#include "metalgb/engine.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <deque>
#include <numeric>
#include <optional>
#include <queue>
#include <set>

namespace metalgb {
namespace {
template<class T> Buffer upload(std::vector<T> const& v) { return Buffer(v.data(), v.size() * sizeof(T)); }

std::vector<int> monotone_constraints(std::string text, u32 columns) {
  std::vector<int> result(columns, 0);
  if (text.empty() || text == "()") return result;
  std::replace(text.begin(), text.end(), '(', '['); std::replace(text.begin(), text.end(), ')', ']');
  auto values = parse_json(text).get<std::vector<int>>();
  require(values.size() <= columns, "monotone_constraints has more entries than features.");
  for (std::size_t i = 0; i < values.size(); ++i) {
    require(values[i] >= -1 && values[i] <= 1, "Monotone constraints must be -1, 0 or 1.");
    result[i] = values[i];
  }
  return result;
}

std::vector<u32> sample_columns(std::vector<u32> values, float fraction, std::mt19937& context_rng,
                                std::span<float const> weights) {
  if (fraction < 1) {
    std::mt19937 rng(context_rng());
    if (weights.empty()) std::shuffle(values.begin(), values.end(), rng);
    else {
      // The host samples feature IDs, using the reference column sampler's
      // exponential keys. All row numerics remain on Metal.
      std::uniform_real_distribution<float> uniform;
      std::vector<float> keys(weights.size(), 0);
      for (auto feature : values) keys[feature] = std::log(uniform(rng)) / std::max(weights[feature], 1.0e-6f);
      std::stable_sort(values.begin(), values.end(), [&](u32 a, u32 b) { return keys[a] > keys[b]; });
    }
    values.resize(std::max<std::size_t>(1, std::floor(values.size() * fraction)));
  }
  std::sort(values.begin(), values.end());
  return values;
}

struct NodePlan {
  u32 id, depth;
  Pair bounds;
  std::vector<Pair> vector_bounds;
  IntPair sum{};
  std::vector<IntPair> vector_sums;
};
struct HistogramPage { Buffer data; std::size_t nodes; };
struct Leaf {
  NodePlan node;
  Split split;
  std::vector<u32> categories;
  std::shared_ptr<HistogramPage> page;
  u32 histogram, begin, count;
  std::vector<VectorWeight> weights;
};
struct LeafOrder {
  bool lossguide;
  bool operator()(Leaf const& a, Leaf const& b) const {
    if (lossguide && a.split.gain != b.split.gain) return a.split.gain < b.split.gain;
    return a.node.id > b.node.id;
  }
};
struct RowSegments { Buffer counts, offsets; };

class TreeWorkspace {
 public:
  TreeWorkspace(bool reuse, std::vector<Buffer>& buffers, std::vector<std::vector<std::uint8_t>>& inputs)
      : reuse_(reuse), buffers_(buffers), inputs_(inputs) {}
  // A batch owns distinct slots until its command completes and its results
  // have been copied into leaves. Cached histograms never use these slots.
  void reset() { next_ = retained_; }
  // These slots stay live throughout a tree; only later per-batch slots reset.
  void retain_prefix() { retained_ = next_; }
  Buffer get(std::size_t bytes) {
    ++requests;
    if (!reuse_) { ++allocations; return Buffer(bytes); }
    auto& buffer = slot();
    // Any other use may rewrite this slot, so it no longer holds known input.
    inputs_[next_ - 1].clear();
    if (!buffer || buffer.size() < bytes) { buffer = Buffer(bytes); ++allocations; }
    return buffer.prefix(bytes);
  }
  template<class Container> Buffer copy(Container const& values) {
    auto bytes = checked_product(values.size(), sizeof(*values.data()));
    auto buffer = get(bytes);
    if (bytes) std::memcpy(buffer.data(), values.data(), bytes);
    return buffer;
  }
  // Host inputs that usually repeat between trees (monotone constraints, GPU
  // feature masks and bounds; all read-only on the GPU). With `reuse`, a slot
  // whose last host write had identical bytes, and which no other request has
  // received since, is returned without a store. When committed asynchronous
  // rounds may still read the slot (`shared`), changed bytes go to a new
  // allocation and the slot keeps its previous contents.
  template<class Container> Buffer copy_input(Container const& values, bool reuse, bool shared) {
    auto bytes = checked_product(values.size(), sizeof(*values.data()));
    if (!reuse_ || !bytes) return copy(values);
    ++requests;
    auto& buffer = slot();
    auto& input = inputs_[next_ - 1];
    auto raw = reinterpret_cast<std::uint8_t const*>(values.data());
    if (reuse && buffer && buffer.size() >= bytes && input.size() == bytes &&
        std::equal(raw, raw + bytes, input.begin())) return buffer.prefix(bytes);
    if (shared) { ++allocations; return Buffer(values.data(), bytes); }
    if (!buffer || buffer.size() < bytes) { buffer = Buffer(bytes); ++allocations; }
    std::memcpy(buffer.data(), raw, bytes);
    if (reuse) input.assign(raw, raw + bytes);
    else input.clear();
    return buffer.prefix(bytes);
  }
  u32 requests{0}, allocations{0};
 private:
  Buffer& slot() {
    if (next_ == buffers_.size()) buffers_.emplace_back();
    if (inputs_.size() < buffers_.size()) inputs_.resize(buffers_.size());
    return buffers_[next_++];
  }
  bool reuse_;
  std::size_t next_{0}, retained_{0};
  std::vector<Buffer>& buffers_;
  std::vector<std::vector<std::uint8_t>>& inputs_;
};

class TreeBuilder {
 public:
  TreeBuilder(Learner& learner, Matrix& data, Buffer const& gradient, Buffer const& scales,
              std::mt19937& rng, u32 group, float eta, bool compact, u32 targets, u32 outputs,
              std::vector<Buffer>& workspace, std::vector<std::vector<std::uint8_t>>& inputs,
              Command* root_command, GpuTreeRound* async_round)
      : learner_(learner), data_(data), gradient_(gradient), scales_(scales), rng_(rng),
        tree_{{Node{}}, group, learner.rounds, {}, {0}, 1, 1, {}}, eta_(eta), targets_(targets),
        max_depth_(learner.integer("max_depth", 6)), max_leaves_(learner.integer("max_leaves", 0)),
        min_gain_(learner.number("min_split_loss", 0)),
        vector_(outputs > 1), compact_(compact),
        lossguide_(learner.str("grow_policy", "depthwise") == "lossguide"),
        scan_(learner.integer("metal_partition_scan", 1) != 0),
        fused_(scan_ && learner.integer("metal_fused_partition", 1) != 0),
        subtract_(learner.integer("metal_histogram_subtraction", 1) != 0),
        coalesce_(learner.integer("metal_coalesce_encoders", 0) != 0),
        compute_fill_(learner.integer("metal_compute_fill", 0) != 0),
        parallel_splits_(learner.integer("metal_parallel_splits", 1) != 0),
        fixed_workspace_(learner.integer("metal_tree_fixed_workspace", 1) != 0),
        histogram_missing_(learner.integer("metal_histogram_missing", 1) != 0 && !data.quantized_dense),
        gpu_row_init_(learner.integer("metal_gpu_row_init", 1) != 0),
        constant_inputs_(learner.integer("metal_tree_constant_inputs", 1) != 0),
        path_elision_(learner.integer("metal_replay_path_elision", 1) != 0),
        workspace_(learner.integer("metal_tree_workspace", 1) != 0, workspace, inputs),
        root_command_(root_command), async_round_(async_round), queue_(LeafOrder{lossguide_}) {
    partition_input_ = data.partition_data();
    partition_bounds_ = data.partition_category_bounds();
    auto cache = learner.str("max_cached_hist_node", "18446744073709551615");
    std::size_t read;
    require(!cache.empty() && cache[0] != '-', "max_cached_hist_node must be positive.");
    auto limit = std::stoull(cache, &read);
    require(read == cache.size() && limit > 0, "max_cached_hist_node must be a positive integer.");
    cache_limit_ = limit == UINT64_MAX ? 4096 : limit;
    monotone_ = monotone_constraints(learner.str("monotone_constraints"), data.cols);
    monotone_buffer_ = fixed_workspace_ ?
        workspace_.copy_input(monotone_, constant_inputs_, learner.has_inflight_rounds()) : upload(monotone_);
    if (vector_) {
      tree_.outputs = outputs; tree_.split_outputs = targets_;
      tree_.base_weights.resize(targets_);
      tree_.nodes[0].threshold = std::numeric_limits<float>::denorm_min();
    }
    if (!learner.str("interaction_constraints").empty())
      interactions_ = parse_json(learner.str("interaction_constraints")).get<std::vector<std::vector<u32>>>();
    for (auto const& set : interactions_) for (auto feature : set)
      require(feature < data.cols, "Interaction constraint feature is out of bounds.");
    std::vector<u32> all_features(data.cols); std::iota(all_features.begin(), all_features.end(), 0);
    tree_features_ = sample_columns(std::move(all_features), learner.number("colsample_bytree", 1),
                                    rng_, data.get_info("feature_weights"));
    // Row identities are written once the driver is known: on the GPU for
    // eligible GPU-driven trees, otherwise by initialize_rows().
    auto row_bytes = checked_product(data.rows, sizeof(u32));
    if (fixed_workspace_) {
      rows_ = workspace_.get(row_bytes);
      scratch_rows_ = workspace_.get(row_bytes);
      ranks_ = workspace_.get(row_bytes);
      workspace_.retain_prefix();
    } else {
      rows_ = Buffer(row_bytes);
      scratch_rows_ = Buffer(row_bytes);
      ranks_ = Buffer(row_bytes);
    }
    has_categories_ = std::any_of(data.categorical.as<std::uint8_t>(), data.categorical.as<std::uint8_t>() + data.cols,
                                [](auto type) { return type != 0; });
    sp_ = {1, data.cols, data.histogram_bins, compact ? 0u : group, learner.number("reg_lambda", 1),
           learner.number("reg_alpha", 0), learner.number("min_child_weight", 1),
           learner.number("max_delta_step", 0), learner.integer("max_cat_to_onehot", 4),
           learner.integer("max_cat_threshold", 64),
           static_cast<u32>(div_up(data.histogram_bins, 256) * 256), static_cast<u32>(has_categories_)};
    if (!vector_) {
      auto tile = learner.integer("metal_scalar_split_bin_tile", 256);
      require(tile == 0 || (tile >= 32 && tile <= 16384 && tile % 32 == 0),
              "metal_scalar_split_bin_tile must be zero or a multiple of 32 in [32, 16384].");
      u32 count = 0;
      if (parallel_splits_ && tile && data.histogram_bins > tile) {
        auto counts = data.cut_counts.as<u32>();
        auto categorical = data.categorical.as<std::uint8_t>();
        for (u32 feature = 0; feature < data.cols; ++feature)
          if (!categorical[feature]) count = std::max(count, counts[feature]);
      }
      scalar_tiles_ = tile ? std::max(1u, static_cast<u32>(div_up(count, tile))) : 1u;
      scalar_tile_ = scalar_tiles_ > 1 ? tile : 0;
      // The 256-thread host descendant scorer keeps one lane per bin, so it
      // overtakes the 32-lane evaluator plus separate subtraction only when
      // most lanes hold a cut: on an M4 Max, above a mean of about 192 numeric
      // cuts per feature for dense input and 160 with missing values.
      auto counts = data.cut_counts.as<u32>();
      auto categorical = data.categorical.as<std::uint8_t>();
      std::uint64_t cuts = 0, numeric = 0;
      for (u32 feature = 0; feature < data.cols; ++feature)
        if (!categorical[feature]) { cuts += counts[feature]; ++numeric; }
      fallback_wide_cuts_ = numeric && cuts > numeric * (data.quantized_dense ? 192u : 160u);
    }
    split_skip_empty_bins_ = learner.integer("metal_skip_empty_bins", 1) != 0 &&
                             !data.quantized_dense && !vector_ && parallel_splits_ && scalar_tiles_ == 1;
    bool large = data.rows >= 262144 && data.cols >= 32;
    rows_per_tile_ = learner.integer("metal_histogram_rows", large ? 4096 : 2048);
    // Packed uint8 bins serve four features with one 4-byte load, so dense
    // numeric inputs beyond a few row tiles also group four features. Lossguide
    // trees never use packed bins; their small nodes favor ungrouped histograms.
    bool packed_grouping = data.quantized_dense && !vector_ && !has_categories_ && !lossguide_ &&
                           data.rows >= 16384 &&
                           learner.integer("metal_histogram_packed_bins", 16) != 0;
    auto feature_group = data.histogram_bins <= 256 ?
        (large || packed_grouping ? 4 : !data.quantized_dense && !vector_ && !has_categories_ ? 2 : 1) : 1;
    features_per_group_ = learner.integer("metal_features_per_group", feature_group);
    require(features_per_group_ >= 1 && features_per_group_ <= 4, "metal_features_per_group must be in [1, 4].");
    if (static_cast<std::size_t>(data.histogram_bins) * features_per_group_ > 1024) features_per_group_ = 1;
    require(rows_per_tile_ >= 256 && rows_per_tile_ <= 16384 && rows_per_tile_ % 256 == 0,
            "metal_histogram_rows must be a multiple of 256 in [256, 16384].");
    histogram_words_ = checked_product(checked_product(targets_, data.cols), (static_cast<std::size_t>(data.histogram_bins) + 1) * 4);
    hp_ = {data.rows, data.cols, data.histogram_bins, compact || vector_ ? targets_ : learner.groups,
             compact ? 0u : group, 0, features_per_group_, targets_};
    learner.active_tuning = {{"histogram_rows", rows_per_tile_}, {"features_per_group", features_per_group_},
                            {"histogram_subtraction", subtract_}, {"partition_scan", scan_},
                            {"fused_partition", fused_}, {"lossguide", lossguide_}, {"histogram_rebuilds", 0},
                            {"coalesced_encoders", coalesce_}, {"compute_fill", compute_fill_}, {"tree_encoders", 0},
                            {"tree_workspace", learner.integer("metal_tree_workspace", 1)},
                            {"tree_fixed_workspace", fixed_workspace_},
                            {"histogram_missing", histogram_missing_},
                            {"split_skip_empty_bins", split_skip_empty_bins_},
                            {"tree_control_fusion", 0}, {"fused_subtraction", 0}, {"fast_weight", 0}, {"fast_gain", 0},
                            {"histogram_present", 0}, {"histogram_compact_scratch", 0}, {"histogram_clear_built", 0},
                            {"histogram_schedule", 0}, {"histogram_single_owner", 0}, {"histogram_packed_bins", 0},
                            {"histogram_unit_hessian", 0},
                            {"split_threads", 32}, {"fast_stats", 0}, {"compact_splits", 0}, {"parallel_root", 0}, {"fused_score", 0},
                            {"fallback_parallel_root", 0}, {"fallback_parallel_splits", 0},
                            {"gpu_tree", 0}, {"gpu_tree_levels", 0}, {"gpu_tree_scratch_bytes", 0},
                            {"tree_fixed_buffer_allocations", fixed_workspace_ ? workspace_.allocations : 4},
                            {"tree_persistent_workspace", learner.integer("metal_tree_workspace", 1) &&
                                                          learner.integer("metal_tree_persistent_workspace", 1)},
                            {"parallel_splits", parallel_splits_},
                            {"scalar_split_bin_tile", scalar_tile_}, {"scalar_split_bin_tiles", scalar_tiles_},
                            {"quantile_cut_tile", data.cut_tile_size}, {"quantile_cut_tiles", data.cut_tile_count},
                            {"quantile_summary_group", data.summary_group_size}, {"quantile_summary_batches", data.summary_batch_count},
                            {"quantile_weighted_summary_stages", data.weighted_summary_stages},
                            {"quantile_weighted_summary_batches", data.weighted_summary_batch_count},
                            {"terminal_histogram_elision", 0},
                            {"parallel_select", 0}, {"deferred_parent_statistics", 0},
                            {"parallel_tree_control", 0},
                            {"parallel_partition_plan", 0}, {"partition_single_pass", 0},
                            {"partition_bin_predicate", 0},
                            {"split_screening", 0}, {"root_screening", 0}, {"split_node_cutoff", 0},
                            {"split_inherit_parent", 0},
                            {"gpu_row_init", 0}, {"replay_path_elision", 0},
                            {"tree_constant_inputs", constant_inputs_ && fixed_workspace_ &&
                                                     learner.integer("metal_tree_workspace", 1)},
                            {"tree_buffer_requests", 0}, {"tree_buffer_allocations", 0}};
  }

  Tree grow() {
    if (gpu_tree_eligible()) {
      if (!gpu_row_init_) initialize_rows();
      return grow_on_gpu();
    }
    // Host-driven trees store into retained workspace between waits. Retire
    // committed asynchronous rounds before the first such store.
    if (learner_.has_inflight_rounds()) learner_.drain();
    initialize_rows();
    bool constrained = std::any_of(monotone_.begin(), monotone_.end(), [](auto value) { return value != 0; });
    float bound = constrained ? std::numeric_limits<float>::max() : std::numeric_limits<float>::infinity();
    NodePlan initial{0, 0, {-bound, bound}, {}, {}, {}};
    if (vector_) initial.vector_bounds.assign(targets_, initial.bounds);
    auto root = evaluate({initial}, {}, root_command_).front();
    if (vector_) {
      // The vector reference stores an eta-scaled root, and accumulates its
      // initial cover in f32. Expanded nodes use the wider split statistics.
      tree_.nodes[0].cover = 0;
      for (u32 target = 0; target < targets_; ++target) {
        tree_.base_weights[target] = root.weights[target].parent * eta_;
        tree_.nodes[0].cover += root.weights[target].hessian;
      }
    } else {
      tree_.base_weights[0] = root.split.parent_weight;
      tree_.nodes[0].leaf = root.split.parent_weight * eta_;
    }
    queue_.push(std::move(root));
    for (auto selected = pop(); !selected.empty(); selected = pop()) {
      std::vector<Leaf> parents;
      std::vector<NodePlan> children;
      for (auto const& leaf : selected) {
        auto next = apply(leaf);
        // Driver::Pop has already counted every split in this batch.
        if ((!max_depth_ || leaf.node.depth + 1 < max_depth_) && (!max_leaves_ || leaves_ < max_leaves_)) {
          parents.push_back(leaf);
          children.insert(children.end(), next.begin(), next.end());
        }
      }
      if (!children.empty()) for (auto& leaf : evaluate(children, parents))
        if (leaf.split.gain > 1.0e-6f) queue_.push(std::move(leaf));
    }
    learner_.active_tuning["tree_buffer_requests"] = workspace_.requests;
    learner_.active_tuning["tree_buffer_allocations"] = workspace_.allocations;
    if (vector_) for (std::size_t id = 0; id < tree_.nodes.size(); ++id) if (tree_.nodes[id].left < 0) {
      tree_.nodes[id].right = tree_.leaf_weights.size() / tree_.outputs;
      if (targets_ == tree_.outputs) {
        auto begin = tree_.base_weights.begin() + id * targets_;
        tree_.leaf_weights.insert(tree_.leaf_weights.end(), begin, begin + targets_);
      } else tree_.leaf_weights.resize(tree_.leaf_weights.size() + tree_.outputs);
    }
    return std::move(tree_);
  }

 private:
  // Host row identities. A retained slot that a committed asynchronous round
  // may still read is replaced rather than overwritten.
  void initialize_rows() {
    if (fixed_workspace_ && !learner_.has_inflight_rounds()) {
      auto row_ids = rows_.as<u32>();
      std::iota(row_ids, row_ids + data_.rows, u32{0});
    } else {
      std::vector<u32> initial_rows(data_.rows); std::iota(initial_rows.begin(), initial_rows.end(), 0);
      rows_ = upload(initial_rows);
    }
  }

  bool gpu_tree_eligible() const {
    // Through depth ten, every nonterminal frontier fits the CPU driver's
    // 256-parent batch. Its final depth-nine frontier may contain 512 nodes,
    // but both CPU batches only append terminal children: no row partition or
    // child evaluation occurs between them. Applying those parents in node-ID
    // order therefore preserves the exact tree layout. Depth eleven needs
    // deferred-parent batching and remains on the original driver.
    // Unsupported settings use this same builder, preserving its RNG.
    if (!learner_.integer("metal_gpu_tree", 1) || vector_ || compact_ || lossguide_ ||
        has_categories_ || !scan_ || !fused_ || !subtract_ || max_leaves_ ||
        max_depth_ < 2 || max_depth_ > 10 || data_.histogram_bins > 256 || scalar_tiles_ != 1 ||
        !interactions_.empty() ||
        learner_.number("colsample_bylevel", 1) != 1 || learner_.number("colsample_bynode", 1) != 1 ||
        std::any_of(monotone_.begin(), monotone_.end(), [](auto value) { return value != 0; })) return false;
    if (cache_limit_ < (std::size_t{1} << max_depth_) - 1) return false;
    return gpu_tree_scratch_bytes() <= 256 * 1024 * 1024;
  }

  std::size_t gpu_tree_scratch_bytes() const {
    auto n = std::size_t{1} << (max_depth_ - 1);
    auto parents = std::max<std::size_t>(1, n / 2);
    auto nodes = (std::size_t{1} << (max_depth_ + 1)) - 1;
    auto hist_tasks = checked_product(div_up(data_.rows, rows_per_tile_) + n,
                                      div_up(data_.cols, features_per_group_));
    auto part_tasks = div_up(data_.rows, 256) + parents;
    return checked_product(2 * n, histogram_words_ * sizeof(u32)) +
        checked_product(n, data_.cols * (sizeof(Split) + std::size_t{1})) +
        checked_product(hist_tasks, sizeof(HistTask)) +
        checked_product(part_tasks, sizeof(PartitionTask) + sizeof(u32)) +
        parents * (sizeof(PartitionSegment) + sizeof(Split) + 5 * sizeof(u32) + sizeof(HistSubtract)) +
        n * (2 * (3 * sizeof(u32) + sizeof(IntPair)) + sizeof(Pair) + sizeof(Split)) +
        nodes * (sizeof(Split) + sizeof(u32)) + (std::size_t{1} << max_depth_) * sizeof(u32) + 528;
  }

  // Certified screening (scalar_screen.metal.h) changes only which candidates
  // run the unchanged exact scorer. Its enclosures assume zero L1, no
  // max_delta_step clamp and unconstrained weights; kernels also recheck the
  // node bounds, feature constraints and parent gain before screening.
  bool split_screening_eligible() const {
    auto magnitude = [](float value) { return std::bit_cast<u32>(value) & 0x7fffffffu; };
    return learner_.integer("metal_split_screening", 1) != 0 && !magnitude(sp_.reg_alpha) &&
           !magnitude(sp_.max_delta_step) && std::isfinite(sp_.reg_lambda) &&
           std::isfinite(sp_.min_child_weight) &&
           std::none_of(monotone_.begin(), monotone_.end(), [](auto value) { return value != 0; });
  }

  // A dense root scores one candidate per lane in a latency-bound dispatch;
  // screening only lengthens it. Missing values add a second, backward one.
  bool root_screening_eligible() const {
    return split_screening_eligible() &&
           learner_.integer("metal_split_screening_root", !data_.quantized_dense) != 0;
  }

  static char const* screened_kernel(u32 mode, bool fast_stats, bool fused_score) {
    constexpr char const* names[3][3] = {
      {"evaluate_splits_screened_subtracted_gain", "evaluate_splits_screened_subtracted_gain_stats",
       "evaluate_splits_screened_subtracted_gain_stats_score"},
      {"evaluate_splits_screened_terminal_gain", "evaluate_splits_screened_terminal_gain_stats",
       "evaluate_splits_screened_terminal_gain_stats_score"},
      {"evaluate_splits_screened_root_gain", "evaluate_splits_screened_root_gain_stats",
       "evaluate_splits_screened_root_gain_stats_score"}};
    return names[mode][fused_score ? 2 : fast_stats ? 1 : 0];
  }

  // Counters are zeroed before encoding and read after the command completes.
  Buffer screening_statistics() {
    auto buffer = workspace_.get(6 * sizeof(u32));
    buffer.zero();
    return buffer;
  }
  void record_screening(Buffer const& counters) {
    auto values = counters.as<u32>();
    auto& stats = learner_.split_screening;
    stats.candidates += values[0]; stats.certified += values[1];
    stats.uncertain += values[2]; stats.exact += values[3];
    stats.groups += values[4]; stats.exact_groups += values[5];
  }

  Tree grow_on_gpu() {
    auto requested_split_threads = learner_.integer("metal_split_threads", 256);
    require(requested_split_threads == 32 || requested_split_threads == 256,
            "metal_split_threads must be 32 or 256.");
    // Select this only after the real GPU-tree eligibility check. Borrowed
    // commands retain their already encoded prediction/gradient boundaries.
    coalesce_ = learner_.integer("metal_coalesce_encoders",
                                data_.quantized_dense && max_depth_ <= 6) != 0;
    bool histogram_present = histogram_missing_ && learner_.integer("metal_histogram_present", 0) != 0;
    bool control_fusion = learner_.integer("metal_tree_control_fusion", 1) != 0;
    // Parallel control kernels: SIMD-group node selection, a threadgroup scan
    // for expansion (fused control only), and threadgroup partition planning.
    // The single-pass partition replaces flags/scan/scatter with one two-ended
    // scatter plus the existing O(1)-per-parent offset kernel.
    bool parallel_select = learner_.integer("metal_parallel_select", 1) != 0;
    // Applied parent statistics feed only the host replay and asynchronous
    // node emission (dispatched after them); compute them once
    // per tree over expanded nodes instead of on every level's critical path.
    // The extra pass pays off from two per-level selections (a dense root
    // keeps its own select_dense_root_split statistics).
    auto selections = max_depth_ - static_cast<u32>(data_.quantized_dense);
    bool deferred_statistics = parallel_select &&
        learner_.integer("metal_deferred_parent_statistics", selections >= 2) != 0;
    bool parallel_control = control_fusion && learner_.integer("metal_parallel_tree_control", 1) != 0;
    bool parallel_plan = learner_.integer("metal_parallel_partition_plan", 1) != 0;
    bool single_pass = learner_.integer("metal_partition_single_pass", 1) != 0;
    // Bins are quantize(data, cuts) of the matrix's own float data, which a
    // QuantileDMatrix decodes back to the same bins. Recoded partition inputs
    // (changed feature types) keep the float comparison.
    bool bin_predicate = single_pass && parallel_plan &&
                         partition_input_.native() == data_.data.native() &&
                         learner_.integer("metal_partition_bin_predicate", 1) != 0;
    bool fused_subtraction = parallel_splits_ && learner_.integer("metal_fused_subtraction", 1) != 0;
    bool histogram_compact_scratch = !histogram_present && features_per_group_ <= 2 &&
        learner_.integer("metal_histogram_compact_scratch", !data_.quantized_dense) != 0;
    bool histogram_clear_built = fused_subtraction && learner_.integer("metal_histogram_clear_built", 1) != 0;
    std::string selected_histogram_kernel(histogram_kernel());
    if (histogram_compact_scratch)
      selected_histogram_kernel = histogram_missing_ ?
          (features_per_group_ == 1 ? "histogram_missing_compact" : "histogram_grouped_missing_compact") :
          (features_per_group_ == 1 ? "histogram_compact" : "histogram_grouped_compact");
    std::string present_histogram_kernel =
        features_per_group_ > 1 ? "histogram_grouped_present" : "histogram_present";
    // Histogram dataflow variants keep each row's exact integer contribution.
    // They need dense input or private missing sums; metal_histogram_missing=0
    // with missing values therefore keeps the original kernels.
    auto packed_request = learner_.integer("metal_histogram_packed_bins", 16);
    require(packed_request == 0 || packed_request == 4 || packed_request == 16,
            "metal_histogram_packed_bins must be 0, 4 or 16.");
    bool tiled_histograms = data_.quantized_dense || histogram_missing_;
    // A packed word holds four features, so a group must not straddle words.
    // Code 255 marks missing values, which therefore need at most 255 bins.
    // Narrow matrices interleave only the word groups they have.
    Buffer packed_bins;
    auto packed_chunk = std::max<u32>(1, std::min<u32>(packed_request / 4, div_up(data_.cols, 4)));
    if (tiled_histograms && packed_request && 4 % features_per_group_ == 0 &&
        data_.histogram_bins <= (data_.quantized_dense ? 256u : 255u))
      packed_bins = data_.packed_bins(packed_chunk);
    // Group-major records keep one feature column hot across a level's nodes;
    // tile-major records let the groups sharing a 16-byte packed chunk, row
    // IDs and gradients run back to back.
    auto histogram_schedule = learner_.integer("metal_histogram_schedule",
                                              packed_bins && packed_request == 16 ? 2 : 1);
    require(histogram_schedule <= 2, "metal_histogram_schedule must be 0, 1 or 2.");
    if (!tiled_histograms) histogram_schedule = 0;
    bool single_owner = tiled_histograms && learner_.integer("metal_histogram_single_owner", 1) != 0;
    // Unweighted squared error quantizes every row's Hessian to one constant.
    // A per-tree GPU check verifies that before count-based Hessians are used,
    // so caller-supplied gradients reaching this path keep the 16-byte sums.
    bool unit_hessian = tiled_histograms && data_.rows && learner_.objective() == 0 &&
                        data_.get_info("weight").empty() &&
                        learner_.integer("metal_histogram_unit_hessian", 1) != 0;
    bool tiled = packed_bins || histogram_schedule || single_owner || unit_hessian;
    HistParams histogram_params_value = hp_;
    u32 plan_flags = 0;
    if (tiled) {
      std::string scratch = histogram_compact_scratch ? (features_per_group_ == 1 ? "_c1" : "_c2") :
                                                        (features_per_group_ == 1 ? "_w1" : "_w4");
      std::string source = packed_bins ? "_packed" : "";
      if (unit_hessian) source += "_unit";
      selected_histogram_kernel = std::string(data_.quantized_dense ? "histogram_tiled_dense" :
                                              "histogram_tiled_missing") + scratch + source;
      present_histogram_kernel = std::string("histogram_tiled_present") +
                                 (features_per_group_ == 1 ? "_w1" : "_w4") + source;
      histogram_params_value.tasks = (histogram_schedule ? u32(kHistogramTileRecords) : 0u) |
          (histogram_schedule == 2 ? u32(kHistogramTileMajor) : 0u) |
          (packed_bins ? packed_chunk << kHistogramPackedShift : 0u);
      plan_flags = (histogram_schedule ? u32(kHistogramPlanTiles) : 0u) |
                   (single_owner ? u32(kHistogramPlanSingleOwner) : 0u);
    }
    auto const& histogram_bins = packed_bins ? packed_bins : data_.bins;
    bool dense_skip_empty_bins = fused_subtraction && data_.quantized_dense &&
                                 learner_.integer("metal_skip_empty_bins", 1) != 0;
    bool fused_skip_empty_bins = split_skip_empty_bins_ || dense_skip_empty_bins;
    bool fast_weight = fused_subtraction && learner_.integer("metal_fast_weight", 0) != 0;
    bool fast_gain = fused_subtraction && learner_.integer("metal_fast_gain", 1) != 0;
    bool parallel_split_groups = requested_split_threads == 256 && fast_gain && !fast_weight;
    bool fast_stats = fast_gain && !fast_weight && learner_.integer("metal_fast_stats", 1) != 0;
    constexpr char const* fused_split_kernels[2][2][2] = {
      {{"evaluate_splits_simd_subtracted", "evaluate_splits_simd_subtracted_gain"},
       {"evaluate_splits_simd_subtracted_fast", "evaluate_splits_simd_subtracted_fast_gain"}},
      {{"evaluate_splits_simd_sparse_subtracted", "evaluate_splits_simd_sparse_subtracted_gain"},
       {"evaluate_splits_simd_sparse_subtracted_fast", "evaluate_splits_simd_sparse_subtracted_fast_gain"}}
    };
    char const* fused_split_kernel = fused_split_kernels[fused_skip_empty_bins][fast_weight][fast_gain];
    if (parallel_split_groups)
      fused_split_kernel = fused_skip_empty_bins ? "evaluate_splits_parallel_sparse_subtracted_gain" :
                                                  "evaluate_splits_parallel_subtracted_gain";
    if (fast_stats)
      fused_split_kernel = parallel_split_groups ?
          (fused_skip_empty_bins ? "evaluate_splits_parallel_sparse_subtracted_gain_stats" : "evaluate_splits_parallel_subtracted_gain_stats") :
          (fused_skip_empty_bins ? "evaluate_splits_simd_sparse_subtracted_gain_stats" : "evaluate_splits_simd_subtracted_gain_stats");
    bool compact_splits = fast_gain && !fast_weight && fused_skip_empty_bins &&
                          learner_.integer("metal_compact_splits", !data_.quantized_dense || max_depth_ >= 8) != 0;
    if (compact_splits) {
      parallel_split_groups = true;
      fused_split_kernel = fast_stats ? "evaluate_splits_compact_subtracted_gain_stats" :
                                       "evaluate_splits_compact_subtracted_gain";
    }
    bool parallel_root = fast_gain && !fast_weight && learner_.integer("metal_parallel_root", 1) != 0;
    bool fused_score = compact_splits && fast_stats && learner_.integer("metal_fused_score", 1) != 0;
    if (fused_score) fused_split_kernel = "evaluate_splits_compact_subtracted_gain_stats_score";
    char const* parallel_root_kernel = fused_score ? "evaluate_splits_parallel_root_gain_stats_score" :
        fast_stats ? "evaluate_splits_parallel_root_gain_stats" : "evaluate_splits_parallel_root_gain";
    // A terminal evaluated level only creates leaf children. Its derived
    // histograms have no consumers after split evaluation; earlier levels
    // still materialize complete histograms for the next subtraction.
    bool terminal_histogram_elision = fused_subtraction && parallel_split_groups &&
        learner_.integer("metal_terminal_histogram_elision", 1) != 0;
    char const* terminal_split_kernel = fused_split_kernel;
    if (terminal_histogram_elision) {
      if (compact_splits)
        terminal_split_kernel = fused_score ? "evaluate_splits_compact_terminal_gain_stats_score" :
            fast_stats ? "evaluate_splits_compact_terminal_gain_stats" : "evaluate_splits_compact_terminal_gain";
      else if (fast_stats)
        terminal_split_kernel = fused_skip_empty_bins ? "evaluate_splits_parallel_sparse_terminal_gain_stats" :
                                                       "evaluate_splits_parallel_terminal_gain_stats";
      else
        terminal_split_kernel = fused_skip_empty_bins ? "evaluate_splits_parallel_sparse_terminal_gain" :
                                                       "evaluate_splits_parallel_terminal_gain";
    }
    // Screened twins reproduce the skip-empty sparse/compact kernels' per-feature
    // records and the parallel root's; other splitters remain unscreened.
    bool split_screening = split_screening_eligible() && fused_subtraction && parallel_split_groups &&
                           fused_skip_empty_bins;
    bool root_screening = root_screening_eligible() && parallel_root;
    if (split_screening) {
      fused_split_kernel = screened_kernel(0, fast_stats, fused_score);
      terminal_split_kernel = screened_kernel(terminal_histogram_elision ? 1 : 0, fast_stats, fused_score);
    }
    if (root_screening) parallel_root_kernel = screened_kernel(2, fast_stats, fused_score);
    // A node-wide cutoff shares certified lower bounds between the feature
    // groups of one node through a per-level 32-bit atomic maximum.
    bool node_cutoff = split_screening && learner_.integer("metal_split_node_cutoff", 1) != 0;
    // Counters are read after the command completes, so asynchronous rounds
    // do not collect them.
    bool screening_stats = (split_screening || root_screening) && !async_round_ &&
                           learner_.integer("metal_split_screening_stats", 0) != 0;
    // Children read their exact weight and cover from the parent's selected
    // split, written by the GPU apply step, instead of dividing again.
    bool inherit_parent = split_screening && learner_.integer("metal_split_inherit_parent", 1) != 0;
    // Reused scratch is written by GPU commands between levels. Never reset
    // the workspace or memcpy over buffers referenced by an unsubmitted level.
    auto capacity = u32{1} << (max_depth_ - 1);
    auto parent_capacity = std::max(1u, capacity / 2);
    auto node_capacity = (u32{1} << (max_depth_ + 1)) - 1;
    auto get = [&](std::size_t count, std::size_t width) { return workspace_.get(checked_product(count, width)); };
    struct Bank { Buffer ids, sums, counts, offsets, histogram; };
    auto bank = [&]() {
      return Bank{get(capacity, sizeof(u32)), get(capacity, sizeof(IntPair)),
                  get(capacity, sizeof(u32)), get(capacity, sizeof(u32)),
                  get(capacity, histogram_words_ * sizeof(u32))};
    };
    std::array<Bank, 2> banks{bank(), bank()};
    // Asynchronous rounds retain their readback buffers until host replay.
    Buffer state, history, flags;
    if (async_round_) {
      state = learner_.round_output(sizeof(TreeDriverState));
      history = learner_.round_output(checked_product(node_capacity, sizeof(Split)));
      flags = learner_.round_output(checked_product(node_capacity, sizeof(u32)));
    } else {
      state = get(1, sizeof(TreeDriverState));
      history = get(node_capacity, sizeof(Split)); flags = get(node_capacity, sizeof(u32));
    }
    auto selected_splits = get(parent_capacity, sizeof(Split));
    auto selected_counts = get(parent_capacity, sizeof(u32)), selected_offsets = get(parent_capacity, sizeof(u32));
    auto selected_children = get(parent_capacity, 2 * sizeof(i32)), parent_indices = get(parent_capacity, sizeof(u32));
    auto partition_capacity = div_up(data_.rows, 256) + parent_capacity;
    auto histogram_capacity = checked_product(div_up(data_.rows, rows_per_tile_) + capacity,
                                              div_up(data_.cols, features_per_group_));
    require(histogram_capacity <= UINT32_MAX && partition_capacity <= UINT32_MAX,
            "GPU tree task count exceeds the indirect dispatch index range.");
    auto partition_tasks = get(partition_capacity, sizeof(PartitionTask));
    auto segments = get(parent_capacity, sizeof(PartitionSegment)), block_counts = get(partition_capacity, sizeof(u32));
    auto histogram_tasks = get(histogram_capacity, sizeof(HistTask));
    auto subtraction = get(parent_capacity, sizeof(HistSubtract));
    auto candidates = get(checked_product(capacity, data_.cols), sizeof(Split)), best = get(capacity, sizeof(Split));
    auto split_params = get(1, sizeof(SplitParams)), histogram_params = get(1, sizeof(HistogramPlanParams));
    auto partition_params = get(1, sizeof(PartitionParams)), parent_count = get(1, sizeof(u32));
    auto zero_bytes = get(1, sizeof(u64));
    std::array<Buffer, 9> args;
    for (auto& argument : args) argument = get(1, sizeof(TreeDriverDispatch));
    auto& [plan_hist_args, hist_args, plan_part_args, part_args, scan_args,
           evaluate_args, select_args, subtract_args, zero_args] = args;
    std::vector<std::uint8_t> mask(checked_product(capacity, data_.cols), 0);
    for (u32 n = 0; n < capacity; ++n)
      for (auto feature : tree_features_) mask[std::size_t(n) * data_.cols + feature] = 1;
    bool shared_inputs = learner_.has_inflight_rounds();
    auto allowed = workspace_.copy_input(mask, constant_inputs_, shared_inputs);
    auto infinity = std::numeric_limits<float>::infinity();
    auto bounds = workspace_.copy_input(std::vector<Pair>(capacity, Pair{-infinity, infinity}),
                                        constant_inputs_, shared_inputs);
    auto unused = get(0, 1);
    // Level d uses cells [2^d - 1, 2^(d + 1) - 1); one clear per tree suffices.
    auto node_cut = node_cutoff ? get(std::size_t{1} << max_depth_, sizeof(u32)) : unused;
    // Counters are host-zeroed; only synchronous rounds collect them. The
    // unit-Hessian flag is cleared on the GPU, so committed asynchronous
    // rounds never see a host store into a slot they may still read.
    auto screen_counters = screening_stats ?
        workspace_.copy_input(std::array<u32, 6>{}, false, shared_inputs) : unused;
    auto unit_mismatch = unit_hessian ? get(1, sizeof(u32)) : unused;
    TreeDriverParams p{sp_, {1, data_.cols, features_per_group_, rows_per_tile_, 1, plan_flags},
                       {data_.rows, data_.cols, 0, learner_.integer("metal_partition_aggregation", 1)},
                       max_depth_, node_capacity, static_cast<u32>(parallel_splits_),
                       static_cast<u32>(histogram_words_ / 4), min_gain_,
                       (parallel_select ? kTreeDriverParallelSelect : 0u) |
                           (parallel_plan ? kTreeDriverParallelPlan : 0u) |
                           (data_.quantized_dense ? kTreeDriverDenseRoot : 0u), 0, 0};
    // One threadgroup scans a frontier of at most `capacity` slots.
    auto control_threads = std::clamp<std::size_t>(capacity, 32, kTreeDriverSlots);
    char const* plan_kernel = parallel_plan ?
        (bin_predicate ? "plan_partitions_parallel_bins" :
         single_pass ? "plan_partitions_parallel_cursors" : "plan_partitions_parallel") :
        (single_pass ? "plan_partitions_cursors" : "plan_partitions");
    std::optional<Command> owned;
    if (!root_command_) owned.emplace("GPU tree construction", coalesce_, compute_fill_);
    auto& command = root_command_ ? *root_command_ : *owned;
    // Row identities, the unit-Hessian check and the node-cutoff clear share
    // the initialization encoder; serial dispatch makes them visible to the
    // first partition, histogram and split kernels.
    command.set_coalescing(coalesce_ || gpu_row_init_ || unit_hessian || node_cutoff);
    if (gpu_row_init_)
      command.dispatch("tree_rows_init", {rows_}, data_.rows, {div_up(data_.rows, 256), 1, 1});
    if (unit_hessian) {
      command.dispatch("fill_zero", {unit_mismatch}, static_cast<u64>(unit_mismatch.size()), {1, 1, 1});
      command.dispatch("histogram_unit_hessian_check", {gradient_, unit_mismatch}, hp_, {div_up(data_.rows, 256), 1, 1});
    }
    if (node_cutoff)
      command.dispatch("fill_zero", {node_cut}, static_cast<u64>(node_cut.size()), {1, 1, 1});
    if (control_fusion)
      command.dispatch(parallel_control ? "tree_driver_init_prepare_parallel" : "tree_driver_init_prepare",
                       {state, banks[0].ids, banks[0].sums,
                       banks[0].counts, banks[0].offsets, flags, split_params, histogram_params,
                       partition_params, parent_count, plan_hist_args, hist_args, plan_part_args,
                       part_args, scan_args, evaluate_args, select_args, subtract_args,
                       zero_bytes, zero_args}, p, {1, 1, 1}, {1, 1, 1});
    else
      command.dispatch("tree_driver_init", {state, banks[0].ids, banks[0].sums,
                       banks[0].counts, banks[0].offsets, flags}, p, {1, 1, 1}, {1, 1, 1});
    command.set_coalescing(coalesce_);
    for (u32 depth = 0; depth < max_depth_; ++depth) {
      auto& current = banks[depth & 1u];
      auto& other = banks[(depth & 1u) ^ 1u];
      if (!control_fusion)
        command.dispatch("tree_driver_prepare", {state, split_params, histogram_params, partition_params,
                       parent_count, plan_hist_args, hist_args, plan_part_args, part_args, scan_args,
                       evaluate_args, select_args, subtract_args, zero_bytes, zero_args}, p,
                       {1, 1, 1}, {1, 1, 1});
      if (depth) {
        if (!parallel_plan && !single_pass)
          command.dispatch_indirect("plan_partitions", {selected_counts, selected_offsets, selected_children,
                                    partition_tasks, segments, part_args, parent_count}, nullptr, 0,
                                    plan_part_args, {64, 1, 1});
        else if (bin_predicate)
          // Single-pass partitioning has no per-tile counts; its planner stores
          // each parent's equal-cut bin index in that slot instead.
          command.dispatch_indirect(plan_kernel, {selected_counts, selected_offsets, selected_children,
                                    partition_tasks, segments, part_args, current.counts, parent_count,
                                    selected_splits, data_.cuts, data_.cut_counts, block_counts, split_params},
                                    nullptr, 0, plan_part_args, {256, 1, 1});
        else
          command.dispatch_indirect(plan_kernel, {selected_counts, selected_offsets, selected_children,
                                    partition_tasks, segments, part_args, current.counts, parent_count},
                                    nullptr, 0, plan_part_args, {parallel_plan ? 256u : 64u, 1, 1});
        if (single_pass) {
          // Children counts are the final atomic cursors; offsets follow from
          // them exactly as in the stable scan's per-segment epilogue.
          if (bin_predicate)
            command.dispatch_indirect("partition_two_ended_bins", {partition_input_, rows_, selected_splits,
                                      partition_tasks, segments, current.counts, scratch_rows_, unused,
                                      partition_params, data_.bins, block_counts}, nullptr, 0, part_args);
          else
            command.dispatch_indirect("partition_two_ended", {partition_input_, rows_, selected_splits,
                                      partition_tasks, segments, current.counts, scratch_rows_, unused,
                                      partition_params}, nullptr, 0, part_args);
          command.dispatch_indirect("partition_atomic_offsets", {segments, current.counts, current.offsets,
                                    partition_params}, nullptr, 0, scan_args, {64, 1, 1});
        } else {
          command.dispatch_indirect("partition_segment_flags", {partition_input_, rows_, selected_splits,
                                    partition_tasks, ranks_, block_counts, unused, partition_params},
                                    nullptr, 0, part_args);
          command.dispatch_indirect("partition_segment_scan", {segments, block_counts, current.counts,
                                    current.offsets, partition_params}, nullptr, 0, scan_args, {64, 1, 1});
          command.dispatch_indirect("partition_segment_scatter", {rows_, partition_tasks, segments, ranks_,
                                    block_counts, current.offsets, scratch_rows_, partition_params}, nullptr, 0, part_args);
        }
        // No deferred leaves remain in this bounded depthwise batch. Only the
        // selected segments are needed by the next frontier after this swap.
        std::swap(rows_, scratch_rows_);
      }
      if (depth && histogram_clear_built)
        command.dispatch_indirect("fill_histogram_built", {current.histogram, current.counts, histogram_params},
                                  hp_, evaluate_args);
      else
        command.dispatch_indirect("fill_zero", {current.histogram, zero_bytes}, nullptr, 0, zero_args);
      command.dispatch_indirect("plan_histograms", {current.counts, current.offsets, parent_indices,
                                histogram_tasks, subtraction, hist_args, histogram_params}, nullptr, 0,
                                plan_hist_args, {64, 1, 1});
      auto const& histogram_kernel_name = depth && histogram_present ? present_histogram_kernel : selected_histogram_kernel;
      if (tiled)
        command.dispatch_indirect(histogram_kernel_name, {histogram_bins, rows_, gradient_, scales_, histogram_tasks,
                                  current.histogram, unit_mismatch}, histogram_params_value, hist_args);
      else
        command.dispatch_indirect(histogram_kernel_name, {data_.bins, rows_, gradient_, scales_, histogram_tasks,
                                  current.histogram}, hp_, hist_args);
      if (depth && !fused_subtraction)
        command.dispatch_indirect("subtract_histograms", {other.histogram, subtraction, current.histogram},
                                  hp_, subtract_args);
      ScreenParams screen{(1u << depth) - 1, static_cast<u32>(node_cutoff), static_cast<u32>(screening_stats),
                          static_cast<u32>(inherit_parent && depth)};
      if (depth && fused_subtraction && split_screening)
        command.dispatch_indirect(terminal_histogram_elision && depth + 1 == max_depth_ ?
                                      terminal_split_kernel : fused_split_kernel,
                                  {current.histogram, scales_, data_.cuts, data_.cut_counts, data_.cut_minima,
                                   monotone_buffer_, bounds, allowed, candidates, data_.categorical,
                                   unused, unused, unused, current.sums, split_params,
                                   other.histogram, subtraction, node_cut, screen_counters, selected_splits},
                                  screen, evaluate_args, {256, 1, 1});
      else if (depth && fused_subtraction)
        command.dispatch_indirect(terminal_histogram_elision && depth + 1 == max_depth_ ?
                                      terminal_split_kernel : fused_split_kernel,
                                  {current.histogram, scales_, data_.cuts, data_.cut_counts, data_.cut_minima,
                                   monotone_buffer_, bounds, allowed, candidates, data_.categorical,
                                   unused, unused, unused, current.sums, split_params,
                                   other.histogram, subtraction}, nullptr, 0, evaluate_args,
                                  {parallel_split_groups ? 256u : 32u, 1, 1});
      else if (!depth && root_screening)
        command.dispatch_indirect(parallel_root_kernel,
                                  {current.histogram, scales_, data_.cuts, data_.cut_counts, data_.cut_minima,
                                   monotone_buffer_, bounds, allowed, candidates, data_.categorical,
                                   unused, unused, unused, current.sums, split_params, screen_counters},
                                  screen, evaluate_args, {256, 1, 1});
      else
        command.dispatch_indirect(!depth && parallel_root ?
                                  parallel_root_kernel :
                                  split_kernel(),
                                  {current.histogram, scales_, data_.cuts, data_.cut_counts, data_.cut_minima,
                                   monotone_buffer_, bounds, allowed, candidates, data_.categorical,
                                   unused, unused, unused, current.sums, split_params},
                                  nullptr, 0, evaluate_args, {!depth && parallel_root ? 256u : parallel_splits_ ? 32u : 64u, 1, 1});
      if (!depth && data_.quantized_dense)
        command.dispatch_indirect("select_dense_root_split", {candidates, best, scales_, bounds,
                                  current.histogram, data_.cut_counts, data_.cuts, data_.cut_minima,
                                  split_params}, nullptr, 0, select_args);
      else if (parallel_select)
        command.dispatch_indirect(deferred_statistics ? "select_splits_simd_deferred" : "select_splits_simd",
                                  {candidates, best, scales_, bounds, split_params},
                                  nullptr, 0, select_args, {32, 1, 1});
      else
        command.dispatch_indirect("select_splits", {candidates, best, scales_, bounds, split_params},
                                  nullptr, 0, select_args, {64, 1, 1});
      if (control_fusion)
        command.dispatch(parallel_control ? "tree_driver_apply_prepare_parallel" : "tree_driver_apply_prepare",
                         {state, current.ids, best, current.counts, current.offsets,
                         history, flags, other.ids, other.sums, selected_splits, selected_counts,
                         selected_offsets, selected_children, parent_indices, split_params, histogram_params,
                         partition_params, parent_count, plan_hist_args, hist_args, plan_part_args, part_args,
                         scan_args, evaluate_args, select_args, subtract_args, zero_bytes, zero_args},
                         p, {1, 1, 1}, {parallel_control ? control_threads : 1, 1, 1});
      else
        command.dispatch("tree_driver_apply", {state, current.ids, best, current.counts, current.offsets,
                         history, flags, other.ids, other.sums, selected_splits, selected_counts,
                         selected_offsets, selected_children, parent_indices}, p, {1, 1, 1}, {1, 1, 1});
    }
    // Deferred applied statistics complete the history that host replay and
    // asynchronous node emission read.
    if (deferred_statistics)
      command.dispatch("tree_driver_applied_statistics", {state, history, flags, scales_, bounds}, p,
                       {div_up(node_capacity, 64), 1, 1}, {64, 1, 1});
    // Path selections are known once encoded; only the evaluated depth
    // depends on GPU results.
    learner_.active_tuning["gpu_tree"] = 1;
    learner_.active_tuning["coalesced_encoders"] = coalesce_;
    learner_.active_tuning["tree_control_fusion"] = control_fusion;
    learner_.active_tuning["fused_subtraction"] = fused_subtraction;
    learner_.active_tuning["split_skip_empty_bins"] = fused_skip_empty_bins;
    learner_.active_tuning["fast_weight"] = fast_weight;
    learner_.active_tuning["fast_gain"] = fast_gain;
    learner_.active_tuning["fast_stats"] = fast_stats;
    learner_.active_tuning["compact_splits"] = compact_splits;
    learner_.active_tuning["parallel_root"] = parallel_root;
    learner_.active_tuning["fused_score"] = fused_score;
    learner_.active_tuning["histogram_present"] = histogram_present;
    learner_.active_tuning["histogram_compact_scratch"] = histogram_compact_scratch;
    learner_.active_tuning["histogram_clear_built"] = histogram_clear_built;
    learner_.active_tuning["histogram_schedule"] = histogram_schedule;
    learner_.active_tuning["histogram_single_owner"] = single_owner;
    learner_.active_tuning["histogram_packed_bins"] = packed_bins ? packed_request : 0;
    // Selection; synchronous rounds replace it with the GPU check's result.
    learner_.active_tuning["histogram_unit_hessian"] = unit_hessian;
    learner_.active_tuning["terminal_histogram_elision"] = terminal_histogram_elision;
    learner_.active_tuning["parallel_select"] = parallel_select;
    learner_.active_tuning["deferred_parent_statistics"] = deferred_statistics;
    learner_.active_tuning["parallel_tree_control"] = parallel_control;
    learner_.active_tuning["parallel_partition_plan"] = parallel_plan;
    learner_.active_tuning["partition_single_pass"] = single_pass;
    learner_.active_tuning["partition_bin_predicate"] = bin_predicate;
    learner_.active_tuning["split_screening"] = split_screening;
    learner_.active_tuning["root_screening"] = root_screening;
    learner_.active_tuning["split_node_cutoff"] = node_cutoff;
    learner_.active_tuning["split_inherit_parent"] = inherit_parent;
    learner_.active_tuning["split_threads"] = parallel_split_groups ? 256 : 32;
    learner_.active_tuning["gpu_tree_scratch_bytes"] = gpu_tree_scratch_bytes();
    learner_.active_tuning["gpu_row_init"] = gpu_row_init_;
    learner_.active_tuning["replay_path_elision"] = path_elision_;
    learner_.active_tuning["tree_buffer_requests"] = workspace_.requests;
    learner_.active_tuning["tree_buffer_allocations"] = workspace_.allocations;
    GpuTreeRound round{state, flags, history, {}, {}, eta_, tree_.group, tree_.round, node_capacity, !path_elision_};
    if (async_round_) {
      // The next round's margin update reads this table on the GPU. The
      // learner replays the same history and compares before publishing.
      require(node_capacity <= 2048, "GPU tree node emission supports at most depth ten.");
      round.nodes = learner_.round_output(checked_product(node_capacity, sizeof(Node)));
      round.meta = learner_.round_output(sizeof(TreeMeta));
      TreeEmitParams emit{eta_, tree_.group, tree_.round, node_capacity};
      command.dispatch("tree_emit_nodes", {state, flags, history, round.nodes, round.meta}, emit,
                       {1, 1, 1}, {std::min<std::size_t>(256, div_up(node_capacity, 32) * 32), 1, 1});
      // Test-only fault injection: clear the emitted metadata (1) or the
      // driver state (2) after emission. Node links stay valid for the next
      // margin update; retirement must reject the round and roll back.
      auto fault = learner_.integer("metal_async_fault", 0);
      if (fault && tree_.round == learner_.integer("metal_async_fault_round", 0))
        command.zero(fault == 1 ? round.meta : state);
      learner_.active_tuning["tree_encoders"] += command.encoder_count();
      *async_round_ = std::move(round);
      return {};
    }
    learner_.record(root_command_ ? "gradient_tree" : "gpu_tree", command.wait());
    learner_.active_tuning["tree_encoders"] += command.encoder_count();
    if (unit_hessian) learner_.active_tuning["histogram_unit_hessian"] = !unit_mismatch.as<u32>()[0];
    if (screening_stats) record_screening(screen_counters);
    u32 evaluated_levels = 0;
    tree_ = replay_gpu_tree(round, evaluated_levels);
    learner_.active_tuning["gpu_tree_levels"] = evaluated_levels;
    return std::move(tree_);
  }

  std::vector<Leaf> pop() {
    std::vector<Leaf> result;
    if (queue_.empty()) return result;
    auto depth = queue_.top().node.depth;
    // Match the reference CPU driver's batch size, including column-sampling
    // order when a level exceeds 256 nodes. CUDA uses the same driver with 1024.
    do {
      Leaf leaf = queue_.top(); queue_.pop();
      if (leaf.split.feature >= 0 && leaf.split.gain > 1.0e-6f && leaf.split.gain >= min_gain_ &&
          (!max_depth_ || leaf.node.depth < max_depth_) && (!max_leaves_ || leaves_ < max_leaves_)) {
        ++leaves_;
        result.push_back(std::move(leaf));
      }
      if (lossguide_) break;
    } while (!queue_.empty() && queue_.top().node.depth == depth && result.size() < 256);
    return result;
  }

  std::array<NodePlan, 2> apply(Leaf const& leaf) {
    require(tree_.nodes.size() <= static_cast<std::size_t>(INT32_MAX) - 2, "Tree exceeds the node index range.");
    auto id = leaf.node.id;
    auto const& split = leaf.split;
    auto left = static_cast<u32>(tree_.nodes.size()), right = left + 1;
    auto& node = tree_.nodes[id];
    node.left = left; node.right = right; node.feature = split.feature;
    node.threshold = split.threshold; node.default_left = split.default_left; node.gain = split.gain;
    node.split_type = split.split_type;
    if (!vector_) {
      tree_.base_weights[id] = split.applied_parent_weight;
      node.cover = split.applied_parent_hessian;
    }
    else {
      node.cover = split.total_h;
      for (u32 target = 0; target < targets_; ++target)
        tree_.base_weights[std::size_t(id) * targets_ + target] = leaf.weights[target].parent;
    }
    if (split.split_type) {
      node.threshold = std::numeric_limits<float>::denorm_min();
      node.category_begin = tree_.categories.size(); node.category_count = leaf.categories.size();
      tree_.categories.insert(tree_.categories.end(), leaf.categories.begin(), leaf.categories.end());
    }
    Node l, r; l.parent = id; r.parent = id;
    l.leaf = split.left_weight * eta_; l.cover = split.left_h;
    r.leaf = split.right_weight * eta_; r.cover = split.right_h;
    if (vector_) l.threshold = r.threshold = std::numeric_limits<float>::denorm_min();
    tree_.nodes.push_back(l); tree_.nodes.push_back(r);
    if (!vector_) { tree_.base_weights.push_back(l.leaf); tree_.base_weights.push_back(r.leaf); }
    else {
      for (auto const& weight : leaf.weights) tree_.base_weights.push_back(weight.left * eta_);
      for (auto const& weight : leaf.weights) tree_.base_weights.push_back(weight.right * eta_);
    }
    auto path = paths_[id]; path.insert(split.feature); paths_.push_back(path); paths_.push_back(path);
    Pair lb = leaf.node.bounds, rb = leaf.node.bounds;
    float middle = (split.left_weight + split.right_weight) * 0.5f;
    if (monotone_[split.feature] > 0) { lb.h = std::min(lb.h, middle); rb.g = std::max(rb.g, middle); }
    if (monotone_[split.feature] < 0) { lb.g = std::max(lb.g, middle); rb.h = std::min(rb.h, middle); }
    std::array<NodePlan, 2> children{{{left, leaf.node.depth + 1, lb, {}, {}, {}},
                                     {right, leaf.node.depth + 1, rb, {}, {}, {}}}};
    children[0].sum = split.left_sum; children[1].sum = split.right_sum;
    if (vector_) {
      children[0].vector_bounds = children[1].vector_bounds = leaf.node.vector_bounds;
      for (u32 target = 0; target < targets_; ++target) {
        auto& left_bound = children[0].vector_bounds[target];
        auto& right_bound = children[1].vector_bounds[target];
        auto const& weight = leaf.weights[target];
        children[0].vector_sums.push_back(weight.left_sum);
        children[1].vector_sums.push_back(weight.right_sum);
        float midpoint = weight.left + 0.5f * (weight.right - weight.left);
        if (monotone_[split.feature] > 0) { left_bound.h = midpoint; right_bound.g = midpoint; }
        if (monotone_[split.feature] < 0) { left_bound.g = midpoint; right_bound.h = midpoint; }
      }
    }
    return children;
  }

  RowSegments partition(Command& command, std::vector<Leaf> const& parents) {
    std::vector<u32> counts, offsets, categories;
    std::vector<Split> splits;
    std::vector<std::array<i32, 2>> children;
    std::size_t capacity = 0;
    for (auto const& leaf : parents) {
      auto split = leaf.split;
      if (split.split_type && partition_bounds_) {
        split.split_type = 0;
        split.threshold = partition_bounds_.as<float>()[split.feature];
      }
      split.category_begin = categories.size();
      categories.insert(categories.end(), leaf.categories.begin(), leaf.categories.end());
      splits.push_back(split);
      children.push_back({static_cast<i32>(2 * counts.size()), static_cast<i32>(2 * counts.size() + 1)});
      counts.push_back(leaf.count); offsets.push_back(leaf.begin);
      capacity += div_up(leaf.count, 256);
    }
    require(capacity <= UINT32_MAX, "Partition task count exceeds the dispatch index range.");
    auto nodes = static_cast<u32>(2 * parents.size());
    RowSegments output{workspace_.get(nodes * sizeof(u32)), workspace_.get(nodes * sizeof(u32))};
    Buffer tasks = workspace_.get(capacity * sizeof(PartitionTask));
    Buffer segments = workspace_.get(parents.size() * sizeof(PartitionSegment));
    auto arguments = dispatch_arguments();
    if (fused_) {
      command.dispatch("plan_partitions", {workspace_.copy(counts), workspace_.copy(offsets), workspace_.copy(children), tasks, segments, arguments},
                       static_cast<u32>(parents.size()), {div_up(parents.size(), 64), 1, 1}, {64, 1, 1});
    } else {
      std::vector<PartitionTask> host_tasks;
      std::vector<PartitionSegment> host_segments;
      for (u32 n = 0; n < parents.size(); ++n) {
        auto first = static_cast<u32>(host_tasks.size());
        for (u32 i = 0; i < counts[n]; i += 256)
          host_tasks.push_back({offsets[n] + i, offsets[n] + std::min(i + 256, counts[n]), n, n});
        host_segments.push_back({offsets[n], offsets[n] + counts[n], first,
                                 static_cast<u32>(host_tasks.size()), 2 * n, 2 * n + 1});
      }
      tasks = workspace_.copy(host_tasks); segments = workspace_.copy(host_segments);
    }
    Buffer split_buffer = workspace_.copy(splits), category_buffer = workspace_.copy(categories);
    Buffer block_counts = workspace_.get(capacity * sizeof(u32));
    PartitionParams pp{data_.rows, data_.cols, nodes, learner_.integer("metal_partition_aggregation", 1)};
    if (scan_) {
      if (fused_) command.dispatch_indirect("partition_segment_flags", {partition_input_, rows_, split_buffer, tasks, ranks_, block_counts, category_buffer}, pp, arguments);
      else command.dispatch("partition_segment_flags", {partition_input_, rows_, split_buffer, tasks, ranks_, block_counts, category_buffer}, pp, {capacity, 1, 1});
      command.dispatch("partition_segment_scan", {segments, block_counts, output.counts, output.offsets},
                       pp, {div_up(parents.size(), 64), 1, 1}, {64, 1, 1});
      if (fused_) command.dispatch_indirect("partition_segment_scatter", {rows_, tasks, segments, ranks_, block_counts, output.offsets, scratch_rows_}, pp, arguments);
      else command.dispatch("partition_segment_scatter", {rows_, tasks, segments, ranks_, block_counts, output.offsets, scratch_rows_}, pp, {capacity, 1, 1});
    } else {
      Buffer cursors = workspace_.get(nodes * sizeof(u32));
      command.zero(output.counts); command.zero(cursors);
      command.dispatch("partition_atomic_count", {partition_input_, rows_, split_buffer, tasks, ranks_, output.counts, category_buffer}, pp, {capacity, 1, 1});
      command.dispatch("partition_atomic_offsets", {segments, output.counts, output.offsets},
                       pp, {div_up(parents.size(), 64), 1, 1}, {64, 1, 1});
      command.dispatch("partition_atomic_scatter", {rows_, tasks, ranks_, output.offsets, cursors, scratch_rows_}, pp, {capacity, 1, 1});
    }
    if (queue_.empty()) std::swap(rows_, scratch_rows_);
    else {
      // Deferred leaves keep their original, disjoint row ranges. Copy back
      // only the expanded ranges; copying all rows per leaf is unnecessary.
      if (fused_) command.dispatch_indirect("partition_segment_copy", {scratch_rows_, tasks, rows_}, pp, arguments);
      else command.dispatch("partition_segment_copy", {scratch_rows_, tasks, rows_}, pp, {capacity, 1, 1});
    }
    return output;
  }

  void histogram_tasks(std::vector<HistTask>& tasks, u32 begin, u32 count, u32 node) const {
    for (u32 feature = 0; feature < data_.cols; feature += features_per_group_)
      for (u32 i = 0; i < count; i += rows_per_tile_)
        tasks.push_back({begin + i, begin + std::min(i + rows_per_tile_, count), node, feature});
  }

  std::pair<Buffer, std::vector<u32>> parent_histograms(Command& command, std::vector<Leaf> const& parents) {
    if (!subtract_ || parents.empty()) return {workspace_.get(0), {}};
    auto const& page = parents.front().page;
    bool shared = page && page->data && std::all_of(parents.begin(), parents.end(), [&](auto const& leaf) { return leaf.page == page; });
    std::vector<u32> indices;
    if (shared) {
      for (auto const& leaf : parents) indices.push_back(leaf.histogram);
      return {page->data, std::move(indices)};
    }
    Buffer combined = workspace_.get(checked_product(parents.size(), histogram_words_) * sizeof(u32));
    command.zero(combined);
    std::map<HistogramPage*, std::vector<std::array<u32, 2>>> copies;
    std::vector<HistTask> rebuild;
    for (u32 i = 0; i < parents.size(); ++i) {
      auto const& leaf = parents[i];
      indices.push_back(i);
      if (leaf.page && leaf.page->data) copies[leaf.page.get()].push_back({leaf.histogram, i});
      else {
        histogram_tasks(rebuild, leaf.begin, leaf.count, i);
        ++learner_.active_tuning["histogram_rebuilds"];
      }
    }
    for (auto const& [source, mapping] : copies)
      command.dispatch("copy_histograms", {source->data, workspace_.copy(mapping), combined}, hp_,
                       {div_up(histogram_words_, 256), mapping.size(), 1});
    command.dispatch(histogram_kernel(), {data_.bins, rows_, gradient_, scales_, workspace_.copy(rebuild), combined}, hp_, {rebuild.size(), targets_, 1});
    return {std::move(combined), std::move(indices)};
  }

  Buffer allowed_features(std::vector<NodePlan> const& nodes) {
    std::vector<std::uint8_t> allowed(checked_product(nodes.size(), data_.cols), 0);
    for (u32 n = 0; n < nodes.size(); ++n) {
      auto [it, inserted] = level_features_.try_emplace(nodes[n].depth);
      if (inserted) it->second = sample_columns(tree_features_, learner_.number("colsample_bylevel", 1), rng_, data_.get_info("feature_weights"));
      auto features = sample_columns(it->second, learner_.number("colsample_bynode", 1), rng_, data_.get_info("feature_weights"));
      auto const& path = paths_[nodes[n].id];
      std::set<u32> valid;
      if (!interactions_.empty() && !path.empty()) {
        for (auto const& set : interactions_) {
          bool compatible = true;
          for (auto feature : path) if (std::find(set.begin(), set.end(), feature) == set.end()) compatible = false;
          if (compatible) valid.insert(set.begin(), set.end());
        }
        valid.insert(path.begin(), path.end());
      }
      for (auto feature : features) if (path.empty() || interactions_.empty() || valid.contains(feature))
        allowed[n * data_.cols + feature] = 1;
    }
    return workspace_.copy(allowed);
  }

  std::vector<Leaf> evaluate(std::vector<NodePlan> const& nodes, std::vector<Leaf> const& parents,
                             Command* root_command = nullptr) {
    workspace_.reset();
    auto n = static_cast<u32>(nodes.size());
    sp_.nodes = n;
    sp_.use_parent_sums = nodes.front().depth != 0;
    RowSegments segments{workspace_.copy(std::array<u32, 1>{data_.rows}), workspace_.copy(std::array<u32, 1>{0})};
    if (!parents.empty() && !fused_) {
      Command command(scan_ ? "segmented row partition" : "atomic row partition", coalesce_, compute_fill_);
      segments = partition(command, parents);
      learner_.record("partition", command.wait());
      learner_.active_tuning["tree_encoders"] += command.encoder_count();
    }
    // The first tree can consume its pending gradients in this same command.
    // All dispatches remain serial; CPU split readback still happens after wait.
    std::optional<Command> owned_command;
    if (!root_command)
      owned_command.emplace(fused_ ? "partition, histogram and split search" : "histogram and split search", coalesce_, compute_fill_);
    Command& command = root_command ? *root_command : *owned_command;
    if (!parents.empty() && fused_) segments = partition(command, parents);
    auto [parent_hist, parent_indices] = parent_histograms(command, parents);
    auto subtraction_count = subtract_ ? parents.size() : 0;
    // Every host expansion supplies adjacent child pairs and their inherited
    // integer totals. Reuse the existing numeric fused scorer without changing
    // the host's feature masks, bounds, queue order, or final split selection.
    bool fallback_parallel_splits = !vector_ && targets_ == 1 && !has_categories_ &&
        sp_.use_parent_sums && subtraction_count && n == 2 * subtraction_count &&
        parallel_splits_ && scalar_tiles_ == 1 && data_.histogram_bins <= 256 &&
        learner_.integer("metal_fallback_parallel_splits", fallback_wide_cuts_) != 0 &&
        learner_.integer("metal_fused_subtraction", 1) != 0 &&
        learner_.integer("metal_split_threads", 256) == 256 &&
        learner_.integer("metal_fast_gain", 1) != 0 &&
        learner_.integer("metal_fast_weight", 0) == 0;
    bool fallback_fast_stats = fallback_parallel_splits && learner_.integer("metal_fast_stats", 1) != 0;
    Buffer hist(checked_product(n, histogram_words_) * sizeof(u32));
    command.zero(hist);
    Buffer subtraction_buffer = workspace_.get(subtraction_count * sizeof(HistSubtract));
    if (fused_) {
      std::size_t rows = parents.empty() ? data_.rows : 0;
      for (auto const& leaf : parents) rows += leaf.count;
      auto capacity = checked_product(div_up(rows, rows_per_tile_) + n, div_up(data_.cols, features_per_group_));
      require(capacity <= UINT32_MAX, "Histogram task count exceeds the indirect dispatch index range.");
      Buffer tasks = workspace_.get(capacity * sizeof(HistTask));
      auto arguments = dispatch_arguments(targets_);
      HistogramPlanParams plan{n, data_.cols, features_per_group_, rows_per_tile_, static_cast<u32>(subtraction_count != 0), 0};
      command.dispatch("plan_histograms", {segments.counts, segments.offsets, workspace_.copy(parent_indices), tasks, subtraction_buffer, arguments},
                       plan, {div_up(n, 64), 1, 1}, {64, 1, 1});
      command.dispatch_indirect(histogram_kernel(), {data_.bins, rows_, gradient_, scales_, tasks, hist}, hp_, arguments);
    } else {
      std::vector<HistTask> tasks;
      std::vector<HistSubtract> subtraction;
      for (u32 i = 0; i < n; ++i) {
        if (subtraction_count) {
          u32 first = i & ~1u;
          u32 small = segments.counts.as<u32>()[first] <= segments.counts.as<u32>()[first + 1] ? first : first + 1;
          if (i == first) subtraction.push_back({parent_indices[first / 2], small, small ^ 1u, 0});
          if (i != small) continue;
        }
        histogram_tasks(tasks, segments.offsets.as<u32>()[i], segments.counts.as<u32>()[i], i);
      }
      subtraction_buffer = workspace_.copy(subtraction);
      command.dispatch(histogram_kernel(), {data_.bins, rows_, gradient_, scales_, workspace_.copy(tasks), hist}, hp_, {tasks.size(), targets_, 1});
    }
    if (subtraction_count && !fallback_parallel_splits)
      command.dispatch("subtract_histograms", {parent_hist, subtraction_buffer, hist}, hp_,
                       {div_up(histogram_words_ / 4, 256), subtraction_count, 1});
    std::vector<Pair> bounds; for (auto const& node : nodes) bounds.push_back(node.bounds);
    Buffer allowed = allowed_features(nodes);
    Buffer best = workspace_.get(n * sizeof(Split));
    auto category_entries = has_categories_ ? checked_product(checked_product(n, data_.cols), data_.histogram_bins) : 0;
    require(category_entries <= UINT32_MAX, "Categorical split storage exceeds the 32-bit index range.");
    Buffer values = workspace_.get(category_entries * sizeof(u32)), ranks = workspace_.get(category_entries * sizeof(u32));
    Buffer vector_weights, screen_counters;
    bool screening = false, root_screening = false;
    bool screening_stats = learner_.integer("metal_split_screening_stats", 0) != 0;
    if (vector_) vector_weights = evaluate_vector(command, nodes, hist, allowed, best, values, ranks);
    else {
      // Keep the statistics selected by the parent, as the CUDA and CPU
      // evaluators do. Metadata changes can make them differ from the sums
      // of rows subsequently assigned to this node.
      std::vector<IntPair> node_sums;
      for (auto const& node : nodes) node_sums.push_back(node.sum);
      auto feature_nodes = checked_product(n, data_.cols);
      Buffer candidates = workspace_.get(feature_nodes * sizeof(Split));
      auto sorted_size = has_categories_ ? checked_product(checked_product(n, data_.cols), sp_.sort_stride) * sizeof(Pair) : 0;
      Buffer sorted = workspace_.get(sorted_size), scratch = workspace_.get(sorted_size);
      if (has_categories_) {
        command.dispatch("category_sort_blocks", {hist, scales_, data_.cut_counts, data_.categorical, sorted},
                         sp_, {div_up(sp_.sort_stride, 256), static_cast<std::size_t>(n) * data_.cols, 1});
        for (u32 width = 256; width < sp_.sort_stride; width *= 2) {
          SortParams sort{data_.histogram_bins, n * data_.cols, sp_.sort_stride, width};
          command.dispatch("merge_sorted", {sorted, scratch}, sort,
                           {div_up(sp_.sort_stride, 256), static_cast<std::size_t>(n) * data_.cols, 1});
          std::swap(sorted, scratch);
        }
        command.dispatch("category_sort_ranks", {sorted, data_.cut_counts, data_.categorical, ranks},
                         sp_, {div_up(data_.histogram_bins, 256), static_cast<std::size_t>(n) * data_.cols, 1});
      }
      auto split_bounds = workspace_.copy(bounds), parent_sums = workspace_.copy(node_sums);
      // Unused bindings reuse existing buffers: the (empty) category buffer
      // for counters and cutoff cells, and the Split array for parent splits.
      screen_counters = screening_stats ? screening_statistics() : values;
      if (scalar_tiles_ > 1) {
        ScalarTileParams p{sp_, scalar_tile_, scalar_tiles_, 0, 0};
        auto tile_count = checked_product(feature_nodes, scalar_tiles_);
        auto sums = workspace_.get(checked_product(tile_count, sizeof(IntPair)));
        auto carries = workspace_.get(checked_product(checked_product(feature_nodes, scalar_tiles_ + 1), sizeof(IntPair)));
        auto tiles = workspace_.get(checked_product(tile_count, sizeof(ScalarTileChoice)));
        command.dispatch("scalar_tile_sums", {hist, data_.cut_counts, data_.categorical, sums},
                         p, {data_.cols, n, scalar_tiles_}, {32, 1, 1});
        command.dispatch("scalar_tile_carries", {sums, data_.categorical, carries},
                         p, {data_.cols, n, 1}, {32, 1, 1});
        command.dispatch("evaluate_scalar_tiles", {hist, scales_, data_.cuts, data_.cut_counts, data_.cut_minima,
                         monotone_buffer_, split_bounds, allowed, candidates, data_.categorical, sorted, ranks,
                         values, parent_sums, carries, tiles}, p, {data_.cols, n, scalar_tiles_}, {32, 1, 1});
        command.dispatch("reduce_scalar_tiles", {tiles, data_.categorical, candidates},
                         p, {data_.cols, n, 1}, {32, 1, 1});
      } else if (fallback_parallel_splits) {
        // Inherited totals, feature bounds and masks are fixed throughout a
        // feature's scan. Zero integer pairs therefore duplicate the previous
        // candidate even for dense input; the sparse kernel keeps each scan
        // direction's first candidate and still materializes every histogram.
        bool fallback_skip_empty_bins = learner_.integer("metal_skip_empty_bins", 1) != 0;
        bool fallback_compact_splits = fallback_skip_empty_bins &&
            learner_.integer("metal_compact_splits", !data_.quantized_dense) != 0;
        bool fallback_fused_score = fallback_compact_splits && fallback_fast_stats &&
            learner_.integer("metal_fused_score", 1) != 0;
        char const* kernel = fallback_fast_stats ?
            (fallback_skip_empty_bins ? "evaluate_splits_parallel_sparse_subtracted_gain_stats" :
                                        "evaluate_splits_parallel_subtracted_gain_stats") :
            (fallback_skip_empty_bins ? "evaluate_splits_parallel_sparse_subtracted_gain" :
                                        "evaluate_splits_parallel_subtracted_gain");
        if (fallback_compact_splits)
          kernel = fallback_fused_score ? "evaluate_splits_compact_subtracted_gain_stats_score" :
                   fallback_fast_stats ? "evaluate_splits_compact_subtracted_gain_stats" :
                                         "evaluate_splits_compact_subtracted_gain";
        // These kernels place SplitParams at slot 14, before the two histogram
        // subtraction inputs. Bind a buffer there instead of appended bytes.
        auto split_params = workspace_.copy(std::array<SplitParams, 1>{sp_});
        // Host batches keep per-feature screening only; their node cutoff
        // cells would need a clear per batch.
        bool fallback_screening = fallback_skip_empty_bins && split_screening_eligible();
        if (fallback_screening) {
          screening = true;
          ScreenParams screen{0, 0, static_cast<u32>(screening_stats), 0};
          command.dispatch(screened_kernel(0, fallback_fast_stats, fallback_fused_score),
                           {hist, scales_, data_.cuts, data_.cut_counts, data_.cut_minima,
                            monotone_buffer_, split_bounds, allowed, candidates,
                            data_.categorical, sorted, ranks, values, parent_sums,
                            split_params, parent_hist, subtraction_buffer, values,
                            screen_counters, best}, screen, {data_.cols, n, 1}, {256, 1, 1});
        } else
          command.dispatch(kernel,
                           {hist, scales_, data_.cuts, data_.cut_counts, data_.cut_minima,
                            monotone_buffer_, split_bounds, allowed, candidates,
                            data_.categorical, sorted, ranks, values, parent_sums,
                            split_params, parent_hist, subtraction_buffer}, nullptr, 0,
                           {data_.cols, n, 1}, {256, 1, 1});
        learner_.active_tuning["fallback_parallel_splits"] = 1;
        learner_.active_tuning["split_skip_empty_bins"] = fallback_skip_empty_bins;
        learner_.active_tuning["split_threads"] = 256;
        learner_.active_tuning["fast_gain"] = 1;
        learner_.active_tuning["fast_stats"] = fallback_fast_stats;
        learner_.active_tuning["compact_splits"] = fallback_compact_splits;
        learner_.active_tuning["fused_score"] = fallback_fused_score;
      } else {
        // Reuse only the numeric root kernel: it ignores inherited sums and
        // covers at most 256 bins. The host still supplies the exact sampled
        // feature mask and monotone bounds, then selects and applies the split.
        bool parallel_root = n == 1 && nodes.front().id == 0 && nodes.front().depth == 0 &&
            parents.empty() && !sp_.use_parent_sums && parallel_splits_ && !has_categories_ &&
            data_.histogram_bins <= 256 &&
            learner_.integer("metal_fallback_parallel_root", 1) != 0 &&
            learner_.integer("metal_parallel_root", 1) != 0 &&
            learner_.integer("metal_fast_gain", 1) != 0 &&
            learner_.integer("metal_fast_weight", 0) == 0;
        if (parallel_root && root_screening_eligible()) {
          root_screening = true;
          ScreenParams screen{0, 0, static_cast<u32>(screening_stats), 0};
          command.dispatch(screened_kernel(2, false, false),
                           {hist, scales_, data_.cuts, data_.cut_counts, data_.cut_minima,
                            monotone_buffer_, split_bounds, allowed, candidates,
                            data_.categorical, sorted, ranks, values, parent_sums,
                            workspace_.copy(std::array<SplitParams, 1>{sp_}), screen_counters},
                           screen, {data_.cols, n, 1}, {256, 1, 1});
        } else
          command.dispatch(parallel_root ? "evaluate_splits_parallel_root_gain" : split_kernel(),
                           {hist, scales_, data_.cuts, data_.cut_counts, data_.cut_minima,
                            monotone_buffer_, split_bounds, allowed, candidates,
                            data_.categorical, sorted, ranks, values, parent_sums}, sp_,
                           {parallel_splits_ ? data_.cols : div_up(data_.cols, 64), n, 1},
                           {parallel_root ? 256u : parallel_splits_ ? 32u : 64u, 1, 1});
        if (parallel_root) {
          learner_.active_tuning["parallel_root"] = 1;
          learner_.active_tuning["fallback_parallel_root"] = 1;
        }
      }
      if (!sp_.use_parent_sums && data_.quantized_dense) {
        command.dispatch("select_dense_root_split", {candidates, best, scales_, split_bounds, hist,
                         data_.cut_counts, data_.cuts, data_.cut_minima}, sp_, {1, 1, 1}, {256, 1, 1});
      } else {
        command.dispatch("select_splits", {candidates, best, scales_, split_bounds}, sp_,
                         {div_up(n, 64), 1, 1}, {64, 1, 1});
      }
    }
    learner_.record(root_command ? "gradient_root" : fused_ ? "tree_level" : "histogram_split", command.wait());
    learner_.active_tuning["tree_encoders"] += command.encoder_count();
    if ((screening || root_screening) && screening_stats) record_screening(screen_counters);
    if (screening) learner_.active_tuning["split_screening"] = 1;
    if (root_screening) learner_.active_tuning["root_screening"] = 1;
    auto page = subtract_ ? std::make_shared<HistogramPage>(HistogramPage{hist, n}) : nullptr;
    std::vector<Leaf> result;
    for (u32 i = 0; i < n; ++i) {
      auto split = best.as<Split>()[i];
      std::vector<u32> categories;
      if (split.feature >= 0 && split.split_type)
        categories.assign(values.as<u32>() + split.category_begin, values.as<u32>() + split.category_begin + split.category_count);
      tree_.nodes[nodes[i].id].cover = split.total_h;
      std::vector<VectorWeight> weights;
      if (vector_) {
        auto begin = vector_weights.as<VectorWeight>() + std::size_t(i) * targets_;
        weights.assign(begin, begin + targets_);
      }
      result.push_back({nodes[i], split, std::move(categories), page, i, segments.offsets.as<u32>()[i], segments.counts.as<u32>()[i], std::move(weights)});
    }
    if (page) {
      cache_.push_back(page); cached_nodes_ += page->nodes;
      while (cached_nodes_ > cache_limit_) {
        // A leaf retains the page's identity, so an evicted parent can be
        // rebuilt on the GPU from its persistent row segment when selected.
        cached_nodes_ -= cache_.front()->nodes;
        cache_.front()->data = {};
        cache_.pop_front();
      }
    }
    return result;
  }

  Buffer evaluate_vector(Command& command, std::vector<NodePlan> const& nodes, Buffer const& hist,
                         Buffer const& allowed, Buffer const& best, Buffer const& values, Buffer const& ranks) {
    auto tile = learner_.integer("metal_vector_split_bin_tile", 256);
    require(tile == 0 || (tile >= 32 && tile <= 16384 && tile % 32 == 0),
            "metal_vector_split_bin_tile must be zero or a multiple of 32 in [32, 16384].");
    u32 max_candidates = 0;
    if (tile && data_.histogram_bins > tile) {
      auto counts = data_.cut_counts.as<u32>();
      auto categorical = data_.categorical.as<std::uint8_t>();
      for (u32 feature = 0; feature < data_.cols; ++feature) {
        auto count = counts[feature];
        auto candidates = categorical[feature] && count >= sp_.max_cat_to_onehot ?
            std::min(count, sp_.max_cat_threshold) - static_cast<u32>(count != 0) : count;
        max_candidates = std::max(max_candidates, candidates);
      }
    }
    auto tiles = tile ? std::max(1u, static_cast<u32>(div_up(max_candidates, tile))) : 1u;
    auto has_monotone = std::any_of(monotone_.begin(), monotone_.end(), [](int v) { return v != 0; });
    VectorSplitParams p{sp_, targets_, static_cast<u32>(has_monotone), tiles > 1 ? tile : 0, tiles};
    learner_.active_tuning["vector_split_bin_tile"] = tiles > 1 ? tile : 0;
    learner_.active_tuning["vector_split_bin_tiles"] = tiles;
    auto n = nodes.size();
    auto node_targets = checked_product(n, targets_);
    auto feature_nodes = checked_product(n, data_.cols);
    Buffer totals;
    auto weights = workspace_.get(checked_product(node_targets, sizeof(VectorWeight)));
    auto candidates = workspace_.get(checked_product(feature_nodes, sizeof(VectorSplitChoice)));
    auto tiled_candidates = tiles > 1 ? workspace_.get(checked_product(checked_product(feature_nodes, tiles),
                                                                      sizeof(VectorSplitChoice))) : candidates;
    auto scans = workspace_.get(checked_product(checked_product(feature_nodes, targets_),
                                               checked_product(data_.histogram_bins, 4 * sizeof(std::int64_t))));
    std::vector<Pair> node_bounds;
    for (auto const& node : nodes) node_bounds.insert(node_bounds.end(), node.vector_bounds.begin(), node.vector_bounds.end());
    auto bounds = workspace_.copy(node_bounds);
    if (sp_.use_parent_sums) {
      std::vector<IntPair> inherited;
      for (auto const& node : nodes) inherited.insert(inherited.end(), node.vector_sums.begin(), node.vector_sums.end());
      totals = workspace_.copy(inherited);
    } else {
      totals = workspace_.get(checked_product(node_targets, sizeof(IntPair)));
      command.dispatch("vector_node_sums", {hist, data_.cut_counts, totals}, p, {targets_, n, 1}, {32, 1, 1});
    }
    auto sorted_size = has_categories_ ? checked_product(checked_product(feature_nodes, sp_.sort_stride), 16) : 0;
    auto sorted = workspace_.get(sorted_size), scratch = workspace_.get(sorted_size);
    if (has_categories_) {
      command.dispatch("vector_category_sort", {hist, scales_, data_.cut_counts, data_.categorical, totals, sorted},
                       p, {div_up(sp_.sort_stride, 256), feature_nodes, 1});
      for (u32 width = 256; width < sp_.sort_stride; width *= 2) {
        SortParams sort{data_.histogram_bins, static_cast<u32>(feature_nodes), sp_.sort_stride, width};
        command.dispatch("vector_category_merge", {sorted, scratch}, sort, {div_up(sp_.sort_stride, 256), feature_nodes, 1});
        std::swap(sorted, scratch);
      }
      command.dispatch("vector_category_ranks", {sorted, data_.cut_counts, data_.categorical, ranks},
                       p, {div_up(data_.histogram_bins, 256), feature_nodes, 1});
    }
    command.dispatch("vector_histogram_scan", {hist, data_.cut_counts, data_.categorical, sorted, scans, totals},
                     p, {data_.cols, n, targets_}, {32, 1, 1});
    command.dispatch("evaluate_vector_splits", {hist, scales_, data_.cut_counts, data_.categorical, monotone_buffer_,
                                               bounds, allowed, totals, scans, tiled_candidates},
                     p, {data_.cols, n, tiles}, {32, 1, 1});
    if (tiles > 1)
      command.dispatch("reduce_vector_split_tiles", {tiled_candidates, data_.cut_counts, data_.categorical, candidates},
                       p, {data_.cols, n, 1}, {32, 1, 1});
    command.dispatch("select_vector_splits", {candidates, scales_, data_.cut_counts, data_.categorical, monotone_buffer_,
                                             bounds, totals, scans, data_.cuts, data_.cut_minima, ranks, best, weights, values},
                     p, {div_up(n, 64), 1, 1}, {64, 1, 1});
    return weights;
  }

  Buffer dispatch_arguments(u32 groups = 1) {
    return workspace_.copy(std::array<u32, 3>{0, groups, 1});
  }
  std::string_view histogram_kernel() const {
    if (histogram_missing_)
      return features_per_group_ > 1 ? "histogram_grouped_missing" : "histogram_missing";
    return features_per_group_ > 1 ? "histogram_grouped" : "histogram";
  }
  std::string_view split_kernel() const {
    if (!parallel_splits_) return "evaluate_splits";
    return split_skip_empty_bins_ ? "evaluate_splits_simd_sparse" : "evaluate_splits_simd";
  }

  Learner& learner_;
  Matrix& data_;
  Buffer const &gradient_, &scales_;
  std::mt19937& rng_;
  Tree tree_;
  float eta_;
  u32 targets_, max_depth_, max_leaves_, leaves_{1};
  float min_gain_;
  bool vector_, compact_, lossguide_, scan_, fused_, subtract_, coalesce_, compute_fill_, parallel_splits_, fixed_workspace_, histogram_missing_, has_categories_;
  bool gpu_row_init_, constant_inputs_, path_elision_;
  bool split_skip_empty_bins_{false}, fallback_wide_cuts_{false};
  u32 rows_per_tile_, features_per_group_;
  u32 scalar_tile_{0}, scalar_tiles_{1};
  std::size_t histogram_words_, cache_limit_, cached_nodes_{0};
  SplitParams sp_;
  HistParams hp_;
  Buffer rows_, scratch_rows_, ranks_, monotone_buffer_, partition_input_, partition_bounds_;
  TreeWorkspace workspace_;
  Command* root_command_;
  GpuTreeRound* async_round_;
  std::vector<int> monotone_;
  std::vector<std::vector<u32>> interactions_;
  std::vector<std::set<u32>> paths_{1};
  std::vector<u32> tree_features_;
  std::map<u32, std::vector<u32>> level_features_;
  std::priority_queue<Leaf, std::vector<Leaf>, LeafOrder> queue_;
  std::deque<std::shared_ptr<HistogramPage>> cache_;
};
}  // namespace

Tree Learner::grow(Matrix& data, Buffer const& gradient, Buffer const& scales, u32 group, float eta,
                   bool compact, u32 targets, u32 outputs, Command* root_command, GpuTreeRound* async_round) {
  bool persistent = integer("metal_tree_workspace", 1) && integer("metal_tree_persistent_workspace", 1);
  std::vector<Buffer> temporary;
  std::vector<std::vector<std::uint8_t>> temporary_inputs;
  if (!persistent) { tree_workspace_.clear(); tree_workspace_inputs_.clear(); }
  auto& workspace = persistent ? tree_workspace_ : temporary;
  auto& inputs = persistent ? tree_workspace_inputs_ : temporary_inputs;
  return TreeBuilder(*this, data, gradient, scales, rng_, group, eta, compact, targets, outputs, workspace,
                     inputs, root_command, async_round).grow();
}

Tree replay_gpu_tree(GpuTreeRound const& round, u32& evaluated_levels) {
  auto result = *round.state.as<TreeDriverState>();
  require(!result.error && result.node_count <= round.node_capacity, "GPU tree driver produced an invalid tree.");
  // Replay the original host assignments in compact node-ID order. In
  // particular, eta multiplication must preserve host subnormal rounding.
  // This is TreeBuilder::apply for the GPU driver's scalar, numeric and
  // unconstrained leaves; interaction paths are recorded only as a control.
  Tree tree{{Node{}}, round.group, round.round, {}, {0}, 1, 1, {}};
  auto const* flags = round.flags.as<u32>();
  auto const* history = round.history.as<Split>();
  std::vector<std::set<u32>> paths;
  if (round.record_paths) paths.resize(1);
  else { tree.nodes.reserve(result.node_count); tree.base_weights.reserve(result.node_count); }
  std::vector<u32> depths(result.node_count, 0);
  evaluated_levels = 0;
  for (u32 id = 0; id < result.node_count; ++id) {
    require(id < tree.nodes.size(), "GPU tree driver produced an invalid node order.");
    auto status = flags[id];
    if (!(status & kTreeDriverEvaluated)) continue;
    auto split = history[id];
    tree.nodes[id].cover = split.total_h;
    evaluated_levels = std::max(evaluated_levels, depths[id] + 1);
    if (!id) {
      tree.base_weights[0] = split.parent_weight;
      tree.nodes[0].leaf = split.parent_weight * round.eta;
    }
    if (!(status & kTreeDriverExpanded)) continue;
    require(tree.nodes.size() <= static_cast<std::size_t>(INT32_MAX) - 2, "Tree exceeds the node index range.");
    auto left = static_cast<u32>(tree.nodes.size()), right = left + 1;
    auto& node = tree.nodes[id];
    node.left = left; node.right = right; node.feature = split.feature;
    node.threshold = split.threshold; node.default_left = split.default_left; node.gain = split.gain;
    node.split_type = split.split_type;
    tree.base_weights[id] = split.applied_parent_weight;
    node.cover = split.applied_parent_hessian;
    if (split.split_type) {
      node.threshold = std::numeric_limits<float>::denorm_min();
      node.category_begin = tree.categories.size(); node.category_count = 0;
    }
    Node l, r; l.parent = id; r.parent = id;
    l.leaf = split.left_weight * round.eta; l.cover = split.left_h;
    r.leaf = split.right_weight * round.eta; r.cover = split.right_h;
    tree.nodes.push_back(l); tree.nodes.push_back(r);
    tree.base_weights.push_back(l.leaf); tree.base_weights.push_back(r.leaf);
    if (round.record_paths) {
      auto path = paths[id]; path.insert(split.feature); paths.push_back(path); paths.push_back(path);
    }
    for (auto child : {left, right}) {
      require(child < result.node_count, "GPU tree driver produced invalid children.");
      depths[child] = depths[id] + 1;
    }
  }
  require(tree.nodes.size() == result.node_count, "GPU tree driver node count differs from replay.");
  return tree;
}

std::uint64_t validate_leaf_scaling(float eta, std::uint64_t begin, std::uint64_t count, std::uint32_t* first_mismatch) {
  require(begin + count <= (std::uint64_t{1} << 32) && count <= (std::uint64_t{1} << 32),
          "Leaf-scaling sweep exceeds the binary32 range.");
  constexpr std::uint64_t chunk = std::uint64_t{1} << 24;
  Buffer output(chunk * sizeof(u32));
  std::uint64_t mismatches = 0;
  for (std::uint64_t offset = 0; offset < count; offset += chunk) {
    auto size = static_cast<u32>(std::min(chunk, count - offset));
    auto first = static_cast<u32>(begin + offset);
    Command command("leaf scaling validation");
    command.dispatch("tree_emit_multiply_sweep", {output}, TreeEmitSweepParams{first, size, eta, 0},
                     {div_up(size, 256), 1, 1});
    command.wait();
    auto const* gpu = output.as<u32>();
    for (u32 i = 0; i < size; ++i) {
      // The host model's standalone binary32 product (tree assembly above).
      auto expected = std::bit_cast<u32>(std::bit_cast<float>(first + i) * eta);
      if (gpu[i] != expected && !mismatches++ && first_mismatch) *first_mismatch = first + i;
    }
  }
  return mismatches;
}

void Learner::update_vector_leaves(Tree& tree, Matrix const& data, Buffer const& gradient,
                                    Buffer const& scales, float eta) {
  SplitParams split{};
  split.nodes = tree.nodes.size(); split.cols = data.cols;
  split.reg_lambda = number("reg_lambda", 1); split.reg_alpha = number("reg_alpha", 0);
  split.max_delta_step = number("max_delta_step", 0);
  VectorLeafParams p{split, data.rows, groups, eta, 0};
  auto nodes = upload(tree.nodes), categories = upload(tree.categories);
  Buffer sums(checked_product(checked_product(tree.nodes.size(), groups), 4 * sizeof(u32)));
  Buffer weights(checked_product(tree.leaf_weights.size(), sizeof(float)));
  Command command("vector leaf value gradients", integer("metal_coalesce_encoders", 0), integer("metal_compute_fill", 0));
  command.zero(sums);
  auto bounds = data.partition_category_bounds(); p.padding = static_cast<u32>(static_cast<bool>(bounds));
  command.dispatch("vector_leaf_sums", {data.partition_data(), nodes, categories, gradient, sums, bounds ? bounds : scales}, p, {div_up(data.rows, 256), groups, 1});
  command.dispatch("vector_leaf_weights", {nodes, sums, scales, weights}, p, {div_up(tree.nodes.size(), 256), groups, 1});
  record("vector_leaf_update", command.wait());
  std::memcpy(tree.leaf_weights.data(), weights.data(), tree.leaf_weights.size() * sizeof(float));
}
}  // namespace metalgb
