#include "metalgb/engine.hpp"
#include <algorithm>
#include <array>
#include <deque>
#include <numeric>
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
  TreeWorkspace(bool reuse, std::vector<Buffer>& buffers) : reuse_(reuse), buffers_(buffers) {}
  // A batch owns distinct slots until its command completes and its results
  // have been copied into leaves. Cached histograms never use these slots.
  void reset() { next_ = 0; }
  Buffer get(std::size_t bytes) {
    ++requests;
    if (!reuse_) { ++allocations; return Buffer(bytes); }
    if (next_ == buffers_.size()) buffers_.emplace_back();
    auto& buffer = buffers_[next_++];
    if (!buffer || buffer.size() < bytes) { buffer = Buffer(bytes); ++allocations; }
    return buffer.prefix(bytes);
  }
  template<class Container> Buffer copy(Container const& values) {
    auto bytes = checked_product(values.size(), sizeof(*values.data()));
    auto buffer = get(bytes);
    if (bytes) std::memcpy(buffer.data(), values.data(), bytes);
    return buffer;
  }
  u32 requests{0}, allocations{0};
 private:
  bool reuse_;
  std::size_t next_{0};
  std::vector<Buffer>& buffers_;
};

class TreeBuilder {
 public:
  TreeBuilder(Learner& learner, Matrix& data, Buffer const& gradient, Buffer const& scales,
              std::mt19937& rng, u32 group, float eta, bool compact, u32 targets, u32 outputs,
              std::vector<Buffer>& workspace)
      : learner_(learner), data_(data), gradient_(gradient), scales_(scales), rng_(rng),
        tree_{{Node{}}, group, learner.rounds, {}, {0}, 1, 1, {}}, eta_(eta), targets_(targets),
        max_depth_(learner.integer("max_depth", 6)), max_leaves_(learner.integer("max_leaves", 0)),
        min_gain_(learner.number("min_split_loss", 0)),
        vector_(outputs > 1),
        lossguide_(learner.str("grow_policy", "depthwise") == "lossguide"),
        scan_(learner.integer("metal_partition_scan", 1) != 0),
        fused_(scan_ && learner.integer("metal_fused_partition", 1) != 0),
        subtract_(learner.integer("metal_histogram_subtraction", 1) != 0),
        coalesce_(learner.integer("metal_coalesce_encoders", 0) != 0),
        compute_fill_(learner.integer("metal_compute_fill", 0) != 0),
        parallel_splits_(learner.integer("metal_parallel_splits", 1) != 0),
        workspace_(learner.integer("metal_tree_workspace", 1) != 0, workspace),
        queue_(LeafOrder{lossguide_}) {
    partition_input_ = data.partition_data();
    partition_bounds_ = data.partition_category_bounds();
    auto cache = learner.str("max_cached_hist_node", "18446744073709551615");
    std::size_t read;
    require(!cache.empty() && cache[0] != '-', "max_cached_hist_node must be positive.");
    auto limit = std::stoull(cache, &read);
    require(read == cache.size() && limit > 0, "max_cached_hist_node must be a positive integer.");
    cache_limit_ = limit == UINT64_MAX ? 4096 : limit;
    monotone_ = monotone_constraints(learner.str("monotone_constraints"), data.cols);
    monotone_buffer_ = upload(monotone_);
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
    std::vector<u32> initial_rows(data.rows); std::iota(initial_rows.begin(), initial_rows.end(), 0);
    rows_ = upload(initial_rows);
    scratch_rows_ = Buffer(static_cast<std::size_t>(data.rows) * sizeof(u32));
    ranks_ = Buffer(static_cast<std::size_t>(data.rows) * sizeof(u32));
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
    }
    bool large = data.rows >= 262144 && data.cols >= 32;
    rows_per_tile_ = learner.integer("metal_histogram_rows", large ? 4096 : 2048);
    features_per_group_ = learner.integer("metal_features_per_group", large && data.histogram_bins <= 256 ? 4 : 1);
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
                            {"tree_persistent_workspace", learner.integer("metal_tree_workspace", 1) &&
                                                          learner.integer("metal_tree_persistent_workspace", 1)},
                            {"parallel_splits", parallel_splits_},
                            {"scalar_split_bin_tile", scalar_tile_}, {"scalar_split_bin_tiles", scalar_tiles_},
                            {"quantile_cut_tile", data.cut_tile_size}, {"quantile_cut_tiles", data.cut_tile_count},
                            {"quantile_summary_group", data.summary_group_size}, {"quantile_summary_batches", data.summary_batch_count},
                            {"quantile_weighted_summary_stages", data.weighted_summary_stages},
                            {"quantile_weighted_summary_batches", data.weighted_summary_batch_count},
                            {"tree_buffer_requests", 0}, {"tree_buffer_allocations", 0}};
  }

  Tree grow() {
    bool constrained = std::any_of(monotone_.begin(), monotone_.end(), [](auto value) { return value != 0; });
    float bound = constrained ? std::numeric_limits<float>::max() : std::numeric_limits<float>::infinity();
    NodePlan initial{0, 0, {-bound, bound}, {}, {}, {}};
    if (vector_) initial.vector_bounds.assign(targets_, initial.bounds);
    auto root = evaluate({initial}, {}).front();
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

  std::vector<Leaf> evaluate(std::vector<NodePlan> const& nodes, std::vector<Leaf> const& parents) {
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
    Command command(fused_ ? "partition, histogram and split search" : "histogram and split search", coalesce_, compute_fill_);
    if (!parents.empty() && fused_) segments = partition(command, parents);
    auto [parent_hist, parent_indices] = parent_histograms(command, parents);
    auto subtraction_count = subtract_ ? parents.size() : 0;
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
    if (subtraction_count)
      command.dispatch("subtract_histograms", {parent_hist, subtraction_buffer, hist}, hp_,
                       {div_up(histogram_words_ / 4, 256), subtraction_count, 1});
    std::vector<Pair> bounds; for (auto const& node : nodes) bounds.push_back(node.bounds);
    Buffer allowed = allowed_features(nodes);
    Buffer best = workspace_.get(n * sizeof(Split));
    auto category_entries = has_categories_ ? checked_product(checked_product(n, data_.cols), data_.histogram_bins) : 0;
    require(category_entries <= UINT32_MAX, "Categorical split storage exceeds the 32-bit index range.");
    Buffer values = workspace_.get(category_entries * sizeof(u32)), ranks = workspace_.get(category_entries * sizeof(u32));
    Buffer vector_weights;
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
      } else {
        command.dispatch(parallel_splits_ ? "evaluate_splits_simd" : "evaluate_splits", {hist, scales_, data_.cuts, data_.cut_counts, data_.cut_minima,
                                            monotone_buffer_, split_bounds, allowed, candidates,
                                            data_.categorical, sorted, ranks, values, parent_sums}, sp_,
                         {parallel_splits_ ? data_.cols : div_up(data_.cols, 64), n, 1}, {parallel_splits_ ? 32u : 64u, 1, 1});
      }
      if (!sp_.use_parent_sums && data_.quantized_dense) {
        command.dispatch("select_dense_root_split", {candidates, best, scales_, split_bounds, hist,
                         data_.cut_counts, data_.cuts, data_.cut_minima}, sp_, {1, 1, 1}, {256, 1, 1});
      } else {
        command.dispatch("select_splits", {candidates, best, scales_, split_bounds}, sp_,
                         {div_up(n, 64), 1, 1}, {64, 1, 1});
      }
    }
    learner_.record(fused_ ? "tree_level" : "histogram_split", command.wait());
    learner_.active_tuning["tree_encoders"] += command.encoder_count();
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
  std::string_view histogram_kernel() const { return features_per_group_ > 1 ? "histogram_grouped" : "histogram"; }

  Learner& learner_;
  Matrix& data_;
  Buffer const &gradient_, &scales_;
  std::mt19937& rng_;
  Tree tree_;
  float eta_;
  u32 targets_, max_depth_, max_leaves_, leaves_{1};
  float min_gain_;
  bool vector_, lossguide_, scan_, fused_, subtract_, coalesce_, compute_fill_, parallel_splits_, has_categories_;
  u32 rows_per_tile_, features_per_group_;
  u32 scalar_tile_{0}, scalar_tiles_{1};
  std::size_t histogram_words_, cache_limit_, cached_nodes_{0};
  SplitParams sp_;
  HistParams hp_;
  Buffer rows_, scratch_rows_, ranks_, monotone_buffer_, partition_input_, partition_bounds_;
  TreeWorkspace workspace_;
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
                   bool compact, u32 targets, u32 outputs) {
  bool persistent = integer("metal_tree_workspace", 1) && integer("metal_tree_persistent_workspace", 1);
  std::vector<Buffer> temporary;
  if (!persistent) tree_workspace_.clear();
  auto& workspace = persistent ? tree_workspace_ : temporary;
  return TreeBuilder(*this, data, gradient, scales, rng_, group, eta, compact, targets, outputs, workspace).grow();
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
