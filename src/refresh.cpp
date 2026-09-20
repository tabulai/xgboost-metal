#include "metalgb/engine.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <sstream>

namespace metalgb {
bool Learner::has_refresh() const {
  return std::find(tree_updaters_.begin(), tree_updaters_.end(), "refresh") != tree_updaters_.end();
}
void Learner::clear_tree_updater() {
  // The pinned learner configures each argument batch separately. A batch
  // without `updater` selects the tree method's default updater again.
  params.erase("updater"); params.erase("updater_seq");
  tree_updaters_ = {"grow_metal_hist"};
}

void Learner::configure_tree() {
  if (auto it = params.find("max_bin"); it != params.end()) {
    std::string_view value = it->second;
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.remove_prefix(1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.remove_suffix(1);
    if (!value.empty() && value.front() == '+') value.remove_prefix(1);
    std::int32_t bins{};
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), bins);
    require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && bins >= 2,
            "max_bin must be an integer in [2, 2147483647].");
    it->second = std::to_string(bins);
  }
  auto process = str("process_type", "default");
  require(process == "default" || process == "update", "process_type must be default or update.");
  std::istringstream sequence(str("updater", "grow_metal_hist"));
  std::vector<std::string> updaters;
  for (std::string name; std::getline(sequence, name, ',');) {
    if (name == "grow_quantile_histmaker" || name == "grow_gpu_hist") name = "grow_metal_hist";
    require(name == "grow_metal_hist" || name == "refresh" || name == "prune",
            "Unknown or unsupported Metal tree updater: " + name);
    require(name != "grow_metal_hist" || updaters.empty(), "The grow updater must be first in the updater sequence.");
    updaters.push_back(std::move(name));
  }
  require(!updaters.empty(), "The tree updater sequence must not be empty.");
  tree_updaters_ = std::move(updaters);
  if (params.contains("updater")) {
    static std::once_flag warned;
    std::call_once(warned, [this] {
      if (integer("verbosity", 1)) warning("You have manually specified the `updater` parameter. The `tree_method` parameter will be ignored. Incorrect sequence of updaters will produce undefined behavior. For common uses, we recommend using `tree_method` parameter instead.");
    });
  }
  (void)boolean("refresh_leaf", true);
  if (process == "update" && trees_to_update_.empty()) {
    update_original_weights_ = !tree_weights_.empty();
    trees_to_update_ = std::move(trees);
    trees.clear(); rounds = 0;
    reset(false);
  }
}

void Learner::validate_tree_updaters() const {
  bool vector = groups > 1 && str("multi_strategy", "one_output_per_tree") == "multi_output_tree";
  if (vector) for (auto const& name : tree_updaters_)
    require(name == "grow_metal_hist", name + " is not supported with multi-target trees by the reference updater.");
  if (is_tree_update()) {
    for (auto const& name : tree_updaters_)
      require(name != "grow_metal_hist", "The grow updater cannot modify existing trees. Set process_type to default to build new trees.");
    auto layer = checked_product(groups, integer("num_parallel_tree", 1));
    require(trees.size() <= trees_to_update_.size() && layer <= trees_to_update_.size() - trees.size(),
            "No more tree left for updating. Boosting rounds cannot exceed previous training rounds.");
    require(!update_original_weights_, "Updating a model with stored dropout weights is not supported by the reference tree updater.");
  } else {
    require(tree_updaters_.front() == "grow_metal_hist",
            "A refresh or prune updater cannot create new trees. Set process_type to update to modify existing trees.");
  }
  if (has_refresh()) {
    auto text = str("monotone_constraints", "()");
    if (!text.empty() && text != "()") {
      std::replace(text.begin(), text.end(), '(', '['); std::replace(text.begin(), text.end(), ')', ']');
      auto constraints = parse_json(text).get<std::vector<int>>();
      require(std::all_of(constraints.begin(), constraints.end(), [](int value) { return value == 0; }),
              "Monotonic constraint is not supported by the refresh updater.");
    }
  }
}

Buffer Learner::refresh_buffer(std::string key, std::size_t bytes) {
  auto& buffer = refresh_workspace_[std::move(key)];
  if (!buffer || buffer.size() < bytes) buffer = Buffer(bytes);
  return buffer.prefix(bytes);
}

void Learner::modify_tree(Tree& tree, Matrix const& data, Buffer const& gradient, float eta,
                          std::string_view updater, bool compact) {
  require(tree.outputs == 1, std::string(updater) + " is not supported with multi-target trees by the reference updater.");
  struct Level { u32 begin, count; };
  std::vector<Level> levels;
  std::vector<u32> parents, leaves, live, frontier{0};
  std::size_t max_parents = 0;
  while (!frontier.empty()) {
    auto begin = parents.size();
    std::vector<u32> next;
    for (auto id : frontier) {
      live.push_back(id);
      auto const& node = tree.nodes[id];
      if (node.left < 0) leaves.push_back(id);
      else {
        parents.push_back(id); next.push_back(node.left); next.push_back(node.right);
      }
    }
    levels.push_back({static_cast<u32>(begin), static_cast<u32>(parents.size() - begin)});
    max_parents = std::max(max_parents, parents.size() - begin);
    frontier = std::move(next);
  }
  auto copy = [&]<class T>(std::string key, std::vector<T> const& values) {
    auto buffer = refresh_buffer(std::move(key), checked_product(values.size(), sizeof(T)));
    if (!values.empty()) std::memcpy(buffer.data(), values.data(), buffer.size());
    return buffer;
  };
  auto nodes = copy("nodes", tree.nodes), weights = copy("weights", tree.base_weights);
  auto parent_ids = copy("parents", parents);
  RefreshParams p{data.rows, data.cols, compact ? 1u : groups, compact ? 0u : tree.group,
                   0, static_cast<u32>(tree.nodes.size()), 0, 0,
                   eta, number("reg_alpha", 0), number("reg_lambda", 1), number("max_delta_step", 0),
                   number("min_split_loss", 0), integer("max_depth", 6), boolean("refresh_leaf", true), 0};
  Command command(updater, integer("metal_coalesce_encoders", 0), integer("metal_compute_fill", 0));
  Buffer transposed_source;
  if (updater == "prune") {
    for (std::size_t depth = levels.size(); depth-- > 0;) {
      p.begin = levels[depth].begin; p.count = levels[depth].count; p.depth = depth;
      command.dispatch("prune_level", {parent_ids, nodes, weights}, p, {div_up(p.count, 64), 1, 1}, {64, 1, 1});
    }
  } else {
    auto row_bytes = checked_product(data.rows, sizeof(u32));
    auto node_bytes = checked_product(tree.nodes.size(), sizeof(u32));
    auto rows = refresh_buffer("rows", row_bytes), scratch = refresh_buffer("scratch", row_bytes);
    auto ranks = refresh_buffer("ranks", row_bytes);
    auto counts = refresh_buffer("counts", node_bytes), offsets = refresh_buffer("offsets", node_bytes);
    auto capacity = std::max(div_up(data.rows, 256) + max_parents,
                             div_up(data.rows, kRefreshRowsPerTile) + leaves.size());
    require(capacity <= UINT32_MAX, "Refresh task count exceeds the dispatch index range.");
    auto tasks = refresh_buffer("tasks", checked_product(capacity, sizeof(PartitionTask)));
    auto segments = refresh_buffer("segments", checked_product(max_parents, sizeof(PartitionSegment)));
    auto block_counts = refresh_buffer("block_counts", checked_product(capacity, sizeof(u32)));
    auto arguments = refresh_buffer("arguments", 3 * sizeof(u32));
    std::array<u32, 3> initial_arguments{0, 1, 1};
    std::memcpy(arguments.data(), initial_arguments.data(), sizeof(initial_arguments));
    auto category_buffer = copy("categories", tree.categories);
    // Like histogram growth, refresh traverses the matrix's stored category
    // codes. Prediction separately maps category names into the model's codes.
    auto input = data.data;
    p.column_major = boolean("metal_refresh_columns", false);
    if (p.column_major && !parents.empty()) {
      auto columns = refresh_buffer("columns", input.size());
      if (refresh_matrix_.lock().get() != &data || refresh_source_.native() != input.native()) {
        LinearParams transpose{}; transpose.rows = data.rows; transpose.cols = data.cols;
        command.dispatch("linear_columns", {input, columns}, transpose,
                         {div_up(data.rows, 32), div_up(data.cols, 32), 1}, {32, 8, 1});
        transposed_source = input;
      }
      input = columns;
    }
    active_tuning["refresh_columns"] = p.column_major;
    command.dispatch("refresh_rows", {rows, counts, offsets}, p, {div_up(std::max(p.rows, p.nodes), 256), 1, 1});
    for (auto level : levels) {
      if (!level.count) continue;
      p.begin = level.begin; p.count = level.count;
      command.zero(arguments.prefix(sizeof(u32)));
      command.dispatch("refresh_partition_plan", {parent_ids, nodes, counts, offsets, tasks, segments, arguments},
                       p, {div_up(p.count, 64), 1, 1}, {64, 1, 1});
      PartitionParams pp{data.rows, data.cols, 2 * p.count, 0};
      command.dispatch_indirect("refresh_partition_flags", {input, rows, nodes, tasks, ranks, block_counts, category_buffer}, p, arguments);
      command.dispatch("partition_segment_scan", {segments, block_counts, counts, offsets}, pp, {div_up(p.count, 64), 1, 1}, {64, 1, 1});
      command.dispatch_indirect("partition_segment_scatter", {rows, tasks, segments, ranks, block_counts, offsets, scratch}, pp, arguments);
      // Completed leaves retain their disjoint segments while deeper parents
      // continue partitioning, just as deferred leaves do in loss-guided growth.
      command.dispatch_indirect("partition_segment_copy", {scratch, tasks, rows}, pp, arguments);
    }
    auto leaf_ids = copy("leaves", leaves), live_ids = copy("live", live);
    auto ranges = refresh_buffer("ranges", checked_product(leaves.size(), 2 * sizeof(u32)));
    auto partial = refresh_buffer("partial", checked_product(capacity, kSurvivalSumBytes));
    auto sums = refresh_buffer("sums", checked_product(tree.nodes.size(), kSurvivalSumBytes));
    p.count = leaves.size();
    command.zero(arguments.prefix(sizeof(u32)));
    command.dispatch("refresh_sum_plan", {leaf_ids, counts, offsets, tasks, ranges, arguments}, p, {div_up(p.count, 64), 1, 1}, {64, 1, 1});
    command.dispatch_indirect("refresh_partial", {gradient, rows, tasks, partial}, p, arguments);
    command.dispatch("refresh_leaf_sums", {partial, ranges, leaf_ids, sums}, p, {p.count, 1, 1});
    for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
      p.begin = it->begin; p.count = it->count;
      command.dispatch("refresh_parent_sums", {parent_ids, nodes, sums}, p, {div_up(p.count, 64), 1, 1}, {64, 1, 1});
    }
    p.count = live.size();
    command.dispatch("refresh_apply", {live_ids, nodes, sums, weights}, p, {div_up(p.count, 64), 1, 1}, {64, 1, 1});
  }
  record(std::string(updater), command.wait());
  if (transposed_source) { refresh_source_ = transposed_source; refresh_matrix_ = data.weak_from_this(); }
  std::memcpy(tree.nodes.data(), nodes.data(), nodes.size());
  std::memcpy(tree.base_weights.data(), weights.data(), weights.size());
}
}  // namespace metalgb
