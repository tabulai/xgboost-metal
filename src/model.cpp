#include "metalgb/engine.hpp"
#include <algorithm>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace metalgb {
namespace {
u32 string_uint(Json const& value) { return value.is_string() ? static_cast<u32>(std::stoul(value.get<std::string>())) : value.get<u32>(); }
std::string json_string(Json const& value) { return value.is_string() ? value.get<std::string>() : value.dump(); }
Json objective_config(Learner const& learner) {
  Json out{{"name", learner.str("objective", "reg:squarederror")}};
  auto objective = learner.objective();
  if (objective >= 17) return learner.ranking_config();
  if (objective == 1 || objective == 2) out["reg_loss_param"] = {{"scale_pos_weight", learner.str("scale_pos_weight", "1")}};
  if (objective == 3 || objective == 4) out["softmax_multiclass_param"] = {{"num_class", std::to_string(learner.num_class)}};
  if (objective == 8) out["tweedie_regression_param"] = {{"tweedie_variance_power", learner.str("tweedie_variance_power", "1.5")}};
  if (objective == 9) out["pseudo_huber_param"] = {{"huber_slope", learner.str("huber_slope", "1")}};
  if (objective == 12) out["quantile_loss_param"] = {{"quantile_alpha", Json(learner.loss_alphas("quantile")).dump()}};
  if (objective == 13) out["expectile_loss_param"] = {{"expectile_alpha", Json(learner.loss_alphas("expectile")).dump()}};
  if (objective == 15) out["aft_loss_param"] = {
      {"aft_loss_distribution", learner.str("aft_loss_distribution", "normal")},
      {"aft_loss_distribution_scale", learner.str("aft_loss_distribution_scale", "1")}};
  return out;
}
}
Json Learner::model() {
  drain();
  configure();
  require(tree_weights_.size() <= trees.size(), "Tree weights cannot exceed the number of active trees.");
  Json serialized = Json::array(), tree_info = Json::array(), iteration_ptr = Json::array();
  for (u32 t = 0; t < trees.size(); ++t) {
    auto const& tree = trees[t];
    auto deleted = std::count_if(tree.nodes.begin(), tree.nodes.end(), [](auto const& node) {
      return node.feature == INT32_MAX && node.default_left;
    });
    Json value{{"id", t}, {"tree_param", {{"num_deleted", std::to_string(deleted)}, {"num_feature", std::to_string(num_feature)},
                 {"num_nodes", std::to_string(tree.nodes.size())}, {"size_leaf_vector", std::to_string(tree.outputs)}}}};
    for (auto name : {"base_weights", "categories", "categories_nodes", "categories_segments", "categories_sizes", "default_left", "left_children", "right_children", "loss_changes", "parents", "split_conditions", "split_indices", "split_type", "sum_hessian"}) value[name] = Json::array();
    for (u32 id = 0; id < tree.nodes.size(); ++id) {
      auto const& n = tree.nodes[id];
      value["default_left"].push_back(n.default_left);
      value["left_children"].push_back(n.left); value["right_children"].push_back(n.right);
      value["loss_changes"].push_back(n.gain); value["parents"].push_back(n.parent < 0 && tree.outputs == 1 ? 2147483647 : n.parent);
      value["split_conditions"].push_back(n.left < 0 && tree.outputs == 1 ? n.leaf : n.threshold);
      value["split_indices"].push_back(n.feature < 0 ? 0 : n.feature);
      value["split_type"].push_back(n.split_type); value["sum_hessian"].push_back(n.cover);
      if (n.split_type) {
        value["categories_nodes"].push_back(id);
        value["categories_segments"].push_back(value["categories"].size());
        value["categories_sizes"].push_back(n.category_count);
        for (u32 i = 0; i < n.category_count; ++i) value["categories"].push_back(tree.categories.at(n.category_begin + i));
      }
    }
    value["base_weights"] = tree.base_weights;
    if (tree.outputs > 1) value["leaf_weights"] = tree.leaf_weights;
    serialized.push_back(std::move(value)); tree_info.push_back(tree.group);
  }
  std::size_t cursor = 0;
  for (u32 round = 0; round <= rounds; ++round) {
    while (cursor < trees.size() && trees[cursor].round < round) ++cursor;
    iteration_ptr.push_back(cursor);
  }
  Json booster_model{{"trees", serialized}, {"tree_info", tree_info}, {"iteration_indptr", iteration_ptr},
    {"gbtree_model_param", {{"num_parallel_tree", str("num_parallel_tree", "1")}, {"num_trees", std::to_string(trees.size())}}},
    {"cats", serialize_categories(categories)}};
  if (!tree_weights_.empty()) booster_model["weight_drop"] = tree_weights_;
  if (is_linear()) booster_model = {{"weights", linear_weights}, {"boosted_rounds", rounds}};
  Json learner{{"attributes", attributes}, {"feature_names", feature_names}, {"feature_types", feature_types},
    {"learner_model_param", {{"base_score", Json(base_score).dump()}, {"boost_from_average", str("boost_from_average", params.contains("base_score") ? "0" : "1")},
          {"num_class", std::to_string(num_class)}, {"num_feature", std::to_string(num_feature)}, {"num_target", std::to_string(num_target)}}},
    {"objective", objective_config(*this)}, {"gradient_booster", {{"name", is_linear() ? "gblinear" : "gbtree"}, {"model", booster_model}}}};
  return {{"version", {3, 5, 0}}, {"learner", learner}};
}
void Learner::load_model(Json const& input) {
  drain();
  auto const& learner = input.at("learner");
  auto const& booster = learner.at("gradient_booster");
  auto name = booster.at("name").get<std::string>();
  require(name == "gbtree" || name == "dart" || name == "gblinear", "Model booster is not supported: " + name);
  auto const& mp = learner.at("learner_model_param");
  auto new_features = string_uint(mp.at("num_feature"));
  auto new_classes = string_uint(mp.at("num_class"));
  auto new_targets = mp.contains("num_target") ? string_uint(mp.at("num_target")) : 1;
  auto new_groups = std::max({1u, new_classes, new_targets});
  auto score_text = json_string(mp.at("base_score"));
  std::vector<float> scores = score_text.starts_with('[') ? parse_json(score_text).get<std::vector<float>>() : std::vector<float>{std::stof(score_text)};
  if (scores.size() == 1) scores.resize(new_groups, scores[0]);
  require(scores.size() == new_groups, "Invalid model base score shape.");
  auto const& tree_booster = name == "dart" ? booster.at("gbtree") : booster;
  auto const& source = tree_booster.at("model");
  auto loaded_categories = source.contains("cats") ? deserialize_categories(source["cats"]) : Json::array();
  require(loaded_categories.empty() || loaded_categories.size() == new_features, "Invalid category feature count in model.");
  std::vector<Tree> loaded;
  std::vector<float> loaded_linear, weights;
  u32 parallel = 1, new_rounds = 0;
  if (name == "gblinear") {
    loaded_linear = source.at("weights").get<std::vector<float>>();
    require(loaded_linear.empty() || loaded_linear.size() == checked_product(static_cast<std::size_t>(new_features) + 1, new_groups),
            "Invalid linear model weight shape.");
    auto count = source.value("boosted_rounds", std::int64_t{0});
    require(count >= 0 && count <= INT32_MAX, "Invalid linear model boosted_rounds.");
    new_rounds = count;
  } else {
  auto const& source_trees = source.at("trees");
  weights = source.value("weight_drop", std::vector<float>{});
  // Legacy DART stored weights beside its nested gbtree model. Early gbtree
  // versions also accepted them beside the model object.
  if (booster.contains("weight_drop")) weights = booster["weight_drop"].get<std::vector<float>>();
  require(weights.empty() || weights.size() == source_trees.size(), "Tree weights must have one entry per tree.");
  require(std::all_of(weights.begin(), weights.end(), [](float w) { return std::isfinite(w); }), "Tree weights must be finite.");
  auto const& info = source.at("tree_info");
  require(info.size() == source_trees.size(), "Invalid tree_info length.");
  std::vector<u32> round_ptr;
  if (source.contains("iteration_indptr")) round_ptr = source["iteration_indptr"].get<std::vector<u32>>();
  parallel = string_uint(source.at("gbtree_model_param").at("num_parallel_tree"));
  require(parallel > 0, "Invalid num_parallel_tree in model.");
  if (round_ptr.empty()) {
    bool vector = !source_trees.empty() && string_uint(source_trees[0].at("tree_param").at("size_leaf_vector")) > 1;
    auto per_round = (vector ? 1u : new_groups) * parallel;
    require(source_trees.size() % per_round == 0, "Invalid model tree count.");
    for (u32 i = 0; i <= source_trees.size(); i += per_round) round_ptr.push_back(i);
  }
  require(!round_ptr.empty() && round_ptr.front() == 0 && round_ptr.back() == source_trees.size() && std::is_sorted(round_ptr.begin(), round_ptr.end()), "Invalid iteration pointers.");
  for (u32 t = 0; t < source_trees.size(); ++t) {
    auto const& value = source_trees[t];
    auto count = string_uint(value.at("tree_param").at("num_nodes"));
    require(count > 0, "A tree must have a root node.");
    for (auto field : {"left_children", "right_children", "parents", "split_conditions", "split_indices", "default_left", "sum_hessian", "loss_changes"}) require(value.at(field).size() == count, "Inconsistent node-array length.");
    if (value.contains("split_type")) require(value["split_type"].size() == count, "Invalid model split-type array.");
    Tree tree; tree.group = info[t].get<u32>();
    tree.outputs = std::max(1u, string_uint(value.at("tree_param").at("size_leaf_vector")));
    require(tree.outputs == 1 || tree.outputs == new_groups, "Vector leaf width must match model output groups.");
    bool legacy_vector = tree.outputs > 1 && !value.contains("leaf_weights");
    if (tree.outputs > 1 && !legacy_vector) {
      tree.leaf_weights = value["leaf_weights"].get<std::vector<float>>();
      require(!tree.leaf_weights.empty() && tree.leaf_weights.size() % tree.outputs == 0,
              "Invalid vector leaf-weight array.");
    }
    if (value.contains("base_weights")) {
      tree.base_weights = value["base_weights"].get<std::vector<float>>();
      require(tree.base_weights.size() % count == 0, "Invalid model base-weight array.");
      tree.split_outputs = tree.base_weights.size() / count;
      require(tree.split_outputs > 0 && tree.split_outputs <= tree.outputs, "Invalid model split-output width.");
      require(!legacy_vector || tree.split_outputs == tree.outputs, "Legacy vector model requires full-width node weights.");
    } else {
      require(tree.outputs == 1, "Vector models require base weights.");
      tree.base_weights.resize(count);
    }
    require(tree.group < new_groups, "Tree output group is out of range.");
    tree.round = std::upper_bound(round_ptr.begin(), round_ptr.end(), t) - round_ptr.begin() - 1;
    tree.nodes.resize(count);
    for (u32 i = 0; i < count; ++i) {
      Node& n = tree.nodes[i];
      n.left = value["left_children"][i].get<i32>(); n.right = value["right_children"][i].get<i32>();
      n.parent = value["parents"][i].get<i32>(); if (i == 0) n.parent = -1;
      n.feature = value["split_indices"][i].get<i32>();
      n.default_left = value["default_left"][i].is_boolean() ? value["default_left"][i].get<bool>() : value["default_left"][i].get<u32>();
      n.cover = value["sum_hessian"][i].get<float>(); n.gain = value["loss_changes"][i].get<float>();
      n.threshold = value["split_conditions"][i].get<float>();
      n.split_type = value.contains("split_type") ? value["split_type"][i].get<u32>() : 0;
      require(n.split_type <= 1, "Invalid model split type.");
      if (n.left < 0) {
        if (tree.outputs > 1) {
          if (legacy_vector) {
            n.right = tree.leaf_weights.size() / tree.outputs;
            auto begin = tree.base_weights.begin() + std::size_t(i) * tree.outputs;
            tree.leaf_weights.insert(tree.leaf_weights.end(), begin, begin + tree.outputs);
          }
          require(n.right >= 0 && std::size_t(n.right) < tree.leaf_weights.size() / tree.outputs,
                  "Vector leaf-weight index is out of bounds.");
        } else { require(n.right < 0, "Invalid leaf child."); n.leaf = n.threshold; }
      }
      else require(n.right >= 0 && static_cast<u32>(n.left) < count && static_cast<u32>(n.right) < count && n.feature >= 0 && static_cast<u32>(n.feature) < new_features, "Model contains an out-of-range tree index.");
      if (!value.contains("base_weights")) tree.base_weights[i] = n.leaf;
    }
    std::vector<std::uint8_t> categorical_seen(count, 0);
    if (value.contains("categories_nodes")) {
      auto const& nodes = value["categories_nodes"];
      auto const& segments = value.at("categories_segments");
      auto const& sizes = value.at("categories_sizes");
      auto const& categories = value.at("categories");
      require(nodes.size() == segments.size() && nodes.size() == sizes.size(), "Invalid model categorical segments.");
      for (u32 i = 0; i < nodes.size(); ++i) {
        auto id = nodes[i].get<u32>(), begin = segments[i].get<u32>(), size = sizes[i].get<u32>();
        require(id < count && !categorical_seen[id] && tree.nodes[id].split_type == 1,
                "Invalid categorical node in model.");
        require(std::uint64_t(begin) + size <= categories.size(), "Categorical model segment is out of bounds.");
        categorical_seen[id] = 1;
        std::vector<u32> selected;
        for (u32 j = 0; j < size; ++j) {
          auto category = categories[begin + j].get<u32>();
          require(category < 16777216, "Invalid category value in model.");
          selected.push_back(category);
        }
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
        tree.nodes[id].category_begin = tree.categories.size();
        tree.nodes[id].category_count = selected.size();
        tree.categories.insert(tree.categories.end(), selected.begin(), selected.end());
      }
    }
    for (u32 i = 0; i < count; ++i) require(!tree.nodes[i].split_type || categorical_seen[i], "Categorical model node is missing its category set.");
    std::vector<std::uint8_t> seen(count, 0);
    std::function<void(u32, u32)> visit = [&](u32 n, u32 depth) {
      require(depth <= 1024 && !seen[n], "Model contains a cyclic or excessively deep tree.");
      seen[n] = 1;
      if (tree.nodes[n].left >= 0) { visit(tree.nodes[n].left, depth + 1); visit(tree.nodes[n].right, depth + 1); }
    };
    visit(0, 0);
    auto deleted = std::count_if(tree.nodes.begin(), tree.nodes.end(), [](auto const& node) {
      return node.feature == INT32_MAX && node.default_left;
    });
    require(deleted == string_uint(value.at("tree_param").value("num_deleted", Json("0"))),
            "Model deleted-node count does not match its node records.");
    for (u32 i = 0; i < count; ++i)
      require(!seen[i] || tree.nodes[i].feature != INT32_MAX || !tree.nodes[i].default_left,
              "A deleted model node must not be reachable from the root.");
    loaded.push_back(std::move(tree));
  }
  new_rounds = round_ptr.size() - 1;
  }
  params.clear(); metrics.clear(); ranking_metric_configs_.clear();
  rng_.seed(0); rng_seed_ = "0";
  params["booster"] = name == "gblinear" ? "gblinear" : "gbtree";
  params["objective"] = learner.at("objective").at("name").get<std::string>();
  params["num_parallel_tree"] = std::to_string(parallel);
  params["num_class"] = std::to_string(new_classes);
  params["num_target"] = std::to_string(new_targets);
  params["boost_from_average"] = mp.contains("boost_from_average") ? json_string(mp["boost_from_average"]) : "1";
  for (auto const& [key, val] : learner.at("objective").items()) if (val.is_object()) for (auto const& [name, v] : val.items()) params[name] = json_string(v);
  load_ranking_state(learner.at("objective"));
  num_feature = new_features; num_class = new_classes; num_target = new_targets; groups = new_groups;
  base_score = std::move(scores); base_margin = score_margins(base_score);
  attributes = learner.value("attributes", std::map<std::string, std::string, std::less<>>{});
  feature_names = learner.value("feature_names", std::vector<std::string>{});
  feature_types = learner.value("feature_types", std::vector<std::string>{});
  categories = std::move(loaded_categories);
  configuration_dirty_ = false; base_score_changed_ = false;
  trees = std::move(loaded); tree_weights_ = std::move(weights);
  linear_weights = std::move(loaded_linear);
  rounds = new_rounds; initialized = true; reset();
}
Json Learner::config() {
  drain();
  configure();
  Json train;
  for (auto const& [key, value] : std::map<std::string, std::string>{{"learning_rate", "0.300000012"}, {"reg_lambda", "1"}, {"reg_alpha", "0"}, {"min_child_weight", "1"}, {"min_split_loss", "0"}, {"max_depth", "6"}, {"max_leaves", "0"}, {"max_bin", "256"}, {"subsample", "1"}, {"colsample_bytree", "1"}, {"colsample_bylevel", "1"}, {"colsample_bynode", "1"}, {"max_delta_step", "0"}, {"grow_policy", "depthwise"}, {"sampling_method", "uniform"}, {"monotone_constraints", "()"}, {"interaction_constraints", ""}, {"max_cat_to_onehot", "4"}, {"max_cat_threshold", "64"}, {"refresh_leaf", "1"}, {"sparse_threshold", "0.2"}}) train[key] = str(key, value);
  train["eta"] = train["learning_rate"]; train["lambda"] = train["reg_lambda"]; train["alpha"] = train["reg_alpha"]; train["gamma"] = train["min_split_loss"];
  Json metric = Json::array();
  for (auto const& m : metrics) {
    Json value{{"name", m}};
    if (m.starts_with("ndcg") || m.starts_with("map") || m.starts_with("pre@") || m == "pre" || m == "pre-")
      value = ranking_metric_config(m);
    if (m == "quantile" || m == "expectile") value[m + "_loss_param"] = {{m + "_alpha", Json(loss_alphas(m)).dump()}};
    if (m == "aft-nloglik") value["aft_loss_param"] = {
        {"aft_loss_distribution", str("aft_loss_distribution", "normal")},
        {"aft_loss_distribution_scale", str("aft_loss_distribution_scale", "1")}};
    metric.push_back(std::move(value));
  }
  auto model_params = model()["learner"]["learner_model_param"];
  std::ostringstream rng_state; rng_state << std::hex << rng_;
  Json updater_configs = Json::array();
  std::string updater_sequence;
  for (auto const& name : tree_updaters_) {
    if (!updater_sequence.empty()) updater_sequence += ',';
    updater_sequence += name;
    Json value{{"name", name}};
    if (name == "grow_metal_hist") value["hist_train_param"] = {
      {"max_cached_hist_node", str("max_cached_hist_node", "18446744073709551615")},
      {"debug_synchronize", boolean("debug_synchronize", false) ? "1" : "0"}};
    updater_configs.push_back(std::move(value));
  }
  Json booster = is_linear() ? linear_config() : Json{
    {"name", "gbtree"}, {"gbtree_model_param", {{"num_trees", std::to_string(trees.size())}, {"num_parallel_tree", str("num_parallel_tree", "1")}}},
    {"gbtree_train_param", {{"process_type", "default"}, {"tree_method", str("tree_method", "hist")}, {"updater", updater_sequence}, {"updater_seq", updater_sequence}}},
    {"tree_train_param", train}, {"dart_train_param", dropout_config()},
    {"specified_updater", params.contains("updater")}, {"updater", updater_configs}};
  return {{"version", {3, 5, 0}}, {"learner", {
    {"generic_param", {{"device", "metal:0"}, {"nthread", str("nthread", "0")}, {"n_jobs", str("nthread", "0")}, {"seed", str("seed", "0")}, {"random_state", str("seed", "0")}, {"seed_per_iteration", boolean("seed_per_iteration", false) ? "1" : "0"}, {"validate_parameters", "1"}, {"rng_state", rng_state.str()}}},
    {"learner_model_param", model_params},
    {"learner_train_param", {{"booster", is_linear() ? "gblinear" : "gbtree"}, {"objective", str("objective", "reg:squarederror")}, {"multi_strategy", str("multi_strategy", "one_output_per_tree")}, {"disable_default_eval_metric", str("disable_default_eval_metric", "0")}, {"base_score", str("base_score", "[]")}, {"boost_from_average", str("boost_from_average", params.contains("base_score") ? "0" : "1")}, {"num_class", std::to_string(num_class)}, {"num_target", std::to_string(num_target)}}},
    {"gradient_booster", booster},
    {"metrics", metric}, {"objective", objective_config(*this)}}}};
}
void Learner::load_config(Json const& document) {
  drain();
  auto const& learner = document.at("learner");
  auto const& objective = learner.at("objective");
  set_param("objective", objective.at("name").get<std::string>());
  for (auto const& [key, val] : objective.items())
    if (val.is_object()) for (auto const& [name, v] : val.items()) set_param(name, json_string(v));
  load_ranking_state(objective, false);
  for (auto const& [key, val] : learner.at("generic_param").items()) if (key != "device" && key != "rng_state") params[key] = json_string(val);
  auto const& booster = learner.at("gradient_booster");
  auto const& tree_booster = booster.value("name", "gbtree") == "dart" ? booster.at("gbtree") : booster;
  if (booster.value("name", "gbtree") == "gblinear") {
    clear_linear_parameters();
    for (auto const& [key, val] : booster.at("gblinear_train_param").items()) params[key] = json_string(val);
  }
  else {
    clear_tree_updater();
    params["process_type"] = "default";
    if (tree_booster.contains("gbtree_train_param"))
      params["tree_method"] = json_string(tree_booster["gbtree_train_param"]["tree_method"]);
  }
  if (tree_booster.contains("tree_train_param")) for (auto const& [key, val] : tree_booster["tree_train_param"].items()) params[key] = json_string(val);
  for (auto key : {"max_cached_hist_node", "debug_synchronize"}) params.erase(key);
  if (tree_booster.contains("updater")) for (auto const& updater : tree_booster["updater"])
    if (updater.contains("hist_train_param")) for (auto const& [key, val] : updater["hist_train_param"].items()) params[key] = json_string(val);
  if (tree_booster.contains("gbtree_model_param")) params["num_parallel_tree"] = json_string(tree_booster["gbtree_model_param"]["num_parallel_tree"]);
  for (auto key : {"sample_type", "normalize_type", "rate_drop", "skip_drop", "one_drop"}) params.erase(key);
  auto const& dropout = tree_booster.contains("dart_train_param") ? tree_booster : booster;
  if (dropout.contains("dart_train_param"))
    for (auto const& [key, value] : dropout["dart_train_param"].items()) params[key] = json_string(value);
  for (auto const& [key, val] : learner.at("learner_train_param").items()) if (key != "base_score" || val != "[]") params[key] = json_string(val);
  configure_rng();
  if (learner.at("generic_param").contains("rng_state")) {
    std::istringstream state(learner["generic_param"]["rng_state"].get<std::string>());
    state >> std::hex >> rng_;
    require(!state.fail(), "Invalid saved random generator state.");
  }
  metrics.clear(); ranking_metric_configs_.clear();
  ranking_metric_caches_.clear();
  for (auto const& m : learner.at("metrics")) {
    metrics.push_back(m.at("name").get<std::string>());
    if (m.contains("lambdarank_param")) ranking_metric_configs_[metrics.back()] = m["lambdarank_param"];
    for (auto const& [key, val] : m.items())
      if (val.is_object() && key != "lambdarank_param") for (auto const& [name, v] : val.items()) set_param(name, json_string(v));
  }
  reset(false);
}
std::vector<std::string> Learner::dump(std::string_view format, bool stats) {
  drain();
  configure();
  if (is_linear()) return dump_linear(format);
  require(format == "text" || format == "json" || format == "dot", "Unknown dump format.");
  std::vector<std::string> result;
  auto feature_name = [&](Node const& n) {
    auto name = feature_names.empty() ? "f" + std::to_string(n.feature) : feature_names.at(n.feature);
    if (static_cast<std::size_t>(n.feature) < feature_types.size()) {
      bool categorical = feature_types[n.feature] == "c";
      require(categorical == static_cast<bool>(n.split_type), name +
              (categorical ? " in feature map is categorical but the tree node is numerical."
                           : " in feature map is numerical but the tree node is categorical."));
    }
    return name;
  };
  for (auto const& tree : trees) {
    auto leaf_value = [&](Node const& node) -> Json {
      if (tree.outputs == 1) return node.leaf;
      Json values = Json::array();
      for (u32 target = 0; target < tree.outputs; ++target)
        values.push_back(tree.leaf_weights.at(std::size_t(node.right) * tree.outputs + target));
      return values;
    };
    auto leaf_text = [&](Node const& node) {
      std::ostringstream out; out << std::setprecision(9);
      if (tree.outputs == 1) out << node.leaf;
      else {
        out << '[';
        for (u32 target = 0; target < tree.outputs; ++target) {
          if (target) out << ", ";
          out << tree.leaf_weights.at(std::size_t(node.right) * tree.outputs + target);
        }
        out << ']';
      }
      return out.str();
    };
    auto selected_categories = [&](Node const& n) {
      auto begin = tree.categories.begin() + n.category_begin;
      return std::vector<u32>(begin, begin + n.category_count);
    };
    auto condition = [&](Node const& n) {
      std::ostringstream text; text << std::setprecision(9) << feature_name(n);
      if (n.split_type) {
        text << ":{";
        for (u32 i = 0; i < n.category_count; ++i) { if (i) text << ','; text << tree.categories[n.category_begin + i]; }
        text << '}';
      } else text << '<' << n.threshold;
      return text.str();
    };
    if (format == "json") {
      std::function<Json(u32, u32)> node = [&](u32 id, u32 depth) {
        auto const& n = tree.nodes.at(id); Json out{{"nodeid", id}};
        if (n.left < 0) out["leaf"] = leaf_value(n);
        else {
          out["depth"] = depth; out["split"] = feature_name(n);
          out["split_condition"] = n.split_type ? Json(selected_categories(n)) : Json(n.threshold);
          out["yes"] = n.split_type ? n.right : n.left; out["no"] = n.split_type ? n.left : n.right;
          out["missing"] = n.default_left ? n.left : n.right;
          if (stats) out["gain"] = n.gain;
          out["children"] = Json::array({node(n.left, depth + 1), node(n.right, depth + 1)});
        }
        if (stats) out["cover"] = n.cover;
        return out;
      };
      result.push_back(dump_json(node(0, 0), true, 2));
    } else {
      std::ostringstream out; out << std::setprecision(9);
      if (format == "dot") out << "digraph {\n    graph [ rankdir=TB ]\n";
      std::function<void(u32, u32)> node = [&](u32 id, u32 depth) {
        auto const& n = tree.nodes.at(id);
        if (format == "dot") {
          out << "    " << id << " [ label=\"";
          if (n.left < 0) out << "leaf=" << leaf_text(n);
          else out << condition(n);
          if (stats) out << "\\ncover=" << n.cover;
          out << "\" ]\n";
          if (n.left >= 0) {
            out << "    " << id << " -> " << n.left << " [ label=\"" << (n.split_type ? "no" : "yes") << (n.default_left ? ", missing" : "") << "\" ]\n";
            out << "    " << id << " -> " << n.right << " [ label=\"" << (n.split_type ? "yes" : "no") << (!n.default_left ? ", missing" : "") << "\" ]\n";
          }
        } else {
          out << std::string(depth, '\t') << id << ':';
          if (n.left < 0) out << "leaf=" << leaf_text(n);
          else out << '[' << condition(n) << "] yes=" << (n.split_type ? n.right : n.left) << ",no=" << (n.split_type ? n.left : n.right) << ",missing=" << (n.default_left ? n.left : n.right);
          if (stats) { if (n.left >= 0) out << ",gain=" << n.gain; out << ",cover=" << n.cover; }
          out << '\n';
        }
        if (n.left >= 0) { node(n.left, depth + 1); node(n.right, depth + 1); }
      };
      node(0, 0); if (format == "dot") out << "}\n";
      result.push_back(out.str());
    }
  }
  return result;
}
std::unique_ptr<Learner> Learner::slice(int begin, int end, int step) {
  drain();
  configure();
  require(!is_linear(), "Slice is not supported by the current booster.");
  require(trees_to_update_.empty() || trees_to_update_.size() == trees.size(),
          "Not all trees are updated. Slice the model before updating only a portion of its trees.");
  require(step > 0 && begin >= 0, "Invalid model slice.");
  if (!end) end = rounds;
  require(end <= static_cast<int>(rounds) && begin < end, "Model slice index out of range.");
  auto result = std::make_unique<Learner>(); result->load_model(model());
  result->params = params; result->metrics = metrics; result->trees.clear(); result->rounds = 0;
  result->tree_weights_.clear();
  result->rng_ = rng_; result->rng_seed_ = rng_seed_;
  result->ranking_metric_configs_ = ranking_metric_configs_;
  result->attributes.erase("best_iteration"); result->attributes.erase("best_score");
  for (int r = begin; r < end; r += step) {
    for (std::size_t t = 0; t < trees.size(); ++t) if (trees[t].round == static_cast<u32>(r)) {
      auto copy = trees[t]; copy.round = result->rounds; result->trees.push_back(std::move(copy));
      if (!tree_weights_.empty()) result->tree_weights_.push_back(tree_weights_[t]);
    }
    ++result->rounds;
  }
  result->reset(); return result;
}
}  // namespace metalgb
