#include "metalgb/engine.hpp"
#include <algorithm>
#include <functional>
#include <iterator>

namespace metalgb {
void Learner::upload_shap() {
  if (shap_size_ == trees.size()) return;
  bool rebuild = shap_size_ > trees.size();
  u32 first = rebuild ? 0u : static_cast<u32>(shap_size_);
  std::size_t path_base = rebuild ? 0 : shap_path_count_;
  std::size_t leaf_base = rebuild ? 0 : static_cast<std::size_t>(shap_leaf_count_);
  std::size_t cat_base = rebuild ? 0 : shap_category_count_;
  std::size_t mean_base = rebuild ? 0 : shap_mean_count_;
  std::vector<ShapPath> paths;
  std::vector<ShapLeaf> leaves;
  std::vector<u32> categories;
  std::vector<float> means;
  struct Term { ShapPath path; std::vector<u32> categories; };
  for (u32 tree_id = first; tree_id < trees.size(); ++tree_id) {
    auto const& tree = trees[tree_id];
    auto offset = means.size();
    means.resize(offset + tree.nodes.size());
    std::function<double(u32)> fill_means = [&](u32 index) -> double {
      auto const& node = tree.nodes[index];
      require(node.cover >= 0, "SHAP requires nonnegative node cover.");
      double mean = tree.outputs > 1 && node.left < 0 ? tree.leaf_weights.at(std::size_t(node.right) * tree.outputs) : node.leaf;
      if (node.left >= 0) {
        auto left = fill_means(node.left), right = fill_means(node.right);
        mean = node.cover > 0 ? (left * tree.nodes[node.left].cover + right * tree.nodes[node.right].cover) / node.cover
                              : 0.5 * (left + right);
      }
      means[offset + index] = static_cast<float>(mean);
      return mean;
    };
    fill_means(0);
    std::function<void(u32, std::vector<Term>, double)> visit =
        [&](u32 index, std::vector<Term> path, double probability) {
      auto const& node = tree.nodes[index];
      if (node.left < 0) {
        for (u32 target = 0; target < tree.outputs; ++target) {
          float value = tree.outputs > 1 ? tree.leaf_weights.at(std::size_t(node.right) * tree.outputs + target) : node.leaf;
          leaves.push_back({static_cast<u32>(path_base + paths.size()), static_cast<u32>(path.size()),
                            tree.outputs > 1 ? target : tree.group, tree.round, value,
                            static_cast<float>(value * probability), tree_id, 0});
        }
        for (auto& term : path) {
          term.path.category_begin = static_cast<u32>(cat_base + categories.size());
          term.path.category_count = static_cast<u32>(term.categories.size());
          paths.push_back(term.path);
          categories.insert(categories.end(), term.categories.begin(), term.categories.end());
        }
        return;
      }
      for (bool left : {true, false}) {
        auto child = left ? node.left : node.right;
        auto child_path = path;
        auto it = std::find_if(child_path.begin(), child_path.end(),
                               [&](Term const& term) { return term.path.feature == static_cast<u32>(node.feature); });
        if (it == child_path.end()) {
          child_path.push_back({{static_cast<u32>(node.feature), 1, 1,
                                -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::infinity(), 0, 0, node.split_type ? 2u : 0u}, {}});
          it = child_path.end() - 1;
        }
        float weight = node.cover > 0 ? tree.nodes[child].cover / node.cover : 0.5f;
        it->path.zero *= std::max(weight, 1.0e-12f);
        it->path.missing &= static_cast<u32>(left == static_cast<bool>(node.default_left));
        if (node.split_type) {
          auto begin = tree.categories.begin() + node.category_begin, end = begin + node.category_count;
          auto const& current = it->categories;
          std::vector<u32> merged;
          // Merge repeated categorical conditions without enumerating a universe:
          // either retain an allowed set, or accumulate an excluded set.
          if (it->path.category_mode == 1) {
            if (left) std::set_difference(current.begin(), current.end(), begin, end, std::back_inserter(merged));
            else std::set_intersection(current.begin(), current.end(), begin, end, std::back_inserter(merged));
          } else if (left) std::set_union(current.begin(), current.end(), begin, end, std::back_inserter(merged));
          else {
            std::set_difference(begin, end, current.begin(), current.end(), std::back_inserter(merged));
            it->path.category_mode = 1;
          }
          it->categories = std::move(merged);
        } else if (left) it->path.upper = std::min(it->path.upper, node.threshold);
        else it->path.lower = std::max(it->path.lower, node.threshold);
        visit(child, std::move(child_path), probability * weight);
      }
    };
    visit(0, {}, 1);
  }
  auto append_buf = [](Buffer& dst, std::size_t base_count, auto const& src) {
    using T = typename std::decay_t<decltype(src)>::value_type;
    std::size_t total = base_count + src.size();
    dst.reserve(checked_product(total, sizeof(T)), checked_product(base_count, sizeof(T)));
    if (!src.empty()) std::memcpy(dst.as<T>() + base_count, src.data(), src.size() * sizeof(T));
  };
  append_buf(shap_paths_, path_base, paths);
  append_buf(shap_leaves_, leaf_base, leaves);
  append_buf(shap_categories_, cat_base, categories);
  append_buf(node_means_, mean_base, means);
  shap_path_count_ = path_base + paths.size();
  shap_category_count_ = cat_base + categories.size();
  shap_mean_count_ = mean_base + means.size();
  shap_leaf_count_ = static_cast<u32>(leaf_base + leaves.size());
  shap_size_ = trees.size();
}
}  // namespace metalgb
