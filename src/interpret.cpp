#include "metalgb/engine.hpp"
#include <algorithm>
#include <functional>
#include <iterator>

namespace metalgb {
void Learner::upload_shap() {
  if (shap_size_ == trees.size()) return;
  std::vector<ShapPath> paths;
  std::vector<ShapLeaf> leaves;
  std::vector<u32> categories;
  std::vector<float> means;
  struct Term { ShapPath path; std::vector<u32> categories; };
  for (u32 tree_id = 0; tree_id < trees.size(); ++tree_id) {
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
          leaves.push_back({static_cast<u32>(paths.size()), static_cast<u32>(path.size()),
                            tree.outputs > 1 ? target : tree.group, tree.round, value,
                            static_cast<float>(value * probability), tree_id, 0});
        }
        for (auto& term : path) {
          term.path.category_begin = categories.size();
          term.path.category_count = term.categories.size();
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
  shap_paths_ = Buffer(paths.data(), paths.size() * sizeof(ShapPath));
  shap_leaves_ = Buffer(leaves.data(), leaves.size() * sizeof(ShapLeaf));
  shap_categories_ = Buffer(categories.data(), categories.size() * sizeof(u32));
  node_means_ = Buffer(means.data(), means.size() * sizeof(float));
  shap_leaf_count_ = leaves.size(); shap_size_ = trees.size();
}
}  // namespace metalgb
