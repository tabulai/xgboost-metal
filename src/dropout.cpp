#include "metalgb/engine.hpp"
#include <iomanip>
#include <sstream>

namespace metalgb {
Learner::DropoutParams Learner::dropout_parameters() const {
  auto rate = number("rate_drop", 0), skip = number("skip_drop", 0);
  require(rate >= 0 && rate <= 1 && skip >= 0 && skip <= 1,
          "rate_drop and skip_drop must be in [0, 1].");
  auto sample = str("sample_type", "uniform"), normalize = str("normalize_type", "tree");
  require(sample == "uniform" || sample == "weighted", "sample_type must be uniform or weighted.");
  require(normalize == "tree" || normalize == "forest", "normalize_type must be tree or forest.");
  return {rate, skip, boolean("one_drop", false), sample == "weighted", normalize == "forest"};
}
Json Learner::dropout_config() const {
  auto p = dropout_parameters();
  auto text = [](float value) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return stream.str();
  };
  return {{"rate_drop", text(p.rate)}, {"skip_drop", text(p.skip)},
          {"one_drop", p.one ? "1" : "0"}, {"sample_type", p.weighted ? "weighted" : "uniform"},
          {"normalize_type", p.forest ? "forest" : "tree"}};
}

std::vector<float> Learner::drop_trees() {
  auto p = dropout_parameters();
  if (tree_weights_.empty()) {
    if (!p.active() || trees.empty()) return {};
    tree_weights_.assign(trees.size(), 1);
    forest_weights_ = Buffer(tree_weights_.data(), tree_weights_.size() * sizeof(float));
  }
  dropped_trees_.clear();
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  if (p.skip > 0 && uniform(rng_) < p.skip) return {};
  if (p.weighted) {
    float total = 0;
    for (auto weight : tree_weights_) total += weight;
    for (std::size_t t = 0; t < tree_weights_.size(); ++t)
      if (uniform(rng_) < p.rate * tree_weights_.size() * tree_weights_[t] / total)
        dropped_trees_.push_back(t);
    if (p.one && dropped_trees_.empty()) {
      std::discrete_distribution<std::size_t> sample(tree_weights_.size(), 0.0,
          static_cast<double>(tree_weights_.size()),
          [this](double index) { return tree_weights_[static_cast<std::size_t>(index)]; });
      dropped_trees_.push_back(sample(rng_));
    }
  } else {
    for (std::size_t t = 0; t < tree_weights_.size(); ++t)
      if (uniform(rng_) < p.rate) dropped_trees_.push_back(t);
    if (p.one && dropped_trees_.empty())
      dropped_trees_.push_back(std::uniform_int_distribution<std::size_t>(0, tree_weights_.size() - 1)(rng_));
  }
  if (dropped_trees_.empty()) return {};
  auto masked = tree_weights_;
  for (auto t : dropped_trees_) masked[t] = 0;
  return masked;
}

void Learner::normalize_trees(std::size_t added) {
  auto p = dropout_parameters();
  if (tree_weights_.empty() && !p.active()) return;
  tree_weights_.resize(trees.size() - added, 1);
  auto count = dropped_trees_.size();
  auto rate = number("learning_rate", 0.3f);
  float new_weight = 1;
  if (count) {
    // Each parallel tree already uses eta / num_parallel_tree. Normalization
    // uses the complete layer's eta, as in the pinned 3.5 implementation.
    float factor = p.forest ? 1.0 / (1.0 + rate) : 1.0 * count / (count + rate);
    for (auto t : dropped_trees_) tree_weights_.at(t) *= factor;
    new_weight = p.forest ? factor : 1.0 / (count + rate);
  }
  tree_weights_.insert(tree_weights_.end(), added, new_weight);
  dropped_trees_.clear();
}
}  // namespace metalgb
