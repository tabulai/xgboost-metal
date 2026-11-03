// Native validation of certified split screening (kernels/scalar_screen.metal.h).
//
// 1. Candidate enclosures: random and adversarial integer statistics run
//    through every exact scalar scorer family and the screening bounds. The
//    exact float32 gain must lie inside [lo, hi] for certified candidates and
//    be rejected for certainly invalid ones; parent bounds must enclose the
//    exact parent gain.
// 2. Feature records: the production screened kernels must write per-feature
//    Splits byte-identical to their unscreened twins, and with the node-wide
//    cutoff, select_splits must choose byte-identical node winners.
//
// Usage: split_screening_bounds_test [candidate cases per setting]
#include "metalgb/runtime.hpp"
#include "metalgb/shared.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace metalgb;

namespace {
constexpr u32 kNegativeInfinity = 0xff800000u;

void check(bool condition, std::string const& message) {
  if (!condition) throw std::runtime_error(message);
}
float bits_float(u32 bits) { return std::bit_cast<float>(bits); }
u32 float_bits(float value) { return std::bit_cast<u32>(value); }

struct Counts { std::uint64_t valid{}, invalid{}, uncertain{}, parent_fallback{}; };

// ---------------------------------------------------------------- candidates
struct CandidateSet {
  std::vector<std::array<i64, 4>> sums;
  std::vector<std::array<float, 2>> scales;
  void add(i64 lg, i64 lh, i64 rg, i64 rh, float gs, float hs) {
    sums.push_back({lg, lh, rg, rh});
    scales.push_back({gs, hs});
  }
};

CandidateSet candidate_cases(std::size_t count, std::mt19937_64& rng) {
  CandidateSet set;
  auto uniform = [&](i64 lo, i64 hi) { return std::uniform_int_distribution<i64>(lo, hi)(rng); };
  auto bits = [&](int width) { return width >= 63 ? i64(rng() >> 1) : i64(rng() & ((1ull << width) - 1)); };
  auto signed_bits = [&](int width) { auto v = bits(width); return (rng() & 1) ? -v : v; };
  // Weight quotients just inside/outside a binary64 -> float32 midpoint cell.
  for (int parity : {0, 1}) for (int inside : {0, 1}) for (int sign : {-1, 1}) {
    i64 q = (1 << 23) + parity;
    i64 delta = (parity ? -1 : 1) * ((1 << 23) + (inside ? -1 : 1));
    u64 odd = 2 * q + 1, inverse = 1;
    for (int k = 0; k < 6; ++k) inverse *= 2 - odd * inverse;
    i64 denominator = (1ll << 52) + ((u64(-delta) * inverse) & ((1 << 24) - 1));
    i64 numerator = static_cast<i64>((static_cast<__int128>(odd) * denominator + delta) / (1 << 24));
    set.add(sign * numerator, denominator, -sign * numerator / 3, denominator / 2, 0x1p40f, 0x1p40f);
    set.add(sign * numerator, denominator, 0, 1ll << 52, 0x1p40f, 0x1p52f);
  }
  while (set.sums.size() < count) {
    auto kind = set.sums.size() % 12;
    switch (kind) {
      case 0: case 1: {  // Quantized regression-like counts and residual sums.
        i64 nl = uniform(0, 1 << 20), nr = uniform(0, 1 << 20);
        int q = int(uniform(20, 40));
        i64 gl = uniform(-2 * nl, 2 * nl) << q, gr = uniform(-2 * nr, 2 * nr) << q;
        gl += signed_bits(q); gr += signed_bits(q);
        set.add(gl, nl << q, gr, nr << q, std::ldexp(1.0f, q), std::ldexp(1.0f, q));
        break;
      }
      case 2: {  // Wide magnitudes and scales, including outside the guards.
        int ge = int(uniform(-110, 140)), he = int(uniform(-110, 140));
        set.add(signed_bits(int(uniform(1, 62))), bits(int(uniform(1, 62))) + 1,
                signed_bits(int(uniform(1, 62))), bits(int(uniform(1, 62))) + 1,
                std::ldexp(1.0f, std::clamp(ge, -126, 127)), std::ldexp(1.0f, std::clamp(he, -126, 127)));
        break;
      }
      case 3: {  // Zero gradients, zero/negative/signed Hessians.
        i64 lg = (rng() % 3) ? signed_bits(40) : 0, rg = (rng() % 3) ? signed_bits(40) : 0;
        i64 lh = bits(40), rh = bits(40);
        if (rng() % 4 == 0) lh = -lh;
        if (rng() % 5 == 0) rh = 0;
        set.add(lg, lh, rg, rh, 0x1p30f, 0x1p30f);
        break;
      }
      case 4: {  // min_child_weight boundaries: H within a few integer units of 1, 5.5 or 1e-3.
        int k = int(uniform(20, 62));
        double targets[] = {1.0, 5.5, 1e-3, 0.0};
        double target = targets[rng() % 4];
        i64 base = static_cast<i64>(std::ldexp(target, k));
        i64 lh = base + uniform(-3, 3), rh = (rng() & 1) ? base + uniform(-3, 3) : bits(k + 2);
        set.add(signed_bits(k), std::max<i64>(lh, 1), signed_bits(k), std::max<i64>(rh, 1),
                std::ldexp(1.0f, k), std::ldexp(1.0f, k));
        break;
      }
      case 5: {  // Flat nodes: proportional children make the exact gain rounding noise.
        i64 a = uniform(-8, 8), lh = uniform(1, 1 << 24), rh = uniform(1, 1 << 24);
        int q = int(uniform(10, 30));
        i64 jitter = (rng() % 4 == 0) ? uniform(-2, 2) : 0;
        set.add((a * lh << q) + jitter, lh << q, (a * rh) << q, rh << q, std::ldexp(1.0f, q), std::ldexp(1.0f, q));
        break;
      }
      case 6: {  // Integer limits: 64-bit extremes and their neighbors.
        i64 extremes[] = {std::numeric_limits<i64>::min(), std::numeric_limits<i64>::min() + 1,
                          std::numeric_limits<i64>::max(), -(1ll << 62), (1ll << 62) - 1, (1ll << 61) + 1};
        auto pick = [&] { return extremes[rng() % 6]; };
        int e = int(uniform(0, 127));
        set.add(pick(), std::abs(pick() / 2) + 1, pick() / 3, (1ll << 62) - 1,
                std::ldexp(1.0f, e), std::ldexp(1.0f, e));
        break;
      }
      case 7: {  // Tiny Hessians where lambda dominates D.
        int q = int(uniform(40, 62));
        set.add(signed_bits(int(uniform(1, 30))), bits(int(uniform(1, 20))) + 1,
                signed_bits(int(uniform(1, 30))), bits(int(uniform(1, 20))) + 1,
                std::ldexp(1.0f, q - 30), std::ldexp(1.0f, q));
        break;
      }
      case 8: {  // Near guard edges: |G|, H close to 2^-60, 2^61 and weights near 2^+-62.
        int q = int(uniform(0, 30));
        int shift = int(uniform(-64, 64));
        float s = std::ldexp(1.0f, std::clamp(q + shift, -126, 127));
        set.add(signed_bits(int(uniform(20, 62))), bits(int(uniform(1, 62))) + 1,
                signed_bits(int(uniform(20, 62))), bits(int(uniform(1, 62))) + 1, s,
                std::ldexp(1.0f, std::clamp(q - shift, -126, 127)));
        break;
      }
      case 9: {  // Mirrored children and exact cancellations.
        i64 g = signed_bits(50), h = bits(50) + 1;
        set.add(g, h, (rng() & 1) ? -g : g, h, 0x1p40f, 0x1p40f);
        break;
      }
      case 10: {  // Logistic-like: tiny Hessians per row, nearly balanced gradients.
        i64 nl = uniform(1, 1 << 18), nr = uniform(1, 1 << 18);
        int q = int(uniform(30, 44));
        i64 hl = nl * uniform(1, 1 << 6), hr = nr * uniform(1, 1 << 6);
        set.add(uniform(-hl, hl), hl << 8, uniform(-hr, hr), hr << 8, std::ldexp(1.0f, q), std::ldexp(1.0f, q + 8));
        break;
      }
      default: {  // Uniform 61-bit statistics at typical scales.
        i64 lh = bits(int(uniform(1, 61))) + 1, rh = bits(int(uniform(1, 61))) + 1;
        set.add(signed_bits(61), lh, signed_bits(61), rh, 0x1p40f, 0x1p40f);
        break;
      }
    }
  }
  return set;
}

Counts check_candidates(std::size_t cases, std::mt19937_64& rng) {
  auto set = candidate_cases(cases, rng);
  auto n = set.sums.size();
  Buffer sums(set.sums.data(), n * sizeof(set.sums[0]));
  Buffer scales(set.scales.data(), n * sizeof(set.scales[0]));
  Buffer exact(n * 4 * sizeof(u32)), enclosure(n * 4 * sizeof(float)), parents(n * 4 * sizeof(u32));
  Buffer inherited(n * 2 * sizeof(u32));
  Counts counts;
  const float subnormal = std::numeric_limits<float>::denorm_min() * 3;
  struct Setting { float lambda, minimum; };
  Setting settings[] = {{1, 1}, {0, 0}, {0.31f, 5.5f}, {0x1p-20f, 1e-3f}, {0x1p20f, 1},
                        {1e-30f, -0.0f}, {subnormal, subnormal}, {0, 1}, {1, 0}};
  for (auto setting : settings) {
    SplitParams p{};
    p.nodes = static_cast<u32>(n); p.reg_lambda = setting.lambda; p.min_child_weight = setting.minimum;
    Command command("screen enclosure probe");
    command.dispatch("screen_probe_candidates", {sums, scales, exact, enclosure, parents, inherited}, p,
                     {div_up(n, 256), 1, 1});
    command.wait();
    auto e = exact.as<u32>();
    auto b = enclosure.as<float>();
    auto q = parents.as<u32>();
    for (std::size_t i = 0; i < n; ++i) {
      auto describe = [&] {
        auto const& s = set.sums[i];
        return " case " + std::to_string(i) + " sums " + std::to_string(s[0]) + "," + std::to_string(s[1]) + "," +
               std::to_string(s[2]) + "," + std::to_string(s[3]) + " scales " + std::to_string(set.scales[i][0]) + "," +
               std::to_string(set.scales[i][1]) + " lambda " + std::to_string(setting.lambda) + " mcw " +
               std::to_string(setting.minimum);
      };
      u32 gain = e[4 * i], result = e[4 * i + 3];
      check(e[4 * i + 1] == gain && e[4 * i + 2] == gain, "Exact scorer families disagree:" + describe());
      check(q[4 * i + 1] == q[4 * i] && q[4 * i + 2] == q[4 * i], "Exact parent gains disagree:" + describe());
      auto r = inherited.as<u32>();
      check(r[2 * i] == q[4 * i] && r[2 * i + 1] == q[4 * i], "Gain from an inherited weight differs:" + describe());
      if (q[4 * i + 3]) {
        double exact_parent = bits_float(q[4 * i]);
        check(b[4 * i + 2] <= exact_parent && exact_parent <= b[4 * i + 3],
              "Parent gain escaped its enclosure:" + describe());
      }
      if (result == 3) { ++counts.parent_fallback; continue; }
      if (result == 0) {
        check(gain == kNegativeInfinity, "A certainly invalid candidate was accepted:" + describe());
        ++counts.invalid;
      } else if (result == 1) {
        double value = bits_float(gain), lo = b[4 * i], hi = b[4 * i + 1];
        check(gain != kNegativeInfinity && !std::isnan(value),
              "A certified candidate was rejected by the exact scorer:" + describe());
        check(std::isfinite(lo) && std::isfinite(hi) && lo <= value && value <= hi,
              "Exact gain " + std::to_string(value) + " escaped [" + std::to_string(lo) + ", " +
              std::to_string(hi) + "]:" + describe());
        ++counts.valid;
      } else {
        check(result == 2, "Unknown screening class:" + describe());
        ++counts.uncertain;
      }
    }
  }
  return counts;
}

// ------------------------------------------------------------ feature records
struct Problem {
  u32 nodes, cols, bins;
  float lambda, minimum;
  std::array<float, 2> scale;
  std::vector<u32> counts;
  std::vector<u32> histogram, parent;      // nodes x cols x (bins + 1) x 4 words
  std::vector<std::array<i64, 2>> node_sums;
  std::vector<std::uint8_t> allowed;
  std::vector<HistSubtract> subtraction;
};

void put(std::vector<u32>& words, std::size_t entry, i64 g, i64 h) {
  auto ug = static_cast<u64>(g), uh = static_cast<u64>(h);
  words[entry * 4] = u32(ug); words[entry * 4 + 1] = u32(ug >> 32);
  words[entry * 4 + 2] = u32(uh); words[entry * 4 + 3] = u32(uh >> 32);
}
std::array<i64, 2> get(std::vector<u32> const& words, std::size_t entry) {
  auto g = u64(words[entry * 4]) | (u64(words[entry * 4 + 1]) << 32);
  auto h = u64(words[entry * 4 + 2]) | (u64(words[entry * 4 + 3]) << 32);
  return {static_cast<i64>(g), static_cast<i64>(h)};
}

enum Pattern { kRegression, kSparse, kDuplicate, kFlat, kProportional, kMinimum, kSymmetric, kHuge,
               kExtreme, kSigned, kPatterns };
char const* pattern_name(int pattern) {
  constexpr char const* names[] = {"regression", "sparse", "duplicate", "flat", "proportional",
                                   "min_child_weight", "symmetric", "huge", "extreme", "signed"};
  return names[pattern];
}

// Nodes come in sibling pairs: node 2k is built and 2k + 1 is derived from
// a parent histogram, as in fused subtraction. Every node's inherited total
// covers all features; the remainder of each feature is its missing bin.
// Some features are fully present (no missing statistics, forward scans only).
Problem make_problem(int pattern, u32 nodes, u32 cols, u32 bins, std::mt19937_64& rng) {
  Problem problem{nodes, cols, bins, 1, 1, {0x1p30f, 0x1p30f}, {}, {}, {}, {}, {}, {}};
  auto uniform = [&](i64 lo, i64 hi) { return std::uniform_int_distribution<i64>(lo, hi)(rng); };
  int q = 30;
  if (pattern == kHuge) { q = 40; problem.scale = {0x1p40f, 0x1p40f}; }
  if (pattern == kExtreme) problem.scale = {0x1p-100f, 0x1p100f};
  if (pattern == kMinimum) problem.minimum = 5.5f;
  if (pattern == kFlat || pattern == kProportional) problem.lambda = 0;
  if (pattern == kSigned) problem.minimum = 0;
  problem.counts.resize(cols);
  for (auto& count : problem.counts) count = static_cast<u32>(uniform(1, bins));
  problem.counts[0] = bins;
  auto entries = std::size_t(nodes) * cols * (bins + 1);
  problem.histogram.assign(entries * 4, 0);
  problem.parent.assign(entries * 4, 0);
  problem.node_sums.assign(nodes, {0, 0});
  problem.allowed.assign(std::size_t(nodes) * cols, 1);
  for (auto& value : problem.allowed) value = rng() % 13 != 0;
  i64 slope = uniform(-3, 3);
  auto draw = [&]() -> std::array<i64, 2> {
    i64 rows = uniform(0, 400);
    if ((pattern == kSparse || pattern == kMinimum) && rng() % 10 < 8) rows = 0;
    if (pattern == kMinimum) rows = std::min<i64>(rows, 4);
    i64 h = rows << q, g = 0;
    switch (pattern) {
      case kFlat: break;
      case kProportional: g = slope * h; break;
      case kHuge:
        h = rows ? i64(rng() >> 9) | 1 : 0;
        g = rows ? i64(rng() >> 9) - (1ll << 54) : 0;
        break;
      case kSigned: g = uniform(-rows, rows) << q; h = uniform(-rows / 4, rows) << q; break;
      default: g = (uniform(-2 * rows, 2 * rows) << q) + (rows ? uniform(-(1 << 20), 1 << 20) : 0);
    }
    return {g, h};
  };
  auto entry = [&](u32 node, u32 f, u32 b) { return (std::size_t(node) * cols + f) * (bins + 1) + b; };
  problem.subtraction.resize(std::max(1u, nodes / 2));
  for (u32 pair = 0; pair < nodes / 2; ++pair) {
    u32 built = 2 * pair, derived = built + 1;
    problem.subtraction[pair] = {pair, built, derived, 0};
    for (u32 side = 0; side < 2; ++side) {
      u32 node = built + side;
      std::array<i64, 2> maximum{0, 0};
      std::vector<std::array<i64, 2>> present(cols, {0, 0});
      for (u32 f = 0; f < cols; ++f) {
        bool duplicate = pattern == kDuplicate && (f & 1) && problem.counts[f] <= problem.counts[f - 1];
        for (u32 b = 0; b < problem.counts[f]; ++b) {
          std::array<i64, 2> value;
          if (duplicate) value = get(problem.histogram, entry(node, f - 1, b));
          else if (pattern == kSymmetric && 2 * b >= problem.counts[f])
            value = get(problem.histogram, entry(node, f, problem.counts[f] - 1 - b));
          else value = draw();
          put(problem.histogram, entry(node, f, b), value[0], value[1]);
          present[f][0] += value[0]; present[f][1] += value[1];
        }
        maximum[1] = std::max(maximum[1], present[f][1]);
      }
      // Inherited total: at least every present Hessian, plus missing rows.
      i64 extra = pattern == kHuge ? i64(rng() >> 9) : uniform(0, 3000) << q;
      i64 gradient = pattern == kFlat ? 0 : pattern == kProportional ? slope * (maximum[1] + extra) :
                     present[0][0] + uniform(-extra / 2 - 1, extra / 2 + 1);
      std::array<i64, 2> total{gradient, maximum[1] + extra};
      problem.node_sums[node] = total;
      for (u32 f = 0; f < cols; ++f) {
        if (f % 3 == 2) {
          // Fully present: fold the remainder into the last present bin.
          auto last = get(problem.histogram, entry(node, f, problem.counts[f] - 1));
          put(problem.histogram, entry(node, f, problem.counts[f] - 1),
              static_cast<i64>(u64(last[0]) + u64(total[0] - present[f][0])),
              static_cast<i64>(u64(last[1]) + u64(total[1] - present[f][1])));
          put(problem.histogram, entry(node, f, bins), 0, 0);
        } else {
          put(problem.histogram, entry(node, f, bins), total[0] - present[f][0], total[1] - present[f][1]);
        }
      }
    }
    // The parent is built + derived bin by bin, including the missing bin.
    for (u32 f = 0; f < cols; ++f)
      for (u32 b = 0; b <= bins; ++b) {
        auto x = get(problem.histogram, entry(built, f, b)), y = get(problem.histogram, entry(derived, f, b));
        put(problem.parent, entry(pair, f, b), static_cast<i64>(u64(x[0]) + u64(y[0])),
            static_cast<i64>(u64(x[1]) + u64(y[1])));
      }
  }
  return problem;
}

// Fused subtraction produces the derived siblings' bins; poison the inputs.
Problem poisoned(Problem problem) {
  for (u32 node = 1; node < problem.nodes; node += 2)
    for (u32 f = 0; f < problem.cols; ++f)
      for (u32 b = 0; b <= problem.bins; ++b)
        put(problem.histogram, (std::size_t(node) * problem.cols + f) * (problem.bins + 1) + b, -12345, 54321);
  return problem;
}

struct Inputs {
  Buffer scales, cuts, counts, minima, monotone, bounds, allowed, node_sums, parent, subtraction, params, dummy;
};

Inputs upload(Problem const& problem, bool root) {
  std::vector<float> cuts(std::size_t(problem.cols) * problem.bins), minima(problem.cols);
  for (u32 f = 0; f < problem.cols; ++f) {
    minima[f] = -1.0f - float(f);
    for (u32 b = 0; b < problem.bins; ++b) cuts[std::size_t(f) * problem.bins + b] = float(b) + 0.25f * float(f);
  }
  std::vector<int> monotone(problem.cols, 0);
  std::vector<Pair> bounds(problem.nodes, Pair{-std::numeric_limits<float>::infinity(),
                                               std::numeric_limits<float>::infinity()});
  SplitParams p{problem.nodes, problem.cols, problem.bins, 0, problem.lambda, 0, problem.minimum, 0,
                4, 64, 256, 0};
  p.use_parent_sums = root ? 0 : 1;
  std::vector<IntPair> sums;
  for (auto const& s : problem.node_sums) sums.push_back({s[0], s[1]});
  return {Buffer(problem.scale.data(), sizeof(problem.scale)), Buffer(cuts.data(), cuts.size() * 4),
          Buffer(problem.counts.data(), problem.counts.size() * 4), Buffer(minima.data(), minima.size() * 4),
          Buffer(monotone.data(), monotone.size() * 4), Buffer(bounds.data(), bounds.size() * sizeof(Pair)),
          Buffer(problem.allowed.data(), problem.allowed.size()), Buffer(sums.data(), sums.size() * sizeof(IntPair)),
          Buffer(problem.parent.data(), problem.parent.size() * 4),
          Buffer(problem.subtraction.data(), problem.subtraction.size() * sizeof(HistSubtract)),
          Buffer(&p, sizeof(p)), Buffer(64)};
}

struct Run { std::vector<Split> records; std::array<u32, 6> statistics{}; };

Run evaluate(Problem const& problem, Inputs const& in, std::string const& kernel, bool screened, bool root,
             bool node_cut, std::vector<Split> const* parents = nullptr) {
  Buffer histogram(problem.histogram.data(), problem.histogram.size() * 4);
  Buffer output(std::size_t(problem.nodes) * problem.cols * sizeof(Split));
  std::memset(output.data(), 0x5a, output.size());
  Buffer cut(std::max<std::size_t>(problem.nodes, 1) * 4), statistics(6 * 4);
  cut.zero(); statistics.zero();
  ScreenParams screen{0, static_cast<u32>(node_cut), 1, static_cast<u32>(parents != nullptr)};
  Buffer inherited = parents ? Buffer(parents->data(), parents->size() * sizeof(Split)) : Buffer(sizeof(Split));
  Command command("screening feature records");
  Extent groups{problem.cols, problem.nodes, 1};
  if (root && screened)
    command.dispatch(kernel, {histogram, in.scales, in.cuts, in.counts, in.minima, in.monotone, in.bounds,
                              in.allowed, output, in.dummy, in.dummy, in.dummy, in.dummy, in.node_sums, in.params,
                              statistics}, screen, groups);
  else if (root)
    command.dispatch(kernel, {histogram, in.scales, in.cuts, in.counts, in.minima, in.monotone, in.bounds,
                              in.allowed, output, in.dummy, in.dummy, in.dummy, in.dummy, in.node_sums, in.params},
                     nullptr, 0, groups);
  else if (screened)
    command.dispatch(kernel, {histogram, in.scales, in.cuts, in.counts, in.minima, in.monotone, in.bounds,
                              in.allowed, output, in.dummy, in.dummy, in.dummy, in.dummy, in.node_sums, in.params,
                              in.parent, in.subtraction, cut, statistics, inherited}, screen, groups);
  else
    command.dispatch(kernel, {histogram, in.scales, in.cuts, in.counts, in.minima, in.monotone, in.bounds,
                              in.allowed, output, in.dummy, in.dummy, in.dummy, in.dummy, in.node_sums, in.params,
                              in.parent, in.subtraction}, nullptr, 0, groups);
  command.wait();
  Run run;
  auto records = output.as<Split>();
  run.records.assign(records, records + std::size_t(problem.nodes) * problem.cols);
  std::memcpy(run.statistics.data(), statistics.data(), sizeof(run.statistics));
  return run;
}

std::vector<Split> select(Problem const& problem, Inputs const& in, std::vector<Split> const& records) {
  Buffer candidates(records.data(), records.size() * sizeof(Split)), best(problem.nodes * sizeof(Split));
  SplitParams p{problem.nodes, problem.cols, problem.bins, 0, problem.lambda, 0, problem.minimum, 0,
                4, 64, 256, 0};
  Command command("screening select");
  command.dispatch("select_splits", {candidates, best, in.scales, in.bounds}, p,
                   {div_up(problem.nodes, 64), 1, 1}, {64, 1, 1});
  command.wait();
  auto result = best.as<Split>();
  return {result, result + problem.nodes};
}

bool same(Split const& a, Split const& b) { return !std::memcmp(&a, &b, sizeof(Split)); }

struct FeatureCounts { std::uint64_t records{}, problems{}, candidates{}, exact{}, overflow{}, cut_records{}; };

void check_features(FeatureCounts& totals, std::mt19937_64& rng) {
  struct Twin { char const* unscreened; char const* screened; int mode; };  // 0 subtracted, 1 terminal, 2 root
  Twin twins[] = {
    {"evaluate_splits_compact_subtracted_gain_stats_score", "evaluate_splits_screened_subtracted_gain_stats_score", 0},
    {"evaluate_splits_compact_subtracted_gain_stats", "evaluate_splits_screened_subtracted_gain_stats", 0},
    {"evaluate_splits_parallel_sparse_subtracted_gain_stats", "evaluate_splits_screened_subtracted_gain_stats", 0},
    {"evaluate_splits_compact_subtracted_gain", "evaluate_splits_screened_subtracted_gain", 0},
    {"evaluate_splits_parallel_sparse_subtracted_gain", "evaluate_splits_screened_subtracted_gain", 0},
    {"evaluate_splits_compact_terminal_gain_stats_score", "evaluate_splits_screened_terminal_gain_stats_score", 1},
    {"evaluate_splits_compact_terminal_gain_stats", "evaluate_splits_screened_terminal_gain_stats", 1},
    {"evaluate_splits_parallel_sparse_terminal_gain_stats", "evaluate_splits_screened_terminal_gain_stats", 1},
    {"evaluate_splits_parallel_sparse_terminal_gain", "evaluate_splits_screened_terminal_gain", 1},
    {"evaluate_splits_parallel_root_gain_stats_score", "evaluate_splits_screened_root_gain_stats_score", 2},
    {"evaluate_splits_parallel_root_gain_stats", "evaluate_splits_screened_root_gain_stats", 2},
    {"evaluate_splits_parallel_root_gain", "evaluate_splits_screened_root_gain", 2},
  };
  u32 bin_choices[] = {1, 2, 31, 32, 33, 200, 256};
  for (int pattern = 0; pattern < kPatterns; ++pattern) {
    for (u32 bins : bin_choices) {
      auto complete = make_problem(pattern, 8, 7, bins, rng);
      auto subtracted = poisoned(complete);
      ++totals.problems;
      for (auto const& twin : twins) {
        bool root = twin.mode == 2;
        auto const& problem = root ? complete : subtracted;
        auto in = upload(problem, root);
        auto label = std::string(pattern_name(pattern)) + " bins " + std::to_string(bins) + " " + twin.screened;
        auto expected = evaluate(problem, in, twin.unscreened, false, root, false);
        auto actual = evaluate(problem, in, twin.screened, true, root, false);
        for (std::size_t i = 0; i < expected.records.size(); ++i)
          check(same(expected.records[i], actual.records[i]),
                "Screened record differs from " + std::string(twin.unscreened) + " at " + std::to_string(i) +
                " (" + label + ")");
        totals.records += expected.records.size();
        totals.candidates += actual.statistics[0];
        totals.exact += actual.statistics[3];
        if (actual.statistics[3] > 256) ++totals.overflow;
        if (root) continue;
        // Inherited parent statistics: each child's weight and cover come
        // from its pair's selected split, as the GPU driver provides them.
        std::vector<Split> parents(problem.subtraction.size());
        for (std::size_t pair = 0; pair < parents.size(); ++pair) {
          auto const& left = expected.records[2 * pair * problem.cols];
          auto const& right = expected.records[(2 * pair + 1) * problem.cols];
          parents[pair].left_weight = left.parent_weight; parents[pair].left_h = left.total_h;
          parents[pair].right_weight = right.parent_weight; parents[pair].right_h = right.total_h;
        }
        auto inherited = evaluate(problem, in, twin.screened, true, root, false, &parents);
        for (std::size_t i = 0; i < expected.records.size(); ++i)
          check(same(expected.records[i], inherited.records[i]),
                "Inherited-statistics record differs at " + std::to_string(i) + " (" + label + ")");
        // Node-wide cutoff: winners and every record that can tie them are
        // identical. Other features may skip candidates (or all of them, with
        // a losing -inf record) but stay strictly below the node maximum.
        auto cut = evaluate(problem, in, twin.screened, true, root, true, (bins & 1) ? &parents : nullptr);
        auto best_expected = select(problem, in, expected.records);
        auto best_actual = select(problem, in, cut.records);
        for (u32 node = 0; node < problem.nodes; ++node) {
          check(same(best_expected[node], best_actual[node]),
                "Node winner differs with the node cutoff at node " + std::to_string(node) + " (" + label + ")");
          float maximum = -std::numeric_limits<float>::infinity();
          for (u32 f = 0; f < problem.cols; ++f)
            maximum = std::max(maximum, expected.records[node * problem.cols + f].gain);
          for (u32 f = 0; f < problem.cols; ++f) {
            auto const& e = expected.records[node * problem.cols + f];
            auto const& a = cut.records[node * problem.cols + f];
            if (same(e, a)) continue;
            ++totals.cut_records;
            check(maximum > -std::numeric_limits<float>::infinity() && e.gain < maximum &&
                  (a.gain < maximum || float_bits(a.gain) == kNegativeInfinity),
                  "A node-cut record could reach the node maximum at node " + std::to_string(node) +
                  " feature " + std::to_string(f) + " (" + label + "): unscreened gain " +
                  std::to_string(e.gain) + " order feature " + std::to_string(e.feature) + " threshold " +
                  std::to_string(e.threshold) + ", screened gain " + std::to_string(a.gain) + " feature " +
                  std::to_string(a.feature) + " threshold " + std::to_string(a.threshold) +
                  ", node maximum " + std::to_string(maximum));
          }
        }
      }
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::size_t cases = argc > 1 ? std::stoull(argv[1]) : 250000;
    std::mt19937_64 rng(20261102);
    auto counts = check_candidates(cases, rng);
    std::cout << "candidates: " << counts.valid << " certified enclosures, " << counts.invalid
              << " certain rejections, " << counts.uncertain << " uncertain, " << counts.parent_fallback
              << " parent fallbacks\n";
    check(counts.valid > cases && counts.invalid > cases / 50 && counts.uncertain > cases / 50,
          "Insufficient candidate coverage");
    FeatureCounts features;
    check_features(features, rng);
    std::cout << "features: " << features.records << " records over " << features.problems << " problems, "
              << features.candidates << " candidates, " << features.exact << " exactly scored, "
              << features.overflow << " overflowing groups, " << features.cut_records
              << " node-cut records\n";
    check(features.overflow > 0 && features.cut_records > 0, "Insufficient feature coverage");
    std::cout << "OK\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
  }
}
