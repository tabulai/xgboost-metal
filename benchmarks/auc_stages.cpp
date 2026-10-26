// Standalone timing experiment: unchanged baseline metric kernels and runtime.
// Compile against the baseline's existing build/metal_library.hpp; no Metal build.
#include "metalgb/runtime.hpp"
#include "metalgb/shared.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using metalgb::Buffer;
using metalgb::Command;
using metalgb::MetricParams;
using metalgb::SortParams;
using metalgb::u32;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static_assert(sizeof(MetricParams) == 48);
static_assert(sizeof(SortParams) == 16);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

namespace {
double seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}
void require(bool value, std::string const& message) {
  if (!value) throw std::runtime_error(message);
}
struct Options {
  u32 rows{100000}, groups{1}, repeats{5}, warmup{1}, seed{1729};
  std::string scores{"both"}, metric{"both"};
};
u32 number(std::string const& value) {
  require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
          "Numeric arguments must be unsigned decimal integers.");
  auto n = std::stoull(value);
  require(n <= std::numeric_limits<u32>::max(), "Numeric argument exceeds uint32.");
  return static_cast<u32>(n);
}
Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    require(i + 1 < argc, "Every option requires a value.");
    std::string value = argv[++i];
    if (key == "--rows") o.rows = number(value);
    else if (key == "--groups") o.groups = number(value);
    else if (key == "--repeats") o.repeats = number(value);
    else if (key == "--warmup") o.warmup = number(value);
    else if (key == "--seed") o.seed = number(value);
    else if (key == "--scores") o.scores = value;
    else if (key == "--metric") o.metric = value;
    else throw std::runtime_error("Unknown option: " + key);
  }
  require(o.rows >= 2 && o.rows <= (1u << 24), "Rows must be in [2, 2^24].");
  require(o.groups >= 1 && o.groups <= 256, "Groups must be in [1, 256].");
  require(o.groups == 1 || o.rows >= 2 * o.groups, "Multiclass needs at least two rows per class.");
  require(o.repeats >= 1 && o.repeats <= 100 && o.warmup <= 10,
          "Repeats must be in [1, 100], warmup in [0, 10].");
  require(std::size_t(o.rows) * o.groups <= (std::size_t{1} << 26),
          "Rows * groups exceeds the profiler's memory bound (2^26).");
  require(o.scores == "unique" || o.scores == "ties" || o.scores == "both",
          "Scores must be unique, ties, or both.");
  require(o.metric == "auc" || o.metric == "aucpr" || o.metric == "both",
          "Metric must be auc, aucpr, or both.");
  return o;
}
struct Inputs {
  Buffer predictions, labels, weights;
  std::vector<u32> positives;
  Inputs(Options const& o, bool ties) : positives(o.groups, 0) {
    std::vector<float> p(std::size_t(o.rows) * o.groups), y(o.rows), w(o.rows, 1.0f);
    for (u32 row = 0; row < o.rows; ++row) {
      u32 label = row % (o.groups == 1 ? 2u : o.groups);
      y[row] = static_cast<float>(label);
      if (o.groups == 1) positives[0] += label;
      else ++positives[label];
      for (u32 group = 0; group < o.groups; ++group) {
        // Odd affine permutation modulo 2^24: injective for all allowed rows.
        // Power-of-two scaling preserves every rank exactly as float32.
        u32 rank = (row * 1664525u + group * 1013904223u + o.seed) & 0x00ffffffu;
        // Each complete label cycle shares one tied score, ensuring mixed
        // positive/negative labels within ties even for the smallest input.
        u32 tie_bucket = ((row / (o.groups == 1 ? 2u : o.groups)) * 13u + group * 7u + o.seed) & 31u;
        p[std::size_t(row) * o.groups + group] = ties
            ? static_cast<float>(tie_bucket) * 0x1p-5f
            : static_cast<float>(rank) * 0x1p-24f;
      }
    }
    predictions = Buffer(p.data(), p.size() * sizeof(float));
    labels = Buffer(y.data(), y.size() * sizeof(float));
    weights = Buffer(w.data(), w.size() * sizeof(float));
  }
};
struct Workspace {
  Buffer first, second, output;
  explicit Workspace(MetricParams const& p)
      : first(std::size_t(p.stride) * p.groups * 4 * sizeof(float)),
        second(first.size()), output(std::size_t(p.groups) * 4 * sizeof(float)) {}
  void poison_output() const {
    std::fill_n(output.as<float>(), output.size() / sizeof(float),
                std::numeric_limits<float>::quiet_NaN());
  }
};
Buffer encode_sort(Command& command, Inputs const& in, Workspace const& w, MetricParams const& p) {
  Buffer a = w.first, b = w.second;
  command.dispatch("metric_sort", {in.predictions, in.labels, in.weights, a}, p,
                   {metalgb::div_up(p.rows, 256), p.groups, 1});
  for (u32 width = 256; width < p.stride; width *= 2) {
    SortParams sort{p.rows, p.groups, p.stride, width};
    command.dispatch("metric_merge", {a, b}, sort,
                     {metalgb::div_up(p.stride, 256), p.groups, 1});
    std::swap(a, b);
  }
  return a;
}
void encode_reduce(Command& command, Buffer const& sorted, Workspace const& w, MetricParams const& p) {
  command.dispatch("auc_reduce", {sorted, w.output}, p,
                   {metalgb::div_up(p.groups, 32), 1, 1}, {32, 1, 1});
}
struct Timing {
  double gpu, encode, wait, command;
  std::size_t encoders;
  json as_json() const {
    return {{"gpu_seconds", gpu}, {"host_encode_seconds", encode},
            {"host_wait_seconds", wait}, {"host_command_seconds", command}, {"encoders", encoders}};
  }
};
template <class Encode>
Timing timed_command(char const* label, Encode encode) {
  auto begin = Clock::now();
  Command command(label, false);
  encode(command);
  auto before_wait = Clock::now();
  double gpu = command.wait();
  auto end = Clock::now();
  require(std::isfinite(gpu) && gpu >= 0, "Invalid GPU timestamp duration.");
  return {gpu, seconds(begin, before_wait), seconds(before_wait, end),
          seconds(begin, end), command.encoder_count()};
}
struct Run { Buffer sorted; json timing; };
Run run_separated(Inputs const& in, Workspace const& w, MetricParams const& p) {
  w.poison_output();
  Run result;
  auto begin = Clock::now();
  auto sort = timed_command("AUC profiler sort and merge", [&](Command& c) {
    result.sorted = encode_sort(c, in, w, p);
  });
  auto reduce = timed_command("AUC profiler reduction", [&](Command& c) {
    encode_reduce(c, result.sorted, w, p);
  });
  auto end = Clock::now();
  result.timing = {{"sort_merge", sort.as_json()}, {"reduce", reduce.as_json()},
                   {"gpu_seconds_sum", sort.gpu + reduce.gpu},
                   {"host_whole_seconds", seconds(begin, end)}};
  return result;
}
Run run_combined(Inputs const& in, Workspace const& w, MetricParams const& p) {
  w.poison_output();
  Run result;
  auto begin = Clock::now();
  auto command = timed_command("AUC profiler combined control", [&](Command& c) {
    result.sorted = encode_sort(c, in, w, p);
    encode_reduce(c, result.sorted, w, p);
  });
  auto end = Clock::now();
  result.timing = {{"command", command.as_json()}, {"host_whole_seconds", seconds(begin, end)}};
  return result;
}
double aggregate(Buffer const& output, MetricParams const& p) {
  double sum = 0, weights = 0;
  for (u32 k = 0; k < p.groups; ++k) {
    auto row = output.as<float>() + k * 4;
    if (!(row[1] > 0 && row[2] > 0)) return std::numeric_limits<double>::quiet_NaN();
    double weight = p.multiclass && p.type == 13 ? row[1] : 1;
    sum += row[0] * weight;
    weights += weight;
  }
  return weights > 0 ? sum / weights : std::numeric_limits<double>::quiet_NaN();
}
json verify(Inputs const& in, Workspace const& separated, Workspace const& combined,
            Run const& a, Run const& b, MetricParams const& p) {
  require(std::memcmp(a.sorted.data(), b.sorted.data(), a.sorted.size()) == 0,
          "Separated and combined sorted float4 buffers differ.");
  require(std::memcmp(separated.output.data(), combined.output.data(), separated.output.size()) == 0,
          "Separated and combined per-class float4 output bits differ.");
  json per_class = json::array();
  for (u32 k = 0; k < p.groups; ++k) {
    auto row = combined.output.as<float>() + 4 * k;
    require(std::isfinite(row[0]), "Nonfinite metric area.");
    require(row[1] == static_cast<float>(in.positives[k]) &&
            row[2] == static_cast<float>(p.rows - in.positives[k]) &&
            std::bit_cast<u32>(row[3]) == 0, "Unexpected class masses or output padding.");
    per_class.push_back({{"class", k}, {"area", row[0]}, {"positive_mass", row[1]},
                         {"negative_mass", row[2]}, {"float4_bits", {
                           std::bit_cast<u32>(row[0]), std::bit_cast<u32>(row[1]),
                           std::bit_cast<u32>(row[2]), std::bit_cast<u32>(row[3])}}});
  }
  double av = aggregate(separated.output, p), bv = aggregate(combined.output, p);
  require(std::isfinite(av) && std::bit_cast<std::uint64_t>(av) == std::bit_cast<std::uint64_t>(bv),
          "Host aggregated metric differs or is nonfinite.");
  return {{"sorted_float4_bitwise_equal", true}, {"output_float4_bitwise_equal", true},
          {"host_metric_bitwise_equal", true}, {"metric_value", av}, {"per_class", per_class}};
}
json summarize(json const& records, char const* path) {
  std::vector<double> values;
  for (auto const& record : records) values.push_back(record.at(json::json_pointer(path)).get<double>());
  std::sort(values.begin(), values.end());
  auto n = values.size();
  double median = n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
  double sum = 0;
  for (auto x : values) sum += x;
  return {{"count", n}, {"min", values.front()}, {"median", median},
          {"mean", sum / n}, {"max", values.back()}};
}
}  // namespace

int main(int argc, char** argv) {
  json report = {{"schema_version", 1}, {"status", "running"}, {"cases", json::array()}};
  try {
    auto o = parse(argc, argv);
    report["options"] = {{"rows", o.rows}, {"groups", o.groups}, {"repeats", o.repeats},
                          {"warmup_pairs", o.warmup}, {"seed", o.seed}, {"scores", o.scores}, {"metric", o.metric}};
    report["scope"] = {
      {"weights", "All ones, one float per row; unweighted binary or one-vs-rest multiclass."},
      {"predictions", "Row-major finite ranking scores, not normalized multiclass probabilities."},
      {"inputs", "Labels row modulo class count (binary modulo 2); unique scores use exact 24-bit affine permutation; tied scores use 32 buckets with every complete label cycle sharing a score."},
      {"command_coalescing", false},
      {"timing", "GPU seconds are Command::wait timestamps. Host command time spans construction, encoding and wait; whole time also includes command destruction/bookkeeping. Buffers and pipelines are allocated/prewarmed before timing; validation follows both modes."},
      {"interpretation", "Separated mode adds a command boundary and CPU wait. Its GPU sum need not equal combined GPU time; cache and scheduling can differ. Each mode uses its own reusable scratch buffers and the same immutable inputs; execution order alternates."},
      {"validation", "Same existing kernels in separated versus combined commands; bitwise sorted buffers, all four output floats per class, and host aggregate. Known class masses checked. This is not an independent CPU metric correctness test."}};
    auto setup_begin = Clock::now();
    auto& runtime = metalgb::Runtime::get();
    for (auto name : {"metric_sort", "metric_merge", "auc_reduce"}) (void)runtime.pipeline(name);
    report["runtime"] = json::parse(runtime.info());
    report["runtime_pipeline_setup_host_seconds"] = seconds(setup_begin, Clock::now());
    std::vector<std::string> scores = o.scores == "both" ? std::vector<std::string>{"unique", "ties"} : std::vector<std::string>{o.scores};
    std::vector<std::string> metrics = o.metric == "both" ? std::vector<std::string>{"auc", "aucpr"} : std::vector<std::string>{o.metric};
    for (auto const& score : scores) {
      auto input_begin = Clock::now();
      Inputs inputs(o, score == "ties");
      double input_seconds = seconds(input_begin, Clock::now());
      for (auto const& metric : metrics) {
        MetricParams p{o.rows, o.groups, 1, metric == "auc" ? 13u : 14u,
                       static_cast<u32>(metalgb::div_up(o.rows, 256) * 256),
                       o.groups > 1 ? 1u : 0u, 0, 0, 1, 0, 0, 0};
        Workspace separate(p), combined(p);
        report["cases"].push_back({{"scores", score}, {"metric", metric}, {"stride", p.stride},
                                    {"input_generation_upload_host_seconds", input_seconds},
                                    {"scratch_bytes_per_mode", separate.first.size() * 2 + separate.output.size()},
                                    {"warmups", json::array()}, {"measurements", json::array()}});
        auto& item = report["cases"].back();
        for (u32 i = 0; i < o.warmup + o.repeats; ++i) {
          bool warm = i < o.warmup;
          u32 index = warm ? i : i - o.warmup;
          bool separate_first = index % 2 == 0;
          Run a, b;
          if (separate_first) { a = run_separated(inputs, separate, p); b = run_combined(inputs, combined, p); }
          else { b = run_combined(inputs, combined, p); a = run_separated(inputs, separate, p); }
          auto verified = verify(inputs, separate, combined, a, b, p);
          auto& records = item[warm ? "warmups" : "measurements"];
          records.push_back({{"index", index}, {"execution_order", separate_first ? "separated_then_combined" : "combined_then_separated"},
                             {"separated", a.timing}, {"combined", b.timing}, {"verification", verified}});
          std::cerr << score << ' ' << metric << ' ' << (warm ? "warmup " : "measured ") << index << " exact\n";
        }
        auto const& measured = item.at("measurements");
        item["summary_seconds"] = {
          {"separated_sort_merge_gpu", summarize(measured, "/separated/sort_merge/gpu_seconds")},
          {"separated_reduce_gpu", summarize(measured, "/separated/reduce/gpu_seconds")},
          {"separated_gpu_sum", summarize(measured, "/separated/gpu_seconds_sum")},
          {"combined_gpu", summarize(measured, "/combined/command/gpu_seconds")},
          {"separated_host_whole", summarize(measured, "/separated/host_whole_seconds")},
          {"combined_host_whole", summarize(measured, "/combined/host_whole_seconds")}};
        item["status"] = "passed";
      }
    }
    report["status"] = "passed";
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (std::exception const& error) {
    report["status"] = "failed";
    report["error"] = error.what();
    std::cout << report.dump(2) << '\n';
    std::cerr << error.what() << '\n';
    return 1;
  }
}
