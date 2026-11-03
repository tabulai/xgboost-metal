#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using u32 = uint;
using i32 = int;
using u64 = ulong;
using i64 = long;
#else
#include <cstdint>
namespace metalgb {
using u32 = std::uint32_t;
using i32 = std::int32_t;
using u64 = std::uint64_t;
using i64 = std::int64_t;
#endif

struct Pair { float g, h; };
struct IntPair { i64 g, h; };
struct LinearParams {
  u32 rows, cols, groups, group;
  u32 feature, all_features, tiles, has_weight;
  float learning_rate, reg_alpha, reg_lambda, tolerance;
};
struct LinearUpdate { float value; u32 feature; };
enum : u32 { kLinearRowsPerTile = 4096, kLinearRealBytes = 16 };
struct Node {
  i32 left{-1}, right{-1}, feature{-1}, parent{-1};
  float threshold{0}, leaf{0}, cover{0}, gain{0};
  u32 default_left{0}, split_type{0}, category_begin{0}, category_count{0};
};
struct TreeMeta { u32 begin, count, group, round; u64 leaf_begin{0}; u32 outputs{1}, padding{0}; };
struct ShapPath {
  u32 feature, missing; float zero, lower, upper;
  u32 category_begin{0}, category_count{0}, category_mode{0};
};
struct ShapLeaf { u32 begin, count, group, round; float value, expectation; u32 tree, padding; };
struct MatrixParams { u32 rows, cols, stride, bins, numeric_bins; };
struct CategoryRecodeParams { u32 rows, cols, construction, column_major; };
struct QuantileValueParams { u32 rows, cols, bins, representative, numeric_only; };
struct PackBinsParams { u32 rows, cols, bins, chunk; };
struct NormalizeParams { u32 rows, cols; float missing; u32 block_stride; };
struct SortParams { u32 rows, cols, stride, width; };
struct SketchStats { u32 nonmissing, elements, runs, summary; };
struct SketchInputParams { u32 rows, cols, stride, unit_weights; };
struct SketchInputBlock {
  u32 elements, nonmissing, runs, offset;
  float first, last;
  u32 run_offset, continued;
};
struct SketchBatch { u64 input, offset; u32 size, stride, limit, reserved; };
struct SketchFeature {
  u64 level_offset, work_offset;
  u32 limit, budget, batch_begin, batches, levels, reserved;
};
struct SketchParams { u32 tasks, width, features, reserved; };
struct SketchCutParams { MatrixParams matrix; u32 tiles, capacity, padding; };
struct GradientParams {
  u32 rows, groups, label_cols, objective;
  u32 has_weight, seed, iteration, reserved;
  float scale_pos_weight, subsample, parameter, padding;
};
struct SamplingParams {
  u32 rows, groups, group, sample_rows;
  u32 seed, method, stride, padding;
  float subsample, padding1, padding2, padding3;
};
enum : u32 { kObjectiveMassWords = 10, kObjectiveRowsPerTile = 4096 };
enum : u32 { kGradientPartialWords = 2 * kObjectiveMassWords, kGradientRowsPerTile = 4096 };
enum : u32 { kProductMassWords = 20, kVarianceMassWords = 28 };
struct InitialMean { float value; u32 valid; };
struct SurvivalParams {
  u32 rows, distribution, mode, seed;
  float scale, subsample;
  u32 iteration, reserved;
};
enum : u32 { kSurvivalSumBytes = 32 };
struct CoxScanParams { u32 rows, blocks, segmented, reserved; };
enum : u32 { kCoxScanRows = 256, kCoxEstimateBytes = 48, kCoxTransformBytes = 48 };
struct RankingParams {
  u32 rows, queries, objective, num_pairs;
  u32 mean, normalization, score_normalization, exp_gain;
  u32 unbiased, positions, seed, iteration;
  u32 mode, width, tiles, max_size;
  float subsample, regularizer_high, regularizer_low;
  u32 minus;
  i32 regularizer_exponent;
  u32 padding0, padding1, padding2;
};
struct RankingTile { u64 begin; u32 size, query; };
struct RankingStats { float norm; i32 grad_shift, hess_shift, cost_shift; };
enum : u32 { kRankingPartialBytes = 48, kRankingRealBytes = 16 };
struct QuantileParams { u32 rows, label_cols, outputs, alphas, pass, padding; };
struct QuantileState { u32 prefix, valid; u32 rank[kObjectiveMassWords]; };
struct HistTask { u32 begin, end, node, feature; };
struct HistSubtract { u32 parent, built, other, padding; };
struct HistParams { u32 rows, cols, bins, groups, group, tasks, padding0, padding1; };
struct HistogramPlanParams { u32 nodes, cols, features, rows_per_tile, subtract, padding; };
// GPU-tree histogram dataflow flags. HistParams.tasks is otherwise unused by
// histogram kernels: its low bits select row-tile records and their decode
// order, and its high half holds the packed-bin words per row chunk.
// HistogramPlanParams.padding selects tile records and single-owner tags.
// A tagged HistTask.node is the only task building that histogram slice.
enum : u32 {
  kHistogramTileRecords = 1, kHistogramTileMajor = 2, kHistogramPackedShift = 16,
  kHistogramPlanTiles = 1, kHistogramPlanSingleOwner = 2,
  kHistogramSingleOwnerTag = 0x80000000u
};
struct SplitParams {
  u32 nodes, cols, bins, group;
  float reg_lambda, reg_alpha, min_child_weight, max_delta_step;
  u32 max_cat_to_onehot, max_cat_threshold, sort_stride, has_categorical;
  u32 use_parent_sums{0}, padding0{0}, padding1{0}, padding2{0};
};
// Certified split screening: cut_offset selects this level's node-cutoff
// cells, node_cut enables them, statistics enables candidate counters and
// inherit takes nonroot parent statistics from the parent's selected split.
struct ScreenParams { u32 cut_offset{0}, node_cut{0}, statistics{0}, inherit{0}; };
struct Split {
  // Applied scalar statistics are filled only after the winning feature is
  // selected. parent_weight/total_h retain the original node statistics.
  float gain, threshold, applied_parent_weight, left_h, applied_parent_hessian, right_h;
  i32 feature;
  u32 default_left;
  float left_weight, right_weight, total_g, total_h;
  u32 split_type{0}, category_begin{0}, category_count{0};
  float parent_weight{0};
  IntPair left_sum{}, right_sum{};
};
struct ScalarTileParams { SplitParams split; u32 bin_tile, bin_tiles, padding0, padding1; };
struct ScalarTileChoice { Split split; u32 order, padding0, padding1, padding2; };
struct VectorSplitParams { SplitParams split; u32 targets, has_monotone, bin_tile, bin_tiles; };
struct VectorSplitChoice { float gain; u32 bin, reverse, valid; };
struct VectorWeight { float parent, left, right, hessian; IntPair left_sum{}, right_sum{}; };
struct VectorLeafParams { SplitParams split; u32 rows, targets; float eta; u32 padding; };
struct PartitionParams { u32 rows, cols, nodes, padding; };
struct PartitionTask { u32 begin, end, node, segment; };
struct PartitionSegment { u32 begin, end, task_begin, task_end, left, right; };
// GPU depthwise control records tree history without materializing model
// floating-point values. The host replays the existing ApplyTreeSplit path
// after the single tree completion wait.
struct TreeDriverState { u32 node_count, frontier_count, depth, error; };
struct TreeDriverParams {
  SplitParams split;
  HistogramPlanParams histogram;
  PartitionParams partition;
  u32 max_depth, node_capacity, parallel_splits, histogram_entries;
  float min_gain;
  u32 control_flags, padding1, padding2;
};
struct TreeDriverDispatch { u32 x, y, z; };
enum : u32 { kTreeDriverEvaluated = 1, kTreeDriverExpanded = 2 };
enum : u32 { kTreeDriverCapacityError = 1, kTreeDriverNodeError = 2 };
// control_flags select the grid shape that prepare writes for parallel
// control kernels: one SIMD group per selected node, one threadgroup per
// partition parent. kTreeDriverDenseRoot marks a root whose applied
// statistics come from select_dense_root_split, not deferred selection.
// Parallel expansion handles at most kTreeDriverSlots.
enum : u32 { kTreeDriverParallelSelect = 1, kTreeDriverParallelPlan = 2, kTreeDriverDenseRoot = 4 };
enum : u32 { kTreeDriverSlots = 512 };
// Node-table emission for asynchronous GPU-driven rounds.
struct TreeEmitParams { float eta; u32 group, round, node_capacity; };
struct TreeEmitSweepParams { u32 begin, count; float eta; u32 padding; };
struct RefreshParams {
  u32 rows, cols, groups, group;
  u32 count, nodes, begin, depth;
  float learning_rate, reg_alpha, reg_lambda, max_delta_step;
  float min_split_loss;
  u32 max_depth, refresh_leaf, column_major;
};
enum : u32 { kRefreshRowsPerTile = 4096 };
struct PredictParams {
  u32 rows, cols, groups, trees;
  u32 begin, end, objective, type;
  u32 has_margin, margin_groups, has_tree_weights, bin_partition;
  u32 tree_begin{0};
};
struct MetricParams {
  u32 rows, groups, label_cols, type;
  u32 stride, multiclass, padding0, padding1;
  float parameter, padding2, padding3, padding4;
};
struct MetricRatio { float high, low; i32 exponent; u32 valid; };

#ifndef __METAL_VERSION__
static_assert(sizeof(Node) == 48);
static_assert(sizeof(TreeMeta) == 32);
static_assert(sizeof(PredictParams) == 52);
static_assert(sizeof(IntPair) == 16);
static_assert(sizeof(SplitParams) == 64);
static_assert(sizeof(Split) == 96);
static_assert(sizeof(ScreenParams) == 16);
static_assert(sizeof(ScalarTileParams) == 80);
static_assert(sizeof(ScalarTileChoice) == 112);
static_assert(sizeof(VectorSplitParams) == 80);
static_assert(sizeof(VectorSplitChoice) == 16);
static_assert(sizeof(VectorWeight) == 48);
static_assert(sizeof(VectorLeafParams) == 80);
static_assert(sizeof(ShapPath) == 32);
static_assert(sizeof(ShapLeaf) == 32);
static_assert(sizeof(QuantileState) == 48);
static_assert(sizeof(QuantileValueParams) == 20);
static_assert(sizeof(SketchCutParams) == 32);
static_assert(sizeof(MetricRatio) == 16);
static_assert(sizeof(InitialMean) == 8);
static_assert(sizeof(SurvivalParams) == 32);
static_assert(sizeof(CoxScanParams) == 16);
static_assert(sizeof(RankingParams) == 96);
static_assert(sizeof(RankingTile) == 16);
static_assert(sizeof(RankingStats) == 16);
static_assert(sizeof(RefreshParams) == 64);
static_assert(sizeof(SamplingParams) == 48);
static_assert(sizeof(TreeDriverState) == 16);
static_assert(sizeof(TreeDriverParams) == 136);
static_assert(sizeof(TreeDriverDispatch) == 12);
static_assert(sizeof(TreeEmitParams) == 16);
static_assert(sizeof(TreeEmitSweepParams) == 16);
}  // namespace metalgb
#endif
