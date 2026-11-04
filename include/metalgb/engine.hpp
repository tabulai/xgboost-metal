#pragma once
#include "metalgb/runtime.hpp"
#include "metalgb/shared.h"
#include "metalgb/sparse.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace metalgb {
using Json = nlohmann::json;
inline constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
Json parse_json(std::string_view value, bool ieee_numbers = false);
std::string dump_json(Json const& value, bool tree_dump = false, int indent = -1);
float missing_value(Json const& config);
void require(bool condition, std::string const& message);
void warning(std::string_view message);
std::size_t checked_product(std::size_t a, std::size_t b);
void prefix_sum_u32(Command& command, Buffer const& counts, Buffer const& offsets, u32 n,
                    std::vector<Buffer>& keepalive);

struct IngestColumnSpec {
  std::uintptr_t data_ptr{};
  u32 rows{}, cols{1}, dtype_code{}, mask_mode{kIngestMaskNone};
  i64 stride_r{}, stride_c{}, data_bit_offset{0};
  std::uintptr_t mask_ptr{};
  i64 mask_bit_offset{0};
  bool prefer_zero_copy{false};
};

struct Array {
  std::uintptr_t pointer{};
  std::vector<std::size_t> shape;
  std::vector<std::ptrdiff_t> strides;
  std::string type;
  std::size_t width{};
  Json mask;
  explicit Array(Json const& description);
  explicit Array(IngestColumnSpec const& spec);
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] double at(std::size_t row, std::size_t column = 0) const;
  [[nodiscard]] std::vector<float> floats() const;
  void copy_floats(std::span<float> destination) const;
  [[nodiscard]] IngestColumnSpec to_ingest_spec() const;
 private:
  std::uintptr_t mask_pointer_{};
  std::size_t mask_bit_offset_{0};
  bool mask_bits_{false};
  [[nodiscard]] bool valid(std::size_t row) const;
};
Json array_interface(void const* pointer, std::vector<std::size_t> shape, std::string type = "<f4");
Json category_column(Json const& descriptor);
Json serialize_categories(Json const& categories);
Json deserialize_categories(Json const& encoded);
Buffer recode_category_input(Buffer const& values, u32 rows, u32 cols,
                             Json const& categories, Json const& reference, bool column_major);
struct CategoryStorage {
  Json columns;
  std::vector<Buffer> buffers;
  explicit CategoryStorage(Json const& values) : columns(values) {}
  [[nodiscard]] Json array_interfaces();
};

struct SparseStorage {
  u32 nnz{0};
  Buffer csr_row_ptr;   // (rows + 1) * u32
  Buffer csr_col_idx;   // nnz * u32 (strictly ascending unique column IDs per row)
  Buffer csr_val;       // nnz * float
  Buffer csr_bins;      // nnz * u16 or u32 (quantized bins per nonzero)
  mutable Buffer csc_col_ptr;  // (cols + 1) * u32 (built lazily on GPU)
  mutable Buffer csc_row_idx;  // nnz * u32 (strictly ascending row IDs per column)
  mutable Buffer csc_val;      // nnz * float
};

struct Matrix : std::enable_shared_from_this<Matrix> {
  u32 rows{}, cols{}, label_cols{1};
  std::uint64_t generation{0};
  bool was_zero_copied{false};
  Buffer data;
  // Native GPU CSR/CSC storage for low-density matrices (avoids rows * cols dense allocation).
  std::shared_ptr<SparseStorage> sparse_;
  // Raw CSR is retained for sparse input and binary exchange when duplicates exist.
  std::shared_ptr<SparsePage> sparse_page;
  std::map<std::string, std::vector<float>, std::less<>> info;
  std::array<std::size_t, 2> label_shape{}, margin_shape{};
  std::vector<u32> group_ptr;
  std::vector<std::string> feature_names, feature_types;
  Json categories = Json::array();
  std::vector<std::int32_t> category_sorted_idx;
  Buffer labels, weights, sketch_weights, margins, lower_bounds, upper_bounds;
  u32 max_bin{0}, histogram_bins{0};
  u32 cut_tile_size{0}, cut_tile_count{0};
  u32 summary_group_size{0}, summary_batch_count{0};
  u32 weighted_summary_stages{0}, weighted_summary_batch_count{0};
  bool is_simple{true}, is_quantile{false}, quantized_dense{false};
  double quantize_seconds{0};
  std::uint64_t quantize_commands{0};
  Buffer bins, cuts, cut_counts, cut_minima, categorical;
  std::vector<std::uint64_t> cut_ptr;
  std::vector<float> cut_values;

  Matrix();
  Matrix(u32 rows, u32 cols, std::vector<float> values, float missing = kNaN);
  Matrix(u32 rows, u32 cols, Buffer values, float missing = kNaN, bool column_major = false);
  static std::shared_ptr<Matrix> dense(Json const& array, float missing);
  static std::shared_ptr<Matrix> columnar(Json const& array, float missing);
  static std::shared_ptr<Matrix> from_ingest_2d(IngestColumnSpec const& spec, float missing);
  static std::shared_ptr<Matrix> from_ingest_columns(std::span<IngestColumnSpec const> columns,
                                                     u32 rows, float missing);
  static std::shared_ptr<Matrix> sparse(Json const& indptr, Json const& indices, Json const& values,
                                      std::size_t dimension, bool csc, float missing,
                                      int sparse_mode = -1);
  static std::shared_ptr<Matrix> from_coo(Json const& row_indices, Json const& col_indices,
                                          Json const& values, std::size_t rows, std::size_t cols,
                                          float missing, int sparse_mode = -1);
  static std::shared_ptr<Matrix> load_binary(std::string const& filename);
  void save_binary(std::string const& filename) const;
  [[nodiscard]] bool is_sparse() const noexcept { return static_cast<bool>(sparse_); }
  [[nodiscard]] Buffer dense_data() const;
  void ensure_csc() const;
  void ensure_dense_bins();
  [[nodiscard]] SparsePage raw_csr() const;
  void filter_missing(float missing);
  void set_info(std::string const& field, Array const& values);
  void set_info(std::string const& field, std::vector<float> values,
                std::array<std::size_t, 2> shape = {SIZE_MAX, 1});
  [[nodiscard]] std::vector<float> const& get_info(std::string_view field) const;
  void refresh_metadata();
  [[nodiscard]] Buffer data_for_categories(Json const& reference) const;
  void quantize(u32 bins, Matrix const* reference = nullptr);
  void finish_quantile();
  void update_feature_types();
  [[nodiscard]] Buffer export_data() const;
  [[nodiscard]] Buffer partition_data() const;
  [[nodiscard]] Buffer partition_category_bounds() const;
  // Lazily derived uint8 histogram codes interleaved by feature group; empty
  // when a code does not fit. The canonical uint16 bins remain authoritative.
  [[nodiscard]] Buffer packed_bins(u32 chunk) const;
  void make_quantile_cuts(MatrixParams const& params);
  void export_cuts();
  [[nodiscard]] std::shared_ptr<Matrix> slice(std::span<int const> indices, bool allow_groups) const;
  void append(Matrix const& batch);
  void decode_quantiles(Buffer const& output, bool representative, bool numeric_only = false) const;
 private:
  bool cuts_categorical_{false}, quantized_types_changed_{false};
  mutable Buffer partition_data_, partition_bounds_, materialized_dense_;
  mutable Buffer packed_bins_;
  mutable u32 packed_chunk_{0};
  std::weak_ptr<Matrix const> slice_parent_;
  Buffer slice_indices_;
  std::uint64_t slice_parent_generation_{0};
  mutable std::mutex recode_mutex_;
  mutable Json recoded_categories_;
  mutable Buffer recoded_data_;
  mutable std::uint64_t recoded_generation_{0};
};

struct Tree {
  std::vector<Node> nodes;
  u32 group{}, round{};
  std::vector<u32> categories;
  std::vector<float> base_weights;
  u32 outputs{1}, split_outputs{1};
  std::vector<float> leaf_weights;
};
struct Prediction { Buffer values; std::vector<std::uint64_t> shape; };
// GPU outputs of a committed GPU-driven tree whose host replay is deferred
// (metal_async_rounds). The emitted Node table feeds the next round's margin
// update; the replay of state/flags/history remains the model's only source.
struct GpuTreeRound {
  Buffer state, flags, history, nodes, meta;
  float eta{0};
  u32 group{0}, round{0}, node_capacity{0}, batch_index{0};
  bool record_paths{true}, emit_nodes{true}, root_is_batch_only{false};
  Buffer unit_mismatch{}, screen_counters{};
  explicit operator bool() const { return static_cast<bool>(state); }
};
// Host assembly of a completed GPU-driven tree, shared by both round modes.
Tree replay_gpu_tree(GpuTreeRound const& round, u32& evaluated_levels);
// Compares GPU leaf scaling with the host multiply for consecutive binary32
// bit patterns; returns the mismatch count (validation extension).
std::uint64_t validate_leaf_scaling(float eta, std::uint64_t begin, std::uint64_t count,
                                    std::uint32_t* first_mismatch);

class Learner {
 public:
  mutable std::recursive_mutex mutex;
  std::map<std::string, std::string, std::less<>> params;
  std::vector<std::string> metrics;
  std::map<std::string, std::string, std::less<>> attributes;
  std::vector<std::string> feature_names, feature_types;
  Json categories = Json::array();
  std::vector<Tree> trees;
  std::vector<float> linear_weights;
  std::vector<std::weak_ptr<Matrix>> cache_matrices;
  std::vector<float> base_score{0.5f}, base_margin{0.5f};
  u32 num_feature{0}, groups{1}, num_class{0}, num_target{1}, rounds{0};
  bool initialized{false};
  std::map<std::string, double> gpu_seconds;
  std::map<std::string, u32> active_tuning;
  std::uint64_t command_count{0};
  struct ForestUploadStats {
    std::uint64_t full_uploads{}, append_uploads{}, trees_uploaded{}, bytes_uploaded{};
    std::uint64_t buffer_allocations{}, bytes_relocated{};
  } forest_uploads;
  // Cumulative screened-kernel candidate counts (metal_split_screening_stats).
  struct SplitScreeningStats {
    std::uint64_t candidates{}, certified{}, uncertain{}, exact{}, groups{}, exact_groups{};
  } split_screening;

  Learner();
  ~Learner();
  void prepare_objective(std::string_view name);
  void set_param(std::string key, std::string value);
  void clear_linear_parameters();
  void clear_tree_updater();
  void configure();
  [[nodiscard]] std::string str(std::string_view key, std::string fallback = "") const;
  [[nodiscard]] float number(std::string_view key, float fallback) const;
  [[nodiscard]] u32 integer(std::string_view key, u32 fallback) const;
  [[nodiscard]] bool boolean(std::string_view key, bool fallback) const;
  [[nodiscard]] bool is_linear() const { return str("booster", "gbtree") == "gblinear"; }
  [[nodiscard]] bool is_tree_update() const { return !is_linear() && str("process_type", "default") == "update"; }
  [[nodiscard]] u32 objective() const;
  [[nodiscard]] Json ranking_config() const;
  [[nodiscard]] std::vector<float> loss_alphas(std::string_view loss) const;
  void initialize(Matrix const& data, bool custom = false);
  void update(Matrix& data, int iteration, Array const* grad = nullptr, Array const* hess = nullptr,
              Array const* value_grad = nullptr, Array const* value_hess = nullptr);
  Prediction predict(Matrix const& data, Json const& config);
  [[nodiscard]] std::string evaluate(int iteration, std::span<std::shared_ptr<Matrix> const> data,
                                     std::span<std::string const> names);
  [[nodiscard]] Json model();
  void load_model(Json const& model);
  [[nodiscard]] Json config();
  void load_config(Json const& config);
  [[nodiscard]] std::vector<std::string> dump(std::string_view format, bool with_stats);
  [[nodiscard]] std::unique_ptr<Learner> slice(int begin, int end, int step);
  void reset(bool clear_dropout = true);
  void record(std::string const& label, double elapsed) { gpu_seconds[label] += elapsed; ++command_count; }
  // Asynchronous boosting rounds (metal_async_rounds). drain() retires every
  // committed round in order; a failed round rolls the model back to the
  // last good round and its error is rethrown here.
  void drain();
  [[nodiscard]] bool has_inflight_rounds() const { return !inflight_.empty(); }
  // Reusable buffer for an asynchronous round's GPU outputs.
  Buffer round_output(std::size_t bytes);
 private:
  std::vector<Tree> trees_to_update_;
  bool update_original_weights_{false};
  std::vector<std::string> tree_updaters_{"grow_metal_hist"};
  std::vector<Buffer> tree_workspace_;
  std::vector<std::vector<std::uint8_t>> tree_workspace_inputs_;
  // Persistent per-round scratch (metal_round_buffers). Pooled buffers are
  // reused only when no host reference remains, so a margin buffer returned
  // to a caller is never rewritten.
  struct RoundBuffers {
    Buffer gradient, partial, scale, quantized, value_partial, value_scale, value_quantized;
  } round_buffers_;
  std::vector<Buffer> margin_pool_, round_output_pool_, metric_pool_;
  Buffer pooled_buffer(std::vector<Buffer>& pool, std::size_t bytes);
  // A committed round. Its GPU outputs stay referenced until retirement, so
  // the output pool cannot hand them out while the host may still read them.
  // Margin buffers are written only by GPU commands, which the single queue
  // orders, and are read by the host only after drain().
  struct InFlightRound {
    std::unique_ptr<Command> command;
    GpuTreeRound tree;
  };
  std::deque<InFlightRound> inflight_;
  u32 async_tuning_round_{UINT32_MAX};
  void retire_round();
  void abandon_rounds() noexcept;
  std::map<std::string, std::string, std::less<>> validated_params_;
  std::map<std::string, Buffer, std::less<>> refresh_workspace_;
  Buffer refresh_source_;
  std::weak_ptr<Matrix const> refresh_matrix_;
  struct SampledGradients { Buffer values, quantized, scales; SamplingParams params{}; };
  std::map<std::string, Buffer, std::less<>> sampling_workspace_;
  Buffer sampling_buffer(std::string key, std::size_t bytes);
  SampledGradients sample_gradients(Buffer const& gradient, u32 rows, u32 group,
                                    u32 sample_rows, bool prepare, u32 targets = 1, u32 source_groups = 0);
  SampledGradients sample_gradients(Buffer const& gradient, SamplingParams params, bool prepare, bool replay);
  void configure_tree();
  [[nodiscard]] bool has_refresh() const;
  void validate_tree_updaters() const;
  void modify_tree(Tree& tree, Matrix const& data, Buffer const& gradient, float eta,
                   std::string_view updater, bool compact = false);
  Buffer refresh_buffer(std::string key, std::size_t bytes);
  Buffer linear_model_, linear_mass_, linear_previous_;
  std::map<std::string, Buffer, std::less<>> linear_workspace_;
  std::weak_ptr<Matrix const> linear_matrix_;
  std::uint64_t linear_generation_{0};
  std::vector<u32> linear_order_;
  bool linear_configuration_dirty_{true}, linear_converged_{false};
  void configure_linear();
  void upload_linear();
  void update_linear(Matrix const& data, Buffer const& gradient);
  Prediction predict_linear(Matrix const& data, Json const& config);
  [[nodiscard]] Json linear_config() const;
  [[nodiscard]] std::vector<std::string> dump_linear(std::string_view format);
  Buffer linear_buffer(std::string key, std::size_t bytes);
  Buffer forest_nodes_, forest_trees_, forest_categories_, forest_weights_, forest_leaf_weights_, base_buffer_;
  std::vector<float> tree_weights_;
  std::vector<std::size_t> dropped_trees_;
  struct DropoutParams {
    float rate, skip;
    bool one, weighted, forest;
    [[nodiscard]] bool active() const { return rate != 0 || skip != 0 || one; }
  };
  [[nodiscard]] DropoutParams dropout_parameters() const;
  [[nodiscard]] Json dropout_config() const;
  std::vector<float> drop_trees();
  void normalize_trees(std::size_t added);
  Buffer shap_paths_, shap_leaves_, shap_categories_, node_means_;
  std::size_t shap_size_{std::numeric_limits<std::size_t>::max()};
  std::size_t shap_path_count_{0}, shap_category_count_{0}, shap_mean_count_{0};
  u32 shap_leaf_count_{0};
  std::size_t forest_size_{std::numeric_limits<std::size_t>::max()};
  std::size_t forest_node_count_{0}, forest_category_count_{0}, forest_leaf_count_{0};
  std::weak_ptr<Matrix const> validated_matrix_;
  std::uint64_t validated_generation_{0};
  u32 validated_objective_{0}, validated_classes_{0};
  std::weak_ptr<Matrix const> metric_validated_matrix_;
  std::uint64_t metric_validated_generation_{0};
  u32 metric_validated_groups_{0}, metric_validated_mask_{0};
  struct DeferredPrediction {
    // Retain the exact tail-prediction inputs. No command is submitted until a
    // consumer either combines it with gradients or flushes it synchronously.
    std::array<Buffer, 9> inputs;
    Buffer output;
    PredictParams params;
    u32 rounds;
    Buffer csr_col_idx{}, csr_val{};
  };
  struct PredictionCache {
    std::weak_ptr<Matrix const> matrix;
    std::thread::id thread;
    Buffer margins;
    u32 rounds{0};
    std::optional<DeferredPrediction> deferred;
  };
  std::vector<PredictionCache> prediction_caches_;
  PredictionCache& prediction_cache(Matrix const& data);
  Buffer encode_deferred_prediction(PredictionCache& cache, Command& command);
  void finish_deferred_prediction(PredictionCache& cache);
  void flush_prediction_caches();
  bool configuration_dirty_{false}, base_score_changed_{false};
  std::mt19937 rng_{0};
  std::string rng_seed_{"0"};
  void configure_rng();
  [[nodiscard]] u32 random_seed() const;
  struct RankingCache {
    std::weak_ptr<Matrix const> matrix;
    std::thread::id thread;
    Json parameters;
    std::vector<u32> offsets, tile_offsets;
    u32 max_size{}, tiles{}, pair_tiles{}, pair_method{}, pair_samples{};
    std::size_t pairs{};
    Buffer query_ptr, row_query, row_tiles, tile_ptr, query_weights, unit_weights, weight_norm;
    Buffer label_order, discount, inverse_idcg;
    u32 idcg_topk{}, idcg_exp_gain{2};
    Buffer pair_ptr, pair_work, pair_tile_ptr;
    mutable std::map<std::string, Buffer, std::less<>> workspace;
  };
  std::unique_ptr<RankingCache> ranking_objective_cache_;
  std::map<std::string, std::vector<RankingCache>, std::less<>> ranking_metric_caches_;
  Buffer ranking_bias_plus_, ranking_bias_minus_;
  bool ranking_bias_work_initialized_{false};
  std::map<std::string, Json, std::less<>> ranking_metric_configs_;
  [[nodiscard]] RankingParams ranking_parameters(u32 rows) const;
  [[nodiscard]] Json ranking_metric_config(std::string const& name) const;
  void load_ranking_state(Json const& objective, bool new_objective = true);
  RankingCache make_ranking_cache(Matrix const& data, Json parameters, u32 label_validation);
  RankingCache& ranking_cache(Matrix const& data, Json parameters, std::string const& metric, u32 label_validation);
  void ranking_weights(Matrix const& data, RankingCache& cache, bool allow_extra = false);
  Buffer ranking_sort(Command& command, RankingCache const& cache, Buffer const& primary,
                      Buffer const& secondary, u32 mode, std::string_view purpose = "score");
  Buffer ranking_buffer(RankingCache const& cache, std::string key, std::size_t bytes) const;
  Buffer ranking_scan(Command& command, RankingCache const& cache, Buffer const& input,
                      Buffer const& queries, u32 rows, u32 level = 0);
  void ranking_idcg(Command& command, Matrix const& data, RankingCache& cache, RankingParams const& p);
  void ranking_gradients(Command& command, Matrix const& data, Buffer const& prediction,
                         Buffer const& gradient, GradientParams const& params, int iteration);
  double ranking_metric(Matrix const& data, Buffer const& prediction, std::string const& name);
  struct SurvivalOrder {
    std::weak_ptr<Matrix const> matrix;
    std::uint64_t generation;
    Buffer indices;
  };
  std::vector<SurvivalOrder> survival_orders_;
  void upload_model();
  void upload_shap();
  [[nodiscard]] std::vector<float> score_margins(std::vector<float> const& scores) const;
  std::vector<float> estimate_quantiles(Matrix const& data);
  std::vector<float> estimate_distribution(Matrix const& data);
  void smooth_gradients(Command& command, Matrix const& data, Buffer const& prediction,
                        Buffer const& gradient, GradientParams const& params);
  void distribution_gradients(Command& command, Matrix const& data, Buffer const& prediction,
                              Buffer const& gradient, GradientParams const& params);
  [[nodiscard]] std::string default_metric() const;
  [[nodiscard]] SurvivalParams aft_parameters(u32 rows) const;
  Buffer survival_order(Command& command, Matrix const& data);
  Buffer survival_scan(Command& command, Buffer const& input, u32 rows);
  void survival_gradients(Command& command, Matrix const& data, Buffer const& prediction,
                          Buffer const& gradient, GradientParams const& params);
  double survival_metric(Matrix const& data, Buffer const& prediction, u32 type);
  double metric(Matrix const& data, Buffer const& prediction, std::string const& name);
  Tree grow(Matrix& data, Buffer const& gradients, Buffer const& scales, u32 group, float eta,
            bool compact = false, u32 targets = 1, u32 outputs = 1, Command* root_command = nullptr,
            GpuTreeRound* async_round = nullptr);
  void update_vector_leaves(Tree& tree, Matrix const& data, Buffer const& gradient, Buffer const& scales, float eta);
};
}  // namespace metalgb
