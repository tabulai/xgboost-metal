#include "metalgb/c_api.h"
#include "metalgb/engine.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <unistd.h>

using namespace metalgb;
namespace {
using MatrixHandle = std::shared_ptr<Matrix>;
thread_local std::string last_error;
thread_local void (*log_callback)(char const*) = nullptr;
thread_local Json global_config{{"verbosity", 1}, {"use_rmm", false}, {"nthread", 0}};
struct Results {
  std::string text, text2;
  std::vector<std::uint8_t> bytes;
  std::vector<std::string> strings;
  std::vector<char const*> pointers;
  std::vector<float> floats;
  std::vector<bst_ulong> shape;
  Buffer prediction;
};
thread_local Results result;
MatrixHandle& matrix(DMatrixHandle h) { require(h != nullptr, "Invalid DMatrix handle."); return *static_cast<MatrixHandle*>(h); }
Learner& learner(BoosterHandle h) { require(h != nullptr, "Invalid Booster handle."); return *static_cast<Learner*>(h); }
template<class F> int api(F&& f) noexcept {
  try { f(); return 0; }
  catch (std::exception const& e) { last_error = e.what(); return -1; }
  catch (...) { last_error = "Unknown native error in xgboost-metal."; return -1; }
}
template<class F> int booster_api(BoosterHandle h, F&& f) noexcept {
  return api([&] { auto& b = learner(h); std::lock_guard lock(b.mutex); f(b); });
}
int unsupported(char const* feature) { return api([&] { throw std::invalid_argument(std::string(feature) + " is not implemented by the Metal backend."); }); }
int cuda_error() { return api([] { throw std::invalid_argument("CUDA input is unavailable: this library implements only the Apple Metal backend."); }); }
void string_output(std::vector<std::string> strings, bst_ulong* len, char const*** out) {
  result.strings = std::move(strings); result.pointers.clear();
  for (auto const& s : result.strings) result.pointers.push_back(s.c_str());
  *len = result.pointers.size(); *out = result.pointers.data();
}
std::vector<std::string>& feature_info(Matrix& m, std::string_view field) {
  require(field == "feature_name" || field == "feature_type", "Unknown feature info field.");
  return field == "feature_name" ? m.feature_names : m.feature_types;
}
std::vector<std::string>& feature_info(Learner& b, std::string_view field) {
  require(field == "feature_name" || field == "feature_type", "Unknown feature info field.");
  return field == "feature_name" ? b.feature_names : b.feature_types;
}
void set_feature_info(std::vector<std::string>& dest, char const** source, bst_ulong len, u32 columns, bool types) {
  require(len == 0 || len == columns, "Feature metadata length must match the number of features.");
  dest.clear();
  for (bst_ulong i = 0; i < len; ++i) {
    require(source && source[i], "Null feature metadata.");
    std::string value = source[i];
    if (types) require(value == "q" || value == "c" || value == "int" || value == "float" || value == "i", "Invalid feature type.");
    dest.push_back(std::move(value));
  }
}
void check_matrix_config(Json const& config) { require(config.value("data_split_mode", 0) == 0, "Only row-wise data split is supported."); }
int sparse_mode_from_config(Json const& config) {
  if (config.contains("sparse_mode") && !config["sparse_mode"].is_null()) return config["sparse_mode"].get<int>();
  if (char const* env = std::getenv("XGBOOST_METAL_SPARSE_MODE"); env && *env) return std::atoi(env);
  return -1;
}
Json decode(void const* data, std::size_t size) {
  require(data && size >= 2, "Invalid empty model buffer.");
  auto p = static_cast<std::uint8_t const*>(data);
  if (p[0] == '{' && (p[1] == '"' || p[1] == '\n' || p[1] == ' ')) return parse_json(std::string_view(reinterpret_cast<char const*>(p), size), true);
  return Json::from_ubjson(p, p + size);
}
void ubj_integer(std::vector<std::uint8_t>& out, std::uint64_t value, unsigned bytes = 8) {
  for (unsigned i = bytes; i-- > 0;) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}
void ubj_string(std::vector<std::uint8_t>& out, std::string const& value) {
  out.push_back('L'); ubj_integer(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}
void ubj_write(Json const& value, std::vector<std::uint8_t>& out) {
  // XGBoost's UBJSON reader requires int64 lengths even for short strings.
  // Model floats must use the float32 marker, not the float64 JSON default.
  if (value.is_object()) {
    out.push_back('{');
    for (auto const& [key, child] : value.items()) { ubj_string(out, key); ubj_write(child, out); }
    out.push_back('}');
  } else if (value.is_array()) {
    out.push_back('['); out.push_back('#'); out.push_back('L'); ubj_integer(out, value.size());
    for (auto const& child : value) ubj_write(child, out);
  } else if (value.is_string()) {
    out.push_back('S'); ubj_string(out, value.get<std::string>());
  } else if (value.is_boolean()) out.push_back(value.get<bool>() ? 'T' : 'F');
  else if (value.is_null()) out.push_back('Z');
  else if (value.is_number_float()) {
    out.push_back('d'); ubj_integer(out, std::bit_cast<std::uint32_t>(value.get<float>()), 4);
  } else {
    out.push_back('L'); ubj_integer(out, value.get<std::int64_t>());
  }
}
void encode(Json const& model, std::string const& format) {
  require(format == "json" || format == "ubj", "Model format must be json or ubj.");
  if (format == "json") {
    auto text = dump_json(model);
    result.bytes.assign(text.begin(), text.end());
  }
  else { result.bytes.clear(); ubj_write(model, result.bytes); }
}
void predict_output(Learner& b, Matrix& m, Json const& config, bst_ulong const** shape, bst_ulong* ndim, float const** values) {
  auto output = b.predict(m, config); result.prediction = output.values;
  result.shape.assign(output.shape.begin(), output.shape.end());
  *shape = result.shape.data(); *ndim = result.shape.size(); *values = result.prediction.as<float>();
}
void copy_proxy_metadata(Matrix& m, DMatrixHandle proxy) {
  if (proxy) {
    auto const& p = *matrix(proxy); m.info = p.info; m.label_cols = p.label_cols; m.group_ptr = p.group_ptr; m.refresh_metadata();
    m.label_shape = p.label_shape; m.margin_shape = p.margin_shape;
  }
}
void prepare_parameters(Learner& b, std::string_view next_booster, std::string_view next_objective) {
  b.prepare_objective(next_objective);
  bool next_linear = next_booster == "gblinear";
  if (b.is_linear() != next_linear) {
    for (auto key : {"updater", "updater_seq", "process_type", "tolerance", "max_row_perbatch"}) b.params.erase(key);
  }
  if (b.is_linear() || next_linear) b.clear_linear_parameters();
  if (!next_linear) b.clear_tree_updater();
}
void replace_proxy(DMatrixHandle h, std::shared_ptr<Matrix> value) {
  auto old = matrix(h); value->info = old->info; value->label_cols = old->label_cols;
  value->label_shape = old->label_shape; value->margin_shape = old->margin_shape;
  value->is_simple = old->is_simple;
  value->feature_names = old->feature_names; value->feature_types = old->feature_types;
  value->group_ptr = old->group_ptr; value->refresh_metadata(); matrix(h) = std::move(value);
}
std::shared_ptr<Matrix> from_iterator(DataIterHandle iter, DMatrixHandle proxy, DataIterResetCallback* reset,
                                     XGDMatrixCallbackNext* next, Json const& config) {
  require(reset && next && proxy, "Iterator callbacks and proxy must be provided.");
  auto combined = std::make_shared<Matrix>();
  combined->is_simple = false;
  reset(iter);
  while (true) {
    int status = next(iter);
    require(status >= 0, "Data iterator callback failed.");
    if (!status) break;
    auto batch = matrix(proxy);
    batch->filter_missing(missing_value(config));
    combined->append(*batch);
  }
  reset(iter);
  require(combined->cols > 0, "The iterator produced no feature data.");
  return combined;
}
void category_output(Json const& categories, CategoriesHandle* out, char const** exported = nullptr) {
  bool any = std::any_of(categories.begin(), categories.end(), [](Json const& v) { return !v.is_null(); });
  *out = nullptr;
  if (exported) *exported = nullptr;
  if (!any) return;
  auto owned = std::make_unique<CategoryStorage>(categories);
  if (exported) { result.text = owned->array_interfaces().dump(); *exported = result.text.c_str(); }
  *out = owned.release();
}
}

void metalgb::warning(std::string_view message) {
  if (global_config.value("verbosity", 1) == 0) return;
  auto text = std::string("WARNING: ") + std::string(message);
  if (log_callback) log_callback(text.c_str());
  else std::fprintf(stderr, "%s\n", text.c_str());
}
void XGBoostVersion(int* major, int* minor, int* patch) { if (major) *major = 3; if (minor) *minor = 5; if (patch) *patch = 0; }
char const* XGBGetLastError() { return last_error.c_str(); }
int XGBRegisterLogCallback(void (*callback)(char const*)) { log_callback = callback; return 0; }
int XGBuildInfo(char const** out) { return api([&] { result.text = Runtime::get().info(); *out = result.text.c_str(); }); }
int XGBSetGlobalConfig(char const* text) {
  return api([&] {
    auto config = parse_json(text);
    for (auto const& [key, value] : config.items()) {
      require(key == "verbosity" || key == "use_rmm" || key == "nthread", "Unknown global parameter: " + key);
      if (key == "use_rmm") require(!value.get<bool>(), "RMM is a CUDA allocator; it is unavailable on Metal.");
      if (key == "verbosity") require(value.get<int>() >= 0 && value.get<int>() <= 3, "verbosity must be in [0, 3].");
      global_config[key] = value;
    }
  });
}
int XGBGetGlobalConfig(char const** out) { return api([&] { result.text = global_config.dump(); *out = result.text.c_str(); }); }

int XGDMatrixCreateFromDense(char const* data, char const* config, DMatrixHandle* out) {
  return api([&] { auto c = parse_json(config); check_matrix_config(c); auto m = Matrix::dense(parse_json(data), missing_value(c)); *out = new MatrixHandle(std::move(m)); });
}
int XGDMatrixCreateFromColumnar(char const* data, char const* config, DMatrixHandle* out) {
  return api([&] { auto c = parse_json(config); check_matrix_config(c); *out = new MatrixHandle(Matrix::columnar(parse_json(data), missing_value(c))); });
}
int XGDMatrixCreateFromCSR(char const* indptr, char const* indices, char const* data, bst_ulong columns, char const* config, DMatrixHandle* out) {
  return api([&] { auto c = parse_json(config); check_matrix_config(c); *out = new MatrixHandle(Matrix::sparse(parse_json(indptr), parse_json(indices), parse_json(data), columns, false, missing_value(c), sparse_mode_from_config(c))); });
}
int XGDMatrixCreateFromCSC(char const* indptr, char const* indices, char const* data, bst_ulong rows, char const* config, DMatrixHandle* out) {
  return api([&] { auto c = parse_json(config); check_matrix_config(c); *out = new MatrixHandle(Matrix::sparse(parse_json(indptr), parse_json(indices), parse_json(data), rows, true, missing_value(c), sparse_mode_from_config(c))); });
}
int XGDMatrixCreateFromCOO(char const* row_idx, char const* col_idx, char const* data, bst_ulong rows, bst_ulong cols, char const* config, DMatrixHandle* out) {
  return api([&] { auto c = parse_json(config); check_matrix_config(c); *out = new MatrixHandle(Matrix::from_coo(parse_json(row_idx), parse_json(col_idx), parse_json(data), rows, cols, missing_value(c), sparse_mode_from_config(c))); });
}
int XGDMatrixCreateFromMat(float const* data, bst_ulong rows, bst_ulong cols, float missing, DMatrixHandle* out) {
  return api([&] { require(rows <= UINT32_MAX && cols <= UINT32_MAX, "Matrix dimensions exceed the index range."); auto n = checked_product(rows, cols); require(data || n == 0, "Null input data."); std::vector<float> v(n); if (n) std::copy_n(data, n, v.data()); *out = new MatrixHandle(std::make_shared<Matrix>(rows, cols, std::move(v), missing)); });
}
int XGDMatrixCreateFromMat_omp(float const* data, bst_ulong rows, bst_ulong cols, float missing, DMatrixHandle* out, int nthread) { return XGDMatrixCreateFromMat(data, rows, cols, missing, out); }
int XGDMatrixCreateFromCudaArrayInterface(char const*, char const*, DMatrixHandle*) { return cuda_error(); }
int XGDMatrixCreateFromCudaColumnar(char const*, char const*, DMatrixHandle*) { return cuda_error(); }
int XGDMatrixFree(DMatrixHandle h) { return api([&] { delete static_cast<MatrixHandle*>(h); }); }
int XGProxyDMatrixCreate(DMatrixHandle* out) { return api([&] { auto m = std::make_shared<Matrix>(); m->is_simple = false; *out = new MatrixHandle(std::move(m)); }); }
int XGProxyDMatrixSetDataDense(DMatrixHandle h, char const* data) { return api([&] { replace_proxy(h, Matrix::dense(parse_json(data), kNaN)); }); }
int XGProxyDMatrixSetDataColumnar(DMatrixHandle h, char const* data) { return api([&] { replace_proxy(h, Matrix::columnar(parse_json(data), kNaN)); }); }
int XGProxyDMatrixSetDataCSR(DMatrixHandle h, char const* ip, char const* ix, char const* values, bst_ulong columns) {
  return api([&] { replace_proxy(h, Matrix::sparse(parse_json(ip), parse_json(ix), parse_json(values), columns, false, kNaN, sparse_mode_from_config(Json::object()))); });
}
int XGProxyDMatrixSetDataCudaArrayInterface(DMatrixHandle, char const*) { return cuda_error(); }
int XGProxyDMatrixSetDataCudaColumnar(DMatrixHandle, char const*) { return cuda_error(); }
int XGDMatrixCreateFromCallback(DataIterHandle iter, DMatrixHandle proxy, DataIterResetCallback* reset, XGDMatrixCallbackNext* next, char const* config, DMatrixHandle* out) {
  return api([&] { *out = new MatrixHandle(from_iterator(iter, proxy, reset, next, parse_json(config))); });
}
int XGQuantileDMatrixCreateFromCallback(DataIterHandle iter, DMatrixHandle proxy, DataIterHandle ref, DataIterResetCallback* reset, XGDMatrixCallbackNext* next, char const* config, DMatrixHandle* out) {
  return api([&] {
    auto c = parse_json(config); auto m = from_iterator(iter, proxy, reset, next, c);
    u32 bins = 256;
    if (c.contains("max_bin") && !c["max_bin"].is_null()) {
      require(c["max_bin"].is_number_integer(), "max_bin must be an integer.");
      auto value = c["max_bin"].get<std::int64_t>();
      require(value >= 2 && value <= INT32_MAX, "max_bin must be in [2, 2147483647].");
      bins = static_cast<u32>(value);
    }
    auto reference = ref ? matrix(ref) : nullptr;
    if (reference && !reference->max_bin) reference->quantize(bins);
    m->quantize(bins, reference.get()); m->finish_quantile(); *out = new MatrixHandle(std::move(m));
  });
}
int XGExtMemQuantileDMatrixCreateFromCallback(DataIterHandle, DMatrixHandle, DataIterHandle, DataIterResetCallback*, XGDMatrixCallbackNext*, char const*, DMatrixHandle*) { return unsupported("Paged external-memory quantile training"); }
int XGDMatrixCreateFromDataIter(DataIterHandle, XGBCallbackDataIterNext*, char const*, float, DMatrixHandle*) { return unsupported("The legacy JVM data iterator"); }
int XGDMatrixSetInfoFromInterface(DMatrixHandle h, char const* field, char const* json) { return api([&] { matrix(h)->set_info(field, Array(parse_json(json))); }); }
int XGDMatrixSetFloatInfo(DMatrixHandle h, char const* field, float const* values, bst_ulong len) {
  return api([&] { require(values || !len, "Null metadata input."); std::vector<float> v(len); if (len) std::copy_n(values, len, v.data()); matrix(h)->set_info(field, std::move(v)); });
}
int XGDMatrixSetUIntInfo(DMatrixHandle h, char const* field, unsigned const* values, bst_ulong len) {
  return api([&] { matrix(h)->set_info(field, Array(array_interface(values, {static_cast<std::size_t>(len)}, "<u4"))); });
}
int XGDMatrixSetDenseInfo(DMatrixHandle h, char const* field, void const* values, bst_ulong size, int type) {
  return api([&] {
    static char const* const types[]{"<f4", "<f8", "<u4", "<u8"};
    require(type >= 1 && type <= 4, "Invalid metadata type."); matrix(h)->set_info(field, Array(array_interface(values, {static_cast<std::size_t>(size)}, types[type - 1])));
  });
}
int XGDMatrixGetInfoRef(DMatrixHandle h, char const* field, char const** out) {
  return api([&] {
    auto const& m = *matrix(h);
    if (std::string_view(field) == "group_ptr") result.text = array_interface(m.group_ptr.data(), {m.group_ptr.size()}, "<u4").dump();
    else {
      auto const& values = m.get_info(field); std::vector<std::size_t> shape{values.size()};
      if (std::string_view(field) == "label" || std::string_view(field) == "base_margin") {
        auto tensor = std::string_view(field) == "label" ? m.label_shape : m.margin_shape;
        if (tensor[1] > 1) shape.assign(tensor.begin(), tensor.end());
      }
      result.text = array_interface(values.data(), shape).dump();
    }
    *out = result.text.c_str();
  });
}
int XGDMatrixGetFloatInfo(DMatrixHandle h, char const* field, bst_ulong* len, float const** out) { return api([&] { auto const& v = matrix(h)->get_info(field); *len = v.size(); *out = v.data(); }); }
int XGDMatrixGetUIntInfo(DMatrixHandle h, char const* field, bst_ulong* len, unsigned const** out) {
  return api([&] { require(std::string_view(field) == "group_ptr", "Unknown integer metadata field."); auto const& v = matrix(h)->group_ptr; *len = v.size(); *out = v.data(); });
}
int XGDMatrixNumRow(DMatrixHandle h, bst_ulong* out) { return api([&] { *out = matrix(h)->rows; }); }
int XGDMatrixNumCol(DMatrixHandle h, bst_ulong* out) { return api([&] { *out = matrix(h)->cols; }); }
int XGDMatrixNumNonMissing(DMatrixHandle h, bst_ulong* out) {
  return api([&] {
    auto const& m = *matrix(h);
    if (m.sparse_page && !m.is_quantile) { *out = m.sparse_page->entries.size(); return; }
    if (m.is_sparse()) {
      if (!m.is_quantile) { *out = m.sparse_->nnz; return; }
      auto const* vals = m.sparse_->csr_val.as<float>();
      bst_ulong count = 0;
      for (u32 e = 0; e < m.sparse_->nnz; ++e) count += !std::isnan(vals[e]);
      *out = count;
      return;
    }
    auto input = m.data.as<float>(); bst_ulong count = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(m.rows) * m.cols; ++i) count += !std::isnan(input[i]);
    *out = count;
  });
}
int XGDMatrixDataSplitMode(DMatrixHandle h, bst_ulong* out) { return api([&] { matrix(h); *out = 0; }); }
int XGDMatrixSliceDMatrixEx(DMatrixHandle h, int const* indices, bst_ulong len, DMatrixHandle* out, int allow_groups) {
  return api([&] { require(indices || !len, "Null slice indices."); *out = new MatrixHandle(matrix(h)->slice({indices, static_cast<std::size_t>(len)}, allow_groups)); });
}
int XGDMatrixSliceDMatrix(DMatrixHandle h, int const* indices, bst_ulong len, DMatrixHandle* out) { return XGDMatrixSliceDMatrixEx(h, indices, len, out, 0); }
int XGDMatrixSetStrFeatureInfo(DMatrixHandle h, char const* field, char const** values, bst_ulong size) {
  return api([&] {
    auto& m = *matrix(h); bool type = std::string_view(field) == "feature_type";
    auto previous = feature_info(m, field);
    set_feature_info(feature_info(m, field), values, size, m.cols, type);
    if (previous != feature_info(m, field)) ++m.generation;
    if (type) m.update_feature_types();
  });
}
int XGDMatrixGetStrFeatureInfo(DMatrixHandle h, char const* field, bst_ulong* len, char const*** out) { return api([&] { string_output(feature_info(*matrix(h), field), len, out); }); }
int XGDMatrixGetCategories(DMatrixHandle h, char const*, CategoriesHandle* out) { return api([&] { category_output(matrix(h)->categories, out); }); }
int XGDMatrixGetCategoriesExportToArrow(DMatrixHandle h, char const*, CategoriesHandle* out, char const** exported) {
  return api([&] { category_output(matrix(h)->categories, out, exported); });
}
int XGBCategoriesFree(CategoriesHandle h) { return api([&] { delete static_cast<CategoryStorage*>(h); }); }
int XGDMatrixGetDataAsCSR(DMatrixHandle h, char const*, bst_ulong* indptr, unsigned* indices, float* values) {
  return api([&] {
    auto const& m = *matrix(h);
    require(m.is_simple || m.is_quantile, "Can not obtain a single CSR page for external memory DMatrix");
    if (m.sparse_page && !m.is_quantile) {
      std::copy(m.sparse_page->offsets.begin(), m.sparse_page->offsets.end(), indptr);
      for (std::size_t i = 0; i < m.sparse_page->entries.size(); ++i) {
        auto entry = m.sparse_page->entries[i]; indices[i] = entry.feature; values[i] = entry.value;
      }
      return;
    }
    if (m.is_sparse()) {
      auto const* rp = m.sparse_->csr_row_ptr.as<u32>();
      auto const* ci = m.sparse_->csr_col_idx.as<u32>();
      if (!m.is_quantile) {
        auto const* cv = m.sparse_->csr_val.as<float>();
        for (u32 r = 0; r <= m.rows; ++r) indptr[r] = rp[r];
        std::copy_n(ci, m.sparse_->nnz, indices);
        std::copy_n(cv, m.sparse_->nnz, values);
        return;
      }
      Buffer rep_val(checked_product(m.sparse_->nnz, sizeof(float)));
      m.decode_quantiles(rep_val, true);
      auto const* rv = rep_val.as<float>();
      bst_ulong count = 0;
      indptr[0] = 0;
      for (u32 r = 0; r < m.rows; ++r) {
        for (u32 e = rp[r]; e < rp[r + 1]; ++e) {
          if (!std::isnan(rv[e])) { indices[count] = ci[e]; values[count++] = rv[e]; }
        }
        indptr[r + 1] = count;
      }
      return;
    }
    auto data = m.export_data(); bst_ulong count = 0; indptr[0] = 0;
    auto input = data.as<float>();
    for (u32 r = 0; r < m.rows; ++r) {
      for (u32 c = 0; c < m.cols; ++c) {
        float x = input[static_cast<std::size_t>(r) * m.cols + c];
        if (!std::isnan(x)) { indices[count] = c; values[count++] = x; }
      }
      indptr[r + 1] = count;
    }
  });
}
int XGDMatrixGetQuantileCut(DMatrixHandle h, char const*, char const** indptr, char const** values) {
  return api([&] { auto& m = *matrix(h); require(m.max_bin != 0, "Quantile cuts are not available before quantization."); result.text = array_interface(m.cut_ptr.data(), {m.cut_ptr.size()}, "<u8").dump(); result.text2 = array_interface(m.cut_values.data(), {m.cut_values.size()}).dump(); *indptr = result.text.c_str(); *values = result.text2.c_str(); });
}
int XGDMatrixSaveBinary(DMatrixHandle h, char const* filename, int) {
  return api([&] { require(filename != nullptr, "Null binary filename."); matrix(h)->save_binary(filename); });
}
int XGDMatrixCreateFromURI(char const* config, DMatrixHandle* out) {
  return api([&] {
    auto c = parse_json(config); auto uri = c.at("uri").get<std::string>();
    check_matrix_config(c);
    require(uri.find('#') == std::string::npos, "External memory training with text input has been removed.");
    if (auto binary = Matrix::load_binary(uri)) {
      if (!c.value("silent", 1) && global_config.value("verbosity", 1) >= 2) {
        std::size_t entry_count = binary->sparse_page ? binary->sparse_page->entries.size() :
                                  (binary->is_sparse() ? binary->sparse_->nnz : 0);
        auto text = std::to_string(binary->rows) + "x" + std::to_string(binary->cols) + " matrix with " +
                    std::to_string(entry_count) + " entries loaded from " + uri;
        if (log_callback) log_callback(text.c_str()); else std::fprintf(stderr, "%s\n", text.c_str());
      }
      *out = new MatrixHandle(std::move(binary)); return;
    }
    auto question = uri.find('?'); require(question != std::string::npos, "URI requires an explicit format, such as ?format=libsvm.");
    auto path = uri.substr(0, question); auto options = uri.substr(question + 1);
    std::ifstream file(path); require(file.good(), "Cannot open data file: " + path);
    bool csv = options.find("format=csv") != std::string::npos;
    require(csv || options.find("format=libsvm") != std::string::npos, "Unknown data file format.");
    std::vector<float> labels, values; std::vector<std::vector<std::pair<u32, float>>> rows;
    std::vector<std::uint64_t> query_ids;
    std::string line; u32 cols = 0;
    int label_column = -1;
    if (auto pos = options.find("label_column="); pos != std::string::npos) label_column = std::stoi(options.substr(pos + 13));
    while (std::getline(file, line)) {
      if (line.empty()) continue;
      std::vector<std::pair<u32, float>> row;
      std::istringstream input(line);
      if (csv) {
        std::string cell; int index = 0; u32 feature = 0;
        while (std::getline(input, cell, ',')) {
          float value = cell.empty() ? kNaN : std::stof(cell);
          if (index++ == label_column) labels.push_back(value); else row.emplace_back(feature++, value);
        }
        cols = std::max(cols, feature);
      } else {
        float y; require(static_cast<bool>(input >> y), "Invalid libsvm label."); labels.push_back(y);
        std::string token;
        bool has_query = false;
        while (input >> token) {
          if (token.starts_with('#')) break;
          auto pos = token.find(':'); require(pos != std::string::npos, "Invalid libsvm feature.");
          if (token.starts_with("qid:")) {
            require(!has_query && query_ids.size() == rows.size(), "Each libsvm row must have exactly one query id when query ids are used.");
            auto text = token.substr(4);
            require(!text.empty() && text.front() != '-', "Invalid libsvm query id.");
            std::size_t parsed;
            auto query = std::stoull(text, &parsed);
            require(parsed == text.size() && (query_ids.empty() || query >= query_ids.back()), "Libsvm query ids must be sorted nonnegative integers.");
            query_ids.push_back(query); has_query = true;
            continue;
          }
          auto index = static_cast<u32>(std::stoul(token.substr(0, pos))); row.emplace_back(index, std::stof(token.substr(pos + 1))); cols = std::max(cols, index + 1);
        }
      }
      rows.push_back(std::move(row));
    }
    std::vector<std::uint64_t> offsets{0};
    std::vector<u32> indices;
    for (auto const& row : rows) {
      for (auto [col, value] : row) { indices.push_back(col); values.push_back(value); }
      offsets.push_back(values.size());
    }
    auto m = Matrix::sparse(array_interface(offsets.data(), {offsets.size()}, "<u8"),
                            array_interface(indices.data(), {indices.size()}, "<u4"),
                            array_interface(values.data(), {values.size()}), cols, false, kNaN);
    if (!query_ids.empty()) {
      require(query_ids.size() == rows.size(), "Each libsvm row must have a query id.");
      m->group_ptr = {0};
      for (u32 r = 1; r < rows.size(); ++r) if (query_ids[r] != query_ids[r - 1]) m->group_ptr.push_back(r);
      m->group_ptr.push_back(rows.size());
    }
    if (!labels.empty()) m->set_info("label", std::move(labels));
    *out = new MatrixHandle(std::move(m));
  });
}

int XGBoosterCreate(DMatrixHandle const dmats[], bst_ulong len, BoosterHandle* out) {
  return api([&] {
    auto b = std::make_unique<Learner>();
    for (bst_ulong i = 0; i < len; ++i) {
      auto m = matrix(dmats[i]);
      b->num_feature = std::max(b->num_feature, m->cols);
      b->cache_matrices.push_back(m);
    }
    *out = b.release();
  });
}
int XGBoosterFree(BoosterHandle h) { return api([&] { delete static_cast<Learner*>(h); }); }
int XGBoosterReset(BoosterHandle h) {
  return booster_api(h, [](Learner& b) {
    b.configure();
    if (!b.initialized) for (auto const& weak : b.cache_matrices) {
      if (auto data = weak.lock()) { b.initialize(*data); break; }
    }
    require(b.initialized, "Model is not yet initialized (not fitted).");
    b.cache_matrices.clear(); b.reset();
  });
}
int XGBoosterSetParam(BoosterHandle h, char const* key, char const* value) { return booster_api(h, [&](Learner& b) { require(key && value, "Null parameter."); prepare_parameters(b, std::string_view(key) == "booster" ? value : b.str("booster", "gbtree"), std::string_view(key) == "objective" ? value : b.str("objective", "reg:squarederror")); b.set_param(key, value); b.configure(); }); }
int XGBoosterSetParams(BoosterHandle h, char const* config) {
  return booster_api(h, [&](Learner& b) {
    auto c = parse_json(config); auto const& args = c.at("params");
    if (args.empty()) return;
    auto next_booster = b.str("booster", "gbtree");
    auto next_objective = b.str("objective", "reg:squarederror");
    for (auto const& p : args) {
      if (p.at(0) == "booster") next_booster = p.at(1).get<std::string>();
      if (p.at(0) == "objective") next_objective = p.at(1).get<std::string>();
    }
    prepare_parameters(b, next_booster, next_objective);
    for (auto const& p : args) if (p.at(0) != "ndcg_exp_gain")
      b.set_param(p.at(0).get<std::string>(), p.at(1).get<std::string>());
    // Metrics are constructed before Configure receives the same argument list.
    for (auto const& p : args) if (p.at(0) == "ndcg_exp_gain")
      b.set_param(p.at(0).get<std::string>(), p.at(1).get<std::string>());
    b.configure();
  });
}
int XGBoosterUpdateOneIter(BoosterHandle h, int iteration, DMatrixHandle data) { return booster_api(h, [&](Learner& b) { b.update(*matrix(data), iteration); }); }
int XGBoosterTrainOneIter(BoosterHandle h, DMatrixHandle data, int iteration, char const* grad, char const* hess) {
  return booster_api(h, [&](Learner& b) { Array g(parse_json(grad)), hh(parse_json(hess)); b.update(*matrix(data), iteration, &g, &hh); });
}
int XGBoosterTrainOneIterWithSplitGrad(BoosterHandle h, DMatrixHandle data, int iteration,
                                     char const* grad, char const* hess, char const* value_grad, char const* value_hess) {
  return booster_api(h, [&](Learner& b) {
    Array g(parse_json(grad)), hh(parse_json(hess)), vg(parse_json(value_grad)), vh(parse_json(value_hess));
    b.update(*matrix(data), iteration, &g, &hh, &vg, &vh);
  });
}
int XGBoosterBoostOneIter(BoosterHandle h, DMatrixHandle data, float* grad, float* hess, bst_ulong len) {
  return booster_api(h, [&](Learner& b) { Array g(array_interface(grad, {static_cast<std::size_t>(len)})), hh(array_interface(hess, {static_cast<std::size_t>(len)})); b.update(*matrix(data), b.rounds, &g, &hh); });
}
int XGBoosterEvalOneIter(BoosterHandle h, int iteration, DMatrixHandle data[], char const* names[], bst_ulong len, char const** out) {
  return booster_api(h, [&](Learner& b) { std::vector<MatrixHandle> matrices; std::vector<std::string> labels; for (bst_ulong i = 0; i < len; ++i) { matrices.push_back(matrix(data[i])); labels.emplace_back(names[i]); } result.text = b.evaluate(iteration, matrices, labels); *out = result.text.c_str(); });
}
int XGBoosterGetNumFeature(BoosterHandle h, bst_ulong* out) { return booster_api(h, [&](Learner& b) { b.configure(); *out = b.num_feature; }); }
int XGBoosterBoostedRounds(BoosterHandle h, int* out) { return booster_api(h, [&](Learner& b) { b.drain(); b.configure(); *out = b.rounds; }); }
int XGBoosterSlice(BoosterHandle h, int begin, int end, int step, BoosterHandle* out) {
  int status = booster_api(h, [&](Learner& b) { *out = b.slice(begin, end, step).release(); });
  return status && last_error.find("out of range") != std::string::npos ? -2 : status;
}
int XGBoosterPredictFromDMatrix(BoosterHandle h, DMatrixHandle d, char const* config, bst_ulong const** shape, bst_ulong* ndim, float const** values) {
  return booster_api(h, [&](Learner& b) { predict_output(b, *matrix(d), parse_json(config), shape, ndim, values); });
}
int XGBoosterPredictFromDense(BoosterHandle h, char const* input, char const* config, DMatrixHandle proxy, bst_ulong const** shape, bst_ulong* ndim, float const** values) {
  return booster_api(h, [&](Learner& b) { require(!b.is_linear(), "Inplace predict is not supported by the current booster."); auto c = parse_json(config); auto m = Matrix::dense(parse_json(input), missing_value(c)); copy_proxy_metadata(*m, proxy); predict_output(b, *m, c, shape, ndim, values); });
}
int XGBoosterPredictFromColumnar(BoosterHandle h, char const* input, char const* config, DMatrixHandle proxy, bst_ulong const** shape, bst_ulong* ndim, float const** values) {
  return booster_api(h, [&](Learner& b) { require(!b.is_linear(), "Inplace predict is not supported by the current booster."); auto c = parse_json(config); auto m = Matrix::columnar(parse_json(input), missing_value(c)); copy_proxy_metadata(*m, proxy); predict_output(b, *m, c, shape, ndim, values); });
}
int XGBoosterPredictFromCSR(BoosterHandle h, char const* ip, char const* ix, char const* input, bst_ulong columns, char const* config, DMatrixHandle proxy, bst_ulong const** shape, bst_ulong* ndim, float const** values) {
  return booster_api(h, [&](Learner& b) { require(!b.is_linear(), "Inplace predict is not supported by the current booster."); auto c = parse_json(config); auto m = Matrix::sparse(parse_json(ip), parse_json(ix), parse_json(input), columns, false, missing_value(c)); copy_proxy_metadata(*m, proxy); predict_output(b, *m, c, shape, ndim, values); });
}
int XGBoosterPredictFromCudaArray(BoosterHandle, char const*, char const*, DMatrixHandle, bst_ulong const**, bst_ulong*, float const**) { return cuda_error(); }
int XGBoosterPredictFromCudaColumnar(BoosterHandle, char const*, char const*, DMatrixHandle, bst_ulong const**, bst_ulong*, float const**) { return cuda_error(); }
int XGBoosterSaveModelToBuffer(BoosterHandle h, char const* config, bst_ulong* len, char const** out) {
  return booster_api(h, [&](Learner& b) { encode(b.model(), parse_json(config).value("format", "ubj")); *len = result.bytes.size(); *out = reinterpret_cast<char const*>(result.bytes.data()); });
}
int XGBoosterLoadModelFromBuffer(BoosterHandle h, void const* input, bst_ulong len) { return booster_api(h, [&](Learner& b) { b.load_model(decode(input, len)); }); }
int XGBoosterSaveModel(BoosterHandle h, char const* filename) {
  return booster_api(h, [&](Learner& b) {
    std::filesystem::path path(filename); encode(b.model(), path.extension() == ".json" ? "json" : "ubj");
    std::ofstream file(path, std::ios::binary); require(file.good(), "Cannot write model: " + path.string());
    file.write(reinterpret_cast<char const*>(result.bytes.data()), result.bytes.size()); require(file.good(), "Failed to write model.");
  });
}
int XGBoosterLoadModel(BoosterHandle h, char const* filename) {
  return booster_api(h, [&](Learner& b) {
    std::ifstream file(filename, std::ios::binary); require(file.good(), "Cannot read model file.");
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    b.load_model(decode(bytes.data(), bytes.size()));
  });
}
int XGBoosterSaveJsonConfig(BoosterHandle h, bst_ulong* len, char const** out) { return booster_api(h, [&](Learner& b) { result.text = b.config().dump(); *len = result.text.size(); *out = result.text.c_str(); }); }
int XGBoosterLoadJsonConfig(BoosterHandle h, char const* json) { return booster_api(h, [&](Learner& b) { b.load_config(parse_json(json)); }); }
int XGBoosterSerializeToBuffer(BoosterHandle h, bst_ulong* len, char const** out) {
  return booster_api(h, [&](Learner& b) { encode({{"Model", b.model()}, {"Config", b.config()}}, "ubj"); *len = result.bytes.size(); *out = reinterpret_cast<char const*>(result.bytes.data()); });
}
int XGBoosterUnserializeFromBuffer(BoosterHandle h, void const* input, bst_ulong len) {
  return booster_api(h, [&](Learner& b) { auto state = decode(input, len); b.load_model(state.at("Model")); b.load_config(state.at("Config")); });
}
int XGBoosterGetAttr(BoosterHandle h, char const* key, char const** out, int* success) {
  return booster_api(h, [&](Learner& b) { auto it = b.attributes.find(key); *success = it != b.attributes.end(); if (*success) { result.text = it->second; *out = result.text.c_str(); } else *out = nullptr; });
}
int XGBoosterSetAttr(BoosterHandle h, char const* key, char const* value) { return booster_api(h, [&](Learner& b) { if (value) b.attributes[key] = value; else b.attributes.erase(key); }); }
int XGBoosterGetAttrNames(BoosterHandle h, bst_ulong* len, char const*** out) { return booster_api(h, [&](Learner& b) { std::vector<std::string> names; for (auto const& [key, value] : b.attributes) names.push_back(key); string_output(std::move(names), len, out); }); }
int XGBoosterSetStrFeatureInfo(BoosterHandle h, char const* field, char const** values, bst_ulong len) { return booster_api(h, [&](Learner& b) { set_feature_info(feature_info(b, field), values, len, b.num_feature, std::string_view(field) == "feature_type"); }); }
int XGBoosterGetStrFeatureInfo(BoosterHandle h, char const* field, bst_ulong* len, char const*** out) { return booster_api(h, [&](Learner& b) { string_output(feature_info(b, field), len, out); }); }
int XGBoosterGetCategories(BoosterHandle h, char const*, CategoriesHandle* out) { return booster_api(h, [&](Learner& b) { category_output(b.categories, out); }); }
int XGBoosterGetCategoriesExportToArrow(BoosterHandle h, char const*, CategoriesHandle* out, char const** exported) {
  return booster_api(h, [&](Learner& b) { category_output(b.categories, out, exported); });
}
int XGBoosterDumpModelEx(BoosterHandle h, char const* fmap, int stats, char const* format, bst_ulong* len, char const*** out) {
  return booster_api(h, [&](Learner& b) { require(!fmap || !*fmap, "Feature-map files have not been implemented yet."); string_output(b.dump(format, stats), len, out); });
}
int XGBoosterDumpModel(BoosterHandle h, char const* fmap, int stats, bst_ulong* len, char const*** out) { return XGBoosterDumpModelEx(h, fmap, stats, "text", len, out); }
int XGBoosterDumpModelExWithFeatures(BoosterHandle h, int n, char const** names, char const** types, int stats, char const* format, bst_ulong* len, char const*** out) {
  return booster_api(h, [&](Learner& b) {
    auto old_names = b.feature_names, old_types = b.feature_types;
    try { set_feature_info(b.feature_names, names, n, b.num_feature, false); set_feature_info(b.feature_types, types, n, b.num_feature, true); string_output(b.dump(format, stats), len, out); }
    catch (...) { b.feature_names = std::move(old_names); b.feature_types = std::move(old_types); throw; }
    b.feature_names = std::move(old_names); b.feature_types = std::move(old_types);
  });
}
int XGBoosterDumpModelWithFeatures(BoosterHandle h, int n, char const** names, char const** types, int stats, bst_ulong* len, char const*** out) { return XGBoosterDumpModelExWithFeatures(h, n, names, types, stats, "text", len, out); }
int XGBoosterFeatureScore(BoosterHandle h, char const* config, bst_ulong* nfeatures, char const*** names, bst_ulong* ndim, bst_ulong const** shape, float const** scores) {
  return booster_api(h, [&](Learner& b) {
    b.drain();
    b.configure();
    auto c = parse_json(config); auto kind = c.value("importance_type", "weight");
    if (b.is_linear()) {
      require(!b.linear_weights.empty(), "Model is not initialized.");
      require(!c.contains("trees") || c["trees"].empty(), "gblinear doesn't support number of trees for feature importance.");
      require(kind == "weight", "gblinear only has `weight` defined for feature importance.");
      auto features = c.contains("feature_names") && !c["feature_names"].is_null() ? c["feature_names"].get<std::vector<std::string>>() : b.feature_names;
      require(features.empty() || features.size() == b.num_feature, "Feature name length is invalid.");
      if (features.empty()) for (u32 f = 0; f < b.num_feature; ++f) features.push_back("f" + std::to_string(f));
      result.floats.assign(b.linear_weights.begin(), b.linear_weights.end() - b.groups);
      string_output(std::move(features), nfeatures, names);
      result.shape = {b.num_feature}; if (b.groups > 1) result.shape.push_back(b.groups);
      *ndim = result.shape.size(); *shape = result.shape.data(); *scores = result.floats.data();
      return;
    }
    require(kind == "weight" || kind == "gain" || kind == "cover" || kind == "total_gain" || kind == "total_cover", "Unknown importance type.");
    std::vector<double> total(b.num_feature, 0); std::vector<u32> count(b.num_feature, 0);
    for (auto const& tree : b.trees) for (auto const& node : tree.nodes) if (node.left >= 0) { ++count[node.feature]; total[node.feature] += kind == "weight" ? 1 : (kind == "gain" || kind == "total_gain" ? node.gain : node.cover); }
    auto features = c.contains("feature_names") && !c["feature_names"].is_null() ? c["feature_names"].get<std::vector<std::string>>() : b.feature_names;
    require(features.empty() || features.size() == b.num_feature, "Feature name length is invalid.");
    std::vector<std::string> used; result.floats.clear();
    for (u32 f = 0; f < b.num_feature; ++f) if (count[f]) { used.push_back(features.empty() ? "f" + std::to_string(f) : features[f]); result.floats.push_back(total[f] / ((kind == "gain" || kind == "cover") ? count[f] : 1)); }
    string_output(std::move(used), nfeatures, names); result.shape = {*nfeatures}; *ndim = 1; *shape = result.shape.data(); *scores = result.floats.data();
  });
}

int XGCommunicatorInit(char const* config) { return api([&] { auto c = parse_json(config); require(c.empty(), "Distributed communication has not been implemented yet."); }); }
int XGCommunicatorFinalize() { return 0; }
int XGCommunicatorGetRank() { return 0; }
int XGCommunicatorGetWorldSize() { return 1; }
int XGCommunicatorIsDistributed() { return 0; }
int XGCommunicatorPrint(char const* message) { return api([&] { if (log_callback) log_callback(message); }); }
int XGCommunicatorGetProcessorName(char const** name) { return api([&] { char host[256]; require(gethostname(host, sizeof(host)) == 0, "Cannot read hostname."); result.text = host; *name = result.text.c_str(); }); }
int XGCommunicatorBroadcast(void*, size_t, int root) { return api([&] { require(root == 0, "Single-device communicator requires root=0."); }); }
int XGCommunicatorAllreduce(void*, size_t, int, int) { return 0; }
int XGCommunicatorSignalError() { return unsupported("Distributed error recovery"); }
int XGTrackerCreate(char const*, TrackerHandle*) { return unsupported("Distributed trackers"); }
int XGTrackerWorkerArgs(TrackerHandle, char const**) { return unsupported("Distributed trackers"); }
int XGTrackerRun(TrackerHandle, char const*) { return unsupported("Distributed trackers"); }
int XGTrackerWaitFor(TrackerHandle, char const*) { return unsupported("Distributed trackers"); }
int XGTrackerFree(TrackerHandle) { return 0; }

// Optional profiling extension. Existing XGBoost entry points stay unchanged.
extern "C" __attribute__((visibility("default"))) int XGBoosterMetalProfile(BoosterHandle h, char const** out) {
  return booster_api(h, [&](Learner& b) {
    // Timings and latest-tree fields describe retired rounds.
    b.drain();
    auto const& f = b.forest_uploads;
    result.text = Json{{"gpu_seconds", b.gpu_seconds}, {"command_buffers", b.command_count},
      {"tuning", b.active_tuning}, {"device", parse_json(Runtime::get().info())["METAL_DEVICE"]},
      {"forest_uploads", {{"full_uploads", f.full_uploads}, {"append_uploads", f.append_uploads},
        {"trees_uploaded", f.trees_uploaded}, {"bytes_uploaded", f.bytes_uploaded},
        {"buffer_allocations", f.buffer_allocations}, {"bytes_relocated", f.bytes_relocated}}},
      {"split_screening", {{"candidates", b.split_screening.candidates},
        {"certified", b.split_screening.certified}, {"uncertain", b.split_screening.uncertain},
        {"exact", b.split_screening.exact}, {"groups", b.split_screening.groups},
        {"exact_groups", b.split_screening.exact_groups}}}}.dump();
    *out = result.text.c_str();
  });
}
// Optional validation extension: compares GPU leaf scaling for asynchronous
// rounds with the host multiply over binary32 bit patterns [begin, begin + count).
extern "C" __attribute__((visibility("default"))) int XGBMetalValidateLeafScaling(
    float eta, std::uint64_t begin, std::uint64_t count, std::uint64_t* mismatches, std::uint32_t* first_mismatch) {
  return api([&] {
    require(mismatches && first_mismatch, "Null validation output.");
    *first_mismatch = 0;
    *mismatches = validate_leaf_scaling(eta, begin, count, first_mismatch);
  });
}

namespace {
std::size_t descriptor_dtype_bytes(u32 code) {
  switch (code) {
    case kIngestI8: case kIngestU8: case kIngestBool: return 1;
    case kIngestF16: case kIngestBF16: case kIngestI16: case kIngestU16: return 2;
    case kIngestF32: case kIngestI32: case kIngestU32: return 4;
    case kIngestF64: case kIngestI64: case kIngestU64: return 8;
    case kIngestBoolBits: return 0;
    default: throw std::invalid_argument("Unsupported buffer dtype code.");
  }
}

IngestColumnSpec spec_from_descriptor(XGBufferDescriptor const& d, bool require_2d) {
  if (require_2d) require(d.ndim == 2, "Feature data must be two dimensional.");
  else require(d.ndim == 1 || d.ndim == 2, "Buffer descriptor must be 1D or 2D.");
  require(d.shape[0] <= UINT32_MAX && (d.ndim == 1 || d.shape[1] <= UINT32_MAX),
          "Matrix dimension exceeds the 32-bit row or feature index range.");
  IngestColumnSpec spec;
  spec.rows = static_cast<u32>(d.shape[0]);
  spec.cols = static_cast<u32>(d.ndim == 2 ? d.shape[1] : 1);
  spec.dtype_code = d.dtype_code;
  auto elem_bytes = descriptor_dtype_bytes(d.dtype_code);
  spec.data_ptr = reinterpret_cast<std::uintptr_t>(d.data);
  spec.prefer_zero_copy = (d.flags & 2u) != 0;
  if (d.dtype_code == kIngestBoolBits) {
    require(spec.cols == 1, "Bit-packed boolean buffers must be 1D or single-column.");
    spec.data_bit_offset = (d.flags & 4u) ? d.strides[0] : 0;
    spec.stride_r = 0;
    spec.stride_c = 0;
  } else if (d.flags & 1u) {
    spec.stride_c = d.ndim == 2 ? static_cast<i64>(elem_bytes) : 0;
    spec.stride_r = d.ndim == 2 ? static_cast<i64>(checked_product(spec.cols, elem_bytes))
                                : static_cast<i64>(elem_bytes);
  } else {
    spec.stride_r = d.strides[0];
    spec.stride_c = d.ndim == 2 ? d.strides[1] : 0;
  }
  require(d.mask_mode <= kIngestMaskInvalidBytes, "Invalid buffer mask mode.");
  spec.mask_mode = d.mask_mode;
  spec.mask_ptr = reinterpret_cast<std::uintptr_t>(d.mask_data);
  spec.mask_bit_offset = d.mask_bit_offset;
  if (checked_product(spec.rows, spec.cols) > 0) {
    require(spec.data_ptr != 0, "Null data pointer for a nonempty buffer.");
    if (spec.mask_mode != kIngestMaskNone)
      require(spec.mask_ptr != 0, "Null validity mask for a nonempty buffer.");
  }
  return spec;
}

struct DLDeviceABI {
  std::int32_t device_type;
  std::int32_t device_id;
};
struct DLDataTypeABI {
  std::uint8_t code;
  std::uint8_t bits;
  std::uint16_t lanes;
};
struct DLTensorABI {
  void* data;
  DLDeviceABI device;
  std::int32_t ndim;
  DLDataTypeABI dtype;
  std::int64_t* shape;
  std::int64_t* strides;
  std::uint64_t byte_offset;
};
struct DLManagedTensorABI {
  DLTensorABI dl_tensor;
  void* manager_ctx;
  void (*deleter)(DLManagedTensorABI*);
};
struct DLPackVersionABI {
  std::uint32_t major;
  std::uint32_t minor;
};
struct DLManagedTensorVersionedABI {
  DLPackVersionABI version;
  void* manager_ctx;
  void (*deleter)(DLManagedTensorVersionedABI*);
  std::uint64_t flags;
  DLTensorABI dl_tensor;
};

u32 dlpack_dtype_code(DLDataTypeABI dt) {
  require(dt.lanes == 1, "Vectorized DLPack dtypes (lanes != 1) are not supported.");
  switch (dt.code) {
    case 0:  // kDLInt
      if (dt.bits == 8) return kIngestI8;
      if (dt.bits == 16) return kIngestI16;
      if (dt.bits == 32) return kIngestI32;
      if (dt.bits == 64) return kIngestI64;
      break;
    case 1:  // kDLUInt
      if (dt.bits == 8) return kIngestU8;
      if (dt.bits == 16) return kIngestU16;
      if (dt.bits == 32) return kIngestU32;
      if (dt.bits == 64) return kIngestU64;
      break;
    case 2:  // kDLFloat
      if (dt.bits == 16) return kIngestF16;
      if (dt.bits == 32) return kIngestF32;
      if (dt.bits == 64) return kIngestF64;
      break;
    case 4:  // kDLBfloat
      if (dt.bits == 16) return kIngestBF16;
      break;
    case 6:  // kDLBool
      if (dt.bits == 8) return kIngestBool;
      if (dt.bits == 1) return kIngestBoolBits;
      break;
  }
  throw std::invalid_argument("Unsupported DLPack dtype.");
}

struct ArrowSchemaABI {
  char const* format;
  char const* name;
  char const* metadata;
  std::int64_t flags;
  std::int64_t n_children;
  ArrowSchemaABI** children;
  ArrowSchemaABI* dictionary;
  void (*release)(ArrowSchemaABI*);
  void* private_data;
};
struct ArrowArrayABI {
  std::int64_t length;
  std::int64_t null_count;
  std::int64_t offset;
  std::int64_t n_buffers;
  std::int64_t n_children;
  void const** buffers;
  ArrowArrayABI** children;
  ArrowArrayABI* dictionary;
  void (*release)(ArrowArrayABI*);
  void* private_data;
};

IngestColumnSpec arrow_column_to_spec(ArrowSchemaABI const* cs, ArrowArrayABI const* ca,
                                      std::int64_t parent_offset, u32 expected_rows,
                                      std::string& out_ftype) {
  require(cs && ca && cs->format, "Invalid null Arrow column schema or array.");
  require(!cs->dictionary && !ca->dictionary,
          "Dictionary-encoded Arrow columns must use XGDMatrixCreateFromColumnar.");
  require(ca->length >= 0 && static_cast<std::uint64_t>(ca->length) == expected_rows,
          "All Arrow columns must have the same row count.");
  std::string_view fmt(cs->format);
  u32 code = 0;
  if (fmt == "c") { code = kIngestI8; out_ftype = "int"; }
  else if (fmt == "C") { code = kIngestU8; out_ftype = "int"; }
  else if (fmt == "s") { code = kIngestI16; out_ftype = "int"; }
  else if (fmt == "S") { code = kIngestU16; out_ftype = "int"; }
  else if (fmt == "i") { code = kIngestI32; out_ftype = "int"; }
  else if (fmt == "I") { code = kIngestU32; out_ftype = "int"; }
  else if (fmt == "l") { code = kIngestI64; out_ftype = "int"; }
  else if (fmt == "L") { code = kIngestU64; out_ftype = "int"; }
  else if (fmt == "e") { code = kIngestF16; out_ftype = "float"; }
  else if (fmt == "f") { code = kIngestF32; out_ftype = "float"; }
  else if (fmt == "g") { code = kIngestF64; out_ftype = "float"; }
  else if (fmt == "b") { code = kIngestBoolBits; out_ftype = "i"; }
  else throw std::invalid_argument("Unsupported Arrow column format: " + std::string(fmt));

  IngestColumnSpec spec;
  spec.rows = expected_rows;
  spec.cols = 1;
  spec.dtype_code = code;
  if (expected_rows == 0) return spec;

  require(ca->n_buffers >= 2 && ca->buffers != nullptr, "Arrow primitive array requires at least 2 buffers.");
  std::int64_t total_offset = parent_offset + ca->offset;
  require(total_offset >= 0, "Invalid negative Arrow array offset.");

  if (ca->buffers[0] != nullptr && ca->null_count != 0) {
    spec.mask_mode = kIngestMaskValidBits;
    spec.mask_ptr = reinterpret_cast<std::uintptr_t>(ca->buffers[0]);
    spec.mask_bit_offset = total_offset;
  }
  require(ca->buffers[1] != nullptr, "Null Arrow data buffer for nonempty column.");
  if (code == kIngestBoolBits) {
    spec.data_ptr = reinterpret_cast<std::uintptr_t>(ca->buffers[1]);
    spec.data_bit_offset = total_offset;
  } else {
    auto elem_bytes = descriptor_dtype_bytes(code);
    spec.data_ptr = reinterpret_cast<std::uintptr_t>(
        static_cast<std::uint8_t const*>(ca->buffers[1]) + static_cast<std::size_t>(total_offset) * elem_bytes);
    spec.stride_r = static_cast<i64>(elem_bytes);
    spec.stride_c = 0;
  }
  return spec;
}
}  // namespace

int XGDMatrixCreateFromBufferDescriptor(XGBufferDescriptor const* desc, float missing, DMatrixHandle* out) {
  return api([&] {
    require(desc && out, "Null buffer descriptor or output handle.");
    auto spec = spec_from_descriptor(*desc, true);
    *out = new MatrixHandle(Matrix::from_ingest_2d(spec, missing));
  });
}

int XGDMatrixCreateFromColumnarBuffers(XGBufferDescriptor const* columns, bst_ulong ncol, float missing, DMatrixHandle* out) {
  return api([&] {
    require(columns && out && ncol > 0, "Expected at least one feature column.");
    require(ncol <= UINT32_MAX, "Columnar input exceeds dimension limits.");
    std::vector<IngestColumnSpec> specs;
    specs.reserve(static_cast<std::size_t>(ncol));
    u32 nr = 0;
    for (bst_ulong c = 0; c < ncol; ++c) {
      auto spec = spec_from_descriptor(columns[c], false);
      require(spec.cols == 1, "Each column buffer must contain one value per row.");
      if (c == 0) nr = spec.rows;
      else require(spec.rows == nr, "All feature columns must have the same length.");
      specs.push_back(spec);
    }
    *out = new MatrixHandle(Matrix::from_ingest_columns(specs, nr, missing));
  });
}

int XGDMatrixSetInfoFromBufferDescriptor(DMatrixHandle h, char const* field, XGBufferDescriptor const* desc) {
  return api([&] {
    require(field && desc, "Null metadata field or buffer descriptor.");
    auto spec = spec_from_descriptor(*desc, false);
    matrix(h)->set_info(field, Array(spec));
  });
}

int XGDMatrixCreateFromDLPack(void* dl_managed_tensor, int is_versioned, float missing, DMatrixHandle* out) {
  return api([&] {
    require(dl_managed_tensor && out, "Null DLPack tensor or output handle.");
    struct DeleterGuard {
      void* ptr;
      int versioned;
      ~DeleterGuard() {
        if (!ptr) return;
        if (versioned) {
          auto* v = static_cast<DLManagedTensorVersionedABI*>(ptr);
          if (v->deleter) v->deleter(v);
        } else {
          auto* u = static_cast<DLManagedTensorABI*>(ptr);
          if (u->deleter) u->deleter(u);
        }
      }
    } guard{dl_managed_tensor, is_versioned};

    DLTensorABI const& t = is_versioned
        ? static_cast<DLManagedTensorVersionedABI const*>(dl_managed_tensor)->dl_tensor
        : static_cast<DLManagedTensorABI const*>(dl_managed_tensor)->dl_tensor;

    // kDLCPU = 1, kDLCUDAHost = 3, kDLMetal = 8, kDLROCMHost = 13
    require(t.device.device_type == 1 || t.device.device_type == 3 ||
            t.device.device_type == 8 || t.device.device_type == 13,
            "DLPack tensor must reside in CPU or Apple Metal unified memory.");
    require(t.ndim == 2 && t.shape != nullptr, "Feature data must be two dimensional.");
    require(t.shape[0] >= 0 && t.shape[1] >= 0 &&
            static_cast<std::uint64_t>(t.shape[0]) <= UINT32_MAX &&
            static_cast<std::uint64_t>(t.shape[1]) <= UINT32_MAX,
            "DLPack tensor dimensions exceed the 32-bit range.");

    IngestColumnSpec spec;
    spec.rows = static_cast<u32>(t.shape[0]);
    spec.cols = static_cast<u32>(t.shape[1]);
    spec.dtype_code = dlpack_dtype_code(t.dtype);
    spec.prefer_zero_copy = (t.device.device_type == 8);
    if (checked_product(spec.rows, spec.cols) > 0) {
      require(t.data != nullptr, "Null DLPack data pointer for nonempty tensor.");
      auto const* base = static_cast<std::uint8_t const*>(t.data) + t.byte_offset;
      spec.data_ptr = reinterpret_cast<std::uintptr_t>(base);
      if (spec.dtype_code == kIngestBoolBits) {
        require(spec.cols == 1 && t.strides == nullptr, "Bit-packed DLPack boolean tensors must be contiguous single-column.");
      } else {
        auto elem_bytes = static_cast<i64>(descriptor_dtype_bytes(spec.dtype_code));
        if (t.strides != nullptr) {
          spec.stride_r = t.strides[0] * elem_bytes;
          spec.stride_c = t.strides[1] * elem_bytes;
        } else {
          spec.stride_c = elem_bytes;
          spec.stride_r = static_cast<i64>(spec.cols) * elem_bytes;
        }
      }
    }
    *out = new MatrixHandle(Matrix::from_ingest_2d(spec, missing));
  });
}

int XGDMatrixCreateFromArrow(void const* arrow_schema, void const* arrow_array, float missing, DMatrixHandle* out) {
  return api([&] {
    require(arrow_schema && arrow_array && out, "Null Arrow schema, array, or output handle.");
    auto* sch = const_cast<ArrowSchemaABI*>(static_cast<ArrowSchemaABI const*>(arrow_schema));
    auto* arr = const_cast<ArrowArrayABI*>(static_cast<ArrowArrayABI const*>(arrow_array));
    struct ArrowGuard {
      ArrowSchemaABI* s;
      ArrowArrayABI* a;
      ~ArrowGuard() {
        if (a && a->release) a->release(a);
        if (s && s->release) s->release(s);
      }
    } guard{sch, arr};

    require(sch->format != nullptr, "Released or invalid Arrow schema.");
    require(arr->length >= 0 && static_cast<std::uint64_t>(arr->length) <= UINT32_MAX,
            "Arrow array length exceeds 32-bit row limit.");
    u32 nrows = static_cast<u32>(arr->length);
    std::string_view fmt(sch->format);
    std::vector<IngestColumnSpec> specs;
    std::vector<std::string> names, types;

    if (fmt == "+s") {
      require(sch->n_children > 0 && sch->n_children == arr->n_children &&
              sch->children != nullptr && arr->children != nullptr,
              "Arrow RecordBatch/Struct must have at least one column.");
      require(static_cast<std::uint64_t>(sch->n_children) <= UINT32_MAX, "Arrow column count exceeds limit.");
      specs.reserve(static_cast<std::size_t>(sch->n_children));
      names.reserve(static_cast<std::size_t>(sch->n_children));
      types.reserve(static_cast<std::size_t>(sch->n_children));
      bool all_named = true;
      for (std::int64_t c = 0; c < sch->n_children; ++c) {
        std::string ftype;
        specs.push_back(arrow_column_to_spec(sch->children[c], arr->children[c], arr->offset, nrows, ftype));
        types.push_back(std::move(ftype));
        if (sch->children[c]->name && *sch->children[c]->name) names.emplace_back(sch->children[c]->name);
        else all_named = false;
      }
      if (!all_named) names.clear();
    } else {
      std::string ftype;
      specs.push_back(arrow_column_to_spec(sch, arr, 0, nrows, ftype));
      types.push_back(std::move(ftype));
      if (sch->name && *sch->name) names.emplace_back(sch->name);
    }

    auto m = Matrix::from_ingest_columns(specs, nrows, missing);
    if (names.size() == m->cols) m->feature_names = std::move(names);
    if (types.size() == m->cols) {
      m->feature_types = std::move(types);
      m->update_feature_types();
    }
    *out = new MatrixHandle(std::move(m));
  });
}

int XGBoosterPredictFromBufferDescriptor(BoosterHandle h, XGBufferDescriptor const* desc, char const* config,
                                         DMatrixHandle proxy, bst_ulong const** shape, bst_ulong* ndim,
                                         float const** values) {
  return booster_api(h, [&](Learner& b) {
    require(desc && shape && ndim && values, "Null buffer descriptor or prediction output pointer.");
    require(!b.is_linear(), "Inplace predict is not supported by the current booster.");
    auto c = parse_json(config);
    auto spec = spec_from_descriptor(*desc, true);
    auto m = Matrix::from_ingest_2d(spec, missing_value(c));
    copy_proxy_metadata(*m, proxy);
    predict_output(b, *m, c, shape, ndim, values);
  });
}

int XGBoosterPredictToBuffer(BoosterHandle h, DMatrixHandle dmat, char const* config, float* out_buffer,
                             bst_ulong out_capacity, bst_ulong const** out_shape, bst_ulong* out_dim) {
  return booster_api(h, [&](Learner& b) {
    require(out_shape && out_dim, "Null output shape pointers.");
    auto output = b.predict(*matrix(dmat), parse_json(config));
    auto elems = output.values.size() / sizeof(float);
    result.shape.assign(output.shape.begin(), output.shape.end());
    *out_shape = result.shape.data();
    *out_dim = result.shape.size();
    require(elems <= out_capacity, "Output buffer capacity (" + std::to_string(out_capacity) +
            ") is smaller than prediction size (" + std::to_string(elems) + ").");
    if (elems > 0) {
      require(out_buffer != nullptr, "Null output buffer for nonempty prediction.");
      std::memcpy(out_buffer, output.values.as<float>(), elems * sizeof(float));
    }
  });
}

int XGDMatrixBuildCVFolds(DMatrixHandle h, int nfold, unsigned seed, int shuffle, int stratified,
                          DMatrixHandle* out_train, DMatrixHandle* out_test) {
  return api([&] {
    require(out_train && out_test, "Null CV fold output arrays.");
    require(nfold >= 2, "nfold must be at least 2.");
    auto const& m = *matrix(h);
    std::mt19937 rng(seed);
    std::vector<MatrixHandle> train_folds(nfold), test_folds(nfold);

    if (!m.group_ptr.empty() && m.group_ptr.size() > 1) {
      require(!stratified, "Stratified CV is not supported with query groups.");
      std::size_t ngroups = m.group_ptr.size() - 1;
      require(static_cast<std::size_t>(nfold) <= ngroups, "Number of folds cannot exceed number of groups.");
      std::vector<std::size_t> g_idx(ngroups);
      std::iota(g_idx.begin(), g_idx.end(), 0);
      if (shuffle) std::shuffle(g_idx.begin(), g_idx.end(), rng);

      std::vector<int> group_fold(ngroups, 0);
      std::size_t pos = 0;
      for (int k = 0; k < nfold; ++k) {
        std::size_t chunk = ngroups / static_cast<std::size_t>(nfold) +
                            (static_cast<std::size_t>(k) < (ngroups % static_cast<std::size_t>(nfold)) ? 1 : 0);
        for (std::size_t j = 0; j < chunk; ++j) group_fold[g_idx[pos++]] = k;
      }

      auto const& raw_w = m.get_info("weight");
      bool has_group_weights = !raw_w.empty() && raw_w.size() == ngroups;
      for (int k = 0; k < nfold; ++k) {
        std::vector<int> train_rows, test_rows;
        std::vector<u32> train_gp{0}, test_gp{0};
        std::vector<float> train_gw, test_gw;
        for (std::size_t g = 0; g < ngroups; ++g) {
          u32 beg = m.group_ptr[g], end = m.group_ptr[g + 1];
          if (group_fold[g] == k) {
            for (u32 r = beg; r < end; ++r) test_rows.push_back(static_cast<int>(r));
            test_gp.push_back(static_cast<u32>(test_rows.size()));
            if (has_group_weights) test_gw.push_back(raw_w[g]);
          } else {
            for (u32 r = beg; r < end; ++r) train_rows.push_back(static_cast<int>(r));
            train_gp.push_back(static_cast<u32>(train_rows.size()));
            if (has_group_weights) train_gw.push_back(raw_w[g]);
          }
        }
        auto dtr = m.slice(train_rows, true);
        auto dte = m.slice(test_rows, true);
        dtr->group_ptr = std::move(train_gp);
        dte->group_ptr = std::move(test_gp);
        if (has_group_weights) {
          dtr->info["weight"] = std::move(train_gw);
          dte->info["weight"] = std::move(test_gw);
        }
        dtr->refresh_metadata();
        dte->refresh_metadata();
        train_folds[k] = std::move(dtr);
        test_folds[k] = std::move(dte);
      }
    } else {
      u32 nrows = m.rows;
      require(static_cast<u32>(nfold) <= nrows, "Number of folds cannot exceed number of rows.");
      std::vector<int> row_fold(nrows, 0);
      if (stratified) {
        auto const& y = m.get_info("label");
        require(y.size() == nrows, "Stratified CV requires 1D labels matching the number of rows.");
        std::vector<u32> order(nrows);
        std::iota(order.begin(), order.end(), 0);
        if (shuffle) std::shuffle(order.begin(), order.end(), rng);
        std::stable_sort(order.begin(), order.end(), [&](u32 a, u32 b) { return y[a] < y[b]; });
        for (u32 pos = 0; pos < nrows; ++pos) row_fold[order[pos]] = static_cast<int>(pos % static_cast<u32>(nfold));
      } else {
        std::vector<u32> order(nrows);
        std::iota(order.begin(), order.end(), 0);
        if (shuffle) std::shuffle(order.begin(), order.end(), rng);
        u32 pos = 0;
        for (int k = 0; k < nfold; ++k) {
          u32 chunk = nrows / static_cast<u32>(nfold) +
                      (static_cast<u32>(k) < (nrows % static_cast<u32>(nfold)) ? 1u : 0u);
          for (u32 j = 0; j < chunk; ++j) row_fold[order[pos++]] = k;
        }
      }

      for (int k = 0; k < nfold; ++k) {
        std::vector<int> train_rows, test_rows;
        train_rows.reserve(nrows);
        test_rows.reserve(nrows / static_cast<u32>(nfold) + 1);
        for (u32 r = 0; r < nrows; ++r) {
          if (row_fold[r] == k) test_rows.push_back(static_cast<int>(r));
          else train_rows.push_back(static_cast<int>(r));
        }
        train_folds[k] = m.slice(train_rows, false);
        test_folds[k] = m.slice(test_rows, false);
      }
    }

    for (int k = 0; k < nfold; ++k) {
      out_train[k] = new MatrixHandle(std::move(train_folds[k]));
      out_test[k] = new MatrixHandle(std::move(test_folds[k]));
    }
  });
}

int XGDMatrixWasZeroCopied(DMatrixHandle h, int* out) {
  return api([&] {
    require(out != nullptr, "Null output pointer.");
    *out = matrix(h)->was_zero_copied ? 1 : 0;
  });
}
