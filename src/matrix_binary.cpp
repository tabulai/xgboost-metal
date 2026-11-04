#include "metalgb/engine.hpp"
#include "matrix_binary.hpp"
#include <algorithm>

namespace metalgb {
namespace {
// CatContainer's UBJSON keeps each integer array's declared storage width.
// These widths also preserve unsigned names across the signed boundary.
std::vector<char> category_bytes(Json const& categories, std::vector<std::int32_t> const& sorted_idx) {
  auto encoded = serialize_categories(categories);
  // Models sort category names for lookup. A freshly constructed DMatrix keeps
  // this optional search index empty; loaded and sliced matrices retain it.
  encoded["sorted_idx"] = sorted_idx;
  std::vector<char> out;
  auto integer = [&](std::uint64_t value, unsigned bytes) {
    for (unsigned i = bytes; i-- > 0;) out.push_back(static_cast<char>(value >> (i * 8)));
  };
  auto key = [&](std::string_view name) {
    out.push_back('L'); integer(name.size(), 8); out.insert(out.end(), name.begin(), name.end());
  };
  auto typed = [&](Json const& values, char marker, unsigned width) {
    out.insert(out.end(), {'[', '$', marker, '#', 'L'}); integer(values.size(), 8);
    for (auto const& value : values) integer(value.get<std::uint64_t>(), width);
  };
  out.push_back('{'); key("enc"); out.insert(out.end(), {'[', '#', 'L'});
  integer(encoded.at("enc").size(), 8);
  for (auto const& column : encoded.at("enc")) {
    out.push_back('{');
    if (column.contains("offsets")) {
      key("offsets"); typed(column.at("offsets"), 'l', 4);
      key("values"); typed(column.at("values"), 'i', 1);
    } else {
      auto type = column.at("type").get<unsigned>();
      require(type >= 9 && type <= 16, "Invalid binary category dtype.");
      key("type"); out.push_back('i'); integer(type, 1);
      static constexpr char markers[]{'i', 'U', 'I', 'I', 'l', 'l', 'L', 'L'};
      static constexpr unsigned widths[]{1, 1, 2, 2, 4, 4, 8, 8};
      key("values"); typed(column.at("values"), markers[type - 9], widths[type - 9]);
    }
    out.push_back('}');
  }
  key("feature_segments"); typed(encoded.at("feature_segments"), 'l', 4);
  key("sorted_idx"); typed(encoded.at("sorted_idx"), 'l', 4);
  out.push_back('}');
  return out;
}
}  // namespace

std::shared_ptr<Matrix> Matrix::load_binary(std::string const& filename) {
  // As in DMatrix::Load, probe the entire filename before interpreting URI
  // options. A missing file or nonbinary magic leaves text dispatch to the API.
  std::ifstream probe(filename, std::ios::binary);
  std::uint32_t magic{};
  if (!probe.read(reinterpret_cast<char*>(&magic), sizeof(magic)) || magic != 0xffffab01u) return {};
  probe.close();
  auto document = binary::read(filename);
  require(document.rows <= UINT32_MAX && document.columns <= UINT32_MAX,
          "Binary matrix dimensions exceed the 32-bit index range.");
  require(document.labels.shape[1] <= UINT32_MAX && document.margins.shape[1] <= UINT32_MAX,
          "Binary metadata columns exceed the 32-bit index range.");
  require(document.groups.empty() || (document.groups.front() == 0 && document.groups.back() == document.rows &&
          std::is_sorted(document.groups.begin(), document.groups.end())), "Invalid binary query group offsets.");
  require(document.feature_names.empty() || document.feature_names.size() == document.columns,
          "Binary feature names do not match the number of columns.");
  require(document.feature_types.empty() || document.feature_types.size() == document.columns,
          "Binary feature types do not match the number of columns.");
  for (auto const& type : document.feature_types)
    require(type == "q" || type == "c" || type == "int" || type == "float" || type == "i", "Invalid binary feature type.");
  // Column sampling indexes these by feature, as set_info validates them.
  require(document.feature_weights.empty() || document.feature_weights.size() == document.columns,
          "Binary feature weights do not match the number of columns.");
  require(std::all_of(document.feature_weights.begin(), document.feature_weights.end(),
                      [](float value) { return std::isfinite(value) && value >= 0; }),
          "Binary feature weights must be finite and nonnegative.");
  auto encoded = Json::from_ubjson(document.categories);
  require(encoded.at("enc").is_array() && encoded.at("feature_segments").is_array() &&
          encoded.at("sorted_idx").is_array(), "Invalid binary category arrays.");
  auto categories = deserialize_categories(encoded);
  require(categories.empty() || categories.size() == document.columns,
          "Binary category dictionaries do not match the number of columns.");
  if (document.declared_fields > 13) warning("Binary DMatrix contains more metadata fields than this version expects.");

  auto matrix = std::make_shared<Matrix>();
  matrix->rows = document.rows; matrix->cols = document.columns;
  auto elements = checked_product(matrix->rows, matrix->cols);
  auto raw_nnz = document.entries.size();
  if (matrix->rows > 0 && matrix->cols > 0 && raw_nnz <= 0x7fffffffu &&
      elements >= 16384 && matrix->cols >= 16 && raw_nnz < elements / 4) {
    auto storage = std::make_shared<SparseStorage>();
    storage->csr_row_ptr = Buffer((static_cast<std::size_t>(matrix->rows) + 1) * sizeof(u32));
    auto* rp = storage->csr_row_ptr.as<u32>();
    std::vector<u32> cols_vec;
    std::vector<float> vals_vec;
    cols_vec.reserve(raw_nnz);
    vals_vec.reserve(raw_nnz);
    rp[0] = 0;
    for (u32 row = 0; row < matrix->rows; ++row) {
      auto begin = document.offsets[row], end = document.offsets[row + 1];
      for (auto i = begin; i < end; ++i) {
        auto entry = document.entries[i];
        if (std::isnan(entry.value)) continue;
        // Binary CSR entries in a row are sorted by feature; if duplicate features appear, last write wins.
        if (!cols_vec.empty() && cols_vec.size() > rp[row] && cols_vec.back() == entry.feature) {
          vals_vec.back() = entry.value;
        } else {
          cols_vec.push_back(entry.feature);
          vals_vec.push_back(entry.value);
        }
      }
      rp[row + 1] = static_cast<u32>(cols_vec.size());
    }
    storage->nnz = static_cast<u32>(cols_vec.size());
    storage->csr_col_idx = Buffer(cols_vec.data(), cols_vec.size() * sizeof(u32));
    storage->csr_val = Buffer(vals_vec.data(), vals_vec.size() * sizeof(float));
    matrix->data = {};
    matrix->sparse_ = std::move(storage);
    matrix->sparse_page = std::make_shared<SparsePage>(SparsePage{std::move(document.offsets), std::move(document.entries)});
  } else {
    matrix->data = Buffer(checked_product(elements, sizeof(float)));
    std::fill_n(matrix->data.as<float>(), elements, kNaN);
    for (u32 row = 0; row < matrix->rows; ++row) {
      for (auto i = document.offsets[row]; i < document.offsets[row + 1]; ++i) {
        auto entry = document.entries[i];
        if (!std::isnan(entry.value)) matrix->data.as<float>()[static_cast<std::size_t>(row) * matrix->cols + entry.feature] = entry.value;
      }
    }
    matrix->sparse_page = std::make_shared<SparsePage>(SparsePage{std::move(document.offsets), std::move(document.entries)});
  }
  matrix->label_shape = {document.labels.shape[0], document.labels.shape[1]};
  matrix->margin_shape = {document.margins.shape[0], document.margins.shape[1]};
  matrix->label_cols = std::max<std::size_t>(matrix->label_shape[1], 1);
  matrix->info["label"] = std::move(document.labels.values);
  matrix->info["base_margin"] = std::move(document.margins.values);
  matrix->info["weight"] = std::move(document.weights);
  matrix->info["label_lower_bound"] = std::move(document.lower_bounds);
  matrix->info["label_upper_bound"] = std::move(document.upper_bounds);
  matrix->info["feature_weights"] = std::move(document.feature_weights);
  matrix->group_ptr = std::move(document.groups);
  matrix->feature_names = std::move(document.feature_names);
  matrix->feature_types = std::move(document.feature_types);
  matrix->categories = std::move(categories);
  matrix->category_sorted_idx = encoded.at("sorted_idx").get<std::vector<std::int32_t>>();
  matrix->refresh_metadata();
  return matrix;
}

void Matrix::save_binary(std::string const& filename) const {
  // Reject before opening the destination, including proxy and iterator input.
  require(is_simple && !is_quantile, "binary saving only supported by SimpleDMatrix");
  binary::Document document;
  document.rows = rows; document.columns = cols;
  document.labels = {{label_shape[0], label_shape[1]}, get_info("label")};
  document.margins = {{margin_shape[0], margin_shape[1]}, get_info("base_margin")};
  document.groups = group_ptr;
  document.weights = get_info("weight");
  document.lower_bounds = get_info("label_lower_bound");
  document.upper_bounds = get_info("label_upper_bound");
  document.feature_weights = get_info("feature_weights");
  document.feature_names = feature_names; document.feature_types = feature_types;
  document.categories = category_bytes(categories, category_sorted_idx);
  auto page = raw_csr();
  document.offsets = std::move(page.offsets); document.entries = std::move(page.entries);
  document.nonzero = document.entries.size();
  binary::write(document, filename);
}
}  // namespace metalgb
