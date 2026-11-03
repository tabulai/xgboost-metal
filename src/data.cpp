#include "metalgb/engine.hpp"
#include <algorithm>
#include <bit>
#include <charconv>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace metalgb {
void require(bool condition, std::string const& message) {
  if (!condition) throw std::invalid_argument(message);
}
std::size_t checked_product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::overflow_error("Array size overflow.");
  return a * b;
}
Json parse_json(std::string_view text, bool ieee_numbers) {
  // Python's json encoder emits these IEEE values for missing-data options.
  std::string clean;
  clean.reserve(text.size());
  std::vector<int> string_kinds;
  bool quoted = false, escaped = false;
  for (std::size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    if (!quoted) {
      if (text.substr(i, 3) == "NaN") {
        clean += ieee_numbers ? "\"NaN\"" : "null";
        if (ieee_numbers) string_kinds.push_back(1);
        i += 2; continue;
      }
      if (text.substr(i, 9) == "-Infinity") {
        clean += "\"-Infinity\"";
        if (ieee_numbers) string_kinds.push_back(2);
        i += 8; continue;
      }
      if (text.substr(i, 8) == "Infinity") {
        clean += "\"Infinity\"";
        if (ieee_numbers) string_kinds.push_back(3);
        i += 7; continue;
      }
      if (ieee_numbers && c == '"') string_kinds.push_back(0);
    }
    clean += c;
    if (c == '"' && !escaped) quoted = !quoted;
    escaped = c == '\\' && !escaped;
  }
  if (!ieee_numbers) return Json::parse(clean.empty() ? "{}" : clean);
  // Token positions distinguish bare IEEE numbers from ordinary string
  // attributes, including strings whose contents are themselves Infinity.
  std::size_t string_index = 0;
  return Json::parse(clean.empty() ? "{}" : clean, [&](int, Json::parse_event_t event, Json& value) {
    if (event == Json::parse_event_t::key || (event == Json::parse_event_t::value && value.is_string())) {
      auto kind = string_kinds.at(string_index++);
      if (kind == 1) value = std::numeric_limits<double>::quiet_NaN();
      else if (kind == 2) value = -std::numeric_limits<double>::infinity();
      else if (kind == 3) value = std::numeric_limits<double>::infinity();
    }
    return true;
  });
}
namespace {
bool has_nonfinite_json(Json const& value) {
  if (value.is_number_float()) return !std::isfinite(value.get<double>());
  if (value.is_structured())
    for (auto const& child : value) if (has_nonfinite_json(child)) return true;
  return false;
}
void write_json_ieee(Json const& value, std::string& output, bool lower_case, int indent, std::size_t depth) {
  auto newline = [&](std::size_t level) {
    if (indent >= 0) { output += '\n'; output.append(level * static_cast<std::size_t>(indent), ' '); }
  };
  if (value.is_structured()) {
    bool object = value.is_object(), first = true;
    output += object ? '{' : '[';
    for (auto item = value.cbegin(); item != value.cend(); ++item) {
      if (!first) output += ',';
      first = false; newline(depth + 1);
      if (object) { output += Json(item.key()).dump(); output += indent >= 0 ? ": " : ":"; }
      write_json_ieee(item.value(), output, lower_case, indent, depth + 1);
    }
    if (!value.empty()) newline(depth);
    output += object ? '}' : ']';
  } else if (value.is_number_float() && !std::isfinite(value.get<double>())) {
    double number = value.get<double>();
    if (std::isnan(number)) output += lower_case ? "nan" : "NaN";
    else {
      if (std::signbit(number)) output += '-';
      output += lower_case ? "inf" : "Infinity";
    }
  } else output += value.dump();
}
}
std::string dump_json(Json const& value, bool tree_dump, int indent) {
  if (!has_nonfinite_json(value)) return value.dump(indent);
  std::string output;
  write_json_ieee(value, output, tree_dump, indent, 0);
  return output;
}
float missing_value(Json const& config) {
  if (!config.contains("missing") || config["missing"].is_null()) return kNaN;
  if (config["missing"].is_string()) return std::stof(config["missing"].get<std::string>());
  return config["missing"].get<float>();
}
Array::Array(Json const& a) {
  require(a.is_object(), "Expected an array-interface object.");
  require(!a.contains("stream"), "CUDA pointers cannot be consumed by the Apple Metal backend.");
  shape = a.at("shape").get<std::vector<std::size_t>>();
  require(!shape.empty() && shape.size() <= 2, "Expected a one- or two-dimensional array.");
  type = a.at("typestr").get<std::string>();
  require(type.size() >= 3 && type[0] != '>', "Expected a native little-endian numeric dtype.");
  pointer = a.at("data").at(0).get<std::uintptr_t>();
  width = std::stoul(type.substr(2));
  require(width == 1 || width == 2 || width == 4 || width == 8, "Unsupported numeric dtype: " + type);
  require(type[1] == 'f' || type[1] == 'i' || type[1] == 'u' || type[1] == 'b' || type[1] == 't', "Unsupported numeric dtype: " + type);
  if (a.contains("strides") && !a["strides"].is_null()) strides = a["strides"].get<std::vector<std::ptrdiff_t>>();
  else {
    strides.resize(shape.size());
    std::size_t stride = width;
    for (std::size_t i = shape.size(); i-- > 0;) { strides[i] = static_cast<std::ptrdiff_t>(stride); stride = checked_product(stride, shape[i]); }
  }
  require(strides.size() == shape.size(), "Invalid array strides.");
  require(pointer != 0 || size() == 0, "Null data pointer for a nonempty array.");
  if (a.contains("mask") && !a["mask"].is_null()) {
    mask = a["mask"];
    mask_pointer_ = mask.at("data").at(0).get<std::uintptr_t>();
    auto mask_type = mask.at("typestr").get<std::string>();
    require(mask_type.size() >= 3, "Invalid validity-mask dtype.");
    mask_bits_ = mask_type[1] == 't';
    require(mask_pointer_ != 0 || size() == 0, "Null validity mask for a nonempty array.");
  }
}
std::size_t Array::size() const {
  std::size_t n = 1;
  for (auto dim : shape) n = checked_product(n, dim);
  return n;
}
namespace {
template<class T> T read_scalar(std::byte const* p) { T value; std::memcpy(&value, p, sizeof(T)); return value; }
template<class T, class Convert>
void convert_array(Array const& array, std::span<float> out, Convert convert) {
  if (out.empty()) return;
  auto source = reinterpret_cast<std::byte const*>(array.pointer);
  std::size_t columns = array.shape.size() == 2 ? array.shape[1] : 1;
  if (array.strides.back() == static_cast<std::ptrdiff_t>(sizeof(T)) &&
      array.strides[0] == static_cast<std::ptrdiff_t>(columns * sizeof(T))) {
    if constexpr (std::is_same_v<T, float>) std::memcpy(out.data(), source, out.size_bytes());
    else for (std::size_t i = 0; i < out.size(); ++i) out[i] = convert(read_scalar<T>(source + i * sizeof(T)));
    return;
  }
  auto column_stride = array.shape.size() == 2 ? array.strides[1] : 0;
  for (std::size_t r = 0; r < array.shape[0]; ++r) {
    auto row = source + static_cast<std::ptrdiff_t>(r) * array.strides[0];
    for (std::size_t c = 0; c < columns; ++c)
      out[r * columns + c] = convert(read_scalar<T>(row + static_cast<std::ptrdiff_t>(c) * column_stride));
  }
}
float half_to_float(std::uint16_t bits) {
  unsigned sign = unsigned(bits & 0x8000) << 16;
  unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
  if (exponent == 0) return (bits & 0x8000 ? -1.0f : 1.0f) * std::ldexp(static_cast<float>(fraction), -24);
  if (exponent == 31) return std::bit_cast<float>(sign | 0x7f800000 | (fraction << 13));
  return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (fraction << 13));
}
}
bool Array::valid(std::size_t row) const {
  if (!mask_pointer_) return true;
  auto values = reinterpret_cast<std::uint8_t const*>(mask_pointer_);
  return mask_bits_ ? (values[row / 8] & (1u << (row % 8))) != 0 : values[row] != 0;
}
double Array::at(std::size_t row, std::size_t column) const {
  if (!valid(row)) return kNaN;
  auto p = reinterpret_cast<std::byte const*>(pointer) + static_cast<std::ptrdiff_t>(row) * strides[0];
  if (shape.size() == 2) p += static_cast<std::ptrdiff_t>(column) * strides[1];
  switch (type[1]) {
    case 'f':
      if (width == 4) return read_scalar<float>(p);
      if (width == 8) return read_scalar<double>(p);
      if (width == 2) return half_to_float(read_scalar<std::uint16_t>(p));
      break;
    case 'i':
      if (width == 1) return read_scalar<std::int8_t>(p);
      if (width == 2) return read_scalar<std::int16_t>(p);
      if (width == 4) return read_scalar<std::int32_t>(p);
      if (width == 8) return static_cast<double>(read_scalar<std::int64_t>(p));
      break;
    case 'u': case 'b': case 't':
      if (width == 1) return read_scalar<std::uint8_t>(p);
      if (width == 2) return read_scalar<std::uint16_t>(p);
      if (width == 4) return read_scalar<std::uint32_t>(p);
      if (width == 8) return static_cast<double>(read_scalar<std::uint64_t>(p));
  }
  throw std::invalid_argument("Unsupported array dtype: " + type);
}
std::vector<float> Array::floats() const {
  std::vector<float> values(size());
  copy_floats(values);
  return values;
}
void Array::copy_floats(std::span<float> out) const {
  require(out.size() == size(), "Array destination has an incompatible shape.");
  if (out.empty()) return;
  auto copy = [&]<class T>() { convert_array<T>(*this, out, [](T value) { return static_cast<float>(value); }); };
  switch (type[1]) {
    case 'f':
      if (width == 2) convert_array<std::uint16_t>(*this, out, half_to_float);
      else if (width == 4) copy.template operator()<float>();
      else if (width == 8) copy.template operator()<double>();
      else throw std::invalid_argument("Unsupported array dtype: " + type);
      break;
    case 'i':
      if (width == 1) copy.template operator()<std::int8_t>();
      if (width == 2) copy.template operator()<std::int16_t>();
      if (width == 4) copy.template operator()<std::int32_t>();
      if (width == 8) copy.template operator()<std::int64_t>();
      break;
    case 'u': case 'b': case 't':
      if (width == 1) copy.template operator()<std::uint8_t>();
      if (width == 2) copy.template operator()<std::uint16_t>();
      if (width == 4) copy.template operator()<std::uint32_t>();
      if (width == 8) copy.template operator()<std::uint64_t>();
      break;
  }
  if (mask_pointer_) {
    auto columns = shape.size() == 2 ? shape[1] : 1;
    for (std::size_t row = 0; row < shape[0]; ++row)
      if (!valid(row)) std::fill_n(out.data() + row * columns, columns, kNaN);
  }
}
Json array_interface(void const* pointer, std::vector<std::size_t> shape, std::string type) {
  return {{"data", {reinterpret_cast<std::uintptr_t>(pointer), true}}, {"shape", shape},
          {"strides", nullptr}, {"typestr", type}, {"version", 3}};
}
Matrix::Matrix() : data(0), labels(0), weights(0), margins(0) {}
Matrix::Matrix(u32 nr, u32 nc, std::vector<float> values, float missing)
    : Matrix(nr, nc, Buffer(values.data(), values.size() * sizeof(float)), missing) {}
Matrix::Matrix(u32 nr, u32 nc, Buffer values, float missing, bool column_major) : rows(nr), cols(nc), data(std::move(values)) {
  require(data.size() == checked_product(checked_product(rows, cols), sizeof(float)), "Data shape does not match its storage.");
  auto blocks = column_major ? checked_product(div_up(rows, 32), div_up(cols, 32)) : div_up(data.size() / sizeof(float), 256);
  auto block_stride = static_cast<u32>(std::min<std::size_t>(blocks, 65535));
  if (blocks) {
    NormalizeParams p{rows, cols, missing, block_stride};
    Buffer invalid(sizeof(u32)); invalid.zero();
    Command normalize("normalize and validate input");
    if (column_major) {
      auto columns = data;
      data = Buffer(columns.size());
      normalize.dispatch("transpose_columns", {columns, data, invalid}, p,
                         {block_stride, div_up(blocks, block_stride), 1}, {32, 8, 1});
    } else normalize.dispatch("normalize_data", {data, invalid}, p,
                              {block_stride, div_up(blocks, block_stride), 1});
    normalize.wait();
    require(!invalid.as<u32>()[0], "Input data contains infinity or a value too large for float32.");
  }
  refresh_metadata();
}
std::shared_ptr<Matrix> Matrix::dense(Json const& description, float missing) {
  Array a(description);
  require(a.shape.size() == 2, "Feature data must be two dimensional.");
  require(a.shape[0] <= UINT32_MAX && a.shape[1] <= UINT32_MAX, "Matrix dimension exceeds the 32-bit row or feature index range.");
  if (a.type[1] == 'f' && a.width == sizeof(float) && a.mask.is_null() &&
      a.strides[1] == sizeof(float) && a.strides[0] == static_cast<std::ptrdiff_t>(a.shape[1] * sizeof(float)))
    return std::make_shared<Matrix>(a.shape[0], a.shape[1],
                                   Buffer(reinterpret_cast<void const*>(a.pointer), checked_product(a.size(), sizeof(float))), missing);
  auto nr = a.shape[0], nc = a.shape[1];
  bool column_major = a.mask.is_null() && a.strides[0] == static_cast<std::ptrdiff_t>(a.width) &&
                      a.strides[1] == static_cast<std::ptrdiff_t>(nr * a.width);
  if (column_major) {
    // Convert the contiguous physical sequence, then transpose it on the GPU.
    a.shape = {a.size()};
    a.strides = {static_cast<std::ptrdiff_t>(a.width)};
  }
  Buffer values(checked_product(a.size(), sizeof(float)));
  a.copy_floats({values.as<float>(), a.size()});
  return std::make_shared<Matrix>(nr, nc, std::move(values), missing, column_major);
}
std::shared_ptr<Matrix> Matrix::columnar(Json const& description, float missing) {
  Json const& columns = description.is_array() ? description : description.at("columns");
  require(columns.is_array() && !columns.empty(), "Expected at least one feature column.");
  auto const& first = columns[0].is_array() ? columns[0][1] : columns[0];
  auto nr = Array(first).shape[0];
  require(nr <= UINT32_MAX && columns.size() <= UINT32_MAX, "Columnar input exceeds dimension limits.");
  auto nc = static_cast<u32>(columns.size());
  std::vector<Array> arrays;
  arrays.reserve(nc);
  Json categories = Json::array();
  for (u32 c = 0; c < nc; ++c) {
    bool categorical = columns[c].is_array();
    arrays.emplace_back(categorical ? columns[c][1] : columns[c]);
    auto const& a = arrays.back();
    require(a.shape[0] == nr, "All feature columns must have the same length.");
    require(a.shape.size() == 1 || a.shape[1] == 1, "Each column must contain one value per row.");
    categories.push_back(categorical ? category_column(columns[c][0]) : Json(nullptr));
  }
  auto row_stride = checked_product(nc, arrays[0].width);
  bool row_major = true;
  for (u32 c = 0; c < nc; ++c) {
    auto const& a = arrays[c];
    row_major &= a.type == arrays[0].type && a.mask.is_null() &&
                 a.strides[0] == static_cast<std::ptrdiff_t>(row_stride) &&
                 a.pointer == arrays[0].pointer + c * arrays[0].width;
  }
  Buffer values(checked_product(checked_product(nr, nc), sizeof(float)));
  if (row_major) {
    // A dataframe can expose adjacent views into one row-major numeric block.
    auto a = arrays[0];
    a.shape = {nr, nc};
    a.strides = {static_cast<std::ptrdiff_t>(row_stride), static_cast<std::ptrdiff_t>(a.width)};
    a.copy_floats({values.as<float>(), checked_product(nr, nc)});
  } else for (u32 c = 0; c < nc; ++c) arrays[c].copy_floats({values.as<float>() + c * nr, nr});
  if (description.is_object() && description.contains("ref_categories") && !description["ref_categories"].is_null()) {
    auto pointer = description["ref_categories"].get<std::uintptr_t>();
    if (pointer) {
      auto const& reference = reinterpret_cast<CategoryStorage const*>(pointer)->columns;
      if (std::any_of(reference.begin(), reference.end(), [](auto const& column) { return !column.is_null(); })) {
        // The reference adapter recodes before filtering missing values.
        values = recode_category_input(values, nr, nc, categories, reference, !row_major);
        categories = reference;
        row_major = true;
      }
    }
  }
  auto matrix = std::make_shared<Matrix>(nr, nc, std::move(values), missing, !row_major);
  matrix->categories = std::move(categories);
  for (auto const& column : matrix->categories) matrix->feature_types.push_back(column.is_null() ? "q" : "c");
  return matrix;
}
std::shared_ptr<Matrix> Matrix::sparse(Json const& ip, Json const& ix, Json const& data,
                                     std::size_t dimension, bool csc, float missing) {
  Array indptr(ip), indices(ix), values(data);
  require(indptr.size() >= 1 && indices.size() == values.size(), "Invalid sparse matrix arrays.");
  std::size_t nr = csc ? dimension : indptr.size() - 1, nc = csc ? indptr.size() - 1 : dimension;
  require(nr <= UINT32_MAX && nc <= UINT32_MAX, "Sparse matrix exceeds dimension limits.");
  std::vector<float> dense(checked_product(nr, nc), kNaN);
  auto page = std::make_shared<SparsePage>();
  page->offsets.resize(nr + 1);
  page->entries.reserve(values.size());
  std::vector<u32> entry_rows;
  if (csc) entry_rows.reserve(values.size());
  require(indptr.at(0) == 0 && indptr.at(indptr.size() - 1) == static_cast<double>(values.size()), "Invalid sparse indptr endpoints.");
  for (std::size_t r = 0; r + 1 < indptr.size(); ++r) {
    double begin = indptr.at(r), end = indptr.at(r + 1);
    require(begin <= end && begin >= 0 && end <= values.size() &&
            std::floor(begin) == begin && std::floor(end) == end, "Invalid sparse indptr.");
    for (std::size_t i = static_cast<std::size_t>(begin); i < static_cast<std::size_t>(end); ++i) {
      double col = indices.at(i);
      require(col >= 0 && col < dimension && std::floor(col) == col, "Sparse index is out of bounds.");
      auto dst = csc ? static_cast<std::size_t>(col) * nc + r : r * nc + static_cast<std::size_t>(col);
      float value = static_cast<float>(values.at(i));
      if (!std::isnan(value) && value != missing) {
        dense[dst] = value;
        page->entries.push_back({static_cast<u32>(csc ? r : col), value});
        if (csc) { ++page->offsets[static_cast<std::size_t>(col) + 1]; entry_rows.push_back(col); }
      }
    }
    if (!csc) page->offsets[r + 1] = page->entries.size();
  }
  if (csc) {
    std::partial_sum(page->offsets.begin(), page->offsets.end(), page->offsets.begin());
    auto cursor = page->offsets;
    std::vector<SparseEntry> ordered(page->entries.size());
    for (std::size_t i = 0; i < ordered.size(); ++i) ordered[cursor[entry_rows[i]]++] = page->entries[i];
    page->entries = std::move(ordered);
  }
  auto by_feature = [](SparseEntry const& a, SparseEntry const& b) { return a.feature < b.feature; };
  for (std::size_t r = 0; r < nr; ++r) {
    auto begin = page->entries.begin() + page->offsets[r], end = page->entries.begin() + page->offsets[r + 1];
    if (!std::is_sorted(begin, end, by_feature)) std::stable_sort(begin, end, by_feature);
  }
  auto result = std::make_shared<Matrix>(nr, nc, std::move(dense), missing);
  result->sparse_page = std::move(page);
  return result;
}
SparsePage Matrix::raw_csr() const {
  require(!is_quantile, "Raw CSR is not available for QuantileDMatrix.");
  if (sparse_page) return *sparse_page;
  SparsePage page;
  page.offsets.reserve(static_cast<std::size_t>(rows) + 1);
  page.offsets.push_back(0);
  for (u32 r = 0; r < rows; ++r) {
    for (u32 c = 0; c < cols; ++c) {
      auto value = data.as<float>()[static_cast<std::size_t>(r) * cols + c];
      if (!std::isnan(value)) page.entries.push_back({c, value});
    }
    page.offsets.push_back(page.entries.size());
  }
  return page;
}
void Matrix::filter_missing(float missing) {
  if (std::isnan(missing)) return;
  for (std::size_t i = 0; i < static_cast<std::size_t>(rows) * cols; ++i)
    if (data.as<float>()[i] == missing) data.as<float>()[i] = kNaN;
  if (sparse_page) {
    auto filtered = std::make_shared<SparsePage>();
    filtered->offsets.push_back(0);
    for (u32 r = 0; r < rows; ++r) {
      for (auto i = sparse_page->offsets[r]; i < sparse_page->offsets[r + 1]; ++i) {
        auto entry = sparse_page->entries[i];
        if (entry.value != missing) filtered->entries.push_back(entry);
      }
      filtered->offsets.push_back(filtered->entries.size());
    }
    sparse_page = std::move(filtered);
  }
}
void Matrix::set_info(std::string const& field, Array const& a) {
  if (field == "group" || field == "qid") {
    group_ptr = {0};
    if (field == "group") {
      std::uint64_t sum = 0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        auto v = a.at(i);
        require(v >= 0 && v <= rows && std::floor(v) == v, "Invalid group size.");
        sum += static_cast<std::uint64_t>(v);
        require(sum <= rows, "Invalid group structure.");
        group_ptr.push_back(sum);
      }
      require(sum == rows, "Group sizes must sum to the number of rows.");
    } else {
      require(a.size() == rows, "Query id length must equal the number of rows.");
      double last = -1;
      for (u32 r = 0; r < rows; ++r) {
        double v = a.at(r);
        require(v >= last && v >= 0 && std::floor(v) == v, "Query ids must be sorted nonnegative integers.");
        if (r && v != last) group_ptr.push_back(r);
        last = v;
      }
      group_ptr.push_back(rows);
    }
    ++generation;
    refresh_metadata();
    return;
  }
  set_info(field, a.floats(), {a.shape[0], a.shape.size() == 2 ? a.shape[1] : 1});
}
void Matrix::set_info(std::string const& field, std::vector<float> values,
                      std::array<std::size_t, 2> shape) {
  require(field == "label" || field == "weight" || field == "base_margin" || field == "label_lower_bound" || field == "label_upper_bound" || field == "feature_weights", "Unknown metadata field: " + field);
  if (field == "label" || field == "base_margin") {
    // Inplace prediction attaches margins to an empty proxy before feature
    // data arrives. As in MetaInfo::ReshapeInfo, defer row checks until then.
    if (shape[0] == SIZE_MAX) shape = {values.size(), 1};
    require(checked_product(shape[0], shape[1]) == values.size(), "Invalid metadata tensor shape.");
    if (rows && shape[0] != rows) {
      require(values.size() % rows == 0, "Metadata shape does not match the number of rows.");
      shape = {rows, values.size() / rows};
    }
    require(shape[1] <= UINT32_MAX, "Metadata columns exceed the index range.");
    if (field == "label") {
      require(std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }),
              "Label contains NaN, infinity or a value too large.");
      label_shape = shape;
      label_cols = std::max<std::size_t>(shape[1], 1);
    } else margin_shape = shape;
  }
  if (field == "weight" || field == "feature_weights") {
    require(std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value) && value >= 0; }),
            "Weights must be finite and nonnegative.");
    if (field == "feature_weights") require(values.empty() || values.size() == cols, "Feature weight length must match the number of columns.");
  }
  info[field] = std::move(values);
  ++generation;
  refresh_metadata();
}
std::vector<float> const& Matrix::get_info(std::string_view field) const {
  static std::vector<float> const empty;
  auto it = info.find(field);
  return it == info.end() ? empty : it->second;
}
void Matrix::refresh_metadata() {
  auto const& y = get_info("label"); labels = Buffer(y.data(), y.size() * sizeof(float));
  auto const& w = get_info("weight");
  if (w.empty()) { std::vector<float> ones(rows, 1.0f); weights = Buffer(ones.data(), ones.size() * sizeof(float)); }
  else weights = Buffer(w.data(), w.size() * sizeof(float));
  sketch_weights = weights;
  // A proxy batch keeps the previous batch's group pointers until its own
  // query ids arrive. Expand group weights only for groups covering these rows.
  if (!w.empty() && !group_ptr.empty() && w.size() + 1 == group_ptr.size() && w.size() != rows &&
      group_ptr.back() == rows) {
    // Metadata expansion only. The weighted quantile calculation stays on Metal.
    std::vector<float> expanded(rows);
    for (std::size_t g = 0; g < w.size(); ++g)
      std::fill(expanded.begin() + group_ptr[g], expanded.begin() + group_ptr[g + 1], w[g]);
    sketch_weights = Buffer(expanded.data(), expanded.size() * sizeof(float));
  }
  auto const& m = get_info("base_margin"); margins = Buffer(m.data(), m.size() * sizeof(float));
  auto const& lower = get_info("label_lower_bound");
  auto const& upper = get_info("label_upper_bound");
  lower_bounds = lower.empty() ? Buffer{} : Buffer(lower.data(), lower.size() * sizeof(float));
  upper_bounds = upper.empty() ? Buffer{} : Buffer(upper.data(), upper.size() * sizeof(float));
}
void Matrix::quantize(u32 requested_bins, Matrix const* reference) {
  require(requested_bins >= 2 && requested_bins <= INT32_MAX, "max_bin must be in [2, 2147483647].");
  if (max_bin == requested_bins && bins) return;
  require(!is_quantile, "Inconsistent max_bin between QuantileDMatrix and Booster.");
  if (reference) {
    require(reference->max_bin == requested_bins && reference->cols == cols, "Quantile reference does not match max_bin or feature count.");
    // The reference supplies effective types along with its cuts. Public type
    // names still belong to this input, as in IterativeDMatrix::InitFromCPU.
    categorical = reference->categorical;
  } else {
    auto const& weight = get_info("weight");
    require(weight.empty() || (group_ptr.empty() ? weight.size() == rows : weight.size() + 1 == group_ptr.size()),
            group_ptr.empty() ? "Number of weights must equal the number of rows." : "Ranking weights must have one entry per query group.");
    if (!categorical) update_feature_types();
  }
  auto types = categorical.as<std::uint8_t>();
  cuts_categorical_ = reference ? reference->cuts_categorical_ :
      std::any_of(types, types + cols, [](auto type) { return type != 0; });
  histogram_bins = std::max(1u, std::min(requested_bins, rows));
  MatrixParams p{rows, cols, static_cast<u32>(div_up(rows, 256) * 256), histogram_bins, requested_bins};
  cut_counts = Buffer(cols * sizeof(u32)); cut_counts.zero();
  if (reference) {
    histogram_bins = reference->histogram_bins; p.bins = histogram_bins;
    cuts = reference->cuts; cut_counts = reference->cut_counts; cut_minima = reference->cut_minima;
    cut_tile_size = reference->cut_tile_size; cut_tile_count = reference->cut_tile_count;
    summary_group_size = reference->summary_group_size; summary_batch_count = reference->summary_batch_count;
    weighted_summary_stages = reference->weighted_summary_stages;
    weighted_summary_batch_count = reference->weighted_summary_batch_count;
  } else {
    if (std::any_of(types, types + cols, [](auto type) { return type != 0; })) {
      Buffer invalid(sizeof(u32)); invalid.zero();
      Command statistics("categorical bin counts");
      statistics.dispatch("category_statistics", {data, categorical, cut_counts, invalid}, p,
                          {div_up(rows, 256), cols, 1});
      quantize_seconds += statistics.wait(); ++quantize_commands;
      require(!invalid.as<u32>()[0], "Invalid categorical value: codes must be nonnegative and less than 16777216.");
      for (u32 c = 0; c < cols; ++c) if (types[c]) {
        cut_counts.as<u32>()[c] = std::max(1u, cut_counts.as<u32>()[c]);
        histogram_bins = std::max(histogram_bins, cut_counts.as<u32>()[c]);
      }
      p.bins = histogram_bins;
    }
    make_quantile_cuts(p);
  }
  bins = Buffer(checked_product(checked_product(rows, cols), histogram_bins <= 65534 ? sizeof(std::uint16_t) : sizeof(u32)));
  Buffer missing(sizeof(u32)); missing.zero();
  Command command("quantize");
  command.dispatch("quantize", {data, cuts, cut_counts, bins, categorical, missing}, p, {div_up(rows, 256), cols, 1});
  quantize_seconds += command.wait(); ++quantize_commands;
  quantized_dense = !missing.as<u32>()[0];
  max_bin = requested_bins;
  quantized_types_changed_ = reference && reference->quantized_types_changed_;
  partition_data_ = {}; partition_bounds_ = {};
  packed_bins_ = {}; packed_chunk_ = 0;
  export_cuts();
}
void Matrix::decode_quantiles(Buffer const& output, bool representative, bool numeric_only) const {
  QuantileValueParams p{rows, cols, histogram_bins, static_cast<u32>(representative), static_cast<u32>(numeric_only)};
  Command command("reconstruct quantile values");
  command.dispatch("quantile_values", {bins, cuts, categorical, output}, p,
                   {div_up(rows, 32), div_up(cols, 32), 1}, {32, 8, 1});
  command.wait();
}
void Matrix::finish_quantile() {
  require(max_bin && bins, "QuantileDMatrix must be quantized before reconstruction.");
  is_quantile = true;
  sparse_page.reset();
  decode_quantiles(data, false);
  std::lock_guard lock(recode_mutex_);
  recoded_data_ = {}; recoded_categories_ = nullptr;
}
void Matrix::update_feature_types() {
  if (!cols) return;
  std::vector<std::uint8_t> types(cols, 0);
  for (u32 c = 0; c < feature_types.size(); ++c) types[c] = feature_types[c] == "c";
  if (categorical.size() == types.size() && categorical &&
      std::equal(types.begin(), types.end(), categorical.as<std::uint8_t>())) return;
  categorical = Buffer(types.data(), types.size());
  quantized_types_changed_ = quantized_types_changed_ || max_bin != 0;
  if (max_bin) export_cuts();
  if (is_quantile) decode_quantiles(data, false);
  ++generation;
  std::lock_guard lock(recode_mutex_);
  recoded_data_ = {}; recoded_categories_ = nullptr;
  partition_data_ = {}; partition_bounds_ = {};
}
Buffer Matrix::export_data() const {
  if (!is_quantile) return data;
  Buffer output(data.size());
  decode_quantiles(output, true);
  return output;
}
Buffer Matrix::partition_data() const {
  if (!quantized_types_changed_ || (is_quantile && cuts_categorical_)) return data;
  std::lock_guard lock(recode_mutex_);
  if (!partition_data_) {
    partition_data_ = Buffer(data.size());
    decode_quantiles(partition_data_, false, !cuts_categorical_);
  }
  return partition_data_;
}
Buffer Matrix::packed_bins(u32 chunk) const {
  if (!bins || !rows || !cols || histogram_bins > 256) return {};
  std::lock_guard lock(recode_mutex_);
  if (packed_chunk_ != chunk) {
    // Built once per quantization and chunk width. An unrepresentable code
    // is also remembered, so later trees do not repeat the conversion.
    auto groups = div_up(div_up(cols, 4), chunk) * chunk;
    Buffer packed(checked_product(checked_product(groups, rows), sizeof(u32)));
    Buffer invalid(sizeof(u32)); invalid.zero();
    PackBinsParams p{rows, cols, histogram_bins, chunk};
    Command command("pack histogram bins");
    command.dispatch("pack_histogram_bins", {bins, packed, invalid}, p, {div_up(rows, 256), groups, 1});
    command.wait();
    packed_bins_ = invalid.as<u32>()[0] ? Buffer{} : std::move(packed);
    packed_chunk_ = chunk;
  }
  return packed_bins_;
}
Buffer Matrix::partition_category_bounds() const {
  if (!quantized_types_changed_ || cuts_categorical_) return {};
  std::lock_guard lock(recode_mutex_);
  if (!partition_bounds_) {
    partition_bounds_ = Buffer(cols * sizeof(float));
    MatrixParams p{rows, cols, 0, histogram_bins, max_bin};
    Command command("quantile partition boundaries");
    command.dispatch("quantile_category_bounds", {cuts, cut_counts, partition_bounds_}, p,
                     {div_up(cols, 256), 1, 1});
    command.wait();
  }
  return partition_bounds_;
}
void Matrix::export_cuts() {
  cut_ptr = {0}; cut_values.clear();
  for (u32 c = 0; c < cols; ++c) {
    auto n = cut_counts.as<u32>()[c];
    auto values = cuts.as<float>() + static_cast<std::size_t>(c) * histogram_bins;
    if (!categorical.as<std::uint8_t>()[c]) cut_values.push_back(-std::numeric_limits<float>::infinity());
    cut_values.insert(cut_values.end(), values, values + n);
    cut_ptr.push_back(cut_values.size());
  }
}
std::shared_ptr<Matrix> Matrix::slice(std::span<int const> indices, bool allow_groups) const {
  require(group_ptr.empty() || allow_groups, "Slicing does not support group structure unless allow_groups=True.");
  require(is_simple && !is_quantile, "Slicing DMatrix is not supported for external memory.");
  require(indices.size() <= UINT32_MAX, "DMatrix slice exceeds the row index range.");
  std::vector<float> values(checked_product(indices.size(), cols));
  for (std::size_t i = 0; i < indices.size(); ++i) {
    require(indices[i] >= 0 && static_cast<u32>(indices[i]) < rows, "DMatrix slice index out of bounds.");
    std::copy_n(data.as<float>() + static_cast<std::size_t>(indices[i]) * cols, cols, values.data() + i * cols);
  }
  auto result = std::make_shared<Matrix>();
  result->rows = indices.size(); result->cols = cols;
  result->data = Buffer(values.data(), values.size() * sizeof(float));
  if (sparse_page) {
    result->sparse_page = std::make_shared<SparsePage>();
    auto& page = *result->sparse_page;
    page.offsets.push_back(0);
    for (auto row : indices) {
      auto begin = sparse_page->entries.begin() + sparse_page->offsets[row];
      auto end = sparse_page->entries.begin() + sparse_page->offsets[row + 1];
      page.entries.insert(page.entries.end(), begin, end);
      page.offsets.push_back(page.entries.size());
    }
  }
  for (auto const& [field, metadata] : info) {
    if (field == "label" || field == "base_margin") continue;
    if (field == "feature_weights") { result->info[field] = metadata; continue; }
    // Like the reference, grouped slicing drops group weights and pointers.
    // Python's CV adapter reconstructs the group boundaries after slicing.
    if (field == "weight" && !group_ptr.empty() && metadata.size() + 1 == group_ptr.size()) continue;
    require(metadata.empty() || (rows && metadata.size() % rows == 0), "Cannot slice group weights as row weights.");
    auto columns = rows ? metadata.size() / rows : 0;
    std::vector<float> out(indices.size() * columns);
    for (std::size_t i = 0; i < indices.size(); ++i) std::copy_n(metadata.data() + static_cast<std::size_t>(indices[i]) * columns, columns, out.data() + i * columns);
    result->info[field] = std::move(out);
  }
  for (auto field : {"label", "base_margin"}) {
    auto const& metadata = get_info(field);
    auto shape = std::string_view(field) == "label" ? label_shape : margin_shape;
    auto columns = metadata.size() == rows ? 1 : shape[1];
    require(metadata.empty() || (rows && shape[0] == rows), "Cannot slice metadata with an inconsistent row count.");
    std::vector<float> out(checked_product(indices.size(), columns));
    if (columns) for (std::size_t i = 0; i < indices.size(); ++i)
      std::copy_n(metadata.data() + static_cast<std::size_t>(indices[i]) * columns, columns, out.data() + i * columns);
    result->info[field] = std::move(out);
    auto& output_shape = std::string_view(field) == "label" ? result->label_shape : result->margin_shape;
    output_shape = {indices.size(), columns};
  }
  result->feature_names = feature_names; result->feature_types = feature_types;
  result->label_cols = std::max<std::size_t>(result->label_shape[1], 1); result->categories = categories;
  result->category_sorted_idx = category_sorted_idx;
  result->refresh_metadata();
  return result;
}
void Matrix::append(Matrix const& batch) {
  require(rows == 0 || cols == batch.cols, "Iterator batches have different feature counts.");
  require(std::uint64_t(rows) + batch.rows <= UINT32_MAX, "Iterator exceeds row index range.");
  require(batch.group_ptr.empty() || batch.group_ptr.back() == batch.rows,
          "Query group sizes must sum to the number of rows in each iterator batch.");
  auto new_rows = rows + batch.rows;
  auto stack_shape = [&](auto left, auto const& right, std::string_view field) {
    require(!left[1] || left[1] == right[1], "Iterator batches have inconsistent metadata columns.");
    if (!left[1]) left[1] = right[1];
    require(left[0] <= SIZE_MAX - right[0], "Iterator metadata exceeds the shape range.");
    left[0] += right[0];
    require(checked_product(left[0], left[1]) == get_info(field).size() + batch.get_info(field).size(),
            "Inconsistent metadata size after concatenation.");
    return left;
  };
  auto new_label_shape = stack_shape(label_shape, batch.label_shape, "label");
  auto new_margin_shape = stack_shape(margin_shape, batch.margin_shape, "base_margin");
  std::vector<float> values(checked_product(new_rows, batch.cols));
  if (rows) std::copy_n(data.as<float>(), static_cast<std::size_t>(rows) * cols, values.data());
  auto batch_data = rows ? batch.data_for_categories(categories) : batch.data;
  std::copy_n(batch_data.as<float>(), static_cast<std::size_t>(batch.rows) * batch.cols, values.data() + static_cast<std::size_t>(rows) * batch.cols);
  if (sparse_page || batch.sparse_page) {
    auto page = raw_csr(), incoming = batch.raw_csr();
    auto offset = page.entries.size();
    page.entries.insert(page.entries.end(), incoming.entries.begin(), incoming.entries.end());
    for (std::size_t r = 1; r < incoming.offsets.size(); ++r) page.offsets.push_back(offset + incoming.offsets[r]);
    sparse_page = std::make_shared<SparsePage>(std::move(page));
  }
  for (auto const& [field, metadata] : batch.info) {
    if (field == "feature_weights") info[field] = metadata;
    else info[field].insert(info[field].end(), metadata.begin(), metadata.end());
  }
  if (!batch.group_ptr.empty()) {
    if (group_ptr.empty()) group_ptr = {0};
    for (std::size_t i = 1; i < batch.group_ptr.size(); ++i) group_ptr.push_back(rows + batch.group_ptr[i]);
  }
  if (!rows) { categories = batch.categories; category_sorted_idx = batch.category_sorted_idx; }
  rows = new_rows; cols = batch.cols;
  label_shape = new_label_shape; margin_shape = new_margin_shape;
  label_cols = std::max<std::size_t>(label_shape[1], 1);
  data = Buffer(values.data(), values.size() * sizeof(float));
  feature_names = batch.feature_names; feature_types = batch.feature_types;
  max_bin = 0; ++generation; refresh_metadata();
  packed_bins_ = {}; packed_chunk_ = 0;
}
}  // namespace metalgb
