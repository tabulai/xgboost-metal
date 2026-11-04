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
namespace {
std::size_t ingest_dtype_width(u32 code) {
  switch (code) {
    case kIngestI8: case kIngestU8: case kIngestBool: return 1;
    case kIngestF16: case kIngestBF16: case kIngestI16: case kIngestU16: return 2;
    case kIngestF32: case kIngestI32: case kIngestU32: return 4;
    case kIngestF64: case kIngestI64: case kIngestU64: return 8;
    case kIngestBoolBits: return 0;
    default: throw std::invalid_argument("Unsupported buffer dtype code.");
  }
}
std::string ingest_dtype_typestr(u32 code) {
  switch (code) {
    case kIngestF32: return "<f4";
    case kIngestF64: return "<f8";
    case kIngestF16: return "<f2";
    case kIngestI8: return "|i1";
    case kIngestI16: return "<i2";
    case kIngestI32: return "<i4";
    case kIngestI64: return "<i8";
    case kIngestU8: return "|u1";
    case kIngestU16: return "<u2";
    case kIngestU32: return "<u4";
    case kIngestU64: return "<u8";
    case kIngestBool: return "|b1";
    default: throw std::invalid_argument("Unsupported metadata buffer dtype code.");
  }
}
void encode_ingest_spec(Command& cmd, IngestColumnSpec const& spec, u32 dst_col_offset, u32 dst_cols,
                        float missing, Buffer const& dst, Buffer const& invalid,
                        std::vector<Buffer>& keepalive, bool* any_zero_copied) {
  if (!spec.rows || !spec.cols) return;
  require(spec.data_ptr != 0, "Null data pointer for a nonempty array.");
  IngestParams p{};
  p.rows = spec.rows;
  p.cols = spec.cols;
  p.dtype_code = spec.dtype_code;
  p.mask_mode = spec.mask_mode;
  p.stride_r = spec.stride_r;
  p.stride_c = spec.stride_c;
  p.missing = missing;
  p.dst_col_offset = dst_col_offset;
  p.dst_cols = dst_cols;
  auto blocks = div_up(div_up(checked_product(spec.rows, spec.cols), 4), 256);
  p.block_stride = static_cast<u32>(std::min<std::size_t>(blocks, 65535));

  Buffer src_buf;
  bool zc = false;
  if (spec.dtype_code == kIngestBoolBits) {
    i64 first_bit = spec.data_bit_offset;
    i64 last_bit = first_bit + static_cast<i64>(spec.rows) - 1;
    require(first_bit >= 0, "Invalid negative bit offset.");
    auto low_addr = spec.data_ptr + static_cast<std::size_t>(first_bit / 8);
    auto span_bytes = static_cast<std::size_t>((last_bit / 8) - (first_bit / 8) + 1);
    std::size_t sub_off = 0;
    src_buf = Buffer::view_subrange_readonly(reinterpret_cast<void const*>(low_addr), span_bytes, sub_off, &zc, spec.prefer_zero_copy);
    p.data_offset = static_cast<i64>(sub_off) * 8 + (first_bit % 8);
  } else {
    auto elem_width = ingest_dtype_width(spec.dtype_code);
    i64 max_r = static_cast<i64>(spec.rows - 1) * spec.stride_r;
    i64 max_c = static_cast<i64>(spec.cols - 1) * spec.stride_c;
    i64 min_rel = std::min<i64>(0, max_r) + std::min<i64>(0, max_c);
    i64 max_rel = std::max<i64>(0, max_r) + std::max<i64>(0, max_c) + static_cast<i64>(elem_width);
    auto low_addr = static_cast<std::uintptr_t>(static_cast<std::intptr_t>(spec.data_ptr) + min_rel);
    auto span_bytes = static_cast<std::size_t>(max_rel - min_rel);
    std::size_t sub_off = 0;
    src_buf = Buffer::view_subrange_readonly(reinterpret_cast<void const*>(low_addr), span_bytes, sub_off, &zc, spec.prefer_zero_copy);
    p.data_offset = static_cast<i64>(sub_off) - min_rel;
  }
  if (any_zero_copied && zc) *any_zero_copied = true;

  auto total = checked_product(spec.rows, spec.cols);
  if (spec.mask_mode == kIngestMaskNone && dst_col_offset == 0 && dst_cols == spec.cols && p.data_offset >= 0) {
    bool is_c_contig_4 = (spec.rows == 1 ? (spec.cols == 1 || spec.stride_c == 4)
                                         : ((spec.cols == 1 || spec.stride_c == 4) &&
                                            spec.stride_r == static_cast<i64>(spec.cols) * 4));
    if (spec.dtype_code == kIngestF32 && is_c_contig_4 && (p.data_offset & 3) == 0) {
      auto vec_blocks = div_up(div_up(total, 4), 256);
      p.block_stride = static_cast<u32>(std::min<std::size_t>(vec_blocks, 65535));
      cmd.dispatch("ingest_contiguous_f32", {src_buf, dst, invalid}, p,
                   {p.block_stride, div_up(vec_blocks, p.block_stride), 1});
      keepalive.push_back(std::move(src_buf));
      return;
    }
    bool is_c_contig_8 = (spec.rows == 1 ? (spec.cols == 1 || spec.stride_c == 8)
                                         : ((spec.cols == 1 || spec.stride_c == 8) &&
                                            spec.stride_r == static_cast<i64>(spec.cols) * 8));
    if (spec.dtype_code == kIngestF64 && is_c_contig_8 && (p.data_offset & 7) == 0) {
      auto vec_blocks = div_up(div_up(total, 2), 256);
      p.block_stride = static_cast<u32>(std::min<std::size_t>(vec_blocks, 65535));
      cmd.dispatch("ingest_contiguous_f64", {src_buf, dst, invalid}, p,
                   {p.block_stride, div_up(vec_blocks, p.block_stride), 1});
      keepalive.push_back(std::move(src_buf));
      return;
    }
    if (spec.dtype_code == kIngestF32 && spec.rows > 1 && spec.cols > 1 &&
        spec.stride_r == 4 && spec.stride_c == static_cast<i64>(spec.rows) * 4 && (p.data_offset & 3) == 0) {
      auto tile_blocks = checked_product(div_up(spec.rows, 32), div_up(spec.cols, 32));
      p.block_stride = static_cast<u32>(std::min<std::size_t>(tile_blocks, 65535));
      cmd.dispatch("ingest_transpose_f32", {src_buf, dst, invalid}, p,
                   {p.block_stride, div_up(tile_blocks, p.block_stride), 1}, {32, 8, 1});
      keepalive.push_back(std::move(src_buf));
      return;
    }
  }

  Buffer mask_buf = src_buf;
  if (spec.mask_mode != kIngestMaskNone) {
    require(spec.mask_ptr != 0, "Null validity mask for a nonempty array.");
    std::size_t m_off = 0;
    if (spec.mask_mode == kIngestMaskValidBits) {
      i64 first_bit = spec.mask_bit_offset;
      i64 last_bit = first_bit + static_cast<i64>(spec.rows) - 1;
      require(first_bit >= 0, "Invalid negative mask bit offset.");
      auto m_low = spec.mask_ptr + static_cast<std::size_t>(first_bit / 8);
      auto m_span = static_cast<std::size_t>((last_bit / 8) - (first_bit / 8) + 1);
      mask_buf = Buffer::view_subrange_readonly(reinterpret_cast<void const*>(m_low), m_span, m_off);
      p.mask_bit_offset = static_cast<i64>(m_off) * 8 + (first_bit % 8);
    } else {
      require(spec.mask_bit_offset >= 0, "Invalid negative mask offset.");
      auto m_low = spec.mask_ptr + static_cast<std::size_t>(spec.mask_bit_offset);
      mask_buf = Buffer::view_subrange_readonly(reinterpret_cast<void const*>(m_low), spec.rows, m_off);
      p.mask_bit_offset = static_cast<i64>(m_off);
    }
    keepalive.push_back(mask_buf);
  }
  cmd.dispatch("ingest_strided_typed", {src_buf, mask_buf, dst, invalid}, p,
               {p.block_stride, div_up(blocks, p.block_stride), 1});
  keepalive.push_back(std::move(src_buf));
}
}
Array::Array(IngestColumnSpec const& spec) {
  pointer = spec.data_ptr;
  type = ingest_dtype_typestr(spec.dtype_code);
  width = ingest_dtype_width(spec.dtype_code);
  if (spec.cols > 1) {
    shape = {spec.rows, spec.cols};
    strides = {static_cast<std::ptrdiff_t>(spec.stride_r), static_cast<std::ptrdiff_t>(spec.stride_c)};
  } else {
    shape = {spec.rows};
    strides = {static_cast<std::ptrdiff_t>(spec.stride_r)};
  }
  require(pointer != 0 || size() == 0, "Null data pointer for a nonempty array.");
  if (spec.mask_mode != kIngestMaskNone) {
    require(spec.mask_mode != kIngestMaskInvalidBytes, "Masked array is not supported.");
    require(spec.mask_bit_offset >= 0, "Invalid negative mask offset.");
    mask_pointer_ = spec.mask_ptr;
    mask_bit_offset_ = static_cast<std::size_t>(spec.mask_bit_offset);
    mask_bits_ = spec.mask_mode == kIngestMaskValidBits;
    require(mask_pointer_ != 0 || size() == 0, "Null validity mask for a nonempty array.");
  }
}
IngestColumnSpec Array::to_ingest_spec() const {
  IngestColumnSpec spec;
  spec.data_ptr = pointer;
  spec.rows = static_cast<u32>(shape[0]);
  spec.cols = static_cast<u32>(shape.size() == 2 ? shape[1] : 1);
  spec.stride_r = static_cast<i64>(strides[0]);
  spec.stride_c = static_cast<i64>(shape.size() == 2 ? strides[1] : 0);
  switch (type[1]) {
    case 'f':
      spec.dtype_code = width == 2 ? kIngestF16 : width == 4 ? kIngestF32 : kIngestF64;
      break;
    case 'i':
      spec.dtype_code = width == 1 ? kIngestI8 : width == 2 ? kIngestI16 : width == 4 ? kIngestI32 : kIngestI64;
      break;
    case 'u': case 'b': case 't':
      spec.dtype_code = width == 1 ? kIngestU8 : width == 2 ? kIngestU16 : width == 4 ? kIngestU32 : kIngestU64;
      break;
    default:
      throw std::invalid_argument("Unsupported array dtype: " + type);
  }
  if (mask_pointer_) {
    spec.mask_ptr = mask_pointer_;
    spec.mask_mode = mask_bits_ ? kIngestMaskValidBits : kIngestMaskValidBytes;
  }
  return spec;
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
  auto idx = row + mask_bit_offset_;
  return mask_bits_ ? (values[idx / 8] & (1u << (idx % 8))) != 0 : values[idx] != 0;
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
std::shared_ptr<Matrix> Matrix::from_ingest_2d(IngestColumnSpec const& spec, float missing) {
  auto result = std::make_shared<Matrix>();
  result->rows = spec.rows;
  result->cols = spec.cols;
  auto total = checked_product(spec.rows, spec.cols);
  result->data = Buffer(checked_product(total, sizeof(float)));
  if (total) {
    Buffer invalid(sizeof(u32));
    invalid.zero();
    std::vector<Buffer> keepalive;
    Command ingest("ingest strided typed", true);
    bool zc = false;
    encode_ingest_spec(ingest, spec, 0, spec.cols, missing, result->data, invalid, keepalive, &zc);
    ingest.wait();
    require(!invalid.as<u32>()[0], "Input data contains infinity or a value too large for float32.");
    result->was_zero_copied = zc;
  }
  result->refresh_metadata();
  return result;
}
std::shared_ptr<Matrix> Matrix::from_ingest_columns(std::span<IngestColumnSpec const> columns,
                                                    u32 rows, float missing) {
  require(!columns.empty(), "Expected at least one feature column.");
  require(columns.size() <= UINT32_MAX, "Columnar input exceeds dimension limits.");
  auto nc = static_cast<u32>(columns.size());
  auto result = std::make_shared<Matrix>();
  result->rows = rows;
  result->cols = nc;
  auto total = checked_product(rows, nc);
  result->data = Buffer(checked_product(total, sizeof(float)));
  if (total) {
    Buffer invalid(sizeof(u32));
    invalid.zero();
    std::vector<Buffer> keepalive;
    keepalive.reserve(nc * 2);
    Command ingest("ingest columnar fused", true);
    bool zc = false;
    for (u32 c = 0; c < nc; ++c) {
      require(columns[c].rows == rows && columns[c].cols == 1, "All feature columns must have the same length.");
      encode_ingest_spec(ingest, columns[c], c, nc, missing, result->data, invalid, keepalive, &zc);
    }
    ingest.wait();
    require(!invalid.as<u32>()[0], "Input data contains infinity or a value too large for float32.");
    result->was_zero_copied = zc;
  }
  result->refresh_metadata();
  return result;
}
std::shared_ptr<Matrix> Matrix::dense(Json const& description, float missing) {
  Array a(description);
  require(a.shape.size() == 2, "Feature data must be two dimensional.");
  require(a.shape[0] <= UINT32_MAX && a.shape[1] <= UINT32_MAX, "Matrix dimension exceeds the 32-bit row or feature index range.");
  return from_ingest_2d(a.to_ingest_spec(), missing);
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
  Json const* ref_cat_ptr = nullptr;
  if (description.is_object() && description.contains("ref_categories") && !description["ref_categories"].is_null()) {
    auto pointer = description["ref_categories"].get<std::uintptr_t>();
    if (pointer) {
      auto const& reference = reinterpret_cast<CategoryStorage const*>(pointer)->columns;
      if (std::any_of(reference.begin(), reference.end(), [](auto const& column) { return !column.is_null(); }))
        ref_cat_ptr = &reference;
    }
  }
  if (!ref_cat_ptr) {
    std::vector<IngestColumnSpec> specs;
    specs.reserve(nc);
    for (u32 c = 0; c < nc; ++c) {
      auto spec = arrays[c].to_ingest_spec();
      spec.cols = 1;
      specs.push_back(spec);
    }
    auto matrix = from_ingest_columns(specs, static_cast<u32>(nr), missing);
    matrix->categories = std::move(categories);
    for (auto const& column : matrix->categories) matrix->feature_types.push_back(column.is_null() ? "q" : "c");
    return matrix;
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
    auto a = arrays[0];
    a.shape = {nr, nc};
    a.strides = {static_cast<std::ptrdiff_t>(row_stride), static_cast<std::ptrdiff_t>(a.width)};
    a.copy_floats({values.as<float>(), checked_product(nr, nc)});
  } else for (u32 c = 0; c < nc; ++c) arrays[c].copy_floats({values.as<float>() + c * nr, nr});
  // The reference adapter recodes before filtering missing values.
  values = recode_category_input(values, nr, nc, categories, *ref_cat_ptr, !row_major);
  categories = *ref_cat_ptr;
  auto matrix = std::make_shared<Matrix>(nr, nc, std::move(values), missing, false);
  matrix->categories = std::move(categories);
  for (auto const& column : matrix->categories) matrix->feature_types.push_back(column.is_null() ? "q" : "c");
  return matrix;
}
void prefix_sum_u32(Command& command, Buffer const& counts, Buffer const& offsets, u32 n,
                    std::vector<Buffer>& keepalive) {
  if (n == 0) {
    offsets.zero();
    return;
  }
  u32 tiles = static_cast<u32>(div_up(n, 1024));
  Buffer tile_sums(tiles * sizeof(u32));
  command.dispatch("sparse_prefix_tiles", {counts, offsets, tile_sums}, n, {tiles, 1, 1});
  if (tiles > 1) {
    Buffer tile_offsets((tiles + 1) * sizeof(u32));
    prefix_sum_u32(command, tile_sums, tile_offsets, tiles, keepalive);
    command.dispatch("sparse_prefix_add", {offsets, tile_offsets, tile_sums}, n, {tiles, 1, 1});
    keepalive.push_back(std::move(tile_offsets));
  }
  keepalive.push_back(std::move(tile_sums));
}

namespace {
bool extract_u32_array(Array const& arr, Buffer& out, u32 upper_exclusive, bool check_endpoints, u32 expected_last) {
  if (arr.shape.size() != 1 || !arr.mask.is_null() || arr.strides.empty() ||
      arr.strides[0] != static_cast<std::ptrdiff_t>(arr.width))
    return false;
  auto n = arr.size();
  auto const* base = reinterpret_cast<std::byte const*>(arr.pointer);
  if (arr.type == "<i4" || arr.type == "<u4") {
    if (check_endpoints && n > 0) {
      if (arr.type == "<i4") {
        auto const* ptr = reinterpret_cast<std::int32_t const*>(base);
        if (ptr[0] != 0 || ptr[n - 1] != static_cast<std::int32_t>(expected_last)) return false;
      } else {
        auto const* ptr = reinterpret_cast<u32 const*>(base);
        if (ptr[0] != 0u || ptr[n - 1] != expected_last) return false;
      }
    }
    out = Buffer(base, n * sizeof(u32));
    return true;
  }
  if (arr.type == "<i8" || arr.type == "<u8") {
    out = Buffer(n * sizeof(u32));
    auto* dst = out.as<u32>();
    if (arr.type == "<i8") {
      auto const* src = reinterpret_cast<std::int64_t const*>(base);
      if (check_endpoints && n > 0 && (src[0] != 0 || src[n - 1] != static_cast<std::int64_t>(expected_last)))
        return false;
      for (std::size_t i = 0; i < n; ++i) {
        std::int64_t v = src[i];
        dst[i] = (v < 0 || v >= static_cast<std::int64_t>(upper_exclusive)) ? 0xffffffffu : static_cast<u32>(v);
      }
    } else {
      auto const* src = reinterpret_cast<std::uint64_t const*>(base);
      if (check_endpoints && n > 0 && (src[0] != 0u || src[n - 1] != expected_last)) return false;
      for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t v = src[i];
        dst[i] = v >= upper_exclusive ? 0xffffffffu : static_cast<u32>(v);
      }
    }
    return true;
  }
  return false;
}

bool extract_f32_array(Array const& arr, Buffer& out) {
  if (arr.shape.size() != 1 || !arr.mask.is_null()) return false;
  auto n = arr.size();
  if (arr.type == "<f4" && !arr.strides.empty() && arr.strides[0] == sizeof(float)) {
    out = Buffer(reinterpret_cast<void const*>(arr.pointer), n * sizeof(float));
    return true;
  }
  if (arr.type == "<f8" && !arr.strides.empty() && arr.strides[0] == sizeof(double)) {
    out = Buffer(n * sizeof(float));
    auto const* src = reinterpret_cast<double const*>(arr.pointer);
    auto* dst = out.as<float>();
    for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
    return true;
  }
  return false;
}

void gpu_transpose_csr_to_csc(u32 rows, u32 cols, u32 nnz,
                              Buffer const& csr_row_ptr, Buffer const& csr_col_idx, Buffer const& csr_val,
                              Buffer& csc_col_ptr, Buffer& csc_row_idx, Buffer& csc_val) {
  csc_col_ptr = Buffer((static_cast<std::size_t>(cols) + 1) * sizeof(u32));
  csc_row_idx = Buffer(static_cast<std::size_t>(nnz) * sizeof(u32));
  csc_val = Buffer(static_cast<std::size_t>(nnz) * sizeof(float));
  if (cols == 0) {
    csc_col_ptr.zero();
    return;
  }
  if (nnz == 0) {
    csc_col_ptr.zero();
    return;
  }
  Buffer col_counts(static_cast<std::size_t>(cols) * sizeof(u32));
  col_counts.zero();
  Buffer cursors(static_cast<std::size_t>(cols) * sizeof(u32));
  Buffer max_col_len(sizeof(u32));
  max_col_len.zero();
  SparseParams sp{rows, cols, nnz, 0, kNaN, 0, 0, 1};
  std::vector<Buffer> keepalive;
  Command cmd("CSR to CSC transpose");
  cmd.dispatch("csr_to_csc_counts", {csr_col_idx, col_counts}, sp, {div_up(nnz, 256), 1, 1});
  prefix_sum_u32(cmd, col_counts, csc_col_ptr, cols, keepalive);
  cmd.dispatch("csr_to_csc_init_cursors", {csc_col_ptr, cursors, max_col_len}, cols, {div_up(cols, 256), 1, 1});
  if (rows > 0)
    cmd.dispatch("csr_to_csc_scatter", {csr_row_ptr, csr_col_idx, csr_val, cursors, csc_row_idx, csc_val}, sp,
                 {div_up(rows, 256), 1, 1});
  cmd.dispatch("csc_sort_small_columns", {csc_col_ptr, csc_row_idx, csc_val}, cols, {cols, 1, 1});
  cmd.wait();

  u32 max_len = *max_col_len.as<u32>();
  if (max_len > 1024) {
    auto const* col_ptr = csc_col_ptr.as<u32>();
    Buffer tmp_row(static_cast<std::size_t>(max_len) * sizeof(u32));
    Buffer tmp_val(static_cast<std::size_t>(max_len) * sizeof(float));
    Command large_cmd("CSC sort large columns");
    for (u32 c = 0; c < cols; ++c) {
      u32 begin = col_ptr[c], len = col_ptr[c + 1] - begin;
      if (len <= 1024) continue;
      struct MergeP { u32 begin, len, width, pad; } mp{begin, len, 256, 0};
      large_cmd.dispatch("csc_sort_blocks_256", {csc_row_idx, csc_val}, mp, {div_up(len, 256), 1, 1});
      for (u32 w = 256; w < len; w <<= 1) {
        mp.width = w;
        large_cmd.dispatch("csc_merge_pass", {csc_row_idx, csc_val, tmp_row, tmp_val}, mp, {div_up(len, 256), 1, 1});
        large_cmd.dispatch("csc_copy_back", {tmp_row, tmp_val, csc_row_idx, csc_val}, mp, {div_up(len, 256), 1, 1});
      }
    }
    large_cmd.wait();
  }
}

std::shared_ptr<SparsePage> build_raw_csr_page_with_dups(Array const& indptr, Array const& indices,
                                                         Array const& values, std::size_t nr, std::size_t nc,
                                                         std::size_t dimension, bool csc, float missing) {
  auto page = std::make_shared<SparsePage>();
  page->offsets.resize(nr + 1);
  page->entries.reserve(values.size());
  std::vector<u32> entry_rows;
  if (csc) entry_rows.reserve(values.size());
  for (std::size_t r = 0; r + 1 < indptr.size(); ++r) {
    auto begin = static_cast<std::size_t>(indptr.at(r)), end = static_cast<std::size_t>(indptr.at(r + 1));
    for (std::size_t i = begin; i < end; ++i) {
      auto col = static_cast<std::size_t>(indices.at(i));
      float value = static_cast<float>(values.at(i));
      if (!std::isnan(value) && value != missing) {
        page->entries.push_back({static_cast<u32>(csc ? r : col), value});
        if (csc) { ++page->offsets[col + 1]; entry_rows.push_back(static_cast<u32>(col)); }
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
  return page;
}
}  // namespace

std::shared_ptr<Matrix> Matrix::sparse(Json const& ip, Json const& ix, Json const& data,
                                       std::size_t dimension, bool csc, float missing,
                                       int sparse_mode) {
  Array indptr(ip), indices(ix), values(data);
  require(indptr.size() >= 1 && indices.size() == values.size(), "Invalid sparse matrix arrays.");
  std::size_t nr = csc ? dimension : indptr.size() - 1, nc = csc ? indptr.size() - 1 : dimension;
  require(nr <= UINT32_MAX && nc <= UINT32_MAX, "Sparse matrix exceeds dimension limits.");
  auto raw_nnz = values.size();
  bool eligible_sparse = sparse_mode == 1 ||
                         (sparse_mode < 0 && nr > 0 && nc > 0 && raw_nnz <= 0x7fffffffu &&
                          std::uint64_t(nr) * nc >= 16384 && nc >= 16 &&
                          raw_nnz < (std::uint64_t(nr) * nc) / 4);
  if (eligible_sparse && nr > 0 && nc > 0 && raw_nnz <= 0x7fffffffu) {
    u32 major = static_cast<u32>(csc ? nc : nr);
    u32 minor = static_cast<u32>(csc ? nr : nc);
    u32 nnz32 = static_cast<u32>(raw_nnz);
    Buffer in_ptr, in_idx, in_val;
    if (extract_u32_array(indptr, in_ptr, nnz32 + 1u, true, nnz32) &&
        extract_u32_array(indices, in_idx, minor, false, 0) &&
        extract_f32_array(values, in_val)) {
      Buffer row_counts(static_cast<std::size_t>(major) * sizeof(u32));
      Buffer row_dirty(static_cast<std::size_t>(major));
      Buffer flags(2 * sizeof(u32));
      flags.zero();
      Buffer out_ptr((static_cast<std::size_t>(major) + 1) * sizeof(u32));
      SparseParams sp{major, minor, nnz32, 0, missing, 0, 0, 1};
      std::vector<Buffer> keepalive;
      {
        Command cmd("canonicalize sparse input counts");
        cmd.dispatch("sparse_count_csr", {in_ptr, in_idx, in_val, row_counts, row_dirty, flags}, sp,
                     {div_up(major, 256), 1, 1});
        prefix_sum_u32(cmd, row_counts, out_ptr, major, keepalive);
        cmd.wait();
      }
      u32 err = flags.as<u32>()[0];
      require(!(err & 1u), "Invalid sparse indptr.");
      require(!(err & 2u), "Sparse index is out of bounds.");
      require(!(err & 4u), "Input data contains infinity or a value too large.");
      u32 valid_nnz = out_ptr.as<u32>()[major];
      Buffer out_idx(static_cast<std::size_t>(valid_nnz) * sizeof(u32));
      Buffer out_val(static_cast<std::size_t>(valid_nnz) * sizeof(float));
      bool had_dups = false;
      if (valid_nnz > 0) {
        Buffer has_dups(sizeof(u32));
        has_dups.zero();
        Buffer orig_pos, new_counts;
        Command cmd("compact sparse input");
        cmd.dispatch("sparse_compact_csr", {in_ptr, in_idx, in_val, out_ptr, out_idx, out_val}, sp,
                     {div_up(major, 256), 1, 1});
        if (flags.as<u32>()[1] != 0) {
          orig_pos = Buffer(static_cast<std::size_t>(valid_nnz) * sizeof(u32));
          new_counts = Buffer(static_cast<std::size_t>(major) * sizeof(u32));
          cmd.dispatch("sparse_sort_dedup_rows",
                       {out_ptr, row_dirty, out_idx, out_val, orig_pos, new_counts, has_dups}, sp,
                       {div_up(major, 256), 1, 1});
        }
        cmd.wait();
        if (flags.as<u32>()[1] != 0 && *has_dups.as<u32>() != 0) {
          had_dups = true;
          Buffer dedup_ptr((static_cast<std::size_t>(major) + 1) * sizeof(u32));
          std::vector<Buffer> keep2;
          {
            Command pcmd("deduplicate sparse offsets");
            prefix_sum_u32(pcmd, new_counts, dedup_ptr, major, keep2);
            pcmd.wait();
          }
          u32 dedup_nnz = dedup_ptr.as<u32>()[major];
          Buffer dedup_idx(static_cast<std::size_t>(dedup_nnz) * sizeof(u32));
          Buffer dedup_val(static_cast<std::size_t>(dedup_nnz) * sizeof(float));
          if (dedup_nnz > 0) {
            Command rcmd("recompact deduplicated sparse");
            rcmd.dispatch("sparse_recompact_csr", {out_ptr, out_idx, out_val, dedup_ptr, dedup_idx, dedup_val}, sp,
                          {div_up(major, 256), 1, 1});
            rcmd.wait();
          }
          out_ptr = std::move(dedup_ptr);
          out_idx = std::move(dedup_idx);
          out_val = std::move(dedup_val);
          valid_nnz = dedup_nnz;
        }
      }

      auto storage = std::make_shared<SparseStorage>();
      storage->nnz = valid_nnz;
      if (!csc) {
        storage->csr_row_ptr = std::move(out_ptr);
        storage->csr_col_idx = std::move(out_idx);
        storage->csr_val = std::move(out_val);
      } else {
        storage->csc_col_ptr = std::move(out_ptr);
        storage->csc_row_idx = std::move(out_idx);
        storage->csc_val = std::move(out_val);
        gpu_transpose_csr_to_csc(major, minor, valid_nnz,
                                 storage->csc_col_ptr, storage->csc_row_idx, storage->csc_val,
                                 storage->csr_row_ptr, storage->csr_col_idx, storage->csr_val);
      }
      auto result = std::make_shared<Matrix>();
      result->data = {};
      result->rows = static_cast<u32>(nr);
      result->cols = static_cast<u32>(nc);
      result->sparse_ = std::move(storage);
      if (had_dups) {
        result->sparse_page = build_raw_csr_page_with_dups(indptr, indices, values, nr, nc, dimension, csc, missing);
      }
      result->refresh_metadata();
      return result;
    }
  }

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

std::shared_ptr<Matrix> Matrix::from_coo(Json const& row_desc, Json const& col_desc, Json const& val_desc,
                                         std::size_t nr, std::size_t nc, float missing, int sparse_mode) {
  Array row_arr(row_desc), col_arr(col_desc), val_arr(val_desc);
  require(row_arr.size() == col_arr.size() && row_arr.size() == val_arr.size(), "Invalid COO sparse matrix arrays.");
  require(nr <= UINT32_MAX && nc <= UINT32_MAX, "Sparse matrix exceeds dimension limits.");
  auto raw_nnz = val_arr.size();
  bool eligible_sparse = sparse_mode == 1 ||
                         (sparse_mode < 0 && nr > 0 && nc > 0 && raw_nnz <= 0x7fffffffu &&
                          std::uint64_t(nr) * nc >= 16384 && nc >= 16 &&
                          raw_nnz < (std::uint64_t(nr) * nc) / 4);
  if (eligible_sparse && nr > 0 && nc > 0 && raw_nnz <= 0x7fffffffu) {
    u32 rows32 = static_cast<u32>(nr), cols32 = static_cast<u32>(nc), nnz32 = static_cast<u32>(raw_nnz);
    Buffer coo_r, coo_c, coo_v;
    if (extract_u32_array(row_arr, coo_r, rows32, false, 0) &&
        extract_u32_array(col_arr, coo_c, cols32, false, 0) &&
        extract_f32_array(val_arr, coo_v)) {
      Buffer row_counts(static_cast<std::size_t>(rows32) * sizeof(u32));
      row_counts.zero();
      Buffer flags(sizeof(u32));
      flags.zero();
      Buffer row_ptr((static_cast<std::size_t>(rows32) + 1) * sizeof(u32));
      SparseParams sp{rows32, cols32, nnz32, 0, missing, 0, 0, 1};
      std::vector<Buffer> keepalive;
      {
        Command cmd("COO row counts");
        if (nnz32 > 0)
          cmd.dispatch("sparse_coo_count_rows", {coo_r, coo_c, coo_v, row_counts, flags}, sp,
                       {div_up(nnz32, 256), 1, 1});
        prefix_sum_u32(cmd, row_counts, row_ptr, rows32, keepalive);
        cmd.wait();
      }
      require(!(*flags.as<u32>() & 2u), "Sparse index is out of bounds.");
      u32 valid_nnz = row_ptr.as<u32>()[rows32];
      Buffer csr_col(static_cast<std::size_t>(valid_nnz) * sizeof(u32));
      Buffer csr_val(static_cast<std::size_t>(valid_nnz) * sizeof(float));
      if (valid_nnz > 0) {
        Buffer cursors(row_ptr.data(), static_cast<std::size_t>(rows32) * sizeof(u32));
        Buffer orig_pos(static_cast<std::size_t>(valid_nnz) * sizeof(u32));
        Buffer new_counts(static_cast<std::size_t>(rows32) * sizeof(u32));
        Buffer has_dups(sizeof(u32));
        has_dups.zero();
        Command cmd("COO scatter and canonicalize CSR");
        cmd.dispatch("sparse_coo_scatter_csr", {coo_r, coo_c, coo_v, cursors, csr_col, csr_val, orig_pos}, sp,
                     {div_up(nnz32, 256), 1, 1});
        cmd.dispatch("sparse_coo_sort_dedup_rows", {row_ptr, csr_col, csr_val, orig_pos, new_counts, has_dups}, sp,
                     {div_up(rows32, 256), 1, 1});
        cmd.wait();
        u32 dup_flags = *has_dups.as<u32>();
        require(!(dup_flags & 4u), "Input data contains infinity or a value too large.");
        if (dup_flags & 1u) {
          Buffer dedup_ptr((static_cast<std::size_t>(rows32) + 1) * sizeof(u32));
          std::vector<Buffer> keep2;
          {
            Command pcmd("COO dedup prefix");
            prefix_sum_u32(pcmd, new_counts, dedup_ptr, rows32, keep2);
            pcmd.wait();
          }
          u32 dedup_nnz = dedup_ptr.as<u32>()[rows32];
          Buffer dedup_col(static_cast<std::size_t>(dedup_nnz) * sizeof(u32));
          Buffer dedup_val(static_cast<std::size_t>(dedup_nnz) * sizeof(float));
          if (dedup_nnz > 0) {
            Command rcmd("COO recompact CSR");
            rcmd.dispatch("sparse_recompact_csr", {row_ptr, csr_col, csr_val, dedup_ptr, dedup_col, dedup_val}, sp,
                          {div_up(rows32, 256), 1, 1});
            rcmd.wait();
          }
          row_ptr = std::move(dedup_ptr);
          csr_col = std::move(dedup_col);
          csr_val = std::move(dedup_val);
          valid_nnz = dedup_nnz;
        }
      }
      auto storage = std::make_shared<SparseStorage>();
      storage->nnz = valid_nnz;
      storage->csr_row_ptr = std::move(row_ptr);
      storage->csr_col_idx = std::move(csr_col);
      storage->csr_val = std::move(csr_val);
      auto result = std::make_shared<Matrix>();
      result->data = {};
      result->rows = rows32;
      result->cols = cols32;
      result->sparse_ = std::move(storage);
      result->refresh_metadata();
      return result;
    }
  }
  // Dense fallback for small / high-density COO inputs (summing duplicate coordinates like COO->CSR).
  std::vector<float> dense(checked_product(nr, nc), kNaN);
  std::vector<std::uint8_t> present(checked_product(nr, nc), 0);
  for (std::size_t i = 0; i < raw_nnz; ++i) {
    double r = row_arr.at(i), c = col_arr.at(i);
    require(r >= 0 && r < nr && std::floor(r) == r && c >= 0 && c < nc && std::floor(c) == c,
            "Sparse index is out of bounds.");
    float v = static_cast<float>(val_arr.at(i));
    auto idx = static_cast<std::size_t>(r) * nc + static_cast<std::size_t>(c);
    if (!present[idx]) {
      dense[idx] = v;
      present[idx] = 1;
    } else {
      dense[idx] += v;
    }
  }
  auto page = std::make_shared<SparsePage>();
  page->offsets.resize(nr + 1, 0);
  for (std::size_t r = 0; r < nr; ++r) {
    for (std::size_t c = 0; c < nc; ++c) {
      auto idx = r * nc + c;
      if (present[idx]) {
        float v = dense[idx];
        if (std::isnan(v) || v == missing) dense[idx] = kNaN;
        else page->entries.push_back({static_cast<u32>(c), v});
      }
    }
    page->offsets[r + 1] = page->entries.size();
  }
  auto result = std::make_shared<Matrix>(nr, nc, std::move(dense), missing);
  result->sparse_page = std::move(page);
  return result;
}

Buffer Matrix::dense_data() const {
  if (data) return data;
  if (!is_sparse()) return data;
  std::lock_guard lock(recode_mutex_);
  if (!materialized_dense_) {
    auto total = static_cast<std::uint64_t>(checked_product(rows, cols));
    materialized_dense_ = Buffer(checked_product(total, sizeof(float)));
    if (total > 0) {
      SparseParams sp{rows, cols, sparse_->nnz, 0, kNaN, 0, 0, 1};
      Command cmd("materialize sparse matrix");
      cmd.dispatch("sparse_fill_nan", {materialized_dense_}, total, {div_up(total, 256), 1, 1});
      if (rows > 0 && sparse_->nnz > 0)
        cmd.dispatch("sparse_scatter_dense",
                     {sparse_->csr_row_ptr, sparse_->csr_col_idx, sparse_->csr_val, materialized_dense_},
                     sp, {div_up(rows, 256), 1, 1});
      cmd.wait();
    }
  }
  return materialized_dense_;
}

void Matrix::ensure_csc() const {
  if (!is_sparse()) return;
  std::lock_guard lock(recode_mutex_);
  if (sparse_->csc_col_ptr) return;
  gpu_transpose_csr_to_csc(rows, cols, sparse_->nnz,
                           sparse_->csr_row_ptr, sparse_->csr_col_idx, sparse_->csr_val,
                           sparse_->csc_col_ptr, sparse_->csc_row_idx, sparse_->csc_val);
}

void Matrix::ensure_dense_bins() {
  if (bins || !is_sparse() || !max_bin) return;
  bins = Buffer(checked_product(checked_product(rows, cols), histogram_bins <= 65534 ? sizeof(std::uint16_t) : sizeof(u32)));
  if (rows > 0 && cols > 0) {
    SparseParams sp{rows, cols, sparse_->nnz, histogram_bins, kNaN, 0, 0, 1};
    Command cmd("materialize dense bins from sparse");
    cmd.dispatch("sparse_fill_missing_bins", {bins}, sp, {div_up(rows, 256), cols, 1});
    if (sparse_->nnz > 0)
      cmd.dispatch("sparse_scatter_dense_bins",
                   {sparse_->csr_row_ptr, sparse_->csr_col_idx, sparse_->csr_bins, bins},
                   sp, {div_up(rows, 256), 1, 1});
    cmd.wait();
  }
}

SparsePage Matrix::raw_csr() const {
  require(!is_quantile, "Raw CSR is not available for QuantileDMatrix.");
  if (sparse_page) return *sparse_page;
  if (is_sparse()) {
    SparsePage page;
    page.offsets.resize(static_cast<std::size_t>(rows) + 1);
    auto const* rp = sparse_->csr_row_ptr.as<u32>();
    for (std::size_t r = 0; r <= rows; ++r) page.offsets[r] = rp[r];
    page.entries.resize(sparse_->nnz);
    auto const* ci = sparse_->csr_col_idx.as<u32>();
    auto const* cv = sparse_->csr_val.as<float>();
    for (u32 e = 0; e < sparse_->nnz; ++e) page.entries[e] = {ci[e], cv[e]};
    return page;
  }
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
  if (is_sparse()) {
    Buffer row_counts(static_cast<std::size_t>(rows) * sizeof(u32));
    Buffer row_dirty(static_cast<std::size_t>(rows));
    Buffer flags(2 * sizeof(u32));
    flags.zero();
    Buffer new_ptr((static_cast<std::size_t>(rows) + 1) * sizeof(u32));
    SparseParams sp{rows, cols, sparse_->nnz, 0, missing, 0, 0, 1};
    std::vector<Buffer> keepalive;
    {
      Command cmd("filter missing CSR");
      if (rows > 0)
        cmd.dispatch("sparse_count_csr",
                     {sparse_->csr_row_ptr, sparse_->csr_col_idx, sparse_->csr_val, row_counts, row_dirty, flags},
                     sp, {div_up(rows, 256), 1, 1});
      prefix_sum_u32(cmd, row_counts, new_ptr, rows, keepalive);
      cmd.wait();
    }
    u32 new_nnz = new_ptr.as<u32>()[rows];
    if (new_nnz < sparse_->nnz) {
      Buffer new_col(static_cast<std::size_t>(new_nnz) * sizeof(u32));
      Buffer new_val(static_cast<std::size_t>(new_nnz) * sizeof(float));
      if (new_nnz > 0) {
        Command cmd("compact filtered CSR");
        cmd.dispatch("sparse_compact_csr",
                     {sparse_->csr_row_ptr, sparse_->csr_col_idx, sparse_->csr_val, new_ptr, new_col, new_val},
                     sp, {div_up(rows, 256), 1, 1});
        cmd.wait();
      }
      auto updated = std::make_shared<SparseStorage>();
      updated->nnz = new_nnz;
      updated->csr_row_ptr = std::move(new_ptr);
      updated->csr_col_idx = std::move(new_col);
      updated->csr_val = std::move(new_val);
      sparse_ = std::move(updated);
      max_bin = 0;
      bins = {};
      materialized_dense_ = {};
    }
  } else {
    for (std::size_t i = 0; i < static_cast<std::size_t>(rows) * cols; ++i)
      if (data.as<float>()[i] == missing) data.as<float>()[i] = kNaN;
  }
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
  ++generation;
  slice_parent_.reset(); slice_indices_ = {};
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
  if (w.empty()) {
    weights = Buffer(static_cast<std::size_t>(rows) * sizeof(float));
    std::fill_n(weights.as<float>(), rows, 1.0f);
  }
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
  if (max_bin == requested_bins && (bins || (is_sparse() && sparse_->csr_bins))) return;
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
      if (is_sparse()) {
        if (sparse_->nnz > 0) {
          SparseParams sp{rows, cols, sparse_->nnz, 0, kNaN, 0, 0, 1};
          statistics.dispatch("category_statistics_csr",
                              {sparse_->csr_col_idx, sparse_->csr_val, categorical, cut_counts, invalid}, sp,
                              {div_up(sparse_->nnz, 256), 1, 1});
        }
      } else {
        statistics.dispatch("category_statistics", {data, categorical, cut_counts, invalid}, p,
                            {div_up(rows, 256), cols, 1});
      }
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
  auto bin_elem_bytes = histogram_bins <= 65534 ? sizeof(std::uint16_t) : sizeof(u32);
  if (is_sparse()) {
    bins = {};
    auto slice_parent = slice_parent_.lock();
    bool pre_sliced_bins = reference && slice_parent.get() == reference &&
                           reference->generation == slice_parent_generation_ &&
                           static_cast<bool>(sparse_->csr_bins);
    if (!pre_sliced_bins) {
      sparse_->csr_bins = Buffer(checked_product(sparse_->nnz, bin_elem_bytes));
      if (sparse_->nnz > 0) {
        SparseParams sp{rows, cols, sparse_->nnz, histogram_bins, kNaN, 0, 0, 1};
        Command command("quantize sparse CSR");
        command.dispatch("sparse_quantize_csr",
                         {sparse_->csr_col_idx, sparse_->csr_val, cuts, cut_counts, sparse_->csr_bins, categorical},
                         sp, {div_up(sparse_->nnz, 256), 1, 1});
        quantize_seconds += command.wait(); ++quantize_commands;
      }
    }
    quantized_dense = (static_cast<std::uint64_t>(sparse_->nnz) == static_cast<std::uint64_t>(rows) * cols) && rows > 0 && cols > 0;
  } else {
    bins = Buffer(checked_product(checked_product(rows, cols), bin_elem_bytes));
    Buffer missing(sizeof(u32)); missing.zero();
    auto slice_parent = slice_parent_.lock();
    if (reference && slice_parent.get() == reference &&
        reference->generation == slice_parent_generation_ &&
        reference->bins && slice_indices_ && rows > 0 && cols > 0) {
      SliceParams sp{rows, reference->rows, cols, histogram_bins};
      Command command("slice quantized bins");
      command.dispatch(histogram_bins <= 65534 ? "slice_bins_u16" : "slice_bins_u32",
                       {reference->bins, slice_indices_, bins, missing}, sp, {div_up(rows, 256), cols, 1});
      quantize_seconds += command.wait(); ++quantize_commands;
    } else {
      Command command("quantize");
      command.dispatch("quantize", {data, cuts, cut_counts, bins, categorical, missing}, p, {div_up(rows, 256), cols, 1});
      quantize_seconds += command.wait(); ++quantize_commands;
    }
    quantized_dense = !missing.as<u32>()[0];
  }
  max_bin = requested_bins;
  quantized_types_changed_ = reference && reference->quantized_types_changed_;
  partition_data_ = {}; partition_bounds_ = {};
  packed_bins_ = {}; packed_chunk_ = 0;
  export_cuts();
}
void Matrix::decode_quantiles(Buffer const& output, bool representative, bool numeric_only) const {
  if (is_sparse()) {
    QuantileValueParams p{sparse_->nnz, cols, histogram_bins, static_cast<u32>(representative), static_cast<u32>(numeric_only)};
    if (sparse_->nnz > 0) {
      Command command("reconstruct sparse quantile values");
      command.dispatch("sparse_decode_quantiles_csr",
                       {sparse_->csr_col_idx, sparse_->csr_bins, cuts, categorical, output}, p,
                       {div_up(sparse_->nnz, 256), 1, 1});
      command.wait();
    }
    return;
  }
  QuantileValueParams p{rows, cols, histogram_bins, static_cast<u32>(representative), static_cast<u32>(numeric_only)};
  Command command("reconstruct quantile values");
  command.dispatch("quantile_values", {bins, cuts, categorical, output}, p,
                   {div_up(rows, 32), div_up(cols, 32), 1}, {32, 8, 1});
  command.wait();
}
void Matrix::finish_quantile() {
  require(max_bin && (bins || (is_sparse() && sparse_->csr_bins)), "QuantileDMatrix must be quantized before reconstruction.");
  is_quantile = true;
  sparse_page.reset();
  if (is_sparse()) {
    decode_quantiles(sparse_->csr_val, false);
    std::lock_guard lock(recode_mutex_);
    sparse_->csc_col_ptr = {}; sparse_->csc_row_idx = {}; sparse_->csc_val = {};
    materialized_dense_ = {};
    recoded_data_ = {}; recoded_categories_ = nullptr;
    return;
  }
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
  if (is_quantile) {
    if (is_sparse()) {
      decode_quantiles(sparse_->csr_val, false);
      sparse_->csc_col_ptr = {}; sparse_->csc_row_idx = {}; sparse_->csc_val = {};
    } else decode_quantiles(data, false);
  }
  ++generation;
  slice_parent_.reset(); slice_indices_ = {};
  std::lock_guard lock(recode_mutex_);
  materialized_dense_ = {};
  recoded_data_ = {}; recoded_categories_ = nullptr;
  partition_data_ = {}; partition_bounds_ = {};
}
Buffer Matrix::export_data() const {
  if (!is_quantile) return dense_data();
  if (is_sparse()) {
    auto total = static_cast<std::uint64_t>(checked_product(rows, cols));
    Buffer output(checked_product(total, sizeof(float)));
    if (total > 0) {
      Buffer rep_val(checked_product(sparse_->nnz, sizeof(float)));
      decode_quantiles(rep_val, true);
      SparseParams sp{rows, cols, sparse_->nnz, 0, kNaN, 0, 0, 1};
      Command cmd("export sparse quantile matrix");
      cmd.dispatch("sparse_fill_nan", {output}, total, {div_up(total, 256), 1, 1});
      if (rows > 0 && sparse_->nnz > 0)
        cmd.dispatch("sparse_scatter_dense",
                     {sparse_->csr_row_ptr, sparse_->csr_col_idx, rep_val, output},
                     sp, {div_up(rows, 256), 1, 1});
      cmd.wait();
    }
    return output;
  }
  Buffer output(data.size());
  decode_quantiles(output, true);
  return output;
}
Buffer Matrix::partition_data() const {
  if (is_sparse()) {
    if (!quantized_types_changed_ || (is_quantile && cuts_categorical_)) return sparse_->csr_val;
    std::lock_guard lock(recode_mutex_);
    if (!partition_data_) {
      partition_data_ = Buffer(checked_product(sparse_->nnz, sizeof(float)));
      decode_quantiles(partition_data_, false, !cuts_categorical_);
    }
    return partition_data_;
  }
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
  auto result = std::make_shared<Matrix>();
  result->rows = indices.size(); result->cols = cols;
  if (is_sparse()) {
    for (auto idx : indices)
      require(idx >= 0 && static_cast<u32>(idx) < rows, "DMatrix slice index out of bounds.");
    auto sliced = std::make_shared<SparseStorage>();
    u32 srows = static_cast<u32>(indices.size());
    sliced->csr_row_ptr = Buffer((static_cast<std::size_t>(srows) + 1) * sizeof(u32));
    if (srows > 0 && cols > 0) {
      Buffer idx_buf(indices.data(), indices.size() * sizeof(int));
      Buffer counts(static_cast<std::size_t>(srows) * sizeof(u32));
      std::vector<Buffer> keepalive;
      {
        Command cmd("slice sparse CSR counts");
        cmd.dispatch("sparse_slice_counts", {sparse_->csr_row_ptr, idx_buf, counts}, srows,
                     {div_up(srows, 256), 1, 1});
        prefix_sum_u32(cmd, counts, sliced->csr_row_ptr, srows, keepalive);
        cmd.wait();
      }
      u32 out_nnz = sliced->csr_row_ptr.as<u32>()[srows];
      sliced->nnz = out_nnz;
      sliced->csr_col_idx = Buffer(static_cast<std::size_t>(out_nnz) * sizeof(u32));
      sliced->csr_val = Buffer(static_cast<std::size_t>(out_nnz) * sizeof(float));
      bool copy_bins = max_bin != 0 && static_cast<bool>(sparse_->csr_bins);
      if (copy_bins) {
        auto bin_elem_bytes = histogram_bins <= 65534 ? sizeof(std::uint16_t) : sizeof(u32);
        sliced->csr_bins = Buffer(checked_product(out_nnz, bin_elem_bytes));
      }
      if (out_nnz > 0) {
        Buffer dummy(sizeof(u32));
        SparseParams sp{srows, cols, out_nnz, histogram_bins, kNaN, copy_bins ? 1u : 0u, 0, 1};
        Command scmd("slice sparse CSR scatter");
        scmd.dispatch("sparse_slice_scatter",
                      {sparse_->csr_row_ptr, sparse_->csr_col_idx, sparse_->csr_val,
                       copy_bins ? sparse_->csr_bins : dummy, idx_buf,
                       sliced->csr_row_ptr, sliced->csr_col_idx, sliced->csr_val,
                       copy_bins ? sliced->csr_bins : dummy},
                      sp, {div_up(srows, 256), 1, 1});
        scmd.wait();
      }
      result->slice_parent_ = shared_from_this();
      result->slice_indices_ = std::move(idx_buf);
      result->slice_parent_generation_ = generation;
    } else {
      for (auto idx : indices)
        require(idx >= 0 && static_cast<u32>(idx) < rows, "DMatrix slice index out of bounds.");
      sliced->csr_row_ptr.zero();
    }
    result->data = {};
    result->sparse_ = std::move(sliced);
  } else if (!indices.empty() && cols > 0) {
    Buffer idx_buf(indices.data(), indices.size() * sizeof(int));
    result->data = Buffer(checked_product(indices.size(), cols) * sizeof(float));
    Buffer invalid(sizeof(u32)); invalid.zero();
    SliceParams sp{static_cast<u32>(indices.size()), rows, cols, 0};
    Command command("slice rows");
    command.dispatch("slice_rows_dense", {data, idx_buf, result->data, invalid}, sp,
                     {div_up(cols, 16), div_up(indices.size(), 16), 1}, {16, 16, 1});
    command.wait();
    require(!invalid.as<u32>()[0], "DMatrix slice index out of bounds.");
    result->slice_parent_ = shared_from_this();
    result->slice_indices_ = std::move(idx_buf);
    result->slice_parent_generation_ = generation;
  } else {
    for (auto idx : indices)
      require(idx >= 0 && static_cast<u32>(idx) < rows, "DMatrix slice index out of bounds.");
    result->data = Buffer(0);
  }
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
  bool can_append_sparse = batch.is_sparse() && (rows == 0 || (is_sparse() && categories == batch.categories)) &&
                           ((rows == 0 ? 0ull : std::uint64_t(sparse_->nnz)) + batch.sparse_->nnz <= 0x7fffffffu);
  if (can_append_sparse) {
    u32 old_nnz = rows == 0 ? 0u : sparse_->nnz;
    u32 add_nnz = batch.sparse_->nnz;
    u32 tot_nnz = old_nnz + add_nnz;
    auto merged = std::make_shared<SparseStorage>();
    merged->nnz = tot_nnz;
    merged->csr_row_ptr = Buffer((static_cast<std::size_t>(new_rows) + 1) * sizeof(u32));
    merged->csr_col_idx = Buffer(static_cast<std::size_t>(tot_nnz) * sizeof(u32));
    merged->csr_val = Buffer(static_cast<std::size_t>(tot_nnz) * sizeof(float));
    auto* dst_rp = merged->csr_row_ptr.as<u32>();
    if (rows > 0) {
      std::copy_n(sparse_->csr_row_ptr.as<u32>(), rows + 1, dst_rp);
      if (old_nnz) {
        std::copy_n(sparse_->csr_col_idx.as<u32>(), old_nnz, merged->csr_col_idx.as<u32>());
        std::copy_n(sparse_->csr_val.as<float>(), old_nnz, merged->csr_val.as<float>());
      }
    } else {
      dst_rp[0] = 0;
    }
    auto const* add_rp = batch.sparse_->csr_row_ptr.as<u32>();
    for (u32 r = 1; r <= batch.rows; ++r) dst_rp[rows + r] = old_nnz + add_rp[r];
    if (add_nnz) {
      std::copy_n(batch.sparse_->csr_col_idx.as<u32>(), add_nnz, merged->csr_col_idx.as<u32>() + old_nnz);
      std::copy_n(batch.sparse_->csr_val.as<float>(), add_nnz, merged->csr_val.as<float>() + old_nnz);
    }
    if (sparse_page || batch.sparse_page) {
      auto page = rows ? raw_csr() : SparsePage{{0}, {}};
      auto incoming = batch.raw_csr();
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
    data = {};
    sparse_ = std::move(merged);
    materialized_dense_ = {};
    feature_names = batch.feature_names; feature_types = batch.feature_types;
    max_bin = 0; bins = {}; ++generation; refresh_metadata();
    slice_parent_.reset(); slice_indices_ = {};
    packed_bins_ = {}; packed_chunk_ = 0;
    return;
  }
  std::vector<float> values(checked_product(new_rows, batch.cols));
  if (rows) std::copy_n(dense_data().as<float>(), static_cast<std::size_t>(rows) * cols, values.data());
  auto batch_data = rows ? batch.data_for_categories(categories) : batch.dense_data();
  if (!batch_data) batch_data = batch.dense_data();
  std::copy_n(batch_data.as<float>(), static_cast<std::size_t>(batch.rows) * batch.cols, values.data() + static_cast<std::size_t>(rows) * batch.cols);
  if (sparse_page || batch.sparse_page || is_sparse() || batch.is_sparse()) {
    auto page = rows ? raw_csr() : SparsePage{{0}, {}};
    auto incoming = batch.raw_csr();
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
  sparse_.reset(); materialized_dense_ = {};
  data = Buffer(values.data(), values.size() * sizeof(float));
  feature_names = batch.feature_names; feature_types = batch.feature_types;
  max_bin = 0; ++generation; refresh_metadata();
  slice_parent_.reset(); slice_indices_ = {};
  packed_bins_ = {}; packed_chunk_ = 0;
}
}  // namespace metalgb
