// SimpleDMatrix wire format at XGBoost 6d2dbe19a099aa046178be8cfc303bfec1a623a6.
// Lengths are bounded by remaining file bytes before allocation or access.
#pragma once
#include "metalgb/sparse.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace metalgb::binary {
using U64 = std::uint64_t;
using U32 = std::uint32_t;
static_assert(std::endian::native == std::endian::little);
inline void check(bool valid, std::string const& message) {
  if (!valid) throw std::runtime_error("DMatrix binary: " + message);
}
inline U64 product(U64 a, U64 b) {
  check(!b || a <= std::numeric_limits<U64>::max() / b, "shape product overflows");
  return a * b;
}
using Entry = SparseEntry;
static_assert(sizeof(Entry) == 8 && offsetof(Entry, value) == 4);
static_assert(std::is_trivially_copyable_v<Entry>);
struct Tensor {
  std::array<U64, 2> shape{};
  std::vector<float> values;
};
struct Document {
  std::array<std::int32_t, 3> version{3, 5, 0};
  U64 declared_fields{13}, rows{}, columns{}, nonzero{};
  Tensor labels, margins;
  std::vector<U32> groups;
  std::vector<float> weights, lower_bounds, upper_bounds, feature_weights;
  std::vector<std::string> feature_names, feature_types;
  // UBJSON category dictionaries use typed arrays, including signed storage
  // for wider unsigned names. They are decoded independently of the wire fields.
  std::vector<char> categories;
  std::vector<U64> offsets;
  std::vector<Entry> entries;
  std::vector<char> trailing;
};

class Reader {
 public:
  explicit Reader(std::filesystem::path const& path) : stream_(path, std::ios::binary) {
    check(stream_.is_open(), "cannot open input file");
    stream_.seekg(0, std::ios::end);
    auto length = stream_.tellg();
    check(length >= 0, "cannot determine file length");
    remaining_ = static_cast<U64>(length);
    stream_.seekg(0);
    check(bool(stream_), "cannot seek input file");
  }
  U64 remaining() const { return remaining_; }
  void bytes(void* destination, U64 count) {
    check(count <= remaining_, "truncated payload");
    auto pointer = static_cast<char*>(destination);
    while (count) {
      auto chunk = static_cast<std::streamsize>(std::min<U64>(count, std::numeric_limits<std::streamsize>::max()));
      stream_.read(pointer, chunk);
      check(bool(stream_), "input read failed");
      pointer += chunk; count -= chunk; remaining_ -= chunk;
    }
  }
  template<class T> T number() {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{}; bytes(&value, sizeof(value)); return value;
  }
  std::string string() {
    auto count = number<U64>();
    check(count <= remaining_ && count <= std::string{}.max_size(), "invalid string length");
    std::string result(static_cast<std::size_t>(count), '\0');
    bytes(result.data(), count); return result;
  }
  template<class T> std::vector<T> vector() {
    static_assert(std::is_trivially_copyable_v<T>);
    auto count = number<U64>();
    check(count <= remaining_ / sizeof(T) && count <= std::vector<T>{}.max_size(), "invalid vector length");
    std::vector<T> result(static_cast<std::size_t>(count));
    bytes(result.data(), count * sizeof(T)); return result;
  }
  void field(std::string_view expected, unsigned type, bool scalar) {
    check(string() == expected, "unexpected field name, expected " + std::string(expected));
    check(number<std::uint8_t>() == type, "unexpected type for " + std::string(expected));
    check(number<std::uint8_t>() == static_cast<unsigned>(scalar), "unexpected scalar flag for " + std::string(expected));
  }
  U64 scalar(std::string_view name) {
    field(name, 4, true); return number<U64>();
  }
  std::array<U64, 2> shape() { return {number<U64>(), number<U64>()}; }
  Tensor tensor(std::string_view name) {
    field(name, 1, false);
    auto dimensions = shape();
    auto values = vector<float>();
    check(product(dimensions[0], dimensions[1]) == values.size(), "tensor shape/count mismatch");
    return {dimensions, std::move(values)};
  }
  template<class T> std::vector<T> vector_field(std::string_view name, unsigned type) {
    field(name, type, false);
    auto dimensions = shape();
    auto values = vector<T>();
    check(dimensions[1] == 1 && dimensions[0] == values.size(), "vector shape/count mismatch");
    return values;
  }
  std::vector<std::string> strings(std::string_view name) {
    field(name, 5, false);
    auto dimensions = shape();
    auto count = number<U64>();
    check(dimensions[1] == 1 && dimensions[0] == count, "string-vector shape/count mismatch");
    check(count <= remaining_ / sizeof(U64) && count <= std::vector<std::string>{}.max_size(), "invalid string-vector length");
    std::vector<std::string> result; result.reserve(static_cast<std::size_t>(count));
    for (U64 i = 0; i < count; ++i) result.push_back(string());
    return result;
  }
 private:
  std::ifstream stream_;
  U64 remaining_{};
};

inline void validate(Document const& document) {
  check(document.version[0] >= 3 && document.version[1] >= 1, "unsupported binary version");
  check(document.declared_fields >= 13, "insufficient metadata fields");
  check(document.rows < std::numeric_limits<U64>::max() && document.offsets.size() == document.rows + 1,
        "row-offset count does not match rows");
  check(!document.offsets.empty() && document.offsets.front() == 0 && document.offsets.back() == document.entries.size(),
        "invalid row-offset endpoints");
  check(std::is_sorted(document.offsets.begin(), document.offsets.end()), "row offsets are not monotone");
  check(document.nonzero == document.entries.size(), "nonmissing count does not match entries");
  for (auto entry : document.entries) check(entry.feature < document.columns, "feature index out of bounds");
  for (auto tensor : {&document.labels, &document.margins})
    check(product(tensor->shape[0], tensor->shape[1]) == tensor->values.size(), "tensor shape/count mismatch");
  // Sorted row indices and unique entries are deliberately not requirements.
  // The format itself can represent both, independently of native training.
}

inline Document read(std::filesystem::path const& path) {
  Reader input(path);
  check(input.number<U32>() == 0xffffab01u, "magic number mismatch");
  std::array<char, 8> marker{}; input.bytes(marker.data(), marker.size());
  check(std::string_view(marker.data(), marker.size()) == "version:", "version marker mismatch");
  Document document;
  for (auto& value : document.version) value = input.number<std::int32_t>();
  check(document.version[0] >= 3 && document.version[1] >= 1, "unsupported binary version");
  document.declared_fields = input.number<U64>();
  check(document.declared_fields >= 13, "insufficient metadata fields");
  // The pinned reader warns about a higher count but reads these 13 fields
  // immediately followed by CSR. Do not invent a skip format for extra fields.
  document.rows = input.scalar("num_row");
  document.columns = input.scalar("num_col");
  document.nonzero = input.scalar("num_nonzero");
  document.labels = input.tensor("labels");
  document.groups = input.vector_field<U32>("group_ptr", 3);
  document.weights = input.vector_field<float>("weights", 1);
  document.margins = input.tensor("base_margin");
  document.lower_bounds = input.vector_field<float>("labels_lower_bound", 1);
  document.upper_bounds = input.vector_field<float>("labels_upper_bound", 1);
  document.feature_names = input.strings("feature_names");
  document.feature_types = input.strings("feature_types");
  document.feature_weights = input.vector_field<float>("feature_weights", 1);
  document.categories = input.vector_field<char>("cats", 5);
  document.offsets = input.vector<U64>();
  document.entries = input.vector<Entry>();
  check(input.remaining() <= document.trailing.max_size(), "trailing payload too large");
  document.trailing.resize(static_cast<std::size_t>(input.remaining()));
  input.bytes(document.trailing.data(), document.trailing.size());
  validate(document);
  return document;
}

class Writer {
 public:
  explicit Writer(std::filesystem::path const& path) : stream_(path, std::ios::binary | std::ios::trunc) {
    check(stream_.is_open(), "cannot open output file");
  }
  void bytes(void const* source, U64 count) {
    auto pointer = static_cast<char const*>(source);
    while (count) {
      auto chunk = static_cast<std::streamsize>(std::min<U64>(count, std::numeric_limits<std::streamsize>::max()));
      stream_.write(pointer, chunk); check(bool(stream_), "output write failed");
      pointer += chunk; count -= chunk;
    }
  }
  template<class T> void number(T value) {
    static_assert(std::is_trivially_copyable_v<T>); bytes(&value, sizeof(value));
  }
  void string(std::string_view value) { number<U64>(value.size()); bytes(value.data(), value.size()); }
  template<class T> void vector(std::vector<T> const& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    number<U64>(values.size()); bytes(values.data(), product(values.size(), sizeof(T)));
  }
  void field(std::string_view name, unsigned type, bool scalar) {
    string(name); number<std::uint8_t>(type); number<std::uint8_t>(scalar);
  }
  void scalar(std::string_view name, U64 value) { field(name, 4, true); number(value); }
  void tensor(std::string_view name, Tensor const& value) {
    field(name, 1, false); number(value.shape[0]); number(value.shape[1]); vector(value.values);
  }
  template<class T> void vector_field(std::string_view name, unsigned type, std::vector<T> const& values) {
    field(name, type, false); number<U64>(values.size()); number<U64>(1); vector(values);
  }
  void strings(std::string_view name, std::vector<std::string> const& values) {
    field(name, 5, false); number<U64>(values.size()); number<U64>(1); number<U64>(values.size());
    for (auto const& value : values) string(value);
  }
  void finish() { stream_.flush(); check(bool(stream_), "output flush failed"); }
 private:
  std::ofstream stream_;
};

inline void write(Document const& document, std::filesystem::path const& path) {
  validate(document);
  Writer output(path);
  output.number<U32>(0xffffab01u); output.bytes("version:", 8);
  for (auto value : document.version) output.number(value);
  output.number(document.declared_fields);
  output.scalar("num_row", document.rows); output.scalar("num_col", document.columns);
  output.scalar("num_nonzero", document.nonzero);
  output.tensor("labels", document.labels); output.vector_field("group_ptr", 3, document.groups);
  output.vector_field("weights", 1, document.weights); output.tensor("base_margin", document.margins);
  output.vector_field("labels_lower_bound", 1, document.lower_bounds);
  output.vector_field("labels_upper_bound", 1, document.upper_bounds);
  output.strings("feature_names", document.feature_names); output.strings("feature_types", document.feature_types);
  output.vector_field("feature_weights", 1, document.feature_weights);
  output.vector_field("cats", 5, document.categories);
  output.vector(document.offsets); output.vector(document.entries);
  output.bytes(document.trailing.data(), document.trailing.size()); output.finish();
}
}  // namespace metalgb::binary
