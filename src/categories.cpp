#include "metalgb/engine.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>

namespace metalgb {
namespace {
template<class T> Json integer_name(Array const& array, std::size_t i) {
  T value;
  auto source = reinterpret_cast<std::byte const*>(array.pointer) +
                static_cast<std::ptrdiff_t>(i) * array.strides[0];
  std::memcpy(&value, source, sizeof(T));
  return value;
}
bool has_categories(Json const& columns) {
  return std::any_of(columns.begin(), columns.end(), [](auto const& c) { return !c.is_null(); });
}
template<class T> Json export_names(Json const& names, std::vector<Buffer>& buffers, std::string dtype) {
  auto values = names.get<std::vector<T>>();
  buffers.emplace_back(values.data(), values.size() * sizeof(T));
  return array_interface(buffers.back().data(), {values.size()}, std::move(dtype));
}
}

Json category_column(Json const& descriptor) {
  Json names = Json::array();
  if (descriptor.contains("offsets")) {
    Array offsets(descriptor.at("offsets")), text(descriptor.at("values"));
    require(offsets.size() > 1, "Categorical feature must have at least one category.");
    auto chars = reinterpret_cast<char const*>(text.pointer);
    for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
      auto begin = offsets.at(i), end = offsets.at(i + 1);
      require(begin >= 0 && begin <= end && end <= text.size(), "Invalid string category offsets.");
      names.push_back(end == begin ? std::string{} : std::string(chars + static_cast<std::size_t>(begin), static_cast<std::size_t>(end - begin)));
    }
    return {{"type", 0}, {"values", std::move(names)}};
  }
  Array array(descriptor);
  require(array.shape.size() == 1 && (array.type[1] == 'i' || array.type[1] == 'u'),
          "Category names must be strings or integers; floating-point category names are not supported.");
  require(array.size() > 0, "Categorical feature must have at least one category.");
  auto width = std::stoul(array.type.substr(2));
  bool signed_type = array.type[1] == 'i';
  int type = (width == 1 ? 9 : width == 2 ? 11 : width == 4 ? 13 : 15) + !signed_type;
  for (std::size_t i = 0; i < array.size(); ++i) {
    if (signed_type) {
      if (width == 1) names.push_back(integer_name<std::int8_t>(array, i));
      if (width == 2) names.push_back(integer_name<std::int16_t>(array, i));
      if (width == 4) names.push_back(integer_name<std::int32_t>(array, i));
      if (width == 8) names.push_back(integer_name<std::int64_t>(array, i));
    } else {
      if (width == 1) names.push_back(integer_name<std::uint8_t>(array, i));
      if (width == 2) names.push_back(integer_name<std::uint16_t>(array, i));
      if (width == 4) names.push_back(integer_name<std::uint32_t>(array, i));
      if (width == 8) names.push_back(integer_name<std::uint64_t>(array, i));
    }
  }
  return {{"type", type}, {"values", std::move(names)}};
}

Json serialize_categories(Json const& categories) {
  Json encoded = Json::array(), segments = Json::array(), sorted = Json::array();
  if (has_categories(categories)) {
    std::size_t total = 0;
    segments.push_back(0);
    for (auto const& column : categories) {
      if (column.is_null()) {
        encoded.push_back({{"offsets", Json::array()}, {"values", Json::array()}});
        segments.push_back(total);
        continue;
      }
      auto const& names = column.at("values");
      if (column.at("type") == 0) {
        Json offsets = Json::array({0}), bytes = Json::array();
        for (auto const& name : names) {
          for (unsigned char ch : name.get_ref<std::string const&>()) bytes.push_back(static_cast<std::int8_t>(ch));
          offsets.push_back(bytes.size());
        }
        encoded.push_back({{"offsets", offsets}, {"values", bytes}});
      } else {
        if (column.at("type") == 16) for (auto const& name : names)
          require(name.get<std::uint64_t>() <= INT64_MAX, "Category index values must not exceed the signed 64-bit range.");
        encoded.push_back(column);
      }
      std::vector<u32> order(names.size());
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](u32 a, u32 b) {
        if (column.at("type") != 0) return names[a] < names[b];
        // XGBoost's category StringView compares signed UTF-8 bytes. Its saved
        // search index must use that order even though std::string uses unsigned.
        auto const& left = names[a].get_ref<std::string const&>();
        auto const& right = names[b].get_ref<std::string const&>();
        return std::lexicographical_compare(left.begin(), left.end(), right.begin(), right.end(),
                                            [](char x, char y) { return static_cast<std::int8_t>(x) < static_cast<std::int8_t>(y); });
      });
      for (auto i : order) sorted.push_back(i);
      total += names.size(); segments.push_back(total);
    }
  }
  return {{"enc", encoded}, {"feature_segments", segments}, {"sorted_idx", sorted}};
}

Json deserialize_categories(Json const& encoded) {
  Json categories = Json::array();
  for (auto const& column : encoded.at("enc")) {
    if (!column.contains("offsets")) {
      auto type = column.at("type").get<int>();
      require(type >= 9 && type <= 16, "Category names must use an integer or string dtype.");
      auto decoded = column;
      auto& names = decoded.at("values");
      // UBJSON stores wider unsigned arrays in same-width signed containers.
      // Restore their declared dtype before comparing or sorting category names.
      if (type == 12) names = names.get<std::vector<std::uint16_t>>();
      if (type == 14) names = names.get<std::vector<std::uint32_t>>();
      if (type == 16) names = names.get<std::vector<std::uint64_t>>();
      categories.push_back(std::move(decoded));
      continue;
    }
    auto const& offsets = column.at("offsets");
    if (offsets.empty()) { categories.push_back(nullptr); continue; }
    auto const& values = column.at("values");
    Json names = Json::array();
    for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
      auto begin = offsets[i].get<std::size_t>(), end = offsets[i + 1].get<std::size_t>();
      require(begin <= end && end <= values.size(), "Invalid model category string offsets.");
      std::string name;
      name.reserve(end - begin);
      for (auto j = begin; j < end; ++j) name.push_back(static_cast<char>(values[j].get<int>()));
      names.push_back(std::move(name));
    }
    categories.push_back({{"type", 0}, {"values", names}});
  }
  return categories;
}

Json CategoryStorage::array_interfaces() {
  Json result = Json::array();
  for (auto const& column : columns) {
    if (column.is_null() || column.at("values").empty()) { result.push_back(nullptr); continue; }
    auto const& names = column.at("values");
    switch (column.at("type").get<int>()) {
      case 0: {
        std::vector<std::int32_t> offsets{0};
        std::vector<std::int8_t> values;
        for (auto const& name : names) {
          auto const& text = name.get_ref<std::string const&>();
          values.insert(values.end(), text.begin(), text.end());
          require(values.size() <= INT32_MAX, "Category string storage exceeds the 32-bit offset range.");
          offsets.push_back(values.size());
        }
        buffers.emplace_back(offsets.data(), offsets.size() * sizeof(std::int32_t));
        auto offset_interface = array_interface(buffers.back().data(), {offsets.size()}, "<i4");
        buffers.emplace_back(values.data(), values.size());
        result.push_back({{"offsets", offset_interface}, {"values", array_interface(buffers.back().data(), {values.size()}, "|i1")}});
        break;
      }
      case 9: result.push_back(export_names<std::int8_t>(names, buffers, "|i1")); break;
      case 10: result.push_back(export_names<std::uint8_t>(names, buffers, "|u1")); break;
      case 11: result.push_back(export_names<std::int16_t>(names, buffers, "<i2")); break;
      case 12: result.push_back(export_names<std::uint16_t>(names, buffers, "<u2")); break;
      case 13: result.push_back(export_names<std::int32_t>(names, buffers, "<i4")); break;
      case 14: result.push_back(export_names<std::uint32_t>(names, buffers, "<u4")); break;
      case 15: result.push_back(export_names<std::int64_t>(names, buffers, "<i8")); break;
      case 16: result.push_back(export_names<std::uint64_t>(names, buffers, "<u8")); break;
      default: throw std::invalid_argument("Invalid category name dtype.");
    }
  }
  return result;
}

namespace {
Buffer recode_values(Buffer const& data, u32 rows, u32 cols, Json const& categories,
                     Json const& reference, bool construction, bool column_major) {
  require(reference.size() == cols && categories.size() == cols, "Category metadata does not match the feature count.");
  std::vector<std::array<u32, 2>> offsets(cols, {UINT32_MAX, 0});
  std::vector<u32> mapping;
  for (u32 c = 0; c < cols; ++c) {
    auto const& source = categories[c];
    auto const& target = reference[c];
    require(source.is_null() == target.is_null(), "Categorical feature types do not match the training data.");
    if (source.is_null() || (!construction && source == target)) continue;
    require(source.at("type") == target.at("type"), "Category name dtypes do not match the training data.");
    auto const& names = source.at("values");
    auto const& target_names = target.at("values");
    std::map<Json, u32> codes;
    for (u32 i = 0; i < target_names.size(); ++i) codes.emplace(target_names[i], i);
    offsets[c] = {static_cast<u32>(mapping.size()), static_cast<u32>(names.size())};
    for (auto const& name : names) {
      auto it = codes.find(name);
      require(it != codes.end(), "Found a category not in the training data for feature " + std::to_string(c) + ": " + name.dump());
      mapping.push_back(it->second);
    }
  }
  Buffer offset_buffer(offsets.data(), offsets.size() * sizeof(offsets[0]));
  Buffer mapping_buffer(mapping.data(), mapping.size() * sizeof(u32));
  Buffer output(data.size()), invalid(sizeof(u32)); invalid.zero();
  CategoryRecodeParams p{rows, cols, static_cast<u32>(construction), static_cast<u32>(column_major)};
  Command command("recode categorical values");
  command.dispatch("recode_categories", {data, offset_buffer, mapping_buffer, output, invalid},
                   p, {div_up(rows, 256), cols, 1});
  command.wait();
  require(!invalid.as<u32>()[0], "Categorical input contains an invalid category code.");
  return output;
}
}

Buffer recode_category_input(Buffer const& values, u32 rows, u32 cols, Json const& categories,
                             Json const& reference, bool column_major) {
  return recode_values(values, rows, cols, categories, reference, true, column_major);
}

Buffer Matrix::data_for_categories(Json const& reference) const {
  if (!has_categories(reference) || !has_categories(categories) || reference == categories) return data;
  std::lock_guard lock(recode_mutex_);
  if (recoded_data_ && recoded_generation_ == generation && recoded_categories_ == reference) return recoded_data_;
  auto output = recode_values(data, rows, cols, categories, reference, false, false);
  recoded_categories_ = reference; recoded_generation_ = generation; recoded_data_ = output;
  return output;
}
}  // namespace metalgb
