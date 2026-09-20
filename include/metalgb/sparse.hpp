#pragma once
#include <cstdint>
#include <vector>

namespace metalgb {
struct SparseEntry {
  std::uint32_t feature;
  float value;
};
struct SparsePage {
  std::vector<std::uint64_t> offsets;
  std::vector<SparseEntry> entries;
};
}  // namespace metalgb
