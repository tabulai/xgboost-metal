#pragma once
#include "metalgb/runtime.hpp"
#include "metalgb/shared.h"

namespace metalgb {
inline Buffer cox_scan_estimates(Command& command, Buffer const& input, u32 rows) {
  auto blocks = static_cast<u32>(div_up(rows, kCoxScanRows));
  Buffer output(std::size_t(rows) * kCoxEstimateBytes), totals(std::size_t(blocks) * kCoxEstimateBytes);
  CoxScanParams p{rows, blocks, 0, 0};
  command.dispatch("cox_estimate_scan_tiles", {input, output, totals}, p, {blocks, 1, 1});
  if (blocks > 1) {
    auto scanned = cox_scan_estimates(command, totals, blocks);
    command.dispatch("cox_estimate_scan_offsets", {output, scanned}, p, {blocks, 1, 1});
  }
  return output;
}

inline Buffer cox_ordered_scan(Command& command, Buffer const& input, Buffer const& labels,
                                Buffer const& order, Buffer const& seed, u32 rows, bool segmented) {
  auto blocks = static_cast<u32>(div_up(rows, kCoxScanRows));
  CoxScanParams p{rows, blocks, static_cast<u32>(segmented), 0};
  Buffer output(std::size_t(rows) * 2 * sizeof(u64));
  if (!rows) return output;
  if (blocks == 1) {
    command.dispatch("cox_scan_replay", {input, labels, order, seed, output}, p,
                     {1, 1, 1}, {32, 1, 1});
    return output;
  }
  Buffer totals(std::size_t(blocks) * kCoxEstimateBytes);
  Buffer transforms(std::size_t(blocks) * 2 * kCoxTransformBytes);
  Buffer starts((std::size_t(blocks) + 1) * 2 * sizeof(u64));
  command.dispatch("cox_scan_estimates", {input, labels, order, totals}, p, {blocks, 1, 1});
  auto estimates = cox_scan_estimates(command, totals, blocks);
  command.dispatch("cox_scan_transforms", {input, labels, order, seed, estimates, transforms}, p,
                   {blocks, 1, 1}, {32, 1, 1});
  command.dispatch("cox_scan_carries", {input, labels, order, seed, transforms, starts}, p,
                   {1, 1, 1}, {32, 1, 1});
  command.dispatch("cox_scan_replay", {input, labels, order, starts, output}, p,
                   {blocks, 1, 1}, {32, 1, 1});
  return output;
}
}  // namespace metalgb
