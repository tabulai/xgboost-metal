#include "metalgb/engine.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>

namespace metalgb {
namespace {
// Only allocation/work metadata is computed here. Values, weights, ranks,
// pruning and cuts are calculated on Metal.
u32 sketch_budget(u32 bins, u32 count) {
  if (!count) return 1;
  double epsilon = 1.0 / std::min(bins, count);
  for (u32 level = 1; level < 32; ++level) {
    auto limit = std::min<std::uint64_t>(count, static_cast<std::uint64_t>(std::ceil(level / (epsilon / 2.0))) + 1);
    if ((std::uint64_t{1} << level) * limit >= count) return static_cast<u32>(limit);
  }
  return count;
}
}

void Matrix::make_quantile_cuts(MatrixParams const& p) {
  bool unit_weights = get_info("weight").empty();
  summary_group_size = summary_batch_count = 0;
  weighted_summary_stages = weighted_summary_batch_count = 0;
  bool sparse_csc = is_sparse();
  if (sparse_csc) ensure_csc();
  Buffer queue(checked_product(sparse_csc ? static_cast<std::size_t>(sparse_->nnz) : checked_product(p.stride, cols), sizeof(Pair)));
  Buffer statistics(checked_product(cols, sizeof(SketchStats)));
  if (sparse_csc) {
    statistics.zero();
    if (cols > 0 && sparse_->nnz > 0) {
      Buffer tile_counts(checked_product(cols, sizeof(u32)));
      Buffer col_tile_offsets(checked_product(static_cast<std::size_t>(cols) + 1, sizeof(u32)));
      std::vector<Buffer> keepalive;
      {
        Command tcmd("CSC quantile tile offsets");
        tcmd.dispatch("sketch_csc_tile_counts", {sparse_->csc_col_ptr, categorical, tile_counts}, cols,
                      {div_up(cols, 256), 1, 1});
        prefix_sum_u32(tcmd, tile_counts, col_tile_offsets, cols, keepalive);
        quantize_seconds += tcmd.wait(); ++quantize_commands;
      }
      u32 total_tiles = col_tile_offsets.as<u32>()[cols];
      if (total_tiles > 0) {
        Buffer tile_map(checked_product(total_tiles, 2 * sizeof(u32)));
        Buffer temp_values(checked_product(checked_product(total_tiles, 128), sizeof(Pair)));
        Buffer compact(queue.size());
        Buffer starts(checked_product(sparse_->nnz, sizeof(u32)));
        Buffer blocks(checked_product(total_tiles, sizeof(SketchInputBlock)));
        SketchCscParams input{rows, cols, sparse_->nnz, total_tiles, static_cast<u32>(unit_weights), 0, 0, 0};
        Command prepare("CSC quantile input queues");
        prepare.dispatch("sketch_csc_populate_tiles", {col_tile_offsets, tile_map}, cols,
                         {div_up(cols, 256), 1, 1});
        prepare.dispatch("sketch_csc_blocks",
                         {sparse_->csc_col_ptr, sparse_->csc_row_idx, sparse_->csc_val,
                          sketch_weights, tile_map, temp_values, blocks},
                         input, {div_up(total_tiles, 8), 1, 1});
        prepare.dispatch("sketch_csc_offsets", {col_tile_offsets, blocks, statistics}, input,
                         {div_up(cols, 64), 1, 1}, {64, 1, 1});
        prepare.dispatch("sketch_csc_compact",
                         {sparse_->csc_col_ptr, tile_map, temp_values, blocks, compact, starts},
                         input, {div_up(total_tiles, 8), 1, 1});
        prepare.dispatch("sketch_csc_runs",
                         {sparse_->csc_col_ptr, col_tile_offsets, tile_map, compact, starts, statistics, queue},
                         input, {div_up(static_cast<std::size_t>(total_tiles) * 128, 256), 1, 1});
        quantize_seconds += prepare.wait(); ++quantize_commands;
      }
    }
  } else {
    auto tiles = div_up(rows, 128);
    Buffer compact(queue.size());
    Buffer starts(checked_product(checked_product(p.stride, cols), sizeof(u32)));
    Buffer blocks(checked_product(checked_product(tiles, cols), sizeof(SketchInputBlock)));
    SketchInputParams input{rows, cols, p.stride, static_cast<u32>(unit_weights)};
    Command prepare("quantile input queues");
    prepare.dispatch("sketch_input_blocks", {data, sketch_weights, categorical, queue, blocks}, input,
                     {tiles, div_up(cols, 8), 1});
    prepare.dispatch("sketch_input_offsets", {blocks, statistics}, input,
                     {div_up(cols, 64), 1, 1}, {64, 1, 1});
    prepare.dispatch("sketch_input_compact", {queue, blocks, compact, starts}, input,
                     {tiles, div_up(cols, 8), 1});
    prepare.dispatch("sketch_input_runs", {compact, starts, statistics, queue}, input,
                     {div_up(rows, 256), cols, 1});
    quantize_seconds += prepare.wait(); ++quantize_commands;
  }

  std::vector<SketchBatch> batches;
  std::vector<SketchFeature> features;
  std::size_t slots = 0, level_slots = 0, work_slots = 0;
  u32 max_stride = 0, max_batch_size = 0, cut_capacity = 0;
  u32 max_integer_batch_size = 0, max_integer_prune_batch_size = 0;
  auto const* csc_offsets = sparse_csc ? sparse_->csc_col_ptr.as<u32>() : nullptr;
  for (u32 feature = 0; feature < cols; ++feature) {
    auto const& stat = statistics.as<SketchStats>()[feature];
    // Each retained summary has at most one entry per input run. This also
    // excludes categorical and empty columns from the numeric work bound.
    cut_capacity = std::max(cut_capacity, std::min(p.numeric_bins, stat.runs));
    u32 limit = sketch_budget(p.numeric_bins, stat.nonmissing);
    u32 budget = sketch_budget(p.numeric_bins, stat.elements);
    u32 first = static_cast<u32>(batches.size());
    std::uint64_t capacity = std::uint64_t{2} * limit;
    std::uint64_t col_base = sparse_csc ? static_cast<std::uint64_t>(csc_offsets[feature])
                                        : std::uint64_t{feature} * p.stride;
    for (std::uint64_t begin = 0; begin < stat.runs; begin += capacity) {
      auto size = static_cast<u32>(std::min<std::uint64_t>(capacity, stat.runs - begin));
      auto stride = static_cast<u32>(div_up(size, 256) * 256);
      u32 exact_integer_weights = unit_weights && stat.elements <= (1u << 24);
      batches.push_back({col_base + begin, slots, size, stride, limit, exact_integer_weights});
      slots += stride;
      max_stride = std::max(max_stride, stride);
      max_batch_size = std::max(max_batch_size, size);
      if (exact_integer_weights) {
        max_integer_batch_size = std::max(max_integer_batch_size, size);
        if (size > limit) max_integer_prune_batch_size = std::max(max_integer_prune_batch_size, size);
      }
    }
    u32 count = static_cast<u32>(batches.size()) - first;
    u32 levels = std::bit_width(std::max(1u, count)) + 1;
    features.push_back({level_slots, work_slots, limit, budget, first, count, levels, 0});
    level_slots += checked_product(limit, levels);
    work_slots += checked_product(limit, 4);
  }
  Buffer plans(batches.data(), checked_product(batches.size(), sizeof(SketchBatch)));
  Buffer columns(features.data(), checked_product(features.size(), sizeof(SketchFeature)));
  Buffer a(checked_product(slots, sizeof(Pair))), b(a.size());
  Buffer summaries(checked_product(slots, 4 * sizeof(float)));
  Buffer sizes(checked_product(batches.size(), sizeof(u32)));
  Buffer levels(checked_product(level_slots, 4 * sizeof(float)));
  Buffer workspace(checked_product(work_slots, 4 * sizeof(float)));
  cuts = Buffer(checked_product(checked_product(cols, histogram_bins), sizeof(float)));
  cut_minima = Buffer(checked_product(cols, sizeof(float)));
  cuts.zero();
  SketchParams s{static_cast<u32>(batches.size()), 0, cols, 0};
  Command command("weighted quantile summaries");
  if (!batches.empty()) {
    command.dispatch("sketch_sort_blocks", {queue, plans, a}, s,
                     {div_up(max_stride, 256), batches.size(), 1});
    for (u32 width = 256; width < max_stride; width *= 2) {
      s.width = width;
      command.dispatch("sketch_merge_sorted", {a, plans, b}, s,
                       {div_up(max_stride, 256), batches.size(), 1});
      std::swap(a, b);
    }
    auto summary_control = std::getenv("METALGB_SERIAL_SUMMARY");
    bool serial_summary = summary_control && std::string_view(summary_control) == "1";
    auto force_control = std::getenv("METALGB_FORCE_PARALLEL_SUMMARY");
    bool force_summary = force_control && std::string_view(force_control) == "1";
    // Small and full-retention batches had faster isolated summaries but could
    // regress full training on M5 Max. Default to large batches with pruning
    // work; diagnostics can also exercise the smaller and unpruned cases.
    u32 minimum_summary_batch = force_summary ? 32u : 32768u;
    u32 summary_work = force_summary ? max_integer_batch_size : max_integer_prune_batch_size;
    if (!serial_summary && !unit_weights && max_batch_size >= 256) {
      // Separate tie repair from ordered formation. The size array first holds
      // repair flags, then summary counts; serial encoders finish all consumers
      // before it is reused. Keep tiny plans on the single-dispatch path.
      weighted_summary_stages = 3;
      weighted_summary_batch_count = static_cast<u32>(batches.size());
      command.dispatch("sketch_weighted_detect", {a, plans, sizes}, s,
                       {batches.size(), 1, 1}, {256, 1, 1});
      command.dispatch("sketch_weighted_repair", {a, plans, sizes, queue}, s,
                       {batches.size(), 1, 1}, {1, 1, 1});
      command.dispatch("sketch_weighted_form", {a, plans, summaries, sizes}, s,
                       {div_up(batches.size(), 64), 1, 1}, {64, 1, 1});
    } else if (!serial_summary && summary_work >= minimum_summary_batch) {
      summary_group_size = 256;
      summary_batch_count = static_cast<u32>(batches.size());
      // Both sort buffers are dead after entry formation. Reuse them as two
      // float2 halves of prune scratch, including batches just above capacity.
      auto summary_params = s;
      summary_params.reserved = !force_summary;
      command.dispatch("sketch_summarize_parallel", {a, plans, summaries, sizes, queue, b}, summary_params,
                       {batches.size(), 1, 1});
      if (!force_summary) {
        // Grouping can remove the planned pruning work. The GPU marks those
        // batches in the size buffer and restores the old per-thread mapping.
        // Serial encoders complete both passes before reduction; no new wait.
        command.dispatch("sketch_summarize_pending", {a, plans, summaries, sizes, queue}, s,
                         {div_up(batches.size(), 64), 1, 1}, {64, 1, 1});
      }
    } else {
      command.dispatch("sketch_summarize", {a, plans, summaries, sizes, queue}, s,
                       {div_up(batches.size(), 64), 1, 1}, {64, 1, 1});
    }
  }
  auto control = std::getenv("METALGB_SERIAL_SKETCH");
  bool serial = control && std::string_view(control) == "1";
  command.dispatch(serial ? "sketch_reduce_serial" : "sketch_reduce_parallel",
                   {summaries, sizes, plans, columns, levels, workspace, statistics}, s,
                   {serial ? div_up(cols, 64) : cols, 1, 1}, {serial ? 64u : 256u, 1, 1});
  auto cut_control = std::getenv("METALGB_SERIAL_CUTS");
  bool serial_cuts = cut_control && std::string_view(cut_control) == "1";
  cut_tile_size = cut_tile_count = 0;
  if (!serial_cuts && cut_capacity >= 32) {
    cut_tile_size = 256;
    cut_tile_count = static_cast<u32>(div_up(cut_capacity - 1, cut_tile_size));
    SketchCutParams selection{p, cut_tile_count, cut_capacity, 0};
    // Query prefixes reuse the reduction workspace. Only the inter-tile
    // carries need additional storage, and every pass stays in this command.
    Buffer carries(checked_product(checked_product(cols, cut_tile_count), sizeof(u32)));
    command.dispatch("sketch_cut_queries", {workspace, columns, statistics, categorical, carries}, selection,
                     {cut_tile_count, cols, 1});
    command.dispatch("sketch_cut_carries", {workspace, columns, statistics, cuts, cut_counts, cut_minima, categorical, carries},
                     selection, {cols, 1, 1});
    command.dispatch("sketch_cut_gather", {workspace, columns, statistics, cuts, cut_counts, categorical, carries}, selection,
                     {cut_tile_count, cols, 1});
  } else {
    command.dispatch("sketch_cuts", {workspace, columns, statistics, cuts, cut_counts, cut_minima, categorical}, p,
                     {div_up(cols, 64), 1, 1}, {64, 1, 1});
  }
  quantize_seconds += command.wait(); ++quantize_commands;
}
}  // namespace metalgb
