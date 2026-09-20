#include "metalgb/runtime.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  try {
    std::size_t mebibytes = argc > 1 ? std::stoull(argv[1]) : 256;
    unsigned repeats = argc > 2 ? std::stoul(argv[2]) : 11;
    if (!mebibytes || !repeats || mebibytes > 4096) throw std::invalid_argument("Use 1..4096 MiB and positive repeats.");
    std::size_t bytes = mebibytes * 1024 * 1024;
    metalgb::Buffer input(bytes), output(bytes);
    for (std::size_t i = 0; i < bytes / sizeof(float); ++i) input.as<float>()[i] = static_cast<float>(i % 4093);
    std::uint64_t count = bytes / 16;
    nlohmann::json measurements = nlohmann::json::array();
    for (std::size_t threads : {128, 256, 512}) {
      std::vector<double> timings;
      for (unsigned i = 0; i < repeats + 3; ++i) {
        metalgb::Command command("measured streaming bandwidth");
        command.dispatch("stream_copy", {input, output}, count,
                         {metalgb::div_up(count, threads), 1, 1}, {threads, 1, 1});
        double elapsed = command.wait();
        if (i >= 3) timings.push_back(elapsed);
      }
      for (std::size_t i = 0; i < bytes / sizeof(float); i += 4099)
        if (output.as<float>()[i] != input.as<float>()[i]) throw std::runtime_error("GPU copy validation failed.");
      if (output.as<float>()[bytes / sizeof(float) - 1] != input.as<float>()[bytes / sizeof(float) - 1])
        throw std::runtime_error("GPU copy tail validation failed.");
      std::sort(timings.begin(), timings.end());
      auto median = timings[timings.size() / 2];
      measurements.push_back({{"threads", threads}, {"bytes_read", bytes}, {"bytes_written", bytes},
                               {"samples", timings}, {"median_gpu_seconds", median},
                               {"median_GB_per_second", 2.0 * bytes / median / 1.0e9},
                               {"best_GB_per_second", 2.0 * bytes / timings.front() / 1.0e9}});
    }
    auto info = nlohmann::json::parse(metalgb::Runtime::get().info());
    std::cout << nlohmann::json{{"device", info}, {"operation", "float4 copy"},
                                {"verification", "sampled input/output equality including tail"},
                                {"measurements", measurements}}.dump(2) << '\n';
    return 0;
  } catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
