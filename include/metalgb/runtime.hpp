#pragma once
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace metalgb {
struct Extent { std::size_t x{1}, y{1}, z{1}; };

class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(std::size_t bytes);
  Buffer(void const* data, std::size_t bytes);
  template <class T> explicit Buffer(std::span<T const> data) : Buffer(data.data(), data.size_bytes()) {}
  [[nodiscard]] void* data() const;
  [[nodiscard]] std::size_t size() const { return bytes_; }
  [[nodiscard]] void* native() const { return object_.get(); }
  [[nodiscard]] Buffer prefix(std::size_t bytes) const;
  template <class T> [[nodiscard]] T* as() const { return static_cast<T*>(data()); }
  explicit operator bool() const { return static_cast<bool>(object_); }
  void zero() const { if (bytes_) std::memset(data(), 0, bytes_); }
 private:
  std::shared_ptr<void> object_;
  std::size_t bytes_{0};
};

struct Timing { std::string label; double gpu_seconds; };
class Runtime {
 public:
  static Runtime& get();
  Runtime(Runtime const&) = delete;
  Runtime& operator=(Runtime const&) = delete;
  [[nodiscard]] std::string info() const;
  [[nodiscard]] void* device() const;
  [[nodiscard]] void* queue() const;
  [[nodiscard]] void* pipeline(std::string_view name);
 private:
  Runtime();
  ~Runtime();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class Command {
 public:
  explicit Command(std::string_view label, bool coalesce = false, bool compute_fill = false);
  ~Command();
  Command(Command const&) = delete;
  Command& operator=(Command const&) = delete;
  void dispatch(std::string_view kernel, std::initializer_list<Buffer> buffers,
                void const* params, std::size_t size, Extent groups, Extent threads = {256, 1, 1},
                std::size_t threadgroup_bytes = 0);
  template<class P> void dispatch(std::string_view kernel, std::initializer_list<Buffer> buffers,
                                  P const& params, Extent groups, Extent threads = {256, 1, 1},
                                  std::size_t threadgroup_bytes = 0) {
    dispatch(kernel, buffers, &params, sizeof(P), groups, threads, threadgroup_bytes);
  }
  void dispatch_indirect(std::string_view kernel, std::initializer_list<Buffer> buffers,
                          void const* params, std::size_t size, Buffer const& arguments,
                          Extent threads = {256, 1, 1});
  template<class P> void dispatch_indirect(std::string_view kernel, std::initializer_list<Buffer> buffers,
                                          P const& params, Buffer const& arguments,
                                          Extent threads = {256, 1, 1}) {
    dispatch_indirect(kernel, buffers, &params, sizeof(P), arguments, threads);
  }
  void zero(Buffer const& buffer);
  double wait();
  [[nodiscard]] std::size_t encoder_count() const { return encoder_count_; }
 private:
  void* begin_compute();
  void end_compute();
  void* object_{nullptr};
  void* encoder_{nullptr};
  std::size_t encoder_count_{0};
  bool coalesce_{false}, compute_fill_{false};
  bool submitted_{false};
};
inline std::size_t div_up(std::size_t n, std::size_t d) { return (n + d - 1) / d; }
}  // namespace metalgb
