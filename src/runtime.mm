#include "metalgb/runtime.hpp"
#include "metal_library.hpp"
#include <nlohmann/json.hpp>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>

namespace metalgb {
namespace {
std::string error_text(NSError* error) {
  return error ? std::string([[error localizedDescription] UTF8String]) : "unknown Metal error";
}
unsigned bind_buffers(id<MTLComputeCommandEncoder> encoder,
                      std::initializer_list<Buffer> buffers) {
  // Bind the same contiguous argument range in one call. Keep the original
  // loop for larger lists instead of imposing a new limit on Command callers.
  constexpr std::size_t capacity = 32;
  unsigned index = 0;
  if (buffers.size() > capacity) {
    for (auto const& buffer : buffers)
      [encoder setBuffer:(id<MTLBuffer>)buffer.native() offset:0 atIndex:index++];
    return index;
  }
  id<MTLBuffer> native_buffers[capacity];
  NSUInteger offsets[capacity]{};
  for (auto const& buffer : buffers)
    native_buffers[index++] = (id<MTLBuffer>)buffer.native();
  if (index) [encoder setBuffers:native_buffers offsets:offsets withRange:NSMakeRange(0, index)];
  return index;
}
}
struct Runtime::Impl {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  std::map<std::string, id<MTLComputePipelineState>, std::less<>> pipelines;
  std::mutex mutex;
  ~Impl() {
    for (auto const& [name, pipeline] : pipelines) [pipeline release];
    [library release]; [queue release]; [device release];
  }
};
Runtime& Runtime::get() { static Runtime runtime; return runtime; }
Runtime::Runtime() : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device || ![impl_->device hasUnifiedMemory] ||
        ![impl_->device supportsFamily:MTLGPUFamilyApple7])
      throw std::runtime_error("xgboost-metal requires an Apple silicon Metal GPU; no CPU fallback exists.");
    impl_->queue = [impl_->device newCommandQueue];
    if (!impl_->queue) throw std::runtime_error("Cannot create the Metal command queue.");
    dispatch_data_t source = dispatch_data_create(kMetalLibrary, kMetalLibrarySize, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError* error = nil;
    impl_->library = [impl_->device newLibraryWithData:source error:&error];
    dispatch_release(source);
    if (!impl_->library) throw std::runtime_error("Cannot load embedded Metal kernels: " + error_text(error));
  }
}
Runtime::~Runtime() = default;
void* Runtime::device() const { return (void*)impl_->device; }
void* Runtime::queue() const { return (void*)impl_->queue; }
void* Runtime::pipeline(std::string_view name) {
  std::lock_guard lock(impl_->mutex);
  if (auto it = impl_->pipelines.find(name); it != impl_->pipelines.end()) return (void*)it->second;
  @autoreleasepool {
    NSString* symbol = [NSString stringWithUTF8String:std::string(name).c_str()];
    id<MTLFunction> function = [impl_->library newFunctionWithName:symbol];
    if (!function) throw std::runtime_error("Missing Metal kernel: " + std::string(name));
    NSError* error = nil;
    auto pipeline = [impl_->device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (!pipeline) throw std::runtime_error("Metal pipeline " + std::string(name) + ": " + error_text(error));
    impl_->pipelines.emplace(std::string(name), pipeline);
    return (void*)pipeline;
  }
}
std::string Runtime::info() const {
  @autoreleasepool {
    nlohmann::json info{{"USE_METAL", true}, {"USE_CUDA", false}, {"USE_OPENMP", false},
      {"USE_NCCL", false}, {"USE_DLOPEN_NCCL", false}, {"USE_RMM", false},
      {"DEBUG", false}, {"BUILTIN_PREFETCH_PRESENT", false},
      {"METAL_DEVICE", [[impl_->device name] UTF8String]},
      {"METAL_UNIFIED_MEMORY", static_cast<bool>([impl_->device hasUnifiedMemory])},
      {"METAL_MAX_BUFFER_LENGTH", [impl_->device maxBufferLength]},
      {"METAL_MAX_THREADGROUP_MEMORY", [impl_->device maxThreadgroupMemoryLength]},
      {"METAL_RECOMMENDED_WORKING_SET", [impl_->device recommendedMaxWorkingSetSize]},
      {"BACKEND", "metal"}, {"XGBOOST_REFERENCE", "6d2dbe19a099aa046178be8cfc303bfec1a623a6"}};
    return info.dump();
  }
}
Buffer::Buffer(std::size_t bytes) : bytes_(bytes) {
  @autoreleasepool {
    id<MTLDevice> device = (id<MTLDevice>)Runtime::get().device();
    if (bytes > [device maxBufferLength]) throw std::runtime_error("Requested buffer exceeds Metal maxBufferLength.");
    // Even an unused array must fit its element type for Metal argument validation.
    id<MTLBuffer> buffer = [device newBufferWithLength:std::max<std::size_t>(bytes, 64)
                                           options:MTLResourceStorageModeShared];
    if (!buffer) throw std::bad_alloc();
    object_ = std::shared_ptr<void>((void*)buffer, [](void* p) { [(id<MTLBuffer>)p release]; });
  }
}
Buffer::Buffer(void const* data, std::size_t bytes) : Buffer(bytes) {
  if (bytes) std::memcpy(this->data(), data, bytes);
}
void* Buffer::data() const { return object_ ? [(id<MTLBuffer>)object_.get() contents] : nullptr; }
Buffer Buffer::prefix(std::size_t bytes) const {
  if (bytes > bytes_) throw std::invalid_argument("Buffer prefix exceeds its allocation.");
  Buffer result = *this;
  result.bytes_ = bytes;
  return result;
}
bool Buffer::reserve(std::size_t bytes, std::size_t preserve_bytes) {
  if (preserve_bytes > bytes_ || preserve_bytes > bytes)
    throw std::invalid_argument("Buffer reserve prefix exceeds its size.");
  if (object_ && bytes <= bytes_) return false;
  auto device = (id<MTLDevice>)Runtime::get().device();
  auto limit = static_cast<std::size_t>(device.maxBufferLength);
  if (bytes > limit) throw std::runtime_error("Requested buffer exceeds Metal maxBufferLength.");
  auto capacity = std::max(bytes, std::max<std::size_t>(64, bytes_ > limit / 2 ? limit : bytes_ * 2));
  Buffer grown(capacity);
  if (preserve_bytes) std::memcpy(grown.data(), data(), preserve_bytes);
  *this = std::move(grown);
  return true;
}
namespace {
// Diagnostic only: isolates every dispatch in its own encoder so that
// stage-boundary timestamps attribute GPU time to individual kernels.
struct DispatchTotals {
  struct Entry { std::uint64_t count{0}; double seconds{0}; };
  std::mutex mutex;
  std::map<std::string, Entry> kernels;
  std::string path;
  id<MTLCounterSet> counters{nil};
  static DispatchTotals* get() {
    static DispatchTotals* totals = [] () -> DispatchTotals* {
      char const* path = std::getenv("METALGB_DISPATCH_PROFILE");
      if (!path || !*path) return nullptr;
      auto result = new DispatchTotals;
      result->path = path;
      auto device = (id<MTLDevice>)Runtime::get().device();
      for (id<MTLCounterSet> set in device.counterSets)
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) result->counters = [set retain];
      if (!result->counters) throw std::runtime_error("No Metal timestamp counters.");
      std::atexit([] {
        auto t = DispatchTotals::get();
        nlohmann::json out = nlohmann::json::object();
        for (auto const& [name, e] : t->kernels) out[name] = {{"count", e.count}, {"seconds", e.seconds}};
        if (auto f = std::fopen(t->path.c_str(), "w")) { auto text = out.dump(1); std::fwrite(text.data(), 1, text.size(), f); std::fclose(f); }
      });
      return result;
    }();
    return totals;
  }
};
}
struct Command::Profile {
  // Metal caps live counter sample buffers per process (32 on M4) and a
  // buffer at 32 KB (4096 timestamps). Dispatches beyond that are counted
  // under "<kernel> (untimed)" instead of failing the command.
  static constexpr NSUInteger kSamples = 4096;
  std::vector<id<MTLCounterSampleBuffer>> buffers;
  std::vector<std::string> names, untimed;
  MTLTimestamp cpu0{0}, gpu0{0};
  ~Profile() { for (auto b : buffers) [b release]; }
  std::pair<id<MTLCounterSampleBuffer>, NSUInteger> next(std::string_view kernel) {
    auto index = names.size() * 2 % kSamples;
    if (index == 0) {
      auto descriptor = [[[MTLCounterSampleBufferDescriptor alloc] init] autorelease];
      descriptor.counterSet = DispatchTotals::get()->counters;
      descriptor.storageMode = MTLStorageModeShared;
      descriptor.sampleCount = kSamples;
      NSError* error = nil;
      auto buffer = [(id<MTLDevice>)Runtime::get().device() newCounterSampleBufferWithDescriptor:descriptor error:&error];
      if (!buffer) { untimed.emplace_back(kernel); return {nil, 0}; }
      buffers.push_back(buffer);
    }
    names.emplace_back(kernel);
    return {buffers.back(), index};
  }
  void finish() {
    MTLTimestamp cpu1 = 0, gpu1 = 0;
    [(id<MTLDevice>)Runtime::get().device() sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    double scale = (gpu1 > gpu0 && cpu1 > cpu0) ? 1e-9 * double(cpu1 - cpu0) / double(gpu1 - gpu0) : 1e-9;
    auto totals = DispatchTotals::get();
    std::lock_guard lock(totals->mutex);
    for (std::size_t b = 0; b < buffers.size(); ++b) {
      auto first = b * (kSamples / 2);
      auto count = std::min<std::size_t>(kSamples / 2, names.size() - first);
      NSData* data = [buffers[b] resolveCounterRange:NSMakeRange(0, count * 2)];
      auto stamps = data ? static_cast<MTLCounterResultTimestamp const*>(data.bytes) : nullptr;
      for (std::size_t i = 0; i < count; ++i) {
        auto& entry = totals->kernels[names[first + i]];
        ++entry.count;
        if (!stamps) continue;
        auto begin = stamps[2 * i].timestamp, end = stamps[2 * i + 1].timestamp;
        if (begin && begin != MTLCounterErrorValue && end != MTLCounterErrorValue && end >= begin)
          entry.seconds += double(end - begin) * scale;
      }
    }
    for (auto const& name : untimed) ++totals->kernels[name + " (untimed)"].count;
  }
};
namespace {
id<MTLCommandBuffer> new_command_buffer(std::string const& label) {
  id<MTLCommandBuffer> command = [(id<MTLCommandQueue>)Runtime::get().queue() commandBuffer];
  if (!command) throw std::runtime_error("Cannot allocate Metal command buffer.");
  [command retain];
  command.label = [NSString stringWithUTF8String:label.c_str()];
  return command;
}
}
Command::Command(std::string_view label, bool coalesce, bool compute_fill)
    : label_(label), coalesce_(coalesce), compute_fill_(compute_fill) {
  @autoreleasepool {
    if (DispatchTotals::get()) { profile_ = std::make_unique<Profile>(); coalesce_ = false; }
    object_ = (void*)new_command_buffer(label_);
  }
}
Command::~Command() {
  end_compute();
  for (auto command : flushed_) [(id<MTLCommandBuffer>)command release];
  [(id<MTLCommandBuffer>)object_ release];
}
void* Command::begin_compute(std::string_view kernel) {
  if (!encoder_) {
    // Serial encoders make each dispatch's writes visible to the next one,
    // including GPU-produced indirect arguments. No concurrent dispatches are
    // used here: the histogram and partition kernels depend on their inputs.
    id<MTLComputeCommandEncoder> encoder;
    if (profile_) {
      auto [sample, index] = profile_->next(kernel);
      auto descriptor = [MTLComputePassDescriptor computePassDescriptor];
      descriptor.dispatchType = MTLDispatchTypeSerial;
      if (sample) {
        descriptor.sampleBufferAttachments[0].sampleBuffer = sample;
        descriptor.sampleBufferAttachments[0].startOfEncoderSampleIndex = index;
        descriptor.sampleBufferAttachments[0].endOfEncoderSampleIndex = index + 1;
      }
      encoder = [(id<MTLCommandBuffer>)object_ computeCommandEncoderWithDescriptor:descriptor];
    } else {
      encoder = [(id<MTLCommandBuffer>)object_ computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    }
    if (!encoder) throw std::runtime_error("Cannot allocate Metal compute encoder.");
    [encoder retain];
    encoder_ = (void*)encoder;
    ++encoder_count_;
  }
  return encoder_;
}
void Command::end_compute() {
  if (encoder_) {
    [(id<MTLComputeCommandEncoder>)encoder_ endEncoding];
    [(id<MTLComputeCommandEncoder>)encoder_ release];
    encoder_ = nullptr;
  }
}
void Command::dispatch(std::string_view kernel, std::initializer_list<Buffer> buffers,
                       void const* params, std::size_t size, Extent groups, Extent threads,
                       std::size_t threadgroup_bytes) {
  if (submitted_) throw std::logic_error("Cannot encode a committed Metal command buffer.");
  if (!groups.x || !groups.y || !groups.z) return;
  @autoreleasepool {
    auto pipeline = (id<MTLComputePipelineState>)Runtime::get().pipeline(kernel);
    if (threads.x * threads.y * threads.z > pipeline.maxTotalThreadsPerThreadgroup)
      throw std::runtime_error("Threadgroup exceeds the Metal pipeline limit.");
    if (threadgroup_bytes) {
      auto device = (id<MTLDevice>)Runtime::get().device();
      if (threadgroup_bytes + pipeline.staticThreadgroupMemoryLength > device.maxThreadgroupMemoryLength)
        throw std::runtime_error("Threadgroup memory exceeds the Metal device limit.");
    }
    auto encoder = (id<MTLComputeCommandEncoder>)begin_compute(kernel);
    [encoder setComputePipelineState:pipeline];
    auto index = bind_buffers(encoder, buffers);
    if (size) [encoder setBytes:params length:size atIndex:index];
    if (threadgroup_bytes) [encoder setThreadgroupMemoryLength:threadgroup_bytes atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(groups.x, groups.y, groups.z)
           threadsPerThreadgroup:MTLSizeMake(threads.x, threads.y, threads.z)];
    if (!coalesce_) end_compute();
  }
}
void Command::dispatch_indirect(std::string_view kernel, std::initializer_list<Buffer> buffers,
                                void const* params, std::size_t size, Buffer const& arguments, Extent threads) {
  if (submitted_) throw std::logic_error("Cannot encode a committed Metal command buffer.");
  if (arguments.size() < sizeof(MTLDispatchThreadgroupsIndirectArguments))
    throw std::invalid_argument("Indirect dispatch requires three uint32 arguments.");
  @autoreleasepool {
    auto pipeline = (id<MTLComputePipelineState>)Runtime::get().pipeline(kernel);
    if (threads.x * threads.y * threads.z > pipeline.maxTotalThreadsPerThreadgroup)
      throw std::runtime_error("Threadgroup exceeds the Metal pipeline limit.");
    auto encoder = (id<MTLComputeCommandEncoder>)begin_compute(kernel);
    [encoder setComputePipelineState:pipeline];
    auto index = bind_buffers(encoder, buffers);
    if (size) [encoder setBytes:params length:size atIndex:index];
    [encoder dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)arguments.native() indirectBufferOffset:0
                              threadsPerThreadgroup:MTLSizeMake(threads.x, threads.y, threads.z)];
    if (!coalesce_) end_compute();
  }
}
void Command::zero(Buffer const& buffer) {
  if (submitted_) throw std::logic_error("Cannot encode a committed Metal command buffer.");
  if (!buffer.size()) return;
  if (compute_fill_) {
    auto bytes = buffer.size();
    dispatch("fill_zero", {buffer}, bytes, {std::min<std::size_t>(div_up(bytes, 16 * 256), 65535), 1, 1});
    return;
  }
  @autoreleasepool {
    end_compute();
    id<MTLBlitCommandEncoder> encoder;
    if (profile_) {
      auto [sample, index] = profile_->next("fill_buffer");
      auto descriptor = [MTLBlitPassDescriptor blitPassDescriptor];
      if (sample) {
        descriptor.sampleBufferAttachments[0].sampleBuffer = sample;
        descriptor.sampleBufferAttachments[0].startOfEncoderSampleIndex = index;
        descriptor.sampleBufferAttachments[0].endOfEncoderSampleIndex = index + 1;
      }
      encoder = [(id<MTLCommandBuffer>)object_ blitCommandEncoderWithDescriptor:descriptor];
    } else {
      encoder = [(id<MTLCommandBuffer>)object_ blitCommandEncoder];
    }
    if (!encoder) throw std::runtime_error("Cannot allocate Metal blit encoder.");
    ++encoder_count_;
    [encoder fillBuffer:(id<MTLBuffer>)buffer.native() range:NSMakeRange(0, buffer.size()) value:0];
    [encoder endEncoding];
  }
}
void Command::set_coalescing(bool enabled) {
  if (submitted_) throw std::logic_error("Cannot change a committed Metal command buffer.");
  if (profile_) return;
  if (coalesce_ != enabled) { end_compute(); coalesce_ = enabled; }
}
void Command::submit_current() {
  end_compute();
  if (profile_ && !started_)
    [(id<MTLDevice>)Runtime::get().device() sampleTimestamps:&profile_->cpu0 gpuTimestamp:&profile_->gpu0];
  started_ = true;
  [(id<MTLCommandBuffer>)object_ commit];
}
void Command::flush() {
  if (submitted_) throw std::logic_error("Cannot flush a committed Metal command buffer.");
  @autoreleasepool {
    // Allocate first: a failed allocation leaves this command unchanged.
    auto next = new_command_buffer(label_);
    submit_current();
    flushed_.push_back(object_);
    object_ = (void*)next;
  }
}
void Command::commit() {
  if (submitted_) return;
  @autoreleasepool { submit_current(); submitted_ = true; }
}
bool Command::completed() const {
  if (!submitted_) return false;
  // One queue completes command buffers in submission order.
  return [(id<MTLCommandBuffer>)object_ status] >= MTLCommandBufferStatusCompleted;
}
double Command::wait() {
  @autoreleasepool {
    auto command = (id<MTLCommandBuffer>)object_;
    commit();
    [command waitUntilCompleted];
    if (profile_) { profile_->finish(); profile_.reset(); }
    double elapsed = 0;
    for (std::size_t i = 0; i <= flushed_.size(); ++i) {
      auto buffer = i < flushed_.size() ? (id<MTLCommandBuffer>)flushed_[i] : command;
      // Earlier buffers completed before the last one on this queue.
      if (buffer.status != MTLCommandBufferStatusCompleted && buffer.status != MTLCommandBufferStatusError)
        [buffer waitUntilCompleted];
      if (buffer.status == MTLCommandBufferStatusError)
        throw std::runtime_error("Metal execution failed: " + error_text(buffer.error));
      elapsed += buffer.GPUEndTime - buffer.GPUStartTime;
    }
    return elapsed;
  }
}
}  // namespace metalgb
