// Cross-process IPC for Intel XPU tensors via SYCL ipc_memory. The handle is a
// self-contained, portable byte blob: the consumer opens it from the bytes alone
// (no dma-buf fd, no offset to carry -- get() takes torch's interior data_ptr and
// open() restores it). Context and device come from torch (c10::xpu) so the
// mapping lands on torch's own SYCL context.
#include <torch/extension.h>
#include <c10/xpu/XPUFunctions.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

// Upstream split this API (functions -> ipc::memory, types -> parent ipc) and
// deprecated flat ipc_memory; no oneAPI release ships it yet, so probe for it.
#if __has_include(<sycl/ext/oneapi/experimental/detail/ipc_common.hpp>)
namespace ipc = sycl::ext::oneapi::experimental::ipc::memory;
namespace ipc_types = sycl::ext::oneapi::experimental::ipc;
#else
namespace ipc = sycl::ext::oneapi::experimental::ipc_memory;
namespace ipc_types = sycl::ext::oneapi::experimental::ipc_memory;
#endif

namespace {

std::vector<std::byte> to_bytes(const std::vector<uint8_t>& in) {
  return {reinterpret_cast<const std::byte*>(in.data()),
          reinterpret_cast<const std::byte*>(in.data()) + in.size()};
}

// Exporter-side handles kept alive until ipc_release_handle(). We must not
// ipc::put() before the consumer opens: under the UR level-zero-v2 adapter
// put_ipc_handle frees the exporter fd and can race the consumer's open.
// The handle is a copyable, non-owning value (freed only via ipc::put), so
// storing it by value needs no manual new/delete.
std::mutex g_handles_mu;
std::unordered_map<uintptr_t, ipc_types::handle> g_handles;

}  // namespace

// Portable IPC handle bytes for the allocation backing `ptr` (interior pointers
// are fine -- the offset is in the blob). Handle retained until ipc_release_handle().
std::vector<uint8_t> ipc_get_handle(uintptr_t ptr) {
  sycl::context ctx = c10::xpu::get_device_context();
  ipc_types::handle h = ipc::get(reinterpret_cast<void*>(ptr), ctx);
  ipc_types::handle_data_t data = h.data();  // owning copy of the blob, independent of `h`
  {
    std::lock_guard<std::mutex> lk(g_handles_mu);
    auto it = g_handles.find(ptr);
    if (it != g_handles.end()) {
      ipc::put(it->second, ctx);  // release stale handle for a reused address
      it->second = h;
    } else {
      g_handles.emplace(ptr, h);
    }
  }
  return {reinterpret_cast<uint8_t*>(data.data()),
          reinterpret_cast<uint8_t*>(data.data()) + data.size()};
}

// Release the exporter handle from ipc_get_handle(ptr); no-op if unregistered.
// Call only after all consumers have opened their mappings (see level-zero-v2 note).
void ipc_release_handle(uintptr_t ptr) {
  std::optional<ipc_types::handle> h;
  {
    std::lock_guard<std::mutex> lk(g_handles_mu);
    auto it = g_handles.find(ptr);
    if (it == g_handles.end()) {
      return;
    }
    h = it->second;
    g_handles.erase(it);
  }
  ipc::put(*h, c10::xpu::get_device_context());
}

// Open a handle from another process -> mapped device pointer (offset included);
// pass it back to ipc_close_handle to release the mapping.
uintptr_t ipc_open_handle(const std::vector<uint8_t>& blob, int64_t device) {
  sycl::context ctx = c10::xpu::get_device_context();
  sycl::device dev = c10::xpu::get_raw_device(static_cast<c10::DeviceIndex>(device));
  std::vector<std::byte> data = to_bytes(blob);
  void* p = ipc::open(data, ctx, dev);
  return reinterpret_cast<uintptr_t>(p);
}

// Close an IPC mapping; `ptr` must be from ipc_open_handle.
void ipc_close_handle(uintptr_t ptr) {
  ipc::close(reinterpret_cast<void*>(ptr), c10::xpu::get_device_context());
}

// Wrap an external device pointer as a non-owning torch XPU uint8 tensor (the XPU
// analogue of rebuild_cuda_tensor) so the worker can read weight slices.
torch::Tensor ipc_wrap_tensor(uintptr_t dptr, int64_t nbytes, int64_t device) {
  auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(
      torch::kXPU, static_cast<c10::DeviceIndex>(device));
  return torch::from_blob(reinterpret_cast<void*>(dptr), {nbytes}, [](void*) {}, opts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ipc_get_handle", &ipc_get_handle, "Get portable IPC handle bytes for a device ptr");
  m.def("ipc_release_handle", &ipc_release_handle, "Release the retained exporter handle for a ptr");
  m.def("ipc_open_handle", &ipc_open_handle, "Open an IPC handle blob -> device ptr");
  m.def("ipc_close_handle", &ipc_close_handle, "Close an IPC mapping");
  m.def("ipc_wrap_tensor", &ipc_wrap_tensor, "Wrap an external XPU ptr as a torch tensor");
}
