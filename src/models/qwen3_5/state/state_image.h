#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/host_context_arena.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/transfer_work.h"
#include "ninfer/types.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5 {

struct DFlashLocalStateSpec {
    std::uint32_t layers   = 0;
    std::uint32_t capacity = 0;
    std::int32_t kv_heads  = 0;
    std::int32_t head_dim  = 0;
};

struct StateImageSpec {
    LinearAttentionStatePoolSpec linear;
    std::int32_t hidden = 0;
    std::optional<DFlashLocalStateSpec> dflash_local;
    // Host encoding of the FP32 Device recurrent state. Conv, hidden and DFlash components are
    // always stored in their Device dtype.
    HostStateStorage host_recurrent = HostStateStorage::Fp32;
};

// Byte layout of one layer's encoded recurrent state inside a Host StateImage. FP32 and BF16
// use only `values`; NVFP4 stores E2M1 codes in `values`, then E4M3 group scales and FP32
// per-value-head scales.
struct StateImageRecurrentEncoding {
    std::size_t values_bytes       = 0;
    std::size_t group_scale_offset = 0;
    std::size_t group_scale_bytes  = 0;
    std::size_t slice_scale_offset = 0;
    std::size_t slice_scale_bytes  = 0;
};

struct StateImageHostLayout {
    StateImageSpec spec;
    LayoutRegion linear_conv;
    std::size_t linear_conv_layer_bytes = 0;
    LayoutRegion linear_recurrent;
    std::size_t linear_recurrent_layer_bytes = 0;
    StateImageRecurrentEncoding recurrent_encoding;
    LayoutRegion continuation_hidden;
    std::optional<LayoutRegion> dflash_local_k;
    std::optional<LayoutRegion> dflash_local_v;
    std::size_t dflash_local_layer_bytes = 0;
    std::size_t image_bytes              = 0;
};

struct StateImageDeviceLayout {
    LinearAttentionStatePoolLayout linear;
    TensorRegion continuation_hidden;
    std::optional<CyclicKVCacheLayout> dflash_local;
    // One layer of encoded recurrent state, staged between the encode/decode Op and the
    // Host copy. Present only when the Host encoding differs from the Device FP32 state.
    std::optional<LayoutRegion> recurrent_staging;
    StateImageHostLayout host;
};

[[nodiscard]] TransferWork state_image_transfer_work(const StateImageHostLayout& layout);
[[nodiscard]] TransferWork dflash_local_transfer_work(const StateImageHostLayout& layout);

[[nodiscard]] StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                                  const StateImageSpec& spec);

struct HostStateImageView {
    std::byte* data                    = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

struct HostStateImageConstView {
    const std::byte* data              = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

class HostStatePool;

struct HostStateSlotHandle {
    std::uint32_t index        = 0;
    std::uint32_t generation   = 0;
    const HostStatePool* owner = nullptr;
};

/** Typed StateImage descriptors backed on demand by the shared Host arena; owns no cache policy.
 */
class HostStatePool {
public:
    HostStatePool(HostContextArena& arena, StateImageHostLayout layout);

    HostStatePool(const HostStatePool&)            = delete;
    HostStatePool& operator=(const HostStatePool&) = delete;
    HostStatePool(HostStatePool&&)                 = delete;
    HostStatePool& operator=(HostStatePool&&)      = delete;

    [[nodiscard]] std::optional<HostStateSlotHandle> allocate() noexcept;
    [[nodiscard]] bool can_allocate() const noexcept;
    [[nodiscard]] bool publish(HostStateSlotHandle handle) noexcept;
    [[nodiscard]] bool release(HostStateSlotHandle handle) noexcept;

    [[nodiscard]] HostStateImageView writable_view(HostStateSlotHandle handle);
    [[nodiscard]] HostStateImageConstView view(HostStateSlotHandle handle) const;

    [[nodiscard]] std::uint32_t capacity() const noexcept;

    [[nodiscard]] std::uint32_t occupied() const noexcept { return occupied_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept {
        return static_cast<std::size_t>(occupied_) * layout_.image_bytes;
    }

    [[nodiscard]] const StateImageHostLayout& layout() const noexcept { return layout_; }

private:
    struct Slot {
        HostContextAllocation storage;
        std::uint32_t generation = 1;
    };

    [[nodiscard]] bool valid(HostStateSlotHandle handle) const noexcept;
    [[nodiscard]] std::byte* slot_data(std::uint32_t index) const noexcept;

    StateImageHostLayout layout_;
    HostContextArena* arena_ = nullptr;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_slots_;
    std::uint32_t free_count_ = 0;
    std::uint32_t occupied_   = 0;
};

struct StateImageDeviceSlotView {
    LinearAttentionStateSlotView linear;
    Tensor continuation_hidden;
    std::optional<CyclicKVCacheSlotView> dflash_local;
};

/**
 * Caller-backed fixed storage for Qwen3.6 continuation state.
 *
 * Every absolute slot contains common GDN/hidden state and, for a DFlash Program, its local cyclic
 * K/V state. The pool owns neither slot roles nor logical checkpoint identity.
 */
class StateImageDevicePool {
public:
    StateImageDevicePool(DeviceSpan backing, const StateImageDeviceLayout& layout);

    StateImageDevicePool(const StateImageDevicePool&)            = delete;
    StateImageDevicePool& operator=(const StateImageDevicePool&) = delete;
    StateImageDevicePool(StateImageDevicePool&&)                 = delete;
    StateImageDevicePool& operator=(StateImageDevicePool&&)      = delete;

    [[nodiscard]] std::int32_t slot_count() const noexcept { return linear_.slot_count(); }

    [[nodiscard]] StateImageDeviceSlotView slot_view(std::int32_t slot) const;
    [[nodiscard]] Tensor continuation_hidden_slot(std::int32_t slot) const;

    [[nodiscard]] LinearAttentionStatePool& linear() noexcept { return linear_; }

    [[nodiscard]] const LinearAttentionStatePool& linear() const noexcept { return linear_; }

    [[nodiscard]] Tensor& continuation_hidden_store() noexcept { return continuation_hidden_; }

    [[nodiscard]] const Tensor& continuation_hidden_store() const noexcept {
        return continuation_hidden_;
    }

    [[nodiscard]] CyclicKVCache* dflash_local() noexcept;
    [[nodiscard]] const CyclicKVCache* dflash_local() const noexcept;

    [[nodiscard]] const StateImageHostLayout& host_layout() const noexcept { return host_layout_; }

    void zero_slot(std::int32_t slot, cudaStream_t stream = nullptr);
    void zero_all(cudaStream_t stream = nullptr);
    void copy_slot(std::int32_t source, std::int32_t destination, cudaStream_t stream = nullptr);
    void copy_dflash_local(std::int32_t source, std::int32_t destination,
                           cudaStream_t stream = nullptr);
    void copy_to_host(std::int32_t source, HostStateImageView destination,
                      cudaStream_t stream = nullptr) const;
    void copy_from_host(HostStateImageConstView source, std::int32_t destination,
                        cudaStream_t stream = nullptr);

private:
    void validate_host_layout(const StateImageHostLayout* layout, const std::byte* data) const;
    void encode_recurrent_to_host(const Tensor& recurrent, std::byte* host_layer,
                                  cudaStream_t stream) const;
    void decode_recurrent_from_host(const std::byte* host_layer, Tensor& recurrent,
                                    cudaStream_t stream);

    LinearAttentionStatePool linear_;
    Tensor continuation_hidden_;
    std::optional<CyclicKVCache> dflash_local_;
    // Bound only for an encoded Host recurrent state; transfers on one stream reuse it in order.
    DeviceSpan recurrent_staging_;
    StateImageHostLayout host_layout_;
};

} // namespace ninfer::models::qwen3_5
