#include "ops/launcher/nvfp4_slice_codec.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/nvfp4_slice_codec.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kDecodeBlock   = 256;
constexpr int kDecodeGridCap = 4096;

} // namespace

void encode_nvfp4_slices_launch(const Tensor& source, std::int64_t slice_elements, Tensor& codes,
                                Tensor& group_scales, Tensor& slice_scales, cudaStream_t stream) {
    const std::int64_t slices = source.numel() / slice_elements;
    if (slices > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("encode_nvfp4_slices: slice count exceeds the CUDA grid");
    }
    encode_nvfp4_slices_kernel<<<static_cast<int>(slices), kNvfp4SliceEncodeBlock, 0, stream>>>(
        static_cast<const float*>(source.data), slice_elements,
        static_cast<uint2*>(codes.data), static_cast<std::uint8_t*>(group_scales.data),
        static_cast<float*>(slice_scales.data));
    CUDA_CHECK(cudaGetLastError());
}

void decode_nvfp4_slices_launch(const Tensor& codes, const Tensor& group_scales,
                                const Tensor& slice_scales, std::int64_t slice_elements,
                                Tensor& destination, cudaStream_t stream) {
    const std::int64_t total_groups = destination.numel() / kNvfp4SliceGroup;
    const int grid                  = static_cast<int>(std::max<std::int64_t>(
        1, std::min<std::int64_t>(div_up(total_groups, static_cast<std::int64_t>(kDecodeBlock)),
                                  kDecodeGridCap)));
    decode_nvfp4_slices_kernel<<<grid, kDecodeBlock, 0, stream>>>(
        static_cast<const uint2*>(codes.data),
        static_cast<const std::uint8_t*>(group_scales.data),
        static_cast<const float*>(slice_scales.data), slice_elements / kNvfp4SliceGroup,
        total_groups, static_cast<float4*>(destination.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
