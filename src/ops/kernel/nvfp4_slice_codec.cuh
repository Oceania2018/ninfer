#pragma once

#include "ninfer/ops/nvfp4_slice_codec.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kNvfp4SliceEncodeBlock = 256;

__global__ void __launch_bounds__(kNvfp4SliceEncodeBlock)
    encode_nvfp4_slices_kernel(const float* source, std::int64_t slice_elements,
                               uint2* codes, std::uint8_t* group_scales,
                               float* slice_scales) {
    __shared__ float warp_max[kNvfp4SliceEncodeBlock / 32];
    __shared__ float slice_scale_shared;

    const std::int64_t slice  = blockIdx.x;
    const float4* slice_data  = reinterpret_cast<const float4*>(source + slice * slice_elements);
    const std::int64_t quads  = slice_elements / 4;
    const std::int64_t groups = slice_elements / kNvfp4SliceGroup;

    float local_max = 0.0F;
    for (std::int64_t i = threadIdx.x; i < quads; i += blockDim.x) {
        const float4 value = slice_data[i];
        local_max          = fmaxf(local_max, fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                                    fmaxf(fabsf(value.z), fabsf(value.w))));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
    }
    if ((threadIdx.x & 31) == 0) { warp_max[threadIdx.x >> 5] = local_max; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float slice_max = 0.0F;
        for (int warp = 0; warp < kNvfp4SliceEncodeBlock / 32; ++warp) {
            slice_max = fmaxf(slice_max, warp_max[warp]);
        }
        const float slice_scale = __fdiv_rn(slice_max, kNvfp4SliceRange);
        slice_scales[slice]     = slice_scale;
        slice_scale_shared      = slice_scale;
    }
    __syncthreads();
    const float slice_scale = slice_scale_shared;

    for (std::int64_t group = threadIdx.x; group < groups; group += blockDim.x) {
        float2 values[8];
        float group_max = 0.0F;
#pragma unroll
        for (int quad = 0; quad < 4; ++quad) {
            const float4 value  = slice_data[group * 4 + quad];
            values[2 * quad]     = make_float2(value.x, value.y);
            values[2 * quad + 1] = make_float2(value.z, value.w);
            group_max = fmaxf(group_max, fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                               fmaxf(fabsf(value.z), fabsf(value.w))));
        }

        const std::int64_t global_group = slice * groups + group;
        std::uint8_t scale_code         = 0;
        if (slice_scale != 0.0F) {
            const float relative = __fdiv_rn(__fdiv_rn(group_max, 6.0F), slice_scale);
            scale_code           = __nv_cvt_float_to_fp8(relative, __NV_SATFINITE, __NV_E4M3);
        }
        const float scale = __fmul_rn(detail::decode_nvfp4_e4m3(scale_code), slice_scale);
        uint2 packed = make_uint2(0U, 0U);
        if (scale != 0.0F) {
#pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                values[pair].x = __fdiv_rn(values[pair].x, scale);
                values[pair].y = __fdiv_rn(values[pair].y, scale);
            }
            detail::pack_nvfp4_e2m1x16(values, packed.x, packed.y);
        }
        codes[global_group]        = packed;
        group_scales[global_group] = scale_code;
    }
}

__global__ void decode_nvfp4_slices_kernel(const uint2* codes,
                                           const std::uint8_t* group_scales,
                                           const float* slice_scales, std::int64_t slice_groups,
                                           std::int64_t total_groups, float4* destination) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t group = start; group < total_groups; group += stride) {
        const float slice_scale = slice_scales[group / slice_groups];
        const float scale = __fmul_rn(detail::decode_nvfp4_e4m3(group_scales[group]), slice_scale);
        const uint2 packed = codes[group];
        const std::uint32_t words[2] = {packed.x, packed.y};
#pragma unroll
        for (int quad = 0; quad < 4; ++quad) {
            const std::uint32_t word = words[quad >> 1];
            const int shift          = (quad & 1) * 16;
            const float2 lo = detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(word >> shift));
            const float2 hi =
                detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(word >> (shift + 8)));
            destination[group * 4 + quad] =
                make_float4(__fmul_rn(lo.x, scale), __fmul_rn(lo.y, scale),
                            __fmul_rn(hi.x, scale), __fmul_rn(hi.y, scale));
        }
    }
}

} // namespace ninfer::ops
