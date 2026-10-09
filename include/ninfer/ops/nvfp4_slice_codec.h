#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Two-level NVFP4 for contiguous FP32 slices: one FP32 scale per slice, one E4M3 scale per
// group of 16 contiguous values, and one E2M1 code per value.
inline constexpr std::int64_t kNvfp4SliceGroup = 16;
// Largest E2M1 magnitude times largest finite E4M3 scale: the slice scale maps the slice maximum
// to the top of the representable two-level range.
inline constexpr float kNvfp4SliceRange = 6.0F * 448.0F;

/**
 * Op: encode_nvfp4_slices
 *
 * Math / indexing:
 *   The source is S = numel / L slices of L contiguous values; slice s holds source[s*L + k].
 *   Groups are the L/16 runs of 16 contiguous values of a slice, numbered globally
 *   j = s*(L/16) + k/16. Every division and product below is one IEEE FP32 round-to-nearest-even
 *   operation, evaluated in the written order.
 *     slice_scales[s] = max_k |x[s,k]| / 2688
 *     group_scales[j] = e4m3_rne_satfinite((max over group j of |x|) / 6 / slice_scales[s]),
 *                       or 0 when slice_scales[s] == 0
 *     scale(j)        = e4m3(group_scales[j]) * slice_scales[s]
 *     code(i)         = e2m1_rne_satfinite(x[i] / scale(j)), or +0 when scale(j) == 0
 *   codes[i/2] holds code(i) in its low nibble for even i and in its high nibble for odd i.
 *
 * Logical shapes:
 *   source: FP32, numel = S*L. codes: U8, numel/2. group_scales: U8, numel/16.
 *   slice_scales: FP32, S. All tensors are contiguous; their declared shapes are not otherwise
 *   interpreted.
 *
 * Supported domain:
 *   L is a positive multiple of 16 and divides numel. Every source value is finite. source and
 *   slice_scales are 16-byte and 4-byte aligned, codes is 8-byte aligned.
 *
 * Numeric:
 *   Exact with respect to the formulas above: the result is bit-identical to an FP32 host
 *   evaluation with RNE E4M3/E2M1 satfinite encoders. A group scale encodes only magnitudes, so
 *   its sign bit is always clear. Decoding with decode_nvfp4_slices reconstructs every value
 *   within half an E2M1 step of its group scale; groups below the E4M3 subnormal range of their
 *   slice decode to zero.
 *
 * Effects:
 *   Writes all of codes, group_scales and slice_scales. No output aliases the source or another
 *   output.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   One CUDA block per slice on stream; capturable in a CUDA Graph.
 */
void encode_nvfp4_slices(const Tensor& source, std::int64_t slice_elements, Tensor& codes,
                         Tensor& group_scales, Tensor& slice_scales, cudaStream_t stream);

/**
 * Op: decode_nvfp4_slices
 *
 * Math / indexing:
 *   With the indexing of encode_nvfp4_slices, every FP32 product is one RNE operation evaluated
 *   in the written order:
 *     destination[i] = e2m1(code(i)) * (e4m3(group_scales[j]) * slice_scales[s])
 *
 * Logical shapes:
 *   codes: U8, numel/2. group_scales: U8, numel/16. slice_scales: FP32, numel/L.
 *   destination: FP32, numel = S*L. All tensors are contiguous.
 *
 * Supported domain:
 *   L is a positive multiple of 16 and divides the destination numel; slice_scales holds finite
 *   non-negative values. destination is 16-byte aligned and codes is 8-byte aligned.
 *
 * Numeric:
 *   Exact with respect to the formula above.
 *
 * Effects:
 *   Writes the full destination. The destination aliases no input.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Grid-stride over groups on stream; capturable in a CUDA Graph.
 */
void decode_nvfp4_slices(const Tensor& codes, const Tensor& group_scales,
                         const Tensor& slice_scales, std::int64_t slice_elements,
                         Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops
