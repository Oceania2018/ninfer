#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void encode_nvfp4_slices_launch(const Tensor& source, std::int64_t slice_elements, Tensor& codes,
                                Tensor& group_scales, Tensor& slice_scales, cudaStream_t stream);
void decode_nvfp4_slices_launch(const Tensor& codes, const Tensor& group_scales,
                                const Tensor& slice_scales, std::int64_t slice_elements,
                                Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops::detail
