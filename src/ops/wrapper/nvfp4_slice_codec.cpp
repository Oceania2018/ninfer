#include "ninfer/ops/nvfp4_slice_codec.h"

#include "ops/launcher/nvfp4_slice_codec.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned(const void* data, std::uintptr_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(data) & (alignment - 1U)) == 0;
}

void require_contiguous(const Tensor& tensor, DType dtype, std::int64_t numel, const char* op,
                        const char* name) {
    if (tensor.dtype != dtype || !tensor.is_contiguous() || tensor.data == nullptr ||
        tensor.numel() != numel) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " has the wrong dtype, size or layout");
    }
}

std::int64_t validate_geometry(std::int64_t numel, std::int64_t slice_elements, const char* op) {
    if (slice_elements <= 0 || slice_elements % kNvfp4SliceGroup != 0 || numel <= 0 ||
        numel % slice_elements != 0) {
        throw std::invalid_argument(std::string(op) +
                                    ": slices must be positive multiples of 16 that divide numel");
    }
    return numel / slice_elements;
}

void require_distinct(const Tensor& left, const Tensor& right, const char* op) {
    const auto* left_begin  = static_cast<const std::byte*>(left.data);
    const auto* right_begin = static_cast<const std::byte*>(right.data);
    if (left_begin < right_begin + right.bytes() && right_begin < left_begin + left.bytes()) {
        throw std::invalid_argument(std::string(op) + ": tensors must not alias");
    }
}

} // namespace

void encode_nvfp4_slices(const Tensor& source, std::int64_t slice_elements, Tensor& codes,
                         Tensor& group_scales, Tensor& slice_scales, cudaStream_t stream) {
    constexpr const char* op  = "encode_nvfp4_slices";
    const std::int64_t numel  = source.numel();
    const std::int64_t slices = validate_geometry(numel, slice_elements, op);
    require_contiguous(source, DType::FP32, numel, op, "source");
    require_contiguous(codes, DType::U8, numel / 2, op, "codes");
    require_contiguous(group_scales, DType::U8, numel / kNvfp4SliceGroup, op, "group_scales");
    require_contiguous(slice_scales, DType::FP32, slices, op, "slice_scales");
    if (!aligned(source.data, 16) || !aligned(codes.data, 8) || !aligned(slice_scales.data, 4)) {
        throw std::invalid_argument("encode_nvfp4_slices: tensors are under-aligned");
    }
    require_distinct(source, codes, op);
    require_distinct(source, group_scales, op);
    require_distinct(source, slice_scales, op);
    require_distinct(codes, group_scales, op);
    require_distinct(codes, slice_scales, op);
    require_distinct(group_scales, slice_scales, op);
    detail::encode_nvfp4_slices_launch(source, slice_elements, codes, group_scales, slice_scales,
                                       stream);
}

void decode_nvfp4_slices(const Tensor& codes, const Tensor& group_scales,
                         const Tensor& slice_scales, std::int64_t slice_elements,
                         Tensor& destination, cudaStream_t stream) {
    constexpr const char* op  = "decode_nvfp4_slices";
    const std::int64_t numel  = destination.numel();
    const std::int64_t slices = validate_geometry(numel, slice_elements, op);
    require_contiguous(destination, DType::FP32, numel, op, "destination");
    require_contiguous(codes, DType::U8, numel / 2, op, "codes");
    require_contiguous(group_scales, DType::U8, numel / kNvfp4SliceGroup, op, "group_scales");
    require_contiguous(slice_scales, DType::FP32, slices, op, "slice_scales");
    if (!aligned(destination.data, 16) || !aligned(codes.data, 8) ||
        !aligned(slice_scales.data, 4)) {
        throw std::invalid_argument("decode_nvfp4_slices: tensors are under-aligned");
    }
    require_distinct(destination, codes, op);
    require_distinct(destination, group_scales, op);
    require_distinct(destination, slice_scales, op);
    detail::decode_nvfp4_slices_launch(codes, group_scales, slice_scales, slice_elements,
                                       destination, stream);
}

} // namespace ninfer::ops
