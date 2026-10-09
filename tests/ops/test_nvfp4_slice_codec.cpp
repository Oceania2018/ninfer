#include "ninfer/ops/nvfp4_slice_codec.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// Independent host oracle: exhaustive nearest-value search with ties to the even code.
float decode_e4m3_positive(std::uint8_t code) {
    const int exponent = (code >> 3) & 0x0f;
    const int mantissa = code & 0x07;
    if (exponent == 0) return std::ldexp(static_cast<float>(mantissa), -9);
    return std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
}

std::uint8_t encode_e4m3_positive_rne_satfinite(float magnitude) {
    if (magnitude >= 448.0F) return 0x7e;
    for (std::uint8_t upper = 1; upper <= 0x7e; ++upper) {
        const float upper_value = decode_e4m3_positive(upper);
        if (upper_value < magnitude) continue;
        const auto lower        = static_cast<std::uint8_t>(upper - 1);
        const float lower_error = magnitude - decode_e4m3_positive(lower);
        const float upper_error = upper_value - magnitude;
        if (lower_error < upper_error) return lower;
        if (upper_error < lower_error) return upper;
        return (lower & 1U) == 0U ? lower : upper;
    }
    return 0x7e;
}

float decode_e2m1(std::uint8_t code) {
    constexpr std::array<float, 8> values{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude = values[code & 0x07U];
    return (code & 0x08U) != 0 ? -magnitude : magnitude;
}

std::uint8_t encode_e2m1_rne_satfinite(float value) {
    const auto sign       = static_cast<std::uint8_t>(std::signbit(value) ? 0x08U : 0U);
    const float magnitude = std::abs(value);
    if (magnitude >= 6.0F) return static_cast<std::uint8_t>(7U | sign);
    for (std::uint8_t upper = 1; upper <= 7; ++upper) {
        const float upper_value = decode_e2m1(upper);
        if (upper_value < magnitude) continue;
        const auto lower        = static_cast<std::uint8_t>(upper - 1);
        const float lower_error = magnitude - decode_e2m1(lower);
        const float upper_error = upper_value - magnitude;
        std::uint8_t selected   = lower_error < upper_error   ? lower
                                  : upper_error < lower_error ? upper
                                  : (lower & 1U) == 0U        ? lower
                                                              : upper;
        return static_cast<std::uint8_t>(selected | sign);
    }
    return static_cast<std::uint8_t>(7U | sign);
}

struct Encoded {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint8_t> group_scales;
    std::vector<float> slice_scales;
};

Encoded oracle_encode(const std::vector<float>& source, std::size_t slice_elements) {
    const std::size_t slices = source.size() / slice_elements;
    const std::size_t groups = slice_elements / 16;
    Encoded out{std::vector<std::uint8_t>(source.size() / 2, 0),
                std::vector<std::uint8_t>(source.size() / 16, 0), std::vector<float>(slices)};
    for (std::size_t s = 0; s < slices; ++s) {
        float slice_max = 0.0F;
        for (std::size_t k = 0; k < slice_elements; ++k) {
            slice_max = std::max(slice_max, std::abs(source[s * slice_elements + k]));
        }
        const float slice_scale = slice_max / ops::kNvfp4SliceRange;
        out.slice_scales[s]     = slice_scale;
        for (std::size_t g = 0; g < groups; ++g) {
            const std::size_t first = s * slice_elements + g * 16;
            float group_max         = 0.0F;
            for (std::size_t k = 0; k < 16; ++k) {
                group_max = std::max(group_max, std::abs(source[first + k]));
            }
            std::uint8_t scale_code = 0;
            if (slice_scale != 0.0F) {
                const float sixth    = group_max / 6.0F;
                const float relative = sixth / slice_scale;
                scale_code           = encode_e4m3_positive_rne_satfinite(relative);
            }
            out.group_scales[first / 16] = scale_code;
            const float scale            = decode_e4m3_positive(scale_code) * slice_scale;
            for (std::size_t k = 0; k < 16; ++k) {
                const std::size_t i = first + k;
                const std::uint8_t code =
                    scale == 0.0F ? 0 : encode_e2m1_rne_satfinite(source[i] / scale);
                out.codes[i / 2] |= static_cast<std::uint8_t>((i % 2) == 0 ? code : code << 4);
            }
        }
    }
    return out;
}

std::vector<float> oracle_decode(const Encoded& encoded, std::size_t count,
                                 std::size_t slice_elements) {
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint8_t byte  = encoded.codes[i / 2];
        const auto code          = static_cast<std::uint8_t>((i % 2) == 0 ? byte & 0x0f : byte >> 4);
        const float scale        = decode_e4m3_positive(encoded.group_scales[i / 16]) *
                                   encoded.slice_scales[i / slice_elements];
        out[i]                   = decode_e2m1(code) * scale;
    }
    return out;
}

std::vector<std::uint32_t> float_bits(const std::vector<float>& values) {
    std::vector<std::uint32_t> bits(values.size());
    std::memcpy(bits.data(), values.data(), values.size() * sizeof(float));
    return bits;
}

int run_case(const char* label, const std::vector<float>& source, std::size_t slice_elements,
             bool report_error) {
    const std::size_t count  = source.size();
    const std::size_t slices = count / slice_elements;
    const Encoded expected   = oracle_encode(source, slice_elements);
    const auto n             = static_cast<std::int32_t>(count);

    GuardedDeviceBuffer device_source(count * sizeof(float));
    GuardedDeviceBuffer device_codes(count / 2);
    GuardedDeviceBuffer device_group_scales(count / 16);
    GuardedDeviceBuffer device_slice_scales(slices * sizeof(float));
    GuardedDeviceBuffer device_decoded(count * sizeof(float));
    device_source.copy_from_host(source.data(), device_source.bytes());

    Tensor source_tensor(device_source.data(), DType::FP32, {n});
    Tensor codes(device_codes.data(), DType::U8, {n / 2});
    Tensor group_scales(device_group_scales.data(), DType::U8, {n / 16});
    Tensor slice_scales(device_slice_scales.data(), DType::FP32, {static_cast<std::int32_t>(slices)});
    Tensor decoded(device_decoded.data(), DType::FP32, {n});
    const auto slice = static_cast<std::int64_t>(slice_elements);
    ops::encode_nvfp4_slices(source_tensor, slice, codes, group_scales, slice_scales, nullptr);
    ops::decode_nvfp4_slices(codes, group_scales, slice_scales, slice, decoded, nullptr);
    cuda_synchronize();

    int failures = 0;
    failures += verify_exact(label, from_device<std::uint8_t>(device_codes.data(), count / 2),
                             expected.codes);
    failures += verify_exact(label, from_device<std::uint8_t>(device_group_scales.data(), count / 16),
                             expected.group_scales);
    failures += verify_exact(label,
                             float_bits(from_device<float>(device_slice_scales.data(), slices)),
                             float_bits(expected.slice_scales));
    const std::vector<float> reference = oracle_decode(expected, count, slice_elements);
    const std::vector<float> got       = from_device<float>(device_decoded.data(), count);
    failures += verify_exact(label, float_bits(got), float_bits(reference));
    failures += verify_exact("encode source unchanged",
                             float_bits(from_device<float>(device_source.data(), count)),
                             float_bits(source));
    failures += device_source.verify_guards("source");
    failures += device_codes.verify_guards("codes");
    failures += device_group_scales.verify_guards("group scales");
    failures += device_slice_scales.verify_guards("slice scales");
    failures += device_decoded.verify_guards("decoded");

    if (report_error) {
        double error_sq = 0.0;
        double value_sq = 0.0;
        double max_rel  = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double error = static_cast<double>(got[i]) - source[i];
            error_sq += error * error;
            value_sq += static_cast<double>(source[i]) * source[i];
        }
        for (std::size_t s = 0; s < slices; ++s) {
            double slice_max = 0.0;
            double slice_err = 0.0;
            for (std::size_t k = 0; k < slice_elements; ++k) {
                const std::size_t i = s * slice_elements + k;
                slice_max = std::max(slice_max, std::abs(static_cast<double>(source[i])));
                slice_err = std::max(slice_err, std::abs(static_cast<double>(got[i]) - source[i]));
            }
            if (slice_max > 0.0) max_rel = std::max(max_rel, slice_err / slice_max);
        }
        std::cout << "  " << label << ": relative RMS error " << std::sqrt(error_sq / value_sq)
                  << ", max error / head max " << max_rel << '\n';
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // Production geometry: 128x128 recurrent matrix per value head, 48 heads in one layer.
    // Gaussian values with per-row magnitudes spanning four decades.
    {
        constexpr std::size_t head = 128 * 128;
        std::vector<float> source(head * 48);
        std::mt19937 rng(11U);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (std::size_t i = 0; i < source.size(); ++i) {
            const float decade = std::pow(10.0F, -static_cast<float>((i / 128) % 4));
            source[i]          = normal(rng) * decade * 0.05F;
        }
        failures += run_case("nvfp4 slices [16384 x 48]", source, head, true);
    }

    // Edge slices of one group each: all zeros, a group below the E4M3 range of its slice, exact
    // grid values, saturation boundaries, negative zero and ties between E2M1 codes.
    {
        std::vector<float> source(16 * 6, 0.0F);
        for (std::size_t k = 0; k < 16; ++k) source[16 + k] = (k == 0) ? 1.0F : 1e-9F;
        const float grid[16] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
                                -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
        std::copy(std::begin(grid), std::end(grid), source.begin() + 32);
        const float ties[16] = {0.25F, 0.75F, 1.25F, 1.75F, 2.5F, 3.5F, 5.0F, 6.0F,
                                -0.25F, -0.75F, -1.25F, -1.75F, -2.5F, -3.5F, -5.0F, -6.0F};
        std::copy(std::begin(ties), std::end(ties), source.begin() + 48);
        for (std::size_t k = 0; k < 16; ++k) {
            source[64 + k] = (k % 2 == 0 ? 1.0F : -1.0F) * 3.0e38F / static_cast<float>(k + 1);
            source[80 + k] = std::ldexp(1.0F, -126) * static_cast<float>(k);
        }
        failures += run_case("nvfp4 edge slices [16 x 6]", source, 16, false);
    }

    // Two-group slices: the second group sits below the E4M3 subnormal range of its slice and
    // decodes to zero, or lands on the smallest subnormal group scales.
    {
        std::vector<float> source(32 * 3);
        fill_uniform(source, 31U, -1.0F, 1.0F);
        source[0] = 1.0F;
        source[32] = 1.0F;
        source[64] = 1.0F;
        for (std::size_t k = 16; k < 32; ++k) {
            source[k] *= 1e-7F;
            source[32 + k] *= 2.5e-5F;
            source[64 + k] *= 7.0e-5F;
        }
        failures += run_case("nvfp4 underflowing groups [32 x 3]", source, 32, false);
    }

    // Multi-group slices whose groups take distinct E4M3 scales.
    {
        std::vector<float> source(48 * 5);
        fill_uniform(source, 23U, -8.0F, 8.0F);
        for (std::size_t i = 0; i < source.size(); ++i) {
            source[i] *= std::pow(2.0F, -static_cast<float>((i / 16) % 3) * 5.0F);
        }
        failures += run_case("nvfp4 slices [48 x 5]", source, 48, false);
    }

    std::cout << (failures ? "FAIL" : "OK") << " encode/decode_nvfp4_slices\n";
    return failures ? 1 : 0;
}
