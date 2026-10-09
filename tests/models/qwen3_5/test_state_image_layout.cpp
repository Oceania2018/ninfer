#include "core/layout.h"
#include "models/qwen3_5/state/state_image.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

namespace q36 = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q36::StateImageDeviceLayout plan(bool dflash) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 2,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 3,
                .slot_count     = 2,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 7,
    };
    if (dflash) {
        spec.dflash_local =
            q36::DFlashLocalStateSpec{.layers = 2, .capacity = 17, .kv_heads = 2, .head_dim = 4};
    }
    ninfer::LayoutBuilder builder;
    return q36::plan_state_image_device_pool(builder, spec);
}

q36::StateImageDeviceLayout plan_encoded(ninfer::HostStateStorage storage,
                                         std::int32_t key_head_dim = 16) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 3,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 3,
                .value_head_dim = 4,
                .key_head_dim   = key_head_dim,
                .slot_count     = 2,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden         = 7,
        .host_recurrent = storage,
    };
    ninfer::LayoutBuilder builder;
    return q36::plan_state_image_device_pool(builder, spec);
}

void test_encoded_layouts() {
    constexpr std::size_t elements = 16 * 4 * 3;
    const auto fp32                = plan_encoded(ninfer::HostStateStorage::Fp32);
    const auto bf16                = plan_encoded(ninfer::HostStateStorage::BFloat16);
    const auto nvfp4               = plan_encoded(ninfer::HostStateStorage::Nvfp4Group16);

    expect(fp32.host.linear_recurrent_layer_bytes == elements * 4 && !fp32.recurrent_staging,
           "FP32 Host recurrent state keeps the Device bytes and needs no staging");
    expect(bf16.host.linear_recurrent_layer_bytes == elements * 2 && bf16.recurrent_staging &&
               bf16.recurrent_staging->bytes == bf16.host.linear_recurrent_layer_bytes,
           "BF16 Host recurrent layer is half the FP32 layer and staged whole");

    const q36::StateImageRecurrentEncoding& encoding = nvfp4.host.recurrent_encoding;
    // 96 code bytes, 12 group scales padded to offset 112, then 12 bytes of head scales.
    expect(encoding.values_bytes == elements / 2 && encoding.group_scale_offset == 96 &&
               encoding.group_scale_bytes == elements / 16 && encoding.slice_scale_offset == 112 &&
               encoding.slice_scale_bytes == 3 * sizeof(float) &&
               nvfp4.host.linear_recurrent_layer_bytes == 128,
           "NVFP4 Host recurrent layer packs codes, group scales and head scales");
    expect(nvfp4.recurrent_staging &&
               nvfp4.recurrent_staging->bytes == nvfp4.host.linear_recurrent_layer_bytes,
           "NVFP4 Host recurrent layer is staged whole");
    expect(nvfp4.host.image_bytes < bf16.host.image_bytes &&
               bf16.host.image_bytes < fp32.host.image_bytes,
           "Host StateImage size orders FP32 > BF16 > NVFP4");

    const auto fp32_work  = q36::state_image_transfer_work(fp32.host);
    const auto nvfp4_work = q36::state_image_transfer_work(nvfp4.host);
    expect(nvfp4_work.copy_operations == fp32_work.copy_operations + 3,
           "an encoded recurrent layer adds one encode/decode launch per layer");
    expect(nvfp4_work.payload_bytes == nvfp4.host.linear_conv.bytes +
                                           nvfp4.host.linear_recurrent.bytes +
                                           nvfp4.host.continuation_hidden.bytes,
           "encoded StateImage work moves the encoded bytes");

    bool rejected = false;
    try {
        (void)plan_encoded(ninfer::HostStateStorage::Nvfp4Group16, 3);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "NVFP4 Host state rejects value heads that are not whole 16-value groups");
}

} // namespace

int main() {
    const q36::StateImageDeviceLayout common = plan(false);
    const auto common_work                   = q36::state_image_transfer_work(common.host);
    expect(common_work.payload_bytes == common.host.linear_conv.bytes +
                                            common.host.linear_recurrent.bytes +
                                            common.host.continuation_hidden.bytes,
           "common StateImage work payload includes layout padding");
    expect(common_work.copy_operations == 2 * common.host.spec.linear.layers + 1,
           "common StateImage work does not match physical CUDA copies");

    const q36::StateImageDeviceLayout dflash = plan(true);
    const auto full_work                     = q36::state_image_transfer_work(dflash.host);
    const auto local_work                    = q36::dflash_local_transfer_work(dflash.host);
    const std::uint64_t local_bytes =
        2ULL * dflash.host.dflash_local_layer_bytes * dflash.host.spec.dflash_local->layers;
    expect(local_work.payload_bytes == local_bytes &&
               local_work.copy_operations == 2 * dflash.host.spec.dflash_local->layers,
           "DFlash-local work does not match physical K/V layer copies");
    expect(full_work.payload_bytes == common_work.payload_bytes + local_work.payload_bytes &&
               full_work.copy_operations ==
                   common_work.copy_operations + local_work.copy_operations,
           "full DFlash StateImage work is not common plus local state");

    test_encoded_layouts();

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
