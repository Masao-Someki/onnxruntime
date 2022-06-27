// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications by Masao Someki
// Copyright (c) 2022 Masao Someki

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
enum MaskIndexType {
  kMaskIndexEnd = 0,
  kMaskIndexEndAndStart,
  kMaskRaw,
  kMask3D,
  kMaskDummy,  // Dummy mask with shape [1, 1] or [batch_size, 1]
  kMask4D      // Megatron GPT2 mask with shape [batch_size, 1, max_sequence_length, max_sequence_length]
};

static void RunCrossAttentionTest(
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& q_weights_data,
    const std::vector<float>& kv_weights_data,
    const std::vector<float>& q_bias_data,
    const std::vector<float>& kv_bias_data,
    const std::vector<int32_t>& mask_index_data,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int kv_sequence_length,
    int hidden_size,
    int num_heads,
    const std::vector<int32_t> qkv_sizes = {},
    MaskIndexType mask_index_type = kMaskIndexEnd,
    bool only_enable_cpu = false,
    bool only_enable_cuda = false,
    bool use_float16 = false
) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !only_enable_cpu;
//   bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get()) && !is_weights_constant && !only_enable_cpu;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !only_enable_cuda;

  if (enable_cpu || enable_cuda) {
    OpTester tester("CrossAttention", 1, onnxruntime::kENDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));

    if (qkv_sizes.size() != 0) {
      std::vector<int64_t> sizes_attribute{qkv_sizes[0], qkv_sizes[1], qkv_sizes[2]};
      tester.AddAttribute<std::vector<int64_t>>("qkv_hidden_sizes", sizes_attribute);
    }

    std::vector<int64_t> query_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> key_dims = {batch_size, kv_sequence_length, hidden_size};
    std::vector<int64_t> q_weights_dims = {hidden_size, hidden_size};
    std::vector<int64_t> kv_weights_dims = {hidden_size, 2 * hidden_size};
    std::vector<int64_t> q_bias_dims = {hidden_size};
    std::vector<int64_t> kv_bias_dims = {2 * hidden_size};

    std::vector<int64_t> mask_index_dims_1 = {batch_size};
    std::vector<int64_t> mask_index_dims_2 = {2 * batch_size};
    std::vector<int64_t> mask_index_dims_3 = {batch_size, kv_sequence_length};
    std::vector<int64_t> mask_index_dims_4 = {batch_size, 1};
    std::vector<int64_t> mask_index_dims_5 = {batch_size, sequence_length, kv_sequence_length};
    std::vector<int64_t> mask_index_dims;
    switch (mask_index_type) {
      case kMaskIndexEnd:
        mask_index_dims = mask_index_dims_1;
        break;
      case kMaskIndexEndAndStart:
        mask_index_dims = mask_index_dims_2;
        break;
      case kMaskRaw:
        mask_index_dims = mask_index_dims_3;
        break;
      case kMaskDummy:
        mask_index_dims = mask_index_dims_4;
        break;
      case kMask3D:
        mask_index_dims = mask_index_dims_5;
        break;
      default:
        assert(0);  // shall not reach here.
        break;
    }

    const std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

    tester.AddInput<float>("query", query_dims, query_data);
    tester.AddInput<float>("key", key_dims, key_data);
    tester.AddInput<float>("q_weight", q_weights_dims, q_weights_data);
    tester.AddInput<float>("kv_weight", kv_weights_dims, kv_weights_data);
    tester.AddInput<float>("q_bias", q_bias_dims, q_bias_data);
    tester.AddInput<float>("kv_bias", kv_bias_dims, kv_bias_data);

    if (mask_index_data.size() > 0) {  // mask index is optional.
      tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);
    } else {
      tester.AddOptionalInputEdge<int32_t>();
    }

    tester.AddOutput<float>("output", output_dims, output_data);

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    // if (enable_rocm) {
    //   std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    //   execution_providers.push_back(DefaultRocmExecutionProvider());
    //   tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    // }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}


TEST(CrossAttentionTest, CrossAttentionBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index < sequence_length
  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  //self-attn without cacheint batch_size,
  RunCrossAttentionTest(input_data, input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data, mask_index_data,
    output_data, batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

TEST(CrossAttentionTest, CrossAttentionBatch1DiffLength) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 4;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> q_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> kv_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index < sequence_length
  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  //self-attn without cacheint batch_size,
  RunCrossAttentionTest(q_input_data, kv_input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data, mask_index_data,
    output_data, batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

TEST(CrossAttentionTest, CrossAttentionMaskBatch2) {
  int batch_size = 2;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L, 2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  //self-attn without cacheint batch_size,
  RunCrossAttentionTest(input_data, input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data, mask_index_data,
    output_data, batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

TEST(CrossAttentionTest, CrossAttentionMaskBatch2DiffLength) {
  int batch_size = 2;
  int sequence_length = 2;
  int kv_sequence_length = 4;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> q_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> kv_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index < sequence_length
  std::vector<int32_t> mask_index_data = {2L, 2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  //self-attn without cacheint batch_size,
  RunCrossAttentionTest(q_input_data, kv_input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data, mask_index_data,
    output_data, batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

TEST(CrossAttentionMaskTest, CrossAttentionMaskPartialSequence) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> q_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> kv_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {1L};

  std::vector<float> output_data = {
      8.6899995803833008f, -0.13000002503395081f, 4.25f, 5.6499996185302734f,
      8.6899995803833008f, -0.13000002503395081f, 4.2499995231628418f, 5.6499991416931152f};

  //self-attn without cacheint batch_size,
  RunCrossAttentionTest(q_input_data, kv_input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data, mask_index_data,
    output_data, batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

TEST(CrossAttentionMaskTest, CrossAttentionMaskPartialSequenceDiffLength) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 4;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> q_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> kv_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {1L};

  std::vector<float> output_data = {
      8.6899995803833008f, -0.13000002503395081f, 4.25f, 5.6499996185302734f,
      8.6899995803833008f, -0.13000002503395081f, 4.2499995231628418f, 5.6499991416931152f};

  //self-attn without cacheint batch_size,
  RunCrossAttentionTest(q_input_data, kv_input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data, mask_index_data,
    output_data, batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

TEST(CrossAttentionMaskTest, CrossAttentionMaskExceedSequence) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> q_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> kv_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {3L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  //self-attn without cacheint batch_size,
  RunCrossAttentionTest(q_input_data, kv_input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data, mask_index_data,
    output_data, batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

TEST(CrossAttentionMaskTest, CrossAttentionNoMaskIndex) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> q_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> kv_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  //self-attn without cacheint batch_size,
  RunCrossAttentionTest(q_input_data, kv_input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data, mask_index_data,
    output_data, batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

TEST(CrossAttentionMaskTest, CrossAttentionNoMaskIndexDiffLength) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 4;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> q_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> kv_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  //self-attn without cacheint batch_size,
  RunCrossAttentionTest(q_input_data, kv_input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data, mask_index_data,
    output_data, batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

TEST(CrossAttentionMaskTest, CrossAttentionBatch2MaskIndex2) {
  int batch_size = 2;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> q_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> kv_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2, 2};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  //self-attn without cacheint batch_size,
  RunCrossAttentionTest(q_input_data, kv_input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data, mask_index_data,
    output_data, batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

TEST(CrossAttentionMaskTest, CrossAttentionBatch2MaskIndex2DiffLength) {
  int batch_size = 2;
  int sequence_length = 2;
  int kv_sequence_length = 4;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> q_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> kv_input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2, 2};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  //self-attn without cacheint batch_size,
  RunCrossAttentionTest(q_input_data, kv_input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data, mask_index_data,
    output_data, batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

}  // namespace test
}  // namespace onnxruntime