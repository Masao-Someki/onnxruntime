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

static void RunRelPosAttentionTest(
    const std::vector<float>& input_data,
    const std::vector<float>& weights_data,
    const std::vector<float>& bias_data,
    const std::vector<float>& pos_emb,
    const std::vector<float>& pos_weights,
    const std::vector<float>& pos_bias_u,
    const std::vector<float>& pos_bias_v,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int pos_sequence_length,
    int hidden_size,
    int num_heads,
    int head_size,
    const std::vector<int32_t> qkv_sizes = {},
    bool only_enable_cpu = false,
    bool only_enable_cuda = false,
    bool use_float16 = false) {
  // int min_cuda_architecture = use_float16 ? 530 : 0;
  // bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !only_enable_cpu;
  bool enable_cuda = false;
  //   bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get()) && !is_weights_constant && !only_enable_cpu;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !only_enable_cuda;

  if (enable_cpu || enable_cuda) {
    OpTester tester("RelPosAttention", 1, onnxruntime::kENDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));

    if (qkv_sizes.size() != 0) {
      std::vector<int64_t> sizes_attribute{qkv_sizes[0], qkv_sizes[1], qkv_sizes[2]};
      tester.AddAttribute<std::vector<int64_t>>("qkv_hidden_sizes", sizes_attribute);
    }

    std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> weights_dims = {hidden_size, 3 * hidden_size};
    std::vector<int64_t> bias_dims = {3 * hidden_size};
    std::vector<int64_t> pos_emb_dims = {batch_size, pos_sequence_length, hidden_size};
    std::vector<int64_t> pos_weights_dims = {hidden_size, hidden_size};
    std::vector<int64_t> pos_bias_u_dims = {num_heads, head_size};
    std::vector<int64_t> pos_bias_v_dims = {num_heads, head_size};

    const std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

    tester.AddInput<float>("input", input_dims, input_data);
    tester.AddInput<float>("weights", weights_dims, weights_data);
    tester.AddInput<float>("pos_emb", pos_emb_dims, pos_emb);
    tester.AddInput<float>("pos_weights", pos_weights_dims, pos_weights);
    tester.AddInput<float>("bias", bias_dims, bias_data);
    tester.AddInput<float>("pos_bias_u", pos_bias_u_dims, pos_bias_u);
    tester.AddInput<float>("pos_bias_v", pos_bias_v_dims, pos_bias_v);

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

TEST(RelPosAttentionTest, RelPosAttentionBatch1) {
  int batch_size = 1;
  int sequence_length = 3;
  int pos_sequence_length = 2 * sequence_length - 1;
  int hidden_size = 8;
  int number_of_heads = 2;
  int head_size = hidden_size / number_of_heads;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f, 0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f, 0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f, 0.8f, -0.5f, 0.0f, 1.0f};

  std::vector<float> pos_emb = {
      0.2f, -0.0f, 0.2f, 0.4f, 0.2f, -0.0f, 0.2f, 0.4f,
      -0.7f, 1.0f, -0.3f, -0.5f, -0.7f, 1.0f, -0.3f, -0.5f,
      -1.3f, 1.0f, -0.5f, 1.0f, -1.3f, 1.0f, -0.5f, 1.0f,
      0.2f, -0.0f, 0.2f, 0.4f, 0.2f, -0.0f, 0.2f, 0.4f,
      -0.7f, 1.0f, -0.3f, -0.5f, -0.7f, 1.0f, -0.3f, -0.5f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 0.1f, -0.2f, 0.3f, 1.0f,
      1.1f, 0.3f, 0.5f, 0.2f, 1.1f, 0.3f, 0.5f, 0.2f, 
      0.3f, -0.6f, 1.5f, 2.0f, 0.3f, -0.6f, 1.5f, 2.0f,

      0.5f, 0.1f, 0.4f, 1.6f, 0.5f, 0.1f, 0.4f, 1.6f,
      1.0f, 2.0f, 0.4f, 0.8f, 1.0f, 2.0f, 0.4f, 0.8f,
      0.9f, 0.1f, -1.3f, 0.7f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 0.3f, 0.2f, 4.0f, 2.2f,
      1.6f, 1.1f, 0.7f, 0.2f, 1.6f, 1.1f, 0.7f, 0.2f,
      0.4f, 1.0f, 1.2f, 0.5f, 0.4f, 1.0f, 1.2f, 0.5f,

      0.2f, 0.1f, 0.4f, 1.6f, 0.2f, 0.1f, 0.4f, 1.6f,
      2.4f, 3.3f, 2.1f, 4.2f, 2.4f, 3.3f, 2.1f, 4.2f,
      8.4f, 0.0f, 2.1f, 3.2f, 8.4f, 0.0f, 2.1f, 3.2f,

      0.1f, -0.2f, 0.3f, 1.0f, 0.1f, -0.2f, 0.3f, 1.0f,
     1.1f, 0.3f, 0.5f, 0.2f, 1.1f, 0.3f, 0.5f, 0.2f, 
      0.3f, -0.6f, 1.5f, 2.0f, 0.3f, -0.6f, 1.5f, 2.0f,

      0.5f, 0.1f, 0.4f, 1.6f, 0.5f, 0.1f, 0.4f, 1.6f,
      1.0f, 2.0f, 0.4f, 0.8f, 1.0f, 2.0f, 0.4f, 0.8f,
      0.9f, 0.1f, -1.3f, 0.7f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 0.3f, 0.2f, 4.0f, 2.2f,
      1.6f, 1.1f, 0.7f, 0.2f, 1.6f, 1.1f, 0.7f, 0.2f,
      0.4f, 1.0f, 1.2f, 0.5f, 0.4f, 1.0f, 1.2f, 0.5f,

      0.2f, 0.1f, 0.4f, 1.6f, 0.2f, 0.1f, 0.4f, 1.6f,
      2.4f, 3.3f, 2.1f, 4.2f, 2.4f, 3.3f, 2.1f, 4.2f,
      8.4f, 0.0f, 2.1f, 3.2f, 8.4f, 0.0f, 2.1f, 3.2f,
      };

  std::vector<float> pos_weight_data = {
     1.1f, 0.3f, 0.5f, 0.2f, 1.1f, 0.3f, 0.5f, 0.2f, 
      1.0f, 2.0f, 0.4f, 0.8f, 1.0f, 2.0f, 0.4f, 0.8f,
      1.6f, 1.1f, 0.7f, 0.2f, 1.6f, 1.1f, 0.7f, 0.2f,
      2.4f, 3.3f, 2.1f, 4.2f, 2.4f, 3.3f, 2.1f, 4.2f,
     1.1f, 0.3f, 0.5f, 0.2f, 1.1f, 0.3f, 0.5f, 0.2f, 
      1.0f, 2.0f, 0.4f, 0.8f, 1.0f, 2.0f, 0.4f, 0.8f,
      1.6f, 1.1f, 0.7f, 0.2f, 1.6f, 1.1f, 0.7f, 0.2f,
      2.4f, 3.3f, 2.1f, 4.2f, 2.4f, 3.3f, 2.1f, 4.2f};

  std::vector<float> bias_data = {
    -0.5f, 0.6f, 1.2f, 2.1f, -0.5f, 0.6f, 1.2f, 2.1f,
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.7f, 0.2f, 1.2f,
    0.5f, 0.4f, 0.3f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f
  };

  std::vector<float> pos_bias_u = {
      0.3f, -0.6f, 0.3f, -0.6f,
      0.9f, 0.1f, 0.9f, 0.1f};

  std::vector<float> pos_bias_v = {
      1.2f, 0.5f, 1.2f, 0.5f,
      8.4f, 0.0f, 8.4f, 0.0f};

  std::vector<float> output_data = {
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.879861831665039f, -0.6599940657615662f, 8.1999530792236328f, 10.0999450683593750f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f};

  // self-attn without cacheint batch_size,
  RunRelPosAttentionTest(input_data, weight_data, bias_data, pos_emb, pos_weight_data,
                         pos_bias_u, pos_bias_v, output_data,
                         batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, head_size);
}

TEST(RelPosAttentionTest, RelPosAttentionBatch2) {
  int batch_size = 2;
  int sequence_length = 3;
  int pos_sequence_length = 2 * sequence_length - 1;
  int hidden_size = 8;
  int number_of_heads = 2;
  int head_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.0f, 0.8f, -0.5f, 0.0f, 1.0f,
      0.5f, 0.2f, 0.3f, -0.6f, 0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f, 0.8f, -0.5f, 0.0f, 1.0f,
      0.8f, -0.5f, 0.0f, 1.f, 0.8f, -0.5f, 0.0f, 1.0f,
      0.5f, 0.2f, 0.3f, -0.6f, 0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.0f, 0.8f, -0.5f, 0.0f, 1.0f};

  std::vector<float> pos_emb = {
      0.2f, -0.0f, 0.2f, 0.4f, 0.2f, -0.0f, 0.2f, 0.4f,
      -0.7f, 1.0f, -0.3f, -0.5f, -0.7f, 1.0f, -0.3f, -0.5f,
      -1.3f, 1.0f, -0.5f, 1.0f, -1.3f, 1.0f, -0.5f, 1.0f,
      0.2f, -0.0f, 0.2f, 0.4f, 0.2f, -0.0f, 0.2f, 0.4f,
      -0.7f, 1.0f, -0.3f, -0.5f, -0.7f, 1.0f, -0.3f, -0.5f,
      0.2f, -0.0f, 0.2f, 0.4f, 0.2f, -0.0f, 0.2f, 0.4f,
      -0.7f, 1.0f, -0.3f, -0.5f, -0.7f, 1.0f, -0.3f, -0.5f,
      -1.3f, 1.0f, -0.5f, 1.0f, -1.3f, 1.0f, -0.5f, 1.0f,
      0.2f, -0.0f, 0.2f, 0.4f, 0.2f, -0.0f, 0.2f, 0.4f,
      -0.7f, 1.0f, -0.3f, -0.5f, -0.7f, 1.0f, -0.3f, -0.5f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 0.1f, -0.2f, 0.3f, 1.0f,
      1.1f, 0.3f, 0.5f, 0.2f, 1.1f, 0.3f, 0.5f, 0.2f, 
      0.3f, -0.6f, 1.5f, 2.0f, 0.3f, -0.6f, 1.5f, 2.0f,

      0.5f, 0.1f, 0.4f, 1.6f, 0.5f, 0.1f, 0.4f, 1.6f,
      1.0f, 2.0f, 0.4f, 0.8f, 1.0f, 2.0f, 0.4f, 0.8f,
      0.9f, 0.1f, -1.3f, 0.7f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 0.3f, 0.2f, 4.0f, 2.2f,
      1.6f, 1.1f, 0.7f, 0.2f, 1.6f, 1.1f, 0.7f, 0.2f,
      0.4f, 1.0f, 1.2f, 0.5f, 0.4f, 1.0f, 1.2f, 0.5f,

      0.2f, 0.1f, 0.4f, 1.6f, 0.2f, 0.1f, 0.4f, 1.6f,
      2.4f, 3.3f, 2.1f, 4.2f, 2.4f, 3.3f, 2.1f, 4.2f,
      8.4f, 0.0f, 2.1f, 3.2f, 8.4f, 0.0f, 2.1f, 3.2f,

      0.1f, -0.2f, 0.3f, 1.0f, 0.1f, -0.2f, 0.3f, 1.0f,
     1.1f, 0.3f, 0.5f, 0.2f, 1.1f, 0.3f, 0.5f, 0.2f, 
      0.3f, -0.6f, 1.5f, 2.0f, 0.3f, -0.6f, 1.5f, 2.0f,

      0.5f, 0.1f, 0.4f, 1.6f, 0.5f, 0.1f, 0.4f, 1.6f,
      1.0f, 2.0f, 0.4f, 0.8f, 1.0f, 2.0f, 0.4f, 0.8f,
      0.9f, 0.1f, -1.3f, 0.7f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 0.3f, 0.2f, 4.0f, 2.2f,
      1.6f, 1.1f, 0.7f, 0.2f, 1.6f, 1.1f, 0.7f, 0.2f,
      0.4f, 1.0f, 1.2f, 0.5f, 0.4f, 1.0f, 1.2f, 0.5f,

      0.2f, 0.1f, 0.4f, 1.6f, 0.2f, 0.1f, 0.4f, 1.6f,
      2.4f, 3.3f, 2.1f, 4.2f, 2.4f, 3.3f, 2.1f, 4.2f,
      8.4f, 0.0f, 2.1f, 3.2f, 8.4f, 0.0f, 2.1f, 3.2f,
      };

  std::vector<float> pos_weight_data = {
     1.1f, 0.3f, 0.5f, 0.2f, 1.1f, 0.3f, 0.5f, 0.2f, 
      1.0f, 2.0f, 0.4f, 0.8f, 1.0f, 2.0f, 0.4f, 0.8f,
      1.6f, 1.1f, 0.7f, 0.2f, 1.6f, 1.1f, 0.7f, 0.2f,
      2.4f, 3.3f, 2.1f, 4.2f, 2.4f, 3.3f, 2.1f, 4.2f,
     1.1f, 0.3f, 0.5f, 0.2f, 1.1f, 0.3f, 0.5f, 0.2f, 
      1.0f, 2.0f, 0.4f, 0.8f, 1.0f, 2.0f, 0.4f, 0.8f,
      1.6f, 1.1f, 0.7f, 0.2f, 1.6f, 1.1f, 0.7f, 0.2f,
      2.4f, 3.3f, 2.1f, 4.2f, 2.4f, 3.3f, 2.1f, 4.2f};

  std::vector<float> bias_data = {
    -0.5f, 0.6f, 1.2f, 2.1f, -0.5f, 0.6f, 1.2f, 2.1f,
    0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.7f, 0.2f, 1.2f,
    0.5f, 0.4f, 0.3f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f
  };

  std::vector<float> pos_bias_u = {
      0.3f, -0.6f, 0.3f, -0.6f,
      0.9f, 0.1f, 0.9f, 0.1f};

  std::vector<float> pos_bias_v = {
      1.2f, 0.5f, 1.2f, 0.5f,
      8.4f, 0.0f, 8.4f, 0.0f};

  std::vector<float> output_data = {
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.879861831665039f, -0.6599940657615662f, 8.1999530792236328f, 10.0999450683593750f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.879861831665039f, -0.6599940657615662f, 8.1999530792236328f, 10.0999450683593750f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f};

  // self-attn without cacheint batch_size,
  RunRelPosAttentionTest(input_data, weight_data, bias_data, pos_emb, pos_weight_data,
                         pos_bias_u, pos_bias_v, output_data,
                         batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, head_size);
}

}  // namespace test
}  // namespace onnxruntime