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

static void RunRelativeShiftTest(
    const std::vector<float>& matrix_ac,
    const std::vector<float>& matrix_bd,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int pos_sequence_length,
    int num_heads,
    bool is_legacy = false,
    bool only_enable_cpu = false,
    bool only_enable_cuda = false,
    bool use_float16 = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !only_enable_cpu;
  // bool enable_cuda = false;
  //   bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get()) && !is_weights_constant && !only_enable_cpu;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !only_enable_cuda;

  if (enable_cpu || enable_cuda) {
    OpTester tester("RelativeShift", 1, onnxruntime::kENDomain);

    if (is_legacy) {
      tester.AddAttribute<int64_t>("legacy", 1);
    }
    std::vector<int64_t> matrix_ac_dims = {batch_size, num_heads, sequence_length, sequence_length};
    std::vector<int64_t> matrix_bd_dims = {batch_size, num_heads, sequence_length, pos_sequence_length};

    const std::vector<int64_t> output_dims = {batch_size, num_heads, sequence_length, sequence_length};

    tester.AddInput<float>("matrix_ac", matrix_ac_dims, matrix_ac);
    tester.AddInput<float>("matrix_bd", matrix_bd_dims, matrix_bd);

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

TEST(RelativeShiftTest, LatestBatch1) {
  int batch_size = 1;
  int sequence_length = 3;
  int pos_sequence_length = 2 * sequence_length - 1;
  int number_of_heads = 1;
  bool is_legacy = false;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f,
      0.5f, 0.2f, 0.3f,
      0.8f, -0.5f, 0.0f};

  std::vector<float> pos_bias = {
      0.2f, -0.0f, 0.2f, 0.4f, 0.2f,
      -0.7f, 1.0f, -0.3f, -0.5f, -0.7f,
      -1.3f, 1.0f, -0.5f, 1.0f, -1.3f};

  std::vector<float> output_data = {
      1.0f, -0.1f, 0.2f,
      1.5f, -0.1f, -0.2f,
      -0.5f, 0.5f, -0.5f};

  // self-attn without cacheint batch_size,
  RunRelativeShiftTest(input_data, pos_bias, output_data,
                         batch_size, sequence_length, pos_sequence_length,
                         number_of_heads, is_legacy);
}

TEST(RelativeShiftTest, LegacyBatch1) {
  int batch_size = 1;
  int sequence_length = 3;
  int pos_sequence_length = sequence_length;
  int number_of_heads = 1;
  bool is_legacy = true;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f,
      0.5f, 0.2f, 0.3f,
      0.8f, -0.5f, 0.0f};

  std::vector<float> pos_bias = {
      0.2f, -0.0f, 0.2f,
      -0.7f, 1.0f, -0.3f,
      -1.3f, 1.0f, -0.5f};

  std::vector<float> output_data = {
      1.0f, -0.5f, -0.7f,
      1.5f, -0.1f, 0.3f,
      -0.5f, 0.5f, -0.5f};

  // self-attn without cacheint batch_size,
  RunRelativeShiftTest(input_data, pos_bias, output_data,
                         batch_size, sequence_length, pos_sequence_length,
                         number_of_heads, is_legacy);
}

}  // namespace test
}  // namespace onnxruntime