// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications by Masao Someki
// Copyright (c) 2022 Masao Someki

#include <algorithm>
#include <cfenv>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/quantization_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/qmath.h"
#include "core/quantization/quantization.h"

namespace onnxruntime {
namespace test {

enum class EP : char {
  CPU,
  CUDA,
  DNNL
};

// input:      [batch_size, sequence_length, hidden_size]
// weights:    [hidden_size, 3 * hidden_size]
// bias:       [3 * hidden_size]
// mask_index: [batch_size]
// output:     [batch_size, sequence_length, hidden_size]
template <typename QInput, typename QWeight, EP ep>
void RunQCrossAttention(const std::vector<float>& query,
                   const std::vector<float>& key,
                   const std::vector<float>& q_weights_data,
                   const std::vector<float>& kv_weights_data,
                   const std::vector<float>& q_bias_data,
                   const std::vector<float>& kv_bias_data,
                   const std::vector<int32_t>& mask_index_data,
                   const std::vector<float>& output_data,
                   quantization::Params<QInput>& query_quant_params,
                   quantization::Params<QInput>& key_quant_params,
                   quantization::Params<QWeight>& q_weight_quant_params,
                   quantization::Params<QWeight>& k_weight_quant_params,
                   int batch_size,
                   int sequence_length,
                   int kv_sequence_length,
                   int hidden_size,
                   int number_of_heads,
                   bool use_float16 = false,
                   int input_hidden_size = 0) {
  input_hidden_size = (input_hidden_size == 0) ? hidden_size : input_hidden_size;

  OpTester tester("QCrossAttention", 1, onnxruntime::kENDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));

  std::vector<int64_t> q_input_dims = {batch_size, sequence_length, input_hidden_size};
  std::vector<int64_t> kv_input_dims = {batch_size, kv_sequence_length, input_hidden_size};
  std::vector<int64_t> q_weights_dims = {input_hidden_size, static_cast<int64_t>(hidden_size)};
  std::vector<int64_t> kv_weights_dims = {input_hidden_size, static_cast<int64_t>(2 * hidden_size)};
  std::vector<int64_t> q_bias_dims = {static_cast<int64_t>(hidden_size)};
  std::vector<int64_t> kv_bias_dims = {static_cast<int64_t>(2 * hidden_size)};
  std::vector<int64_t> mask_index_dims = {batch_size};
  if constexpr (ep == EP::DNNL) {
    //onednn only supports raw mask
    if (mask_index_data.size() == static_cast<size_t>(batch_size * sequence_length)) {
      mask_index_dims = {batch_size, sequence_length};
    }
  }
  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

  if (query_quant_params.scale != 0.0f) {
    tester.AddInput<QInput>("query",
                            q_input_dims,
                            QuantizeTestVector<QInput>(query, query_quant_params));
    tester.AddInput<QInput>("key",
                            kv_input_dims,
                            QuantizeTestVector<QInput>(key, key_quant_params));
    tester.AddInput<QWeight>("q_weight",
                             q_weights_dims,
                             QuantizeTestVector<QWeight>(q_weights_data, q_weight_quant_params));
    tester.AddInput<QWeight>("kv_weight",
                             kv_weights_dims,
                             QuantizeTestVector<QWeight>(kv_weights_data, k_weight_quant_params));
  } else {
    bool force_symmetric = false;
    if constexpr (ep == EP::CUDA) {
      force_symmetric = true;
    }
    tester.AddInput<QInput>(
        "query",
        q_input_dims,
        QuantizeLinearTestVector<QInput>(query, query_quant_params, force_symmetric));
    tester.AddInput<QInput>(
        "key",
        kv_input_dims,
        QuantizeLinearTestVector<QInput>(key, key_quant_params, force_symmetric));
    tester.AddInput<QWeight>(
        "q_weight",
        q_weights_dims,
        QuantizeLinearTestVector<QWeight>(q_weights_data, q_weight_quant_params, force_symmetric));
    tester.AddInput<QWeight>(
        "kv_weight",
        kv_weights_dims,
        QuantizeLinearTestVector<QWeight>(kv_weights_data, k_weight_quant_params, force_symmetric));
  }
  if (use_float16) {
    tester.AddInput<MLFloat16>("q_bias", q_bias_dims, ToFloat16(q_bias_data));
    tester.AddInput<MLFloat16>("kv_bias", kv_bias_dims, ToFloat16(kv_bias_data));
    tester.AddInput<MLFloat16>("query_scale", {1}, ToFloat16({query_quant_params.scale}));
    tester.AddInput<MLFloat16>("key_scale", {1}, ToFloat16({key_quant_params.scale}));
    tester.AddInput<MLFloat16>("q_weight_scale", {1}, ToFloat16({q_weight_quant_params.scale}));
    tester.AddInput<MLFloat16>("k_weight_scale", {1}, ToFloat16({k_weight_quant_params.scale}));
    tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
  } else {
    tester.AddInput<float>("q_bias", q_bias_dims, q_bias_data);
    tester.AddInput<float>("kv_bias", kv_bias_dims, kv_bias_data);
    tester.AddInput<float>("query_scale", {1}, {query_quant_params.scale});
    tester.AddInput<float>("key_scale", {1}, {key_quant_params.scale});
    tester.AddInput<float>("q_weight_scale", {1}, {q_weight_quant_params.scale});
    tester.AddInput<float>("k_weight_scale", {1}, {k_weight_quant_params.scale});
    tester.AddOutput<float>("output", output_dims, output_data);
  }

  if (mask_index_data.size() > 0) {
    tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);
  } else {
    // mask index is optional.
    tester.AddOptionalInputEdge<int32_t>();
  }

  tester.AddInput<QInput>("query_zero_point", {1}, {query_quant_params.zero_point});
  tester.AddInput<QInput>("key_zero_point", {1}, {key_quant_params.zero_point});
  tester.AddInput<QWeight>("q_weight_zero_point", {1}, {q_weight_quant_params.zero_point});
  tester.AddInput<QWeight>("k_weight_zero_point", {1}, {k_weight_quant_params.zero_point});

  if constexpr (ep == EP::CUDA) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
  } else if constexpr (ep == EP::CPU) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
  } else {  // onednn ep
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultDnnlExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);
  }
}

// static void RunQCrossAttentionCUDA(
//     const std::vector<float>& query,
//     const std::vector<float>& key,
//     const std::vector<float>& q_weights_data,
//     const std::vector<float>& kv_weights_data,
//     const std::vector<float>& q_bias_data,
//     const std::vector<float>& kv_bias_data,
//     const std::vector<int32_t>& mask_index_data,
//     const std::vector<float>& output_data,
//     int batch_size,
//     int sequence_length,
//     int hidden_size,
//     int number_of_heads,
//     bool use_special_quantize_parameter = true,
//     bool use_float16 = false,
//     int input_hidden_size = 0) {
//   int min_cuda_architecture = 530;
//   bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);

//   if (enable_cuda) {
//     quantization::Params<int8_t> input_quant_params = {0.0f, 0};
//     quantization::Params<int8_t> weights_quant_params = {0.0f, 0};
//     if (use_special_quantize_parameter) {
//       input_quant_params.scale = 0.1f;
//       weights_quant_params.scale = 0.1f;
//     }
//     RunQCrossAttention<int8_t, int8_t, EP::CUDA>(
//         query, key, q_weights_data, kv_weights_data, q_bias_data, kv_bias_data, mask_index_data, output_data, input_quant_params, weights_quant_params,
//         batch_size, sequence_length, hidden_size, number_of_heads, use_float16, input_hidden_size);
//   }
// }

static void RunQCrossAttentionDNNL(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& q_weights_data,
    const std::vector<float>& kv_weights_data,
    const std::vector<float>& q_bias_data,
    const std::vector<float>& kv_bias_data,
    const std::vector<int32_t>& mask_index_data,  // onednn only support raw mask data
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int kv_sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_special_quantize_parameter = true) {
  // Return without running code if USE_DNNL is not defined
#ifdef USE_DNNL
  quantization::Params<uint8_t> query_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<uint8_t> key_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<int8_t> q_weights_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<int8_t> k_weights_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  if (use_special_quantize_parameter) {
    query_quant_params.scale = 0.1f;
    key_quant_params.scale = 0.1f;
    q_weights_quant_params.scale = 0.1f;
    k_weights_quant_params.scale = 0.1f;
    query_quant_params.zero_point = 128;
    key_quant_params.zero_point = 128;
    q_weights_quant_params.zero_point = 1;
    k_weights_quant_params.zero_point = 1;
  }

  RunQCrossAttention<uint8_t, int8_t, EP::DNNL>(
      query, key, q_weights_data, kv_weights_data, q_bias_data, k_bias_data, mask_index_data, output_data,
      query_quant_params, key_quant_params, q_weights_quant_params, k_weights_quant_params,
      batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
#else
  ORT_UNUSED_PARAMETER(query);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(q_weights_data);
  ORT_UNUSED_PARAMETER(kv_weights_data);
  ORT_UNUSED_PARAMETER(q_bias_data);
  ORT_UNUSED_PARAMETER(kv_bias_data);
  ORT_UNUSED_PARAMETER(mask_index_data);
  ORT_UNUSED_PARAMETER(output_data);
  ORT_UNUSED_PARAMETER(batch_size);
  ORT_UNUSED_PARAMETER(sequence_length);
  ORT_UNUSED_PARAMETER(kv_sequence_length);
  ORT_UNUSED_PARAMETER(hidden_size);
  ORT_UNUSED_PARAMETER(number_of_heads);
  ORT_UNUSED_PARAMETER(use_special_quantize_parameter);
#endif
}

static void RunQCrossAttentionU8U8(
    const std::vector<float>& query,
    const std::vector<float>& key,
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
    int number_of_heads,
    bool use_special_quantize_parameter = true) {
  quantization::Params<uint8_t> query_quant_params = {0.0f, 0};
  quantization::Params<uint8_t> key_quant_params = {0.0f, 0};
  quantization::Params<uint8_t> q_weights_quant_params = {0.0f, 0};
  quantization::Params<uint8_t> k_weights_quant_params = {0.0f, 0};
  if (use_special_quantize_parameter) {
    query_quant_params.scale = 0.1f;
    key_quant_params.scale = 0.1f;
    q_weights_quant_params.scale = 0.1f;
    k_weights_quant_params.scale = 0.1f;
    query_quant_params.zero_point = 128;
    key_quant_params.zero_point = 128;
    q_weights_quant_params.zero_point = 128;
    k_weights_quant_params.zero_point = 128;
  }

  RunQCrossAttention<uint8_t, uint8_t, EP::CPU>(
      query, key, q_weights_data, kv_weights_data, q_bias_data, kv_bias_data, mask_index_data, output_data,
      query_quant_params, key_quant_params, q_weights_quant_params, k_weights_quant_params,
      batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

static void RunQCrossAttentionU8S8(
    const std::vector<float>& query,
    const std::vector<float>& key,
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
    int number_of_heads,
    bool use_special_quantize_parameter = true) {
  quantization::Params<uint8_t> query_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<uint8_t> key_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<int8_t> q_weights_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<int8_t> k_weights_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  if (use_special_quantize_parameter) {
    query_quant_params.scale = 0.1f;
    weights_quant_params.scale = 0.1f;
    key_quant_params.scale = 0.1f;
    input_quant_params.zero_point = 128;
    q_weights_quant_params.scale = 0.1f;
    weights_quant_params.zero_point = 1;
    k_weights_quant_params.scale = 0.1f;
    query_quant_params.zero_point = 128;
    key_quant_params.zero_point = 128;
    q_weights_quant_params.zero_point = 1;
    k_weights_quant_params.zero_point = 1;
  }

  RunQCrossAttention<uint8_t, int8_t, EP::CPU>(
      query, key, q_weights_data, kv_weights_data, q_bias_data, kv_bias_data, mask_index_data, output_data,
      query_quant_params, key_quant_params, q_weights_quant_params, k_weights_quant_params,
      batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

static void RunQCrossAttentionAll(
    const std::vector<float>& query,
    const std::vector<float>& key,
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
    int number_of_heads,
    bool use_special_quantize_parameter = true) {
  RunQCrossAttentionU8U8(
      query, key, q_weights_data, kv_weights_data, q_bias_data, kv_bias_data, mask_index_data, output_data,
                    batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
  RunQCrossAttentionU8S8(
      query, key, q_weights_data, kv_weights_data, q_bias_data, kv_bias_data, mask_index_data, output_data,
                    batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
  // RunQCrossAttentionCUDA(
  //     query, key, q_weights_data, kv_weights_data, q_bias_data, kv_bias_data, mask_index_data, output_data,
  //                   batch_size, sequence_length, hidden_size, number_of_heads,
  //                   use_special_quantize_parameter, is_unidirectional, use_float16, input_hidden_size);
  RunQCrossAttentionDNNL(
      query, key, q_weights_data, kv_weights_data, q_bias_data, kv_bias_data, mask_index_data, output_data,
                    batch_size, sequence_length, kv_sequence_length, hidden_size, number_of_heads);
}

//ONEDNN EP only supports 2D raw mask
#ifdef USE_DNNL
TEST(QCrossAttentionTest, QCrossAttentionDNNLBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f
  }

  std::vector<float> kv_weight_data = {
       1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
       1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
       1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
       2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f}

  std::vector<float> kv_bias_data = {
      0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {1L, 1L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunQCrossAttentionDNNL(input_data, input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data,
    mask_index_data, output_data, batch_size, sequence_length, sequence_length, hidden_size, number_of_heads);
}
#endif  // USE_DNNL

TEST(QCrossAttentionTest, QCrossAttentionBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f
  };

  std::vector<float> kv_weight_data = {
       1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
       1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
       1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
       2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
      0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunQCrossAttentionAll(input_data, input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data,
    mask_index_data, output_data, batch_size, sequence_length, sequence_length, hidden_size, number_of_heads);
}

TEST(QCrossAttentionTest, QCrossAttentionBatch2) {
  int batch_size = 2;
  int sequence_length = 2;
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
      0.2f, 0.1f, 0.4f, 1.6f
  };

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

  RunQCrossAttentionAll(input_data, input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data,
    mask_index_data, output_data, batch_size, sequence_length, sequence_length, hidden_size, number_of_heads);
}

//ONEDNN EP only support 2D raw mask
#ifdef USE_DNNL
TEST(QCrossAttentionTest, QCrossAttentionDNNLBatch2) {
  int batch_size = 2;
  int sequence_length = 2;
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
      0.2f, 0.1f, 0.4f, 1.6f
  };

  std::vector<float> kv_weight_data = {
       1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
       1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
       1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
       2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
      0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {1L, 1L, 1L, 1L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunQCrossAttentionDNNL(input_data, input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data,
    mask_index_data, output_data, batch_size, sequence_length, sequence_length, hidden_size, number_of_heads);
}
#endif  // USE_DNNL

TEST(QCrossAttentionTest, QCrossAttentionMaskPartialSequence) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f
  };

  std::vector<float> kv_weight_data = {
       1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
       1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
       1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
       2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
      0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index < sequence_length
  std::vector<int32_t> mask_index_data = {1L};

  std::vector<float> output_data = {
      8.6899995803833008f, -0.13000002503395081f, 4.25f, 5.6499996185302734f,
      8.6899995803833008f, -0.13000002503395081f, 4.2499995231628418f, 5.6499991416931152f};

  RunQCrossAttentionAll(input_data, input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data,
    mask_index_data, output_data, batch_size, sequence_length, sequence_length, hidden_size, number_of_heads);
}

//oneDNN EP only supports 2D raw mask
#ifdef USE_DNNL
TEST(QCrossAttentionTest, QCrossAttentionDNNLMaskPartialSequence) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f
  };

  std::vector<float> kv_weight_data = {
       1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
       1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
       1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
       2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
      0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {1L, 0L};

  std::vector<float> output_data = {
      8.6899995803833008f, -0.13000002503395081f, 4.25f, 5.6499996185302734f,
      8.6899995803833008f, -0.13000002503395081f, 4.2499995231628418f, 5.6499991416931152f};

  RunQCrossAttentionDNNL(input_data, input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data,
    mask_index_data, output_data, batch_size, sequence_length, sequence_length, hidden_size, number_of_heads);
}
#endif  // USE_DNNL

TEST(QCrossAttentionTest, QCrossAttentionMaskExceedSequence) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f
  };

  std::vector<float> kv_weight_data = {
       1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
       1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
       1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
       2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
      0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index > sequence_length
  std::vector<int32_t> mask_index_data = {3L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunQCrossAttentionAll(input_data, input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data,
    mask_index_data, output_data, batch_size, sequence_length, sequence_length, hidden_size, number_of_heads);
}

TEST(QCrossAttentionTest, QCrossAttentionNoMaskIndex) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f
  };

  std::vector<float> kv_weight_data = {
       1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
       1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
       1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
       2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> q_bias_data = {-0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> kv_bias_data = {
      0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunQCrossAttentionAll(input_data, input_data, q_weight_data, kv_weight_data, q_bias_data, kv_bias_data,
    mask_index_data, output_data, batch_size, sequence_length, sequence_length, hidden_size, number_of_heads);
}

}  // namespace test
}  // namespace onnxruntime