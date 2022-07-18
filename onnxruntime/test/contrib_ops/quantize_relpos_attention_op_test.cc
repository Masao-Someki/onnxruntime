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
void RunQRelPosAttention(const std::vector<float>& input,
                   const std::vector<float>& weights,
                   const std::vector<float>& pos_emb,
                   const std::vector<float>& pos_weights,
                   const std::vector<float>& bias,
                   const std::vector<float>& pos_bias_u,
                   const std::vector<float>& pos_bias_v,
                   const std::vector<int32_t>& mask_index_data,
                   const std::vector<float>& output_data,
                   quantization::Params<QInput>& input_quant_params,
                   quantization::Params<QWeight>& weights_quant_params,
                   quantization::Params<QInput>& pos_emb_quant_params,
                   quantization::Params<QWeight>& pos_weights_quant_params,
                   int batch_size,
                   int sequence_length,
                   int pos_sequence_length,
                   int hidden_size,
                   int number_of_heads,
                   bool is_legacy = false,
                   bool use_float16 = false,
                   int input_hidden_size = 0) {
  input_hidden_size = (input_hidden_size == 0) ? hidden_size : input_hidden_size;

  OpTester tester("QRelPosAttention", 1, onnxruntime::kENDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
  if (is_legacy) {
    tester.AddAttribute<int64_t>("legacy", static_cast<int64_t>(is_legacy ? 1 : 0));
  }

  std::vector<int64_t> i_input_dims = {batch_size, sequence_length, input_hidden_size};
  std::vector<int64_t> pos_input_dims = {batch_size, pos_sequence_length, input_hidden_size};
  std::vector<int64_t> i_weights_dims = {input_hidden_size, static_cast<int64_t>(3 * hidden_size)};
  std::vector<int64_t> pos_weights_dims = {input_hidden_size, static_cast<int64_t>(hidden_size)};
  std::vector<int64_t> bias_dims = {static_cast<int64_t>(3 * hidden_size)};
  std::vector<int64_t> bias_u_dims = {number_of_heads, static_cast<int64_t>(hidden_size / number_of_heads)};
  std::vector<int64_t> bias_v_dims = {number_of_heads, static_cast<int64_t>(hidden_size / number_of_heads)};
  std::vector<int64_t> mask_index_dims = {batch_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};
  if constexpr (ep == EP::DNNL) {
    //onednn only supports raw mask
    if (mask_index_data.size() == static_cast<size_t>(batch_size * sequence_length)) {
      mask_index_dims = {batch_size, sequence_length};
    }
  }
  if (input_quant_params.scale != 0.0f) {
    tester.AddInput<QInput>("input",
                            i_input_dims,
                            QuantizeTestVector<QInput>(input, input_quant_params));
    tester.AddInput<QWeight>("weights",
                            i_weights_dims,
                            QuantizeTestVector<QWeight>(weights, weights_quant_params));
    tester.AddInput<QInput>("pos_emb",
                             pos_input_dims,
                             QuantizeTestVector<QInput>(pos_emb, pos_emb_quant_params));
    tester.AddInput<QWeight>("pos_weights",
                             pos_weights_dims,
                             QuantizeTestVector<QWeight>(pos_weights, pos_weights_quant_params));
  } else {
    bool force_symmetric = false;
    if constexpr (ep == EP::CUDA) {
      force_symmetric = true;
    }
    tester.AddInput<QInput>(
        "input",
        i_input_dims,
        QuantizeLinearTestVector<QInput>(input, input_quant_params, force_symmetric));
    tester.AddInput<QWeight>(
        "weights",
        i_weights_dims,
        QuantizeLinearTestVector<QWeight>(weights, weights_quant_params, force_symmetric));
    tester.AddInput<QInput>(
        "pos_emb",
        pos_input_dims,
        QuantizeLinearTestVector<QInput>(pos_emb, pos_emb_quant_params, force_symmetric));
    tester.AddInput<QWeight>(
        "pos_weights",
        pos_weights_dims,
        QuantizeLinearTestVector<QWeight>(pos_weights, pos_weights_quant_params, force_symmetric));
  }

  if (use_float16) {
    tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias));
    tester.AddInput<MLFloat16>("pos_bias_u", bias_u_dims, ToFloat16(pos_bias_u));
    tester.AddInput<MLFloat16>("pos_bias_v", bias_v_dims, ToFloat16(pos_bias_v));
    tester.AddInput<MLFloat16>("input_scale_tensor", {1}, ToFloat16({input_quant_params.scale}));
    tester.AddInput<MLFloat16>("weights_scale_tensor", {1}, ToFloat16({weights_quant_params.scale}));
    tester.AddInput<MLFloat16>("pos_emb_scale_tensor", {1}, ToFloat16({pos_emb_quant_params.scale}));
    tester.AddInput<MLFloat16>("pos_weights_scale_tensor", {1}, ToFloat16({pos_weights_quant_params.scale}));
    tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
  } else {
    tester.AddInput<float>("bias", bias_dims, bias);
    tester.AddInput<float>("pos_bias_u", bias_u_dims, pos_bias_u);
    tester.AddInput<float>("pos_bias_v", bias_v_dims, pos_bias_v);
    tester.AddInput<float>("input_scale_tensor", {1}, {input_quant_params.scale});
    tester.AddInput<float>("weights_scale_tensor", {1}, {weights_quant_params.scale});
    tester.AddInput<float>("pos_emb_scale_tensor", {1}, {pos_emb_quant_params.scale});
    tester.AddInput<float>("pos_weights_scale_tensor", {1}, {pos_weights_quant_params.scale});
    tester.AddOutput<float>("output", output_dims, output_data);
  }
  if (mask_index_data.size() > 0) {
    tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);
  } else {
    // mask index is optional.
    tester.AddOptionalInputEdge<int32_t>();
  }
  tester.AddInput<QInput>("input_zero_point", {1}, {input_quant_params.zero_point});
  tester.AddInput<QWeight>("iw_zero_point", {1}, {weights_quant_params.zero_point});
  tester.AddInput<QInput>("pos_zero_point", {1}, {pos_emb_quant_params.zero_point});
  tester.AddInput<QWeight>("pw_zero_point", {1}, {pos_weights_quant_params.zero_point});

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


static void RunQRelPosAttentionDNNL(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& pos_emb,
    const std::vector<float>& pos_weights,
    const std::vector<float>& bias,
    const std::vector<float>& bias_u,
    const std::vector<float>& bias_v,
    const std::vector<int32_t>& mask_index_data,  // onednn only support raw mask data
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int pos_sequence_length,
    int hidden_size,
    int number_of_heads,
    bool is_legacy,
    bool use_special_quantize_parameter = true) {
  // Return without running code if USE_DNNL is not defined
#ifdef USE_DNNL
  quantization::Params<uint8_t> input_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<uint8_t> pos_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<int8_t> iw_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<int8_t> pw_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  if (use_special_quantize_parameter) {
    input_quant_params.scale = 0.1f;
    pos_quant_params.scale = 0.1f;
    iw_quant_params.scale = 0.1f;
    pw_quant_params.scale = 0.1f;
    input_quant_params.zero_point = 128;
    pos_quant_params.zero_point = 128;
    iw_quant_params.zero_point = 1;
    pw_quant_params.zero_point = 1;
  }

  RunQRelPosAttention<uint8_t, int8_t, EP::DNNL>(
      input, weight, pos_emb, pos_weights, bias, bias_u, bias_v,mask_index_data,  output_data,
      input_quant_params, iw_quant_params, pos_quant_params, pw_quant_params,
      batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, is_legacy);
#else
  ORT_UNUSED_PARAMETER(input);
  ORT_UNUSED_PARAMETER(weight);
  ORT_UNUSED_PARAMETER(pos_emb);
  ORT_UNUSED_PARAMETER(pos_weights);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(bias_u);
  ORT_UNUSED_PARAMETER(bias_v);
  ORT_UNUSED_PARAMETER(mask_index_data);
  ORT_UNUSED_PARAMETER(output_data);
  ORT_UNUSED_PARAMETER(batch_size);
  ORT_UNUSED_PARAMETER(sequence_length);
  ORT_UNUSED_PARAMETER(pos_sequence_length);
  ORT_UNUSED_PARAMETER(hidden_size);
  ORT_UNUSED_PARAMETER(number_of_heads);
  ORT_UNUSED_PARAMETER(use_special_quantize_parameter);
#endif
}

static void RunQRelPosAttentionU8U8(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& pos_emb,
    const std::vector<float>& pos_weights,
    const std::vector<float>& bias,
    const std::vector<float>& bias_u,
    const std::vector<float>& bias_v,
    const std::vector<int32_t>& mask_index_data,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int pos_sequence_length,
    int hidden_size,
    int number_of_heads,
    bool is_legacy,
    bool use_special_quantize_parameter = true) {
  quantization::Params<uint8_t> input_quant_params = {0.0f, 0};
  quantization::Params<uint8_t> pos_quant_params = {0.0f, 0};
  quantization::Params<uint8_t> iw_quant_params = {0.0f, 0};
  quantization::Params<uint8_t> pw_quant_params = {0.0f, 0};

  if (use_special_quantize_parameter) {
    input_quant_params.scale = 0.1f;
    pos_quant_params.scale = 0.1f;
    iw_quant_params.scale = 0.1f;
    pw_quant_params.scale = 0.1f;
    input_quant_params.zero_point = 128;
    pos_quant_params.zero_point = 128;
    iw_quant_params.zero_point = 128;
    pw_quant_params.zero_point = 128;
  }

  RunQRelPosAttention<uint8_t, uint8_t, EP::CPU>(
      input, weight, pos_emb, pos_weights, bias, bias_u, bias_v, mask_index_data, output_data,
      input_quant_params, iw_quant_params, pos_quant_params, pw_quant_params,
      batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, is_legacy);
}

static void RunQRelPosAttentionU8S8(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& pos_emb,
    const std::vector<float>& pos_weights,
    const std::vector<float>& bias,
    const std::vector<float>& bias_u,
    const std::vector<float>& bias_v,
    const std::vector<int32_t>& mask_index_data,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int pos_sequence_length,
    int hidden_size,
    int number_of_heads,
    bool is_legacy,
    bool use_special_quantize_parameter = true) {
  quantization::Params<uint8_t> input_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<uint8_t> pos_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<int8_t> iw_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<int8_t> pw_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  if (use_special_quantize_parameter) {
    input_quant_params.scale = 0.1f;
    pos_quant_params.scale = 0.1f;
    iw_quant_params.scale = 0.1f;
    pw_quant_params.scale = 0.1f;
    input_quant_params.zero_point = 128;
    pos_quant_params.zero_point = 128;
    iw_quant_params.zero_point = 1;
    pw_quant_params.zero_point = 1;
  }

  RunQRelPosAttention<uint8_t, int8_t, EP::CPU>(
      input, weight, pos_emb, pos_weights, bias, bias_u, bias_v, mask_index_data, output_data,
      input_quant_params, iw_quant_params, pos_quant_params, pw_quant_params,
      batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, is_legacy);
}

static void RunQRelPosAttentionAll(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& pos_emb,
    const std::vector<float>& pos_weights,
    const std::vector<float>& bias,
    const std::vector<float>& bias_u,
    const std::vector<float>& bias_v,
    const std::vector<int32_t>& mask_index_data,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int pos_sequence_length,
    int hidden_size,
    int number_of_heads,
    bool is_legacy,
    bool use_special_quantize_parameter = true) {
  RunQRelPosAttentionU8U8(
      input, weight, pos_emb, pos_weights, bias, bias_u, bias_v, mask_index_data, output_data,
                batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, is_legacy);
  RunQRelPosAttentionU8S8(
      input, weight, pos_emb, pos_weights, bias, bias_u, bias_v, mask_index_data, output_data,
                batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, is_legacy);
  RunQRelPosAttentionDNNL(
      input, weight, pos_emb, pos_weights, bias, bias_u, bias_v, mask_index_data, output_data,
                batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, is_legacy);
}

//ONEDNN EP only supports 2D raw mask
#ifdef USE_DNNL
TEST(QRelPosAttentionTest, QRelPosAttentionDNNLBatch1) {
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

  std::vector<int32_t> mask_index_data = {3L};

  std::vector<float> output_data = {
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.879861831665039f, -0.6599940657615662f, 8.1999530792236328f, 10.0999450683593750f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f};

  RunQRelPosAttentionDNNL(input_data, weight_data, pos_emb, pos_weight_data, bias_data, pos_bias_u, pos_bias_v,
    mask_index_data, output_data, batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads);
}
#endif  // USE_DNNL

TEST(QRelPosAttentionTest, QRelPosAttentionBatch1) {
  int batch_size = 1;
  int sequence_length = 3;
  int pos_sequence_length = 2 * sequence_length - 1;
  int hidden_size = 8;
  int number_of_heads = 2;
  bool is_legacy = false;

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

  std::vector<int32_t> mask_index_data = {3L};

  std::vector<float> output_data = {
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.879861831665039f, -0.6599940657615662f, 8.1999530792236328f, 10.0999450683593750f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f};

  RunQRelPosAttentionAll(input_data, weight_data, pos_emb, pos_weight_data, bias_data, pos_bias_u, pos_bias_v,
    mask_index_data, output_data, batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, is_legacy);
}

TEST(QRelPosAttentionTest, QRelPosttentionBatch2) {
  int batch_size = 2;
  int sequence_length = 3;
  int pos_sequence_length = 2 * sequence_length - 1;
  int hidden_size = 8;
  int number_of_heads = 2;
  bool is_legacy = false;

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

  std::vector<int32_t> mask_index_data = {3L, 3L};

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

  RunQRelPosAttentionAll(input_data, weight_data, pos_emb, pos_weight_data, bias_data, pos_bias_u, pos_bias_v,
    mask_index_data, output_data, batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, is_legacy);
}

TEST(QRelPosAttentionTest, QLegacyRelPosAttentionBatch1) {
  int batch_size = 1;
  int sequence_length = 3;
  int pos_sequence_length = sequence_length;
  int hidden_size = 8;
  int number_of_heads = 2;
  bool is_legacy = true;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f, 0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f, 0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f, 0.8f, -0.5f, 0.0f, 1.0f};

  std::vector<float> pos_emb = {
      0.2f, -0.0f, 0.2f, 0.4f, 0.2f, -0.0f, 0.2f, 0.4f,
      -0.7f, 1.0f, -0.3f, -0.5f, -0.7f, 1.0f, -0.3f, -0.5f,
      -1.3f, 1.0f, -0.5f, 1.0f, -1.3f, 1.0f, -0.5f, 1.0f};

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

  std::vector<int32_t> mask_index_data = {3L};

  std::vector<float> output_data = {
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      -6.4679698944091797f, 0.3448030948638916f, 0.2346523106098175f, 0.8192735314369202f,
      -8.6457786560058594f, 0.4385272860527039f, -0.5083251595497131f, -0.0463974177837372f,

      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f};

  RunQRelPosAttentionAll(input_data, weight_data, pos_emb, pos_weight_data, bias_data, pos_bias_u, pos_bias_v,
    mask_index_data, output_data, batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, is_legacy);
}

TEST(QRelPosAttentionTest, QLegacyRelPosttentionBatch2) {
  int batch_size = 2;
  int sequence_length = 3;
  int pos_sequence_length = sequence_length;
  int hidden_size = 8;
  int number_of_heads = 2;
  bool is_legacy = true;

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
      -1.3f, 1.0f, -0.5f, 1.0f, -1.3f, 1.0f, -0.5f, 1.0f};

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

  std::vector<int32_t> mask_index_data = {3L, 3L};

  std::vector<float> output_data = {
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      -6.4679698944091797f, 0.3448030948638916f, 0.2346523106098175f, 0.8192735314369202f,
      -8.6457786560058594f, 0.4385272860527039f, -0.5083251595497131f, -0.0463974177837372f,

      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,

      -6.4679698944091797f, 0.3448030948638916f, 0.2346523106098175f, 0.8192735314369202f,
      -8.6457786560058594f, 0.4385272860527039f, -0.5083251595497131f, -0.0463974177837372f,

      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f,
      16.8799991607666016f, -0.6599999666213989f, 8.1999998092651367f, 10.0999994277954102f};

  RunQRelPosAttentionAll(input_data, weight_data, pos_emb, pos_weight_data, bias_data, pos_bias_u, pos_bias_v,
    mask_index_data, output_data, batch_size, sequence_length, pos_sequence_length, hidden_size, number_of_heads, is_legacy);
}

}  // namespace test
}  // namespace onnxruntime
