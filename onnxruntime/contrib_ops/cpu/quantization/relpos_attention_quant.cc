// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications by Masao Someki
// Copyright (c) 2022 Masao Someki

#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/bert/relpos_attention_cpu_base.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/qmath.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename T>
class QRelPosAttention : public OpKernel, public RelPosAttentionCPUBase {
 public:
  QRelPosAttention(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  BufferUniquePtr packed_weights_;
  size_t packed_weights_size_;
  TensorShape q_weight_shape_;
  TensorShape kv_weight_shape_;
  bool weights_is_signed_;
};

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QRelPosAttention,
    kENDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QRelPosAttention<float>);

template <typename T>
QRelPosAttention<T>::QRelPosAttention(const OpKernelInfo& info) : OpKernel(info), RelPosAttentionCPUBase(info) {}

template <typename T>
Status QRelPosAttention<T>::Compute(OpKernelContext* context) const {
  // Input and output shapes:
  //   Input  0 - input             : (batch_size, sequence_length, input_hidden_size)
  //   Input  1 - weights           : (input_hidden_size, 3 * hidden_size)
  //   Input  2 - bias              : (3 * hidden_size)
  //   Input  3 - input_scale       : scalar
  //   Input  4 - weight_scale      : scalar for per tensor quantization, (3 * hidden_size) for per column quantization
  //   Input  5 - mask_index        : nullptr, (batch_size), (2 * batch_size), (batch_size, 1), (1, 1) or (batch_size, past_sequence_length + sequence_length)
  //   Input  6 - input_zero_point  : scalar
  //   Input  7 - weight_zero_point : scalar for per tensor quantization, (3 * hidden_size) for per column quantization
  //   Input  8 - past              : (2, batch_size, num_heads, past_sequence_length, head_size)
  //   Output 0                     : (batch_size, sequence_length, hidden_size)
  //   ORT_RETURN_IF_ERROR(CheckInputs(context));
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* pos_emb = context->Input<Tensor>(2);
  const Tensor* pos_weights = context->Input<Tensor>(3);
  const Tensor* bias = context->Input<Tensor>(4);
  const Tensor* pos_bias_u = context->Input<Tensor>(5);
  const Tensor* pos_bias_v = context->Input<Tensor>(6);

  const Tensor* input_scale_tensor = context->Input<Tensor>(7);
  const Tensor* weights_scale_tensor = context->Input<Tensor>(8);
  const Tensor* pos_emb_scale_tensor = context->Input<Tensor>(9);
  const Tensor* pos_weights_scale_tensor = context->Input<Tensor>(10);
  const Tensor* mask_index = context->Input<Tensor>(11);
  const Tensor* input_zp_tensor = context->Input<Tensor>(12);
  const Tensor* iw_zp_tensor = context->Input<Tensor>(13);
  const Tensor* pos_zp_tensor = context->Input<Tensor>(14);
  const Tensor* pw_zp_tensor = context->Input<Tensor>(15);

  ORT_RETURN_IF_ERROR(RelPosAttentionBase::CheckInputs(input->Shape(),
                                                       weights->Shape(),
                                                       bias->Shape(),
                                                       pos_emb->Shape(),
                                                       pos_bias_u->Shape(),
                                                       pos_bias_v->Shape(),
                                                       mask_index));

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(input_scale_tensor),
                    "input scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(pos_emb_scale_tensor),
                    "input scale must be a scalar or 1D tensor of size 1");
  T input_scale = *(input_scale_tensor->template Data<T>());
  T pos_scale = *(pos_emb_scale_tensor->template Data<T>());

  bool is_i_weight_scale_per_column = !IsScalarOr1ElementVector(weights_scale_tensor);
  bool is_p_weight_scale_per_column = !IsScalarOr1ElementVector(pos_weights_scale_tensor);
  const T* i_weight_scale_data = weights_scale_tensor->template Data<T>();
  const T* p_weight_scale_data = pos_weights_scale_tensor->template Data<T>();

  std::vector<T> i_dequant_scales(i_weight_scale_data, i_weight_scale_data + weights_scale_tensor->Shape().Size());
  std::vector<T> p_dequant_scales(p_weight_scale_data, p_weight_scale_data + pos_weights_scale_tensor->Shape().Size());
  std::for_each(i_dequant_scales.begin(), i_dequant_scales.end(), [&input_scale](float& i_dequant_scale) {
    return i_dequant_scale *= input_scale;
  });
  std::for_each(p_dequant_scales.begin(), p_dequant_scales.end(), [&pos_scale](float& p_dequant_scale) {
    return p_dequant_scale *= pos_scale;
  });

  uint8_t input_zero_point = 0;
  if (input_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(input_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    input_zero_point = *input_zp_tensor->template Data<uint8_t>();
  }
  uint8_t pos_zero_point = 0;
  if (pos_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(pos_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    pos_zero_point = *pos_zp_tensor->template Data<uint8_t>();
  }

  bool is_i_weight_zp_per_column = false;
  uint8_t i_weight_zp_default = 0;
  const uint8_t* i_weight_zp_data = nullptr;
  if (iw_zp_tensor != nullptr) {
    is_i_weight_zp_per_column = !IsScalarOr1ElementVector(iw_zp_tensor);
    i_weight_zp_data = static_cast<const uint8_t*>(iw_zp_tensor->DataRaw());
  }
  bool is_p_weight_zp_per_column = false;
  uint8_t p_weight_zp_default = 0;
  const uint8_t* p_weight_zp_data = nullptr;
  if (pw_zp_tensor != nullptr) {
    is_p_weight_zp_per_column = !IsScalarOr1ElementVector(pw_zp_tensor);
    p_weight_zp_data = static_cast<const uint8_t*>(pw_zp_tensor->DataRaw());
  }

  const auto& input_shape = input->Shape();
  const auto& pos_shape = pos_emb->Shape();
  const int batch_size = static_cast<int>(input_shape[0]);
  const int sequence_length = static_cast<int>(input_shape[1]);
  const int pos_sequence_length = static_cast<int>(pos_shape[1]);

  const int input_hidden_size = static_cast<int>(input_shape[2]);
  const auto pos_hidden_size = static_cast<int>(pos_shape[2]);
  const int head_size = input_hidden_size / num_heads_;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = input_shape[0];
  output_shape[1] = input_shape[1];
  output_shape[2] = static_cast<int64_t>(input_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  constexpr size_t element_size = sizeof(T);

  auto* tp = context->GetOperatorThreadPool();
  // STEP.1: gemm_data(BS, 3NH) = Scale(input(BS, D) x weights(D, 3NH)) + bias(3NH)
  // D is hidden dimension of input, where input_hidden_size (D) could be larger than hidden_size (NH) when model is pruned.
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * (sequence_length * 3 * input_hidden_size + pos_sequence_length * pos_hidden_size) * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + static_cast<int64_t>(batch_size) * sequence_length * input_hidden_size;
  auto V = K + static_cast<int64_t>(batch_size) * sequence_length * input_hidden_size;
  auto P = V + static_cast<int64_t>(batch_size) * sequence_length * input_hidden_size;

  T* QKV[3] = {Q, K, V};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto* input_data = input->template Data<uint8_t>();
    const auto* bias_data = bias->template Data<T>();

    const auto* weights_data = packed_weights_ ? nullptr : static_cast<const uint8_t*>(weights->DataRaw());
    const bool weights_is_signed = packed_weights_ ? weights_is_signed_ : weights->IsDataType<int8_t>();

    MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
    gemm_shape.M = sequence_length;
    gemm_shape.N = head_size;
    gemm_shape.K = input_hidden_size;
    gemm_shape.BIsSigned = weights_is_signed;

    std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_data_vec(loop_len);
    std::vector<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> scale_bias_procs;
    scale_bias_procs.reserve(loop_len);

    for (int i = 0; i < loop_len; i++) {
      const int batch_index = static_cast<int>((i / 3) / num_heads_);
      const int head_index = static_cast<int>((i / 3) % num_heads_);
      const int qkv_index = static_cast<int>(i % 3);

      int input_offset = batch_index * sequence_length * input_hidden_size;
      int weights_offset = qkv_index * input_hidden_size + head_index * head_size;
      int weights_scale_offset = is_i_weight_scale_per_column ? weights_offset : 0;
      int weights_zp_offset = is_i_weight_zp_per_column ? weights_offset : 0;
      float* qkv_dest = QKV[qkv_index];
      int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size);

      //                   original           transposed            iteration
      // A: input          (BxSxD)            (B.)S x D             S x D
      // B: weights        (Dx3xNxH)          D  x (3.N.)H          D x H
      // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H

      scale_bias_procs.emplace_back(qkv_dest + qkv_offset,
                                    head_size,
                                    i_dequant_scales.data() + weights_scale_offset,
                                    bias_data + weights_offset,
                                    MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                                    is_i_weight_scale_per_column ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);

      auto& gemm_params = gemm_data_vec[i];
      gemm_params.A = input_data + input_offset;
      gemm_params.lda = input_hidden_size;
      gemm_params.ZeroPointA = input_zero_point;
      if (packed_weights_) {
        const auto* packed_weight =
            static_cast<const uint8_t*>(packed_weights_.get()) + packed_weights_size_ * (weights_offset / head_size);
        gemm_params.B = packed_weight;
        gemm_params.BIsPacked = true;
      } else {
        gemm_params.B = weights_data + weights_offset;
        gemm_params.ldb = static_cast<int64_t>(3) * input_hidden_size;
      }
      gemm_params.ZeroPointB = nullptr != i_weight_zp_data ? i_weight_zp_data + weights_zp_offset : &i_weight_zp_default;
      gemm_params.PerColumnZeroPoints = is_i_weight_zp_per_column;
      gemm_params.C = reinterpret_cast<int32_t*>(qkv_dest + qkv_offset);
      gemm_params.ldc = head_size;
      gemm_params.OutputProcessor = &(scale_bias_procs[i]);
    }

    MlasGemmBatch(gemm_shape, gemm_data_vec.data(), loop_len, tp);
  }

  {
    const int loop_len = batch_size * num_heads_;
    const auto* input_data = pos_emb->template Data<uint8_t>();

    const auto* weights_data = packed_weights_ ? nullptr : static_cast<const uint8_t*>(pos_weights->DataRaw());
    const bool weights_is_signed = packed_weights_ ? weights_is_signed_ : pos_weights->IsDataType<int8_t>();

    MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
    gemm_shape.M = pos_sequence_length;
    gemm_shape.N = head_size;
    gemm_shape.K = input_hidden_size;
    gemm_shape.BIsSigned = weights_is_signed;

    std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_data_vec(loop_len);
    std::vector<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> scale_bias_procs;
    scale_bias_procs.reserve(loop_len);

    for (int i = 0; i < loop_len; i++) {
      const int batch_index = static_cast<int>(i / num_heads_);
      const int head_index = static_cast<int>(i % num_heads_);

      int input_offset = batch_index * pos_sequence_length * input_hidden_size;
      int weights_offset = head_index * head_size;
      int weights_scale_offset = is_p_weight_scale_per_column ? weights_offset : 0;
      int weights_zp_offset = is_p_weight_zp_per_column ? weights_offset : 0;
      int p_offset = (batch_index * num_heads_ + head_index) * (pos_sequence_length * head_size);

      //                   original           transposed            iteration
      // A: input          (BxSxD)            (B.)S x D             S x D
      // B: weights        (Dx3xNxH)          D  x (3.N.)H          D x H
      // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H

      // initialize bias data for position embedding.
      scale_bias_procs.emplace_back(P + p_offset,
                                    head_size,
                                    p_dequant_scales.data() + weights_scale_offset,
                                    nullptr,
                                    MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                                    is_p_weight_scale_per_column ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);

      auto& gemm_params = gemm_data_vec[i];
      gemm_params.A = input_data + input_offset;
      gemm_params.lda = input_hidden_size;
      gemm_params.ZeroPointA = pos_zero_point;
      if (packed_weights_) {
        const auto* packed_weight =
            static_cast<const uint8_t*>(packed_weights_.get()) + packed_weights_size_ * (weights_offset / head_size);
        gemm_params.B = packed_weight;
        gemm_params.BIsPacked = true;
      } else {
        gemm_params.B = weights_data + weights_offset;
        gemm_params.ldb = input_hidden_size;
      }
      gemm_params.ZeroPointB = nullptr != p_weight_zp_data ? p_weight_zp_data + weights_zp_offset : &p_weight_zp_default;
      gemm_params.PerColumnZeroPoints = is_p_weight_zp_per_column;
      gemm_params.C = reinterpret_cast<int32_t*>(P + p_offset);
      gemm_params.ldc = head_size;
      gemm_params.OutputProcessor = &(scale_bias_procs[i]);
    }

    MlasGemmBatch(gemm_shape, gemm_data_vec.data(), loop_len, tp);
  }

  // Compute the attention score and apply the score to V
  return ApplyRelPosAttention(Q, K, V, P, pos_bias_u, pos_bias_v, mask_index, output,
                              batch_size, sequence_length, pos_sequence_length,
                              head_size, head_size, input_hidden_size, context);
}

}  // namespace contrib
}  // namespace onnxruntime