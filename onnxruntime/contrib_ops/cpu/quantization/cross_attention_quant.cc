// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications by Masao Someki
// Copyright (c) 2022 Masao Someki

#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/bert/cross_attention_cpu_base.h"
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
class QCrossAttention : public OpKernel, public CrossAttentionCPUBase {
 public:
  QCrossAttention(const OpKernelInfo& info);

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
    QCrossAttention,
    kENDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QCrossAttention<float>);

template <typename T>
QCrossAttention<T>::QCrossAttention(const OpKernelInfo& info) : OpKernel(info), CrossAttentionCPUBase(info) {}

template <typename T>
Status QCrossAttention<T>::Compute(OpKernelContext* context) const {
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
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* q_weights = packed_weights_ ? nullptr : context->Input<Tensor>(2);
  const Tensor* kv_weights = packed_weights_ ? nullptr : context->Input<Tensor>(3);
  const Tensor* q_bias = context->Input<Tensor>(4);
  const Tensor* kv_bias = context->Input<Tensor>(5);
  const Tensor* query_scale_tensor = context->Input<Tensor>(6);
  const Tensor* key_scale_tensor = context->Input<Tensor>(7);
  const Tensor* query_weight_scale_tensor = context->Input<Tensor>(8);
  const Tensor* key_weight_scale_tensor = context->Input<Tensor>(9);
  const Tensor* mask_index = context->Input<Tensor>(10);
  const Tensor* query_zp_tensor = context->Input<Tensor>(11);
  const Tensor* key_zp_tensor = context->Input<Tensor>(12);
  const Tensor* qw_zp_tensor = context->Input<Tensor>(13);
  const Tensor* kw_zp_tensor = context->Input<Tensor>(14);

  const TensorShape& q_weights_shape = (packed_weights_ ? q_weight_shape_ : q_weights->Shape());
  const TensorShape& kv_weights_shape = (packed_weights_ ? kv_weight_shape_ : kv_weights->Shape());
  ORT_RETURN_IF_ERROR(CrossAttentionBase::CheckInputs(query->Shape(),
                                                 key->Shape(),
                                                 q_weights_shape,
                                                 kv_weights_shape,
                                                 q_bias->Shape(),
                                                 kv_bias->Shape(),
                                                 mask_index));

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(query_scale_tensor),
                    "input scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(key_scale_tensor),
                    "input scale must be a scalar or 1D tensor of size 1");
  T query_scale = *(query_scale_tensor->template Data<T>());
  T key_scale = *(key_scale_tensor->template Data<T>());

  bool is_q_weight_scale_per_column = !IsScalarOr1ElementVector(query_weight_scale_tensor);
  bool is_k_weight_scale_per_column = !IsScalarOr1ElementVector(key_weight_scale_tensor);
  const T* q_weight_scale_data = query_weight_scale_tensor->template Data<T>();
  const T* k_weight_scale_data = key_weight_scale_tensor->template Data<T>();

  std::vector<T> q_dequant_scales(q_weight_scale_data, q_weight_scale_data + query_weight_scale_tensor->Shape().Size());
  std::vector<T> k_dequant_scales(k_weight_scale_data, k_weight_scale_data + key_weight_scale_tensor->Shape().Size());
  std::for_each(q_dequant_scales.begin(), q_dequant_scales.end(), [&query_scale](float& q_dequant_scale) {
    return q_dequant_scale *= query_scale;
  });
  std::for_each(k_dequant_scales.begin(), k_dequant_scales.end(), [&key_scale](float& k_dequant_scale) {
    return k_dequant_scale *= key_scale;
  });

  uint8_t query_zero_point = 0;
  if (query_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(query_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    query_zero_point = *query_zp_tensor->template Data<uint8_t>();
  }
  uint8_t key_zero_point = 0;
  if (key_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(key_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    key_zero_point = *key_zp_tensor->template Data<uint8_t>();
  }

  bool is_q_weight_zp_per_column = false;
  uint8_t q_weight_zp_default = 0;
  const uint8_t* q_weight_zp_data = nullptr;
  if (qw_zp_tensor != nullptr) {
    is_q_weight_zp_per_column = !IsScalarOr1ElementVector(qw_zp_tensor);
    q_weight_zp_data = static_cast<const uint8_t*>(qw_zp_tensor->DataRaw());
  }
  bool is_k_weight_zp_per_column = false;
  uint8_t k_weight_zp_default = 0;
  const uint8_t* k_weight_zp_data = nullptr;
  if (kw_zp_tensor != nullptr) {
    is_k_weight_zp_per_column = !IsScalarOr1ElementVector(kw_zp_tensor);
    k_weight_zp_data = static_cast<const uint8_t*>(kw_zp_tensor->DataRaw());
  }

  const auto& q_shape = query->Shape();
  const auto& kv_shape = key->Shape();
  const int batch_size = static_cast<int>(q_shape[0]);
  const int sequence_length = static_cast<int>(q_shape[1]);
  const int kv_sequence_length = static_cast<int>(kv_shape[1]);

  const int q_hidden_size = static_cast<int>(q_shape[2]);
  const auto kv_hidden_size = static_cast<int>(kv_shape[2]);
  const int head_size = q_hidden_size / num_heads_;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = q_shape[0];
  output_shape[1] = q_shape[1];
  output_shape[2] = static_cast<int64_t>(q_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  constexpr size_t element_size = sizeof(T);

  auto* tp = context->GetOperatorThreadPool();
  // STEP.1: gemm_data(BS, 3NH) = Scale(input(BS, D) x weights(D, 3NH)) + bias(3NH)
  // D is hidden dimension of input, where input_hidden_size (D) could be larger than hidden_size (NH) when model is pruned.
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * (sequence_length * q_hidden_size + kv_sequence_length * 2 * kv_hidden_size) * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + static_cast<int64_t>(batch_size) * sequence_length * q_hidden_size;
  auto V = K + static_cast<int64_t>(batch_size) * kv_sequence_length * kv_hidden_size;

  {
    const int loop_len = batch_size * num_heads_;
    const auto* input_data = query->template Data<uint8_t>();
    const auto* bias_data = q_bias->template Data<T>();

    const auto* weights_data = packed_weights_ ? nullptr : static_cast<const uint8_t*>(q_weights->DataRaw());
    const bool weights_is_signed = packed_weights_ ? weights_is_signed_ : q_weights->IsDataType<int8_t>();

    MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
    gemm_shape.M = sequence_length;
    gemm_shape.N = head_size;
    gemm_shape.K = q_hidden_size;
    gemm_shape.BIsSigned = weights_is_signed;

    std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_data_vec(loop_len);
    std::vector<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> scale_bias_procs;
    scale_bias_procs.reserve(loop_len);

    for (int i = 0; i < loop_len; i++) {
      const int batch_index = static_cast<int>(i / num_heads_);
      const int head_index = static_cast<int>(i % num_heads_);

      int input_offset = batch_index * sequence_length * q_hidden_size;
      int weights_offset = head_index * head_size;
      int weights_scale_offset = is_q_weight_scale_per_column ? weights_offset : 0;
      int weights_zp_offset = is_q_weight_zp_per_column ? weights_offset : 0;
      int q_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size);

      //                   original           transposed            iteration
      // A: input          (BxSxD)            (B.)S x D             S x D
      // B: weights        (Dx3xNxH)          D  x (3.N.)H          D x H
      // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H

      scale_bias_procs.emplace_back(Q + q_offset,
                                    head_size,
                                    q_dequant_scales.data() + weights_scale_offset,
                                    bias_data + weights_offset,
                                    MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                                    is_q_weight_scale_per_column ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);

      auto& gemm_params = gemm_data_vec[i];
      gemm_params.A = input_data + input_offset;
      gemm_params.lda = q_hidden_size;
      gemm_params.ZeroPointA = query_zero_point;
      if (packed_weights_) {
        const auto* packed_weight =
            static_cast<const uint8_t*>(packed_weights_.get()) + packed_weights_size_ * (weights_offset / head_size);
        gemm_params.B = packed_weight;
        gemm_params.BIsPacked = true;
      } else {
        gemm_params.B = weights_data + weights_offset;
        gemm_params.ldb = q_hidden_size;
      }
      gemm_params.ZeroPointB = nullptr != q_weight_zp_data ? q_weight_zp_data + weights_zp_offset : &q_weight_zp_default;
      gemm_params.PerColumnZeroPoints = is_q_weight_zp_per_column;
      gemm_params.C = reinterpret_cast<int32_t*>(Q + q_offset);
      gemm_params.ldc = head_size;
      gemm_params.OutputProcessor = &(scale_bias_procs[i]);
    }

    MlasGemmBatch(gemm_shape, gemm_data_vec.data(), loop_len, tp);
  }

  T* KV[2] = {K, V};
  {
    const int loop_len = 2 * batch_size * num_heads_;
    const auto* input_data = key->template Data<uint8_t>();
    const auto* bias_data = kv_bias->template Data<T>();

    const auto* weights_data = packed_weights_ ? nullptr : static_cast<const uint8_t*>(kv_weights->DataRaw());
    const bool weights_is_signed = packed_weights_ ? weights_is_signed_ : kv_weights->IsDataType<int8_t>();

    MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
    gemm_shape.M = kv_sequence_length;
    gemm_shape.N = head_size;
    gemm_shape.K = kv_hidden_size;
    gemm_shape.BIsSigned = weights_is_signed;

    std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_data_vec(loop_len);
    std::vector<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> scale_bias_procs;
    scale_bias_procs.reserve(loop_len);

    for (int i = 0; i < loop_len; i++) {
      const int batch_index = static_cast<int>((i / 2) / num_heads_);
      const int head_index = static_cast<int>((i / 2) % num_heads_);
      const int kv_index = static_cast<int>(i % 2);

      int input_offset = batch_index * kv_sequence_length * kv_hidden_size;
      int weights_offset = kv_index * kv_hidden_size + head_index * head_size;
      int weights_scale_offset = is_k_weight_scale_per_column ? weights_offset : 0;
      int weights_zp_offset = is_k_weight_zp_per_column ? weights_offset : 0;
      float* kv_dest = KV[kv_index];
      int kv_offset = (batch_index * num_heads_ + head_index) * (kv_sequence_length * head_size);

      //                   original           transposed            iteration
      // A: input          (BxSxD)            (B.)S x D             S x D
      // B: weights        (Dx3xNxH)          D  x (3.N.)H          D x H
      // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H

      scale_bias_procs.emplace_back(kv_dest + kv_offset,
                                    head_size,
                                    k_dequant_scales.data() + weights_scale_offset,
                                    bias_data + weights_offset,
                                    MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                                    is_k_weight_scale_per_column ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);

      auto& gemm_params = gemm_data_vec[i];
      gemm_params.A = input_data + input_offset;
      gemm_params.lda = kv_hidden_size;
      gemm_params.ZeroPointA = key_zero_point;
      if (packed_weights_) {
        const auto* packed_weight =
            static_cast<const uint8_t*>(packed_weights_.get()) + packed_weights_size_ * (weights_offset / head_size);
        gemm_params.B = packed_weight;
        gemm_params.BIsPacked = true;
      } else {
        gemm_params.B = weights_data + weights_offset;
        gemm_params.ldb = static_cast<int64_t>(2) * kv_hidden_size;
      }
      gemm_params.ZeroPointB = nullptr != k_weight_zp_data ? k_weight_zp_data + weights_zp_offset : &k_weight_zp_default;
      gemm_params.PerColumnZeroPoints = is_k_weight_zp_per_column;
      gemm_params.C = reinterpret_cast<int32_t*>(kv_dest + kv_offset);
      gemm_params.ldc = head_size;
      gemm_params.OutputProcessor = &(scale_bias_procs[i]);
    }

    MlasGemmBatch(gemm_shape, gemm_data_vec.data(), loop_len, tp);
  }

  // Compute the attention score and apply the score to V
  return ApplyCrossAttention(Q, K, V, mask_index, output,
                        batch_size, sequence_length, kv_sequence_length,
                        head_size, head_size, q_hidden_size, context);
}

}  // namespace contrib
}  // namespace onnxruntime
