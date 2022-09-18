// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modification by Masao Someki

#include "cross_attention.h"
#include "attention_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      CrossAttention,                                                  \
      kENDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      CrossAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
CrossAttention<T>::CrossAttention(const OpKernelInfo& info) : CudaKernel(info), CrossAttentionBase(info) {}

template <typename T>
Status CrossAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* q_weights = context->Input<Tensor>(2);
  const Tensor* kv_weights = context->Input<Tensor>(3);
  const Tensor* q_bias = context->Input<Tensor>(4);
  const Tensor* kv_bias = context->Input<Tensor>(5);
  const Tensor* mask_index = context->Input<Tensor>(6);

  auto& device_prop = GetDeviceProp();
  // ORT_RETURN_IF_ERROR(CheckInputs(
  //   query->Shape(), key->Shape(), q_weights->Shape(), kv_weights->Shape(),
  //   q_bias->Shape(), kv_bias->Shape(), mask_index, device_prop.maxThreadsPerBlock));

  // input shape (batch_size, sequence_length, input_hidden_size)
  const auto& q_shape = query->Shape();
  const auto& kv_shape = key->Shape();
  int batch_size = static_cast<int>(q_shape[0]);
  int sequence_length = static_cast<int>(q_shape[1]);
  int kv_sequence_length = static_cast<int>(kv_shape[1]);
  int input_hidden_size = static_cast<int>(q_shape[2]);

  int head_size = input_hidden_size / num_heads_;

  TensorShapeVector output_shape(3);
  output_shape[0] = q_shape[0];
  output_shape[1] = q_shape[1];
  output_shape[2] = static_cast<int64_t>(input_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  cublasHandle_t cublas = CublasHandle();
  constexpr size_t element_size = sizeof(T);

  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = input_hidden_size;
  int k = input_hidden_size;
  auto q_gemm_buffer = GetScratchBuffer<T>(batch_size * sequence_length * input_hidden_size * element_size);

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // compute query
  // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  // TODO: use custom kernel of expand to improve the performance.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
      reinterpret_cast<const CudaT*>(q_bias->template Data<T>()), n,
      GetConstOnes<CudaT>(m), 1,
      &zero, reinterpret_cast<CudaT*>(q_gemm_buffer.get()), n, device_prop));

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(q_weights->template Data<T>()), n,
      reinterpret_cast<const CudaT*>(query->template Data<T>()), k,
      &one, reinterpret_cast<CudaT*>(q_gemm_buffer.get()), n, device_prop));

  // Use GEMM for fully connection.
  m = batch_size * kv_sequence_length;
  n = input_hidden_size;
  k = input_hidden_size;
  auto kv_gemm_buffer = GetScratchBuffer<T>(batch_size * kv_sequence_length * 2 * input_hidden_size * element_size);

  // compute query
  // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  // TODO: use custom kernel of expand to improve the performance.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
      reinterpret_cast<const CudaT*>(kv_bias->template Data<T>()), n,
      GetConstOnes<CudaT>(m), 1,
      &zero, reinterpret_cast<CudaT*>(kv_gemm_buffer.get()), n, device_prop));

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(kv_weights->template Data<T>()), n,
      reinterpret_cast<const CudaT*>(key->template Data<T>()), k,
      &one, reinterpret_cast<CudaT*>(kv_gemm_buffer.get()), n, device_prop));

  auto qkv_buffer_p = GetScratchBuffer<void>(batch_size * (sequence_length + 2 * kv_sequence_length) * input_hidden_size * element_size);
  auto workspace_p = GetScratchBuffer<void>(2 * batch_size * sequence_length * num_heads_ * element_size * (2 * head_size + kv_sequence_length));

  if (!LaunchCrossAttentionKernel(
          device_prop,
          Stream(),
          reinterpret_cast<const CudaT*>(q_gemm_buffer.get()),
          reinterpret_cast<const CudaT*>(kv_gemm_buffer.get()),
          nullptr == mask_index ? nullptr : mask_index->template Data<int>(),
          nullptr == mask_index ? gsl::span<const int64_t>() : mask_index->Shape().GetDims(),
          output->template MutableData<T>(),
          batch_size,
          sequence_length,
          kv_sequence_length,
          num_heads_,
          head_size,
          qkv_buffer_p.get(),
          workspace_p.get(),
          cublas,
          element_size)) {
    // Get last error to reset it to cudaSuccess.
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
