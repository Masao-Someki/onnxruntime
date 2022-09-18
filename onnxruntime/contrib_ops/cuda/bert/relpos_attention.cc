// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modification by Masao Someki

#include "relpos_attention.h"
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
      RelPosAttention,                                            \
      kENDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      RelPosAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
RelPosAttention<T>::RelPosAttention(const OpKernelInfo& info) : CudaKernel(info), RelPosAttentionBase(info) {}

template <typename T>
Status RelPosAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* pos_emb = context->Input<Tensor>(2);
  const Tensor* pos_weights = context->Input<Tensor>(3);
  const Tensor* bias = context->Input<Tensor>(4);
  const Tensor* pos_bias_u = context->Input<Tensor>(5);
  const Tensor* pos_bias_v = context->Input<Tensor>(6);
  const Tensor* mask_index = context->Input<Tensor>(7);

  auto& device_prop = GetDeviceProp();

  // ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
  //                                 weights->Shape(),
  //                                 bias->Shape(),
  //                                 pos_emb->Shape(),
  //                                 pos_bias_u->Shape(),
  //                                 pos_bias_v->Shape(),
  //                                 mask_index,
  //                                 device_prop.maxThreadsPerBlock));

  // input shape (batch_size, sequence_length, input_hidden_size)
  const auto& input_shape = input->Shape();
  const auto& p_shape = pos_emb->Shape();
  int batch_size = static_cast<int>(input_shape[0]);
  int sequence_length = static_cast<int>(input_shape[1]);
  int pos_sequence_length = static_cast<int>(p_shape[1]);
  int input_hidden_size = static_cast<int>(input_shape[2]);

  int head_size = input_hidden_size / num_heads_;

  TensorShapeVector output_shape(3);
  output_shape[0] = input_shape[0];
  output_shape[1] = input_shape[1];
  output_shape[2] = static_cast<int64_t>(input_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  cublasHandle_t cublas = CublasHandle();
  constexpr size_t element_size = sizeof(T);

  // Use GEMM for fully connection.
  int m = batch_size * 3 * sequence_length;
  int n = input_hidden_size;
  int k = input_hidden_size;
  auto gemm_buffer = GetScratchBuffer<T>(
      batch_size * 4 * sequence_length * input_hidden_size * element_size);
  int Q2_offset = batch_size * 3 * sequence_length * input_hidden_size * element_size;
  // Q1 (for u), K, V, Q2 (for v)

  typedef typename ToCudaType<T>::MappedType CudaT;
  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // compute projection for Q,K,V
  // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  // TODO: use custom kernel of expand to improve the performance.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
      reinterpret_cast<const CudaT*>(bias->template Data<T>()), n,
      GetConstOnes<CudaT>(m), 1,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->template Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->template Data<T>()), k,
      &one, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  // Copy Q1 to the Q2
  int copy_size = batch_size * sequence_length * input_hidden_size * element_size;
  CUBLAS_RETURN_IF_ERROR(cublasCopyHelper(
    Stream(),
    cublas,
    copy_size,
    reinterpret_cast<CudaT*>(gemm_buffer.get()), 1,
    reinterpret_cast<CudaT*>(gemm_buffer.get() + Q2_offset), 1));

  // Compute projection for positional embedding
  m = batch_size * pos_sequence_length;
  n = input_hidden_size;
  k = input_hidden_size;
  const int pos_buffer_size = batch_size * pos_sequence_length * input_hidden_size * element_size;
  auto pos_gemm_buffer = GetScratchBuffer<T>(pos_buffer_size);

  // set beta to be zero, since bias is zero and the buffer is not initialized.
  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(pos_weights->template Data<T>()), n,
      reinterpret_cast<const CudaT*>(pos_emb->template Data<T>()), k,
      &zero, reinterpret_cast<CudaT*>(pos_gemm_buffer.get()), n, device_prop));

  int workspace_size = (batch_size * num_heads_ * sequence_length * (3 * sequence_length - 1)) * element_size + batch_size * num_heads_ * sequence_length * sequence_length * element_size + batch_size * sequence_length * input_hidden_size * element_size;
  auto workspace_p = GetScratchBuffer<void>(workspace_size);
  int qkvp_buffer_size = 4 * batch_size * sequence_length * input_hidden_size * element_size + batch_size * pos_sequence_length * input_hidden_size * element_size;
  auto qkvp_buffer = GetScratchBuffer<void>(qkvp_buffer_size);

  if (!LaunchRelPosAttentionKernel(
          device_prop,
          Stream(),
          reinterpret_cast<const CudaT*>(gemm_buffer.get()),
          reinterpret_cast<const CudaT*>(pos_gemm_buffer.get()),
          nullptr == mask_index ? nullptr : mask_index->template Data<int>(),
          nullptr == mask_index ? gsl::span<const int64_t>() : mask_index->Shape().GetDims(),
          output->template MutableData<T>(),
          reinterpret_cast<const CudaT*>(pos_bias_u->template Data<T>()),
          reinterpret_cast<const CudaT*>(pos_bias_v->template Data<T>()),
          batch_size,
          sequence_length,
          pos_sequence_length,
          num_heads_,
          head_size,
          qkvp_buffer.get(),
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
