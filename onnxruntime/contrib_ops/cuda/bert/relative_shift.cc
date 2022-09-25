// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modification by Masao Someki

#include "relative_shift.h"
#include "relative_shift_impl.h"
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
      RelativeShift,                                            \
      kENDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      RelativeShift<T>);

REGISTER_KERNEL_TYPED(float)
// REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
RelativeShift<T>::RelativeShift(const OpKernelInfo& info) : CudaKernel(info) {}

template <typename T>
Status RelativeShift<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* matrix_ac = context->Input<Tensor>(0);
  const Tensor* matrix_bd = context->Input<Tensor>(1);

  // input shape (batch_size, sequence_length, input_hidden_size)
  const auto& input_shape = matrix_ac->Shape();
  const int batch_size = static_cast<int>(input_shape[0]);
  const int num_heads = static_cast<int>(input_shape[1]);
  const int sequence_length = static_cast<int>(input_shape[2]);
  const int pos_sequence_length = static_cast<int>(input_shape[3]);

  TensorShapeVector output_shape(4);
  output_shape[0] = input_shape[0];
  output_shape[1] = input_shape[1];
  output_shape[2] = input_shape[2];
  output_shape[3] = input_shape[2];
  Tensor* output = context->Output(0, output_shape);

  // apply relative shift to input tensor
  typedef typename ToCudaType<T>::MappedType CudaT;
  if (!LaunchRelShiftAdd(
    Stream(),
    sequence_length,
    pos_sequence_length,
    batch_size,
    num_heads,
    reinterpret_cast<const CudaT*>(matrix_ac->template Data<T>()),
    reinterpret_cast<const CudaT*>(matrix_bd->template Data<T>()),
    output->template MutableData<T>())) {
      CUDA_CALL(cudaGetLastError());
      return Status(common::ONNXRUNTIME, common::FAIL);
    }
  // cublasHandle_t cublas = CublasHandle();
  // constexpr size_t element_size = sizeof(T);

  // // Use GEMM for fully connection.
  // int m = batch_size * 2 * sequence_length;
  // int n = input_hidden_size;
  // int k = input_hidden_size;
  // auto gemm_buffer = GetScratchBuffer<T>(
  //     batch_size * 3 * sequence_length * input_hidden_size * element_size);
  // int Q2_offset = batch_size * 2 * sequence_length * input_hidden_size * element_size;
  // // Q1 (for u), K, V, Q2 (for v)

  // typedef typename ToCudaType<T>::MappedType CudaT;
  // CudaT one = ToCudaType<T>::FromFloat(1.0f);
  // CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // // compute projection for Q,K
  // // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
  // // TODO: use custom kernel of expand to improve the performance.
  // CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
  //     cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &one,
  //     reinterpret_cast<const CudaT*>(bias->template Data<T>()), n,
  //     GetConstOnes<CudaT>(m), 1,
  //     &zero, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  // // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
  // CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
  //     cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
  //     reinterpret_cast<const CudaT*>(weights->template Data<T>()), n,
  //     reinterpret_cast<const CudaT*>(input->template Data<T>()), k,
  //     &one, reinterpret_cast<CudaT*>(gemm_buffer.get()), n, device_prop));

  // // Copy Q1 to the Q2
  // int copy_size = batch_size * sequence_length * input_hidden_size * element_size;
  // CUBLAS_RETURN_IF_ERROR(cublasCopyHelper(
  //   Stream(),
  //   cublas,
  //   copy_size,
  //   reinterpret_cast<CudaT*>(gemm_buffer.get()), 1,
  //   reinterpret_cast<CudaT*>(gemm_buffer.get() + Q2_offset), 1));

  // // Compute projection for positional embedding
  // m = batch_size * pos_sequence_length;
  // n = input_hidden_size;
  // k = input_hidden_size;
  // const int pos_buffer_size = batch_size * pos_sequence_length * input_hidden_size * element_size;
  // auto pos_gemm_buffer = GetScratchBuffer<T>(pos_buffer_size);

  // // set beta to be zero, since bias is zero and the buffer is not initialized.
  // // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x B.
  // CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
  //     cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
  //     reinterpret_cast<const CudaT*>(pos_weights->template Data<T>()), n,
  //     reinterpret_cast<const CudaT*>(pos_emb->template Data<T>()), k,
  //     &zero, reinterpret_cast<CudaT*>(pos_gemm_buffer.get()), n, device_prop));

  // int workspace_size = batch_size * num_heads_ * sequence_length * (2 * sequence_length - 1) * element_size
  //   + batch_size * num_heads_ * sequence_length * sequence_length * element_size
  //   + batch_size * sequence_length * input_hidden_size * element_size;

  // auto workspace_p = GetScratchBuffer<void>(workspace_size);
  // int qkvp_buffer_size = 3 * batch_size * sequence_length * input_hidden_size * element_size
  //   + batch_size * pos_sequence_length * input_hidden_size * element_size;
  // auto qkvp_buffer = GetScratchBuffer<void>(qkvp_buffer_size);

  // if (!LaunchRelPosAttentionKernel(
  //         device_prop,
  //         stream,
  //         reinterpret_cast<const CudaT*>(gemm_buffer.get()),
  //         reinterpret_cast<const CudaT*>(pos_gemm_buffer.get()),
  //         nullptr == mask_index ? nullptr : mask_index->template Data<int>(),
  //         nullptr == mask_index ? gsl::span<const int64_t>() : mask_index->Shape().GetDims(),
  //         output->template MutableData<T>(),
  //         reinterpret_cast<const CudaT*>(pos_bias_u->template Data<T>()),
  //         reinterpret_cast<const CudaT*>(pos_bias_v->template Data<T>()),
  //         batch_size,
  //         sequence_length,
  //         pos_sequence_length,
  //         num_heads_,
  //         head_size,
  //         qkvp_buffer.get(),
  //         workspace_p.get(),
  //         cublas,
  //         element_size)) {
  //   // Get last error to reset it to cudaSuccess.
  //   CUDA_CALL(cudaGetLastError());
  //   return Status(common::ONNXRUNTIME, common::FAIL);
  // }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
