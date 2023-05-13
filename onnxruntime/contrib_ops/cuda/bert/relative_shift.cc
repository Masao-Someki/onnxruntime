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
    Stream(context),
    sequence_length,
    pos_sequence_length,
    batch_size,
    num_heads,
    reinterpret_cast<const CudaT*>(matrix_ac->template Data<T>()),
    reinterpret_cast<const CudaT*>(matrix_bd->template Data<T>()),
    output->template MutableData<T>())) {
      return CUDA_CALL(cudaGetLastError());
    }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
