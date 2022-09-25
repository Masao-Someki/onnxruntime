// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications by Masao Someki
// Copyright (c) 2022 Masao Someki

#include "relative_shift_base.h"

#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename T>
class RelativeShift : public OpKernel, public RelativeShiftBase {
 public:
  explicit RelativeShift(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    RelativeShift,
    kENDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    RelativeShift<float>);

Status RelativeShiftBase::CheckInputs(const TensorShape& input_shape,
                                        const TensorShape& pos_bias_shape) const {

  const auto& dims = input_shape.GetDims();
  if (dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is expected to have 4 dimensions, got ",
                           dims.size());
  }

  const auto& pos_dims = pos_bias_shape.GetDims();
  if (pos_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'pos_u_bias' is expected to have 4 dimension, got ",
                           pos_dims.size());
  }

  int sequence_length = dims[3];
  int pos_sequence_length = pos_dims[3];
  if (is_legacy_) {
    if (sequence_length != pos_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "sequence length must be same as pos_sequencE_length");
    }
  }
  else {
    if (pos_sequence_length != 2 * sequence_length - 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "pos sequence length must be same as 2 * sequence_length - 1");
    }
  }

  return Status::OK();
}

// Status RelPosAttentionBase::CheckInputs(const TensorShape& input_shape,
//                                         const TensorShape& weight_shape,
//                                         const TensorShape& bias_shape,
//                                         const TensorShape& pos_emb_shape,
//                                         const TensorShape& pos_bias_u_shape,
//                                         const TensorShape& pos_bias_v_shape,
//                                         const Tensor*& mask_index,
//                                         const int max_threads_per_block) const {
//   if (num_heads_ > max_threads_per_block) {
//     return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
//   }

//   return CheckInputs(input_shape, weight_shape, bias_shape, pos_emb_shape, pos_bias_u_shape, pos_bias_v_shape, mask_index);
// }

template <typename T>
RelativeShift<T>::RelativeShift(const OpKernelInfo& info) : OpKernel(info), RelativeShiftBase(info) {
}

template <typename T>
void ComputeLegacyRelShiftAdd(
                      const T* matrix_ac,                             // output buffer for the attention probs. Its size is BxNxSxS*
                      const T* matrix_bd,                       // Q data. Its size is BxNxSxH
                      T* output,
                      int batch_size,                           // batch size of self-attention
                      int sequence_length,                      // sequence length
                      int pos_sequence_length,                  // sequence length
                      int num_heads,
                      ThreadPool* tp                            // thread pool)
)
{
  {
    const int loop_len = batch_size * num_heads;

    const double cost = static_cast<double>(sequence_length) * sequence_length;
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end){
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int output_offset = static_cast<int>(i) * sequence_length * sequence_length;
        int bd_batch_offset = 0;
        int bd_sequence_offset = sequence_length - 1;

        for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
          const int oidx = static_cast<int>(seq_index) * sequence_length + output_offset;
          for (int j = 0; j < sequence_length; j++) {
            if (bd_sequence_offset == sequence_length) {
              bd_batch_offset += 1;
              output[oidx + j] = matrix_ac[oidx + j];
              bd_sequence_offset = 0;
            } else {
              const int bid = output_offset + bd_batch_offset * sequence_length + bd_sequence_offset;
              output[oidx + j] = matrix_ac[oidx + j] + matrix_bd[bid];
              bd_sequence_offset += 1;
            }
          }
        }
      }
    });
  }
}

template <typename T>
void ComputeRelShiftAdd(
                      const T* matrix_ac,                             // output buffer for the attention probs. Its size is BxNxSxS*
                      const T* matrix_bd,                       // Q data. Its size is BxNxSxH
                      T* output,
                      int batch_size,                           // batch size of self-attention
                      int sequence_length,                      // sequence length
                      int pos_sequence_length,                  // sequence length
                      int num_heads,
                      ThreadPool* tp                            // thread pool)
)
{
  {
    const int diff_pos_length = sequence_length - 1; // 2 * seq_len - 1 - seq_len = seq_len - 1
    const int loop_len = batch_size * num_heads;
    const double cost = static_cast<double>(sequence_length) * sequence_length;
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end){
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int output_offset = static_cast<int>(i) * sequence_length * sequence_length;
        int bd_input_offset = sequence_length * pos_sequence_length * static_cast<int>(i);

        for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
          const int oidx = static_cast<int>(seq_index) * sequence_length + output_offset;
          const int ibd = static_cast<int>(bd_input_offset) + seq_index * pos_sequence_length + (diff_pos_length - seq_index);
          for (int j = 0; j < sequence_length; j++) {
            output[oidx + j] = matrix_ac[oidx + j] + matrix_bd[ibd + j];
          }
        }
      }
    });
  }
}

template <typename T>
Status RelativeShift<T>::Compute(OpKernelContext* context) const {
  const Tensor* matrix_ac = context->Input<Tensor>(0);
  const Tensor* matrix_bd = context->Input<Tensor>(1);
  ORT_RETURN_IF_ERROR(CheckInputs(matrix_ac->Shape(),
                                  matrix_bd->Shape()
                                  ));

  const auto shape = matrix_ac->Shape().GetDims();
  const auto pos_shape = matrix_bd->Shape().GetDims();
  const int batch_size = static_cast<int>(shape[0]);
  const int num_heads = static_cast<int>(shape[1]);
  const int sequence_length = static_cast<int>(shape[2]);
  const int pos_sequence_length = static_cast<int>(pos_shape[3]);

  auto* tp = context->GetOperatorThreadPool();

  std::vector<int64_t> output_shape(4);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = shape[2];
  output_shape[3] = shape[2];
  Tensor* output = context->Output(0, output_shape);

  const auto* matrix_ac_data = matrix_ac->template Data<T>();
  const auto* matrix_bd_data = matrix_bd->template Data<T>();
  if (is_legacy_) {
    ComputeLegacyRelShiftAdd<T>(matrix_ac_data, matrix_bd_data, output->template MutableData<T>(),
                      batch_size, sequence_length, pos_sequence_length, num_heads, tp);
  } else {
    ComputeRelShiftAdd<T>(matrix_ac_data, matrix_bd_data, output->template MutableData<T>(),
                      batch_size, sequence_length, pos_sequence_length, num_heads, tp);
  }
  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
