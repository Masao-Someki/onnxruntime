// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications by Masao Someki
// Copyright (c) 2022 Masao Someki

#include "relpos_attention_cpu_base.h"
#include "attention_helper.h"

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
class RelPosAttention : public OpKernel, public RelPosAttentionCPUBase {
 public:
  explicit RelPosAttention(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    RelPosAttention,
    kENDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    RelPosAttention<float>);

Status RelPosAttentionBase::CheckInputs(const TensorShape& input_shape,
                                        const TensorShape& weights_shape,
                                        const TensorShape& bias_shape,
                                        const TensorShape& pos_emb_shape,
                                        const TensorShape& pos_bias_u_shape,
                                        const TensorShape& pos_bias_v_shape,
                                        const Tensor*& mask_index) const {
  // Input shapes:
  //   input       : (batch_size, sequence_length, input_hidden_size)
  //   weights     : (input_hidden_size, 3 * hidden_size)
  //   bias        : (3 * hidden_size)
  //   mask_index  : nullptr, (batch_size), (2 * batch_size),
  //                 or (batch_size, 1), (1, 1)
  //                 or (batch_size, past_sequence_length + sequence_length)
  //                 or (batch_size, sequence_length, past_sequence_length + sequence_length)
  //   past        : (2, batch_size, num_heads, past_sequence_length, head_size)
  //   extra_add_qk: (batch_size, num_heads, sequence_length, sequence_length)
  //
  // Where hidden_size = num_heads * head_size.
  // When a model is pruned (like some attention heads are removed), hidden_size < input_hidden_size.

  const auto& dims = input_shape.GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is expected to have 3 dimensions, got ",
                           dims.size());
  }
  int sequence_length = static_cast<int>(dims[1]);

  const auto& weights_dims = weights_shape.GetDims();
  if (weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'weights' is expected to have 2 dimensions, got ",
                           weights_dims.size());
  }
  if (weights_dims[0] != dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 0 should have same length as dimension 2 of input 0");
  }

  const auto& bias_dims = bias_shape.GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                           bias_dims.size());
  }

  const auto& pos_u_dims = pos_bias_u_shape.GetDims();
  if (pos_u_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'pos_u_bias' is expected to have 2 dimension, got ",
                           pos_u_dims.size());
  }
  const auto& pos_v_dims = pos_bias_v_shape.GetDims();
  if (pos_v_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'pos_v_bias' is expected to have 2 dimension, got ",
                           pos_v_dims.size());
  }

  int hidden_size = 0;

  if (qkv_hidden_sizes_.size() == 0) {
    hidden_size = static_cast<int>(weights_dims[1]) / 3;
    if (3 * hidden_size != static_cast<int>(weights_dims[1])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 1 dimension 1 should be 3 times of hidden dimension");
    }

    if (hidden_size % num_heads_ != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "hidden_size should be divisiable by num_heads.");
    }
  } else {
    int qkv_sizes = 0;

    if (qkv_hidden_sizes_.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "qkv_hidden_sizes attribute should have 3 elements");
    }

    if (qkv_hidden_sizes_[0] != qkv_hidden_sizes_[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "qkv_hidden_sizes first element should be same as the second");
    }

    for (size_t i = 0; i < qkv_hidden_sizes_.size(); i++) {
      if (qkv_hidden_sizes_[i] % num_heads_ != 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "hidden_size should be divisiable by num_heads:", qkv_hidden_sizes_[i]);
      }

      qkv_sizes += static_cast<int>(qkv_hidden_sizes_[i]);
    }

    int qkv_hidden_sizes_sum = static_cast<int>(weights_dims[1]);
    if (qkv_hidden_sizes_sum != qkv_sizes) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "qkv_sizes doesn't match the wights dimension");
    }

    hidden_size = static_cast<int>(qkv_hidden_sizes_[2]);
  }

  if (bias_dims[0] != weights_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'bias' dimension 0 should have same length as dimension 1 of input 'weights'");
  }

  const auto& pos_dims = pos_emb_shape.GetDims();
  if (pos_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'pos_emb' should be 3 dimensional tensor.");
  }

  int pos_length = static_cast<int>(pos_dims[1]);
  if (is_legacy_) {
    if (pos_length != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "The length of input 'pos_emb' should be sequence_length in legacy version.");
    }
  } else {
    if (pos_length != 2 * sequence_length - 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "The length of input 'pos_emb' should be 2 * sequence_length - 1.");
    }
  }

  int batch_size = dims[0];
  if (mask_index != nullptr) {  // mask_index is optional
    const auto& mask_dims = mask_index->Shape().GetDims();
    if (mask_dims.size() == 1) {
      if (static_cast<int>(mask_dims[0]) != batch_size && static_cast<int>(mask_dims[0]) != 2 * batch_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mask_index' with 1D data shall have length of batch_size or 2 * batch_size");
      }
    } else if (mask_dims.size() == 2) {
      if (static_cast<int>(mask_dims[0]) != batch_size || static_cast<int>(mask_dims[1]) != sequence_length) {
        // Add operator supports broadcasting. Here we handle a case with only one element in the 2nd dimension.
        if ((static_cast<int>(mask_dims[0]) == batch_size || static_cast<int>(mask_dims[0]) == 1) && static_cast<int>(mask_dims[1]) == 1) {
          // Mask will have same value after propogation, which has same effect as no mask.
          mask_index = nullptr;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mask_index' with 2D data shall have shape batch_size x (past_sequence_length + sequence_length)");
        }
      }
    } else if (mask_dims.size() == 3) {
      if (static_cast<int>(mask_dims[0]) != batch_size || static_cast<int>(mask_dims[2]) != sequence_length) {
        return ORT_MAKE_STATUS(
          ONNXRUNTIME,
          INVALID_ARGUMENT,
          "Inputs 'mask_index' with 3D data shall have shape batch_size x sequence_length x sequence_length ");
      }
    } else if (mask_dims.size() == 4) {
      if (static_cast<int>(mask_dims[0]) != batch_size || mask_dims[1] != 1 || mask_dims[2] != mask_dims[3] || mask_dims[2] < sequence_length) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mask_index' with 4D data shall have shape batch_size x 1 x max_sequence_length x max_sequence_length)");
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'mask_index' is expected to have 1, 2, 3 or 4 dimensions, got ",
                             mask_dims.size());
    }
  }

  return Status::OK();
}

Status RelPosAttentionBase::CheckInputs(const TensorShape& input_shape,
                                        const TensorShape& weight_shape,
                                        const TensorShape& bias_shape,
                                        const TensorShape& pos_emb_shape,
                                        const TensorShape& pos_bias_u_shape,
                                        const TensorShape& pos_bias_v_shape,
                                        const Tensor*& mask_index,
                                        const int max_threads_per_block) const {
  if (num_heads_ > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(input_shape, weight_shape, bias_shape, pos_emb_shape, pos_bias_u_shape, pos_bias_v_shape, mask_index);
}

template <typename T>
RelPosAttention<T>::RelPosAttention(const OpKernelInfo& info) : OpKernel(info), RelPosAttentionCPUBase(info) {
}

template <typename T>
Status RelPosAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* pos_emb = context->Input<Tensor>(2);
  const Tensor* pos_weights = context->Input<Tensor>(3);
  const Tensor* bias = context->Input<Tensor>(4);
  const Tensor* pos_bias_u = context->Input<Tensor>(5);
  const Tensor* pos_bias_v = context->Input<Tensor>(6);
  const Tensor* mask_index = context->Input<Tensor>(7);
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias->Shape(),
                                  pos_emb->Shape(),
                                  pos_bias_u->Shape(),
                                  pos_bias_v->Shape(),
                                  mask_index
                                  ));

  const auto shape = input->Shape().GetDims();
  const auto pos_shape = pos_emb->Shape().GetDims();
  const int batch_size = static_cast<int>(shape[0]);
  const int sequence_length = static_cast<int>(shape[1]);
  const int pos_sequence_length = pos_shape[1];
  const int input_hidden_size = static_cast<int>(shape[2]);

  int hidden_size;

  if (qkv_hidden_sizes_.size() == 0) {
    const auto& weights_dims = weights->Shape().GetDims();
    hidden_size = static_cast<int>(weights_dims[1]) / 3;
  } else {
    hidden_size = static_cast<int>(qkv_hidden_sizes_[2]);
  }

  const int head_size = hidden_size / num_heads_;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = static_cast<int64_t>(hidden_size);
  Tensor* output = context->Output(0, output_shape);

  constexpr size_t element_size = sizeof(T);

  int q_hidden_size = 0;
  int k_hidden_size = 0;
  int v_hidden_size = 0;
  if (qkv_hidden_sizes_.size() == 0) {
    q_hidden_size = hidden_size;
    k_hidden_size = hidden_size;
    v_hidden_size = hidden_size;
  } else {
    q_hidden_size = static_cast<int>(qkv_hidden_sizes_[0]);
    k_hidden_size = static_cast<int>(qkv_hidden_sizes_[1]);
    v_hidden_size = static_cast<int>(qkv_hidden_sizes_[2]);
  }
  const int qkv_head_size[3] = {q_hidden_size / num_heads_, k_hidden_size / num_heads_, v_hidden_size / num_heads_};

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto* tp = context->GetOperatorThreadPool();
  // Compute Q, K, V
  // gemm_data(BS, NT) = input(BS, D) x weights(D, NT) + bias(NT)
  // D (input_hidden_size) is hidden dimension of input, where D could be larger than any of the hidden_sizes
  // (NH) when model is pruned. T = H1 + H2 + H3, where H1, H2, H3 are head sizes of Q, K, V respectively
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) *
                                    (sequence_length * (q_hidden_size + k_hidden_size + v_hidden_size) + pos_sequence_length * q_hidden_size) * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + static_cast<size_t>(batch_size) * sequence_length * q_hidden_size;
  auto V = K + static_cast<size_t>(batch_size) * sequence_length * k_hidden_size;
  auto P = V + static_cast<size_t>(batch_size) * sequence_length * v_hidden_size;

  T* QKV[3] = {Q, K, V};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto* input_data = input->template Data<T>();
    const auto* weights_data = weights ? weights->template Data<T>() : nullptr;
    const auto* bias_data = bias->template Data<T>();

    const double cost = static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(input_hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3);

        int input_offset = batch_index * sequence_length * input_hidden_size;

        T* qkv_dest = QKV[qkv_index];
        int head_size = qkv_head_size[qkv_index];
        int weights_offset = 0;
        int bias_offset = qkv_index * q_hidden_size + head_index * head_size;

        weights_offset = bias_offset;

        int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size);

        // TODO!! memcpy here makes it not worthwhile to use Gemm batch. Possible to post process?
        // broadcast NH -> (B.N.S.H) for each of Q, K, V
        const T* broadcast_data_src = bias_data + bias_offset;
        T* broadcast_data_dest = QKV[qkv_index] + qkv_offset;

        for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
          memcpy(broadcast_data_dest, broadcast_data_src, head_size * sizeof(T));
          broadcast_data_dest += head_size;
        }

        //                   original           transposed            iteration
        // A: input          (BxSxD)            (B.)S x D             S x D
        // B: weights        (DxNxT)             D x (N.)T            D x H
        // C: QKV[qkv_index] (BxNxSxT)          (B.N.)S x T           S x H
        math::GemmEx<float, ThreadPool>(
            CblasNoTrans,                                   // TransA = no
            CblasNoTrans,                                   // TransB = no
            sequence_length,                                // M      = S
            head_size,                                      // N      = H
            input_hidden_size,                              // K      = D
            1.0f,                                           // alpha
            input_data + input_offset,                      // A
            input_hidden_size,                              // lda    = D
            weights_data + weights_offset,                  // B
            q_hidden_size + k_hidden_size + v_hidden_size,  // ldb = NH1 + NH2 + NH3
            1.0f,                                           // beta
            qkv_dest + qkv_offset,                          // C
            head_size,                                      // ldc
            nullptr                                         // use single-thread
        );
      }
    });
  }

  {
    const int loop_len = batch_size * num_heads_;
    const auto* pos_input_data = pos_emb->template Data<T>();
    const auto* pos_weights_data = pos_weights->template Data<T>();

    const double cost =
        static_cast<double>(pos_sequence_length) * static_cast<double>(head_size) * static_cast<double>(q_hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>(i / num_heads_);
        const int head_index = static_cast<int>(i % num_heads_);

        int input_offset = batch_index * pos_sequence_length * q_hidden_size;

        int head_size = qkv_head_size[0];
        int weights_offset = 0;
        int bias_offset = head_index * head_size;
        weights_offset = bias_offset;
        int p_offset = (batch_index * num_heads_ + head_index) * (pos_sequence_length * head_size);
        
        T* broadcast_data_dest = P + p_offset;
        for (int seq_index = 0; seq_index < pos_sequence_length; seq_index++) {
          memset(broadcast_data_dest, 0, SafeInt<size_t>(head_size) * sizeof(T));
          broadcast_data_dest += head_size;
        }
        //                   original           transposed            iteration
        // A: input          (BxSxD)            (B.)S x D             S x D
        // B: weights        (DxNxT)             D x (N.)T            D x H
        // C: QKV[qkv_index] (BxNxSxT)          (B.N.)S x T           S x H
        math::GemmEx<float, ThreadPool>(
            CblasNoTrans,                   // TransA = no
            CblasNoTrans,                   // TransB = no
            pos_sequence_length,            // M      = S
            head_size,                      // N      = H
            q_hidden_size,              // K      = D
            1.0f,                           // alpha
            pos_input_data + input_offset,      // A
            q_hidden_size,              // lda    = D
            pos_weights_data + weights_offset,  // B
            q_hidden_size,                  // ldb = NH1
            1.0f,                           // beta
            P + p_offset,                   // C
            head_size,                      // ldc
            nullptr                         // use single-thread
        );
      }
    });
  }
  // Compute the attention score and apply the score to V
  return ApplyRelPosAttention(Q, K, V, P, pos_bias_u, pos_bias_v, mask_index, output,
                              batch_size, sequence_length, pos_sequence_length,
                              qkv_head_size[0], qkv_head_size[2], v_hidden_size, context);
}
}  // namespace contrib
}  // namespace onnxruntime
