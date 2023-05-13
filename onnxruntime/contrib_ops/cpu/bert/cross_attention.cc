// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications by Masao Someki
// Copyright (c) 2022 Masao Someki

#include "attention_helper.h"

#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "contrib_ops/cpu/bert/cross_attention_cpu_base.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename T>
class CrossAttention : public OpKernel, public CrossAttentionCPUBase {
 public:
  explicit CrossAttention(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    CrossAttention,
    kENDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    CrossAttention<float>);

Status CrossAttentionBase::CheckInputs(const TensorShape& query_shape,
                                  const TensorShape& key_shape,
                                  const TensorShape& q_weights_shape,
                                  const TensorShape& kv_weights_shape,
                                  const TensorShape& q_bias_shape,
                                  const TensorShape& kv_bias_shape,
                                  const Tensor*& mask_index) const {

  const auto& query_shape_dims = query_shape.GetDims();
  if (query_shape_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'query' is expected to have 3 dimensions, got ",
                           query_shape_dims.size());
  }

  int hidden_size = static_cast<int>(query_shape_dims[2]);

  const auto& key_shape_dims = key_shape.GetDims();
  if (key_shape_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 3 dimensions, got ",
                           key_shape_dims.size());
  }

  if (query_shape_dims[0] != key_shape_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "query and key shall have the same batch size");
  }

  if (query_shape_dims[2] != key_shape_dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "query and key shall have the same hidden size");
  }

  const auto& q_weights_dims = q_weights_shape.GetDims();
  if (q_weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'q_weights' is expected to have 2 dimensions, got ",
                           q_weights_dims.size());
  }

  const auto& kv_weights_dims = kv_weights_shape.GetDims();
  if (kv_weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'kv_weights' is expected to have 2 dimensions, got ",
                           kv_weights_dims.size());
  }

  if (q_weights_dims[0] != hidden_size || q_weights_dims[1] != hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "q_weights shall have shape (hidden size, hidden size)");
  }

  if (kv_weights_dims[0] != hidden_size || kv_weights_dims[1] != 2 * hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "kv_weights shall have shape (hidden size, 2 * hidden size)");
  }

  const auto& q_bias_dims = q_bias_shape.GetDims();
  if (q_bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'q_bias' is expected to have 1 dimension, got ",
                           q_bias_dims.size());
  }
  if (q_bias_dims[0] != hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "q_bias shall have shape (hidden size)");
  }

  const auto& kv_bias_dims = kv_bias_shape.GetDims();
  if (kv_bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'kv_bias' is expected to have 1 dimension, got ",
                           kv_bias_dims.size());
  }

  if (kv_bias_dims[0] != 2 * hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "kv_bias shall have shape (2 * hidden size)");
  }

  int batch_size = static_cast<int>(query_shape_dims[0]);
  int kv_batch_size = static_cast<int>(key_shape_dims[0]);
  int sequence_length = static_cast<int>(query_shape_dims[1]);
  int kv_sequence_length = static_cast<int>(key_shape_dims[1]);

  if (batch_size != kv_batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "batch_size of key must be the same or larger than batch_size of query.");
  }

  if (kv_sequence_length < sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "sequence length of key must be the same or larger than sequence length of query.");
  }

  if (mask_index != nullptr) {  // mask_index is optional
    const auto& mask_dims = mask_index->Shape().GetDims();
    if (mask_dims.size() == 1) {
      if (static_cast<int>(mask_dims[0]) != batch_size && static_cast<int>(mask_dims[0]) != 2 * batch_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mask_index' with 1D data shall have length of batch_size or 2 * batch_size", batch_size);
      }
    } else if (mask_dims.size() == 2) {
      if (static_cast<int>(mask_dims[0]) != batch_size || static_cast<int>(mask_dims[1]) != kv_sequence_length) {
        // Add operator supports broadcasting. Here we handle a case with only one element in the 2nd dimension.
        if ((static_cast<int>(mask_dims[0]) == batch_size || static_cast<int>(mask_dims[0]) == 1) && static_cast<int>(mask_dims[1]) == 1) {
          // Mask will have same value after propogation, which has same effect as no mask.
          mask_index = nullptr;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mask_index' shall have shape batch_size x (past_sequence_length + sequence_length)");
        }
      }
    } else if (mask_dims.size() == 3) {
      if (static_cast<int>(mask_dims[0]) != batch_size || mask_dims[1] != sequence_length || static_cast<int>(mask_dims[2]) != kv_sequence_length) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mask_index' with 3D data shall have shape batch_size x sequence_length x (past_sequence_length + sequence_length)");
      }
    } else if (mask_dims.size() == 4) {
      if (static_cast<int>(mask_dims[0]) != batch_size || mask_dims[1] != 1 || mask_dims[2] != mask_dims[3] || mask_dims[2] < kv_sequence_length) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mask_index' with 4D data shall have shape batch_size x 1 x max_sequence_length x max_sequence_length)");
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'mask_index' is expected to have 1, 2, 3 or 4 dimensions, got ",
                             mask_dims.size());
    }
  }

  return Status::OK();
}

Status CrossAttentionBase::CheckInputs(const TensorShape& query_shape,
                                  const TensorShape& key_shape,
                                  const TensorShape& q_weights_shape,
                                  const TensorShape& kv_weights_shape,
                                  const TensorShape& q_bias_shape,
                                  const TensorShape& kv_bias_shape,
                                  const Tensor*& mask_index,
                                  const int max_threads_per_block) const {
  if (num_heads_ > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(query_shape, key_shape, q_weights_shape, kv_weights_shape, q_bias_shape, kv_bias_shape, mask_index);
}

template <typename T>
CrossAttention<T>::CrossAttention(const OpKernelInfo& info) : OpKernel(info), CrossAttentionCPUBase(info) {
}

template <typename T>
Status CrossAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* q_weights = context->Input<Tensor>(2);
  const Tensor* kv_weights = context->Input<Tensor>(3);
  const Tensor* q_bias = context->Input<Tensor>(4);
  const Tensor* kv_bias = context->Input<Tensor>(5);
  const Tensor* mask_index = context->Input<Tensor>(6);

  ORT_RETURN_IF_ERROR(
    CheckInputs(query->Shape(),
                key->Shape(),
                q_weights->Shape(),
                kv_weights->Shape(),
                q_bias->Shape(),
                kv_bias->Shape(),
                mask_index
    )
  );

  // query shape (batch_size, sequence_length, input_hidden_size)
  const auto& query_shape = query->Shape();
  int batch_size = static_cast<int>(query_shape[0]);
  int sequence_length = static_cast<int>(query_shape[1]);
  int hidden_size = static_cast<int>(query_shape[2]);
  const auto& key_shape = key->Shape();
  int key_sequence_length = static_cast<int>(key_shape[1]);
  int head_size = hidden_size / num_heads_;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = query_shape[0];
  output_shape[1] = query_shape[1];
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
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * (sequence_length * q_hidden_size + key_sequence_length * (k_hidden_size + v_hidden_size)) * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + static_cast<size_t>(batch_size) * sequence_length * q_hidden_size;
  auto V = K + static_cast<size_t>(batch_size) * key_sequence_length * k_hidden_size;

  T* KV[2] = {K, V};

  {
    // compute query
    const int loop_len = batch_size * num_heads_;
    const auto* input_data = query->template Data<T>();
    const auto* weights_data = q_weights->template Data<T>();
    const auto* q_bias_data = q_bias->template Data<T>();

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(q_hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>(i / num_heads_);
        const int head_index = static_cast<int>(i % num_heads_);
        // const int qkv_index = static_cast<int>(i);

        int input_offset = batch_index * sequence_length * q_hidden_size;

        // T* qkv_dest = QKV[qkv_index];
        int head_size = qkv_head_size[0]; // head size of query
        int weights_offset = 0;
        int bias_offset = head_index * head_size; // this loop is for query.

        weights_offset = bias_offset;

        int q_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size);

        // TODO!! memcpy here makes it not worthwhile to use Gemm batch. Possible to post process?
        // broadcast NH -> (B.N.S.H) for each of Q, K, V
        const T* broadcast_data_src = q_bias_data + bias_offset;
        T* broadcast_data_dest = Q + q_offset;

        for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
          memcpy(broadcast_data_dest, broadcast_data_src, head_size * sizeof(T));
          broadcast_data_dest += head_size;
        }

        math::GemmEx<float, ThreadPool>(
          CblasNoTrans, CblasNoTrans,
            sequence_length, head_size, q_hidden_size,
            1.0f,
            input_data + input_offset, q_hidden_size,
            weights_data + weights_offset, q_hidden_size,
            1.0f,
            Q + q_offset, head_size,
            nullptr
        );
      }
    });
  }

  {
    // compute key
    const int loop_len = 2 * batch_size * num_heads_;
    const auto* input_data = key->template Data<T>();
    const auto* weights_data = kv_weights->template Data<T>();
    const auto* kv_bias_data = kv_bias->template Data<T>();

    const double cost =
        static_cast<double>(key_sequence_length) * static_cast<double>(head_size) * static_cast<double>(k_hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 2) / num_heads_);
        const int head_index = static_cast<int>((i / 2) % num_heads_);
        const int kv_index = static_cast<int>(i % 2);

        int input_offset = batch_index * key_sequence_length * k_hidden_size;

        T* kv_dest = KV[kv_index];
        int head_size = qkv_head_size[1 + kv_index]; // head size of key
        int weights_offset = 0;
        int bias_offset = kv_index * k_hidden_size + head_index * head_size; // this loop is for key.

        weights_offset = bias_offset;

        int kv_offset = (batch_index * num_heads_ + head_index) * (key_sequence_length * head_size);

        // TODO!! memcpy here makes it not worthwhile to use Gemm batch. Possible to post process?
        // broadcast NH -> (B.N.S.H) for each of Q, K, V
        const T* broadcast_data_src = kv_bias_data + bias_offset;
        T* broadcast_data_dest = KV[kv_index] + kv_offset;

        for (int seq_index = 0; seq_index < key_sequence_length; seq_index++) {
          memcpy(broadcast_data_dest, broadcast_data_src, head_size * sizeof(T));
          broadcast_data_dest += head_size;
        }

        math::GemmEx<float, ThreadPool>(
          CblasNoTrans, CblasNoTrans,
            key_sequence_length, head_size, k_hidden_size,
            1.0f,
            input_data + input_offset, k_hidden_size,
            weights_data + weights_offset, k_hidden_size + v_hidden_size,
            1.0f,
            kv_dest + kv_offset, head_size,
            nullptr
        );
      }
    });
  }
  // Compute the attention score and apply the score to V
  return ApplyCrossAttention(Q, K, V, mask_index, output,
                        batch_size, sequence_length, key_sequence_length,
                        qkv_head_size[0], qkv_head_size[2], v_hidden_size,
                        context);
}
}  // namespace contrib
}  // namespace onnxruntime
