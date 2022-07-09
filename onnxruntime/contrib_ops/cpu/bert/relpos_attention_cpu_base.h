// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications by Masao Someki
// Copyright (c) 2022 Masao Someki

#pragma once

#include "relpos_attention_base.h"
#include "attention_helper.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class RelPosAttentionCPUBase : public RelPosAttentionBase {
 protected:
  RelPosAttentionCPUBase(const OpKernelInfo& info) : RelPosAttentionBase(info) {}

  template <typename T>
  Status ApplyRelPosAttention(T* Q,                // Q data. Its size is BxNxSxH
                             const T* K,                // K data. Its size is BxNxSxH
                             const T* V,                // V value with size BxNxSxH
                             const T* P,                // V value with size BxNxSxH
                             const Tensor* mask_index,    // mask index. nullptr if no mask or its size is B
                             const Tensor* pos_bias_u,                // V value with size BxNxSxH
                             const Tensor* pos_bias_v,                // V value with size BxNxSxH
                             Tensor* output,            // output tensor
                             int batch_size,            // batch size
                             int sequence_length,       // sequence length
                             int pos_sequence_length,       // sequence length
                             int qk_head_size,          // qk_head_size
                             int v_head_size,           // head_size
                             int v_hidden_size,         // hidden_size
                             OpKernelContext* context) const {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    auto* tp = context->GetOperatorThreadPool();

    // Compute the attention score. It does 3 things:
    //         I. matrix_ac(B, N, S, S*) = (pos_bias_u + Q(B, N, S, H)) x K'(B, N, S*, H -> B, N, H, S*)
    //         II. matrix_bd(B, N, S, S*) = (pos_bias_v + Q(B, N, S, H)) x P(B, N, S*, H -> B, N, H, S*)
    //         III.attention_probs(B, N, S, S*) = Softmax((matrix_ac + matrix_bd) / sqrt(head))
    size_t matrix_ac_bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * sequence_length * sizeof(T);
    auto matrix_ac = allocator->Alloc(matrix_ac_bytes);
    BufferUniquePtr ac_scratch_buffer(matrix_ac, BufferDeleter(allocator));
    size_t matrix_bd_bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * pos_sequence_length * sizeof(T);
    auto matrix_bd = allocator->Alloc(matrix_bd_bytes);
    BufferUniquePtr bd_scratch_buffer(matrix_bd, BufferDeleter(allocator));

    // I compute matrix_ac
    ComputeAttentionProbs<T>(static_cast<T*>(matrix_ac), Q, K, pos_bias_u->template Data<T>(),
                             batch_size, sequence_length, pos_sequence_length,
                             qk_head_size == 0 ? v_head_size : qk_head_size, tp);
    // II compute matrix_bd
    ComputeAttentionProbs<T>(static_cast<T*>(matrix_bd), Q, P, pos_bias_v->template Data<T>(),
                             batch_size, sequence_length, pos_sequence_length,
                             qk_head_size == 0 ? v_head_size : qk_head_size, tp);
    
    // III Compute RelShift.
    const int32_t* mask_index_data = mask_index != nullptr ? mask_index->template Data<int32_t>() : nullptr;
    gsl::span<const int64_t> mask_index_dims = mask_index != nullptr ? mask_index->Shape().GetDims() : gsl::span<const int64_t>{};
    ComputeRelShift<T>(static_cast<T*>(matrix_ac), static_cast<T*>(matrix_bd),
      mask_index_data, mask_index_dims, 
      batch_size, sequence_length, pos_sequence_length, v_head_size, tp);

    // Compute the attentionScore * Value. It does: out_tmp(B, N, S, H) = attention_probs(B, N, S, S*) x V(B, N, S*, H)
    auto out_tmp_data =
        allocator->Alloc(SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * v_head_size * sizeof(T));
    BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(allocator));

    ComputeVxAttentionScore(output->template MutableData<T>(), static_cast<T*>(out_tmp_data), static_cast<T*>(matrix_ac), V,
                            batch_size, sequence_length, sequence_length, v_head_size, v_hidden_size, tp);

    return Status::OK();
  }

 private:
  // Helper function to compute the attention probs. It does 2 things:
  //  I. attention_probs(B, N, S, S*) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, S*, H -> B, N, H, S*) +
  //                                    1 x mask_data(B, N, S, S*)
  //  II.attention_probs(B, N, S, S*) = Softmax(attention_probs)
  template <typename T>
  void ComputeAttentionProbs(T* attention_probs,                                         // output buffer for the attention probs. Its size is BxNxSxS*
                             T* Q,                                                 // Q data. Its size is BxNxSxH
                             const T* K,                                                 // k data. Its size is BxNxSxH
                             const T* pos_bias,                                                 // k data. Its size is BxNxSxH
                             int batch_size,                                             // batch size of self-attention
                             int sequence_length,                                        // sequence length
                             int pos_sequence_length,                                        // sequence length
                             int head_size,                                              // head size of self-attention
                             ThreadPool* tp                                              // thread pool
  ) const {                                                                              // S* = S' + S
    const size_t input_chunk_length = static_cast<size_t>(sequence_length) * head_size;  // S x H

    {
      memset(attention_probs, 0, static_cast<size_t>(batch_size) * num_heads_ * sequence_length * pos_sequence_length * sizeof(T));

      const int loop_len = batch_size * num_heads_;

      // The cost of Gemm
      const double cost = static_cast<double>(head_size) * sequence_length * pos_sequence_length;
      ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t i = begin; i != end; ++i) {
          const int output_offset = static_cast<int>(i) * sequence_length * pos_sequence_length;
          T* output = attention_probs + output_offset;

          const T* k = K + input_chunk_length * i;

          for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
            for (int j = 0; j < head_size; j++) {
              Q[seq_index * sequence_length + j] += pos_bias[j];
            }
          }
          // Compute Q*K' + AttentionMask
          //                     original                 transposed             each iteration
          // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
          // B: K'               (B x N x) S* x H         (B x N x) H x S*       H x S*
          // C: attention_probs  (B x N x) S x S*         (B x N x) S x S*       S x S*
          math::Gemm<T, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, sequence_length, head_size, 1.0,
                                    Q + input_chunk_length * i, k, 1.0,
                                    output, nullptr);
        }
      });
    }
  }

  template<typename T>
  void ComputeRelShift(T* matrix_ac,                                         // output buffer for the attention probs. Its size is BxNxSxS*
                      const T* matrix_bd,                                                 // Q data. Its size is BxNxSxH
                      const int32_t* mask_index,                    // mask index. nullptr if no mask or its size is B
                      gsl::span<const int64_t> mask_index_dims,     // mask index shape
                      int batch_size,                                             // batch size of self-attention
                      int sequence_length,                                        // sequence length
                      int pos_sequence_length,                                        // sequence length
                      int head_size,                                              // head size of self-attention
                      ThreadPool* tp                                              // thread pool)
  ) const {
    const int diff_pos_length = pos_sequence_length - sequence_length;
    {

      const int loop_len = batch_size * num_heads_;
      const float alpha = 1.0f / sqrt(static_cast<float>(head_size));
      
      // The cost of Gemm
      const double cost = static_cast<double>(head_size) * sequence_length * sequence_length;
      ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t i = begin; i != end; ++i) {
          for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
            for (int j = 0; j < head_size; j++) {
              int iac = seq_index * sequence_length + j;
              int ibd = seq_index * pos_sequence_length + (diff_pos_length - seq_index) * head_size + j;
              matrix_ac[iac] += matrix_bd[ibd];
              matrix_ac[iac] /= alpha;
            }
          }
        }
      });
    }
    //  attention_probs(B, N, S, S*) = Softmax(attention_probs)
    {
      if (mask_index != nullptr) {
        ApplyMask(mask_index, mask_index_dims, matrix_ac, batch_size, sequence_length);
      }
      const int N = batch_size * num_heads_ * sequence_length;
      const int D = sequence_length;
      ComputeAttentionSoftmaxInplace(matrix_ac, N, D, tp);
    }
  }

  template <typename T>
  void ApplyMask(const int32_t* mask_index,
                  gsl::span<const int64_t> mask_index_dims,
                  T* p_mask,
                  int batch_size,
                  int sequence_length) const {

    // For 3D mask, convert values 0 to -10000.0, and 1 to 0.0, then apply unidirectional mask if any.
    if (nullptr != mask_index && mask_index_dims.size() == 3) {
      for (int i = 0; i < batch_size * sequence_length * sequence_length; i++) {
        p_mask[i] += (mask_index[i] > 0) ? static_cast<T>(0.0f) : static_cast<T>(-10000.0f);
      }
      return;
    }

    bool is_raw_attention_mask = (nullptr != mask_index && mask_index_dims.size() == 2);
    bool has_mask_start_position = (nullptr != mask_index && mask_index_dims.size() == 1 && static_cast<int>(mask_index_dims.at(0)) == 2 * batch_size);

    for (int b_i = 0; b_i < batch_size; b_i++) {
      // TODO: mask_index can be used in softmax to save some calculation.
      if (nullptr != mask_index) {
        if (is_raw_attention_mask) {
          // Raw attention mask has value 0 or 1. Here we convert 0 to -10000.0, and 1 to 0.0.
          const int32_t* raw_mask = mask_index + b_i * sequence_length;
          for (int m_i = 0; m_i < sequence_length; m_i++) {
            p_mask[m_i] += (raw_mask[m_i] > 0) ? static_cast<T>(0.0f) : static_cast<T>(-10000.0f);
          }
        } else {
          // mask_index is 1D: (B) or (2B) => (Bx)S*

          // Handle right-side padding: mask value at or after the end position will be -10000.0
          int end_position = mask_index[b_i];
          for (int m_i = end_position; m_i < sequence_length; m_i++) {
            p_mask[m_i] += static_cast<T>(-10000.0f);
          }

          // Handle left-side padding: mask value before the start position will be -10000.0
          if (has_mask_start_position) {
            int start_position = std::min(mask_index[b_i + batch_size], sequence_length);
            for (int m_i = 0; m_i < start_position; m_i++) {
              p_mask[m_i] += static_cast<T>(-10000.0f);
            }
          }
        }
      }

      // Broadcast mask from (Bx)S* to (Bx)SxS*
      for (int s_i = 1; s_i < sequence_length; s_i++) {
        memcpy(p_mask + s_i * sequence_length, p_mask, sequence_length * sizeof(T));
      }

      p_mask += sequence_length * sequence_length;
    }
  }

  template <typename T>
  void ComputeVxAttentionScore(T* output,                                                // buffer for the result with size BxSxNxH
                               T* tmp_buffer,                                            // buffer for temp use with size is BxNxSxH
                               const T* attention_probs,                                 // Attention probs with size BxNxSxS*
                               const T* V,                                               // V value with size BxNxSxH
                               int batch_size,                                           // batch size
                               int sequence_length,                                      // sequence length
                               int kv_sequence_length,                                      // sequence length
                               int head_size,                                            // head size
                               int hidden_size,                                          // hidden size
                               ThreadPool* tp) const {                                   // S* = S' + S
    const size_t input_chunk_length = static_cast<size_t>(sequence_length * head_size);  // S x H

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(sequence_length);

    ThreadPool::TryParallelFor(tp, batch_size * num_heads_, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const T* v = V + input_chunk_length * i;

        T* current_tmp_data = reinterpret_cast<T*>(tmp_buffer) + input_chunk_length * i;
        math::MatMul<T>(sequence_length, head_size, sequence_length,
                        attention_probs + sequence_length * sequence_length * i,
                        v, current_tmp_data, nullptr);

        // transpose: out(B, S, N, H) = transpose out_tmp(B, N, S, H)
        const int batch_index = static_cast<int>(i / num_heads_);
        const int head_index = static_cast<int>(i % num_heads_);
        T* src = current_tmp_data;
        T* dest = output + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
        const auto bytes_to_copy = SafeInt<size_t>(head_size) * sizeof(T);
        for (int j = 0; j < sequence_length; j++) {
          memcpy(dest, src, bytes_to_copy);
          src += head_size;
          dest += hidden_size;
        }
      }
    });
  }
};

}  // namespace contrib
}  // namespace onnxruntime
