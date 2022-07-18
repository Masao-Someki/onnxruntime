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

namespace onnxruntime
{
  namespace contrib
  {

    class RelPosAttentionCPUBase : public RelPosAttentionBase
    {
    protected:
      RelPosAttentionCPUBase(const OpKernelInfo &info) : RelPosAttentionBase(info) {}

      template <typename T>
      Status ApplyRelPosAttention(const T* Q,                     // Q data. Its size is BxNxSxH
                                  const T* K,               // K data. Its size is BxNxSxH
                                  const T* V,               // V value with size BxNxSxH
                                  const T* P,               // V value with size BxNxSxH
                                  const Tensor* pos_bias_u, // V value with size BxNxSxH
                                  const Tensor* pos_bias_v, // V value with size BxNxSxH
                                  const Tensor* mask_index,    // mask index. nullptr if no mask or its size is B
                                  Tensor* output,           // output tensor
                                  int batch_size,           // batch size
                                  int sequence_length,      // sequence length
                                  int pos_sequence_length,  // sequence length
                                  int qk_head_size,         // qk_head_size
                                  int v_head_size,          // head_size
                                  int v_hidden_size,        // hidden_size
                                  OpKernelContext* context) const
      {
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

        size_t temp_buffer_bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * qk_head_size * sizeof(T);
        auto temp_buffer_mat = allocator->Alloc(temp_buffer_bytes);
        BufferUniquePtr temp_mat_buffer(temp_buffer_mat, BufferDeleter(allocator));

        // I compute matrix_ac
        const auto* pos_bias_u_data = pos_bias_u->template Data<T>();
        ComputeAttentionProbs<T>(static_cast<T*>(matrix_ac), Q, K, pos_bias_u_data, static_cast<T*>(temp_buffer_mat), 
                                 batch_size, sequence_length, sequence_length,
                                 qk_head_size, tp);
        // II compute matrix_bd
        const auto* pos_bias_v_data = pos_bias_v->template Data<T>();
        ComputeAttentionProbs<T>(static_cast<T*>(matrix_bd), Q, P, pos_bias_v_data, static_cast<T*>(temp_buffer_mat), 
                                 batch_size, sequence_length, pos_sequence_length,
                                 qk_head_size, tp);

        // III Compute RelShift.
        void* mask_data = nullptr;
        if (mask_index != nullptr) {
          size_t mask_data_bytes = SafeInt<size_t>(batch_size) * sequence_length * pos_sequence_length * sizeof(T);
          mask_data = allocator->Alloc(mask_data_bytes);
          memset(mask_data, 0, mask_data_bytes);
        }
        BufferUniquePtr mask_data_buffer(mask_data, BufferDeleter(allocator));

        const int32_t* mask_index_data = mask_index != nullptr ? mask_index->template Data<int32_t>() : nullptr;
        gsl::span<const int64_t> mask_index_dims = mask_index != nullptr ? mask_index->Shape().GetDims() : gsl::span<const int64_t>{};

        if (is_legacy_) {
          ComputeLegacyRelShift<T>(static_cast<T*>(matrix_ac), static_cast<T*>(matrix_bd),
                            mask_index_data, mask_index_dims, static_cast<T*>(mask_data),
                            batch_size, sequence_length, pos_sequence_length, v_head_size, tp);
        } else {
          ComputeRelShift<T>(static_cast<T*>(matrix_ac), static_cast<T*>(matrix_bd),
                            mask_index_data, mask_index_dims, static_cast<T*>(mask_data),
                            batch_size, sequence_length, pos_sequence_length, v_head_size, tp);
        }

        // Compute the attentionScore * Value. It does: out_tmp(B, N, S, H) = attention_probs(B, N, S, S*) x V(B, N, S*, H)
        auto out_tmp_data =
            allocator->Alloc(SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * v_head_size * sizeof(T));
        BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(allocator));

        ComputeVxAttentionScore(output->template MutableData<T>(), static_cast<T *>(out_tmp_data), static_cast<T *>(matrix_ac), V,
                                batch_size, sequence_length, v_head_size, v_hidden_size, tp);

        return Status::OK();
      }

    private:
      // Helper function to compute the attention probs. It does 2 things:
      //  I. attention_probs(B, N, S, S*) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, S*, H -> B, N, H, S*) +
      //                                    1 x mask_data(B, N, S, S*)
      //  II.attention_probs(B, N, S, S*) = Softmax(attention_probs)
      template <typename T>
      void ComputeAttentionProbs(T* attention_probs,      // output buffer for the attention probs. Its size is BxNxSxS*
                                 const T* A,                    // Q data. Its size is BxNxSxH
                                 const T* B,              // k data. Its size is BxNxSxH
                                 const T* pos_bias,       // k data. Its size is BxNxSxH
                                 T* temp_buffer,       // k data. Its size is BxNxSxH
                                 int batch_size,          // batch size of self-attention
                                 int sequence_length,     // sequence length
                                 int pos_sequence_length, // sequence length
                                 int head_size,           // head size of self-attention
                                 ThreadPool* tp           // thread pool
      ) const
      {                                                                                           // S* = S' + S
        const size_t a_input_chunk_length = static_cast<size_t>(sequence_length) * head_size;     // S x H
        const size_t b_input_chunk_length = static_cast<size_t>(pos_sequence_length) * head_size; // S x H

        {
          memset(attention_probs, 0, SafeInt<size_t>(batch_size)*  num_heads_ * sequence_length * pos_sequence_length * sizeof(T));
          memset(temp_buffer, 0, SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * head_size * sizeof(T));
          const int loop_len = batch_size * num_heads_;

          // The cost of Gemm
          const double cost = static_cast<double>(head_size) * sequence_length * pos_sequence_length;
          ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end)
                                     {
        for (std::ptrdiff_t i = begin; i != end; ++i) {
          // const int batch_index = static_cast<int>(i / num_heads_);
          const int head_index = static_cast<int>(i % num_heads_);
          const int output_offset = static_cast<int>(i) * sequence_length * pos_sequence_length;
          T* output = attention_probs + output_offset;
          T* workspace = reinterpret_cast<T*>(temp_buffer) + a_input_chunk_length * i;

          const T* q = A + a_input_chunk_length * i;
          const T* k = B + b_input_chunk_length * i;
          // memset(workspace, q, SafeInt<size_t>(sequence_length) * head_size * sizeof(T));

          const size_t pbid = static_cast<size_t>(head_index * head_size);
          for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
            const size_t qid = static_cast<size_t>(seq_index * head_size);
            // memcpy(workspace + qid, pos_bias + pbid, SafeInt<size_t>(head_size) * sizeof(T));
            for (int j = 0; j < head_size; j++) {
              // const int column_height = j * sequence_length + seq_index;
              workspace[qid + j] = q[qid + j] + pos_bias[pbid + j];
            }
          }
          // Compute Q*K' + AttentionMask
          //                     original                 transposed             each iteration
          // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
          // B: K'               (B x N x) S* x H         (B x N x) H x S*       H x S*
          // C: attention_probs  (B x N x) S x S*         (B x N x) S x S*       S x S*
          math::Gemm<T, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, pos_sequence_length, head_size, 1.0,
                                    workspace, k, 1.0, output, nullptr);
        } });
        }
      }

      template <typename T>
      void ComputeLegacyRelShift(
                           T* matrix_ac,                             // output buffer for the attention probs. Its size is BxNxSxS*
                           T* matrix_bd,                       // Q data. Its size is BxNxSxH
                           const int32_t* mask_index,                    // mask index. nullptr if no mask or its size is B
                           gsl::span<const int64_t> mask_index_dims,     // mask index shape
                           T* mask_data,                                 // buffer for mask data. It is nullptr if mask_index is nullptr and not unidirectional, otherwise its shape is BxSxS*
                           int batch_size,                           // batch size of self-attention
                           int sequence_length,                      // sequence length
                           int pos_sequence_length,                  // sequence length
                           int head_size,                            // head size of self-attention
                           ThreadPool* tp                            // thread pool)
      ) const
      {
        if (mask_data != nullptr) {
          PrepareMask(mask_index, mask_index_dims, mask_data, false, batch_size, sequence_length, 0);
        }
        {
          const int loop_len = batch_size * num_heads_;
          const float alpha = 1.0f / sqrt(static_cast<float>(head_size));

          // The cost of Gemm
          const double cost = static_cast<double>(sequence_length) * sequence_length;
          ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end){
            for (std::ptrdiff_t i = begin; i != end; ++i) {
              const int output_offset = static_cast<int>(i) * sequence_length * sequence_length;
              const int batch_index = static_cast<int>(i) / num_heads_;
              const int mask_offset = batch_index * sequence_length * sequence_length;
              int bd_batch_offset = 0;
              int bd_sequence_offset = sequence_length - 1;

              for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
                const int oidx = static_cast<int>(seq_index) * sequence_length + output_offset;
                const int midx = static_cast<int>(seq_index) * sequence_length + mask_offset;
                for (int j = 0; j < sequence_length; j++) {
                  if (bd_sequence_offset == sequence_length) {
                    matrix_ac[oidx + j] *= alpha;
                    bd_batch_offset += 1;
                    bd_sequence_offset = 0;
                  } else {
                    const int bid = output_offset + bd_batch_offset * sequence_length + bd_sequence_offset;
                    matrix_ac[oidx + j] += matrix_bd[bid];
                    matrix_ac[oidx + j] *= alpha;
                    bd_sequence_offset += 1;
                  }
                  if (mask_data != nullptr) {
                    matrix_ac[oidx + j] += mask_data[midx + j];
                  }
                }
              }
            }
          });
        }
        //  attention_probs(B, N, S, S*) = Softmax(attention_probs)
        {
          const int N = batch_size * num_heads_ * sequence_length;
          const int D = sequence_length;
          ComputeAttentionSoftmaxInplace(matrix_ac, N, D, tp);
        }
      }

      template <typename T>
      void ComputeRelShift(
                           T* matrix_ac,                             // output buffer for the attention probs. Its size is BxNxSxS*
                           T* matrix_bd,                       // Q data. Its size is BxNxSxH
                           const int32_t* mask_index,                    // mask index. nullptr if no mask or its size is B
                           gsl::span<const int64_t> mask_index_dims,     // mask index shape
                           T* mask_data,                                 // buffer for mask data. It is nullptr if mask_index is nullptr and not unidirectional, otherwise its shape is BxSxS*
                           int batch_size,                           // batch size of self-attention
                           int sequence_length,                      // sequence length
                           int pos_sequence_length,                  // sequence length
                           int head_size,                            // head size of self-attention
                           ThreadPool* tp                            // thread pool)
      ) const
      {
        if (mask_data != nullptr) {
          PrepareMask(mask_index, mask_index_dims, mask_data, false, batch_size, sequence_length, 0);
        }
        const int diff_pos_length = sequence_length - 1; // 2 * seq_len - 1 - seq_len = seq_len - 1
        {

          const int loop_len = batch_size * num_heads_;
          const float alpha = 1.0f / sqrt(static_cast<float>(head_size));

          // The cost of Gemm
          const double cost = static_cast<double>(sequence_length) * sequence_length;
          ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end){
            for (std::ptrdiff_t i = begin; i != end; ++i) {
              const int output_offset = static_cast<int>(i) * sequence_length * sequence_length;
              const int batch_index = static_cast<int>(i) / num_heads_;
              const int mask_offset = batch_index * sequence_length * sequence_length;
              int bd_input_offset = sequence_length * pos_sequence_length * static_cast<int>(i); 

              for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
                const int oidx = static_cast<int>(seq_index) * sequence_length + output_offset;
                const int midx = static_cast<int>(seq_index) * sequence_length + mask_offset;
                const int ibd = static_cast<int>(bd_input_offset) + seq_index * pos_sequence_length + (diff_pos_length - seq_index);
                for (int j = 0; j < sequence_length; j++) {
                  matrix_ac[oidx + j] += matrix_bd[ibd + j];
                  matrix_ac[oidx + j] *= alpha;
                  if (mask_data != nullptr) {
                    matrix_ac[oidx + j] += mask_data[midx + j];
                  }
                }
              }
            }
          });
        }
        //  attention_probs(B, N, S, S*) = Softmax(attention_probs)
        {
          const int N = batch_size * num_heads_ * sequence_length;
          const int D = sequence_length;
          ComputeAttentionSoftmaxInplace(matrix_ac, N, D, tp);
        }
      }

      template <typename T>
      void ComputeVxAttentionScore(T* output,                // buffer for the result with size BxSxNxH
                                   T* tmp_buffer,            // buffer for temp use with size is BxNxSxH
                                   const T* attention_probs, // Attention probs with size BxNxSxS*
                                   const T* V,               // V value with size BxNxSxH
                                   int batch_size,           // batch size
                                   int sequence_length,      // sequence length
                                   int head_size,            // head size
                                   int hidden_size,          // hidden size
                                   ThreadPool* tp) const
      {                                                                                     // S* = S' + S
        const size_t input_chunk_length = static_cast<size_t>(sequence_length * head_size); // S x H

        const double cost =
            static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(sequence_length);

        ThreadPool::TryParallelFor(tp, batch_size * num_heads_, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end)
                                   {
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
      } });
      }
    };

  } // namespace contrib
} // namespace onnxruntime
