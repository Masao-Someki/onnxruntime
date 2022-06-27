// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications by Masao Someki
// Copyright (c) 2022 Masao Someki

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class CrossAttentionBase {
 public:
  Status CheckInputs(const TensorShape& query_shape,
                      const TensorShape& key_shape,
                      const TensorShape& q_weights_shape,
                      const TensorShape& kv_weights_shape,
                      const TensorShape& q_bias_shape,
                      const TensorShape& kv_bias_shape,
                     const Tensor*& mask_index,
                      const int max_threads_per_block) const;

 protected:
   CrossAttentionBase(const OpKernelInfo& info) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    if (!info.GetAttrs<int64_t>("qkv_hidden_sizes", qkv_hidden_sizes_).IsOK() || qkv_hidden_sizes_.empty()) {
      qkv_hidden_sizes_.resize(0);
    }
  }

  Status CheckInputs(const TensorShape& query_shape,
                      const TensorShape& key_shape,
                      const TensorShape& q_weights_shape,
                      const TensorShape& kv_weights_shape,
                      const TensorShape& q_bias_shape,
                      const TensorShape& kv_bias_shape,
                     const Tensor*& mask_index) const;

  int num_heads_;
  std::vector<int64_t> qkv_hidden_sizes_;   // Q, K, V path hidden layer sizes
};

}  // namespace contrib
}  // namespace onnxruntime
