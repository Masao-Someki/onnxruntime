// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications by Masao Someki
// Copyright (c) 2022 Masao Someki

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class RelPosAttentionBase {
 public:
  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const TensorShape& pos_emb_shape,
                     const TensorShape& pos_bias_u_shape,
                     const TensorShape& pos_bias_v_shape,
                     const Tensor*& mask_index,
                     const int max_threads_per_block) const;

 protected:
  RelPosAttentionBase(const OpKernelInfo& info) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    is_legacy_ = info.GetAttrOrDefault<int64_t>("legacy", 0) == 1;

    if (!info.GetAttrs<int64_t>("qkv_hidden_sizes", qkv_hidden_sizes_).IsOK() || qkv_hidden_sizes_.empty()) {
      qkv_hidden_sizes_.resize(0);
    }
  }

  // Status CheckInputs(const TensorShape& input_shape,
  //                    const TensorShape& weights_shape,
  //                    const TensorShape& bias_shape,
  //                    const TensorShape& pos_emb_shape,
  //                    const TensorShape& pos_bias_u_shape,
  //                    const TensorShape& pos_bias_v_shape,
  //                    const Tensor*& mask_index) const;

  int num_heads_;
  bool is_legacy_;
  std::vector<int64_t> qkv_hidden_sizes_;  // Q, K, V path hidden layer sizes
};

}  // namespace contrib
}  // namespace onnxruntime