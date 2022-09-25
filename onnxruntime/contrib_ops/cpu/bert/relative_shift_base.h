// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications by Masao Someki
// Copyright (c) 2022 Masao Someki

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class RelativeShiftBase {
 public:
  // Status CheckInputs(const TensorShape& input_shape,
  //                     const TensorShape& weights_shape,
  //                     const TensorShape& bias_shape,
  //                     const TensorShape& pos_emb_shape,
  //                     const TensorShape& pos_bias_u_shape,
  //                     const TensorShape& pos_bias_v_shape,
  //                     const Tensor*& mask_index,
  //                     const int max_threads_per_block) const;

 protected:
   RelativeShiftBase(const OpKernelInfo& info) {
    is_legacy_ = info.GetAttrOrDefault<int64_t>("legacy", 0) == 1;
  }

  Status CheckInputs(const TensorShape& input_shape,
                      const TensorShape& pos_bias_shape) const;

  bool is_legacy_;
};

}  // namespace contrib
}  // namespace onnxruntime
