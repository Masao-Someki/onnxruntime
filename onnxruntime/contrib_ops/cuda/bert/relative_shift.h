// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modification by Masao Someki

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class RelativeShift final : public CudaKernel {
 public:
  RelativeShift(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime