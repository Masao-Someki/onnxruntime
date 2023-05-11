// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cublas_v2.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool LaunchRelShiftAdd(
    cudaStream_t stream,
    const int sequence_length,
    const int pos_sequence_length,
    const int batch_size,
    const int num_heads,
    const float* matrix_ac,
    const float* matrix_bd,
    float* output);

bool LaunchRelShiftAdd(
    cudaStream_t stream,
    const int sequence_length,
    const int pos_sequence_length,
    const int batch_size,
    const int num_heads,
    const half* matrix_ac,
    const half* matrix_bd,
    half* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime