// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cublas_v2.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool LaunchRelPosBiasAdd(
    cudaStream_t stream,
    const int sequence_length,
    const int batch_size,
    const int head_size,
    const int num_heads,
    const float* pos_bias,
    float* query);

bool LaunchRelPosBiasAdd(
    cudaStream_t stream,
    const int sequence_length,
    const int batch_size,
    const int head_size,
    const int num_heads,
    const half* pos_bias,
    half* query);

bool LaunchRelShiftAdd(
    cudaStream_t stream,
    const int sequence_length,
    const int pos_sequence_length,
    const int batch_size,
    const int num_heads,
    float* matrix_ac,
    float* matrix_bd,
    float alpha);

bool LaunchRelShiftAdd(
    cudaStream_t stream,
    const int sequence_length,
    const int pos_sequence_length,
    const int batch_size,
    const int num_heads,
    half* matrix_ac,
    half* matrix_bd,
    float alpha);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
