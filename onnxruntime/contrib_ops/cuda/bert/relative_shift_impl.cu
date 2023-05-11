/*
 The implementation of this file is based on qkvToContext plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/
Copyright 2019 NVIDIA Corporation
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
/* Modifications by Masao Someki. */
#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"
// #include "attention_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {


template <typename T>
__global__ void LegacyRelShiftAddFloat2(const int pos_sequence_length, const T* matrix_ac, const T* matrix_bd, T* output) {
  // matrix_bd: BxNxPSxS
  // matrix_ac: BxNxSxS

  int head_idx = threadIdx.y;
  int sequence_idx = blockIdx.x;
  int batch_idx = blockIdx.y;

  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int SS = sequence_length * sequence_length;
  const int NSS = SS * num_heads;

  const int ac_offset = sequence_idx * sequence_length + head_idx * SS + batch_idx * NSS;
  int bd_offset = 0;
  bd_offset = head_idx * SS + batch_idx * NSS;

  const int i = threadIdx.x;
  if (i < sequence_length) {
    int bd_idx = (sequence_length - 1) + (sequence_idx * sequence_length) - sequence_idx;
    if (i < sequence_idx + 1) {
      output[ac_offset + i].x = matrix_ac[ac_offset + i].x + matrix_bd[bd_offset + bd_idx + i].x;
      output[ac_offset + i].y = matrix_ac[ac_offset + i].y + matrix_bd[bd_offset + bd_idx + i].y;
    } else if (i > sequence_length + 1) {
      output[ac_offset + i].x = matrix_ac[ac_offset + i].x + matrix_bd[bd_offset + bd_idx + i - 1].x;
      output[ac_offset + i].y = matrix_ac[ac_offset + i].y + matrix_bd[bd_offset + bd_idx + i - 1].y;
    }
  }
}

template <typename T>
__global__ void LegacyRelShiftAddHalf2(const int pos_sequence_length, const T* matrix_ac, const T* matrix_bd, T* output) {
  // matrix_bd: BxNxPSxS
  // matrix_ac: BxNxSxS

  int head_idx = threadIdx.y;
  int sequence_idx = blockIdx.x;
  int batch_idx = blockIdx.y;

  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int SS = sequence_length * sequence_length;
  const int NSS = SS * num_heads;

  const int ac_offset = sequence_idx * sequence_length + head_idx * SS + batch_idx * NSS;
  int bd_offset = 0;
  bd_offset = head_idx * SS + batch_idx * NSS;

  const int i = threadIdx.x;
  if (i < sequence_length) {
    int bd_idx = (sequence_length - 1) + (sequence_idx * sequence_length) - sequence_idx;
    if (i < sequence_idx + 1) {
      output[ac_offset + i].x = matrix_ac[ac_offset + i].x + matrix_bd[bd_offset + bd_idx + i].x;
      output[ac_offset + i].y = matrix_ac[ac_offset + i].y + matrix_bd[bd_offset + bd_idx + i].y;
    } else if (i > sequence_length + 1) {
      output[ac_offset + i].x = matrix_ac[ac_offset + i].x + matrix_bd[bd_offset + bd_idx + i - 1].x;
      output[ac_offset + i].y = matrix_ac[ac_offset + i].y + matrix_bd[bd_offset + bd_idx + i - 1].y;
    }
  }
}

template <typename T>
__global__ void LegacyRelShiftAddFloat(const int pos_sequence_length, const T* matrix_ac, const T* matrix_bd, T* output) {
  // matrix_bd: BxNxPSxS
  // matrix_ac: BxNxSxS

  int head_idx = threadIdx.y;
  int sequence_idx = blockIdx.x;
  int batch_idx = blockIdx.y;

  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int SS = sequence_length * sequence_length;
  const int NSS = SS * num_heads;

  const int ac_offset = sequence_idx * sequence_length + head_idx * SS + batch_idx * NSS;
  int bd_offset = 0;
  bd_offset = head_idx * SS + batch_idx * NSS;

  const int i = threadIdx.x;
  if (i < sequence_length) {
    int bd_idx = (sequence_length - 1) + (sequence_idx * sequence_length) - sequence_idx;
    if (i < sequence_idx + 1) {
      output[ac_offset + i] = matrix_ac[ac_offset + i] + matrix_bd[bd_offset + bd_idx + i];
    } else if (i > sequence_length + 1) {
      output[ac_offset + i] = matrix_ac[ac_offset + i] + matrix_bd[bd_offset + bd_idx + i - 1];
    }
  }
}

template <typename T>
__global__ void LegacyRelShiftAddHalf(const int pos_sequence_length, const T* matrix_ac, const T* matrix_bd, T* output) {
  // matrix_bd: BxNxPSxS
  // matrix_ac: BxNxSxS

  int head_idx = threadIdx.y;
  int sequence_idx = blockIdx.x;
  int batch_idx = blockIdx.y;

  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int SS = sequence_length * sequence_length;
  const int NSS = SS * num_heads;

  const int ac_offset = sequence_idx * sequence_length + head_idx * SS + batch_idx * NSS;
  int bd_offset = 0;
  bd_offset = head_idx * SS + batch_idx * NSS;

  const int i = threadIdx.x;
  if (i < sequence_length) {
    int bd_idx = (sequence_length - 1) + (sequence_idx * sequence_length) - sequence_idx;
    if (i < sequence_idx + 1) {
      output[ac_offset + i] = matrix_ac[ac_offset + i] + matrix_bd[bd_offset + bd_idx + i];
    } else if (i > sequence_length + 1) {
      output[ac_offset + i] = matrix_ac[ac_offset + i] + matrix_bd[bd_offset + bd_idx + i - 1];
    }
  }
}

template <typename T>
__global__ void LatestRelShiftAddFloat2(const int pos_sequence_length, const T* matrix_ac, const T* matrix_bd, T* output) {
  // matrix_bd: BxNxPSxS
  // matrix_ac: BxNxSxS

  int head_idx = threadIdx.y;
  int sequence_idx = blockIdx.x;
  int batch_idx = blockIdx.y;

  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int SS = sequence_length * sequence_length;
  const int PSS = pos_sequence_length * sequence_length;
  const int NSS = SS * num_heads;
  const int NPSS = PSS * num_heads;

  const int ac_offset = sequence_idx * sequence_length + head_idx * SS + batch_idx * NSS;
  int bd_offset = 0;
  bd_offset = sequence_idx * pos_sequence_length + head_idx * PSS + batch_idx * NPSS;

  const int i = threadIdx.x;
  if (i < sequence_length) {
    const int bd_idx = sequence_length - 1 - sequence_idx + i;
    output[ac_offset + i].x = matrix_ac[ac_offset + i].x + matrix_bd[bd_offset + bd_idx].x;
    output[ac_offset + i].y = matrix_ac[ac_offset + i].y + matrix_bd[bd_offset + bd_idx].y;
  }
}

template <typename T>
__global__ void LatestRelShiftAddHalf2(const int pos_sequence_length, const T* matrix_ac, const T* matrix_bd, T* output) {
  // matrix_bd: BxNxPSxS
  // matrix_ac: BxNxSxS

  int head_idx = threadIdx.y;
  int sequence_idx = blockIdx.x;
  int batch_idx = blockIdx.y;

  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int SS = sequence_length * sequence_length;
  const int PSS = pos_sequence_length * sequence_length;
  const int NSS = SS * num_heads;
  const int NPSS = PSS * num_heads;

  const int ac_offset = sequence_idx * sequence_length + head_idx * SS + batch_idx * NSS;
  int bd_offset = 0;
  bd_offset = sequence_idx * pos_sequence_length + head_idx * PSS + batch_idx * NPSS;

  const int i = threadIdx.x;
  if (i < sequence_length) {
    const int bd_idx = sequence_length - 1 - sequence_idx + i;
    output[ac_offset + i].x = matrix_ac[ac_offset + i].x + matrix_bd[bd_offset + bd_idx].x;
    output[ac_offset + i].y = matrix_ac[ac_offset + i].y + matrix_bd[bd_offset + bd_idx].y;
  }
}

template <typename T>
__global__ void LatestRelShiftAddFloat(const int pos_sequence_length, const T* matrix_ac, const T* matrix_bd, T* output) {
  // matrix_bd: BxNxPSxS
  // matrix_ac: BxNxSxS

  int head_idx = threadIdx.y;
  int sequence_idx = blockIdx.x;
  int batch_idx = blockIdx.y;

  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int SS = sequence_length * sequence_length;
  const int PSS = pos_sequence_length * sequence_length;
  const int NSS = SS * num_heads;
  const int NPSS = PSS * num_heads;

  const int ac_offset = sequence_idx * sequence_length + head_idx * SS + batch_idx * NSS;
  int bd_offset = 0;
  bd_offset = sequence_idx * pos_sequence_length + head_idx * PSS + batch_idx * NPSS;

  const int i = threadIdx.x;
  if (i < sequence_length) {
    const int bd_idx = sequence_length - 1 - sequence_idx + i;
    output[ac_offset + i] = matrix_ac[ac_offset + i] + matrix_bd[bd_offset + bd_idx];
  }
}

template <typename T>
__global__ void LatestRelShiftAddHalf(const int pos_sequence_length, const T* matrix_ac, const T* matrix_bd, T* output) {
  // matrix_bd: BxNxPSxS
  // matrix_ac: BxNxSxS

  int head_idx = threadIdx.y;
  int sequence_idx = blockIdx.x;
  int batch_idx = blockIdx.y;

  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int SS = sequence_length * sequence_length;
  const int PSS = pos_sequence_length * sequence_length;
  const int NSS = SS * num_heads;
  const int NPSS = PSS * num_heads;

  const int ac_offset = sequence_idx * sequence_length + head_idx * SS + batch_idx * NSS;
  int bd_offset = 0;
  bd_offset = sequence_idx * pos_sequence_length + head_idx * PSS + batch_idx * NPSS;

  const int i = threadIdx.x;
  if (i < sequence_length) {
    const int bd_idx = sequence_length - 1 - sequence_idx + i;
    output[ac_offset + i] = matrix_ac[ac_offset + i] + matrix_bd[bd_offset + bd_idx];
  }
}

bool LaunchRelShiftAdd(cudaStream_t stream,
                       const int sequence_length, const int pos_sequence_length, const int batch_size, const int num_heads,
                       const float* matrix_ac, const float* matrix_bd, float* output) {
  // TODO : Support head_size > 1024
  const dim3 grid(sequence_length, batch_size, 1);

    const dim3 block(sequence_length, num_heads, 1);
    if (sequence_length == pos_sequence_length) {
      LegacyRelShiftAddFloat<float><<<grid, block, 0, stream>>>(sequence_length, matrix_ac, matrix_bd, output);
    } else {
      LatestRelShiftAddFloat<float><<<grid, block, 0, stream>>>(sequence_length, matrix_ac, matrix_bd, output);
    }
  // }
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchRelShiftAdd(cudaStream_t stream,
                       const int sequence_length, const int pos_sequence_length, const int batch_size, const int num_heads,
                       const half* matrix_ac, const half* matrix_bd, half* output) {
  // TODO : Support head_size > 1024
  const dim3 grid(sequence_length, batch_size, 1);
    const dim3 block(sequence_length, num_heads, 1);
    if (sequence_length == pos_sequence_length) {
      LegacyRelShiftAddHalf<half><<<grid, block, 0, stream>>>(sequence_length, matrix_ac, matrix_bd, output);
    } else {
      LatestRelShiftAddHalf<half><<<grid, block, 0, stream>>>(sequence_length, matrix_ac, matrix_bd, output);
    }
  // }
  return CUDA_CALL(cudaPeekAtLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime