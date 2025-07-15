#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>                      

#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC__
#include <cuda.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

size_t coo_compact(const uint64_t *d_RC_in, size_t N, int *&d_offset_out, uint64_t *&d_RC_out, int *&d_V_out, cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif

#endif