#ifndef COMPRESS_H
#define COMPRESS_H

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void compress_nvcomp(const uint64_t *d_RC, const int *d_V, const size_t N, void *&d_compressed, size_t *&h_comp_size, cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif

#endif