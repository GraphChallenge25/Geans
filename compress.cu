#include "compress.cuh"
#include "config.h"

#include <nvcomp/lz4.hpp>
#include <nvcomp/zstd.hpp>
#include <nvcomp/nvcompManager.hpp>

#include <cub/cub.cuh>

inline int bit_width(unsigned long long x) {
    return x == 0 ? 0 : 64 - __builtin_clzll(x);
}

__global__ void pre_encode_kernel(const uint64_t* __restrict__ RC_in, const int* __restrict__ V_in, size_t* __restrict__ out, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    uint32_t R, C;
    int V = V_in[i]; 
    
    if (i == 0) {
        R = RC_in[i] >> 32;
        C = RC_in[i] & (0xFFFF);
    } else {
        R = (RC_in[i] - RC_in[i - 1]) >> 32;
        C = (RC_in[i] - RC_in[i - 1]) & (0xFFFF);
    }

    out[i] = (R ? 38 - __builtin_clz(R) : 7) / 7 + 
             (C ? 38 - __builtin_clz(C) : 7) / 7 +
             (V ? 38 - __builtin_clz(V) : 7) / 7;
}

void cub_exclusive_scan(size_t *d_input_output, size_t count, cudaStream_t stream) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_input_output, d_input_output,
        count, stream
    );

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_input_output, d_input_output,
        count, stream
    );
}

__global__ void encode_kernel(const uint64_t* __restrict__ RC_in, const int* __restrict__ V_in, const size_t *offset_in, uint8_t* __restrict__ out, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    uint8_t *ptr = out + offset_in[i];
    uint32_t R, C;
    int V = V_in[i]; 
    
    if (i == 0) {
        R = RC_in[i] >> 32;
        C = RC_in[i] & (0xFFFF);
    } else {
        R = (RC_in[i] - RC_in[i - 1]) >> 32;
        C = (RC_in[i] - RC_in[i - 1]) & (0xFFFF);
    }

    int k = 0;

    while (R >= 0x80) {
        ptr[k++] = (R & 0x7F) | 0x80;
        R >>= 7;
    }
    ptr[k++] = R;

    while (C >= 0x80) {
        ptr[k++] = (C & 0x7F) | 0x80;
        C >>= 7;
    }
    ptr[k++] = C;
    
    while (V >= 0x80) {
        ptr[k++] = (V & 0x7F) | 0x80;
        V >>= 7;
    }
    ptr[k++] = V;
}

__global__ void copy_RCV_kernel(const uint64_t* __restrict__ RC_in, const int* __restrict__ V_in, uint8_t* __restrict__ out, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    size_t offset = i * (sizeof(uint64_t) + sizeof(int));

    uint64_t rc = RC_in[i];
    int v = V_in[i];

#pragma unroll
    for (int j = 0; j < 8; ++j) {
        out[offset + j] = (rc >> (j * 8)) & 0xFF;
    }
    for (int j = 0; j < 4; ++j) {
        out[offset + 8 + j] = (v >> (j * 8)) & 0xFF;
    }
}

void compress_nvcomp(const uint64_t *d_RC, const int *d_V, const size_t N, void *&d_compressed, size_t *&h_comp_size, cudaStream_t stream) {
#ifdef HHE
    size_t *d_offset;
    cudaMalloc(&d_offset, (N + 1) * sizeof(size_t));

    pre_encode_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_RC, d_V, d_offset, N);

    cub_exclusive_scan(d_offset, N + 1, stream);

    size_t h_bytes = 0;
    cudaMemcpy(&h_bytes, d_offset + N, sizeof(size_t), cudaMemcpyDeviceToHost);

    const size_t BYTES = h_bytes;

    printf("Encode length = %lu Bytes...\n", BYTES);

    uint8_t *d_uncompressed;
    cudaMalloc(&d_uncompressed, BYTES);
    
    encode_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_RC, d_V, d_offset, d_uncompressed, N);
#else
    const size_t BYTES = N * (sizeof(uint64_t) + sizeof(int));
    uint8_t *d_uncompressed;
    cudaMalloc(&d_uncompressed, BYTES);

    copy_RCV_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_RC, d_V, d_uncompressed, N);

#endif
    const size_t comp_CHUNK_SIZE = 1 << 17; 
    nvcompBatchedZstdOpts_t opts = {NVCOMP_TYPE_CHAR};
    
    std::shared_ptr<nvcomp::nvcompManagerBase> manager =
        std::make_shared<nvcomp::ZstdManager>(
            comp_CHUNK_SIZE, opts, stream, nvcomp::NoComputeNoVerify,
            nvcomp::BitstreamKind::NVCOMP_NATIVE
        );
            
    nvcomp::CompressionConfig config = manager->configure_compression(BYTES);

    cudaMalloc(&d_compressed, config.max_compressed_buffer_size);

    cudaMallocHost(&h_comp_size, sizeof(size_t));
    *h_comp_size = 0;

    manager->compress(
        static_cast<const uint8_t*>(d_uncompressed),
        static_cast<uint8_t*>(d_compressed),
        config,
        h_comp_size
    );

    cudaStreamSynchronize(stream);
}