#include "processing.h"

#include "ip_anonymize.h"
#include "matrix.cuh"
#include "compress.cuh"
#include "write.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <stdio.h>
#include <string>

__global__ void random_kernel(uint64_t* d_out, size_t size, uint64_t seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    uint64_t hi = curand(&state);
    uint64_t lo = curand(&state);
    d_out[idx] = (hi << 32) | lo;
}

void PCAP_processing(
    uint8_t* d_pcap,
    const size_t* d_pcap_offsets,
    size_t N,
    const char* output_dir
) {
#ifdef ENABLE_TIME_MEASURE
    auto ip_start = std::chrono::high_resolution_clock::now();
#endif

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint64_t* d_RC = nullptr;
    cudaMalloc(&d_RC, N * sizeof(uint64_t));

    launch_extract_and_anonymize_ip(d_pcap, d_pcap_offsets, N, nullptr, d_RC);

    cudaFree(d_pcap);

#ifdef ENABLE_TIME_MEASURE
    auto ip_stop = std::chrono::high_resolution_clock::now();
    printf("Ip extract+anonymize time = %.3f ms\n", std::chrono::duration<double, std::milli>(ip_stop - ip_start).count());

    auto merge_start = std::chrono::high_resolution_clock::now();
#endif

    int *d_offsets = nullptr;
    uint64_t *d_RC_reduced = nullptr;
    int *d_V_reduced = nullptr;

    size_t reduced_size = coo_compact(d_RC, N, d_offsets, d_RC_reduced, d_V_reduced, stream);

#ifdef ENABLE_TIME_MEASURE
    printf("reduced size = %lu\n", reduced_size);

    auto merge_stop = std::chrono::high_resolution_clock::now();
    printf("Total GPU sort+reduce time = %.3f ms\n", std::chrono::duration<double, std::milli>(merge_stop - merge_start).count());

    auto compress_start = std::chrono::high_resolution_clock::now();
#endif

    void *d_compressed = nullptr;
    size_t *h_comp_size = nullptr;
    compress_nvcomp(d_RC_reduced, d_V_reduced, reduced_size, d_compressed, h_comp_size, stream);

#ifdef ENABLE_TIME_MEASURE
    auto compress_end = std::chrono::high_resolution_clock::now();

    printf("✅ Compressed output: %lu → %lu bytes\n", reduced_size * (sizeof(u_int64_t) + sizeof(int)), *h_comp_size);
    printf("🧩 nvcomp GPU compression time = %.3f ms\n", std::chrono::duration<double, std::milli>(compress_end - compress_start).count());

    auto copy_start = std::chrono::high_resolution_clock::now();
#endif

    void *h_output = nullptr;
    cudaMallocHost(&h_output, *h_comp_size);
    cudaMemcpy(h_output, d_compressed, *h_comp_size, cudaMemcpyDeviceToHost);

#ifdef ENABLE_TIME_MEASURE
    auto copy_stop = std::chrono::high_resolution_clock::now();
    printf("💾 Host memcpy time = %.3f ms\n", std::chrono::duration<double, std::milli>(copy_stop - copy_start).count());
    
    auto write_start = std::chrono::high_resolution_clock::now();
#endif

    write_chunks_parallel(output_dir, h_output, *h_comp_size, 128);

#ifdef ENABLE_TIME_MEASURE
    auto write_stop = std::chrono::high_resolution_clock::now();
    printf("📝 Write files time = %.3f ms\n", std::chrono::duration<double, std::milli>(write_stop - write_start).count());
#endif
}
