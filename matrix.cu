#include "matrix.cuh"
#include "config.h"

#include <algorithm> 

#include <cub/device/device_radix_sort.cuh>

#include <thrust/execution_policy.h>          
#include <thrust/iterator/zip_iterator.h>     
#include <thrust/tuple.h>                   
#include <thrust/reduce.h>                      
#include <thrust/functional.h>                 
#include <thrust/iterator/discard_iterator.h>    
#include <thrust/iterator/constant_iterator.h>   
#include <thrust/sort.h>  
#include <thrust/device_vector.h>

void RadixSort(const uint16_t *d_B, const uint64_t *d_RC, uint16_t *d_B_out, uint64_t *d_RC_out, size_t N, cudaStream_t stream) {
    uint16_t *d_B_temp;
    uint64_t *d_RC_temp;
    cudaMalloc(&d_B_temp, N * sizeof(uint16_t));
    cudaMalloc(&d_RC_temp, N * sizeof(uint64_t));

    void *d_temp_storage = nullptr;
    size_t temp_bytes_RC = 0, temp_bytes_B = 0;

    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes_RC, d_RC, d_RC_temp, d_B, d_B_temp, N, 0, 64, stream);

    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes_B, d_B_temp, d_B_out, d_RC_temp, d_RC_out, N, 0, 13, stream);

    size_t temp_bytes = std::max(temp_bytes_RC, temp_bytes_B);
    cudaMalloc(&d_temp_storage, temp_bytes);

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_bytes_RC, d_RC, d_RC_temp, d_B, d_B_temp, N, 0, 64, stream);

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_bytes_B, d_B_temp, d_B_out, d_RC_temp, d_RC_out, N, 0, 13, stream);

    cudaFree(d_temp_storage);
    cudaFree(d_B_temp);
    cudaFree(d_RC_temp);
}

__global__ void generate_B_V_kernel(uint16_t* B, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        B[i] = i / CHUNK_SIZE;
    }
}

size_t coo_compact(const uint64_t *d_RC_in, size_t N, int *&d_offset_out, uint64_t *&d_RC_out, int *&d_V_out, cudaStream_t stream) {
#ifdef GMA
    const int CHUNK_NUM = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;

    uint16_t *d_B_in;
    cudaMalloc(&d_B_in, N * sizeof(uint16_t));
    generate_B_V_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_B_in, N);

    uint16_t *d_B_out;
    cudaMalloc(&d_B_out, N * sizeof(uint16_t));
    cudaMalloc(&d_RC_out, N * sizeof(uint64_t));
    cudaMalloc(&d_V_out, N * sizeof(int));

    auto policy = thrust::cuda::par.on(stream);

    uint16_t *d_B_sorted;
    uint64_t *d_RC_sorted;
    cudaMalloc(&d_B_sorted, N * sizeof(uint16_t));
    cudaMalloc(&d_RC_sorted, N * sizeof(uint64_t));

    RadixSort(d_B_in, d_RC_in, d_B_sorted, d_RC_sorted, N, stream);

    auto reduced_end = thrust::reduce_by_key(
        policy, 
        thrust::make_zip_iterator(thrust::make_tuple(d_B_sorted, d_RC_sorted)), 
        thrust::make_zip_iterator(thrust::make_tuple(d_B_sorted + N, d_RC_sorted + N)), 
        thrust::make_constant_iterator(1), 
        thrust::make_zip_iterator(thrust::make_tuple(d_B_out, d_RC_out)), 
        d_V_out
    );

    size_t reduced_size = thrust::get<0>(reduced_end.first.get_iterator_tuple()) - d_B_out;
    cudaMalloc(&d_offset_out, CHUNK_NUM * sizeof(int));
    thrust::reduce_by_key(
        policy, 
        d_B_out, 
        d_B_out + reduced_size,     
        thrust::make_constant_iterator(1),                     
        thrust::make_discard_iterator(),       
        d_offset_out        
    );

    return reduced_size;
#else
    const size_t CHUNK_NUM = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;
    std::vector<uint64_t*> chunks_RC;
    std::vector<int*> chunks_V;
    std::vector<size_t> chunks_NNZ;

    size_t total_nnz = 0;

    for (size_t i = 0; i < CHUNK_NUM; ++i) {
        size_t offset = i * CHUNK_SIZE;
        size_t len = std::min(size_t(CHUNK_SIZE), N - offset);

        thrust::device_vector<uint64_t> block_keys(d_RC_in + offset, d_RC_in + offset + len);
        thrust::device_vector<int> block_vals(len, 1);

        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            block_keys.begin(), block_keys.end(),
            block_vals.begin()
        );

        thrust::device_vector<uint64_t> out_keys(len);
        thrust::device_vector<int> out_vals(len);

        auto reduce_end = thrust::reduce_by_key(
            thrust::cuda::par.on(stream),
            block_keys.begin(), block_keys.end(),
            block_vals.begin(),
            out_keys.begin(),
            out_vals.begin()
        );

        size_t nnz = reduce_end.first - out_keys.begin();
        total_nnz += nnz;
        chunks_NNZ.push_back(nnz);

        uint64_t* d_rc;
        int* d_v;
        cudaMalloc(&d_rc, nnz * sizeof(uint64_t));
        cudaMalloc(&d_v, nnz * sizeof(int));
        cudaMemcpyAsync(d_rc, thrust::raw_pointer_cast(out_keys.data()), nnz * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_v, thrust::raw_pointer_cast(out_vals.data()), nnz * sizeof(int), cudaMemcpyDeviceToDevice, stream);

        chunks_RC.push_back(d_rc);
        chunks_V.push_back(d_v);
    }

    cudaMalloc(&d_RC_out, total_nnz * sizeof(uint64_t));
    cudaMalloc(&d_V_out, total_nnz * sizeof(int));
    cudaMalloc(&d_offset_out, CHUNK_NUM * sizeof(int));

    size_t cursor = 0;
    std::vector<int> h_offsets(CHUNK_NUM);
    for (size_t i = 0; i < CHUNK_NUM; ++i) {
        size_t nnz = chunks_NNZ[i];
        cudaMemcpyAsync(d_RC_out + cursor, chunks_RC[i], nnz * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_V_out + cursor, chunks_V[i], nnz * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        h_offsets[i] = cursor;
        cursor += nnz;

        cudaFree(chunks_RC[i]);
        cudaFree(chunks_V[i]);
    }

    cudaMemcpyAsync(d_offset_out, h_offsets.data(), CHUNK_NUM * sizeof(int), cudaMemcpyHostToDevice, stream);
    return total_nnz;
#endif
}