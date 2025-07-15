#include "ip_anonymize.h"
#include "cryptopan.cuh"
#include "config.h"

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define ETH_TYPE_IP    0x0800
#define ETH_TYPE_VLAN  0x8100

#define LINKTYPE_EN10MB 1
#define LINKTYPE_RAW    101

static const uint8_t DEFAULT_CRYPTOPAN_KEY[16] = {
    0x01, 0x23, 0x45, 0x67,
    0x89, 0xAB, 0xCD, 0xEF,
    0x10, 0x32, 0x54, 0x76,
    0x98, 0xBA, 0xDC, 0xFE
};

__constant__ uint8_t d_key[16];

__device__ const uint8_t* find_iphdr(const uint8_t* base, int link_type) {
    if (link_type == LINKTYPE_RAW) {
        return base;
    }
    uint16_t ether_type = ((uint16_t)base[12] << 8) | base[13];
    if (ether_type == ETH_TYPE_IP) {
        return base + 14;
    } else if (ether_type == ETH_TYPE_VLAN) {
        uint16_t inner = ((uint16_t)base[16] << 8) | base[17];
        if (inner == ETH_TYPE_IP) {
            return base + 18;
        }
    }
    return nullptr;
}

__device__ uint32_t load_le32(const uint8_t* p) {
    return ((uint32_t)p[3] << 24) | ((uint32_t)p[2] << 16) | ((uint32_t)p[1] << 8) | ((uint32_t)p[0]);
}

__device__ uint32_t compute_anon_ip(uint32_t ip) {
    return crypto_pan_encrypt_ipv4(ip, d_key);
}

__global__ void extract_ip_kernel(
    const uint8_t* __restrict__ d_pcap,
    const size_t* __restrict__ d_offsets,
    size_t num_packets,
    uint32_t* __restrict__ d_ip,
    size_t* __restrict__ d_ord,
    int link_type
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_packets) {
        return;
    }

    const uint8_t* pkt = d_pcap + d_offsets[idx];
    const uint8_t* iphdr = find_iphdr(pkt, link_type);
    if (!iphdr || (iphdr[0] >> 4) != 4) {
        return;
    }

    uint32_t src = load_le32(iphdr + 12);
    uint32_t dst = load_le32(iphdr + 16);

    d_ip[idx * 2] = src;
    d_ip[idx * 2 + 1] = dst;
    d_ord[idx * 2] = idx * 2;
    d_ord[idx * 2 + 1] = idx * 2 + 1;
}

__global__ void pre_anonymize_ip_kernel(const uint32_t* __restrict__ d_ip, uint32_t* __restrict__ d_ip_anony, size_t num_packets) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_packets) {
        return;
    }

    if (idx % BLOCK_SIZE == 0 || d_ip[idx] != d_ip[idx - 1]) {
        d_ip_anony[idx] = compute_anon_ip(d_ip[idx]);
    }
}

__global__ void scan_anonymize_ip_kernel(const uint32_t* __restrict__ d_ip, uint32_t* __restrict__ d_ip_anony, size_t num_packets) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_packets) {
        return;
    }

    if (idx % BLOCK_SIZE == 0) {
        uint32_t now_ip = d_ip_anony[idx];
        for (int i = idx + 1; i < idx + BLOCK_SIZE && i < num_packets; i++) {
            if (d_ip[i] != d_ip[i - 1]) {
                now_ip = d_ip_anony[i];
            } else {
                d_ip_anony[i] = now_ip;
            }
        }
    }
}

__global__ void allocate_RC_kernel(
    const uint32_t* __restrict__ d_ip_anony, 
    const size_t* __restrict__ d_ord,                                
    uint64_t* __restrict__ d_RC,
    size_t num_packets,
    int flag
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_packets) {
        return;
    }
    
    if (d_ord[idx] % 2 == flag) {
        d_RC[d_ord[idx] / 2] |= flag ? d_ip_anony[idx] : (uint64_t)d_ip_anony[idx] << 32;
    }
}

__global__ void force_kernel(
    const uint8_t* __restrict__ d_pcap,
    const size_t* __restrict__ d_offsets,
    size_t num_packets,
    uint64_t* __restrict__ d_RC,
    int link_type
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_packets) {
        return;
    }

    const uint8_t* pkt = d_pcap + d_offsets[idx];
    const uint8_t* iphdr = find_iphdr(pkt, link_type);
    if (!iphdr || (iphdr[0] >> 4) != 4) {
        return;
    }

    uint32_t src = load_le32(iphdr + 12);
    uint32_t dst = load_le32(iphdr + 16);

    src = compute_anon_ip(src);
    dst = compute_anon_ip(dst);

    d_RC[idx] = (uint64_t)src << 32 | dst;
}

void launch_extract_and_anonymize_ip(
    const uint8_t* d_pcap,
    const size_t* d_offsets,
    size_t num_packets,
    const uint8_t* key,
    uint64_t* d_RC
) {
    const uint8_t* effective_key = key ? key : DEFAULT_CRYPTOPAN_KEY;
    cudaMemcpyToSymbol(d_key, effective_key, 16);

    uint32_t h_linktype = LINKTYPE_EN10MB;
    cudaMemcpy(&h_linktype, d_pcap + 20, sizeof(uint32_t), cudaMemcpyDeviceToHost);

#ifdef LIA
    uint32_t* d_ip = nullptr;
    size_t* d_ord = nullptr;
    cudaMalloc(&d_ip, sizeof(uint32_t) * num_packets * 2);
    cudaMalloc(&d_ord, sizeof(size_t) * num_packets * 2);

    extract_ip_kernel<<<(num_packets + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_pcap, d_offsets, num_packets, d_ip, d_ord, h_linktype
    );

    thrust::device_ptr<uint32_t> key_ptr(d_ip);
    thrust::device_ptr<size_t>   val_ptr(d_ord);

    thrust::sort_by_key(
        key_ptr, key_ptr + num_packets,
        val_ptr
    );
    thrust::sort_by_key(
        key_ptr + num_packets, key_ptr + num_packets + num_packets,
        val_ptr + num_packets
    );

    uint32_t* d_ip_anony = nullptr;
    cudaMalloc(&d_ip_anony, sizeof(uint32_t) * num_packets * 2);
    pre_anonymize_ip_kernel<<<(2 * num_packets + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_ip, d_ip_anony, 2 * num_packets
    );
    scan_anonymize_ip_kernel<<<(2 * num_packets + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_ip, d_ip_anony, 2 * num_packets
    );
    allocate_RC_kernel<<<(2 * num_packets + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_ip_anony, d_ord, d_RC, 2 * num_packets, 0
    );
    allocate_RC_kernel<<<(2 * num_packets + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_ip_anony, d_ord, d_RC, 2 * num_packets, 1
    );

    cudaDeviceSynchronize();

    cudaFree(d_ip);
    cudaFree(d_ip_anony);
    cudaFree(d_ord);
#else
    force_kernel<<<(num_packets + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_pcap, d_offsets, num_packets, d_RC, h_linktype
    );
    cudaDeviceSynchronize();
#endif
}
