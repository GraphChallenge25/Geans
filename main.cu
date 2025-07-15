#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <vector>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>
#include "processing.h"
#include "config.h"

int main(int argc, char* argv[]) {
#ifdef ENABLE_TIME_MEASURE
    auto Start = std::chrono::high_resolution_clock::now();
#endif
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.pcac> <output_dir>\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_dir = argv[2];

#ifdef ENABLE_TIME_MEASURE
    auto offset_start = std::chrono::high_resolution_clock::now();
#endif

    int fd = open(input_path, O_RDONLY);
    if (fd < 0) {
        perror("open input");
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        close(fd);
        return 1;
    }

    size_t filesize = st.st_size;
    uint8_t* file_data = (uint8_t*)mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    std::cout << "PCAP Read successful!\n";

    size_t num_packets = 0;
    size_t* offsets = (size_t*)malloc(size_t(1ull << 30) * sizeof(size_t));
    size_t curr = 24; // PCAP global header

    while (curr + 16 <= filesize) {
        const uint8_t* pkt_hdr = file_data + curr;
        uint32_t incl_len = *(uint32_t*)(pkt_hdr + 8);

        if (curr + 16 + incl_len > filesize) {
            break;
        }

        offsets[num_packets++] = curr + 16;
        curr += 16 + incl_len;
    }

    if (num_packets == 0) {
        fprintf(stderr, "No packets found in %s\n", input_path);
        return 1;
    }

    std::cout << "PCAP packets = " << num_packets << '\n';

    uint8_t* d_pcap = nullptr;
    size_t* d_pcap_offsets = nullptr;

    cudaMalloc(&d_pcap, filesize);
    cudaMemcpy(d_pcap, file_data, filesize, cudaMemcpyHostToDevice);

    cudaMalloc(&d_pcap_offsets, num_packets * sizeof(size_t));
    cudaMemcpy(d_pcap_offsets, offsets, num_packets * sizeof(size_t), cudaMemcpyHostToDevice);

#ifdef ENABLE_TIME_MEASURE
    auto offset_end = std::chrono::high_resolution_clock::now();
    printf("PCAP read and segment and transfer time = %.3f ms\n", std::chrono::duration<double, std::milli>(offset_end - offset_start).count());
#endif

#ifdef ENABLE_TIME_MEASURE
    auto processing_start = std::chrono::high_resolution_clock::now();
#endif

    PCAP_processing(d_pcap, d_pcap_offsets, num_packets, output_dir);

#ifdef ENABLE_TIME_MEASURE
    auto processing_end = std::chrono::high_resolution_clock::now();
    printf("PCAP processing time = %.3f ms\n", std::chrono::duration<double, std::milli>(processing_end - processing_start).count());
#endif

#ifdef ENABLE_TIME_MEASURE
    auto End = std::chrono::high_resolution_clock::now();
    printf("Total time = %.3f ms\n", std::chrono::duration<double, std::milli>(End - Start).count());
#endif

    cudaFree(d_pcap);
    cudaFree(d_pcap_offsets);
    munmap(file_data, filesize);
    close(fd);

    return 0;
}
