#ifndef PROCESSING_H
#define PROCESSING_H

#include "config.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void PCAP_processing(
    uint8_t* d_pcap,
    const size_t* d_pcap_offsets,
    size_t N,
    const char* output_dir
);

#ifdef __cplusplus
}
#endif


#endif 
