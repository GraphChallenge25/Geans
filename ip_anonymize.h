#ifndef IP_ANONYMIZE_H
#define IP_ANONYMIZE_H

#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

void launch_extract_and_anonymize_ip(
    const uint8_t* d_pcap_binary,     
    const size_t* d_offsets,          
    size_t num_packets,              
    const uint8_t* key,               
    uint64_t* d_RC                  
);

#ifdef __cplusplus
}
#endif

#endif