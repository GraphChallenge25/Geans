#ifndef CRYPTOPAN_H
#define CRYPTOPAN_H

#include <stdint.h>
#include <cuda_runtime.h>

__device__ uint32_t crypto_pan_encrypt_ipv4(uint32_t ip, const uint8_t* key);

#endif
