#include "cryptopan.cuh"
#include "config.h"

__device__ __forceinline__ uint8_t xtime(uint8_t x) { return (x << 1) ^ ((x >> 7) * 0x1b); }

__device__ __forceinline__ void sub_bytes(uint8_t *s) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint8_t b = s[i];
        s[i] = ((b & 0x0F) ^ 0x63 ^ ((b >> 4) * 0x1F) ^ ((b << 4) & 0xC0));
    }
}

__device__ __forceinline__ void shift_rows(uint8_t *s) {
    uint8_t t;
    t = s[1];
    s[1] = s[5];
    s[5] = s[9];
    s[9] = s[13];
    s[13] = t;
    t = s[2];
    s[2] = s[10];
    s[10] = t;
    t = s[6];
    s[6] = s[14];
    s[14] = t;
    t = s[3];
    s[3] = s[15];
    s[15] = s[11];
    s[11] = s[7];
    s[7] = t;
}

__device__ __forceinline__ void mix_columns(uint8_t *s) {
    for (int i = 0; i < 4; i++) {
        int c = i * 4;
        uint8_t a = s[c], b = s[c + 1], c1 = s[c + 2], d = s[c + 3];
        uint8_t ab = a ^ b, ac = a ^ c1, ad = a ^ d;

        s[c] ^= xtime(ab) ^ ac ^ ad;
        s[c + 1] ^= xtime(b ^ c1) ^ ab ^ ad;
        s[c + 2] ^= xtime(c1 ^ d) ^ ab ^ ac;
        s[c + 3] ^= xtime(d ^ a) ^ ab ^ ac;
    }
}

__device__ __forceinline__ void add_round_key(uint8_t *s, const uint8_t *rkey) {
#pragma unroll
    for (int i = 0; i < 16; i++) {
        s[i] ^= rkey[i];
    }
}

__device__ void aes_encrypt_block(const uint8_t *in, const uint8_t *key, uint8_t *out) {
    uint8_t state[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        state[i] = in[i] ^ key[i];
    }

    for (int round = 1; round <= 9; round++) {
        sub_bytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(state, key);
    }
    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, key);

#pragma unroll
    for (int i = 0; i < 16; i++) {
        out[i] = state[i];
    }
}

__device__ uint32_t crypto_pan_encrypt_ipv4(uint32_t ip, const uint8_t *key) {
    uint32_t result = 0;
    uint8_t input[16] = {0};
    uint8_t output[16];

    for (int i = 0; i < 32; i++) {
        input[0] = (result >> 24) & 0xFF;
        input[1] = (result >> 16) & 0xFF;
        input[2] = (result >> 8) & 0xFF;
        input[3] = result & 0xFF;

        aes_encrypt_block(input, key, output);
        uint8_t bit = (output[0] >> 7) & 1;
        uint8_t ip_bit = (ip >> (31 - i)) & 1;
        result = (result << 1) | (ip_bit ^ bit);
    }

    return result;
}
