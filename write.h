#ifndef WRITE_H
#define WRITE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void write_chunks_parallel(const char *pwd, const void *buffer, size_t total_size, int num_chunks);

#ifdef __cplusplus
}
#endif

#endif