#define _GNU_SOURCE

#include "write.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>

typedef struct {
    const char *chunk_ptr;
    size_t chunk_size;
    char filename[MAX_FILENAME_LEN];
    int core_id;
} WriteTask;

void write_large_file(const char *path, const void *buffer, size_t size) {
    int fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd == -1) {
        perror("open");
        exit(1);
    }

    size_t remaining = size;
    const char *ptr = (const char *)buffer;

    while (remaining > 0) {
        size_t chunk = remaining > (1UL << 30) ? (1UL << 30) : remaining;
        ssize_t written = write(fd, ptr, chunk);
        if (written < 0) {
            perror("write");
            close(fd);
            exit(1);
        }

        remaining -= written;
        ptr += written;
    }

    close(fd);
}

void *write_thread_func(void *arg) {
    WriteTask *task = (WriteTask *)arg;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(task->core_id % MAX_CPU_CORES, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    write_large_file(task->filename, task->chunk_ptr, task->chunk_size);
    return NULL;
}

void write_chunks_parallel(const char *pwd, const void *buffer, size_t total_size, int num_chunks) {
    size_t chunk_size = (total_size + num_chunks - 1) / num_chunks;
    const char *base_ptr = (const char *)buffer;

    pthread_t *threads = malloc(num_chunks * sizeof(pthread_t));
    WriteTask *tasks = malloc(num_chunks * sizeof(WriteTask));

    for (int i = 0; i < num_chunks; ++i) {
        size_t offset = i * chunk_size;
        size_t this_chunk_size = (i == num_chunks - 1) ? total_size : chunk_size;

        tasks[i].chunk_ptr = base_ptr + offset;
        tasks[i].chunk_size = this_chunk_size;
        tasks[i].core_id = i;
        snprintf(tasks[i].filename, MAX_FILENAME_LEN, "%schunk_%03d.lz4", pwd, i);

        pthread_create(&threads[i], NULL, write_thread_func, &tasks[i]);

        total_size -= this_chunk_size;
    }

    for (int i = 0; i < num_chunks; ++i) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(tasks);
}
