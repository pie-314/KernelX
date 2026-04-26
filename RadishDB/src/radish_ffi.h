#ifndef RADISH_FFI_H
#define RADISH_FFI_H

#include <stddef.h>
#include <stdint.h>

/**
 * FFI Interface for RadishDB - exposed to Rust/C++ consumers
 * Handles binary key-value data for trajectory storage
 */

typedef void* RadishDBHandle;

/* Initialize RadishDB with binary data support */
RadishDBHandle radish_ht_create(int initial_size);

/* Set a binary key-value pair with optional expiration */
void radish_ht_set(RadishDBHandle ht, const void *key, size_t klen,
                   const void *value, size_t vlen, int64_t expires_at);

/* Get a binary value by key (caller must free) */
void* radish_ht_get(RadishDBHandle ht, const void *key, size_t klen,
                    size_t *out_len);

/* Delete a key */
int radish_ht_delete(RadishDBHandle ht, const void *key, size_t klen);

/* Free the hashtable */
void radish_ht_free(RadishDBHandle ht);

/* AOF Operations */
int radish_aof_open(const char *filename);
void radish_aof_close(void);
void radish_aof_append(const void *key, size_t klen,
                       const void *value, size_t vlen, int64_t expires_at);

/* Get AOF file size */
size_t radish_aof_get_size(void);

/* Export database state to JSON */
int radish_export_json(RadishDBHandle ht, const char *filename);

#endif
