#include "radish_ffi.h"
#include "hashtable.h"
#include "aof.h"
#include "persistence.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>

/**
 * FFI Implementation for RadishDB
 * Bridges binary data handling between Rust and C
 */

/* Wrapper to handle binary data by converting to/from strings */

RadishDBHandle radish_ht_create(int initial_size) {
    return (RadishDBHandle)ht_create(initial_size);
}

void radish_ht_set(RadishDBHandle ht, const void *key, size_t klen,
                   const void *value, size_t vlen, int64_t expires_at) {
    if (!ht || !key || !value) return;
    
    /* Create null-terminated string keys/values for storage */
    char *key_str = malloc(klen + 1);
    char *val_str = malloc(vlen + 1);
    
    if (!key_str || !val_str) {
        free(key_str);
        free(val_str);
        return;
    }
    
    memcpy(key_str, key, klen);
    key_str[klen] = '\0';
    
    memcpy(val_str, value, vlen);
    val_str[vlen] = '\0';
    
    ht_set((HashTable *)ht, key_str, val_str, (time_t)expires_at);
    
    free(key_str);
    free(val_str);
}

void* radish_ht_get(RadishDBHandle ht, const void *key, size_t klen,
                    size_t *out_len) {
    if (!ht || !key) return NULL;
    
    char *key_str = malloc(klen + 1);
    if (!key_str) return NULL;
    
    memcpy(key_str, key, klen);
    key_str[klen] = '\0';
    
    char *result = ht_get((HashTable *)ht, key_str);
    free(key_str);
    
    if (result) {
        *out_len = strlen(result);
        return result;
    }
    return NULL;
}

int radish_ht_delete(RadishDBHandle ht, const void *key, size_t klen) {
    if (!ht || !key) return 0;
    
    char *key_str = malloc(klen + 1);
    if (!key_str) return 0;
    
    memcpy(key_str, key, klen);
    key_str[klen] = '\0';
    
    int result = ht_delete((HashTable *)ht, key_str);
    free(key_str);
    
    return result;
}

void radish_ht_free(RadishDBHandle ht) {
    if (ht) ht_free((HashTable *)ht);
}

/* AOF wrapper */
int radish_aof_open(const char *filename) {
    /* Ensure aof directory exists */
    char dir_path[256] = {0};
    strncpy(dir_path, filename, sizeof(dir_path) - 1);
    
    /* Find last slash */
    char *last_slash = strrchr(dir_path, '/');
    if (last_slash) {
        *last_slash = '\0';
        mkdir(dir_path, 0755);  /* Safe even if exists */
    }
    
    int result = aof_open(filename);
    fprintf(stderr, "[RadishDB FFI] aof_open('%s') = %d\n", filename, result);
    return result;
}

void radish_aof_close(void) {
    aof_close();
}

void radish_aof_append(const void *key, size_t klen,
                       const void *value, size_t vlen, int64_t expires_at) {
    if (!key || !value) {
        fprintf(stderr, "[RadishDB FFI] radish_aof_append: null pointer\n");
        return;
    }
    
    char *key_str = malloc(klen + 1);
    char *val_str = malloc(vlen + 1);
    
    if (!key_str || !val_str) {
        fprintf(stderr, "[RadishDB FFI] radish_aof_append: malloc failed\n");
        free(key_str);
        free(val_str);
        return;
    }
    
    memcpy(key_str, key, klen);
    key_str[klen] = '\0';
    
    memcpy(val_str, value, vlen);
    val_str[vlen] = '\0';
    
    /* Convert expire timestamp to string format */
    char expire_str[32];
    snprintf(expire_str, sizeof(expire_str), "%ld", (long)expires_at);
    
    aof_append_set(key_str, val_str, expire_str);
    fprintf(stderr, "[RadishDB FFI] Appended event: key_len=%zu, val_len=%zu\n", klen, vlen);
    
    free(key_str);
    free(val_str);
}

size_t radish_aof_get_size(void) {
    return aof_filesize("aof/radish.aof");
}

/* JSON export stub */
int radish_export_json(RadishDBHandle ht, const char *filename) {
    if (!ht || !filename) return 0;
    
    /* For now, just write a placeholder JSON */
    FILE *f = fopen(filename, "w");
    if (!f) return 0;
    
    fprintf(f, "{\"radishdb\": \"export\", \"status\": \"ok\"}\n");
    fclose(f);
    
    return 1;
}
