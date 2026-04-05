// Minimal standalone GGUF reader using mmap
// No dependencies beyond POSIX + C++ standard library
//
// Usage: ./gguf-reader <model.gguf>
//
// Parses the GGUF header, prints metadata and tensor info,
// mmaps the file, and demonstrates accessing tensor data directly from the mapping.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---- GGUF format constants ----

static constexpr uint32_t GGUF_MAGIC     = 0x46554747; // "GGUF" in little-endian
static constexpr uint32_t GGUF_VERSION   = 3;
static constexpr size_t   GGUF_ALIGNMENT = 32;

enum gguf_type : int32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// ggml tensor data types (subset — enough to compute sizes for common types)
enum ggml_type : int32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8   = 24,
    GGML_TYPE_I16  = 25,
    GGML_TYPE_I32  = 26,
    GGML_TYPE_I64  = 27,
    GGML_TYPE_F64  = 28,
    GGML_TYPE_BF16 = 30,
};

// Returns {type_size_bytes, block_size_elements} for a given ggml_type.
// type_size is the byte size of one block, block_size is how many elements that block covers.
// Total bytes for a tensor = (n_elements / block_size) * type_size
static bool get_type_info(int32_t type, size_t & type_size, size_t & block_size, const char *& name) {
    switch (type) {
        case GGML_TYPE_F32:  type_size = 4;  block_size = 1;   name = "F32";  return true;
        case GGML_TYPE_F16:  type_size = 2;  block_size = 1;   name = "F16";  return true;
        case GGML_TYPE_Q4_0: type_size = 18; block_size = 32;  name = "Q4_0"; return true;  // 2 (f16 delta) + 16 (32 nibbles)
        case GGML_TYPE_Q4_1: type_size = 20; block_size = 32;  name = "Q4_1"; return true;  // 4 (2xf16) + 16
        case GGML_TYPE_Q5_0: type_size = 22; block_size = 32;  name = "Q5_0"; return true;  // 2 + 4 + 16
        case GGML_TYPE_Q5_1: type_size = 24; block_size = 32;  name = "Q5_1"; return true;  // 4 + 4 + 16
        case GGML_TYPE_Q8_0: type_size = 34; block_size = 32;  name = "Q8_0"; return true;  // 2 + 32
        case GGML_TYPE_Q8_1: type_size = 36; block_size = 32;  name = "Q8_1"; return true;  // 4 + 32
        case GGML_TYPE_Q2_K: type_size = 84; block_size = 256; name = "Q2_K"; return true;
        case GGML_TYPE_Q3_K: type_size = 110; block_size = 256; name = "Q3_K"; return true;
        case GGML_TYPE_Q4_K: type_size = 144; block_size = 256; name = "Q4_K"; return true;
        case GGML_TYPE_Q5_K: type_size = 176; block_size = 256; name = "Q5_K"; return true;
        case GGML_TYPE_Q6_K: type_size = 210; block_size = 256; name = "Q6_K"; return true;
        case GGML_TYPE_Q8_K: type_size = 292; block_size = 256; name = "Q8_K"; return true;
        case GGML_TYPE_I8:   type_size = 1;  block_size = 1;   name = "I8";   return true;
        case GGML_TYPE_I16:  type_size = 2;  block_size = 1;   name = "I16";  return true;
        case GGML_TYPE_I32:  type_size = 4;  block_size = 1;   name = "I32";  return true;
        case GGML_TYPE_I64:  type_size = 8;  block_size = 1;   name = "I64";  return true;
        case GGML_TYPE_F64:  type_size = 8;  block_size = 1;   name = "F64";  return true;
        case GGML_TYPE_BF16: type_size = 2;  block_size = 1;   name = "BF16"; return true;
        default:             type_size = 0;  block_size = 0;   name = "???";  return false;
    }
}

// ---- Simple binary reader (wraps a pointer into the mmap'd region) ----

struct reader {
    const uint8_t * data;
    size_t          size;
    size_t          pos;

    template <typename T>
    bool read(T & dst) {
        if (pos + sizeof(T) > size) return false;
        memcpy(&dst, data + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }

    bool read_bytes(void * dst, size_t n) {
        if (pos + n > size) return false;
        memcpy(dst, data + pos, n);
        pos += n;
        return true;
    }

    bool read_string(std::string & dst) {
        uint64_t len;
        if (!read(len)) return false;
        if (len > 1024 * 1024) return false; // sanity limit
        if (pos + len > size) return false;
        dst.assign(reinterpret_cast<const char *>(data + pos), len);
        pos += len;
        return true;
    }

    bool skip(size_t n) {
        if (pos + n > size) return false;
        pos += n;
        return true;
    }

    void align(size_t alignment) {
        size_t rem = pos % alignment;
        if (rem != 0) pos += alignment - rem;
    }
};

// ---- KV value display helper ----

static const char * gguf_type_name(int32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return "u8";
        case GGUF_TYPE_INT8:    return "i8";
        case GGUF_TYPE_UINT16:  return "u16";
        case GGUF_TYPE_INT16:   return "i16";
        case GGUF_TYPE_UINT32:  return "u32";
        case GGUF_TYPE_INT32:   return "i32";
        case GGUF_TYPE_FLOAT32: return "f32";
        case GGUF_TYPE_BOOL:    return "bool";
        case GGUF_TYPE_STRING:  return "str";
        case GGUF_TYPE_ARRAY:   return "arr";
        case GGUF_TYPE_UINT64:  return "u64";
        case GGUF_TYPE_INT64:   return "i64";
        case GGUF_TYPE_FLOAT64: return "f64";
        default:                return "???";
    }
}

// Size in bytes for a non-string, non-array GGUF value type
static size_t gguf_value_size(int32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return 1;
        case GGUF_TYPE_INT8:    return 1;
        case GGUF_TYPE_UINT16:  return 2;
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:  return 4;
        case GGUF_TYPE_INT32:   return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT64:  return 8;
        case GGUF_TYPE_INT64:   return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        default:                return 0;
    }
}

// Skip over a KV value (used for array elements and values we don't print inline)
static bool skip_value(reader & r, int32_t type);

static bool skip_value(reader & r, int32_t type) {
    if (type == GGUF_TYPE_STRING) {
        std::string tmp;
        return r.read_string(tmp);
    }
    if (type == GGUF_TYPE_ARRAY) {
        int32_t elem_type;
        uint64_t count;
        if (!r.read(elem_type) || !r.read(count)) return false;
        for (uint64_t i = 0; i < count; i++) {
            if (!skip_value(r, elem_type)) return false;
        }
        return true;
    }
    size_t sz = gguf_value_size(type);
    if (sz == 0) return false;
    return r.skip(sz);
}

// Print a single scalar value at the current reader position
static bool print_value(reader & r, int32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   { uint8_t  v; if (!r.read(v)) return false; printf("%u",    v); return true; }
        case GGUF_TYPE_INT8:    { int8_t   v; if (!r.read(v)) return false; printf("%d",    v); return true; }
        case GGUF_TYPE_UINT16:  { uint16_t v; if (!r.read(v)) return false; printf("%u",    v); return true; }
        case GGUF_TYPE_INT16:   { int16_t  v; if (!r.read(v)) return false; printf("%d",    v); return true; }
        case GGUF_TYPE_UINT32:  { uint32_t v; if (!r.read(v)) return false; printf("%u",    v); return true; }
        case GGUF_TYPE_INT32:   { int32_t  v; if (!r.read(v)) return false; printf("%d",    v); return true; }
        case GGUF_TYPE_FLOAT32: { float    v; if (!r.read(v)) return false; printf("%g",    v); return true; }
        case GGUF_TYPE_BOOL:    { int8_t   v; if (!r.read(v)) return false; printf("%s", v ? "true" : "false"); return true; }
        case GGUF_TYPE_UINT64:  { uint64_t v; if (!r.read(v)) return false; printf("%lu",   v); return true; }
        case GGUF_TYPE_INT64:   { int64_t  v; if (!r.read(v)) return false; printf("%ld",   v); return true; }
        case GGUF_TYPE_FLOAT64: { double   v; if (!r.read(v)) return false; printf("%g",    v); return true; }
        case GGUF_TYPE_STRING:  { std::string v; if (!r.read_string(v)) return false; printf("\"%s\"", v.c_str()); return true; }
        default: return false;
    }
}

// ---- Tensor info ----

struct tensor_info {
    std::string name;
    uint32_t    n_dims;
    int64_t     ne[4];      // dimension sizes (max 4)
    int32_t     type;
    uint64_t    offset;     // offset within the data section
};

// ---- Main ----

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * path = argv[1];

    // Open and mmap the entire file
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        close(fd);
        return 1;
    }

    size_t file_size = st.st_size;
    void * mapped = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    // Hint to the OS that we'll read sequentially through the metadata
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    reader r = { static_cast<const uint8_t *>(mapped), file_size, 0 };

    // 1. Read and verify magic
    uint32_t magic;
    if (!r.read(magic) || magic != GGUF_MAGIC) {
        fprintf(stderr, "Error: not a GGUF file (bad magic)\n");
        munmap(mapped, file_size);
        close(fd);
        return 1;
    }

    // 2. Read version
    uint32_t version;
    if (!r.read(version)) {
        fprintf(stderr, "Error: failed to read version\n");
        munmap(mapped, file_size);
        close(fd);
        return 1;
    }
    if (version < 2 || version > GGUF_VERSION) {
        fprintf(stderr, "Error: unsupported GGUF version %u (expected 2 or 3)\n", version);
        munmap(mapped, file_size);
        close(fd);
        return 1;
    }

    // 3. Read tensor count and KV count
    int64_t n_tensors, n_kv;
    if (!r.read(n_tensors) || !r.read(n_kv)) {
        fprintf(stderr, "Error: failed to read header counts\n");
        munmap(mapped, file_size);
        close(fd);
        return 1;
    }

    printf("GGUF v%u | %ld tensors | %ld KV pairs\n\n", version, n_tensors, n_kv);

    // 4. Read KV pairs
    size_t alignment = GGUF_ALIGNMENT;

    printf("--- Metadata ---\n");
    for (int64_t i = 0; i < n_kv; i++) {
        std::string key;
        int32_t type;
        if (!r.read_string(key) || !r.read(type)) {
            fprintf(stderr, "Error: failed to read KV pair %ld\n", i);
            munmap(mapped, file_size);
            close(fd);
            return 1;
        }

        printf("  %-50s %4s = ", key.c_str(), gguf_type_name(type));

        if (type == GGUF_TYPE_ARRAY) {
            int32_t elem_type;
            uint64_t count;
            if (!r.read(elem_type) || !r.read(count)) {
                fprintf(stderr, "Error: failed to read array header\n");
                munmap(mapped, file_size);
                close(fd);
                return 1;
            }
            printf("[%s x %lu]\n", gguf_type_name(elem_type), count);
            // Skip array contents
            for (uint64_t j = 0; j < count; j++) {
                if (!skip_value(r, elem_type)) {
                    fprintf(stderr, "Error: failed to skip array element\n");
                    munmap(mapped, file_size);
                    close(fd);
                    return 1;
                }
            }
        } else {
            if (!print_value(r, type)) {
                fprintf(stderr, "Error: failed to read value\n");
                munmap(mapped, file_size);
                close(fd);
                return 1;
            }
            printf("\n");
        }

        // Check for alignment override
        if (key == "general.alignment" && type == GGUF_TYPE_UINT32) {
            // We already advanced past the value, so peek back
            uint32_t align_val;
            memcpy(&align_val, r.data + r.pos - sizeof(uint32_t), sizeof(uint32_t));
            if (align_val > 0) {
                alignment = align_val;
            }
        }
    }

    // 5. Read tensor descriptors
    printf("\n--- Tensors ---\n");
    std::vector<tensor_info> tensors(n_tensors);

    for (int64_t i = 0; i < n_tensors; i++) {
        tensor_info & ti = tensors[i];
        ti.ne[0] = ti.ne[1] = ti.ne[2] = ti.ne[3] = 1;

        if (!r.read_string(ti.name)) {
            fprintf(stderr, "Error: failed to read tensor name\n");
            munmap(mapped, file_size);
            close(fd);
            return 1;
        }

        if (!r.read(ti.n_dims) || ti.n_dims > 4) {
            fprintf(stderr, "Error: bad dimension count for tensor '%s'\n", ti.name.c_str());
            munmap(mapped, file_size);
            close(fd);
            return 1;
        }

        for (uint32_t d = 0; d < ti.n_dims; d++) {
            if (!r.read(ti.ne[d])) {
                fprintf(stderr, "Error: failed to read dimension\n");
                munmap(mapped, file_size);
                close(fd);
                return 1;
            }
        }

        if (!r.read(ti.type) || !r.read(ti.offset)) {
            fprintf(stderr, "Error: failed to read tensor type/offset\n");
            munmap(mapped, file_size);
            close(fd);
            return 1;
        }

        // Compute total elements and byte size
        int64_t n_elements = ti.ne[0] * ti.ne[1] * ti.ne[2] * ti.ne[3];
        size_t type_size, block_size;
        const char * type_name;
        size_t byte_size = 0;
        if (get_type_info(ti.type, type_size, block_size, type_name)) {
            byte_size = (n_elements / block_size) * type_size;
        } else {
            type_name = "???";
        }

        printf("  %-50s %6s [", ti.name.c_str(), type_name);
        for (uint32_t d = 0; d < ti.n_dims; d++) {
            if (d > 0) printf(" x ");
            printf("%ld", ti.ne[d]);
        }
        printf("]  %8.2f MB\n", byte_size / (1024.0 * 1024.0));
    }

    // 6. Compute data section start (aligned after metadata)
    r.align(alignment);
    size_t data_offset = r.pos;

    printf("\n--- Data section ---\n");
    printf("  Offset:    0x%lx (%zu bytes into file)\n", data_offset, data_offset);
    printf("  Alignment: %zu bytes\n", alignment);
    printf("  Remaining: %.2f MB of tensor data\n", (file_size - data_offset) / (1024.0 * 1024.0));

    // 7. Demonstrate direct access to tensor data via mmap
    //    Each tensor's data lives at: mapped + data_offset + tensor.offset
    printf("\n--- Accessing tensor data via mmap ---\n");
    for (int64_t i = 0; i < n_tensors && i < 3; i++) {
        const tensor_info & ti = tensors[i];
        const uint8_t * tensor_data = static_cast<const uint8_t *>(mapped) + data_offset + ti.offset;

        printf("  %s: first 16 bytes = ", ti.name.c_str());
        size_t type_size, block_size;
        const char * type_name;
        get_type_info(ti.type, type_size, block_size, type_name);
        int64_t n_elements = ti.ne[0] * ti.ne[1] * ti.ne[2] * ti.ne[3];
        size_t byte_size = (n_elements / block_size) * type_size;

        size_t preview = byte_size < 16 ? byte_size : 16;
        for (size_t b = 0; b < preview; b++) {
            printf("%02x ", tensor_data[b]);
        }
        printf("\n");
    }

    // In a real inference engine, you would now:
    // - Keep the mmap active for the lifetime of inference
    // - Point your tensor structs' data pointers directly into the mapped region:
    //     tensor.data = (void *)(mapped + data_offset + tensor.offset);
    // - The OS pages in data from disk on first access (or immediately with MAP_POPULATE)
    // - Optionally call mlock() to pin pages and prevent eviction

    printf("\nDone. File remains mmap'd — in a real engine, tensor data pointers\n"
           "would reference directly into this mapping with zero copies.\n");

    munmap(mapped, file_size);
    close(fd);
    return 0;
}
