#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

typedef struct hrx_buffer_s * hrx_buffer_t;

namespace ggml::hrx {

class UnifiedBufferRef {
  public:
    UnifiedBufferRef() = default;
    ~UnifiedBufferRef();

    UnifiedBufferRef(UnifiedBufferRef && other) noexcept;
    UnifiedBufferRef & operator=(UnifiedBufferRef && other) noexcept;

    UnifiedBufferRef(const UnifiedBufferRef &)             = delete;
    UnifiedBufferRef & operator=(const UnifiedBufferRef &) = delete;

    bool valid() const { return buffer_ != nullptr; }

    hrx_buffer_t buffer() const { return buffer_; }

    size_t offset() const { return offset_; }

  private:
    hrx_buffer_t buffer_ = nullptr;
    size_t       offset_ = 0;

    UnifiedBufferRef(hrx_buffer_t buffer, size_t offset);
    friend class UnifiedBufferRegistry;
};

// Tracks mapped buffers owned by the HRX buffer type so pointer-based GGML transfers can remain handle based.
class UnifiedBufferRegistry {
  public:
    void add(hrx_buffer_t buffer, void * base, size_t size);
    void remove(hrx_buffer_t buffer);

    UnifiedBufferRef find(const void * data, size_t size) const;

  private:
    struct Entry {
        hrx_buffer_t buffer = nullptr;
        uintptr_t    base   = 0;
        size_t       size   = 0;
    };

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
};

}  // namespace ggml::hrx
