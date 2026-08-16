#include "unified-memory.h"

#include "hrx_runtime.h"

#include <algorithm>
#include <utility>

namespace ggml::hrx {

UnifiedBufferRef::UnifiedBufferRef(hrx_buffer_t buffer, size_t offset) : buffer_(buffer), offset_(offset) {}

UnifiedBufferRef::~UnifiedBufferRef() {
    if (buffer_ != nullptr) {
        hrx_buffer_release(buffer_);
    }
}

UnifiedBufferRef::UnifiedBufferRef(UnifiedBufferRef && other) noexcept :
    buffer_(std::exchange(other.buffer_, nullptr)),
    offset_(other.offset_) {
    other.offset_ = 0;
}

UnifiedBufferRef & UnifiedBufferRef::operator=(UnifiedBufferRef && other) noexcept {
    if (this != &other) {
        if (buffer_ != nullptr) {
            hrx_buffer_release(buffer_);
        }
        buffer_       = std::exchange(other.buffer_, nullptr);
        offset_       = other.offset_;
        other.offset_ = 0;
    }
    return *this;
}

void UnifiedBufferRegistry::add(hrx_buffer_t buffer, void * base, size_t size) {
    if (buffer == nullptr || base == nullptr || size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.push_back({ buffer, reinterpret_cast<uintptr_t>(base), size });
}

void UnifiedBufferRegistry::remove(hrx_buffer_t buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [buffer](const Entry & entry) { return entry.buffer == buffer; }),
                   entries_.end());
}

UnifiedBufferRef UnifiedBufferRegistry::find(const void * data, size_t size) const {
    if (data == nullptr) {
        return {};
    }
    const uintptr_t             address = reinterpret_cast<uintptr_t>(data);
    std::lock_guard<std::mutex> lock(mutex_);
    for (const Entry & entry : entries_) {
        if (address < entry.base) {
            continue;
        }
        const size_t offset = static_cast<size_t>(address - entry.base);
        if (offset > entry.size || size > entry.size - offset) {
            continue;
        }
        hrx_buffer_retain(entry.buffer);
        return UnifiedBufferRef(entry.buffer, offset);
    }
    return {};
}

}  // namespace ggml::hrx
