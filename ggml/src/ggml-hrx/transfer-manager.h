#pragma once

#include "hrx_runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ggml::hrx {

struct transfer_manager_options {
    // These policy defaults allow recording, transfer, and reuse to overlap while bounding staging memory.
    size_t staging_page_size     = 8ull * 1024ull * 1024ull;
    size_t maximum_staging_bytes = 32ull * 1024ull * 1024ull;
};

struct transfer_manager_stats {
    uint64_t uploads            = 0;
    uint64_t downloads          = 0;
    uint64_t uploaded_bytes     = 0;
    uint64_t downloaded_bytes   = 0;
    uint64_t submissions        = 0;
    uint64_t consumer_waits     = 0;
    uint64_t producer_waits     = 0;
    uint64_t page_allocations   = 0;
    uint64_t page_reuses        = 0;
    uint64_t backpressure_waits = 0;
    uint64_t staging_bytes      = 0;

    std::string format() const;
};

// Host data is copied into owned staging memory before calls return; device consumers are ordered by timeline dependencies.
class transfer_manager {
  public:
    transfer_manager(hrx_device_t device, const transfer_manager_options & options = {});
    ~transfer_manager();
    transfer_manager(transfer_manager &&) noexcept;
    transfer_manager & operator=(transfer_manager &&) noexcept;
    transfer_manager(const transfer_manager &)             = delete;
    transfer_manager & operator=(const transfer_manager &) = delete;

    bool                valid() const;
    const std::string & initialization_error() const;

    // Several uploads may be batched until join() or flush() publishes them.
    std::string upload(const void * host_source, hrx_buffer_t destination, size_t destination_offset, size_t size);

    // Device operations use the transfer timeline; producer identifies compute work that writes source.
    std::string fill(hrx_buffer_t destination,
                     size_t       destination_offset,
                     size_t       size,
                     const void * pattern,
                     size_t       pattern_size);
    std::string copy(hrx_stream_t producer,
                     hrx_buffer_t source,
                     size_t       source_offset,
                     hrx_buffer_t destination,
                     size_t       destination_offset,
                     size_t       size);

    // Publishes pending transfers and inserts a device-side wait into consumer without waiting on the host.
    std::string join(hrx_stream_t consumer);

    // Makes the transfer stream wait for producer before overwriting a previously consumed allocation.
    std::string wait_for_producer(hrx_stream_t producer);

    // Publishes pending transfers without a consumer dependency.
    std::string flush(hrx_timeline_point_t * position = nullptr);

    // Readback waits for host visibility after joining a non-null producer into the transfer stream.
    std::string download(hrx_stream_t producer,
                         hrx_buffer_t source,
                         size_t       source_offset,
                         void *       host_destination,
                         size_t       size);

    // Waits for completion and recycles all staging pages; do not use this on dispatch paths.
    std::string synchronize();

    // Recycles pages whose transfer timeline points completed.
    std::string collect();

    transfer_manager_stats stats() const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace ggml::hrx
