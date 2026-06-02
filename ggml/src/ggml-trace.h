#pragma once

// P058 profiling support: emit per-kernel / per-op trace data so the HIP and HRX backends
// can be compared kernel-by-kernel for the same model (see compare.py).
//
// Shared by two backends with different capture paths:
//   * HRX writes CSV rows directly (one per hrx_stream_dispatch) — rocprofv3 can't see HRX.
//   * HIP emits roctx range labels; rocprofv3 supplies the kernel name + GPU timing, and
//     compare.py joins its kernels to these ranges.
// Both encode the same ggml semantics (node, op, fused set, shapes) so compare.py parses one
// format. Runtime-gated by env: GGML_TRACE_CSV=<path> (HRX output) and/or GGML_PROFILE=1 (HIP
// roctx). When unset, every entry point below is a cheap no-op.

#include "ggml.h"

#include <cstdint>
#include <string>

namespace ggml_trace {

// True if HIP roctx markers should be emitted (GGML_PROFILE or GGML_TRACE_CSV set). Cached.
bool enabled();

// True if the HRX CSV sink is configured (GGML_TRACE_CSV set). Cached. The HRX wrapper gates its
// per-kernel flush+sync on this so non-profiling runs keep their normal async batching.
bool hrx_enabled();

// Monotonic timestamp in microseconds, relative to the first call.
double now_us();

// Encode one tensor's shape as "<type>@<ne0>x<ne1>x<ne2>x<ne3>", e.g. "q8_0@4096x1x1x1".
// '@'/'x' are collision-free with ggml node names (verified: names lack @, x is a letter but
// type names and the dim list never need name-splitting).
std::string pack_tensor(const struct ggml_tensor * t);

// HRX: append one CSV row for a kernel launch. Writes the header on first call; thread-safe;
// flushes per row. `group`/`group_n` are the ggml nodes this kernel covers (1 = unfused).
// No-op unless hrx_enabled().
void emit_hrx_row(uint64_t graph_eval,
                  const struct ggml_tensor *         node,    // op-group starting node (join key)
                  const char *                       kernel,
                  const struct ggml_tensor * const * group,
                  int                                group_n,
                  double t_enqueue_us, double t_gpu_us);

// HIP: append one op-group metadata row to the CSV (no kernel/timing — rocprofv3 supplies those).
// A roctx range labelled "p058corr=<corr>" encloses the kernels; `corr` is repeated here so
// compare.py can join rocprof's kernels (nested in that range by timestamp) to this op-group.
// No-op unless hrx_enabled() (the same GGML_TRACE_CSV sink is used for the HIP meta file).
void emit_hip_meta(uint64_t graph_eval, uint64_t corr,
                   const struct ggml_tensor *         node,
                   const struct ggml_tensor * const * group,
                   int                                group_n);

} // namespace ggml_trace
