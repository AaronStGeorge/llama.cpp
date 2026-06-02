// P058 profiling trace emitter. See ggml-trace.h.

#include "ggml-trace.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace ggml_trace {

// ---- env-cached configuration -------------------------------------------------------------

static const char * env_or_null(const char * name) {
    const char * v = std::getenv(name);
    return (v && v[0]) ? v : nullptr;
}

static const std::string & csv_path() {
    static const std::string path = [] {
        if (const char * p = env_or_null("GGML_TRACE_CSV")) {
            return std::string(p);
        }
        return std::string();
    }();
    return path;
}

bool hrx_enabled() {
    static const bool on = !csv_path().empty();
    return on;
}

bool enabled() {
    static const bool on = env_or_null("GGML_PROFILE") != nullptr || !csv_path().empty();
    return on;
}

double now_us() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::micro>(clock::now() - t0).count();
}

// ---- shape / op-group encoding (shared by HRX CSV and HIP roctx label) ---------------------

std::string pack_tensor(const struct ggml_tensor * t) {
    if (!t) {
        return "none";
    }
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s@%lldx%lldx%lldx%lld",
                  ggml_type_name(t->type),
                  (long long) t->ne[0], (long long) t->ne[1],
                  (long long) t->ne[2], (long long) t->ne[3]);
    return std::string(buf);
}

// A node's stable identifier for the fused-set field: its name, or the op name if unnamed.
static std::string node_id(const struct ggml_tensor * t) {
    if (t->name[0]) {
        return std::string(t->name);
    }
    return std::string(ggml_op_name(t->op));
}

static bool in_group(const struct ggml_tensor * t,
                     const struct ggml_tensor * const * group, int group_n) {
    for (int i = 0; i < group_n; ++i) {
        if (group[i] == t) {
            return true;
        }
    }
    return false;
}

// ';'-joined node ids of the op-group (the fusion composition).
static std::string fused_nodes(const struct ggml_tensor * const * group, int group_n) {
    std::string out;
    for (int i = 0; i < group_n; ++i) {
        if (i) out += ';';
        out += node_id(group[i]);
    }
    return out;
}

// ','-joined shapes of the group's *external* inputs (srcs not produced inside the group),
// deduplicated — i.e. what the (possibly fused) kernel actually reads.
static std::string pack_srcs(const struct ggml_tensor * const * group, int group_n) {
    std::vector<const struct ggml_tensor *> seen;
    std::string out;
    for (int i = 0; i < group_n; ++i) {
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const struct ggml_tensor * s = group[i]->src[j];
            if (!s || in_group(s, group, group_n)) {
                continue;
            }
            bool dup = false;
            for (const auto * p : seen) { if (p == s) { dup = true; break; } }
            if (dup) continue;
            seen.push_back(s);
            if (!out.empty()) out += ',';
            out += pack_tensor(s);
        }
    }
    return out;
}

// The group's output = its terminal node (last in topo order).
static std::string pack_dst(const struct ggml_tensor * const * group, int group_n) {
    return pack_tensor(group[group_n - 1]);
}

// ---- HRX CSV sink -------------------------------------------------------------------------

static std::mutex   g_mu;
static std::FILE *  g_csv     = nullptr;
static bool         g_csv_bad = false;
static int          g_header  = 0;   // 0=none, 1=hrx, 2=hip (whichever emitter opens the file first)

static std::string csv_quote(const std::string & s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += '"';   // double internal quotes
        out += c;
    }
    out += '"';
    return out;
}

// Open the CSV once and write `header` on first use. Caller holds g_mu. Returns false on error.
static bool csv_ready_locked(int kind, const char * header) {
    if (g_csv_bad) {
        return false;
    }
    if (!g_csv) {
        g_csv = std::fopen(csv_path().c_str(), "w");
        if (!g_csv) {
            std::fprintf(stderr, "ggml-trace: cannot open GGML_TRACE_CSV=%s\n", csv_path().c_str());
            g_csv_bad = true;
            return false;
        }
    }
    if (g_header == 0) {
        std::fprintf(g_csv, "%s\n", header);
        g_header = kind;
    }
    return true;
}

void emit_hrx_row(uint64_t graph_eval,
                  const struct ggml_tensor *         node,
                  const char *                       kernel,
                  const struct ggml_tensor * const * group,
                  int                                group_n,
                  double t_enqueue_us, double t_gpu_us) {
    if (!hrx_enabled() || group_n <= 0) {
        return;
    }
    const std::string row_node   = node_id(node);
    const char *      row_op     = ggml_op_desc(node);
    const std::string row_fnodes = fused_nodes(group, group_n);
    const std::string row_srcs   = pack_srcs(group, group_n);
    const std::string row_dst    = pack_dst(group, group_n);

    std::lock_guard<std::mutex> lock(g_mu);
    if (!csv_ready_locked(1,
            "backend,graph_eval,node,op,kernel,fused_span,fused_nodes,srcs,dst,t_enqueue_us,t_gpu_us")) {
        return;
    }
    std::fprintf(g_csv, "hrx,%llu,%s,%s,%s,%d,%s,%s,%s,%.3f,%.3f\n",
                 (unsigned long long) graph_eval,
                 csv_quote(row_node).c_str(),
                 csv_quote(row_op ? row_op : "").c_str(),
                 csv_quote(kernel ? kernel : "").c_str(),
                 group_n,
                 csv_quote(row_fnodes).c_str(),
                 csv_quote(row_srcs).c_str(),
                 csv_quote(row_dst).c_str(),
                 t_enqueue_us, t_gpu_us);
    std::fflush(g_csv);   // durable across a crash mid-run (atexit may not fire)
}

// ---- HIP roctx label ----------------------------------------------------------------------

void emit_hip_meta(uint64_t graph_eval, uint64_t corr,
                   const struct ggml_tensor *         node,
                   const struct ggml_tensor * const * group,
                   int                                group_n) {
    if (csv_path().empty() || group_n <= 0) {
        return;
    }
    const std::string row_node   = node_id(node);
    const char *      row_op     = ggml_op_desc(node);
    const std::string row_fnodes = fused_nodes(group, group_n);
    const std::string row_srcs   = pack_srcs(group, group_n);
    const std::string row_dst    = pack_dst(group, group_n);

    std::lock_guard<std::mutex> lock(g_mu);
    if (!csv_ready_locked(2,
            "backend,graph_eval,corr,node,op,fused_span,fused_nodes,srcs,dst")) {
        return;
    }
    std::fprintf(g_csv, "hip,%llu,%llu,%s,%s,%d,%s,%s,%s\n",
                 (unsigned long long) graph_eval,
                 (unsigned long long) corr,
                 csv_quote(row_node).c_str(),
                 csv_quote(row_op ? row_op : "").c_str(),
                 group_n,
                 csv_quote(row_fnodes).c_str(),
                 csv_quote(row_srcs).c_str(),
                 csv_quote(row_dst).c_str());
    std::fflush(g_csv);
}

} // namespace ggml_trace
