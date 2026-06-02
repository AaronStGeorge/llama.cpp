#!/usr/bin/env python3
"""P058 compare.py — align HIP vs HRX per-kernel traces by ggml node and classify each
HRX-vs-HIP difference as *fusion-divergence* or *kernel-speed*.

Inputs (all CSV; both backends were run on the same model with the same --flash-attn):
  --hrx hrx.csv            : the HRX in-backend emitter (one row per kernel, with GPU timing).
  --hip-meta hip_meta.csv  : the HIP in-backend emitter (one row per ggml op-group, no timing).
  --hip-trace DIR          : rocprofv3 -d output dir (kernel_trace.csv + marker_api_trace.csv).

The HIP timing/kernel names come from rocprofv3; the op-group semantics (node, op, fused set,
shapes) come from hip_meta.csv; they are joined via Correlation_Id -> our "p058corr=N" label.

This is the deterministic comparison table only — interpreting the rows (why-slow, fixes) is the
deferred Phase-2 work.
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

# ----------------------------------------------------------------------------- parsing

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class OpGroup:
    """One (graph_eval, node) execution unit: the op (or fused op-group) + its kernels."""
    __slots__ = ("node", "op", "fnodes", "srcs", "dst", "kernels")

    def __init__(self, node, op, fnodes, srcs, dst):
        self.node = node
        self.op = op
        self.fnodes = fnodes            # tuple of ggml node names this unit covers
        self.srcs = srcs
        self.dst = dst
        self.kernels = []               # list of (kernel_name, gpu_us)

    @property
    def total_us(self):
        return sum(us for _, us in self.kernels)


def parse_hrx(path):
    """HRX CSV -> {graph_eval: {node: OpGroup}}. Rows are per-kernel; group by (eval, node)."""
    out = defaultdict(dict)
    for r in read_csv(path):
        ev = int(r["graph_eval"])
        key = (r["node"], r["op"])     # node name alone is NOT unique (e.g. Qcur = matmul then rope)
        g = out[ev].get(key)
        if g is None:
            g = OpGroup(r["node"], r["op"],
                        tuple(r["fused_nodes"].split(";")) if r["fused_nodes"] else (r["node"],),
                        r["srcs"], r["dst"])
            out[ev][key] = g
        g.kernels.append((r["kernel"], float(r["t_gpu_us"])))
    return out


def parse_hip(meta_path, trace_dir):
    """hip_meta.csv + rocprofv3 trace -> {graph_eval: {node: OpGroup}} with kernels from rocprof."""
    # 1. rocprof correlation: our corr N (from the roctx range "Function") -> [(kernel_name, gpu_us)]
    def find(suffix):
        hits = glob.glob(os.path.join(trace_dir, "**", f"*{suffix}"), recursive=True)
        if not hits:
            sys.exit(f"compare.py: no *{suffix} under {trace_dir}")
        return sorted(hits)[-1]

    markers = read_csv(find("_marker_api_trace.csv"))
    rcorr_to_n = {}                     # rocprof Correlation_Id -> our N
    for m in markers:
        mobj = re.fullmatch(r"p058corr=(\d+)", m["Function"])
        if mobj:
            rcorr_to_n[m["Correlation_Id"]] = int(mobj.group(1))

    n_kernels = defaultdict(list)       # our N -> [(kernel_name, gpu_us)]
    unattributed = 0
    for k in read_csv(find("_kernel_trace.csv")):
        n = rcorr_to_n.get(k["Correlation_Id"])
        if n is None:
            unattributed += 1           # setup kernels outside any p058 range (V7)
            continue
        us = (float(k["End_Timestamp"]) - float(k["Start_Timestamp"])) / 1000.0
        n_kernels[n].append((k["Kernel_Name"], us))

    # 2. op-group semantics from hip_meta.csv, keyed by our corr N
    out = defaultdict(dict)
    for r in read_csv(meta_path):
        ev = int(r["graph_eval"])
        n = int(r["corr"])
        key = (r["node"], r["op"])
        g = out[ev].get(key)
        if g is None:
            g = OpGroup(r["node"], r["op"],
                        tuple(r["fused_nodes"].split(";")) if r["fused_nodes"] else (r["node"],),
                        r["srcs"], r["dst"])
            out[ev][key] = g
        g.kernels.extend(n_kernels.get(n, []))
    return out, unattributed


# ----------------------------------------------------------------------------- aggregation

def steady_evals(per_eval, warmup):
    evals = sorted(per_eval)
    return evals[warmup:] if len(evals) > warmup else evals


def aggregate(per_eval, warmup):
    """Mean total_us per node over steady-state evals; keep one representative group for semantics."""
    evals = steady_evals(per_eval, warmup)
    sums, counts, rep = defaultdict(float), defaultdict(int), {}
    for ev in evals:
        for node, g in per_eval[ev].items():
            sums[node] += g.total_us
            counts[node] += 1
            rep.setdefault(node, g)
    return {node: (sums[node] / counts[node], rep[node]) for node in sums}, len(evals)


# ----------------------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hrx", help="HRX emitter CSV")
    ap.add_argument("--hip-meta", help="HIP emitter op-group CSV")
    ap.add_argument("--hip-trace", help="rocprofv3 -d output directory")
    ap.add_argument("--warmup", type=int, default=2, help="decode steps to skip on each side")
    ap.add_argument("--floor-us", type=float, default=5.0,
                    help="flag rows whose HRX time is within this of the per-kernel sync floor")
    args = ap.parse_args()

    hrx = aggregate(parse_hrx(args.hrx), args.warmup)[0] if args.hrx else {}
    hip, unattr = ({}, 0)
    if args.hip_meta and args.hip_trace:
        per_eval, unattr = parse_hip(args.hip_meta, args.hip_trace)
        hip = aggregate(per_eval, args.warmup)[0]

    if not hrx or not hip:
        # single-backend summary (e.g. before the other run exists)
        only = hrx or hip
        which = "HRX" if hrx else "HIP"
        print(f"\n=== {which}-only summary ({len(only)} op-groups) ===")
        for key, (us, g) in sorted(only.items(), key=lambda kv: -kv[1][0])[:40]:
            print(f"  {us:9.2f}us  {g.op:<16} {g.node:<28} [{','.join(g.fnodes)}]  kernels={len(g.kernels)}")
        if unattr:
            print(f"  ({unattr} rocprof kernels outside any p058 range — setup/unattributed)")
        return

    # ---- align by node name; classify ----
    rows = []
    for key in sorted(set(hrx) | set(hip)):
        node, op = key
        h = hrx.get(key)
        p = hip.get(key)
        if h and p:
            same_fuse = set(h[1].fnodes) == set(p[1].fnodes)
            cause = "kernel-speed" if same_fuse else "fusion-divergence"
            rows.append((h[0] - p[0], node, op, p[0], h[0], cause, p[1], h[1]))
        elif h:           # HRX ran this op; HIP fused it away (or ran it elsewhere)
            rows.append((h[0], node, op, 0.0, h[0], "fusion-divergence(HIP-absorbed)", None, h[1]))
        else:             # HIP ran this op; HRX fused it away
            rows.append((-p[0], node, op, p[0], 0.0, "fusion-divergence(HRX-absorbed)", p[1], None))

    rows.sort(key=lambda r: -r[0])
    print(f"\n{'delta_us':>10} {'hip_us':>9} {'hrx_us':>9}  {'op':<14} {'node':<26} cause")
    print("-" * 100)
    for delta, node, op, hip_us, hrx_us, cause, pg, hg in rows:
        flag = " *floor" if hg and hg.total_us <= args.floor_us else ""
        print(f"{delta:10.2f} {hip_us:9.2f} {hrx_us:9.2f}  {op:<14} {node:<26} {cause}{flag}")

    # ---- summary ----
    tot_hrx = sum(h[0] for h in hrx.values())
    tot_hip = sum(p[0] for p in hip.values())
    fus = sum(r[0] for r in rows if r[0] > 0 and r[5].startswith("fusion"))
    spd = sum(r[0] for r in rows if r[0] > 0 and r[5] == "kernel-speed")
    print("\n=== summary (mean per decode step) ===")
    print(f"  total HRX gpu = {tot_hrx:10.2f}us   total HIP gpu = {tot_hip:10.2f}us   "
          f"HRX-HIP = {tot_hrx - tot_hip:+.2f}us")
    deficit = fus + spd
    if deficit > 0:
        print(f"  of HRX's deficit (Σ positive deltas = {deficit:.2f}us): "
              f"fusion-divergence {100*fus/deficit:.0f}%, kernel-speed {100*spd/deficit:.0f}%")
    if unattr:
        print(f"  ({unattr} rocprof kernels were outside any p058 range — setup/unattributed)")
    # by-op rollup
    by_op = defaultdict(lambda: [0.0, 0.0])
    for key in set(hrx) | set(hip):
        if key in hrx:
            by_op[key[1]][0] += hrx[key][0]
        if key in hip:
            by_op[key[1]][1] += hip[key][0]
    print(f"\n  {'op':<16} {'hrx_us':>10} {'hip_us':>10} {'delta':>10}")
    for op, (hu, pu) in sorted(by_op.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
        print(f"  {op:<16} {hu:10.2f} {pu:10.2f} {hu - pu:+10.2f}")


if __name__ == "__main__":
    main()
