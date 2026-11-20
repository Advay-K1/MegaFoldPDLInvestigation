"""Launch-overhead analysis of an nsys `cuda_gpu_trace` CSV for one training step.

For kernels on the same stream, in start order, the gap g_i = start_i - end_{i-1}:
  * g_i > 0  : GPU idle on that stream between two kernels (launch latency / CPU-bound / sync);
  * g_i < 0  : kernel i started before kernel i-1 finished -- only possible on one stream
               with Programmatic Dependent Launch (PDL overlap).
Memcpy/memset rows are kept as stream activity (they serialize the stream too).
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = list(csv.DictReader(open(a.csv)))
cols = rows[0].keys()
col = lambda pat: next(c for c in cols if re.fullmatch(pat, c))  # noqa: E731
C_START, C_DUR, C_NAME = col(r"Start \((ns|us)\)"), col(r"Duration \((ns|us)\)"), col(r"Name")
C_STRM = col(r"Strm|Stream.*")
scale = 1.0 if "(ns)" in C_START else 1e3


def family(name: str) -> str:
    nm = name.lower()
    if "_attn_" in nm:
        return "megafold_evoattention"
    if "swiglu" in nm:
        return "megafold_swiglu"
    if re.search(r"_(first|second)_(forward|backward)_kernel", nm):
        return "megafold_layernorm_linear"
    if nm.startswith("triton_") or "triton_poi" in nm or "triton_red" in nm or "triton_per" in nm:
        return "inductor_triton"
    if any(s in nm for s in ("gemm", "cutlass", "sm90_xmma", "cublas", "nvjet", "ampere_", "sm80_")):
        return "cublas_gemm"
    if "memcpy" in nm or "memset" in nm:
        return "memcpy_memset"
    return "other_aten"


by_stream = defaultdict(list)
for r in rows:
    if not r[C_DUR]:
        continue
    s = float(r[C_START]) * scale
    d = float(r[C_DUR]) * scale
    by_stream[r[C_STRM]].append((s, s + d, r[C_NAME]))

gaps_pos, overlaps, fam_count, fam_time = [], [], defaultdict(int), defaultdict(float)
pair_overlap = defaultdict(int)
busy = 0.0
t0 = min(s for v in by_stream.values() for s, _, _ in v)
t1 = max(e for v in by_stream.values() for _, e, _ in v)
for strm, evs in by_stream.items():
    evs.sort()
    for i, (s, e, nm) in enumerate(evs):
        f = family(nm)
        fam_count[f] += 1
        fam_time[f] += e - s
        busy += e - s
        if i:
            g = s - evs[i - 1][1]
            if g >= 0:
                gaps_pos.append(g)
            else:
                overlaps.append(-g)
                pair_overlap[f"{family(evs[i-1][2])} -> {f}"] += 1

# busy may double-count concurrent streams; also compute union of busy intervals
iv = sorted((s, e) for v in by_stream.values() for s, e, _ in v)
union, cs, ce = 0.0, *iv[0]
for s, e in iv[1:]:
    if s > ce:
        union += ce - cs
        cs, ce = s, e
    else:
        ce = max(ce, e)
union += ce - cs

q = lambda xs, p: (sorted(xs)[int(p * (len(xs) - 1))] if xs else 0.0)  # noqa: E731
res = dict(
    num_streams=len(by_stream),
    num_gpu_ops=sum(len(v) for v in by_stream.values()),
    span_ms=(t1 - t0) / 1e6,
    gpu_busy_union_ms=union / 1e6,
    gpu_idle_ms=(t1 - t0 - union) / 1e6,
    gpu_idle_frac=(t1 - t0 - union) / (t1 - t0),
    gaps_count=len(gaps_pos),
    gap_total_ms=sum(gaps_pos) / 1e6,
    gap_median_us=(statistics.median(gaps_pos) / 1e3) if gaps_pos else 0.0,
    gap_p90_us=q(gaps_pos, 0.9) / 1e3,
    gap_p99_us=q(gaps_pos, 0.99) / 1e3,
    gaps_under_5us=sum(g < 5e3 for g in gaps_pos),
    pdl_overlaps_count=len(overlaps),
    pdl_overlap_total_ms=sum(overlaps) / 1e6,
    pdl_overlap_median_us=(statistics.median(overlaps) / 1e3) if overlaps else 0.0,
    pdl_overlap_by_pair=dict(sorted(pair_overlap.items(), key=lambda kv: -kv[1])),
    kernel_count_by_family=dict(fam_count),
    kernel_time_ms_by_family={k: v / 1e6 for k, v in fam_time.items()},
)
json.dump(res, open(a.out, "w"), indent=2)
print(json.dumps(res, indent=2))
