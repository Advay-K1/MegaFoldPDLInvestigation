"""Classify every same-stream boundary of one traced training step by (primary family -> secondary family),
by the API the secondary was launched with, and by whether the secondary was already enqueued by the CPU
when the primary finished ("queued" -- the only case in which any PDL, same- or cross-language, can help).

    python benchmarks/boundary_census.py results/nsys/pdl_all.sqlite --out results/report3/census_pdl_all.json
"""
import argparse, json, re, sqlite3, statistics
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("sqlite")
ap.add_argument("--out", required=True)
a = ap.parse_args()
c = sqlite3.connect(a.sqlite)
S = dict(c.execute("select id, value from StringIds"))


def family(nm: str) -> str:
    n = nm.lower()
    if "_attn_" in n or "swiglu" in n or re.search(r"_(first|second)_(forward|backward)_kernel", n):
        return "triton_megafold"
    if n.startswith("triton_"):
        return "triton_inductor"
    if any(s in n for s in ("gemm", "cutlass", "xmma", "cublas", "nvjet", "ampere_", "sm80_", "sm90_")):
        return "cublas"
    if "cudnn" in n or "flash" in n or "fmha" in n:
        return "cudnn_flash"
    if "at::native" in n or "at_cuda_detail" in n or n.startswith("void "):
        return "aten"
    return "other"


rt = {cid: (S.get(nid, "?"), st, en) for cid, nid, st, en in
      c.execute("select correlationId, nameId, start, end from CUPTI_ACTIVITY_KIND_RUNTIME")}
ev = []  # (stream, start, end, family, launch_api, cpu_enqueue_end)
for st, en, strm, cid, nid in c.execute(
        "select start,end,streamId,correlationId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL"):
    api, _, cpu_end = rt.get(cid, ("?", 0, 0))
    ev.append((strm, st, en, family(S.get(nid, "?")), api.split("_v")[0], cpu_end))
for tbl, fam in (("CUPTI_ACTIVITY_KIND_MEMSET", "memset"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy")):
    for st, en, strm, cid in c.execute(f"select start,end,streamId,correlationId from {tbl}"):
        api, _, cpu_end = rt.get(cid, ("?", 0, 0))
        ev.append((strm, st, en, fam, api.split("_v")[0], cpu_end))

launch_api = defaultdict(Counter)
for e in ev:
    launch_api[e[3]][e[4]] += 1

by = defaultdict(list)
for e in ev:
    by[e[0]].append(e)
pairs = Counter(); queued = Counter(); gap_q = defaultdict(list); gap_all = defaultdict(list)
for strm, L in by.items():
    L.sort(key=lambda e: e[1])
    for p, s in zip(L, L[1:]):
        k = f"{p[3]} -> {s[3]}"
        pairs[k] += 1
        g = s[1] - p[2]
        gap_all[k].append(g)
        if s[5] and s[5] < p[2]:          # CPU had finished enqueuing the secondary before the primary ended
            queued[k] += 1
            gap_q[k].append(g)

# reachability buckets: what would have to be true for PDL to act on this boundary?
def bucket(p, s):
    if not (s[5] and s[5] < p[2]):
        return "CPU-bound: secondary not yet enqueued when primary ended"
    if p[3] in ("memset", "memcpy") or s[3] in ("memset", "memcpy"):
        return "queued, memcpy/memset involved (never PDL)"
    if s[3].startswith("triton"):
        return "queued, secondary Triton (reachable today)"
    if s[3] == "aten":
        return "queued, secondary ATen (needs cross-language PDL)"
    return "queued, secondary cuBLAS/other library (closed source)"
buckets = defaultdict(lambda: [0, 0.0])
for strm, L in by.items():
    for p, s in zip(L, L[1:]):
        b = buckets[bucket(p, s)]
        b[0] += 1; b[1] += max(0, s[1] - p[2]) / 1e6
N = sum(pairs.values()); NQ = sum(queued.values())
tri = lambda f: f.startswith("triton")
def frac(pred):
    return (sum(v for k, v in pairs.items() if pred(*k.split(" -> "))) / N,
            sum(v for k, v in queued.items() if pred(*k.split(" -> "))) / max(NQ, 1))
kern = lambda f: f not in ("memset", "memcpy")
res = dict(
    trace=a.sqlite, boundaries=N, queued_boundaries=NQ, queued_frac=NQ / N,
    gap_ms_total=sum(sum(v) for v in gap_all.values() if v) / 1e6,
    gap_ms_on_queued=sum(sum(max(0, x) for x in v) for v in gap_q.values()) / 1e6,
    median_gap_us_queued=statistics.median([x for v in gap_q.values() for x in v]) / 1e3 if NQ else None,
    scenario_frac_all_vs_queued={
        "triton->triton (Report 1 eligibility)": frac(lambda p, s: tri(p) and tri(s)),
        "any kernel -> triton (implicit trigger, today's launches)": frac(lambda p, s: kern(p) and tri(s)),
        "triton -> aten (needs PDL-aware ATen secondary)": frac(lambda p, s: tri(p) and s == "aten"),
        "aten -> aten": frac(lambda p, s: p == "aten" and s == "aten"),
        "any kernel -> any kernel (every kernel PDL-aware)": frac(lambda p, s: kern(p) and kern(s)),
    },
    reachability={k: dict(boundaries=v[0], gap_ms=round(v[1], 2)) for k, v in buckets.items()},
    pairs_top=dict(pairs.most_common(25)),
    queued_top=dict(queued.most_common(25)),
    launch_api_by_family={k: dict(v) for k, v in launch_api.items()},
)
import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("pairs_top", "queued_top")}, indent=1))
print("top pairs:", json.dumps(dict(pairs.most_common(10)), indent=1))
print("top queued:", json.dumps(dict(queued.most_common(10)), indent=1))
