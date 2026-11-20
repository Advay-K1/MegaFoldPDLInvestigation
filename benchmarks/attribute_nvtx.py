"""Attribute every GPU kernel of an nsys trace to a training-step phase / module by CPU launch time.

nsys's own `nvtx_kern_sum` attributes a kernel to an NVTX range only if the *same thread* pushed the
range and launched the kernel; autograd runs backward on its own thread, so that report misses almost
all backward kernels. Here each kernel is matched (via correlationId) to the CPU timestamp of its launch
API call, and that timestamp is placed inside the NVTX ranges by time, regardless of thread. The main
thread is blocked inside `loss.backward()` for the whole backward, so the "backward" range covers it.

    python benchmarks/attribute_nvtx.py results/report2/nsys/<run>.sqlite --out attribution.json
(the .sqlite is produced by `nsys stats` next to the .nsys-rep)
"""

import argparse
import collections
import json
import sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("sqlite")
ap.add_argument("--out", required=True)
a = ap.parse_args()

db = sqlite3.connect(a.sqlite)
S = dict(db.execute("select id, value from StringIds"))
rng = sorted((s, e, t if t else S.get(tid, "?"))
             for s, e, t, tid in db.execute("select start, end, text, textId from NVTX_EVENTS where end is not null"))
launch = dict(db.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))
kern = list(db.execute("select correlationId, shortName, start, end from CUPTI_ACTIVITY_KIND_KERNEL"))
TOP = {"backward", "clip+optimizer"}


def phase(t):
    for s, e, name in rng:
        if name in TOP and s <= t <= e:
            return name
    return "forward (and loss)"


def innermost(t):
    best = None
    for s, e, name in rng:
        if s > t:
            break
        if s <= t <= e and (best is None or s >= best[0]):
            best = (s, e, name)
    return best[2] if best else "(outside ranges)"


ph, phT, fwd = collections.Counter(), collections.Counter(), collections.Counter()
for cid, sn, ks, ke in kern:
    t = launch.get(cid)
    p = "(cuda-graph replay)" if t is None else phase(t)
    ph[p] += 1
    phT[p] += ke - ks
    if p == "forward (and loss)":
        fwd[innermost(t)] += 1

res = dict(kernels=len(kern),
           by_phase={p: dict(kernels=n, gpu_ms=phT[p] / 1e6) for p, n in ph.most_common()},
           forward_by_innermost_module=fwd.most_common(30))
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
