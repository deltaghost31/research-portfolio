import dimod, networkx as nx
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.preprocessing import SpinReversalTransformComposite
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def encode_repetition3(bqm, chain_strength):
    enc = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
    mapping = {}
    for v in bqm.variables:
        reps = [f"{v}_r{i}" for i in range(3)]
        mapping[v] = reps
        for p in reps:
            enc.add_variable(p, bqm.linear[v])
        for i in range(3):
            for j in range(i+1, 3):
                enc.add_interaction(reps[i], reps[j], -chain_strength)
    for (u, v), J in bqm.quadratic.items():
        for pu in mapping[u]:
            for pv in mapping[v]:
                enc.add_interaction(pu, pv, J)
    return enc, mapping

def majority_vote(sample, mapping):
    return {log: 1 if sum(sample[p] for p in chain) >= 0 else -1
            for log, chain in mapping.items()}

def cut_size(sample, edges):
    return sum(sample[u] != sample[v] for u, v in edges)

def summarize_breaks(ss):
    ctx = ss.info.get("embedding_context", {})
    if "chain_break_fraction" in ctx:
        frac = ctx["chain_break_fraction"]
    elif "chain_break_fraction" in ss.record.dtype.names:
        frac = ss.record["chain_break_fraction"]
    else:
        return None, None
    return float(frac.mean()), int((frac == 0).sum())

# ─────────────────────────────────────────────────────────────
# 1. Build random Max-Cut (n=50, p=0.8)
# ─────────────────────────────────────────────────────────────
n, p, seed = 50, 0.8, 123
G = nx.gnp_random_graph(n, p, seed=seed)
for u, v in G.edges:
    G[u][v]["weight"] = 1

bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
for u, v in G.edges:
    bqm.add_interaction(u, v, +1)

# ─────────────────────────────────────────────────────────────
# 2. Sampler & configs (1 µs anneal)
# ─────────────────────────────────────────────────────────────
base = EmbeddingComposite(DWaveSampler())

configs = [
    {"label":"Raw QPU",   "reads":1000, "gauges":0, "rep":False},
    {"label":"Gauge QPU", "reads":1000, "gauges":4, "rep":False},
    {"label":"Rep-3",     "reads":2000, "gauges":0, "rep":True},
    {"label":"Rep-3+2g",  "reads":2000, "gauges":2, "rep":True},
]

results = []

# ─────────────────────────────────────────────────────────────
# 3. Sweep annealing_time for all four methods at cs=16
# ─────────────────────────────────────────────────────────────
anneal_times = [1, 5, 10, 20]  # μs

results = []
for t in anneal_times:
    print(f"\n=== annealing_time = {t} μs ===")
    for cfg in configs:
        label, reads, gauges, use_rep = cfg.values()

        # 1) choose or encode problem
        if use_rep:
            problem, mapping = encode_repetition3(bqm, chain_strength=16)
        else:
            problem, mapping = bqm, None

        # 2) wrap sampler in gauges if needed
        sampler = SpinReversalTransformComposite(base) if gauges else base

        # 3) sample with current anneal time
        params = {"num_reads": reads, "annealing_time": t}
        if gauges:
            params["num_spin_reversal_transforms"] = gauges

        ss = sampler.sample(problem, **params)

        # 4) decode best sample
        phys    = ss.first.sample
        logical = majority_vote(phys, mapping) if use_rep else phys

        # 5) compute metrics
        best_cut, energy = cut_size(logical, G.edges), ss.first.energy
        brk, perfect     = summarize_breaks(ss)

        print(f"{label:12} | reads={reads:4} | best_cut={best_cut:3} "
              f"| energy={energy:6} | break-frac={brk:.3f} | perfect={perfect}")

        # also collect into table
        results.append({
            "anneal_time":  t,
            "method":       label,
            "reads":        reads,
            "gauges":       gauges,
            "repetition":   use_rep,
            "best_cut":     best_cut,
            "energy":       energy,
            "break_frac":   brk,
            "perfect_reads":perfect
        })

# ─────────────────────────────────────────────────────────────
# 4. Show consolidated results table
# ─────────────────────────────────────────────────────────────
import pandas as pd
df = pd.DataFrame(results)
print("\nFull results:")
print(df.to_string(index=False))
