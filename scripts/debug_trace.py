import gzip, json, collections
from pathlib import Path

path = sorted(Path("/tmp/jax-trace").rglob("*.trace.json.gz"))[-1]
with gzip.open(path) as f:
    trace = json.load(f)

events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
by_name = collections.defaultdict(lambda: [0, 0.0])
for e in events:
    rec = by_name[e.get("name", "?")]
    rec[0] += 1
    rec[1] += e.get("dur", 0) / 1e3   # ms

print(f"{'op':<50} {'count':>7} {'total_ms':>10}")
for name, (n, ms) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f"{name[:50]:<50} {n:>7} {ms:>10.2f}")




@partial(jax.shard_map, mesh=gpu_mesh,
         in_specs=(jax.P(gpu_axis),), out_specs=jax.P(gpu_axis), check_vma=False)
def noop_pos(x):
    return x * 2.0

t_pos = bench(noop_pos, x)
print(f"shard_map positional: {t_pos*1e3:8.3f} ms")
