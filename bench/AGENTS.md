# bench/AGENTS.md

Bench guidance for AI coding agents. Repository-wide rules are in the root `AGENTS.md`.

## Comparing against a bench run: the metric definitions

Benchmark communication backend selection defaults to **vLLM's own choice,
not NCCL-only**. Do not force all-reduce environment values. Separately retain
the benchmark's `fuse_allreduce_rms=False` and `fuse_gemm_comms=False`, recording
both under `engine_kwargs.compilation_config`: changing backend selection does
not authorize enabling these two compiler fusions. Ordinary compute compilation
and CUDA graphs stay enabled; model-native fused operations are not governed
universally by these switches. `--disable-custom-all-reduce` is an explicit opt-in, off by default;
the shell template exposes it as `DISABLE_CUSTOM_ALL_REDUCE=1`. This flag alone
does not disable other optimized or model-native fused paths. Record the argument
in `engine_kwargs`, inherited communicator variables (including unset values)
in `hardware.all_reduce_environment`, and resolved parallel/compiler settings
in `resolved_config`. Configuration is not proof of actual collective dispatch.
Do not add source patches to force NCCL as the default. Historical NCCL-only
examples retain their recorded settings and are not relabelled. Simulator-side
NCCL calibration does not select the benchmark backend; keep every comparison's
network model, recorded backend and hardware environment matched.

Snapshot resolved engine settings and KV capacity before shutdown. Normal
serving runs write `engine_start.json` before submitting requests, exclusively;
use a fresh output directory. This startup snapshot is not a completed run.
`bench validate` rejects known partial, aborted, resolve-only and perturbed
diagnostic inputs while retaining legacy request/timeseries compatibility.
`--kv-cache-memory-bytes` fixes an explicit per-GPU budget, but comparisons must
still verify resolved block counts. Dummy weights do not establish equivalent
backend selection, routing or scheduling for a real checkpoint.

`bench/core/kv_cache_metadata.py` observes native READY payloads only during the
standalone synchronous constructor. Restore the callback before decoding or
request submission; add no worker RPC, request hook or simulated latency.
The reviewed vLLM 0.31 client sums connected engine block counts and retains the
last READY block size. Preserve those raw fields; their product is not generally
an aggregate token capacity. Match each instance to its directly observed DP
rank under complete `kv_cache.capacity.per_engine`, never an inferred division.
Unsupported callback/response APIs, replaced callbacks, client/topology mismatches
or incomplete observations remain explicit unknowns. Record the observed version,
callback signature and response fields; exact version equality is not required.
These startup scheduler-pool facts neither measure physical per-GPU allocation
nor track later elastic changes. Do not relabel historical benchmark metadata.

Use `bench/core/validate.py::_bench_latencies`, the implementation shared by
`bench validate` and committed summaries, rather than ad hoc metric formulas.

`requests.jsonl` mixes wall-clock `arrival_time` with monotonic lifecycle
timestamps. Align arrival to the engine clock before computing milliseconds:

```text
offset  = min over requests of (queued_ts - arrival_time)
arrival = arrival_time + offset
TTFT    = (first_token_ts - arrival) * 1000
TPOT    = (last_token_ts - first_token_ts) / (output_toks - 1) * 1000
latency = (last_token_ts - arrival) * 1000
```

The offset is an estimate: if pickup delay is nonnegative, its residual bias
is the minimum pickup delay in that run, not a guaranteed fixed tolerance.
When conversion is unavailable, the validator uses its documented queued-time
fallback. TPOT is defined only for outputs with more than one token.

Using queued time instead of arrival omits frontend-to-engine waiting from
TTFT and end-to-end latency. It does not affect token-to-token intervals.
Neither request latency nor `Total clocks (ns)` is simulator host wall time.

Report all fifteen TTFT, TPOT and latency statistics, including every fixed
repeat when repeated references are available. The simulator's deterministic
regression digests establish reproducibility, not benchmark accuracy. Engine
timing variation does not justify omitting a failing TTFT percentile or
replacing it with run span. Preserve recorded vLLM references when refreshing
simulator-side examples, and explain intentional changes in either direction.
