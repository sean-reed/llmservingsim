# serving/AGENTS.md

Simulator guidance for AI coding agents. Repository-wide rules are in the root `AGENTS.md`; profiler-side contracts are in `profiler/AGENTS.md`.

## Architecture Patterns

### Vocab-parallel target endpoints

Two all-reduces per dense decoder block are not the complete TP forward.
For the ordinary vLLM 0.28 target path, `VocabParallelEmbedding.forward`
reduces the local-vocabulary embedding output, and
`LogitsProcessor._get_logits` gathers vocabulary-sharded logits before
sampling. `_shared_tp_collective` emits these once, gated by the catalog
binding and shared prologue/head placement; it does not infer them for MTP
or alternative heads. The formulas and scope are documented in
`docs/docs/simulator/parallelism-mechanics.md`.

Vocabulary padding precedes TP division. The embedding and head weights
remain per-rank, logits use the configured head dtype, and the sampler reads
the gathered full vocabulary. Per-sequence tensor sizes use the same head
row count as their latency lookup, not all scheduled prompt tokens.
This endpoint contract does not settle idle-DP padding, speculative heads
or expert dispatch/combine; those require their own audits.

### The attention grid's second axis describes the batch, not one sequence
It used to be `kv_prefill`, one prefill sequence's context, and the simulator
fed it `sum(batch.prefill_k_list)`. A step routinely carries several prefill
sequences -- one request finishing its prompt beside another just starting --
and summing their contexts produces a coordinate none of them has: 0, 1200 and
64 came out as 1264.

The axis is **`prefill_key`**: the query-weighted mean of each prefill's
effective causal key length --

    prefill_key = sum_i c_i * (k_i + c_i / 2) / sum_i c_i

`c_i` is what request *i* has scheduled this step and `k_i` what it already
holds; the half is causal masking, since the average query in a chunk sees
half of it. `profiler/core/attention_shape.py` owns this coordinate for both
profiling and serving. An empty prefill contributes zero. Cap each request's
effective key before weighting for a saturating kernel; this remains a window
approximation, not an exact discrete causal-work integral. Equal chunks retain
their old coordinate, but unequal chunks must not count as equally much work:
chunks 1792 and 256 with no history give 800, not the old sequence mean 512.

The following figures describe the earlier sequence-mean migration, not an
independent validation of query weighting. Matched against real vLLM runs -- table built from the shapes the
profiler sweeps, scored on multi-prefill steps -- that moves |err| p50 from
14.18% to 5.97% on DeepSeek-V3.2 and 12.96% to 7.93% on Llama-3.1-8B, and it
improves the single-prefill case too (2.15% -> 1.71%, 10.28% -> 9.88%). Adding
`n_prefill` as a fifth axis on top changes nothing measurable: once the key
coordinate is batch-aware the sequence count has nothing left to say. The grid
stays four axes wide.

**Sweeping it needs multi-sequence shots.** One sequence can only reach
`key >= tokens / 2`, so "many tokens, short keys" -- 4 requests of 500 tokens
with no context -- is unreachable, and **32-39% of a real run's multi-prefill
steps land there**, at the full token budget, which is exactly where a
saturated run's TTFT is set. `_compose_prefill` derives the batch that
realises a target coordinate: the fewest sequences that leave each a
non-negative context. About 27% of the grid's shots are multi-sequence.

**Old bundles are relabelled, not refused.** Every pre-change shot had exactly
one prefill sequence, so `prefill_key = kv_prefill + prefill_chunk / 2` is
exact and `_build_attention_table` applies it on load. What a relabel cannot
recover is the coverage above. `meta.yaml::attention_grid.axes` is the marker
that says which form a bundle holds.

`kv_prefill` still exists on `BatchCtx` and is still the summed context -- it
is a *length*, and tensor sizing and the PIM split need it as one.

### A saturating kernel is capped at lookup, per kernel
A sparse-attention query reads at most a fixed number of selected positions,
so its cost stops growing once a sequence's key window passes that bound:
DeepSeek-V3.2's 16-token prefill costs 968.8 us at no context, 742.4 at 512,
then 139.3 at 2048 and 138.3 at 8192 -- flat from exactly `index_topk`. The
same sweep on Llama rises throughout (8.5, 18.8, 51.2, 181.1), because a dense
query reads its whole causal window.

`stack.probe_key_saturation` reads `index_topk` for DeepSeek/GLM. For MiniMax-M3
it retains the committed tables' historical lookup rule, including the initial
and local block adjustment (17 x 128 = 2176). New native acquisitions instead
use `attention_lookup.py` and `meta.yaml.attention_lookup.per_tp`: the inspected
vLLM 0.31 Triton indexer/attend pair uses the configured selected-block count
with the reviewed 128-token block layout (2048 tokens for the shipped config)
for `sparse_attention` prefill lookup only. Initial/local blocks are part of that
total, not an addition. Exact selected classes, callable API, BF16
indexer/KV storage, selection-buffer layout and model/catalog parameters must
match; speculative acquisition and unreviewed backends/layouts are refused.
Native attention publication reads the current measurement metadata and recognizes
explicitly supported historical row protocols. Recognition does not waive
semantic acquisition compatibility or completion checks and never converts
historical measurements. Source-file hashes and version labels are not vetoes.
The catalog's `key_saturates` says which entries are subject to the bound. That
is a per-**kernel** property, not a per-model one: M3 is sparse only from its
fourth layer on, and a sparse model's `indexer` scores the whole KV to make
the selection and never saturates (52 -> 101 us from kv 2048 to 16384).

The cap is applied at lookup, never at sweep time. Both kernels share one
grid, so a capped column would collapse the indexer's large-key rows onto a
single cell. `BatchCtx` carries the mean both ways because the clip is per
sequence and must precede the mean -- `mean(min(x, cap))` is not
`min(mean(x), cap)`.

Native row protocols bind the per-TP semantic digest to the acquisition identity.
Pending records cannot be consumed; completed attention tables carry a checksum.
Missing contracts for marked rows, changed bytes or unknown schemas fail closed.
Never relabel historical rows, add a contract after acquisition, or change the
historical helper/fingerprint to migrate existing fits. Refresh in a separate
root, or explicitly remeasure the affected category. Serving, skew fitting,
support completion and `plan-skew` share this resolver and select each TP's cap
without mutating the shared cached database. Decode-history coordinates remain
uncapped. Partial non-attention refreshes preserve the authoritative contract.
Native skew publication also checks row identity and completion; a zero-case
plan creates no skew artifact, changes no existing artifact even with `--force`,
and contributes no new skew provenance or fit trigger to the caller.
TP1 prefix acquisition and boundary checks do not certify multi-TP deployment,
memory capacity, a full grid or end-to-end benchmark accuracy. See the
[public contract](docs/docs/profiler/output-bundle.md#native-attention-lookup-contract).

**The same flag governs the skew guard.** `alpha` is where a skewed batch
lands on the line between the two uniform attention-table lookups. That line
does not care which end is higher; the fitter and simulator accept either
sign for a kernel declared saturating. The former `t_max > t_mean` guard
was right for dense -- cost rises with kv, so a negative
gap is noise -- and wrong for a saturating kernel, where the gap is negative
for essentially every case, so a sparse skew sweep would have had almost every
row dropped as `nan`.

### Canonical layer names (simulator ↔ profiler, unified)
The simulator consumes the profiler's per-category CSVs directly. Canonical
layer names match vLLM's own attribute names. `trace_generator` walks the
`blocks:` section of `profiler/models/<model_type>.yaml`; the table below
lists where each layer appears in the profiler CSVs and how the simulator keys
the lookup.

| Layer | Category (CSV) | Key semantics |
|-------|----------------|---------------|
| `embedding` | dense | `tokens = total_len` |
| `layernorm` | dense (tp_stable) | `tokens = total_len` |
| `qkv_proj` | dense | `tokens = total_len` |
| `qk_norm` | dense (TP-dependent; Qwen3 only) | `tokens = total_len` |
| `rotary_emb` | dense | `tokens = total_len` |
| `attention` | attention | `(prefill_chunk, prefill_key, n_decode, kv_decode)` |
| `o_proj` | dense + ALLREDUCE after (TP>1) | `tokens = total_len` |
| `gate_up_proj` | dense | `tokens = total_len` |
| `act_fn` | dense | `tokens = total_len` |
| `down_proj` | dense + ALLREDUCE after (TP>1) | `tokens = total_len` |
| `final_layernorm` | dense (tp_stable) | `tokens = total_len` |
| `lm_head` | per_sequence | `sequences = num_requests` |
| `sampler` | per_sequence (tp_stable) | `sequences = num_requests` |
| `moe` | moe (tp=1, one grid per EP degree; wrapped in the EP all-to-all) | `(ep_total, local_tokens, activated_experts)` |
| `mtp_*` | mtp (only with `--profile-mtp`; emitted N times per step, each pass wrapping a replay of `mtp.decoder_block`) | `tokens = total_len` of the pass |

The **`linear_attention`** category is keyed on `(prefill_tokens, n_decode)`,
and that key is what makes it regime-aware: a gated-DeltaNet block runs a
*different set of kernels* for a pure prefill, a pure decode and a mixed batch,
so a kernel simply has no rows for a regime it does not fire in, and the
simulator emits nothing for it there. The historical Qwen3.8-27B decomposition:

| kernel | prefill | decode | mixed |
|---|---|---|---|
| `gdn_conv_prefill`, `gdn_post_conv`, `gdn_prefill` | yes | — | yes |
| `gdn_conv_decode`, `gdn_decode` | — | yes | — |
| `gdn_decode_mixed` | — | — | yes |
| `gdn_in_proj`, `gdn_out_proj`, `gdn_norm`, `gdn_glue` | yes | yes | yes |

Put a regime-dependent kernel in `dense` or `per_sequence` and it is emitted on
**every** batch, because neither has a notion of regime — that charged a decode
conv on a pure prefill and a prefill conv on a pure decode. The always-on
layers stay in `dense`, which is why the split is per layer rather than per
block. Use `python -m profiler coverage` to find out which is which; it reports
per regime.

The selected packed layout replaces the GDN conv/scan/norm components with
`gdn_core`; projections and outer glue stay separate. Its size is packed qkvz+ba
input, gated-normalized value output, and owned conv/dt_bias/A_log/norm parameters;
persistent state capacity remains per sequence. Use model activation dtype (A_log
is float32), require positive TP-divisible heads, and reject unverified quantized
BA layouts. Sequence expansion is cached by TP and original sequence without
mutating the catalog or changing latency interpolation.

The `attention` category holds **more than one kernel** on a sparse model, and
they are not interchangeable — the lookup takes a layer name
(`attention_by_layer`) and so does the skew alpha. MiniMax-M3 profiles
`attention` (its non-sparse layers), `sparse_attention` and `indexer`;
DeepSeek/GLM profile `attention` (MLA) and `indexer`. There is deliberately no
pooled `tables["attention"]` shortcut: there was one, every lookup took it, and
a sparse layer got the dense kernel's latency (2.1x per layer on M3).

Names beyond this table are per-family and live in the catalogs
(`gdn_*`, `mla_*`, `indexer_*`, `sparse_*`, `mtp_*`, `*_glue`).
**Ordinary catalogs require `calculate_sizes` coverage in `memory_model.py`.**
Admitted native/contextual representations instead use their selected providers
in `component_sizes.py`; their logical trace bytes do not establish resident
capacity. Check every selected canonical name through its actual sizing path.
Catalog presence alone does not establish a valid shape or deployment.

### Trace file format
A trace is a list of per-layer field tuples, handed to the Chakra converter
in memory. `--save-trace-text` also writes it as a whitespace-aligned text
file (fixed-width columns from `utils.py::_FMT`), which is the only
human-readable form of what the simulator emitted — nothing in the pipeline
reads it. The field spec below describes both representations; the text
columns are what `indexed_cols()` and `_write_trace()` agree on. The sample below is illustrative, not byte-accurate — see
`docs/docs/reference/trace-format.md` for the real widths. Every field except the
last carries an **explicit trailing space** in `_FMT`: `{:<15}` pads a short value
but emits nothing for one that already fills the column, so a 15-character value
like `ALLREDUCE:1,0,0` (3-D `involved_dim`) would otherwise run into the next
field and the row would come back one short. `header()` goes through `_FMT` too, so
the whole file has one layout:

```
COLOCATED		model_parallel_NPU_group: {pp_size}		pp_stage_boundaries: 73,145,217
{num_layers}
Layername    comp_time    input_loc    input_size    weight_loc    weight_size    output_loc    output_size    comm_type    comm_size    misc
embedding_0  5621         REMOTE:0     40            LOCAL         1050673152     LOCAL         81920          NONE         0            NONE
...
sampler_291  25933        LOCAL        2565120       LOCAL         0              REMOTE:0      40             NONE         0            NONE
```

- Line 1 carries `key: value` pairs after the mode marker.
  `model_parallel_NPU_group` is `pp_size`. `pp_stage_boundaries` appears only when
  `pp_size > 1` and lists the `pp_size - 1` line indices at which each stage after
  the first begins, counted **after** any leading `kv_load`/`kv_evict` rows (the
  converter strips those before partitioning). The frontend places them on
  transformer-block starts using vLLM's `get_pp_indices` rule — blocks split
  evenly, remainder to the stages *before* the last, which also carries
  `lm_head`
- `comp_time`: latency in nanoseconds (from the per-category CSVs, whose `time_us` is converted at load time)
- `input_loc`/`weight_loc`/`output_loc`: `LOCAL` (NPU), `REMOTE:{node_id}` (CPU), `CXL:{id}`
- `comm_type`: `NONE`, `ALLREDUCE`, `ALLGATHER`, `REDUCESCATTER`, or with
  dimension scoping `ALLREDUCE:1,0`, `ALLGATHER:0,1`, `ALLREDUCE:1,0,0` (the
  `:dim0,...` suffix maps to ASTRA-Sim's `involved_dim` BoolList for
  multi-dimensional topologies, one entry per topology dimension)
- `comm_size` on `qkv_proj` carries the **P/D KV transfer amount** (per layer, per rank,
  K+V only, honouring `kv_cache_dtype`), and is 0 unless `pd_type == "prefill"`.
  `convert_prefill` reads it for the per-layer SEND/RECV to the paired decode NPU. It used
  to read the layer's `output_size`, i.e. the whole QKV activation, which shipped Q as well
  and overstated the transfer by `(q_dim + 2*kv_dim) / (2*kv_dim)` — 3x for Llama-3.1-8B.
  `comm_type` stays `NONE` there: a SEND/RECV pair only needs comm_size/src/dst/tag
- `misc`: `NONE` or batch tag for sub-batch interleaving (`BATCH_1`, `BATCH_2`)
- First layer (embedding) input comes from `REMOTE` (CPU → NPU), last layer (sampler) output goes to `REMOTE` (NPU → CPU)
- MoE uses `EXPERT {i}` / `EXPERT END` markers (comm_type on EXPERT line can include dimension scoping)
- PIM uses `PIM {channel}` / `PIM END` markers

### Performance DB and latency lookup
The simulator loads per-category CSVs via `_load_perf_db()` and dispatches
lookups by catalog category: `_lookup_dense` (1D linear over tokens),
`_lookup_per_sequence` (1D linear over sequences), `_lookup_attention` (4D
linear over `(prefill_chunk, prefill_key, n_decode, kv_decode)`), and
`_lookup_moe` (2D over `(tokens, activated_experts)` within the table for
this run's `ep_total`, profiled at tp=1).
Lookups extrapolate **above** the top profiled value (time_us is linearly
extended); below the bottom they **clamp**. That asymmetry is only safe because
every axis sweeps a 0 or 1 point, so no runtime query can fall under it -- with
one exception, which is why `moe.csv` has an `ep` axis. See the MoE section.

`_axis_bracket` blends on a **linear** scale, not in log space, even
though the profiler sweeps every axis geometrically. Grid spacing decides
where the kernel is sampled; the blend decides how two samples are
combined; the kernel is linear in each axis. Profiled decode attention
fits `time_us = a + b * (n_decode * kv_decode)` with R^2 = 1.0000 (RTX
4090 / Llama-3.1-8B, implied 953 GB/s = 95% of spec — a pure
KV-bandwidth read). Log blending of a per-axis-linear function is
convex-biased upward by up to +6.0% per axis on a doubling grid, and
leave-one-out over the measured grid puts it at +11.6% to +14.4% mean
error against +2.3% to +3.7% for linear, across every bundle in
`profiler/perf/`. Don't "restore" log space because the sweep is
geometric. Latencies are stored as microseconds in the
CSVs and converted to nanoseconds at load time. No calibration scaling —
profiled latencies are used directly.

Attention with skew correction: `_lookup_attention_with_skew` uses the
unchanged uniform mean/max lookups and resolves alpha via `_skew_alpha`.
Pure prefill, one decode and uniform decode bypass correction. For a
heterogeneous batch, both endpoints are needed to compute the lever even
when the selected alpha is zero. New calibration tables use a kernel/query/
prefill/lever partition and binary search over supported N anchors.
`_hydrate_skew_fit_tables` validates and loads the table once; no fitting,
measurement-neighbor interpolation or file access occurs per batch.
Enabled fits without a recognized calibration format are rejected with a
`profiler refit-skew` instruction. The current format is `runtime-skew-calibration`.
Disabled skew bundles, including RTX4090, bypass calibration unchanged.

Profile CSV path: `profiler/perf/<hardware>/<model>/<variant>/tp<N>/{dense,
per_sequence,attention,moe,skew,skew_fit}.csv` (resolved as
`../profiler/perf/...` from the `astra-sim/` working directory).

Variant resolution: `trace_generator.resolve_variant(model_config)` selects the
checkpoint-default folder and is a **pure function of the model config** —
it takes no dtype argument, because the simulator has no dtype
input. Weight dtype is `utils.config_weight_dtype` (`quantization_config`
first, then `torch_dtype` / `dtype`), KV dtype is `utils.config_kv_cache_dtype`
and appends a `-kv<short>` suffix when not `auto`. Runtime lookups verify the
resulting folder exists; a miss raises a clear `FileNotFoundError` pointing at
the missing variant. The profiler can still *write* other variants for the same
model (`--variant`, `--dtype`, `--kv-cache-dtype`), which is how a deliberate
second precision is measured and kept beside the first; the simulator just
never asks for one. Do not assume universal profiler/consumer name equality:
nested-only quantization, FP32 auto conversion and auto-promoted KV dtypes
retain compatibility-label limitations documented in the profiler running guide.

Runtime vs. profiled warnings: on first load of a `(hardware, model, variant)`,
the simulator compares the CLI's `--max-num-batched-tokens` and `--max-num-seqs`
against `meta.yaml`'s `engine_effective` values and logs a one-shot warning
when the runtime exceeds the profiler's sweep bounds (lookups will extrapolate).

### Agentic session support (dependency chains)
The simulator supports closed-loop agentic workloads (SWE-bench, tool-calling agents)
where LLM calls within a session form a dependency chain interleaved with tool calls.

**Dataset format:** Each JSONL line is a session with `sub_requests[]`. Each sub-request
has `input_toks`, `output_toks`, `tool_duration_ns` (wait time after this LLM call before
the next can start). Flat requests (no `sub_requests` key) are also supported for backward
compatibility. Both formats can coexist in the same file.

**Router dependency tracking** (`router.py`):
- `load_requests()` auto-detects flat vs agentic format. For agentic sessions, only the
  first sub-request is queued; the rest are stored in `_deferred_sessions`
- `notify_request_completed(request_id, completion_time_ns)` releases the next sub-request
  at `completion_time + tool_duration_ns` and inserts it sorted into `_pending_requests`
- `has_deferred_sessions()` prevents premature simulation exit while sessions are in-flight
- `scheduler.add_request()` uses `bisect.insort` (not `append`) to maintain arrival-time
  sort order when dynamically released sub-requests enter the queue

**Time advancement:** When all instances are idle but deferred sub-requests have future
arrival times (tool calls still running), `serving/__main__.py` advances `current` to the next pending
arrival time to avoid busy-looping.

### Scheduler and memory model
- `scheduler.py` follows vLLM V1's `schedule()` shape: a persistent `self.running` set is
  served first (phase A), preempting only from its own tail, then `self.waiting` is admitted
  (phase B) while budget and slots remain. Phase B **never** preempts, and it is skipped
  entirely on any step that preempted — that anti-thrash rule is load-bearing
- There is **one** `schedule()` for prefix caching on and off. The pool handles
  `enable_caching=False` the way vLLM does (allocate through the same free list, never
  index), so do not reintroduce a second scheduler
- Admission also refuses a request whose *whole* sequence would not fit, not just
  its first chunk (`--reserve-full-isl`, on by default, per-instance
  `reserve_full_isl`). Mirrors vLLM's `scheduler_reserve_full_isl`, `True` there
  too; checking only the first chunk lets chunked prefill over-admit
- Admission is **one step stale** under `--async-scheduling` (on by default, as
  vLLM's `scheduler_config.async_scheduling` is). vLLM's
  `max_concurrent_batches` is 2 at `pp_size` 1 with it on
  (`config/vllm.py`: "Async scheduling requires 2 concurrent batches to
  overlap"), and `EngineCore.step_with_batch_queue` composes batch k+1 right
  after submitting k, then blocks on k's future -- so the batch that executes
  next was built while the previous one was still on the GPU, and a request
  arriving after that cannot join it. `_arrival_cutoff` reproduces it by lagging
  phase B's arrival horizon to the completed batch's own `batch_time`. Working
  through vLLM's loop, a request arriving at `a` during batch A lands in C, not
  B, so `TTFT = (t_A_end - a) + dt_B + dt_C` on both sides.
  The lag is dropped when the instance was **not** busy right up to `current`
  (an idle engine's batch queue is empty and vLLM admits immediately), and
  `schedule()` retries phase B with the full horizon when the lagged pass
  admitted nothing. That retry is load-bearing, not cosmetic: the main loop
  advances the clock only for arrivals still pending in the *router*, so a
  request already sitting in `waiting` behind a stale cutoff would be invisible
  forever and the run would answer "pass" at a clock that never moves.
  At `pp_size > 1`, this additional lag applies only when the scheduler's
  `_uses_v2_model_runner` classification is true; the modeled V1 pipeline
  already supplies its in-flight window. The classification currently treats
  non-MoE, non-hybrid stacks as V2 and is separate from `cudagraph.runner`.
  Admission and TTFT effects depend on workload and batch composition.
- Token budget controlled by `--max-num-batched-tokens` (default 2048) and `--max-num-seqs` (default 128)
- `--long-prefill-token-threshold` caps per-request tokens per step for chunked prefill
- **There is no prefill phase or decode phase.** A request just catches up to
  `num_tokens_reached`, so `num_new = num_tokens_reached - num_computed_tokens` — 1 in
  steady-state decode, the whole sequence for a resumed request. `Request.is_prefill()` is
  gone on purpose: it read `original_input` and would misread a resumed request's
  recomputation as decoding. The trace classifies by **scheduled token count**
  (>1 = prefill chunk, ==1 = decode)
- `num_computed_tokens` is advanced at **schedule** time, as in vLLM's
  `_update_after_schedule`. `Batch.scheduled_tokens` is the snapshot `add_done` works from.
  Advancing at completion instead lets `pp_size > 1` schedule the same tokens twice
- `num_tokens_reached` (prompt + generated) must never be derived from
  `num_computed_tokens`: preemption resets the latter to 0
- Preemption is vLLM verbatim, including `num_computed_tokens = 0`. That is **not**
  re-prefill — `free_blocks` keeps the blocks' hashes, so on re-admission
  `get_computed_blocks` finds whatever is still resident, a lower tier returns what was
  written down, and only the remainder is recomputed. Do not add a "preserve the decode
  state" special case; the tiers are what preserve it
- The three modes map onto three real vLLM configurations: `--no-enable-prefix-caching` =
  vLLM with prefix caching off (a resume recomputes the whole sequence);
  `--enable-prefix-caching` = default vLLM; plus `--prefix-storage CPU/CXL` = vLLM with
  LMCache / `OffloadingConnector` attached
- KV capacity is `npu_mem.mem_size * npu_mem.mem_util - weight`, divided into `--block-size`
  blocks. vLLM also subtracts the activation peak and CUDA context, which are not modelled,
  so the simulator's capacity is an upper bound at the same utilization. That only matters
  when a run **saturates** the KV cache: below the ceiling nothing is preempted and the
  configured capacity is invisible in the results. **For a run that does saturate,
  calibrate `mem_util` so each instance's block count matches its observed engine
  pool in the bench run's `meta.json` before comparing latency at all.** New runs
  record `kv_cache.capacity.per_engine`; require its complete status rather than
  interpreting a DP client aggregate as one pool. Historical single-engine
  references retain their recorded `kv_cache.num_gpu_blocks`. The RTX 4090
  example matches 2,588 blocks with `mem_util: 0.833919`. The refreshed Qwen3-30B
  example matches 38,256 blocks per DP rank; Llama and Qwen3-32B retain the
  recorded utilization without claiming a complete per-engine capacity match.
  Inspect preemption and request scheduling before attributing latency error
  to capacity; heartbeat occupancy percentages have different cache accounting
  from vLLM and do not establish equivalent pools.
- `block_pool.py` / `kv_cache_manager.py` own everything about which blocks exist and where.
  `memory_model.py` keeps the static sizing math and a byte-level view; `npu_used` is a
  property over the pool, so there is one ledger per tier
- `calculate_sizes(parallel=)` computes per-layer tensor sizes — `parallel` is TP for dense
  layers and EP for MoE experts. Uses `head_dim`, `q_dim`, `kv_dim`
- MoE expert weights are sharded by `ep_size` (not `tp_size`)
- Prompt throughput (`prompt_t` in `add_done()`) includes prefix cache hit tokens,
  not just actually computed prefill tokens. This matches vLLM's reported prompt
  throughput which counts all input tokens including cached ones
- `kv_load` / `kv_evict` trace layers fire **only** with `--prefix-storage`: without a lower
  tier there is nothing to recall from, so both byte counts are 0. `batch.evict` is 0 in
  every mode — eviction from the NPU costs nothing, because the data is either a finished
  request's cache or already written down off the critical path

### Speculative decoding
`--num-speculative-tokens N` turns it on. Which draft tokens the target accepts
is the one thing a simulator cannot compute -- it needs the draft's and the
target's distributions over real tokens -- so acceptance is a **policy** chosen
the way MoE expert routing is, with the default taken from what the model's own
authors published (`configs/spec_decode.json`, one entry per model, each
carrying its source).

**The rate is `accepted / drafted`, and it is marginal**, so

    mean_accept_length = 1 + rate * N

That identity reproduces all nine published (rate, length) pairs to within 0.01
tokens. It is deliberately **not** Leviathan's per-position alpha (ICML 2023),
which is *conditional* -- position i is reached only if 1..i-1 were accepted --
and gives the capped geometric `(1 - a^(N+1))/(1 - a)`. Passing a published
rate to that formula under-predicts the published accept length by 25-30%,
because real acceptance is front-loaded rather than i.i.d. The check that
settles the reading: Qwen's published per-position decline of 95% at p1 to 60%
at p5 averages 0.775 read as marginals against a published 0.779, and 0.621
read as conditionals. **Don't reintroduce the capped-geometric formula.**

A model with no published figure gets no default -- rates run from 0.39 to 0.78
across the four modern families, so there is nothing defensible to guess.

Scheduling follows vLLM exactly, including its framing: `num_tokens_with_spec =
num_tokens + spec_tokens`, a request just catches up to it, and rejection rolls
back with `num_computed_tokens -= num_rejected`. Three details that are easy to
get wrong:

- **Roll back before caching the prefix.** A block holding a rejected token
  must never be hashed, or a later request hits on text the model never emitted
- **Classify a spec step by why it has many tokens, not by the count.**
  `num_new > 1` files a verification step as a prefill chunk; the `1 + N`
  queries of one sequence share that sequence's KV read, a prefill chunk of the
  same size does not
- **Clamp the overshoot.** A step commits `1 + accepted` at once and can run
  past `output`; vLLM stops at max_tokens and discards the excess, so the
  overshoot is not throughput
- **The draft count is per *batch*, not per request.** `add_done` reads it from
  `Batch.spec_scheduled`, snapshotted at build time exactly as
  `Batch.scheduled_tokens` is. Left on the request it is rewritten by the next
  `schedule()` while the batch is still in flight — to 0, since the request is
  no longer caught up — so the rollback never runs and `num_computed_tokens`
  stays `N - 1` past `num_tokens_reached`, which no later step can schedule.
  Invisible at `pp_size == 1`, where an instance holds one batch at a time

The verification forward needs the **fifth attention axis**, `decode_q_len`
(query tokens per decode sequence, `1 + N`). It is opt-in in the profiler
(`--attention-decode-q-lens`, default `1`) because it multiplies the grid. An
unprofiled value falls back to the nearest with a one-shot warning rather than
interpolating: query length changes the kernel's tile shape, not just its size,
and unlike the other four axes there is no measurement saying a value between
two profiled ones lies between their costs.

Being opt-in meant it had **never been run**, and the grid's feasibility
filters sat outside the `decode_q_len` loop — so every one of them counted a
decode request as a single token. Right at `q = 1`, wrong above it: a shot at
`chunk = max_num_batched_tokens` whose `n_dec * q` ran past `max_num_seqs`
passed the filter and overflowed vLLM's own buffer, three and a half hours into
a sweep, as `operands could not be broadcast together with shapes (2304,)
(2432,) (2304,)` — 2304 being `MNBT + MSQ` and 2432 the real token count,
`2048 + 64*6`. The three token-based bounds (the combined sum, a decode's own
sequence length, and the KV block budget) are inside the loop now and take `q`.
At `q = 1` every one is identical to what it was, so committed grids still line
up shot for shot with a re-profile.

Two shapes of the grid are worth knowing when planning a run:

- **Query coverage.** Each requested query length adds decode-containing shots;
  pure prefill is emitted once at `q=1`, even if `1` is not requested. Separate
  sweeps can therefore overlap on prefill rows. Prefer a combined acquisition;
  independent runs need separate output roots, an explicit common KV bound and
  compatible identities before any merge. `decode_q_len` remains part of the row
  key. FULL-prefill anchors union the target configurations' capture sizes.
- **The reach.** `--attention-max-kv` defaults to the model's own context,
  `max_model_len - max(decode_q_len) - 1` — not `max_model_len`, which gets
  the top point filtered (a decode occupies `kv + q` positions and needs one
  more to be a decode) and stops the sweep a doubling short. Cost is close to
  linear in the number of kv values, since the KV-budget filter prunes the
  large-kv x large-n_decode corner: 8,643 shots at 16,384 against 14,653 at
  DeepSeek-V3.2's full 163,834.

**Reach matters more on a sparse model than a dense one.** The simulator
extrapolates linearly past the top profiled kv, which is right for decode
attention — a pure KV read fitting `a + b*(n_decode*kv_decode)` at R^2=1.0000 —
and wrong once selection is involved. Measured on DeepSeek-V3.2 at
`n_decode = 8`, `attention` is 342 us at `kv = 2048` and 350 us at 16,384, flat
from `index_topk` onward because past that it only reads the selected tokens,
while `indexer` goes 52 -> 101 us because it scores the whole KV to make the
selection. The unprofiled region is exactly where a long-context run's cost
lives.

### Linear-attention state, prefix caching and the drafter
Three things a hybrid or speculative run costs that a per-token KV model does
not see. All three are vLLM's rules, and none is guesswork.

**State pages, not state bytes.** vLLM picks the attention block size so one
attention page covers one mamba page — `platforms/interface.py`:
`attn_block_size = alignment * cdiv(mamba_page_size, alignment *
attn_page_size_1_token)` — then sets `mamba_page_size_padded = attn_page_size`,
so a layer's whole recurrent state occupies exactly one page and the padding is
really allocated. Qwen3.8-27B: mamba page 3,207,168 B against an attention page
of 3,211,264 at `block_size 784`, which is how 784 gets chosen. How many pages
per layer is `MambaSpec.max_memory_usage_bytes`:

| cache mode | pages per mamba layer | when |
|---|---|---|
| `none` | `1 + N` | prefix caching off |
| `align` | `2 + N` | prefix caching on — **the default** |
| `all` | `cdiv(max_model_len, block_size)` | opt-in, not modelled |

`align` holds two because one page carries the state being written this step
and the other the last checkpoint at a block boundary, which is what a later
prefix hit resumes from. `N` is `num_speculative_tokens`. Charging one page per
layer understates a prefix-caching run by exactly 2x. Speculative decoding also
**widens the conv state itself** (`conv_kernel_size - 1 + N`), which is the
small half and moves ~2% at N=4.

**Chunk ends must be block-aligned under `align`.** State slot *p* holds the
state after exactly `(p + 1) * block_size` tokens and state is written at chunk
ends, so `Scheduler._mamba_block_aligned_split` floors a prefill chunk to a
block boundary (exempting the prompt's last chunk), stops a mid-block chunk at
the next boundary, and never runs past the last block-aligned position. It can
legitimately return **0** — vLLM's "insufficient budget for a block-aligned
chunk" — and that is not the scheduler deadlock the `num_new <= 0` guard
catches: the split only floors to zero when `block_size <= max_prefill_tokens`,
so a fresh step's budget does cover a block. With `block_size 784` and a 2048
budget a chunk is 784 or 1568, never 2048, so this changes batch composition on
every hybrid run.

**The drafter is not free.** vLLM runs it **N times per step** — once, then
`num_speculative_tokens - 1` more in `llm_base_proposer.py`'s loop — each a
decode-shaped forward at `max_query_len = 1`. A model that drafts with itself
runs an MTP module per pass: two norms, an `eh_proj`, one full decoder layer of
its own family (`DeepseekV2DecoderLayer`, `Glm4DecoderLayer`,
`MiniMaxM3DecoderLayer`, `Qwen3_5DecoderLayer`), then a norm, `lm_head` and the
sampler. The block is **full attention** whatever the target's layers are —
Qwen3.5's MTP passes `layer_type="full_attention"` explicitly — so a hybrid's
drafter carries no recurrent state, but it does carry a KV cache: `+1.6%`
bytes/token on DeepSeek-V3.2's one module, `+11.7%` on MiniMax-M3's seven,
`+6.2%` on Qwen3.8 (one more full-attention layer out of its 16).

Draft **time** is charged: `_emit_drafter` emits N passes after the target's
head (the drafter runs from `sample_tokens()`, after the target has sampled),
each one `mtp.prologue` → a replay of `mtp.decoder_block` → `mtp.head`. The
first pass reuses the target's own token layout and the rest are pure decode at
one query per sequence, which is what `llm_base_proposer.py` does — reusing the
target's shape for all N charged a prefill chunk N times over.

`_require_drafter_cost` still **refuses** a speculative run on a model with MTP
modules whose catalog is missing `mtp.prologue` or `mtp.decoder_block`, rather
than reporting a speedup no engine can deliver; the wrapper has to come from a
live profile dump like every other block. A model with no MTP modules drafts
externally (a second model, or n-gram); that is a serving choice rather than a
checkpoint property, so it warns instead of refusing.

### Dtypes come from the model config, never from an input
There is no `--dtype` and no `--kv-cache-dtype`, and no cluster-config
`dtype` / `kv_cache_dtype` either. A model carries **five** cache dtypes and
they are decided in four different places, so a flag per dtype is both
unusable and unfaithful — the checkpoint already says what it is, and saying
otherwise describes a model nobody can serve. `memory_model.cache_dtype_bytes`
holds the whole table; every rule below is vLLM's, verified against its source:

| Cache | Source | vLLM |
|-------|--------|------|
| weights | `quantization_config.quant_method`, then `torch_dtype` / `dtype` | on a quantized checkpoint the dtype fields are the *activation* dtype |
| KV cache | `quantization_config.kv_cache_scheme` / `kv_cache_quant_algo` → fp8, otherwise `torch_dtype` / `dtype` (the *activation* dtype on a weight-only quantized checkpoint) | `attention.py:281` promotes a declared scheme when the flag is `auto`; otherwise `auto` is `model_config.dtype` (`platforms/interface.py`), so DeepSeek-R1's fp8 weights keep a bf16 cache. Sparse-MLA models (`index_topk`) keep the weight dtype: vLLM picks their format per backend (`fp8_ds_mla` on SM120) |
| mamba conv state | `mamba_cache_dtype`, `auto` → weight dtype | `mamba_utils.py::_mamba_state_dtype` |
| mamba recurrent state | `mamba_ssm_dtype`, `auto` → **conv** dtype (not the weight dtype) | same, plus `models/config.py::Qwen3_5ForConditionalGenerationConfig` bridging the HF field |
| sparse-indexer side cache | fixed by the model | DeepSeek/GLM `torch.uint8`, M3 `resolve_indexer_kv_dtype("bf16")` — neither follows the KV dtype |

Note the weight row is the profiler's **variant folder name**, not vLLM's
`model_config.dtype`: vLLM calls DeepSeek-V3.2 bfloat16 and keeps fp8 in the
quant method, while the folder has to encode the quantization or two bundles
collide. That divergence is deliberate; the other four match vLLM exactly.

Profiler `--dtype auto` uses the omitted-dtype naming path. An explicit BF16/FP16
matching a quantized checkpoint's declared activation dtype keeps the quantization
base; another activation dtype gets `-act<dtype>` (for example `fp8-actfp16`).
Explicit FP32 is not an auto match: vLLM may downcast an FP32 checkpoint.
HF precision-field or any nested `text_config` override requires `--variant`;
ambiguous dtype aliases or nested-only quantization with explicit `--dtype`
also require an intentional label. These names never change engine quantization.
Serving still selects only the checkpoint-default variant. Do not rename or
restamp old acquisitions to represent a new precision or measurement identity.

To simulate a different precision, **profile it** — the profiler's flags write
a separate bundle and the simulator reads the one the checkpoint names.

### Head dimension
Some models (e.g., Qwen3) have `head_dim != hidden_size // num_attention_heads`. Always use:
```python
head_dim = config.get('head_dim', n_embd // n_head)
q_dim = n_head * head_dim        # NOT n_embd
kv_dim = kv_head * head_dim      # NOT n_embd // group
```

### Tensor sizes, block weight and KV shape
`memory_model.calculate_sizes(model, layer_name, ...)` returns
`(input, weight, output)` bytes **per rank** for one canonical layer, and
**raises** on a name it does not know — so a catalog entry without a formula
here makes the model unsimulable, not merely mis-sized.

MiniMax-M3's canonical `indexer` is resolved before the separate DeepSeek
formula. Its index query and key-only cache use their independent BF16 dtype;
its output is int32 `[tokens, local index heads, topk blocks]`. The trace passes
exact prefill and decode cached histories through `kv_lens` for M3 `indexer`
and `sparse_attention` only: the indexer sums them whole, while sparse attention
sums `min(history, sparse_topk_blocks * sparse_block_size)` per request.
Do not reconstruct those histories from a floored decode mean or cap their
sum once. These are logical query plus preexisting-cache bytes, not kernel
DRAM transactions, workspace, capacity or causal-prefill read multiplicity.
The existing byte cap and historical lookup cap remain separate; this sizing
rule does not establish arbitrary runtime indexer dtypes or whole-model TP
correctness. Unknown indexer layouts and malformed M3 geometry are refused.

`get_weight()` walks the architecture yaml's blocks and the checkpoint's own
per-layer composition (`utils.get_architecture` / `utils.get_layer_stack`), one
built weight per distinct block shape. It used to sum a hardcoded
`layernorm + qkv_proj + o_proj + layernorm + mlp`, which cannot describe MLA
(no `qkv_proj`) or a hybrid stack. For PP it takes the **heaviest** contiguous
window of layers, since the first window is the light one on a stack whose
leading layers are dense.

`kv_bytes_per_token_per_layer()` is the one place that knows the KV shapes, and
they are not interchangeable:

| Shape | Per token per layer | TP |
|-------|---------------------|-----|
| GQA | `2 * kv_head * head_dim * kv_fp` (K and V) | sharded |
| MLA | `(kv_lora_rank + qk_rope_head_dim) * kv_fp`, one latent, no separate V | **replicated** (`num_kv_heads = 1`) |
| + sparse indexer | plus `index_head_dim + index_head_dim//128 * 4` bytes (fp8 keys + fp32 scales, uint8) | replicated |
| linear attention | 0 per token — the state is per **sequence** | n/a |

A per-sequence state is charged as blocks the request holds for its whole life
(`MemoryModel._state_blocks_per_request`), in a list separate from the token
blocks: the token list is positional, so a block backing no tokens must not
join it. It is counted in **pages**, not bytes — see the linear-attention
section above for why, and for the `none`/`align` page counts. 153.9 MB per
sequence on Qwen3.8-27B, i.e. 6 blocks per request with prefix caching on at
`block_size 784` — it bounds concurrency the way a KV cache bounds context, and
leaving it out lets the simulator admit requests vLLM could not.

Sizing DeepSeek-V3.2 as GQA read 1,748,992 bytes/token where MLA caches 78,324.

**Verify a new family's shapes against its published parameter count.** It is
the one number that catches a wrong shape anywhere in the stack, and it is
public. Summing `calculate_sizes`' weights over the catalog and the resolved
stack: DeepSeek-V3.2-Exp comes to 671.878B, and minus
the DSA indexer (0.852B) that is 671.026B — V3's published 671B, with the
difference being exactly what V3.2 adds.

### Communication sizes for ASTRA-Sim
ASTRA-Sim's data-size convention depends on the collective: AllReduce and
ReduceScatter take the full pre-partition tensor; AllGather takes a local chunk.
Ring divides the former by its rank count, not the AllGather chunk.
- ALLREDUCE on `o_proj` and `down_proj`: pass full output tensor size at
  activation dtype. An FP8-weight checkpoint still reduces its bf16 output
- Dense `tp_output_reduction: all_reduce` entries reduce their full projection
  output at activation dtype, not the sharded input. For Qwen `gdn_out_proj`,
  that is effective token rows × hidden size × activation bytes.
  No reduction at TP1; the declaration applies only when the row is emitted.
  Historical and packed GDN layouts both use this declaration.
- MoE: **not** one size. Dispatch AllGathers take per-rank chunks, while the
  combine ReduceScatter takes a pre-scatter total. Native dispatch has three
  tensor collectives; the legacy path combines its dispatch payload. See the MoE section.

An MoE entry's `attention_sequence_parallel: true` requires `sequence_parallel: true`.
Serving refuses its TP>1, PP1, DP+EP-spanning deployment before trace emission:
attention-wide sequence shards need different gathers/scatters and local row counts,
not a wrapper-only restore. The guard is not certification of every other topology.

### Multi-dimensional topology and `involved_dim`
For DP configurations the network topology is multi-dimensional, innermost dimension
first: `npus_count: [tp_size, dp_group_size]`, or `[tp_size, pp_size, dp_group_size]`
when `pp_size > 1`. This mirrors vLLM's rank layout
(`parallel_state.initialize_model_parallel`: `all_ranks.reshape(-1, dp, pp, pcp, tp)`);
the `pp_size` dimension is omitted when it is 1, so existing DP+TP configs keep their
2-D topology and their recorded baselines. Collectives are scoped to specific
dimensions via the `involved_dim` BoolList attribute on COMM_COLL_NODE protobuf nodes:
- ALLREDUCE (TP): the TP dim only — `[True, False]`, or `[True, False, False]` with PP
- EP: the DP dim, plus the TP dim when EP spans past one instance's GPUs —
  `[False, True]` / `[True, True]`, or `[False, False, True]` / `[True, False, True]`
  with PP. **Never** the PP dim: vLLM's EP group is
  `all_ranks.transpose(1, 2).reshape(-1, dp*pcp*tp)`, and that transpose pins the PP
  index, so experts are sharded across the DP x TP ranks of one pipeline stage.
  Marking PP involved would drag the other stages' NPUs into a collective they never
  join in vLLM

A DP round only completes if **every NPU of every member instance runs it**.
`Scheduler.add_done` enforces this by refusing to complete a batch until both
`start_npu` and the instance's last NPU appear in `batch.end`, and **nothing raises
when that cannot happen** — the batch simply never finishes and the group's collective
blocks forever. Any code path that creates or serves a batch the start NPU cannot
claim is therefore a silent deadlock; that was issue #65. `schedule()` lets the start
NPU fall through to `_schedule_existing` when the pipeline is full for exactly this
reason, and a DP batch is not servable to any NPU until the barrier has stamped its
`workload_name`.

The `involved_dim` is encoded in the trace `comm_type` field as `ALLGATHER:0,1` (parsed by
the Chakra converter's `_parse_comm_type`). ASTRA-Sim's `Workload::issue_comm()` reads this
and passes it to `generate_all_to_all()`, which skips dimensions where `involved_dim` is false.

The `system.json` collective implementations must have one entry per topology dimension
(e.g., `"all-to-all-implementation": ["ring", "ring"]` for 2D, three entries for 3D).
`config_builder.py` generates this automatically from the topology it emitted.

### Group-limited expert routing
RR uses each token's ordinal in the full gathered/replicated batch, not a
constant zero or a counter restarted at each source partition. Unrestricted
RR chooses `(t + j) % E` for `j = 0..k-1`; grouped RR uses the same advancing
ordinal to select groups and experts. RR resets at each layer invocation,
so block copy is exact for it at a fixed batch shape. RAND draws and the
BALANCED/CUSTOM analytical paths are independent of this ordinal fix.

DeepSeek-V3/V3.2 and GLM restrict a token's experts to `topk_group` of
`n_group` groups (`deepseek_v2.py` passes `num_expert_group=config.n_group`,
`topk_group=config.topk_group`, both defaulting to 1). `GateRouter` reads both
off the checkpoint. It matters because it changes how many EP ranks one token
reaches, and therefore the per-rank MoE token count and the EP collective size:
DeepSeek-V3.2 at EP=8 sends a token to 45.4% of ranks against 66.2%
unrestricted (GLM-5's figure, same `E` and top-`k`). **Only DeepSeek-V3.2 actually restricts** — GLM-5 ships
`n_group: 1` and no other family declares the fields.

`_hit_probs` is exact and deterministic: a DP over which groups the token
selected, not sampling. It draws **without replacement**, matching
`torch.topk`; the older `1 - ((ep-1)/ep)**k` modelled independent draws and
read ~1% low even with no grouping. Don't "simplify" it back — the difference
is measurable, and the grouped and ungrouped cases must not have two different
answers to one question.

### Distinct expert counts and measured routing

Whole-block MoE lookup distinguishes gathered tokens, expert-token assignments
and distinct activated experts. With independent tokens choosing uniform top-k
experts, the expected distinct count per EP rank is:

```text
activated_per_rank = (E / ep) * (1 - ((E - k) / E) ** n)
```

At one token the global count is k and the expected per-rank count is k/ep;
the latter is not an integer placement for a particular token. Runtime counts
are rounded and bounded for lookup. BALANCED is an expectation, not a measured
routing histogram. Do not replace distinct counts with n*k/ep: repeated expert
selections collide, so assignments and distinct weights have different scaling.

`bench run --record-gate-stats --enforce-eager` records a trained gate's counts.
`bench/core/gate_stats.py` reduces them into `gate_stats.json`, consumed through
`--expert-routing-policy CUSTOM --gate-stats`. The curve describes the recorded
weights, inputs and batching, not a universal correction for that architecture.

- Interpolate between recorded batch sizes and clamp at the endpoints. The
  measured count is global; the balanced per-rank estimate divides it by EP.
- Check expert count, top-k and model basename before consuming a curve. The
  basename check is a typo guard, not full checkpoint identity verification.
  Missing, unreadable or mismatched curves warn and fall back to BALANCED.
- Aggregate only between explicit workload start/end markers. Concentrated
  routing with `distinct <= top_k` is valid and must not be classified as
  warmup. Reject unmarked, repeated-boundary and incomplete raw logs.
- The observer requires eager execution and synchronization. Mark its latency
  diagnostic and compare serving latency against a separate uninstrumented run.
  Recording does not change the default routing policy.
- Forced routing can change expert-load histograms and grouped-GEMM kernel
  selection. It is not a control that holds everything except the distinct
  count fixed, and end-to-end agreement cannot bound an individual MoE table's
  error without separating other costs.
- Historical whole-block measurements and native components have different
  contracts. Neither a measured count curve nor matching CSV columns establish
  arbitrary-histogram or deployment coverage.

### MoE dispatch/combine and TP restoration

The modeled transport is vLLM's `allgather_reducescatter` backend.
A matching native component table is preferred; otherwise the retained
whole-block `moe.csv` path is used with a warning for unverified DP coverage.
Other all-to-all backends need separate execution contracts.

Native DP+EP modular execution keeps local gate/routing, three ordered AllGathers
(hidden states, top-k weights, top-k IDs), gathered expert computation,
ReduceScatter and local finalization separate. Tensor widths follow the
recorded dtypes. Sequence-parallel wrappers dispatch over EP and restore the
TP output with AllGather; non-SP wrappers dispatch over DP and restore it
with AllReduce when TP is greater than one. Do not describe this path as a
single hidden-state/router-logit payload.

The legacy fallback approximates dispatch with one hidden-state/router-logit
AllGather and combine with ReduceScatter. At DP=1 there is no dispatch/combine:
replicated input reaches local experts, followed by a TP AllReduce if needed.
Sequence-parallel legacy traces also carry their trailing TP AllGather.
Chakra accepts multiple collective/size pairs on an expert marker and chains
them in order, preserving the tail through skipped expert ranks. Reinstall
the converter with the matching backend; the TP restore is not a missing feature.

For unequal DP counts, retain the full already-padded vector. SP wrappers
ceil-divide each member by TP before gathering; the total and minimum alone
cannot reconstruct that geometry. The Ring envelope uses the gathered row
count minus the smallest dispatch count. For a group of G ranks and tensor
width W bytes, AllGather uses ceil(remote_rows * W / (G - 1)) as its local
chunk; ReduceScatter uses the hidden-state chunk multiplied by G. Equal counts
recover the ordinary local chunk. This is an analytical worst-rank envelope,
not an exact model of grouped ragged NCCL execution or a measured guarantee
for every rank count.

Expert lookup uses all gathered/replicated rows and the appropriate rank's
active experts, not the number of unique tokens with a local expert from
`RoutingResult.local_tokens`. Local gate/finalization and gathered expert
work must keep separate row domains on the native path. Attention/request
geometry and head rows remain separate from model-forward padding.

### MoE expert blocks
Routing vectors use global EP ranks. Each DP member's trace uses
`global_rank = dp_rank * local_ep + local_rank`, where `dp_rank` is its
position in the DP group, never its global instance ID. Both wave-completion
paths must pass that position through ordinary and interleaved synthesis.
PP does not enter this index: each stage has its own EP group. Keep
`EXPERT {local_rank}` markers local, because the converter selects them
within an instance. Equal BALANCED vectors can hide an incorrect offset;
asymmetric routing is required to check rank selection.

Expert blocks use `EXPERT {i}` / `EXPERT END` markers for ASTRA-Sim. Each EP rank
gets a latency from the gathered/replicated token count and its own activated
experts (`key_0=tokens, key_1=activated_experts`, profiled at tp=1). Ranks execute
in parallel and sync at the dispatch/combine collectives. Expert-to-rank assignment uses even
partitioning: `expert_id * ep_size // num_experts`.

### DP+EP wave synchronization
For DP groups (instances with the same `dp_group`), wave synchronization is achieved
through two mechanisms:
1. **Python-side dp_pending barrier**: trace generation is deferred until all DP group
   members have scheduled their batches, and the EP collectives are sized from
   the round's gathered total. `dp_pending[dg][inst]` is a
   **FIFO**, not one slot: at `pp_size > 1` a member can have up to `pp_size` batches
   waiting, and a round pairs the members' *j*-th batches, mirroring vLLM, where DP
   rank A's *j*-th forward joins the same collective as rank B's *j*-th
2. **ASTRA-Sim collective barrier**: all DP group instances' `.et` files are placed in a
   shared workload folder. The EP collectives in both files have matching stream
   IDs, causing ASTRA-Sim to block until both NPUs reach the collective.

**Local graph padding precedes DP synchronization; it also applies at DP=1.**
`serving/core/cudagraph.py` resolves the target MRV1 capture grid and local
FULL/PIECEWISE/NONE mode by default, or MRV2 when selected explicitly or by a
matching serving observation. Both DP completion paths and the independent-instance
path share that resolver. Small prefills can use PIECEWISE graphs. The common
DP mode is the minimum of the local modes: non-NONE broadcasts the largest
locally padded size; NONE retains each local padded count. With an 8-token
capture point and maximum 256, `[6, 1529]` therefore becomes `[8, 1529]`.
Do not replace this with a threshold-only `max <= cap` rule.
MRV2 instead returns to raw local token counts when any DP rank selects NONE;
otherwise it redispatches using the common token maximum and decode eligibility.
Its FULL decode candidate sizes are distinct from PIECEWISE token sizes. A first
one-token prefill is not decode-eligible; a one-token extension over cached history
can be. Preserve historical MRV1 behavior when no target runner was recorded.

The per-instance `cudagraph` object describes deployment behavior independently
of the profiler's eager settings. Defaults derive the grid from scheduler
limits, speculative query length and hardware compute capability; explicit
mode/grid/cap, runner and compiler-SP overrides describe other targets. Use the effective
mode after attention-backend resolution. LoRA keys, cascade/encoder restrictions,
DBO, dynamic/variable speculative decoding and drafter graph dispatch are not inferred. Attention offloading is
outside this GPU graph contract and uses NONE.

Bench startup may record a read-only worker `execution` observation after graph
capture, separately from the frontend's resolved configuration. Unsupported or
incomplete observations remain unknown. Preserve complete capture lists even when
they exceed generic diagnostic list limits. Collect every DP engine's response;
the public load-balancing RPC may return only the first engine's result.
`profiler import-execution` can save
a matching observation under `meta.yaml.serving_execution`; require matching
model configuration, device, precision and deployment scope. Resolve local
declarations through the worker's native config class
before binding aliases/defaults to effective model fields; do not maintain an
offline model-specific config alias table. Keep only the matched declaration,
not a duplicate expanded config. A custom variant label needs explicit matching
acquisition precision. Keep one observation
per scope, not an execution history. Explicit instance `cudagraph` settings win;
otherwise serving uses a matching observation or derives settings with a warning
when saved observations do not match. Versions are provenance, not execution locks.
Acquisition metadata stays separate: importing graph settings neither changes
latency CSVs nor certifies eager measurements as graph measurements. Reduced-stack
diagnostics must not become full-model serving references.

Padding changes model-forward rows, not requests, query/KV lists, decode counts
or head rows. Keep those domains separate in trace lookup and completion
accounting. This is a shape rule, never a benchmark-fitted graph-time correction.
This does not mean vLLM always executes attention without padding. MRV1 sets
`pad_attn` for FULL graphs and passes padded token/request dimensions into
attention metadata; PIECEWISE keeps real dimensions there. Padded request
query boundaries repeat, sequence lengths are zero, and KV slots are invalid.
QKV/norm/RoPE, output projections and MLP/MoE still see padded forward tensors;
selected logits rows do not. Serving retains real attention geometry and does
not separately model FULL-graph empty-slot/backend scheduling overhead.

For non-speculative execution, `_build_batch_ctx` uses the number of actual
requests for per-sequence head lookup, tensor sizes and TP logits gathering.
vLLM selects `hidden_states[logits_indices]` outside the padded model forward;
an idle DP `_dummy_run` returns hidden states without logits or sampling.
`_emit_final_layers` therefore retains the final norm but skips per-sequence
head entries on idle ranks, including their power costs. Keep a zero-byte
REMOTE output on the final trace row so Chakra can emit its terminal store
without transferring nonexistent token IDs. Speculative execution retains its
existing head/drafter row contract: changing it independently can skip
DP-synchronized drafter collectives.
Batch-context construction participates in the skew lookup fingerprint. Refit
enabled bundles through `profiler refit-skew` when it changes; a head-only fix
can refresh the reference identity without changing any numerical CSV values.
Never silence the stale-fit guard by manually replacing a fingerprint.

Idle-head omission also requires communicator-scoped collective numbering.
When Chakra supplies `involved_dim` rather than an explicit communicator,
ASTRA-Sim allocates stream tags from separate dimension-mask counters, with
distinct tag residues for overlapping scopes. A TP-only logits gather must
not advance the next EP collective's sequence. Disjoint groups may reuse a
scope's tags because matching includes source and destination ranks. The
allocator lives on `Sys`, so its counters persist across batch graphs; it
fails on namespace exhaustion rather than wrapping. Explicit communicator
handling and all timing parameters remain unchanged. Rebuild ASTRA-Sim with
this frontend change; omitting the head against the old global counter can
deadlock a later EP wave when DP members have different head participation.

When one DP instance is idle (no requests), a dummy batch (one decode query,
`1 + num_speculative_tokens` tokens, matching `GPUWorker.execute_dummy_batch`) is created
so it can participate in the sync. When one instance finishes all requests, it
continues generating dummy batches until all DP group members are done.

The dummy is gated on `len(inflight) < pp_size`, not `== 0`: vLLM requires every rank
of a DP group to run the same number of forwards, and with PP a rank has `pp_size`
microbatches in flight, so the gate is `schedule()`'s own pipeline-depth rule. Gating
on `== 0` lets the member holding a real request run `pp_size` batches ahead of a round
the idle members can never join. **Any** NPU of the instance may open a round (an empty
`dp_pending[dg][inst]` keeps it to one per member per round: nothing queued means the
member has not joined the round being assembled) — which NPU ASTRA-Sim asks about
is not ours to choose. Because of that the dummy is appended to
`schedulers[i].inflight` the way `_build_batch` registers a real batch, and the start
NPU must be able to join it (see the `involved_dim` section above).

A DP batch is the one place where the simulator cannot do what vLLM does and
schedule-then-dispatch in one step (`schedule()` then `execute_model()` in
`step_with_batch_queue`): its graph needs the group-wide padded `max_total_len`, so
dispatch waits for the barrier. The invariant that survives is that the dispatch still
lands **before the next `schedule()` for that NPU** — `dp_ready_workloads` is keyed by
the NPU that opened the round (`batch.fired[0]`) and is served ahead of `schedule()`.
Letting a new build take that poll instead is what hung `dp>1 x pp>1`: the NPU ran the
*second* microbatch's graph as its first iteration, `add_done` credited it to the first
by `id - 1`, and the other pipeline stage waited forever on a RECV that never came.

### Chakra graph converter
The Chakra converter (`astra-sim/extern/graph_frontend/chakra/src/converter/llm_converter.py`)
transforms text traces into protobuf `.et` files. It creates:
- `MEM_LOAD_NODE` for the first layer's input (from REMOTE/CPU memory)
- `COMP_NODE` for each computation layer
- `MEM_STORE_NODE` for the last layer's output (to REMOTE/CPU memory)
- `COMM_COLL_NODE` for ALLREDUCE / ALLGATHER / REDUCESCATTER (with optional `involved_dim` BoolList attribute)

The converter parses `comm_type` strings like `ALLGATHER:0,1` via `_parse_comm_type()`,
splitting into `comm_type="ALLGATHER"` and `involved_dim=[False, True]`. It also accepts
`ALLTOALL`, which nothing emits -- see the MoE section.

The MEM_STORE node uses the **last layer's** `output_memory_loc` and
`output_memory_size`. This is why the sampler (not lm_head) must have
`output_loc=REMOTE:{node_id}`: what goes back to the host is the sampled token ids
(4 bytes per sequence), not the logits the sampler read.

Memory location types: `LOCAL` (NPU) = 1, `REMOTE` (CPU) = 2, `CXL` = 3, `STORAGE` = 4.
These must match the C++ enum in `astra-sim/astra-sim/system/AstraMemoryAPI.hh`.
