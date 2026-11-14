# profiler/AGENTS.md

Profiler guidance for AI coding agents. Repository-wide rules are in the root `AGENTS.md`; simulator-side contracts are in `serving/AGENTS.md`.

## Architecture Patterns

### Profiler (`profiler/`)
The profiler uses lightweight PyTorch/Kineto capture via a worker extension class.
Explicit dynamic module scopes own native CUDA activities through CPU launch
correlations; CPU duration is never a latency input. No model-specific collector
branches, Python stacks, tensor shapes or memory snapshots are required. Architecture is
dispatched by the HF config's `model_type` field against YAML catalogs under
`profiler/models/<model_type>.yaml`, which bind canonical layer names (dense /
per-sequence / attention / linear-attention / moe) to vLLM class/kernel selectors
or one exact native CPU operator name. Operator CPU intervals establish launch
ownership only; their CUDA activities supply all measured latency.

**Catalog naming rule.** The file is named after the `model_type` it primarily
serves, spelled **verbatim** — including where vendors disagree: DeepSeek
writes V3.2 as `deepseek_v32`, Qwen writes 3.5 as `qwen3_5`, so the files are
`deepseek_v32.yaml` and `qwen3_5.yaml`. Copying upstream's spelling is what
keeps the rule mechanical; normalising it would mean the filename matches no
`model_type` and every lookup falls through to the directory scan. `model_type`
is read from the top level of the config handed to the profiler. Preserve the
shape that the checkpoint's config class consumes: Qwen's text tower is
flattened to `qwen3_5_text`, while MiniMax-M3 must retain its nested
`text_config`. Do not apply one family's flattening rule to another.
When several `model_type` values share a compatible implementation, one file
lists them under `model_types:` rather than duplicating or symlinking the catalog.
The recorded vLLM 0.28 DeepSeek-V3.2/GLM-5 decomposition shares `deepseek_v2`;
that is not a version-independent runtime claim. vLLM 0.31's native DeepSeek-V3.2
path is `vllm.models.deepseek_v32.nvidia.model`. Existing aliases and stored tables
do not certify acquisition on that path: require matching live coverage and
execution contracts before a new sweep. Two files claiming one `model_type` is
an error, not first-wins, including a matching filename alongside another file's
alias. Check all claims before choosing a catalog; a file's own alias is not
a second claimant.

**One layer-order form: `blocks:` + `shared:`.** `blocks:` is keyed by **axis**
(`blocks.attn.<layer_types value>`, `blocks.sparse_attn.<same>` as an overlay,
`blocks.mlp.dense|moe`) rather than by block name, because a layer's identity
is a tuple and enumerating combinations explodes. `shared:` holds `prologue`
and `head`, which run once per iteration. A uniform stack is the degenerate
case — one entry per axis.

There was a second form, `sequence:`, for uniform stacks. **It is gone; do not
reintroduce it.** It was this one flattened (`pre_attn`/`post_attn` = the single
implicit `attn.full_attention`, `mlp_dense`/`mlp_moe` = the `mlp` axis with the
value in the key name), and two forms meant two code paths — the simulator only
implemented the flat one, so Qwen3.5 and MiniMax-M3 could not be simulated at
all. The flattening also encoded a false claim: with the axis in the key there
is no per-layer question left, and the MLP was resolved once per *model*, which
modelled DeepSeek-V3.2's and GLM-5's first three dense layers as MoE.

Which block a given layer runs comes from the *checkpoint's* config
(`layer_types`, `first_k_dense_replace`, `decoder_sparse_step`,
`moe_layer_freq`, `sparse_attention_freq`, `index_topk_pattern`), never from
the yaml. `profiler/core/stack.py` owns those rules and **both** the profiler
and the simulator import it — the profiler to decide how many layers to
instantiate, the simulator to decide which block each layer emits. Two
implementations would drift, and the rules disagree on the off-by-one in
opposite directions between vendors. The profiler automatically resolves the
smallest prefix that instantiates every block type required by each category —
4 for a Qwen3.8-27B category needing every attention type. Leave
`--num-hidden-layers` unset to use that resolution; an explicit lower count can
omit required block types.

**How finely to divide is mostly not a choice.** An entry exists iff the
profile tree holds a node carrying CUDA time that no other entry claims, and
the collector retains invoked modules and their descendant CUDA activity — so
vLLM's implementation sets the count, and the same operation needs seven
entries in one family and one in another. DeepSeek-V3.2's `Indexer` is an
`nn.Module` container (`self.wq_b = ReplicatedLinear`,
`self.wk_weights_proj = MergedColumnParallelLinear`, `self.k_norm = LayerNorm`,
plus `view`/`cat` in its `forward`), so it decomposes into 7; MiniMax-M3's
`MiniMaxM3IndexerTritonImpl` is a `forward` and nothing else, with its index
projections folded into `MinimaxM3QKVParallelLinearWithIndexer`, so it is 1.
That is also why `*_glue` exists only where the forward does eager tensor work
between module calls — Llama, Mixtral and Qwen3 blocks are pure module chains.

What *is* a choice is which level to bind, since a parent node's time already
covers its subtree. Entries may merge only when all three hold: **one scaling
axis** (categories cannot mix — `Indexer` cannot be one node because its
projections scale with tokens and its scoring with kv, and binding the parent
forces the 8,643-shot attention grid on work that `dense` covers in 152);
**nothing pinned** (`o_proj`/`down_proj` carry the TP all-reduce after them,
`attention`/`sparse_attention` have PIM swapped in front); **no `tp_stable`
mixing** (that flag buys one sweep at TP=1 plus replication). `moe` is the rule
in the other direction: one entry for the whole block, because everything in it
scales with tokens.

**Finer than the rule requires is allowed and usually right.** DeepSeek's 7
indexer entries could be 3. They are not merged because a per-entry number can
be checked against physics and a composite cannot — `lm_head` was caught at 9x
against `vocab * hidden * bytes / mem_bw`. Coarsen when trace width actually
costs, and coarsen the **trace** rather than the catalog: measurement fidelity
and graph size need not share a granularity.

**Write a catalog from a live profile dump, not from vLLM's source.** The
module tree and the profile tree differ both ways: `rotary_emb`, `q_norm`,
`k_norm` and `RMSNormGated` are real modules that never become profile nodes
(binding them silently measures nothing), while raw CUDA kernels that are not
modules *can* be bound by name (`vllm: _causal_conv1d_fwd_kernel`) and are the
only way to reach gated DeltaNet's conv and decode recurrence. Kernel identity
also changes with the batch regime — a GDN block runs one set for pure prefill,
another for pure decode, a third for a mixed batch — so a catalog written from
a single shot binds the wrong kernel for the others.

Kernel wildcards must not reclaim a bound module's descendants. DeepSeek/GLM
`indexer_glue` excludes `LayerNorm` and `SparseAttnIndexer`, whose copy/fill
kernels are already included in `indexer_k_norm` and `indexer`. A coverage
report with no gaps does not establish non-overlap: coverage counts a bound
parent once, while latency extraction continues into its children. Audit
nested bindings on the live tree, including across categories; a dense-only
slice still must not claim work owned by an attention entry. Refresh old
`indexer_glue` rows through dense acquisition after changing this guard.
Phi-MoE's `block_glue` likewise excludes `PhiMoE`, `LayerNorm` and
`PhiMoEAttention`; only the decoder's direct residual work belongs there.
Acquisition rejects shared CUDA activity across canonical bindings. Old Phi
glue measurements require a dense-category refresh, not a metadata restamp.

For `deepseek_v32` and `glm_moe_dsa`, normalize the native `layer_types` cache
label `deepseek_sparse_attention` to the catalog's full-attention block axis.
Sparse indexer selection remains a separate axis; this normalization changes
neither native execution nor the stored source config, and admits no new backend.

Every TP degree is profiled on a **single GPU**: the engine is always booted with
`tensor_parallel_size=1`, and per-rank shapes are emulated by dividing `SHARD_FIELDS`
by TP via `hf_overrides`. Collective timings are left to ASTRA-Sim.
Include separately declared `dense_intermediate_size` in this sharding step;
do not leave a dense branch at full width while reducing the routed branch.
Resolve the unsharded local configuration through installed vLLM first, after
user overrides and decoder-prefix materialization. `hooks/tp_shape.py` preserves
the native head width when reducing the head count would otherwise change it;
validate both the sharded candidate and any scalar-width pin by materializing
and reloading the raw config through vLLM, not by updating an already resolved
config object. A nested backbone can derive `head_dim` without declaring it in
JSON; write an admitted pin into that backbone and consume the forwarded key.
Unsupported special-width changes fail rather than guessing. Nested text
fields and user shape overrides feed the same sharding step. Strict dimension
divisibility remains; this does not add replicated-KV-head TP emulation.
Incorrectly shaped timings require fresh acquisition, including affected dense
projections, attention and skew; use a separate output root or explicit category
replacement. Source edits alone do not invalidate ordinary/skew resume identities.
Consume nested backbone-only overrides after materializing them in the local
config; do not reapply them to wrapper properties that may be read-only.
Keep independently declared root fields, including shared root/text token IDs,
forwarded. Validate the final native config, not only the materialized dict.
Recorded engine kwargs are not a complete resolved model configuration.
`probe_limits` snapshots the native text config without another RPC; persist it
as `engine_resolved.per_tp[tp].model_text_config` only for authoritative main
engines. Preserve other TP entries and MTP/short-stack refresh semantics. This
text snapshot does not certify the wrapper, backend or parameter layout, and
missing historical snapshots must not be reconstructed.
Ordinary and skew acquisition metadata record resolved geometry and preparation
semantics. Remeasure wrongly shaped rows, never restamp them as corrected.
TP-stable replication applies to every catalog, not only coordinated layouts.
Preserve copied rows and their source identities during local acquisition;
exclude them from measured-shot resume coverage and from local replacement.
Layout discovery probes remain subject to their own acquisition contract.

**The stack is shrunk per category, not per model.** Instrumentation, warmup
and forward work scale with executed operations. Runtime estimates must come
from a bounded acquisition on the intended model/backend/grid, not historical
Python-call profiler timings.

A category must not pay for an axis it does not measure. The profile tree
merges same-class siblings, so a second layer of a type already present adds
**no information**, only its op count on every shot. `Category.stack_axes`
declares the axes (`attention` / `linear_attention` → `(attn, sparse)`,
everything else → all), `stack.minimal_layer_count_for` answers per axis set,
and `run_full` boots one engine per distinct depth — deepest first, since that
engine's shapes are what `meta.yaml` and `--attention-max-kv` must describe.

Where it lands, from each checkpoint's own config:

| model | all axes | attention only |
|---|---|---|
| DeepSeek-V3.2, GLM-5 | 4 (`first_k_dense_replace 3`) | **1** — every layer has the same attention |
| Qwen3.8-27B | 4 | 4 — gated DeltaNet three times, then full attention |
| MiniMax-M3 | 4 | 4 — non-sparse three times, then sparse |
| Llama, Qwen3, Mixtral, PhiMoE | 1 | 1 |

Only DeepSeek and GLM-5 gain, and only on `attention` — which is the sweep
that dominates a run. **1,057 → 341 ms per shot, 3.1x measured.** Splitting a
category further does not help: cost is per forward and a forward runs the
whole stack, so measuring `dense`'s MLP entries in a separate 1-layer engine
would mean two sweeps rather than one, even though Qwen3.8's MLP axis is
uniform.

Two things to know.

Fewer layers means a larger `num_cache_tokens`, which the feasibility filters
read, so **more shots pass**. Coverage widens and the grid is not row-for-row
comparable with a deeper run's.

And **axes are a conservative proxy** for what a category needs, which is
really "every block its entries live in". The proxy never over-shrinks, so no
data is lost, but it does under-shrink: DeepSeek-V3.2's `dense` entries sit
only in `attn.full_attention` and `mlp.dense` — `moe` is its own category — so
layer 0 alone would serve, while the all-axes answer is 4 because the MLP turns
to MoE at layer 3. Qwen3.8's `linear_attention` is the same story. The slack is
deliberate: `attention` is 8,643 shots against `dense`'s 152,
`linear_attention`'s 78 and `per_sequence`'s 40, so the axis rule already
captures every minute that matters and an entry-to-block resolver would save
three or four.

The reason a category cannot simply be shrunk to one layer is not that its
entries are hard to separate — it is that **at one layer some of them do not
exist**. A 1-layer Qwen3.8 instantiates only the gated-DeltaNet block, so
`qkv_proj` and `o_proj` are never built, their rows are missing from
`dense.csv`, and the simulator charges those layers zero.

**Every field vLLM shards has to be in that list**, or one part of the model is
measured per-rank while the rest is measured at full size, and a mixed
measurement is worse than either. `linear_num_key_heads` /
`linear_num_value_heads` are there because
`MambaStateShapeCalculator.gated_delta_net_state_shape` divides both `conv_dim`
and `num_v_heads` by the TP world size; without them, Qwen3.8-27B at TP=2 had a
halved attention page against a full-size mamba page and the engine resolved
`block_size` 1568 instead of 784 — a rank that exists in no deployment. The
tell is the per-TP `block_size` the run logs: on a hybrid it should be the
*same* at every degree, since both pages scale together. The model's full `config.json` (read from
`configs/model/<org>/<name>.json`, or auto-fetched from the HF Hub on first run)
is written to a tmpdir at spin-up so vLLM never needs Hub access.
When reducing `num_hidden_layers`, retain the first N entries of an explicit
`layer_types` list, in flat and nested text configurations. Validate that N is a
positive integer within that list; retain the original model config unchanged.
If `hf_overrides` explicitly supplies `layer_types`, slice that override too:
vLLM applies it again after reading the materialized config.

Attribution: the original layerwise-profile methodology (worker-extension hooks
and single-GPU TP emulation via `hf_overrides`) is
adapted from [@waneon](https://github.com/waneon). The unified 4D attention
sweep, the heterogeneous-decode skew sweep in `profiler/core/skew.py`, and
the offline reference-aligned skew calibration in `profiler/core/skew_calibration.py` are
developed in this repo.

Acquisition writes the selected categories into a bundle. Historical whole-block
MoE tables may remain alongside new native component measurements:

```text
perf/<hw>/
├── hardware.yaml                 # separately acquired device/interconnect facts
└── <model>/<variant>/
    ├── meta.yaml                 # shared and per-TP acquisition records
    └── tp<N>/
        ├── dense.csv             # layer, tokens, time_us
        ├── per_sequence.csv      # layer, sequences, time_us
        ├── attention.csv         # layer, prefill_chunk, prefill_key, n_decode, kv_decode, decode_q_len, time_us
        ├── linear_attention.csv  # layer, prefill_tokens, n_decode, time_us (hybrid only)
        ├── mtp.csv               # layer, sequences, time_us (opt-in drafter pass)
        ├── moe.csv               # historical whole-block fallback; no new acquisition
        ├── skew.csv              # raw heterogeneous-decode measurements
        ├── skew_fit.csv          # derived per-bucket alpha table
        └── moe_components/       # per-deployment components.csv and optional raw checkpoints
```

**A refresh rewrites only what it measured.** `meta.yaml` describes more than
any one `slice` covers, so `persist_meta` takes two authority flags and carries
everything else over from the file:

- `engine_effective` / `engine_resolved` describe the **deepest main engine** —
  the one whose shapes the simulator will run, which is why `run_full` probes
  limits there. A `--profile-mtp` refresh boots a different model (one extra
  full-attention layer, a conv state widened by `num_speculative_tokens`), and
  a category slice may boot a shallower stack. MTP and shallower slices are
  not authoritative; an ordinary slice whose required depth equals the
  all-axis depth updates the selected TP's main-engine metadata.
  Letting the MTP pass write them put `block_size: 800` and half the KV cache
  into Qwen3.8-27B's bundle where the main engine resolves **784** and
  19,839,182 — and `_resolve_block_size` reads that block size whenever
  `--block-size` is omitted, which is the documented way to get the profiled
  value. Three of the four modern bundles had it.
- `attention_grid` may only be rewritten by a run that swept attention. It
  records **`decode_q_lens`** explicitly: row counts do not establish requested
  query-length coverage, and the once-only `q=1` pure-prefill rows do not imply
  that ordinary `q=1` decode was requested or measured.

An empty `limits` argument now **preserves** `engine_resolved` rather than
erasing it, which is how the MTP refresh could delete what the main engine
resolved.

`attention.csv`, `skew.csv` and `skew_fit.csv` are keyed by **`layer`** as well:
a sparse-attention model has two or three kernels in that category and they
share neither a latency curve nor an alpha (on MiniMax-M3 the same batch fits
0.24 / 0.74 / -0.01 for `attention` / `indexer` / `sparse_attention`). A bundle
profiled before those columns existed holds one kernel and it is `attention`.

`<variant>` retains checkpoint-derived naming and separates explicit activation/KV
overrides (e.g. `bf16`, `fp8-actfp16`, `fp8-kvfp8`) unless `--variant` is set.
Ambiguous precision or nested HF overrides require an intentional label; names
are not complete resolved-precision contracts. Times are in **microseconds**. Layers marked
`tp_stable: true` in the yaml (layernorms, sampler) are profiled once at TP=1.
The writer copies stable `dense` and `per_sequence` rows into other `tp<N>/`
folders; stable MTP rows stay at TP1 and are looked up there by the simulator.

**`tp_stable` is about the tensor a layer walks, not the weight it owns.** The
test is whether the *work* is TP-invariant, and a norm's weight can be
replicated while its input is sharded. Qwen3's `qk_norm` was declared stable on
the strength of `RMSNorm(head_dim)` -- head_dim does not depend on TP -- but
`qwen3.py` applies q_norm/k_norm to `q`/`k` of size `num_heads * head_dim`
where `self.num_heads = self.total_num_heads // tp_size`, i.e. this rank's
heads. So the work halves at TP=2 while the sibling `layernorm`, which
normalises the replicated hidden state, does not. Declaring it stable charged
the TP=1 figure at every degree: a TP=2 re-measure reads **0.52-0.63x** of it
(45.0 us against 87.0 at 2048 tokens on Qwen3-32B), worth +0.6 pp of a mixed
step.

The audit that settles it for a new catalog is one question per entry: what
does the kernel read? Every other `tp_stable` entry in the shipped catalogs
passes -- `layernorm` / `final_layernorm` / MTP norms walk the residual hidden
state (replicated); DeepSeek's `mla_a_layernorm` walks the q/kv latent behind a
**`ReplicatedLinear`**; its `indexer_k_norm` walks a `head_dim` slice of a
`MergedColumnParallelLinear` that passes **`disable_tp=True`**; and `sampler`
runs on logits that `LogitsProcessor._gather_logits` has already all-gathered
to full vocab. Qwen3.5's and MiniMax-M3's q/k norms are not bound at all (their
kernels are fused into `fusedMiniMaxM3QNormRopeKVInsertKernel` and friends), so
the question does not arise there.

The measurement that catches it needs a **control**: re-measure a genuinely
stable layer at the same degree in the same run. `layernorm` re-swept at TP=2
reads 1.00-1.03 while `qk_norm` reads 0.52 -- without the control, a 0.52 could
be the machine rather than the flag.

The profiler Docker targets **vLLM v0.31.0 / CUDA 13**
(`vllm/vllm-openai:v0.31.0`). Changing the environment pin does not certify
every catalog/backend or remeasure the historical bundles. Every symbol under
`profiler/core/hooks/` is a vLLM *internal* API and is version-specific. v0.28
restructured MoE substantially: the old `FusedMoE` module is gone, replaced by
`FusedMoEFactory` returning a `MoERunner` that owns a `router` (`BaseRouter`)
and a `RoutedExperts`.

Whole-block forced-routing acquisition has been removed. New native component
acquisition retains the original global expert count and top-k. Gate/routing
executes natively; expert-only stimuli vary local active experts and assignments
with balanced local histograms and uniform weights. These are independent
measurement boundaries, not trained-gate numerical equivalence. Historical
whole-block CSVs remain readable and must never be relabelled as components.

### Profiling the drafter (MTP)
A model that drafts with itself keeps its MTP module outside the ordinary
model, and it **cannot be loaded standalone**: the MTP config's `model_type`
(`deepseek_mtp` / `qwen3_5_mtp` / `minimax_m3_mtp`) is produced by
`SpeculativeConfig.hf_config_override` and HF Transformers does not know it.
`--profile-mtp` boots with `speculative_config` so vLLM builds the drafter
alongside the target. It takes **no draft count**: the engine is pinned to
`num_speculative_tokens=1`, so `mtp.csv` holds **one** drafter pass — the unit
the simulator multiplies by its own `--num-speculative-tokens`. Booting at N
recorded N passes and the simulator multiplied again, so the cost came out N²;
that is why the flag is not a number.

The kernels then arrive without new tooling, because the drafter runs inside
`sample_tokens()` (`propose_draft_token_ids` → `drafter.propose`) and the fire
path already calls `execute_model` then `sample_tokens(None)` inside one
native CUDA-activity context. **`profiler coverage --profile-mtp` is how the
`mtp:` sections were written** — it names the unbound nodes with their ancestor
paths, so the profile tree is the source, as for every other block. For the
opposite failure — an entry that binds *too much* — coverage checks overlapping
and structurally unrelated claims. Inspect the profile tree and ancestor paths
as well: unexecuted alternatives and structurally plausible but semantically
wrong bindings still require an audit.

The `mtp` category has **one** axis, the pass's token count: the loop's passes
are decode-shaped at `max_query_len = 1` (`llm_base_proposer.py`), where tokens
and sequences are the same number. Shot counts depend on sequence/token limits,
grid factors, target drafter graph anchors and cache feasibility. Use a bounded
sweep to estimate acquisition time for the chosen model and device.

Four things a catalog here must get right:

- **A guard on every entry.** The drafter's modules are the *same classes* as
  the target's (`RMSNorm`, `ColumnParallelLinear`, `Linear`,
  `DeepseekV2DecoderLayer` / `Qwen3_5DecoderLayer` /
  `MiniMaxM3DecoderLayer`), so an unguarded entry claims the target's nodes
  too. DeepSeek and M3 pin `within` to their predictor classes
  (`DeepSeekMultiTokenPredictorLayer`, `MiniMaxM3MultiTokenPredictorLayer`);
  Historical Qwen tables used `not_within:
  [Qwen3_5DecoderLayer, Qwen3_5ForCausalLM]` because the old Python-call tree
  omitted their kernel-less wrapper. The scoped collector retains an invoked
  `Qwen3_5MultiTokenPredictor` even without a direct kernel; methods bypassing
  `Module.__call__` still need explicit ownership. Recheck every MTP binding
  against the current runtime, rather than assuming historical top-level
  ancestry. Getting this wrong is
  invisible in the CSV: the curve stays smooth and monotone. Qwen's `mtp_norms`
  read **1287 us at one sequence** for two RMSNorms before the guard was right.
- **The decoder block is declared, not listed.** `mtp.prologue` / `mtp.head`
  hold only the wrapper; the block itself is the target's own, replayed, and
  `mtp.decoder_block` says *which* one. **`mtp.head` must name something**:
  the Chakra converter reads the trace's MEM_STORE node from the last entry's
  attributes, and a drafter whose block ends in MoE closes with an
  `EXPERT END` marker — which surfaced as `'Layer' object has no attribute
  'output_memory_loc'` from inside the converter on MiniMax-M3. Every family
  has one to name: the wrapper's norms are a **single merged profile node
  spanning both sides of the block** (Qwen's `pre_fc_norm_*` before `fc` plus
  `norm` after; M3's `enorm`/`hnorm` before `eh_proj` plus `final_layernorm`
  after), and they are same-class siblings so the profiler cannot separate
  them — charging the merged node once after the block is exactly as accurate
  as once before. It cannot be read off the checkpoint:
  vLLM's MTP modules force it per family (DeepSeek/GLM build at layer index
  `num_hidden_layers`, Qwen passes `layer_type="full_attention"`, M3 passes
  `force_sparse_attn=True, force_moe=True`), and the resolved stack has exactly
  `num_hidden_layers` entries so any index into it wraps to layer 0 — dense for
  DeepSeek/GLM, non-sparse for M3, linear attention for Qwen3.8. An axis left
  unset is inherited from the stack, and only when the stack agrees on it.
- **The block is the dominant term.** On Qwen3.8-27B at 4 layers, one drafter
  pass measures 597 us for its `Qwen3_5DecoderLayer` against ~136 us for the
  whole wrapper. Emitting only the wrapper read a pass at a fifth of its cost.
- **The families do not share names or TP behaviour.** All three combine
  projections are 2h→h and all dominate the wrapper, but DeepSeek/GLM use a
  plain `nn.Linear` (unsharded), Qwen a `ColumnParallelLinear` (TP-sharded), M3
  a `ReplicatedLinear` (replicated).

### Historical Python-call normalization trap (vLLM 0.28)
The following describes the older acquisition tree. The scoped collector builds
its own grouped tree but deliberately retains the catalog's normalization and
full parameter-signature identity. It does not reproduce the old tree's pruning.

Worth knowing well beyond MTP: it made **every latency in a 0.28-profiled
bundle 3x too large**, not just the top-level ones. Every profile node is
normally divided by **its parent's** invocation count to get a per-call figure, and the
top level has no parent node to read that from — so the whole subtree under it
inherits the error. Two things go wrong at once under 0.28:

- vLLM merges same-class siblings into one node (time summed, `invocations`
  counting them all) and reports it **once** under a parent that is itself a
  node — but once **per sibling module** at the top level, where the owning
  module launched no kernel and was flattened away. Qwen's three drafter norms
  arrive as three identical top-level entries, and `LogitsProcessor` arrives
  once per timed forward. `extract_samples` now drops repeats identical in
  (full module representation, time, invocations). Do not reduce the identity
  to a class name: differently shaped modules can have equal times and counts.
- The root's own invocation count is the **forward count**, so
  `extract_samples` takes `iterations` and uses it as the top level's
  `parent_invocations`. It used to hardcode 1.

Whole-block MoE is a distinct normalization contract. `_entry_dict` marks the
MoE category with `normalization: invocations`; `extract_samples` divides by
the matched MoE node's calls, not the merged decoder parent's calls. A hybrid
stack's parent includes dense layers that do not invoke MoE. Keep the parent
times occurrence denominator for other categories, including intentionally
merged projection pairs. Remeasure affected stored `moe.csv` grids rather
than applying a model-specific multiplier. Native DP+EP components have their
own per-forward attribution and are not changed by this fix.

Two controlled experiments, same model / hardware / flags, against the trusted
`vllm=0.19.0` bundle for Llama-3.1-8B on RTXPRO6000:

| | value | vs trusted |
|---|---|---|
| `lm_head@1`, before | 6416.67 us | 8.99x |
| `lm_head@1`, dedup only | 2139.95 us | 3.00x |
| `lm_head@1`, both fixes | **713.805 us** | **1.00x** (trusted: 714.006) |
| whole `dense.csv`, both fixes | 1368 rows | mean abs delta 1.5-6.5% per layer, i.e. noise |

The physics check that made it findable at all: `lm_head` reads the whole
output embedding, so `vocab * hidden * bytes / mem_bw` = 583 us is a hard floor
and 6417 us is suspiciously far above that floor. A lower bound alone does
not make a slower measurement impossible; the controlled 9x discrepancy
and upstream normalization explain the error. **A profiled curve gives no other signal** — it
stayed smooth and monotone the whole way, and `profiler coverage` passed,
because coverage reports only what is *un*bound.

**Every category** of all four `vllm=0.28.0` bundles was affected, at a factor
of exactly `measurement_iterations`, and every one was re-profiled. Five
categories confirm the factor independently — `dense` p50 2.9995 (2584 rows),
`moe` 2.996 / 2.999, `linear_attention` 3.01-3.12 across all six GDN kernels,
`per_sequence` by the controlled experiment above, and `attention` by the
KV-bandwidth check (28.5% of spec against 84-95% for every 0.19 bundle). The
±5% spread around 3.000 is the noise between two independent measurement runs,
not structure.

The four `vllm=0.19.0` bundles were never affected, which is what made them
usable as the reference: the fix was verified against them before anything was
rewritten. RTXPRO6000 acquisition versions now vary by model: Llama-3.1-8B,
Qwen3-32B and Qwen3-30B have refreshed TP1/TP2 data recording 0.31, while
other bundles and Qwen3-30B's historical whole-block fallback keep their
recorded provenance. `RTX4090/meta-llama/Llama-3.1-8B` stays 0.19.
**A re-profile can move recorded clocks.** Refresh affected
regression baselines and example validation together when adopting new data;
the suite also includes other architecture families and a DeepSeek diagnostic.
Bandwidth efficiency is the estimated memory-time bound divided by measured
time, not a claim that measured time is below the bound. An outlier requires a
controlled attribution check, not an automatic conclusion that it is a bug.
These normalization checks do not certify all stored rows against later
acquisition-protocol changes.

`num_mtp_modules` is capped to 1 in the config **file**, not via
`hf_overrides`: the drafter reads `speculative_config.draft_model_config.hf_config`,
built from disk. M3's 7 modules are ~103 GB otherwise. M3 additionally needs
`scripts/patches/vllm_m3_mtp_layer_name.py` to start at all.

### A dense model's sampler is invisible on vLLM 0.28
The collector's module scopes follow **module** calls, so a component that
is not an `nn.Module` cannot become a node no matter what it launches. vLLM 0.28
has two samplers and picks between them by model:

| runner | chosen when | sampler | a node? |
|---|---|---|---|
| `vllm.v1.worker.gpu_model_runner` (V1) | MoE, hybrid | `vllm.v1.sample.sampler.Sampler`, an `nn.Module` | yes |
| `vllm.v1.worker.gpu.model_runner` (V2) | **everything else** | `vllm.v1.worker.gpu.sample.sampler.Sampler`, a plain object | **no** |

The rule is `VllmConfig._is_default_v2_model_runner_model`:
`is_default_v2_architecture or not model_config.is_moe`. So **every dense model
profiled on 0.28 silently loses its `sampler` row**, and the simulator then
refuses the bundle with `Missing per-sequence profile for layer=sampler`. The
four 0.28 bundles that existed before this are all MoE, which is why it
surfaced only with the first dense refresh.

Two things make it hard to spot from the outside. The two runner classes share
the name `GPUModelRunner`, so `type(runner).__name__` cannot tell them apart —
only `__module__` can. The historical Python-call coverage passed at **100%**:
the sampler's work never entered that tree. Native-activity coverage now retains
it as an explicit gap unless the sampler facade gives it an owning module.

`profiler/core/hooks/sampler_shim.py` wraps a non-module sampler in an
`nn.Module` named `Sampler`, installed once per worker by `Extension.fire` and
`Extension.coverage`. The name is deliberate: a profile node carries its
module's class name and that name is what a catalog binds, so one
`sampler: {vllm: Sampler}` covers both runners and no catalog changes. The
alternative was a second spelling in all nine of them.

The term is not negligible — 21.6 us at one sequence to 107 us at 256 on
RTXPRO6000/Llama-3.1-8B, against `lm_head`'s 714 to 840. Two independent
checks that the shim measures exactly the missing work and nothing else:
CUDA-event timing around the sampler call (29.5 us at 1 sequence, 63.7 at 128,
90.3 at 256, an upper bound since it brackets the Python call too), and
coverage's own total, which rose by 22.3 / 24.1 / 29.4 us across the three
regimes once the shim was in.


### Skew acquisition and offline calibration

`skew_plan.py` generates workload-independent distributions from configured
bounds and resolved engine limits. It covers history families, request order,
token-frontier batches and equal/unequal multi-prefill splits. Decode query
lengths follow the attention sweep. Planning streams one expanded batch at a
time; feasibility uses page-aligned actual KV allocation and reserves the
sampler's final token. No benchmark inputs belong in the planner.

`skew_support.py` checks the base plan in the compiler's actual partition
coordinates using the attention table. It adds deterministic draws where an
observed N/partition lacks `MIN_ROWS` distinct cases. Selection never reads
skew timing targets or benchmark results. CPU preview and live acquisition
share this logic, including the retained raw data's prefill envelope. Extra
cases retain compact identities and regenerate their request arrays lazily.
Bounded candidate search may leave deficits: `support_completion` records
them separately from acquisition completion. Do not equate completing a
sweep with every lookup cell having direct support. Existing raw data remains
preserved unless `--force` is requested.

`hooks/skew_measurement.py` measures the actual heterogeneous batch through
vLLM 0.28. Three independent contexts, each containing three timed forwards
by default, produce a median of forward medians. CPU scopes establish call
ownership; correlated GPU annotations are not kernel work. Each measurement
context initializes its assigned dummy KV pages before warmup.
Warmups verify finite output and executed geometry.
Recurrent state groups are identified by vLLM KV-cache spec, not model names;
the paged attention groups supply the history/query geometry in hybrid stacks.
Backend metadata need not retain query starts: sparse indexers split and
discard them. During untimed warmup, verify the common metadata passed to the
actual builder and retain the exact returned object's identity. Use that
evidence only when direct backend geometry is unavailable; never skip an
unknown backend by name. Restore builder methods before timing and on failure.
Attention and ordinary categories use the same per-call CUDA activity union as
skew, but retain their category-specific averaging and axes.

`hooks/cuda_timing.py` forms the interval union within each raw module call,
separately per device, before the collector groups compatible invocations. Overlapping
streams on one device count once; different calls must not be pooled before
normalization. Resolve a native activity by correlation, name and exact device
interval, but use CPU launch scopes for ownership. Exclude annotations and
device gaps, reject missing/ambiguous activities, and preserve event coverage.
This is device-active time, not whole-step wall time. Catalog components may
overlap each other, so their summed unions need not equal a global union.
Coverage remains a kernel-work audit, not a union-latency percentage.

New ordinary CSVs contain coordinates and `time_us`, not provenance columns.
Record per-category acquisition identities, method descriptions and actual runtime
versions in variant-root `meta.yaml::per_tp[tp].measurements`; preserve earlier acquisitions when
extending compatible tables. Describe the PyTorch/Kineto collector, per-invocation
device-active union, normalization and dummy input preparation directly, without
numbered protocol branding. Resume compares semantic catalog, geometry, method
and execution contracts, not source-file hashes or exact library versions.
Those versions are provenance, not execution vetoes. Historical source hashes
remain observations; new runtime admission does not collect source-file hashes.
Actual API, backend, shape, dtype, layout, finite-output and coverage guards remain.
Missing or incompatible acquisition metadata requires a separate output root or
explicit remeasurement with `--force`; never restamp historical measurements.
TP-stable replication preserves acquisition provenance and rejects mixed methods.
When a replica-only destination begins local acquisition, promote its category
metadata to the local method while retaining each replica's source-TP identity.
Validate preserved rows against that source record during local resume; copying
TP1 rows never claims a local measurement or completed local coverage.
Fully skipped categories do not acquire a new measurement timestamp. Skew records
its per-forward method, contexts and runtime observations in root `per_tp[tp].skew`;
CPU `plan-skew` uses the same semantic projection without importing Torch/vLLM.
Do not read `skew.meta.yaml`; acquisition and calibration use root metadata only.
Native DP+EP component contracts and skew-fit invalidation remain separate.

Use the variant-root `meta.yaml` for all new profile metadata. Per-TP measurements,
category publication, context/callable records, skew acquisition and native MoE
records belong under `per_tp[tp]`; do not create metadata sidecars beside CSVs.
Already-central `engine_resolved.per_tp`, `attention_lookup.per_tp`,
`component_layouts.per_tp` and `skew_fit.per_tp` retain their existing locations.
The empty `.meta.lock` coordinates metadata writers; it is not a measurement
record. Preserve unrelated sections on updates. Narrow legacy reads do not
authorize restamping historical measurements or infer identity from directories.
Raw repetition checkpoints may support acquisition resume; failure-only error
records and diagnostic traces may support debugging. They are optional, not
always-on metadata/evidence files or simulator inputs, and are not committed
experiment artifacts.

Saved policy, method and format names use descriptive hyphen-separated labels,
not numbered suffixes. `metadata_names.py` holds shared names, not old-name aliases.
Obsolete symbolic contracts require fresh acquisition, not renaming stored records.
Legacy file-layout fallbacks still require current method contracts; unmarked
historical mainstream tables retain their existing reads. Real library/model
versions and numeric structural schema fields remain unchanged.

`hooks/history.py` prepares query request state and completes asynchronous output.
`hooks/dummy_cache.py` initializes assigned attention pages
through the installed vLLM `initialize_single_dummy_weight` with its default
uniform range and per-tensor seed. The backend defines the logical block axis;
kernel subdivisions map back to the request's manager block IDs. Leave
unassigned pages untouched. Reinitialize per context, outside warmup and timing,
so preceding queries cannot become the next shot's synthetic history.
FP8 initialization uses bounded slabs, each with the upstream per-tensor seed,
to limit its temporary FP16 conversion. Supported NVIDIA packed caches initialize
floating payloads with that same helper, then quantize and pack the native format:
`fp8_ds_mla` uses 512 FP8 values, four FP32 scales and 64 BF16 RoPE values in each
656-byte record; `DEEPSEEK_V32_INDEXER` uses separate value/scale planes within
each block (128 FP8 values and one FP32 scale per token), optionally shuffled in
16-by-16 tiles. Do not treat the index cache as interleaved 132-byte token records.
Main scales clamp `amax / 448` at `1e-4`; index scales clamp `amax` before dividing
by 448. Both round the scale upward to a power of two. Never initialize scale bytes
randomly, substitute unit scales or run a native forward to generate this history.

Dispatch by the verified cache spec/backend contract, not the model name. Check
dtype, head/state geometry, sparse role, explicit shuffle mode and byte strides;
unknown packed formats remain refusals. Floating `MLAAttentionSpec` indexer
caches use the typed resolver. A three-dimensional `[blocks, tokens, channels]`
view may omit the head axis only when the spec has one KV head and matching
floating dtype; restore that singleton axis for logical shape validation.
Require non-overlapping strides for this squeezed view: in ascending stride
order, each non-singleton axis must start beyond the preceding axes' extent.
Allow padded and permuted views that pass this check.
Keep existing four-dimensional typed views and strict opaque-byte checks.
A runtime without the sparse-role capability retains the original typed-cache
resolver. Preserve null/unassigned
pages and block padding, including when layer views share backing storage.
Bound floating-source chunks independently of total cache capacity; quantization
temporaries add a bounded multiple, so the source cap is not a peak-memory claim.
Initialization stays outside warmup/timing and in acquisition method metadata;
old rows must not be relabelled to bypass incompatible resume.
Recurrent states are zero-start, not generated history. None of these inputs
claims trained-workload KV statistics
or representative data-dependent sparse selection.

Token IDs use a local deterministic generator bounded by the live,
TP-overridden vocabulary. Drain each asynchronous result before reusing input
buffers or request state; fresh IDs must not become streaming extensions of the
previous shot. Metadata cleanup does not erase physical KV pages. Preparation
adds no simulator cost.

Query geometry and request phase are separate contracts. `PreparedQuery` carries
the shot's explicit prefill/decode boundary through dummy KV preparation.
For a history-bearing decode, the prompt ends before the query: V1's fresh
`CachedRequestState` is split before `InputBatch.add_request`, while V2 receives
the prompt separately from all token IDs. Preserve token values, computed
positions and physical pages, and leave sorting/backend thresholds to vLLM.
Restore instance hooks even on failure. Making the whole sequence a prompt
leaves decodes as short extends; a sparse indexer with a lower decode threshold
than the main attention backend can then classify an entire mixed batch as
prefill. Geometry-only checks cannot catch that. Zero-history category shots
remain fresh prompts. Record this request-phase preparation with the CUDA
device-active-union method, not as a numbered protocol suffix.

### Runtime contexts and coordinated component layouts

The worker extension installs capability-checked dummy-weight preparation before
native loading. Ordinary models call the original loader unchanged. For the
admitted native MXFP4/SITU backend, initialize checkpoint-format
expert bytes with the native quantizer in bounded row chunks; preserve native
backend selection and postload processing. Do not substitute arbitrary uint8
values or silently emulate unsupported TP/EP layouts. KDA's nonpersistent
decode conv/norm buffers copy their native parameter transforms after floating
initialization, rather than receiving independent random values. Preparation,
postload structure and measured-forward validity are separate facts. The main
engine persists the preparation receipt and method metadata describes the inputs.
Preparation alone does not certify a forward or serving support.

Architecture `native_profile` selects `NativeProfile` orchestration in
`callable_profile.py`, with native API/backend inspection delegated to an admitted
adapter in `hooks/runtime_admission.py`. Keep common category publication in
`context_tables.py`, strict loading in `context_lookup.py`, and contextual
bundle/sizing integration in `serving/core/component_sizes.py`; do not add
checkpoint-specific orchestration or loader copies. The hybrid/residual adapter
retains native capability, ABI, precision and execution-mode checks. Config-derived
geometry does not establish arbitrary model/backend or device support.

Kimi-K3's local eager adapter preserves the independently supplied nested config.
Shared stack helpers derive category prefixes and original-depth residual roles;
MoE covers the MLP axis and other categories retain their declared axes. Keep
model widths, expert count and top-k native, with distinct MLA and KDA dimensions.
A completed category requires every planned layer/coordinate, independently
rebuilt catalog coverage, compatible semantic/runtime contracts and an unchanged CSV.
Retain measured library versions as observations; do not infer absent versions
from the consuming environment or use an exact version as execution admission. Validate partial
CSVs before deduplication or preload can hide malformed/duplicate rows. A partial
shot is reacquired in full. Old publication schemas and changed execution semantics
require fresh acquisition, never relabeled measurements or inferred completion.
Preparation elapsed times remain engine telemetry but do not participate in
runtime identity. Keep actual layout checks and table-integrity hashes; source
file hashes in historical records do not pin execution. New admission records
retain actual backend/layout and version observations, not source-file digests.
Unscoped runner GPU work remains visible as coverage gaps; complete model
boundaries do not imply complete step-time accounting or a zero-gap CLI result.

KDA's projections and recurrent core are separate. Native MLA's q-LoRA phase
joins concurrently executing gate and attention branches; its per-call device
union must remain one atomic cost. This concurrent boundary is the exception
to splitting independent scaling axes: independently adding the branches
double-counts overlap. The gate application and output projection remain separate.

Schema-3 callable contexts preserve declared scalar arguments and tensor
metadata, never values or CPU duration. Contextual dense rows cannot enter
the token-only sink. The hybrid/residual planner uses original-depth source roles
through the unchanged native function, not the reduced engine's pooled calls.
Its isolated synthetic arguments, native ABI and CUDA call evidence are validated
before publication. Root `per_tp[tp]` retains category records under `categories`,
context contracts/receipts under `contexts.{ordinary,callables}`, and callable
options, plans and compact per-shot summaries under `callables.callables`.
Keep `callables/dense_context.csv` as timing data; do not write new raw CUDA
ledger/evidence JSON files. Historical sidecars remain read fallbacks. Lookup validates
them once, then interpolates tokens only within an identical structural context,
without cross-context fallback or extrapolation. Keep the fixed allocation
bound explicit; do not silently reduce context capacity to fit it.

`ContextualProfileBundle` requires complete ordinary and residual tables, local
TP1/PP1/EP1 eager text execution and explicit `duration_replay_only=True`.
Source-role residuals disable transformer-block copying. Logical byte triples
are not resident capacity or traffic. Distributed, graph-padded, speculative,
offloaded, prefix-cached, power and roofline execution remain unsupported;
do not present bounded profiling/trace readiness as full-depth GPU execution
or end-to-end accuracy. The hybrid/residual path does not enable skew acquisition.
Shared geometry and trace routing read expert top-k through `expert_top_k`;
preserve the existing expert-group selection semantics.

`capture_scope_contract.py` projects the whole architecture to ordinary, coverage
and skew RPCs, preserving exclusions across category slices and including them
in host/worker identities. Methods/operators use schema 1; explicit callable
declarations add schema-2 `callables`, including inactive alternatives, and
argument/layout declarations use schema 3. Common
`embed_input_ids`/`compute_logits` method scopes provide runtime ancestry for direct
calls, not independently bindable costs or arbitrary-method instrumentation.
Restore exact instance state; reject foreign replacement, async/generator methods
and ambiguous normalization. Exact native `operator:` scopes own CUDA activities,
never CPU duration; reject wildcards, launch APIs, collisions and ambiguous or
out-of-forward/thread scopes. Contexts add no descendant normalization units:
consume returned normalized components, not an augmented tree via an old extractor.

Each catalog entry chooses exactly one `vllm`, `operator` or typed `callable`
binding. A declared function resolves an exact attribute of an already loaded
module; a declared method resolves an exact object path under an observed runtime
owner, including subclasses. Do not import targets, dispatch by model family or
replace kernel implementations. Callable scopes remain transparent to descendant
normalization; their own timings are per-call CUDA unions, never CPU duration.
An explicit `normalization` chooses `parent` or `invocations`; omission preserves
ordinary parent and whole-MoE invocation defaults. Check that the selected unit
matches one trace component rather than using normalization as a fitted factor.

Unavailable targets remain visible in `callable_audit` and once-per-runner
warnings. They are not zero measurements or proof of selected-layout coverage.
Reject failing descriptors, unsupported async/generator or data-descriptor targets,
ambiguous ownership, kind collisions and overlapping canonical claims. Restore
exact prior attributes on exit; preserve a foreign replacement and invalidate the
capture rather than overwriting it. Both acquisition identities include the
full declarations and their capture semantics, not source-file hashes.
Preserve historical simulation inputs,
but never relabel them to resume an incompatible acquisition.

`CoverageReport.canonical_invocations` preserves actual per-measured-forward
module/operator/callable counts, positive-CUDA call counts and raw CUDA activity
counts separately. Normalization groups retain their exact divisors; never sum
them into a fabricated call count. Check expected calls against the source-reviewed
execution path, not the denominator or a coverage percentage. CPU-only calls do
not establish CUDA coverage, and missing entries must not become zero timings.
Warmup/preparation calls are excluded; this audit changes no latency formula.

`component_layouts.py` is a shared CPU-only contract for declared historical and
alternative decompositions over `dense`, `linear_attention` and `attention`. Select a
unique variant from positive prefill/decode/mixed CUDA coverage on every worker
and check every acquired shot. Never infer it from model/version names or absent
kernels. Small probes do not certify unmeasured shapes, backends or deployments.
Discovery uses fixed `runner._COVERAGE_REGIMES`, not limits-adapted probes; small
engine envelopes can reject discovery even when their requested grid is valid.
Validate all fixed probes on the host before the first worker RPC. Their minimum
envelope is 5 requests, `max_model_len >= 258` and an acquisition token buffer
of at least 68 (the logical token budget plus `max_num_seqs`). Refuse an undersized
envelope without resizing or skipping probes. `hooks.batch.validate_shot_limits`
also checks ordinary synthetic request count, query-token total and per-request
history plus query plus one output against actual engine buffers before page
allocation or GPU work; preserve this guard for direct worker calls.

`meta.yaml.component_layouts` (`component-layouts`, per TP) is the sole authority.
Publish `pending` before clearing, `writing` with identity, then `complete` only
after row validation. Changed shared identity requires explicit force across
coupled categories; changed representation/definition requires fresh output.
Resume whole valid shots, preserve unrelated TP-stable identities, and refuse
pending/mixed/stale/malformed data or missing complete-table regimes.
`writing` categories may resume before their first measured shot, including when
only valid TP1 replicas remain at TP2; validate replicas before exempting them from
local coverage. Empty/replica-only categories cannot become `complete`. Unmarked
historical tables keep their old sequence; alternative-only rows require metadata.
Qwen's packed core includes its norm, so refresh dense and linear_attention together.
Use one writer per variant root and immutable published bundles: atomic replacement
is recovery, not locking. See the output-bundle guide for the full publication rules.

Site-scoped alternatives name existing `blocks`/`shared` paths. Apply replacements
once; a `layer0` prefix may be consumed only at absolute target layer 0 by the
declared active prologue endpoint after its expected prefix matches. Never remove
an MTP or later-layer norm by analogy. `site-scope-calls` checks actual canonical
calls and timing divisors against the resolved prefix before publishing each shot.

`native-dsa-eager-tail` binds original source configuration and logical TP/EP
to live native runtime facts. Keep configured FP8/BF16 matrix widths separate
from BF16 activations;
do not shrink model width, emulate another TP or accept skipped/shared indexers.
The physical acquisition is local TP1/PP1/DP1/EP1, eager decode q=1, without MTP,
offload, sequence parallelism, dense-MHA prefill or quantized query input.
Retain packed cache geometry and the actual unwrapped sampler class/entrypoint
and entrypoint ABI. Source hashes and library versions are observations, not
admission locks. Sampler payload width is source-backed, not observed
tensor data; V1 returns int32, V2 int64. Do not force a runner to fit a byte formula.

New native DSA publication requires `dense`, `per_sequence` and `attention` with
matching runtime identity and per-category engine facts inside the existing per-TP
component-layout record. Native MoE components publish independently under their
deployment contract; new ordinary records must not require the removed `moe.csv`
acquisition. Historical records containing `moe` retain their original completion
checks. Never bless old rows by adding these fields. The combined tail includes indexer work and must use full-context
coordinates, not the historical sparse-attention cap. Historical representations
and MiniMax's independent native attention contract remain unchanged.

Native DSA traces require explicit `duration_replay_only=True`; default serving
must refuse the native 656+132-byte cache until capacity integration exists.
Logical component bytes are not traffic or allocator residency. Refuse distributed,
P/D, offload, PIM, speculative, graph-padded, prefix-cache, power and roofline modes.
Cache layout decisions and immutable substitutions by TP/site, sharing lazy caches
across TP views; recheck source configuration on native byte-contract cache hits,
including mutations of an already seen dictionary. Do not apply new restrictions
to historical unmarked bundles.

Ordinary category sweeps checkpoint the accumulated CSV between completed
shots, including early in a new run. `DedupSink.flush(clear=False)` retains
all rows, acquisition identities and duplicate counts; writing only the most
recent rows would erase earlier progress. Stage the measurement metadata until
the replacement CSV is serialized and fsynced. Publish
root `per_tp[tp].measurements.<category>.publication` with `state: pending` and the intended
`csv_sha256`, replace the CSV and fsync its directory, then mark the publication
`complete`. Apply the same ordering to later checkpoints and TP-stable replication.
Resume validates the CSV against either state's checksum: matching pending data
are reusable, while mismatches require fresh output or explicit `--force`.
Checking/preloading neither finalizes pending metadata nor stamps older metadata
that lacks a publication record. Keep these data-generation seals separate from
source/version provenance and category-coverage completion. A prior checkpoint
is resumable only with compatible metadata; never accept old CSV bytes under a
new forced-acquisition identity after an interrupted replacement. This
does not turn a partially acquired table into a completed simulation bundle:
finish the requested sweep before adopting it. Checkpoint I/O is outside the
CUDA timing context and does not add simulated CPU time.

Attention resume keys use `_attn_key` unchanged, as does `extract_points`.
The CSV writer formats latency only; coordinate floats keep their full
precision. Do not round only the planned key: weighted prefill coordinates
can have more than three decimal places and would miss their stored rows.
This identity alignment changes neither measurement protocol nor saved timings.

The common `hooks/activity_collector.py` captures actual runtime module
ancestry, including shared objects and repeated calls. Native OS-thread CPU
scopes and launch correlations establish ownership; never use GPU timestamp
containment. Duplicate, missing, cross-forward or ambiguous ownership fails
closed. Ordinary shots require the same positive active canonical set in every
forward, without requiring inactive catalog alternatives to execute. Reject
overlapping canonical claims before storing timings. Coverage keeps a unique
native kernel/copy/memset ledger and explicitly reports unscoped work; never
silently label it preparation or add it to an arbitrary layer. Preserve overlap
errors through the host CoverageReport and CLI exit status.

Untimed recurrent warmup checks must bind each query/history pair to its own
assigned state pages, including reused backend metadata. The audited ordinary
GDN contract uses cache mode `none`. Native vLLM 0.31 KDA has a separate,
native API/metadata contract: it consumes the complete non-speculative view, including
the decode prefix of a mixed batch, and does not populate GDN's prefill-only
views. Keep state-page, query/history, initial-state and token-conservation checks
for both. Reject unsupported APIs/schemas and active speculative, checkpoint,
RecoverSSM or optional FlashInfer partitions. This warmup guard does not certify
a model's catalog, acquisition grid or serving support. Changed acquisition
semantics require remeasurement, not metadata relabeling.
MTP shots require every requested canonical
entry, rather than publishing a partially observed drafter table.

`hooks/cache_pages.py` allocates all cache groups from one shared physical
pool, reserving null block 0. Group IDs must not overlap even when typed views
have different logical shapes. Check the sum of all groups' physical page
requirements against capacity. Initialize only assigned spans, respecting
backend block axes and kernel subdivisions; preserve null and unassigned pages.
Recurrent caches retain zero-start state. Unsupported state modes and packed
layouts fail explicitly. Record allocation semantics in ordinary and skew
method metadata; incompatible rows require remeasurement, not relabelling.

Raw skew CSV rows preserve ordered requests, query roles and actual times;
metadata records repetitions and method. Complete kernel sets and sufficient repetitions are required
for resume. Failed acquisition raises after checkpointing instead of silently
publishing partial coverage. Root `per_tp[tp].skew` records the actual per-TP plan,
resolved limits and completion; metadata must not fabricate this from defaults.
If all planned skew rows were saved before a completion-update interruption,
resume may seal that record without new timings or acquisition provenance.
`profiler plan-skew` previews the same plan and outstanding cases on CPU from
an existing bundle's resolved engine limits. Both `plan-skew` and
`profile --only-skew` accept an independent positive TP selection such as
`--tp 2`; they neither measure nor replicate TP-stable categories. Ordinary
category profiling still requires TP1.

`skew_calibration.py` recomputes both endpoints with serving's unchanged
query-weighted attention lookup. Exact kernel/query reference slices are
required. Bimodal historical raw measurements remain reconstructible; that
input conversion is not an old runtime estimator. General data requires full
geometry, and multi-query data an explicit prefill boundary.

The weighted median minimizes each cell's summed absolute relative
profile-latency error before clipping. N anchors require sufficient distinct
support within their kernel/query/prefill/lever partition. Sparse counts pool
to the nearest supported anchor using geometric midpoints. Prefill boundaries
scale with the measured token envelope; leverage boundaries are dimensionless.
Unsupported cells use their own kernel/query pooled fallback; missing
kernel/query data means zero correction. No cross-model, hardware, TP or
query borrowing is permitted.

Only the supported `runtime-skew-calibration` format can be enabled. Remove, rather than
revive, old inline alpha and legacy bucket lookup code. Disabled bundles need
no migration; the RTX4090 bundle remains untouched. The normal writer and
CPU-only `profiler refit-skew` use the same compiler. Identity, attention,
table, lookup-code and saturation fingerprints are checked before execution.
An empty rebuild is an error and must not enable a disabled fit. A root skew
record marked incomplete also blocks rebuilding until acquisition is resumed.
Repeat-level skew CSVs require complete root acquisition metadata; unmarked
historical tables retain their existing behavior.
`--skip-skew` skips new acquisition, not existing data; `--only-skew` requires
existing attention references before GPU work begins.

Validate every statistic through `bench validate` on committed examples,
without fitting their request latencies. Broader acquisition and a shared
rule do not guarantee improvements on every model. Generalization needs
unseen models, hardware and workloads; endpoint summaries are not sufficient
statistics of arbitrary distributions.

### CUDA graph timing and calibration boundaries

Ordinary profiles measure per-call CUDA activity unions, not eager engine
wall time. CUDA graph dispatch changes execution shapes and may change kernel
selection or overlap; graph padding alone is not a wall-time correction.
Do not subtract an isolated eager-versus-graph wall-time difference from
profiled GPU work or add CPU launch/preparation time to simulated GPU latency.

Use matched query/KV shapes, execution modes and measurement boundaries when
checking a residual. Tick-level throughput aggregates and end-to-end agreement
cannot identify a per-layer error or establish a universal per-step adjustment.
New native MoE components use eager-only acquisition, without isolated graph
capture. Serving graph padding is separate and still determines lookup rows.
Historical eager/graph component tables retain their measured mode selection.

Calibrate communication from primitive measurements using the backend's
actual cost model, not by fitting model benchmark errors. Keep raw samples,
rank count, units and calibration assumptions with the hardware bundle.
Full-depth graph timing and sharded layer profiling are different contracts:
sharding GPU work does not scale host costs or reproduce full-model cache state.


### hardware.yaml: the machine's own facts, measured
A cluster config mixes two kinds of statement:

```
tp_size, num_npus, mem_util, dp_group      what the user wants to simulate
link_bw, link_latency, npu_mem.mem_*       what the hardware actually is
```

The first stays under the user's control. For owned hardware, the second
should be characterised independently of model benchmarks. Fitting link
parameters against end-to-end accuracy can conceal unrelated model errors.

`python -m profiler hardware --hardware <hw> --npus 2` writes
`profiler/perf/<hw>/hardware.yaml`: one file per hardware folder, shared by
every model bundle under it. The hardware command defaults to INFO logging;
an explicit `--log-level` is optional and validated at argument parsing.
Three sections:

- **`spec`** — queried from the device. GPU name, SM count, bus width, clocks,
  PCIe generation and width, driver, power limit, and `memory_bw_gbps` derived
  from the reported clock and bus width rather than hardcoded.
- **`measured`** — the NCCL sweep: **all three collectives** the simulator
  emits (AllReduce, AllGather, ReduceScatter), timed inside CUDA graphs and
  separately in isolated calls, with raw samples retained alongside the fit.
- **`defaults`** — what a cluster config inherits, each entry carrying its own
  `source`: `measured`, `spec`, or `assumed`, and units. A run logs each
  inherited value with that provenance.

**Fit the model the backend executes.** `_ring_terms` in
`profiler/core/hardware.py` represents one local-dimension, single-chunk Ring
on a one-hop FullyConnected topology. With `N` ranks, AllReduce has `2(N-1)`
network phases; AllGather and ReduceScatter have `N-1`, not the same latency
multiplier. Raw sample bytes mean AllReduce input, AllGather local input, or
ReduceScatter local output. Convert them to Ring message size before counting
traffic; they equal charged traffic only at `N=2`.

Network BW uses **GiB/s**, as `NetworkFunction.cpp` multiplies it by `2^30`;
local-memory BW uses decimal **GB/s**. `PacketBundle.cc` separately charges
three local transfers per reduction step and `MemBus.cc` adds endpoint events.
Include these known costs in the fit, using the integer local-memory spec
that `config_builder.py` supplies to ASTRA; do not let them be absorbed into
link BW. The detailed formula is in
`docs/docs/profiler/adding-hardware.md#calibration-contract`.

**One pair is an approximation, not proof that NCCL shares a curve.** The
objective remains relative squared error over the primitive sweep, solved
as nonnegative least squares in latency and inverse BW without model-specific
coefficients or hardware-specific search bounds. The network term is continuous
for fitting; saved predictions use serialized parameters and the backend's
integer truncation. Record fit version, rank count, units, assumptions, timing
source and per-size/per-collective residuals. Invalid or unidentifiable inputs
must fail rather than emit link defaults. A zero-latency boundary is flagged.

The primitive sweep does **not** measure grouped multi-tensor or ragged MoE
communication. Summing their payload bytes into one primitive does not make
their execution equivalent. Other topology, splitting, rank count or memory
settings require validation against that contract. Existing `hardware.yaml`
files are not automatically rewritten by a code update; old fit residuals
cannot certify the corrected formula. Refit retained samples with their
measurement context or re-measure before validating new defaults.

**Optional per-operation links retain physical bytes.** `collective_links`
in the cluster config selects `link_bw` / `link_latency` for `all_reduce`,
`all_gather` and `reduce_scatter`, each scalar or one value per topology
dimension. Missing fields and operations use the common link. The builder
emits same-topology sidecar YAML files referenced by `collective_networks`.
The congestion-unaware analytical frontend supports these overrides for Ring
only; ns-3 and other collective algorithms are rejected. Preserve the original
collective on the stream/request: AllReduce's internal scatter/gather phases
must use its AllReduce curve, not standalone ReduceScatter/AllGather curves.
Do not change message bytes, rank groups or local-memory charges to emulate BW.

The hardware command keeps the joint fit and records `collective_fits` at its
fixed latency, with separate residuals. Each identifiable operation supplies
`defaults.collective_links.<operation>.link_bw`; an unavailable fit records its
reason and supplies no override. These effective primitive fits do not model
NCCL's grouped launch semantics or establish accuracy at other rank counts.
Existing hardware files are unchanged until explicitly refreshed.

Hardware operation defaults are inherited only when neither common link value
nor an explicit `collective_links` key is supplied, and every instance shares
one hardware label. An explicit map replaces the hardware map; `{}` disables
it. This precedence must remain idempotent across both config readers, so a
hypothetical explicit interconnect never silently gains the measured card's
operation curves.

The RTXPRO6000 Qwen3-32B TP2 and Qwen3-30B DP2/EP2 examples omit all three
link keys and consume the measured operation defaults automatically. Their
vLLM 0.31 references preserve the recorded custom-all-reduce setting: disabled
for Qwen3-32B TP2, not disabled for Llama TP1 or Qwen3-30B TP1/DP2/EP2.
Neither TP1 deployment exercises the TP custom-all-reduce path. Refresh
each example's recorded truth, simulator output, validation plots and
regression digests together when adopting a changed interconnect. Do not
pair refreshed hardware defaults with an older transport reference.

**Three inheritance rules** (`serving/core/hardware_defaults.py`, applied at
both config load sites):

1. **An explicit value always wins.** No exceptions and no warning — a config
   describing an 8-GPU NVLink node it does not own must be able to say so.
2. **A gap is filled from the bundle**, and the run logs the value and its
   provenance.
3. **A gap with nothing to fill it raises.** Not a default, not a warning. If
   the interconnect was never measured — one GPU on the machine, or hardware
   that is gone — no number is defensible.

`link_bw` / `link_latency` are cluster-level while `hardware` is per-instance,
so they are inherited only when every instance shares one label. A cluster
mixing two card types has a link that is neither one's intra-node measurement.

**Two GPUs are the floor, and the measurement is topology-specific.** A link has
two ends. `profiler hardware` still writes a useful file on a single-GPU machine
— the spec section needs only a device query — with `interconnect: null` and the
reason, and it **exits non-zero** so a script notices, the same shape as
`profiler coverage`. And `npus` is recorded because an all-reduce across two
PCIe-linked cards is not the physics of eight over NVLink; a config asking for
more is extrapolating.

**RTX4090 is the worked example of the empty case.** The card is gone, so its
`hardware.yaml` is hand-written with `measured: null`, carries no link defaults,
and its bench example keeps `link_bw` / `link_latency` in the config — where a
reader can see they are the author's choice rather than a measurement.

**`mem_bw` is deliberately not measured.** A profiled kernel latency already
contains the card's real memory behaviour; `mem_bw` applies to the
explicitly-modelled memory paths (`--prefix-storage` KV recall, PIM, remote
memory) and ASTRA's local reduction cost. It is carried from the spec,
marked `spec`, and `mem_latency` is
marked `assumed` because every committed config has carried 0 and nobody has
checked it. What will matter is **CPU** memory bandwidth, which those paths do
read.

**`mem_size` is CUDA's number, not the marketed one.** vLLM applies
`gpu_memory_utilization` to `init_snapshot.total_memory`, which is
`torch.cuda.mem_get_info()[1]` — 101,976,440,832 B = **94.97 GiB** on this
card, against a marketed "96 GB". That is not a 1000/1024 slip: 96 decimal
would be 89.41 GiB, so the marketing figure is GiB labelled GB, and the gap to
CUDA's number is ECC/firmware reserve (0.4%) plus driver context (635 MiB).
Since the simulator's `mem_util` mirrors vLLM's flag, its base has to be the
same one. It changes nothing on a run that does not saturate — Qwen3-32B peaks
at 36.8% KV with zero preemptions, and 96 vs 94.97 came out identical to four
decimals — and moves KV capacity by 1.1% on one that does.

### Target graph coordinates and eager acquisition

`profiler/core/sampling.py` combines geometric points with target serving graph
coordinates through the shared `serving/core/cudagraph.py` resolver. Acquisition
always remains eager. Explicit `--cudagraph` JSON overrides the target policy;
otherwise prefer a scope-matching completed worker capture observation and derive
candidates only when none matches. Label observed captures, explicit options and
derived candidates separately. Acquisition-backend capability and a requested
grid do not establish full-model capture. Resolve from logical scheduler limits,
not padded acquisition buffer sizes; retain runner, query length, TP/compiler-SP
and hardware distinctions. Explicit `NONE` disables graph anchors.

Dense token grids include forward anchors; TP1 also unions the requested TP
degrees' anchors for TP-stable replication. Target logits and sampler execute
outside the backbone graph, so per-sequence sampling stays geometric. Attention
and linear attention add pure-decode request anchors only for FULL-capable modes
at the supported query length, and pure-prefill anchors only for `FULL`; mixed
shots retain geometric axes. Keep context, block-rounded KV and scheduler
feasibility checks. MTP adds drafter request anchors only: its table has no
complete first-pass query/token dimension. Skew keeps independent axes.

Root `per_tp[tp].sampling[category]` records the feasible plan, factors, target
coordinates, provenance and category limitations; it does not certify completed
measurements. Report excluded dense capture points. Native MoE records transformed
`expert_sampling.capture_tokens` and `sampling_graph` in each target contract;
do not place metadata in CSV columns or reinterpret eager timings as graph times.

### Attention-grid density and acquisition cost

`--grid-factor` defaults to 2 for ordinary geometric token/request coordinates.
Attention chunk/KV, skew and native MoE retain separate factor controls, all
defaulting to 2; only attention request counts retain `SQRT2 = 2.0 ** 0.5`.
The KV factor controls both prefill-key and decode-KV. The editable `profile.sh` leaves
optional settings unset and forwards only explicit nonempty values, so the
CLI owns its defaults. Keep every `profile` option reachable from the template,
including `GRID_FACTOR`, JSON-valued `CUDAGRAPH`, `DP_DEGREES` (`--dp`),
`EP_DEGREES` (`--ep`), `MOE_ROUNDS` and explicit logging; do not
silently enable a TP sweep or native DP acquisition. The separate multi-model
`profile-all.sh` remains an explicitly configured campaign, not a default-parity
wrapper. What picks sqrt(2)
over 1.5 -- which is *coarser*, not
finer, since 1.5 > sqrt(2) -- is that `_geometric_grid` accumulates in float
and rounds only for output, so the even powers land exactly:

```
sqrt(2)  0, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256
x2       0, 1, 2,    4,    8,     16,     32,     64,      128,      256   <- a strict subset
1.5      0, 1, 2, 3, 5, 8, 11, 17, 26, 38, 58, 86, 130, 195, 256          <- not nested
```

A doubling axis is therefore a **subset** of the sqrt(2) axis. Reuse still
requires compatible acquisition identities and matching complete shot keys;
changing another axis or the request composition can change those matches.
A factor of 1.5 is not nested in either grid. Compare actual coordinates
rather than promising that a density change reuses every previous row.

For a fixed range, interval count scales as `1 / log(factor)`. Changing a
factor from 2 to sqrt(2) roughly doubles intervals per axis, not 1.4x.
Attention combines four shape axes, so their costs multiply before rounding,
deduplication and feasibility filters. Enumerate `AttentionCategory.compose_shots`
with the requested settings and resolved limits for the actual shot count.
Do not infer whole-bundle duration from one axis factor or an old run's ETA;
skew, other TP degrees and native MoE acquisition are separate work.

**The accuracy case is specific to `n_decode`, and it does not extend to
chunk or kv.** That axis is smooth on a *pure decode* batch -- per-sequence
cost on Llama-3.1-8B is flat and slightly falling, 6.60 / 6.33 / 5.94 / 5.81
us at n = 16 / 32 / 64 / 128 -- which is what a doubling grid was chosen on,
and that measurement is right. It just does not describe a **mixed** batch,
where the same model rises 37% across the same interval (chunk 273, kv 2076,
one prefill sequence):

| n_decode | 16 | 32 | 64 | 72 | 80 | 88 | 96 | 104 | 112 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|
| pure us/seq | 6.60 | 6.33 | 5.94 | | | | | | | 5.81 |
| mixed us/seq | 9.18 | 9.28 | 8.99 | 9.33 | 9.45 | 9.47 | 10.39 | 11.03 | 11.55 | 12.31 |

The curve is flat to n=88 and then climbs, so the blend across a 64 -> 128
bracket over-charges the middle. Scored against those measured values, with a
profiled point at 91 standing in for what sqrt(2) samples:

| n | measured us | x2 blend | err | sqrt(2) blend | err |
|---|---|---|---|---|---|
| 72 | 671.8 | 700.4 | +4.3% | 670.0 | -0.3% |
| 80 | 756.0 | 825.4 | +9.2% | 764.7 | +1.2% |
| **88** | 833.4 | 950.5 | **+14.1%** | 859.4 | +3.1% |
| 96 | 997.4 | 1075.5 | +7.8% | 986.9 | -1.1% |
| 104 | 1147.1 | 1200.6 | +4.7% | 1134.1 | -1.1% |
| 112 | 1293.6 | 1325.6 | +2.5% | 1281.3 | -1.0% |

Worst |err| 14.1% -> 3.1%; attention is 54.8% of such a step, so the
step-level bias is 7.7% against 1.7%. And that is where a real run's error
was: matching Llama-3.1-8B's steps against a vLLM run one for one, the
simulator is exact at n_decode 0 and 32 (both grid points, 0.983-1.032) and
5-12% slow at 64-127, at every chunk size from 256 to 1792. A layer
decomposition puts the excess in `attention`. Everything else was ruled out
first -- skew is worth 0.4 points, `n_prefill` 2.0 points, and every other
layer is within a percent across the two shapes.

**Note what that table is and is not.** It is arithmetic on measured values,
so it bounds the interpolation error of the modelled quantity; it is not an
end-to-end result, and how much of Llama's +4.8% TTFT it recovers is settled
only by re-running the refreshed bundle against the truth. This is still a
different kind of evidence from the one that reverted `attention_chunk_factor`
-- that was an end-to-end A/B confounded by the bundle and the truth being on
different vLLM versions, while this is measured against the same engine in the
same run.

Chunk and KV axes default to doubling. The request-count evidence above does
not establish an accuracy benefit from densifying those other axes. Target
graph anchors add relevant pure-regime coordinates without changing their
geometric defaults or turning eager measurements into graph measurements.

**Skew acquisition density is separate from fit resolution.** Its geometric
factors retain their defaults in `ProfileArgs`; the new compiler creates N
anchors only where each partition has direct support. Increasing density can
add support or introduce new regimes, but neither necessarily improves the
fit. Prefill partitions scale with the measured token envelope; lever bins
remain dimensionless and fixed. Other measured shape axes still affect which
observations are pooled. Validate changed coverage through
the committed bench examples rather than assuming that more rows are better.

**A refresh has to restate the other axes.** Read each bundle's recorded
acquisition settings instead of assuming that historical profiles used today's
defaults. A category slice can change the grid when its factors or resolved
limits differ; a lower `--max-num-seqs` narrows support through feasibility
filters. Match the recorded chunk, KV and request factors, decode query lengths,
maximum KV, request/token limits and target graph policy when retaining the
same sampling envelope. A changed envelope is a new acquisition choice, not a
row-for-row refresh.

**A long sweep prints a heartbeat.** The rich progress bar needs a TTY.
Ordinary categories also write periodic atomic CSV checkpoints between
completed shots, independent of progress rendering. Low GPU utilization
alone does not establish a stall: CPU event processing contributes to
acquisition time but not simulated latency.
`_fire_one_category` logs count, rate and ETA every 1% of shots, and
`skew.sample_skew` logs at every 20-case atomic checkpoint. It reports the
actual acquisition rate, excluding reused cases, and retains completed cases
if a later measurement fails.

### Feasibility bounds shared by attention and skew
Both the uniform attention sweep and the skew sweep cap `n_reqs > max_num_seqs`
(strict `>`, not `>=`) so that `n = MSQ` **pure** cases (no prefill chunk) fit.
This uses vLLM V1's `input_batch` buffer exactly up to `MSQ`. Mixed cases at
`n = MSQ` need `MSQ + 1` requests and are still filtered. If a runtime workload
needs mixed-regime data at `n = X`, profile with `MAX_NUM_SEQS ≥ X + 1`.

The bounds that count **tokens** rather than requests have to see
`decode_q_len`, because a decode request submits `q` of them. They are inside
the `q` loop in `AttentionCategory.compose_shots` for that reason — see the
speculative-decoding section for what happens when they are not.

`--attention-max-kv` is resolved **once**, right after `probe_limits`, by
`engine.resolve_attention_max_kv`, and written back into the frozen
`ProfileArgs` with `dataclasses.replace`. Everything downstream — both grids
and the meta the run records — therefore reads one concrete number and never
`None`; `compose_shots` asserts on `None` rather than silently sweeping at the
wrong resolution.

### Native local and DP+EP component profiles

MoE acquisition defaults to native DP1 components; `--dp` selects other degrees.
Use `tp_degrees`, `dp_degrees` and `ep_degrees` for internal settings, with
`--tp`, `--dp`, `--ep` and corresponding uppercase shell variables. Optional
EP degrees filter the supported TP/DP targets; they never change expert sharding
or infer additional TP/DP degrees. Omitted EP keeps every target. Reject unmatched
EP degrees before any engine starts; a slice matches only `--tp-refresh`.
The filter affects only MoE components, not ordinary categories at the other TPs.
The removed whole-block path no longer shrinks global expert count/top-k.
The target is full EP=TP*DP with
DP>=1, independently of the one-GPU acquisition process. `MoeTarget` is shared
with serving: local gate/finalization rows and gathered expert rows differ,
and sequence-parallel wrappers require per-member ceil division before gathering.
Pass the full already-padded DP vector through both wave-completion paths; its
sum/minimum cannot reconstruct the sequence-parallel geometry.

The unquantized modular DP1 path is non-SP. Select its backend from the resolved target,
not the DP+EP selection: native priorities can differ. NoDPEP preparation must
preserve the input and return no scale or token metadata; with a NoOP reduction,
experts write directly to the caller's output alias. Observe no CUDA work in
prepare/finalize before recording `local_execution` and omit `finalize_copy`
entirely. Preserve the alias in every conditioned eager variant. DP1
traces contain gate/routing, experts and optional TP AllReduce, no DP collectives.
Full EP at TP>1 owns whole experts and is not plain tensor-sharded TP. Missing
DP1 native coverage falls back only when a historical whole-block table exists,
without consuming a mismatched EP component table. Match the legacy lookup's TP
choice (TP1 when present, otherwise its first available degree); a fresh bundle
without that table requires a complete matching component acquisition.
Cache missing deployment coverage for the lifetime of the simulation, just like
loaded native tables: bundles are immutable during a run. Do not reread the same
metadata on every layer. Corrupt contracts remain errors, never cached misses.
Initialize `native_moe_index_present` and `native_moe_components` on the persistent
perf DB before creating per-batch shallow copies. Otherwise the copies discard
new cache entries and rebuild metadata and interpolation support every batch.
Keep TP-specific attention caps on each view and native deployment keys separate;
cache reuse changes host work, not measured latency or routing.

Read `FusedMoEConfig.use_deferred_moe_finalize` as a strict boolean effective
capability. `defer_moe_finalize` is a request flag in 0.28 and a request method
in 0.31, not a stable execution-mode predicate. Reject an unknown/non-boolean
API; never invoke the request method or silently assume finalization is local.
Check actual native APIs, geometry, dtypes, backend and conditioning semantics;
source-file hashes and exact versions are not acquisition admission or grouping
keys. Do not restamp historical measurements after a semantic adapter change.

The separate live adapter admits actual APIs, standard token layouts and native
capabilities at TP1/DP1/EP1. Preserve processed weights/scales, native backend,
routing and activation parameters; do not rebuild an unquantized approximation.
Its `gate_routing` includes external routing, or projection only when routing is
inside the expert kernel. `experts` includes native preparation and finalization.
`wrapper` executes the native parent and accounts for shared experts, latent
transforms, normalization, scaling and combination outside the excluded gate/router
and routed-kernel scopes. Use launch ownership and conservation checks, not
subtraction of unrelated profile captures. Sum unique CUDA activity durations;
this is not an interval union or elapsed critical path across concurrent branches.
Compare every routing stimulus and each cloned bank exactly with the native method
outside timing and verify native state remains unchanged.
These guards compare parameter bindings/layouts, available version counters,
effective configuration and callable identities, not all device-memory contents.
Do not describe them as bytewise weight checks or arbitrary CUDA-write detection.

Keep gate dtype/width, model width and physical expert width explicit. Count unique
native weight/scale storage separately from all cloned bank storage; preserve
storage aliases, offsets and strides. Bank allocation uses the actual processed
bank footprint plus native workspace/free-memory reserve, not the unquantized
conversion-buffer heuristic. Missing or unsupported capability statements are
errors, including deferred, input-weighted, fused-shared-gate, biased/redundant,
LoRA, EPLB and padded-hidden execution. Monolithic routing requires native replay
evidence; a matching model name is not an admission.

The isolated MoE engine uses the MLP prefix and declared native-profile settings.
Its private `kv_cache_memory_bytes` override defaults to 1 GiB and accepts only a
positive integer. The internal caller may raise this resource cap, but never
reduce requested token/sequence/context limits to fit it. vLLM must validate native
capacity. Record resolved component-engine settings only in MoE observations,
never as main-engine cache facts or replacement target graph policy. Read nested
source geometry through `text_config`; local Kimi DP1/EP1 selectors are admitted,
distributed selectors remain refused. Preserve DSA/Kimi's duration-only consumers
and their separate capacity/serving restrictions.

Root `per_tp[tp].moe_components` selects deployment entries containing semantic
contracts, observations, coverage and CSV checksums. Paths locate tables; folder
names do not establish contract identity. No new index/contract/coverage JSON
sidecars are written. `moe_components` metadata is separately owned; a MoE refresh must not
restamp attention/skew or the main engine. Individual CUDA activities are counted
once per forward during eager measurements. No CPU duration is added.
No-op resume preserves original runtime observations. Record distinct later
`acquisition_contexts` only after validating new work; this list is not a count
of durable or completed measurement rows.
Expert measurements rotate positions and L2-sized weight banks, compare B/2B
under identical complete-cycle work, independently verify eager outputs and
require the recorded stability gate before publication. Those rules condition
the measurement; they do not prove full-model cache equivalence.
Publish only `coverage.conditioning_summary` with `checked_points`,
`failed_points` and `max_relative_change`; retain the tolerance in
`contract.expert_conditioning`. Individual control points and timings belong to
raw resume checkpoints, not published metadata. Preserve runtime observations,
geometry, dtypes, parallelism, backend, completion and the CSV integrity check.

New plans use `--moe-grid-factor` (default 2) for input rows N, active local
experts A and local assignments M, including feasible endpoints, envelope
corners and balanced-total anchors. At EP1 M=N*k collapses the assignment axis.
`expert_sampling` belongs in the root contract; CSV columns remain unchanged.
Target serving capture tokens G take precedence on token axes after
`target.token_domains((G,) * target.dp)` maps them to local and expert rows.
Use geometric points only in uncovered token ranges, retaining endpoints and
feasibility corners. Decode-only/NONE policies require the full eager range;
an out-of-capture DP peer can require small local eager rows on every member.
Include MRV1 compiler-SP rounding in physical bounds before native initialization.
Capture sizes never determine A or M. Keep transformed anchors, logical budget
and eager ranges in `expert_sampling`; source/target information belongs in
`sampling_graph`. Absent range metadata retains an existing acquisition plan;
a changed semantic plan requires a fresh output root.
New acquisition must not create CUDA graphs; eager timing and serving-padded
coordinates are distinct.

Serving automatically consumes matching tables for the supported modular
local/DP+EP or separately admitted live local path. Local components use token interpolation.
New experts use physical N/A/M interpolation: construct affine-rank-reduced
simplices at table load, then use nonnegative barycentric weights and bounded
query caches. Do not normalize by query-dependent feasible ranges: doing so
can reduce assignment work during interpolation. Check feasible geometry and
measured support; never extrapolate. Historical balanced-total tables retain
their original interpolation and graph/eager selection. The routing policy
supplies A/M; BALANCED and measured active-count CUSTOM use rounded expectations,
RR/RAND use selected pairs and native rank ownership. Within-rank imbalance is
not a fourth measured axis. Missing deployment/policy coverage warns and retains the
legacy EP table only when it exists; corruption and out-of-support queries are
errors. Shared, quantized and supported monolithic local execution require the
live contract, never guessed aliases. Deferred and tensor-sharded execution remain
unsupported. Keep historical acquisition
fingerprints as provenance; do not relabel old samples after changing a method.

DP+EP trace order is gate/routing, hidden/top-k-weights/top-k-IDs AllGathers, experts,
ReduceScatter, finalize-copy, then the wrapper's optional TP AllGather (SP) or
AllReduce (non-SP). Chakra must chain every marker collective and retain the
last dependency through ranks it skips. The ragged Ring envelope and independent
logical collective costs remain approximations of grouped NCCL execution.
See `docs/docs/profiler/native-moe-components.md` for flags and exact limits.

Shipped native data is the runtime subset: index, contract, coverage and CSV.
Its acquisition fingerprints remain historical provenance, not hashes to rewrite
when packaging or changing unrelated code. Original per-forward streams are
required for acquisition resume; use a separate output root when remeasuring a
published bundle. Treat the indexed target and contract bounds as authoritative;
the shipped Qwen3-30B TP1/DP2/EP2 data does not imply TP2 or another model is covered.

Hardware refreshes and historical examples have different ownership. The shipped
RTXPRO6000 operation fits retain the previously calibrated common latency and
refit bandwidth from NCCL primitives with the backend's actual formula. Record
that constraint and residuals; do not relabel it as an unconstrained automatic
fit. If retaining an older multi-GPU reference, explicitly retain its matching
common link and disable operation-map inheritance. The current Qwen examples
instead use refreshed NCCL references and inherit the shared operation defaults.
Refresh benchmark outputs, summaries and public plots together.

### The legacy MoE grid is per EP degree

Historical whole-block acquisition used a per-EP token grid. That acquisition
path has been removed; only its stored `moe.csv` lookup remains until remeasurement.
The following describes those historical measurements and their fallback, not the
native full-top-k component contract above. A reduced-top-k checkpoint is an
approximation and must not be presented as the native distributed kernel.
An EP rank does not run the model's MoE block; it runs a **slice** of it, and
the slice differs three ways at once: it permutes over `E/ep` local experts
rather than all E, a token contributes `k/ep` of its k expert assignments
rather than all k, and so the distinct experts it activates start at `k/ep`
rather than at k. Profiling at ep=1 and charging that per rank gets all three
wrong in the same direction.

The third is the one that bit. `activated_experts` is the only axis in the repo
whose minimum is a positive number the runtime can go under -- it floors at
`top_k`, because a token cannot activate fewer experts than it selects, and the
profiler's `ExpertRoute.forge` refuses to build one that does. The runtime goes
under it constantly: `_balanced_route_ep` gives a decode step
`activated = min(round(total_len * k / ep), E/ep)`, which is 1 at ep=8, and
**98.4% of MoE lookups in a GLM-5 EP=2 run asked for 4 against a floor of 8**.
Clamped, the MoE term reads 1.7x to 4.8x over at the decode operating point:

| model, one token on the rank | ep=1 (what was charged) | ep=8 (the real slice) |
|---|---|---|
| DeepSeek-V3.2 | 84.7 us | **28.1 us** |
| GLM-5 | 162.6 us | **33.7 us** |
| MiniMax-M3 | 139.5 us | **47.3 us** |
| Qwen3-30B-A3B | 47.8 us | **27.8 us** |

**Extrapolating instead of measuring would have over-corrected.** Fitting
`fixed + per_expert * a` through Qwen3-30B-A3B's own grid predicts 16.8 us at
`a=4` where the ep=2 slice measures 36.2 us -- 2.2x low, because the fixed
permute cost does not vanish with the expert count. The curve saturates from
ep=8 on (28.1 / 27.5 / 27.7 us on DeepSeek) for the same reason: once `k/ep`
floors at 1 only one expert plus that fixed cost is left.

These are historical reduced-expert measurements, not the current acquisition
plan or runtime estimate. `--moe-ep-degrees` and its config-slicing path were
removed. New component sweeps use `--tp`, `--dp`, optional `--ep` and unchanged global top-k;
unsupported native adapters fail rather than falling back to old acquisition.

The simulator picks the table for the instance's **total** EP degree and falls
back to the nearest profiled one with a one-shot warning. Not interpolated, for
the same reason `decode_q_len` is not: `E/ep` has to be a whole number of
experts and the permute width is a staircase in it. A bundle with no `ep`
column reads as ep=1 and prices exactly as it did before the axis existed.

### Reduced checkpoints and profile reuse

A reduced checkpoint is a diagnostic configuration, not validation of the
full-size trained model. The stored DeepSeek-V3.2-Exp-16L64E config reduces
layer and expert counts while retaining MLA, sparse indexing, grouped routing,
dense/MoE layer composition and MTP. Dummy loading still allocates its weights.

Reusing `dense`, `per_sequence`, `attention` or `mtp` rows requires matching
kernel shapes, dtypes, backend and acquisition geometry. Record inherited data
with `derived_from` metadata rather than representing it as fresh acquisition.
Reduced-stack profiling is an approximation of full-model execution and cache
state; model depth alone is not evidence that every execution property matches.

Expert count changes require a new `moe` table. Even at the same token,
top-k and active-expert coordinates, routing and permutation operate over a
different expert domain and can select different kernels. Native component
contracts also retain model/deployment identity and must not be relabelled.
