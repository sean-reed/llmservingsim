# AGENTS.md

Guidelines for AI coding agents (Claude Code, Cursor, Copilot, etc.) working in this repository.

## Project Context

LLMServingSim 2.0 is a cycle-level LLM serving simulator. It combines a Python frontend
(`serving/`, run as `python -m serving`) with ASTRA-Sim (C++ analytical network simulator)
as the backend. The profiling pipeline (`profiler/`) generates per-hardware latency data
that drives the simulation, and the bench module (`bench/`) runs vLLM end-to-end to
validate the simulator against ground truth.

### Branches

Per-paper artifact evaluation scripts (the previous `evaluation/`
directory) live on dedicated branches (`ispass26-artifact`, etc.) and
are not part of the main branch's tree.

### Simulation flow

1. `serving/__main__.py` parses CLI args and cluster config
2. `config_builder.py` generates ASTRA-Sim input files (network.yml, system.json, memory_expansion.json)
3. ASTRA-Sim subprocess is launched
4. Per iteration:
   - `scheduler.py` forms a batch under memory and token budget constraints
   - `trace_generator.py` looks up profiled latencies and emits a text trace
   - `graph_generator.py` converts the trace to a Chakra protobuf graph
   - `controller.py` feeds the graph path to ASTRA-Sim, reads back cycle count
   - `scheduler.py` updates request state, marks completions
5. Results are printed and optionally saved to CSV

### Key data flow

```
profiler/perf/<hw>/<model>/<variant>/tp<N>/*.csv (profiled latencies)
    ↓ _load_perf_db() + _lookup_{dense,per_sequence,attention,moe}()
trace_generator.py → per-layer field tuples (TraceData)
    ↓ Chakra converter, in-process, via LLMConverter.convert_rows()
graph_generator.py → .et protobuf file (reused when the rows repeat)
    ↓ stdin/stdout IPC
ASTRA-Sim (C++) → cycle count
    ↓
scheduler.py → next iteration
```

## Code Style & Formatting

- **No enforced formatter** — match surrounding code style in the file you're editing
- **CLI flags**: use hyphens (`--cluster-config`, `--max-num-seqs`)
- **Internal Python**: use underscores (`max_num_seqs`, `enable_chunked_prefill`)
- **JSON config filenames**: descriptive snake_case (`single_node_pim_instance.json`)
- **Imports**: keep minimal and consistent; `serving/` modules use relative imports
- **Comments**: use English only — no Korean or other non-English text in comments, docstrings, or log messages

## Architecture Patterns

Detailed contracts load from per-directory guides when you work in that directory:

- `profiler/AGENTS.md` — catalogs, acquisition, skew, native MoE components, hardware.yaml, grid density
- `serving/AGENTS.md` — trace generation, lookups, scheduler, memory/KV, speculative decoding, MoE/DP+EP traces
- `bench/AGENTS.md` — bench-vs-sim metric definitions

Shared contracts (`profiler/core/stack.py`, attention shape, catalogs) are simulator inputs: when a
change crosses `profiler/` and `serving/`, read both guides.

### CLI argument conventions
CLI flags follow vLLM naming where applicable:
- `--skip-prefill` — legacy first-token bookkeeping flag; it does not skip
  prompt computation or prepare cached KV and is not a decode-only generator
- `--request-routing-policy` (`LOAD`, `RR`, `RAND`, `CUSTOM`) — request routing across instances
- `--expert-routing-policy` (`BALANCED`, `RR`, `RAND`, `CUSTOM`) — expert token routing for MoE.
  `CUSTOM` reads the measured distinct-expert count from `--gate-stats` rather than
  deriving it from a uniform gate; see the gate-stats section
  (block-copy optimization is controlled separately via `--enable-block-copy`, default on)
- Reversible boolean flags use `argparse.BooleanOptionalAction` (e.g.,
  `--enable-prefix-caching` / `--no-enable-prefix-caching`). Opt-in `store_true`
  switches have no automatic negative form; document the actual parser.

Before release, compare each public entry point's registered arguments with
its directory README index and complete website reference, including negative
boolean forms, defaults, choices and subcommand restrictions. Check wrapper
forwarding without starting engines. Distinguish parser defaults from resolved
engine values; an unset profiler block size is chosen by vLLM, while serving
loads the profiled per-TP value and uses 16 only as a metadata fallback.

### Model configs
Model configs live in `configs/model/{org}/{model}.json` and must retain the
fields and structure required by the checkpoint's native configuration class,
not just the fields read by simulator sizing. MiniMax-M3 and Kimi-K3 preserve
their nested text wrappers; the shipped Qwen text tower uses its native flat
text configuration. Do not flatten or prune a different family by analogy.

The simulator loads these via `get_config(model_name)` in `utils.py`.
`get_source_config(model_name)` retains the original wrapper for independent
native-profile validation. Preserve its cache boundary: the flattened simulator
config is a separate deep copy, not a mutation of the original source object.

### Cluster configs
Cluster configs in `configs/cluster/` define hardware topology. Key instance fields:
- `hardware`: must match a directory name in `profiler/perf/<hardware>/`
- `model_name`: must match a config in `configs/model/{model_name}.json`
- `num_npus`: total GPUs for the instance (optional, inferred from `tp_size * pp_size`)
- `tp_size`: tensor parallel degree (required or inferred)
- `pp_size`: pipeline parallel degree (optional, default 1)
- `ep_size`: expert parallel degree (optional, default `tp_size` for MoE, 1 for dense)
- `dp_group`: DP group ID string (optional). Instances with the same string form one
  data-parallel group, wave-synchronized per iteration; for MoE they also share
  experts across the group. Works for dense models too (plain DP replicas)
- `npu_mem.mem_bw`: NPU memory bandwidth (also set as `local-mem-bw` in system.json)
- `npu_mem.mem_util`: fraction of `mem_size` usable for weights plus KV cache (optional,
  default from `--npu-memory-utilization`, itself `0.9`). KV capacity is
  `mem_size * mem_util - weight`. Sits inside `npu_mem` because its only job is to
  scale `mem_size`, and it follows that block's `mem_*` naming
- `cpu_mem.mem_bw`: CPU memory bandwidth (set as remote memory in memory_expansion.json)
- `link_bw`: network link bandwidth in GiB/s (set in network.yml)
- `link_latency`: inter-node link latency in ns

Parallelism inference: users may provide partial info (e.g., `num_npus=4, tp_size=2`)
and `config_builder.py` infers the rest (`pp_size=2`). Validation ensures
`num_npus = tp_size * pp_size`, `pp_size <= num_hidden_layers` (stages are cut on
transformer-block boundaries, so a stage cannot be empty), and `ep_size` divides
`num_local_experts`. Members of a `dp_group` must agree on `tp_size`, `pp_size` and
`ep_size`. For a **MoE** model in a DP group `ep_size` is the total EP degree across
the group, so it must be divisible by `dp_group_size` and
`ep_size / dp_group_size <= tp_size`; for a **dense** model `ep_size` is 1 because
there are no experts to shard, so neither check applies.

TP and EP share the same GPUs: non-MoE layers use TP (ALLREDUCE), MoE layers use EP
(an all-to-all -- see the MoE section for how it is emitted). DP is achieved via
multiple instances with the same `dp_group`.

`config_builder.py` reads the cluster config and generates three ASTRA-Sim input files:
- `astra-sim/inputs/network/network.yml` — topology and bandwidth
- `astra-sim/inputs/system/system.json` — scheduling policy and memory bandwidth
- `astra-sim/inputs/memory/memory_expansion.json` — remote (CPU) memory config

### Working directory
Run `python -m serving` from the repository root. The entry point changes cwd
to `astra-sim/` internally, but relative cluster, dataset, output and gate-stat
paths are handled relative to the invocation root. Use repository-relative
dataset paths: that loader prepends `../` even to an absolute input.
An explicit relative `--inputs-root` instead resolves from `astra-sim/`;
use an absolute path when selecting a separate intermediate directory.

### Docker environments

For long runs, `scripts/monitor_run.py` provides process-tree RSS, host available
memory, optional swap-growth and timeout guards. Choose explicit limits for the
host and a fresh log path; summed RSS can double-count shared pages. It stops
only its own command and descendants. Keep container memory limits as the hard
backstop. Optional GPU monitoring identifies one physical UUID but neither
reserves it nor establishes permission or exclusive access.

- **vLLM container** (used by GPU acquisition, `python -m bench run`, and
  `python -m workloads.generators`): `vllm/vllm-openai:v0.31.0` (CUDA 13;
  requires a compatible NVIDIA driver)
  - Launched via `scripts/docker-vllm.sh`. Set `VLLM_GPUS` to a docker
    device spec to keep it off GPUs someone else is using; the inner
    quotes are part of the value (`VLLM_GPUS='"device=2,3"'`), and
    without them docker reads the second field as a GPU count. Default
    is every GPU on the host
  - Mounts the **LLMServingSim repo root** as `/workspace`; container cwd
    is `/workspace`, so `python -m profiler …` etc. work directly
  - Installs `datasets`, `matplotlib`, `pandas` and Torch's declared
    `nvidia-nccl-cu13==2.29.7` dependency. This package pin does not force
    NCCL-only communication or change the benchmark's communication policy
  - Applies the MiniMax-M3 MTP namespace fix and inert-by-default observers.
    The SM120 sparse-MLA fix is upstream in 0.31.0; do not reapply its obsolete
    backport. `VLLM_STEP_SHAPE_LOG` is an optional CPU-entry cadence diagnostic,
    not a GPU step timer. `bench --record-gate-stats` enables the modular-router
    observer and requires eager execution; monolithic MoE paths can bypass it
  - `scripts/install-vllm.sh` targets Python 3.12 in the repository's `.venv`,
    uses uv's `--torch-backend cu130`, and applies the same dependency pin and
    patches. Check installed versions and live model/backend coverage separately
  - Export `HF_TOKEN` in the shell for gated-config auto-download; the launcher
    forwards it without embedding credentials in the script
- **Simulator container**: `astrasim/tutorial-micro2024` + Python deps
  - Launched via `scripts/docker-sim.sh`
  - Mounts the repo root at `/app/LLMServingSim`; ASTRA-Sim + Chakra are
    built inside via `scripts/compile.sh` on first use
  - `python -m bench validate` compares recorded files here on CPU, without
    booting vLLM or using a GPU

## README and docs split

The repo has two documentation surfaces with deliberate scope:

- **`README.md`** — minimal front door. Latest News / About / Getting Started /
  Publications / Citation only. Logo + link bar (Website / Documentation / Contribute /
  Contact / Changelog) point everything else out to the website. **Do not
  re-add detailed content (CLI flag tables, dataset schema, profiler
  walkthroughs, validation plots, etc.) to the README** — it lives on the
  website now. Keep Getting Started to runnable commands, the CPU/GPU boundary,
  runtime versions and documentation links. Measurement internals, model-specific
  limits and contribution policies belong in the guides.
- **`docs/`** — the public docs site (Docusaurus 3, deployed at
  `https://llmservingsim.ai`). All long-form content lives here. See
  `docs/AGENTS.md` for site-specific conventions.

When you add a new feature with user-visible behavior, document it on the
website (not the README).

Start directory READMEs with a short introduction to their purpose, inputs and
outputs, plus links to the main usage sections. Put detailed measurement rules,
model-specific limits and recovery contracts under topic headings after the
usage entry points; do not accumulate change descriptions in the introduction.
Directory READMEs use `## Layout` for source trees and `## Output schema` for
generated bundles. Use `text` fences, `├──` / `└──` / `│` connectors and four-column
nesting; keep explanations outside trees and label selected inventories as such.
Use language-tagged code fences, sentence-case section headings, and blank lines
after headings and before lists or code blocks. Do not rewrite dependency
READMEs for cosmetic consistency. Latest News links completed work to its PR;
in-progress entries are roadmap statements, not supported behavior or accuracy claims.

## Workload generation

ShareGPT free generation forwards `--vllm-repetition-penalty` to
`SamplingParams.repetition_penalty` with the parsed default of `1.1`;
`1.0` disables the penalty. Keep registered sampling options connected to
the generation call and check both defaults and explicit overrides. This path
requires `--use-vllm` without `--fix-len`; existing workload files are unchanged.
Do not promise an output length or EOS behavior from the penalty alone.

## Commit & Pull Request Guidelines

- Short imperative commit messages: `Fix incorrect evict_size accumulation`,
  `Add Qwen3 model support`
- Keep commits focused — one logical change per commit
- Record verified fixes for established bugs in separate commits. Commit only
  adopted changes, not unresolved experiments.
- **Do not commit intermediate work.** This includes development/session notes,
  intermediate measurements and generated results, experimental code, and
  temporary scripts or test code written to diagnose or verify a fix. Run
  validation locally, but keep these temporary files out of the staged diff.
  Do not delete retained experiments merely because they are excluded.
- **Every commit must include documentation updates**, including docs-only
  commits: the root README.md and affected directory READMEs, AGENTS.md,
  CHANGELOG.md, and relevant public pages under docs/. Review repository
  READMEs, including per-directory files, for affected or stale descriptions.
  Regenerate the site's changelog from CHANGELOG.md. Keep README updates
  concise and consistent with the README/docs split above.
- Public documentation describes final supported behavior, usage, verified
  changes and limitations only. Never publish session diaries, investigation
  checkpoints or intermediate experiment notes, including on contributor pages
  or in this file.
  Public documentation and reproduction instructions must not reference or
  depend on private archives or untracked intermediate experiment scripts/results.
- Keep one changelog section per release and one heading per change category.
  Consolidate related Unreleased entries into short, user-facing summaries;
  retain migration requirements, limitations and contributor credit. Link to
  guides for implementation details instead of accumulating per-commit stories.
  Preserve published release records when tidying pending changes.
- When consolidating release notes, compare commits and the net code diff since
  the previous release, including changed submodules. The old changelog is not
  an exhaustive inventory; do not advertise reverted experiments as features.
- Before committing, inspect the staged file list and diff for both exclusions
  and mandatory documentation coverage. Do not stage the whole dirty worktree.
- Do not add AI-tool references or AI attribution trailers to commit messages.
- Publish required submodule commits before their parents: Chakra, ASTRA-Sim,
  then the frontend. Check remote reachability and use
  `git push --recurse-submodules=check` for the parent; do not publish local backup
  refs or force-push over independent remote work. Rebuild the backend and
  reinstall Chakra after updating the recursive submodules.
- Include the exact command used for validation and note any output CSV path in PRs
- Describe which simulation mode is affected and the config/dataset used

## Testing & Validation

Published accuracy belongs to a fixed profile bundle, recorded benchmark and
execution contract. Reproducing the committed examples does not certify a
fresh-only acquisition from the latest profiler or every catalogued model.
Keep availability, acquisition provenance, deterministic regression and
end-to-end accuracy distinct in all public documentation. Historical coarse-grid
shot counts are examples, not current-default runtime estimates.

The example wrappers discover `bench/examples/*/*/config.json`, including
stored diagnostics beyond the headline benchmarks. Reproduction uses the
example's target graph policy and memory configuration. The RTXPRO6000 headline
configs inherit transport from the hardware bundle. Llama and both Qwen
bundles contain refreshed TP1 and TP2 acquisition data; headline validation
covers Llama TP1, Qwen3-32B TP2 and Qwen3-30B TP1/DP2/EP2 only. Preserve
adopted generated values and rows: profile and hardware metadata, measurement
tables, benchmark metadata, simulation outputs, validation files and logs.
The only publication exception is portable identifiers for environment-dependent
dataset and local model paths. Use repository-relative dataset paths and canonical
model identifiers where equivalent; temporary sharded-model identifiers must
retain their TP-shape semantics. Do not prune records or change measurements,
settings, hashes or validation results. Publish the runtime tables
described by the generated bundle metadata. Qwen3-30B's
instance memory fractions match the complete per-engine READY block counts,
not the aggregate client count or an end-to-end latency target. Reproduction uses the
recorded `kv_cache.block_size` unless `BLOCK_SIZE` explicitly overrides it.
If that metadata is absent, omit the flag so serving resolves its default;
do not force a dense-only page size onto sparse or hybrid examples.

Use focused local checks for changed contracts; temporary diagnosis and
fix-verification scripts or tests stay outside the published tree. The simulator
is deterministic, so its regression validation checks exact equality against
recorded results:

**Several parts of `profiler/` are simulator inputs**, despite the path.
Shared data and CPU-only helpers can change simulation clocks:

- **`profiler/models/*.yaml`** — the layer order. Merging two catalogs into one
  broke all 16 MoE scenarios exactly this way.
- **`profiler/core/stack.py`** — which block each decoder layer runs, resolved
  from the checkpoint's config.
- **`profiler/core/catalog_path.py`** — `model_type` → yaml resolution.
- **Attention shape, skew calibration and MoE contract helpers** — geometry,
  correction-table loading and deployment compatibility affect lookup.
- **`profiler/perf/`** — latency tables and hardware defaults.

These helpers must remain usable in the CPU simulator environment without
importing vLLM or GPU acquisition dependencies. Audit actual imports rather
than assuming only the stack and catalog resolvers can affect simulation.

1. **`./serving/validate.sh`** — the whole check, ~8 min. Stage 1 compares every
   scenario against the `Total clocks (ns)` recorded in
   `serving/validate-baselines.txt`; stage 2 regenerates each `bench/examples`
   entry's `outputs/sim.csv` and `validation/summary.txt` and checks both md5s.
   Anything that moved is printed as a markdown table for the PR.
   `--clocks-only` skips stage 2, `--list` names the scenarios, `--update`
   rewrites the baselines, `--help` prints the rest.
   **Run it after every commit that touches `serving/`** — it is cheap enough,
   and it is how the 8-of-19 regression on the perf branch was caught.
2. For the *size* of an accuracy change, not just its presence:
   `./bench/examples/validate.sh` regenerates `validation/summary.txt` and the
   three plots. A changed `sim.csv` makes those stale, so regenerate and commit
   them in the same commit.
3. For profiler changes: edit `MODEL` / `HARDWARE` in `profiler/profile.sh`
   and run `./profiler/profile.sh` from the repo root inside the vLLM container.
4. For a catalog change (new or edited `profiler/models/*.yaml`), and after a
   vLLM upgrade: `python -m profiler coverage <model> --hardware <hw>` inside
   the vLLM container. It boots once, runs one forward per batch regime, and
   exits non-zero on either defect it can find. A **gap** is CUDA time no
   entry claims. An **over-match** is one entry claiming nodes in
   structurally unrelated places, so its number is the sum of two roles —
   which the coverage percentage cannot show, because over-matching leaves
   nothing unbound. It fires only when the matched nodes' ancestor chains
   share no common prefix, so an entry legitimately binding several kernels
   under one parent (M3's `sparse_attention`, any `*`-prefixed glue) is not
   flagged. Qwen3.5's `mtp_norms` read **1287 us for two RMSNorms** and its
   `embedding` read **2.1x** while passing coverage at 100%. Inspect dynamic
   invocations, native activities and unscoped gaps, not only the static module
   inventory. Coverage does not prove every declared alternative executed:
   an entry absent from every forward can leave no activity gap. Required-entry
   checks, including the MTP slice check, are separate. Shared-activity overlaps
   must never be suppressed by the target/drafter repeated-role exemption.

DP/TP/PP scenarios with uneven request streams can change round pairing when
a component cost changes. Total clocks may then move non-monotonically even
when that component gets faster. Preserve these scenarios as behavior checks;
inspect the resulting batches, request completion and token counts before
attributing a changed digest to a kernel-cost regression. An aggregate clock
alone does not identify its cause.

A scenario whose clock equals an existing one exercises flag parsing and
nothing else. Several knobs only bite once the KV cache is saturated, which is
what the `saturated_*` scenarios are for; `example_trace.jsonl` never gets
there, and its DP members always drain together, which is what the `*_uneven`
scenarios are for.

**A model family is an axis too.** Everything but the `hybrid_*`, `sparse_*`
and `spec_*` scenarios runs Llama-3.1-8B/70B, Qwen3-30B-A3B or Qwen3-32B, so
none of it reaches linear attention, sparse attention, MLA, a heterogeneous
stack or the drafter — a speculative `pp_size > 1` deadlock lived through a
green 58/58 for exactly that reason. Their cluster configs use the **smallest
parallelism in which each model fits** on the profiled hardware (DeepSeek-V3.2
is 625 GB at fp8, so PP=16 is a floor `MemoryModel` enforces, not a choice),
and PP rather than TP because only `tp1` is profiled for three of the four.
`--num-reqs` is deliberately small: what these check is which layers get
emitted, and the stacks are 60-78 deep.

## Common Pitfalls

- **Don't reintroduce `sequence:`, or any second layer-order form.** There is
  one: `blocks:` + `shared:`. A uniform stack is the degenerate case, one entry
  per axis. `sequence:` was that flattened, and two forms meant two code paths —
  the simulator only implemented the flat one, so Qwen3.5 and MiniMax-M3 could
  not be simulated at all. It also encoded a false claim: with the axis in the
  key there is no per-layer question left, and the MLP was resolved once per
  *model*, modelling DeepSeek/GLM's first three dense layers as MoE
- **Don't resolve a per-layer property once per model.** Which block a layer
  runs comes from the checkpoint via `profiler/core/stack.py`, per layer. The
  tell for this class of bug is a name like `is_moe` on the context object
- **Don't reuse a built transformer block across layers without keying on the
  block shape.** `_block_copy_key` returns the layers' `LayerSpec`s, and the
  replay is what keeps trace generation O(1) in depth. Getting the key wrong is
  invisible on a uniform model: an earlier version emitted **one** block
  instead of `num_hidden_layers` whenever block copy was disabled, understating
  the clock 3.1x on 48 layers, and the recorded baseline enshrined it
- **Don't add a layer name without a formula on its selected sizing path.**
  Ordinary catalogs use `memory_model.calculate_sizes`; native/contextual
  profiles use the admitted providers in `component_sizes.py`. Check the
  selected layout, not every native name against the ordinary dispatcher.
  Logical bytes in a duration trace do not imply serving-capacity support
- **Don't edit `astra-sim/`** unless the change targets simulator integration
  (e.g., `llm_converter.py`, `Workload.cc`, input configs). Chakra is *installed*
  into the container's site-packages by `scripts/compile.sh`, so editing
  `llm_converter.py` changes nothing until you reinstall it
  (`cd astra-sim/extern/graph_frontend/chakra && pip3 install .`)
- **Don't commit large files**: generated traces, `.et` files and scratch run
  output are gitignored (`outputs/*` with `!outputs/example_*.csv`,
  `bench/results/`). `astra-sim/inputs/runs/` is cleaned per run, and each
  batch's workload folder already during the run once `pp_size + 1` newer ones
  of the same instance or DP group have been handed out, unless you pass
  `--keep-inputs` (or `--save-trace-text`, which implies it), either of which
  can leave gigabytes behind
- **Don't use machine-specific absolute paths** in configs or code — use relative paths
  rooted at the repo
- **Don't add `getattr` fallbacks** for Request attributes — initialize all attributes
  in `Request.__init__` and access directly
- **Don't reintroduce `is_prefill()` or a prefill/decode branch in the scheduler.** vLLM has
  neither; a request just catches up to `num_tokens_reached`. Classify for the trace by
  scheduled token count instead
- **Don't add a "preserve the decode state on preemption" special case.** `num_computed_tokens
  = 0` is vLLM's own behaviour and is not re-prefill — recovery comes from the block hashes
  surviving in the pool and from the lower tier. Two earlier attempts to special-case this
  cost 375 preemptions / 293k recomputed tokens and 41,569 preemptions / 6 TB of swap
- **Don't derive sequence length from `num_computed_tokens`** — preemption resets it.
  `num_tokens_reached` is the independent counter, mirroring vLLM's `len(_all_token_ids)`
- **Don't split pipeline stages by trace-line count.** `pp_stage_boundaries` in the
  trace header exists because a stage may only be cut on a transformer-block
  boundary: that is the one place where the upstream `output_size` and the
  downstream `input_size` are the same tensor (the hidden state). Inside a block
  they differ — `qkv_proj` emits Q+K+V while `rotary_emb` declares only Q+K — and
  ASTRA-Sim's analytical backend keys its send/recv tracker on
  `(tag, src, dst, chunk_size, chunk_id)`, so a mismatch silently deadlocks the
  receiving NPU instead of raising. That was issue #55: only the `pp_size` values
  whose line-count cuts happened to land between two size-agreeing layers ran
- **Don't gate a DP batch on which NPU is asking.** `sys == inst2npu_mapping[i]` does
  **not** mean "built this batch": any NPU of an instance may open a DP round, so the
  start NPU can arrive holding a batch it joined through `_schedule_existing`. Route on
  whether the batch is new (`len(batch.fired) == 1`) instead, or the round is
  registered twice and the graph regenerated. And never hand an NPU a DP batch whose
  `workload_name` is still `None` — that derives the solo `instance<id>_batch<id>`
  folder, which is never written for a DP batch, and ASTRA-Sim logs
  `[critical] workload file ... does not exist` and then **hangs instead of exiting**.
  Both were issue #65; `add_done` needs `start_npu in batch.end` to complete a batch,
  so any batch the start NPU cannot claim deadlocks silently
- **Don't keep per-batch scheduler state on a `Request`.** If `add_done` reads
  it, it belongs in the `Batch`, snapshotted at build time like
  `scheduled_tokens` and `spec_scheduled`. At `pp_size > 1` the next
  `schedule()` runs while the batch is in flight and rewrites the request's
  fields for the step it is building, so `add_done` reads the wrong step's
  values. That deadlocked every speculative `pp_size > 1` run and is invisible
  at `pp_size == 1`
- **Don't give a DP group one slot per member anywhere.** `dp_pending[dg][inst]` and
  `dp_ready_workloads[npu]` are both FIFOs because at `pp_size > 1` a member has up to
  `pp_size` batches outstanding. Three separate hangs came from single slots: the second
  registration dropped the first batch from the barrier (it never got a `workload_name`,
  so the instance's other NPUs retried joining it forever), and the second round
  overwrote an unconsumed workload (the NPU then ran the *wrong* microbatch's graph). All
  three are invisible at `pp_size == 1`, where an instance holds one batch at a time
- **Don't let the start NPU's join depend on the pipeline being full.** `schedule()` tries
  `_schedule_existing` *before* the `len(inflight) >= pp_size` cap. A dummy opened by a
  non-start NPU leaves `pp_size - 1` slots free, and gating the join on a full pipeline
  sent the start NPU down the build path, where an idle member has nothing to build — so
  it passed forever and `add_done` never saw `start_npu in batch.end`
- **Don't assume `hidden_size == num_heads * head_dim`** — use explicit `head_dim` from config
- **Use canonical vLLM layer names** (`qkv_proj`, `o_proj`, `gate_up_proj`,
  `act_fn`, `down_proj`, `rotary_emb`, `qk_norm`, `attention`, `layernorm`,
  `final_layernorm`, `embedding`, `lm_head`, `sampler`, `moe`). Every name the
  simulator emits must also appear in the architecture yaml's catalog.
- **Profiler CSVs store microseconds** (`time_us` column) — the simulator
  multiplies by 1000 and rounds to nanoseconds at load time
- **First and last trace layers must use REMOTE** — the Chakra converter creates a MEM_LOAD
  node from the first layer's input_loc and a MEM_STORE node from the last layer's output_loc;
  if either is LOCAL without local_mem configured, ASTRA-Sim crashes
- **memory_expansion.json only has remote_mem by default** — local_mem is not configured unless
  `--enable-local-offloading` is used; weight loads from LOCAL go through compute time, not memory
- **`config_builder.py` regenerates ASTRA-Sim inputs on every run** — don't manually edit
  `astra-sim/inputs/` files expecting them to persist
