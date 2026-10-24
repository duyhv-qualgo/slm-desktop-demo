# Why inference is fast or slow — memory, compute, and the CPU/GPU split

Every performance number in this app comes from one fact and its consequences.
This note explains the mechanism, then records what was actually measured on real
devices, so nobody re-derives it or re-runs the experiments that already failed.

Companion to [HANDOFF.md §4](HANDOFF.md), which covers how `ChatLM.mm` works.
Where they overlap, HANDOFF describes the *code*; this describes the *physics*.

> **On the `integration` branch:** the numbers below were measured on `main`,
> with the 3-slot KV prefix cache and the benchmark tooling (`BenchmarkHarness`,
> the probes). Both were removed for handover — every call now pays a full
> prefill, so per-turn suggest latency is higher than the cached figures here.
> The device physics (prefill vs decode, CPU vs GPU, memory) is unchanged. See
> [integration.md](integration.md).

All figures below are measured, on the devices named. Where something is an
estimate or an inference it says so.

---

## 1. The one fact

A dense model uses **every** weight for **every** token, and the weights are far
too big for any cache. So they stream from DRAM into the chip, get multiplied, and
are evicted — then read again for the next token.

```
Flash ──(once, at load via mmap)──> DRAM ──(every token)──> SoC caches ──> ALUs
```

Caches on a phone SoC hold megabytes. `qwen3-0.6b-v6` Q8_0 is **604 MiB**. By the
time layer 28 is read, layer 1 is long gone. Nothing survives between tokens.

Everything below follows from that.

> Dense models only. A mixture-of-experts model activates a fraction of its
> weights per token and breaks this rule — which is why MoE is attractive
> on-device. Neither checkpoint here is MoE.

---

## 2. Decode and prefill read the same bytes — and differ completely

| | weights read | tokens produced | bytes per token | reuse while on-chip |
|---|---|---|---|---|
| **Decode** | whole model | 1 | **604 MiB** | **1×** |
| **Prefill** (`n_ubatch=128`) | whole model | 128 | **~5 MiB** | **128×** |

Prefill does not read less. It gets 128× more work out of each byte before
evicting it. That single difference decides which resource limits each phase:

**Decode is memory-bound.** ~2 arithmetic ops per byte loaded; the processor
waits on DRAM. Time is a division:

```
time per token = model_bytes ÷ memory_bandwidth
```

**Prefill is compute-bound.** The read is amortised 128 ways, so per token it is
trivial. Reading 604 MiB at ~30 GB/s costs 21 ms — measured against the whole
prefill pass on iPhone 14 Pro:

| | 128 tokens take | of which the DRAM read is |
|---|---|---|
| Prefill, CPU | 1160 ms | 21 ms — **2%** |
| Prefill, GPU | 262 ms | 21 ms — **8%** |
| Decode, CPU | 22 ms (1 token) | 22 ms — **~100%** |

### Consequences you can rely on

- **Decode speed is inversely proportional to model size.** Exactly, because it
  is a division. Measured on iPhone 14 Pro CPU: gemma 0.29 GB → 106 tok/s;
  qwen3-v6 0.64 GB → 45 tok/s (2.2× the bytes, 2.4× slower).
- **A smaller model is the only real decode lever.** Quantization helps decode
  for the same reason — fewer bytes to move.
- **Prefill responds to kernels, cores and clocks, not to bandwidth.** Every
  prefill win recorded in §5 left decode untouched, which is the signature.

---

## 3. Why the GPU loses at decode and wins at prefill

Measured on iPhone 14 Pro, same model, same build:

| | prefill | decode |
|---|---|---|
| qwen3-v6 CPU | 110.4 tok/s | 45.2 tok/s |
| qwen3-v6 **GPU** | **489.3** tok/s | **30.8** tok/s |
| gemma CPU | 430.5 tok/s | 106.1 tok/s |
| gemma **GPU** | **818.4** tok/s | **48.2** tok/s |

The GPU is 1.9–4.4× faster at prefill and **32–55% slower at decode**. Two
reasons, and the second is the big one.

**Unified memory means no bandwidth advantage.** CPU and GPU pull from the same
DRAM. Decode asks for bandwidth, not compute, so the GPU brings nothing to the
part that matters.

**Decode pays a fixed per-token round trip.** One token per forward pass means,
every token:

```
CPU encodes work → submit → GPU wakes/ramps → executes
                                                  ↓
CPU blocks (waitUntilCompleted) ← logits synced back
     ↓
CPU samples token N → only now can token N+1 start
```

It cannot be pipelined: token N+1's input is token N's output. Prefill runs the
same round trip **once for the whole prompt**.

### Sizing that overhead — two independent derivations

*Direct subtraction*, assuming equal execution time:

```
qwen3-v6:  32.4 − 22.1 = 10.3 ms/token
gemma:     20.7 −  9.4 = 11.3 ms/token
```

*Bandwidth-normalised.* Convert decode rates to effective GB/s:

| | ms/token | effective GB/s |
|---|---|---|
| qwen3-v6 CPU | 22.1 | **28.9** |
| gemma CPU | 9.4 | **30.9** |
| qwen3-v6 GPU | 32.5 | 19.7 |
| gemma GPU | 20.7 | 14.1 |

The CPU lands within 7% on two models differing 2.2× in size — the signature of a
saturated bus (~60% of the A16's peak, a normal practical ceiling). The GPU's
apparent bandwidth is inconsistent, so it is *not* bandwidth-limited. Assume it
also reaches ~30 GB/s and solve for the remainder:

```
qwen3-v6:  32.5 ms − 21.3 ms execute = 11.2 ms overhead
gemma:     20.7 ms −  9.7 ms execute = 11.0 ms overhead
```

**11.2 and 11.0 ms** — a 2% spread across models 2.2× apart. Two independent
methods agreeing is why this is stated as fact.

> **What is still open:** how that ~11 ms splits between dispatch/synchronisation
> and GPU clock ramping on a workload too short to sustain a boost. An `n_ubatch`
> sweep would separate them. The *existence* and *size* are settled; the *cause*
> is inferred.

### The rule that falls out

**Prompt-heavy work wants the GPU. Output-heavy work wants the CPU.**

```
GPU wins when:  prompt_tokens × (1/cpu_prefill − 1/gpu_prefill)  >  output_tokens × 11 ms
```

Checked against measurement: qwen3 summarize → predicted GPU win ~1.1 s (observed
0.61 s); gemma summarize → predicted GPU **loss** ~0.5 s (observed 0.67 s loss).
Right direction both times.

**The app currently picks one backend globally, so gemma runs on the wrong one on
an iPhone 14 Pro** (CPU 1.17 s vs GPU 1.85 s for the same summarize). Making
backend a per-`SLMModelOption` property is an open improvement.

You cannot run prefill on the GPU and decode on the CPU: the KV cache lives on
whichever device created the context, and the backend is a load-time property.
It would take two contexts and a cache copy every turn.

---

## 4. Device ceilings

| device | model | backend | prefill | decode | TOTAL (summarize) |
|---|---|---|---|---|---|
| iPhone XS (A12) | qwen3-v6 | cpu | 47.0 | 14.8 | **11.56 s** |
| iPhone XS (A12) | gemma | cpu | 182.9 | 49.3 | 2.46 s |
| iPhone 14 Pro (A16) | qwen3-v6 | cpu | 110.4 | 45.2 | 3.56 s |
| iPhone 14 Pro (A16) | qwen3-v6 | gpu | 489.3 | 30.8 | 2.95 s |
| iPhone 14 Pro (A16) | gemma | cpu | 430.5 | 106.1 | **1.17 s** |
| iPhone 14 Pro (A16) | gemma | gpu | 818.4 | 48.2 | 1.85 s |

**A 0.6B model on an A12 does not meet a 3 s budget.** 11.56 s versus gemma's
2.46 s on the same device — 4.7×. No kernel or backend change closes that.

### The A12 is missing SIMD the A16 has

`llama_print_system_info()`, logged once per process at load:

```
iPhone XS (A12):  NEON = 1 | ARM_FMA = 1 |               LLAMAFILE = 1 | ACCELERATE = 1 | REPACK = 1
Simulator (M4):   NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | DOTPROD = 1 | LLAMAFILE = 1 | ACCELERATE = 1 | REPACK = 1
```

No `DOTPROD` (SDOT — 16 int8 MACs per instruction) and no `FP16_VA` on the A12.
Int8 dot products are widened to int16 and run several instructions per MAC.
`MATMUL_INT8` (i8mm) is absent from **both**, so no device gets it from this
xcframework.

> **Correction, recorded so it is not repeated.** The missing `DOTPROD` was
> initially blamed for the A12's flat prefill:decode ratio. The device data
> refutes that as the primary cause: A12 → A16 on CPU scaled decode 3.1× but
> prefill only 2.4×. If SDOT were the dominant term, prefill should have gained
> *more*, not less. The feature gap is real; it is not what caps CPU prefill here.
> The thread count was the next suspect and is ruled out too (§5, measured).

### The XS may not even be bandwidth-saturated

```
XS + qwen3-v6:  9.5 GB/s
XS + gemma:    14.4 GB/s
```

Inconsistent, and only ~30–40% of the A12's peak — unlike the A16's steady
~30 GB/s. So XS decode is likely **partly core-limited**, not purely
bandwidth-bound: too few cores to issue loads fast enough to saturate the bus.

---

## 5. Settled experiments — do not repeat

### Quantization: Q8_0, not a K-quant (CPU)

Identical weights, same code path, same prompt:

| qwen3-0.6b-v6 | prefill | decode | ratio |
|---|---|---|---|
| Q4_K_M (397 MB) | 102 tok/s | 66 | 1.6× |
| **Q8_0 (639 MB)** | **366 tok/s** | **82** | **4.5×** |

**3.5× prefill for +242 MB.** llama.cpp's batched GEMM (`llamafile_sgemm`) covers
Q8_0 and Q4_0 but **not** the K-quants, so Q4_K_M runs per-row dot products for
the prompt pass as well as decode — the 1.6× ratio is the tell. Q8_0 is also the
less lossy quant.

> The Q8 verdict above is a *CPU* verdict. **Q4_0 on GPU has since been
> measured** (iPhone 14 Pro, 2026-10-05): +41–49% decode, prefill unchanged.
> On the CPU, Q4_0 (not only Q4_K_M) also loses to Q8_0, on the simulator and
> on the XS. See §11 and [Q4_0_PERF_REPORT.md](Q4_0_PERF_REPORT.md).

### Do not give BLAS/Accelerate to the CPU backend

| qwen3-v6 Q8_0 | prefill | graph splits |
|---|---|---|
| CPU + BLAS | 105 tok/s | 282 |
| **CPU alone** | **366 tok/s** | **1** |

**3.5× faster with BLAS excluded.** Accelerate implements matmul only, above a
batch threshold. Every other op — norms, softmax, RoPE, activations, and each
small LoRA matmul — stays on the CPU backend, so prefill alternates
CPU→BLAS→CPU through all 28 layers: 282 sync points and tensor copies. BLAS also
needs f32, so quantized weights are dequantized into scratch on every matmul.

At `bs=1` the batch is under BLAS's threshold, so it never engages and decode sees
1 split — which is why this hurt prefill only, and why **the prefill:decode ratio,
not the raw rate, is what exposes it.**

With BLAS on, Q8_0 measured 105 tok/s — *identical to Q4_K_M*. **BLAS was masking
the entire quantization win.** Changing quant without also dropping BLAS would
have cost +242 MB for nothing.

### Do not enable `use_extra_bufts` (weight repacking)

| | prefill before | with repack |
|---|---|---|
| gemma | 1125 | **252** (−4.5×) |
| qwen3-v6 Q4_K_M | 102 | **40** (−2.6×) |
| qwen3-v6 Q8_0 | 554 | **73** (−7.6×) |

Slower everywhere, decode collapsed 209 → 24 tok/s, **and memory doubled**:

```
CPU_Mapped model buffer =  370.96 MiB
CPU_REPACK model buffer =  372.40 MiB     ← 743 MiB total
```

The repacked layout's fast kernels need dotprod/i8mm; without them it falls back
to generic scalar code over a re-laid-out tensor. Worse still, the repacked copy
is *anonymous* memory (§6) — the expensive kind.

### `n_threads_batch`: keep the P-core count (measured, A12)

`ChatLM.mm` gives prefill the same 2 P-cores as decode. The suspicion was that
prefill, being compute-bound, would gain from the 4 idle E-cores. Measured on the
iPhone XS, ~400-token suggest-shaped prompt, reuse off, median of 3
(`GuardrailTests/PrefillThreadsProbe`, `ChatLM.batchThreads`):

| model (Q8_0) | 2 threads | 4 threads | 6 threads |
|---|---|---|---|
| gemma 270M | 216 tok/s | 221 | 195 |
| qwen2.5-0.5b-v6.1 | **62** | 54 | 50 |
| qwen3-0.6b-v6 | **38** | 35 | 38 |

**No gain; the Qwen models get slower.** The E-cores are too slow to share a
matmul evenly with the P-cores, so each layer waits on its slowest thread. The
default stays. On an A12 the only large prefill lever measured is the model:
gemma prefills **3.5× faster than qwen2.5 and 5.7× faster than qwen3**.

---

## 6. Memory: two kinds, and only one can kill you

Measured for the shipping config (`qwen3-0.6b-v6` Q8_0 + one LoRA pack):

```
CPU_Mapped model buffer =  604.15 MiB   ← clean, file-backed
CPU KV buffer           =  336.00 MiB   ← dirty
CPU compute buffer      =   77.46 MiB   ← dirty
LoRA adapter            =  ~39    MiB   ← dirty
                          ─────────────
total resident          = ~1.03 GB      (dirty: ~452 MiB)
```

**Clean, file-backed** — `mp.use_mmap = true` maps weights straight off the
`.gguf`. Under pressure the kernel drops these pages and re-reads from flash. They
appear in RSS but generally not in `phys_footprint`, which is what jetsam acts on.
The code comment says exactly this: *"mmap weights (keeps phys_footprint low)"*.

**Dirty, anonymous** — nothing on disk backs these; they can only be compressed.
**This is the real budget exposure: ~452 MiB, not 1.03 GB.**

### The KV cache is the biggest dirty allocation, and it is a choice

336 MiB comes from `n_ctx = kCtxPerSlot(1024) × kSlotCount(3)` — **112 MiB per
slot**. Two of the three slots exist only for prompt-prefix reuse. Dropping to 2
slots saves 112 MiB of dirty RAM; to 1 saves 224 MiB. On a 4 GB iPhone XS that
trades directly against cache-reuse latency.

> The precise `phys_footprint` accounting rules are Apple's; the above is standard
> behaviour plus the code's stated intent, not a measurement. Xcode's memory gauge
> gives the real number in a minute if a shipping decision depends on it.

---

## 7. Reading the logs

Every generation emits one line. Fields relevant here:

```
[SLMPerf] task/suggest qwen3-0.6b-v6(cpu) | in=257 tok (cached 32, eval 225, slot 0)
  | prefill 4.052s (55.5 tok/s) | ttft 4.056s | out=49 tok (161 chars)
  | decode 1.947s (25.2 tok/s) | lora=…vietnamese.gguf/same +0.000s
  | overhead 0.014s | TOTAL 6.014s | stop=eos | cap=160 temp=0.80
```

- **`prefill … tok/s`** is per **evaluated** token, so a cached turn reports a
  near-meaningless rate. Settings → Diagnostics → **Reuse cached prompt prefix**
  turns reuse off, forcing a full prompt pass, which is the only way to read true
  throughput.
- **`prefill:decode` ratio** is the diagnostic that transfers between devices;
  absolute rates do not. Below ~2× means no batched path is engaged.
- **`stop=`** — `eos` means the model chose to stop and `cap` was not binding.
  Across 13 device turns, **none** hit `max_tokens`, so lowering caps shortens
  nothing; it only truncates.
- **`lora=<pack>/<state>`** — `same` / `warm` / `cold`. `cold` means a pack was
  parsed off disk inside that call.
- Once per process at load: `[SLMPerf] llama system info: …` — the SIMD feature
  set. **Read this before reasoning about any device's prefill ceiling.**

---

## 8. If a turn is too slow, in order

1. **Check `prefill:decode`.** Under ~2× means a batched path is missing —
   suspect quantization (§5) or backend devices, not the model.
2. **Check `cached` vs `in`.** Prefill is linear in *evaluated* tokens. Reuse is
   the only free lever; measured average is 64%, but the window steps every third
   suggest and re-prefills everything (`GuardrailTests/SuggestReuseTests`).
3. **Check which phase dominates.** Prefill-heavy → GPU and §5's levers.
   Decode-heavy → only a smaller model or fewer output tokens will help.
4. **Then consider the model.** On an A12 this is the first question, not the
   last.

---

## 9. Where the tests live

| what | file |
|---|---|
| prefill/decode throughput per model, ± LoRA | `GuardrailTests/TurnMetricsTests.swift` (`PrefillThroughputProbe`) |
| metrics are real, not under-measured | `GuardrailTests/TurnMetricsTests.swift` |
| prefix-reuse switch does defeat the cache | `GuardrailTests/TurnMetricsTests.swift` (`PrefixReuseSwitchTests`) |
| how much of each prompt is reused | `GuardrailTests/SuggestReuseTests.swift` |
| real per-turn `eval` tokens + pre-warm hit rate, stride sweep | `GuardrailTests/SuggestKVHarnessTests.swift` (grep `KVH`) |
| where output tokens go | `GuardrailTests/OutputBudgetTests.swift` |
| Q8_0 vs Q4_0, rank-32 vs rank-64 pack vs merged, CPU or GPU | `GuardrailTests/QuantFormatProbe.swift` (grep `QUANT`) |

```bash
xcodebuild test -scheme SLMChat -destination 'platform=iOS Simulator,name=iPhone 17' \
  -only-testing:GuardrailTests/PrefillThroughputProbe
```

> Simulator figures are **not** device figures — it runs natively on the Mac's
> CPU and takes the `TARGET_OS_SIMULATOR` CPU-only path. Use it to compare
> settings against each other; use a device for absolute numbers.

---

## 10. One defect worth knowing about

`llama_perf_context`'s `t_p_eval_ms` intermittently reports near-zero for a prompt
pass that took seconds — observed as `prefill 0.000s (5977777.8 tok/s)` alongside
`ttft 3.450s`. `TOTAL` stayed correct, so nothing looked broken; the missing
seconds were silently reattributed to `overhead`.

`ChatLM.mm` therefore times the prompt pass itself and uses llama.cpp only for
token *counts*, which are always right. `TurnMetricsTests` pins this — including
an implausible-rate ceiling, since a too-low duration is invisible on its own but
unmistakable once divided into a token count.

---

## 11. Future enhancements — measured, not shipped yet

Everything here was measured on iPhone 14 Pro (A16) GPU unless stated. The
app still ships Q8_0 bases; the Qwen packs are rank 64 since 2026-10-05
(Gemma packs stay rank 32 — CPU only). Full tables, method and
caveats: [Q4_0_PERF_REPORT.md](Q4_0_PERF_REPORT.md).

### QAT Q4_0 for GPU phones (A14+)

GPU decode is bandwidth-bound (§2), so halving the weight bytes pays off;
prefill is compute-bound and doesn't move.

| bare model, en | Q8_0 decode | Q4_0 decode | prefill |
|---|---|---|---|
| qwen2.5-0.5b | 60.8 | **90.5** | unchanged |
| qwen3-0.6b | 46.1 | **64.9** | unchanged |
| qwen3.5-0.8b | 35.3 | **51.6** | unchanged |

- Use llama.cpp's **default** Q4_0 recipe (Q6_K output head). A Q8_0 head is
  15–35% slower on Metal.
- **CPU-only phones (A12) stay on Q8_0.** Q4_0 adds a 4-bit unpack to a
  compute-bound int8 path and was never faster on the CPU.
- Accuracy is QAT's job and is not measured yet; these Q4_0 files were
  requantized from Q8_0.

### LoRA packs at rank ≥ 64, or merged

Rank-32 packs are the biggest GPU prefill loss in the app. Metal tiles a
matmul only when its inner dimension is ≥ 64. In `B·(A·x)` that dimension is
the rank, so a rank-32 B runs on the one-token kernel for the whole prompt.
Zero-padding to rank 64 (double `alpha`) gives identical output.

Q4_0, ~390-token prompt + 45 output tokens:

| model | rank-32 pack | rank-64 pack | merged |
|---|---|---|---|
| qwen2.5-0.5b | 1.95 s | 1.21 s | **0.91 s** |
| qwen3-0.6b | 2.20 s | 1.35 s | **1.00 s** |
| qwen3.5-0.8b | 2.89 s | 1.91 s | **1.48 s** |

- Prefill vs merged: rank 32 24–30%, rank 64 83–87%.
- Decode vs merged: 70–76% for **any** unmerged pack. The cost is the extra
  ops per token (4 per adapted weight), not the rank.
- Merged (`llama-export-lora`, then quantize) runs at base speed, at the cost
  of one full model file per language.
- Padding: `scripts/pad_lora_rank.py in.gguf out.gguf 64`.

### Shipping (`dev-slm`) vs proposed — qwen3.5-0.8b, one session

Re-run on 2026-10-05 with the `dev-slm` source (`c7f8fc7`) and
`QuantFormatProbe` added only for the run. iPhone 14 Pro GPU, en prompt
398 tokens; turn = prefill + 45 output tokens. All configs ran back to back
in one session.

| config | prefill tok/s | decode tok/s | turn |
|---|---|---|---|
| Q8_0, no pack | 866 | 32.5 | — |
| **Q8_0 + rank-32 pack (shipping)** | **261** | **25.9** | **3.26 s** |
| Q8_0 + rank-64 pack | 758 | 25.6 | 2.28 s |
| Q4_0 + rank-32 pack | 262 | 32.6 | 2.90 s |
| **Q4_0 + rank-64 pack (proposed)** | **762** | **33.0** | **1.89 s** |

- The shipping config misses the 3 s budget. The proposed one takes 42% less
  time per turn: 2.9× prefill, 1.27× decode.
- With rank 32, prefill is stuck at ~262 tok/s whatever the base format. Q4_0
  only pays off together with rank ≥ 64.
- **Check the phone's state before trusting a slow run.** An earlier run of
  this same build, right after a 3 GB USB install, gave 231 / 10.3 tok/s
  (no pack) and 68 / 8.8 (rank 32): 3.5–4× slow on every config. Minutes
  later it reproduced the numbers above. If a known config (Q4_0 no pack:
  879 / 43.7) doesn't reproduce, the device is throttled; rerun.

### Order of work

1. ~~Convert every pack to rank ≥ 64~~ — **done 2026-10-05** for all 21 Qwen
   packs (padded; originals in `models-lab/lora-rank32/`; 67–83 MB each).
2. QAT Q4_0 bases for A14+ GPU phones; keep Q8_0 for the A12.
3. Decide merged-per-language vs base + pack (download size vs ~25% decode).
4. Measure an A14/A15 phone; only the A16 is measured.
