# Ternary Bonsai 2 2B on llama.cpp (macOS)

How to run [prism-ml/Ternary-Bonsai-2-2B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf)
with llama.cpp on a Mac, with LoRA language packs, standalone and in SLMChat.
Branch `dev-q2-lora`. Numbers below are from an M4 with 16 GB.

## What to know first

- **Use the `Q2_0` file** (547 MB). It is the only Bonsai 2 format official
  llama.cpp runs. `PQ2_0` and `PTQ1_0` need PrismML's fork. PTQ1_0 is under
  review upstream in [#29672](https://github.com/ggml-org/llama.cpp/pull/29672).
- **llama.cpp b11556 or newer.** Bonsai 2 stores its weights pre-multiplied by a
  Walsh–Hadamard rotation (`W_f = W·D·H`, 512-wide blocks). The runtime must rotate
  each matmul input to match.
  [PR #29600](https://github.com/ggml-org/llama.cpp/pull/29600) (first build:
  b11556) added that. Builds from August up to b11556 know Q2_0 but not the
  rotation: they load the file and print gibberish. Our bundled framework and
  Homebrew b9010 were older still and refused the file.
- **LoRA needs a patch.** Official llama.cpp refuses any LoRA on a rotated
  weight. In the 2B every projection is rotated. See [LoRA](#lora-language-packs).
- **Turn thinking off** (`enable_thinking: false`). It's on by default, and short
  prompts can use 3,000+ thinking tokens.
- It is the same `qwen35` hybrid attention/SSM architecture as Qwen3.5 0.8B.

## 1. Run it standalone

```bash
# official llama.cpp >= b11556 (Homebrew 0.7.0 = build 11669 works)
brew upgrade llama.cpp && llama-cli --version          # must say build >= 11556

curl -LO https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/resolve/main/Ternary-Bonsai-2-2B-Q2_0.gguf

llama-bench -m Ternary-Bonsai-2-2B-Q2_0.gguf -p 512 -n 128 -fa 1 -ngl 99
llama-server -m Ternary-Bonsai-2-2B-Q2_0.gguf -ngl 99 -fa on -c 8192 --jinja
#   request with "chat_template_kwargs": {"enable_thinking": false}
#   PrismML sampling, thinking off: temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5
```

The load log should include `loaded N Hadamard-folded weight(s)`. If output is
gibberish, the build is too old.

## 2. Build the patched llama.cpp (needed for LoRA)

```bash
./scripts/build_llama_macos.sh
```

This clones official llama.cpp **b11669** to `../llama.cpp`, applies
`patches/llama.cpp-b11669-hadamard-lora.patch`, builds a macOS-only
`llama.xcframework` and installs it in `Frameworks/`. The previous framework is
kept as `.pre-b11669`. `scripts/bootstrap.sh` calls it when the framework is
missing. The desktop app no longer takes iOS's framework.

Command-line tools from the same patched source, used for testing packs:

```bash
cd ../llama.cpp && cmake -B build-tools -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build-tools -j --target llama-server llama-cli llama-bench llama-perplexity llama-export-lora
```

## LoRA language packs

**Why official llama.cpp refuses them.** In `build_lora_mm`, the rotated input
`z = H·D·x` overwrites `cur`. The LoRA delta then reads `B·A·z` instead of
`B·A·x`. That is silently wrong, so the PR adds a load-time refusal
(`targets a prism.hadamard folded weight, which is not supported`).

**The patch.** It follows PrismML's fork: the base weight reads the rotated input
and the LoRA reads the original `cur`. It also lifts the refusal for everything
except `token_embd`, which is rotated the other way and doubles as the tied output
head. Two functions in `llama-graph.cpp` and one check in `llama-adapter.cpp`
change, 11 lines added and 8 removed. Packs train and convert like any other LoRA.
No rotation step is needed. The patch has not been sent upstream yet. Until it
is, re-apply it on each llama.cpp bump.

**Proof** (`scripts/lora_parity/run_parity.sh`): the check uses synthetic
adapters on all 150 rotated weights and llama.cpp's own KL divergence, 16×512
tokens.

| Check | Result |
|---|---|
| Official b11669 | refuses the adapter |
| zero adapter vs base | KLD 0.000000 |
| adapter vs base (sensitivity) | KLD 0.049, top token differs at 9% of positions |
| **merged `A·D·H` vs patched adapter** | **KLD 0.000001, 99.975% same top token** |
| merged unrotated vs patched adapter | KLD 0.090, the wrong-basis result |

The team's real Vietnamese pack also matches its merged reference (KLD 0.000001).

### Pack spec for training

- Train against the **frozen Bonsai 2 2B weights in their unrotated form**, as a
  normal PEFT LoRA.
- Targets: `q_proj k_proj v_proj o_proj in_proj_qkv in_proj_z out_proj gate_proj up_proj down_proj`.
  Never the embedding or LM head.
- **Rank 64.** Rank 32 hits a slow Metal path. Measured in ChatLM with the same pack
  zero-padded from r32 to r64 (identical output), 884-token prompt:

  | pack | prefill | time to first token | decode |
  |---|---|---|---|
  | none | 1137 tok/s | 0.91 s | 106 tok/s |
  | rank 32 | 552 tok/s | 1.86 s | 87 tok/s |
  | rank 64 | 958 tok/s | 1.07 s | 83 tok/s |

### Convert and check a pack

```bash
# PEFT -> GGUF (reads lora_alpha from adapter_config.json, or pass --alpha)
python scripts/peft_lora_to_gguf.py adapter_model.safetensors Ternary-Bonsai-2-2B-Q2_0.gguf \
  lora-bonsai2-2b-vietnamese.gguf --alpha 64            # add --pad-rank 64 for a rank-32 pack

# use it standalone (patched build)
../llama.cpp/build-tools/bin/llama-server -m Ternary-Bonsai-2-2B-Q2_0.gguf -ngl 99 -fa on --jinja \
  --lora lora-bonsai2-2b-vietnamese.gguf

# prove the runtime applies it exactly (needs numpy >= 2.2.6, tqdm, pyyaml, requests)
PYTHON=/path/to/python3.13 scripts/lora_parity/run_parity.sh Ternary-Bonsai-2-2B-Q2_0.gguf some-text.txt /tmp/parity
```

`peft_lora_to_gguf.py` doesn't need the base HF folder. For the 2B, the qwen35
converter's V-head reorder is the identity (16 key = 16 value heads), so the
output matches `convert_lora_to_gguf.py`. The script refuses a model where the
counts differ (check this for 4B/9B).

## 3. Run it in SLMChat

```bash
./scripts/bootstrap.sh        # framework, Bonsai and its pack, if ../demo/models-lab/bonsai2-2b has them
xcodegen generate
xcodebuild -project SLMChatMac.xcodeproj -scheme SLMChat -configuration Debug \
  -destination 'platform=macOS' -derivedDataPath build build \
  && open build/Build/Products/Debug/SLMChat.app
```

Settings (⌘,) → model **Bonsai 2 2B**, language **Vietnamese**. The pack row
should read "Language pack installed".

What the app does for Bonsai:

- **Desktop-only evaluation model.** It is listed in `DesktopModels.evaluationModels`
  only when its file is bundled, and never the default. Its packs are in
  `DesktopModels.evaluationAdapters`. `LoRACatalog.allAdapters` merges them into
  every lookup, so the catalog's own rows stay identical to iOS.
- **Language switching** is the existing mechanism: `ChatService` reloads with the
  pack for the language. Languages with no Bonsai pack fall back to the loaded
  one, so English chats currently run under the Vietnamese pack.
- **No prefix KV reuse** for `qwen3.5-0.8b` and `bonsai2-2b`
  (`ChatService.prefixReuseActive`). llama.cpp can't roll back their SSM state, so
  every turn prefills cold, and the suggest pre-warm is skipped.
- `ChatLM.mm`: the only llama.cpp API change was `use_mmap` → `load_mode`.

## Measured (M4 16 GB, Metal)

| | Bonsai 2 2B Q2_0 | Qwen3.5 0.8B Q8_0 (current default) |
|---|---|---|
| file | 521 MiB | 764 MiB |
| prefill pp512 | 994 tok/s | 2221 tok/s |
| decode tg128 | 105 tok/s | 80 tok/s |
| peak RSS, feature run | ~1.0 GB (1.4 GB with the r64 pack) | ~1.4–1.7 GB with packs |

Prefill is 2.2× slower because Bonsai has 2.5× the parameters (prefill is
compute-bound). Both models run at about 3.5 TFLOPS effective. Decode is faster
because it is bandwidth-bound and Bonsai's file is smaller. The new framework is
as fast as or faster than the old one for the Qwen models, with token-identical
output.

**Quality, 17 cases on the app's real prompts.**

- Bare Bonsai with format hints: every format correct, 11/11 event fields, working
  vi→en translation. Weak Vietnamese.
- With the team's Vietnamese pack (no hints): it learned the app's formats.
  Vietnamese is on par with Qwen3.5 0.8B plus its pack: better on some
  translations, worse on summary faithfulness and on the "fix diacritics" compose
  chip.
- Both fine-tunes share the same data bugs: vi→en returns the source untranslated,
  and "meeting moved to 10am" comes back as `no_event`.

## Open items

- Upstream the LoRA patch with the parity results, then drop `patches/`.
- Packs: retrain at rank 64, add an English pack, and fix the shared training-data
  bugs above.
- Not yet tested: switching between two Bonsai packs (only one exists) and the
  Settings flow by hand.
- Not committed: the Bonsai model entry and pack wiring (`DesktopModels`,
  `LoRACatalog.allAdapters`, `EngineUpgradeProbe`). They depend on the
  uncommitted `DesktopModels` work from `dev-agents`.
- Pre-existing, unrelated to this work: `TranslateStreamTests` fails on both
  frameworks (a half-written `[[LANG:` tag in partials).

## Files

| Path | What |
|---|---|
| `patches/llama.cpp-b11669-hadamard-lora.patch` | LoRA on rotated weights |
| `scripts/build_llama_macos.sh` | patched macOS `llama.xcframework` |
| `scripts/peft_lora_to_gguf.py` | PEFT → GGUF pack (`--folded`, `--pad-rank`) |
| `scripts/lora_parity/` | proof that the patch is exact |
| `scripts/bench_slm.py` | feature benchmark on the app's prompts |
| `GuardrailTests/EngineUpgradeProbe.swift` | Metal probe: models × packs, prefix reuse, rank speed |
| `GuardrailTests/PrefixReuseGateTests.swift` | no prefix reuse on qwen35 models |
