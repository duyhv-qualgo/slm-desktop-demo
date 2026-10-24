# SLMChat — handoff to the mobile team

Written by the AI engineer who built this. It is a **working demo, not a
shippable app**. This document is the shortest path from `git clone` to
"I understand what I own."

Read in order: **§1** to get it running, **§2** to know what's yours, **§4** if
llama.cpp is new to you (it assumes no background), **§7** before anyone talks
about a release date. **§5** is the one most often misread — the LoRA rules
changed, and old summaries of them are still circulating. **§9** covers the
guardrail, which is no longer the thing §3 used to describe.

For anything performance-related — why a turn is slow, CPU vs GPU, device
ceilings, which experiments are already settled — see
**[PERFORMANCE.md](PERFORMANCE.md)**.

**On the `integration` branch** the engine was trimmed for handover: one KV slot
cleared per call (no prefix reuse), no benchmark tooling, four metrics fields,
`cancel`, and an `SLMMessage` protocol in place of the demo's `ChatMessage`.
**[integration.md](integration.md)** says which files to transfer and what is
still open. This document is updated to match that branch.

---

## 1. Day 1 — getting it to build

A fresh clone **will not build**. Only 53 files are tracked; every large binary
is gitignored. Three things have to arrive out-of-band (Drive, S3, a USB stick —
whatever your team uses for binaries):

| Missing | Size | Why it's not in git |
|---|---:|---|
| `Frameworks/llama.xcframework` | 234 MB | prebuilt llama.cpp + Metal |
| `Frameworks/CLiteRTLM.xcframework` | 116 MB | TFLite runtime for the two classifiers |
| `Resources/*.gguf` | ~1.2 GB | 2 base models + 9 language packs |
| `Resources/*.tflite`, `guardrail-tokenizer.json` | ~38 MB | guardrail + image tagging |
| `Resources/APIConfig.plist` | — | contains an API key; see §1.2 |

I will send you a single `slmchat-binaries.zip` that unpacks over the repo root.
After that:

```bash
brew install xcodegen        # once
./scripts/bootstrap.sh       # verifies binaries are present, runs xcodegen
SIM_NAME="iPhone 17" ./scripts/run_sim.sh
```

`bootstrap.sh` now **fails loudly** with the list of what's missing instead of
printing a warning and generating a project that won't link. If you keep the
binaries on a share, set `SLMCHAT_ASSETS=/path/to/unpacked` and it copies them
in for you.

### 1.1 Re-run xcodegen whenever you add a file

`SLMChat.xcodeproj` is **generated and gitignored**. `project.yml` is the truth.
Adding a Swift file or a resource means `xcodegen generate`. Editing build
settings in Xcode's UI silently loses them on the next generate — edit
`project.yml` instead.

One non-obvious constraint is already encoded there: `Resources/` must stay a
plain group, **not** `type: folder`. Folder references ship as opaque blue
folders that Xcode's signing step doesn't enumerate, which leaves `Info.plist`
outside the code signature and fails install with "Missing bundle ID".

### 1.2 First-run failure you will otherwise spend an afternoon on

The chat partner "Alex" replies **only** through the remote API. There used to be
an on-device fallback — a second `SLMEngine` loading `chat-qwen.gguf`, ~500 MB —
for when `Resources/APIConfig.plist` was absent. It has been removed: this project
always runs the API, that `.gguf` was never in the repo, so every branch reaching
that engine was dead code nobody could execute. Git history has it if you need it
back.

Consequences:

- **plist present** → friends reply via the API, and only the *task* model is
  on-device. This is the only supported configuration.
- **plist absent** → warm-up still SUCCEEDS and every on-device feature works
  (translate, suggest, summarize, event, guardrail, image tagging). Only the
  friend is mute: any attempt to make one reply shows
  `ChatService.missingAPIMessage`, which names the file to create.

Copy `APIConfig.example.plist` → `Resources/APIConfig.plist`, fill in your own
proxy + key, re-run xcodegen. I'll send credentials separately — they are not in
the zip and must never land in git.

### 1.3 Dead weight you can delete

`Frameworks/GemmaEngine.xcframework` is referenced by nothing — safe to delete.

The four `Resources/*.litertlm` files (~2.2 GB, leftovers from the previous
runtime — the app is 100% GGUF now) have already been moved to `legacy/`, which
is gitignored and not bundled. `legacy/` also holds the three stock Qwen `.gguf`
checkpoints that used to be listed in `ModelCatalog` and were greyed out on every
machine. Nothing there ships; delete the folder if you don't want the ~3.8 GB.

---

## 2. Ownership boundary

The single most useful thing for both of us is agreeing on this line, because
some of these files look like ordinary app code and are not.

### Mine (AI engineer) — send changes back to me, don't edit in place

| File | What it holds |
|---|---|
| `Engine/Prompts.swift` | every prompt template, the turn builder, output parsing |
| `Engine/LoRACatalog.swift` | which language packs exist, hashes, sizes |
| `Engine/ModelCatalog.swift` | which checkpoints are selectable |
| `Resources/*.gguf`, `*.tflite`, `guardrail-tokenizer.json` | the trained artifacts |
| generation params in `ChatService.swift` (`suggestTemp`, `translateTokens`, …) | tuned against the checkpoints |

These are **coupled to the trained weights**. A prompt reworded for readability
can change output quality with no compile error and no crash — the model just
gets worse, and you find out from a user. If a prompt produces bad output, file
it with the exact input and the `[Suggest]`/`[Translate]` log line and I'll
retrain or retune.

### Yours (mobile team) — go ahead

Everything in `View/`, `Store/`, `Model/`, persistence, image handling,
navigation, styling, accessibility, the download UX, error presentation,
threading outside the engine queues.

### Shared — ping me before changing

`ChatLM.mm`, `SLMEngine.swift`, `ChatService.swift` orchestration. Read §4 first,
then see §6 for the
specific lines that are load-bearing.

---

## 3. What the app actually does

A simulated 1:1 and group chat where every intelligent feature runs on-device.

| Feature | Engine | Model |
|---|---|---|
| Translate (native ↔ EN) | `taskEngine` | `model.gguf`, `qwen2.5-0.5b-v6.1.gguf`, `qwen3-0.6b-v6.gguf` or `qwen3.5-0.8b.gguf` + language pack |
| Auto-reply suggestions | `taskEngine` | same |
| Magic Compose (rewrite a draft / write a reply from a chip) | `taskEngine` | same |
| Summarize unread | `taskEngine` | same |
| Event detection | `taskEngine` | same |
| Safety guardrail | `guardEngine` | `guardrail-v16ls-<lang>.tflite` — eight per-language BERT-Tiny classifiers, plus per-region regex rules. No SLM stage (§9) |
| Image tagging / offline photo search | `imageTagEngine` | `image-tagging.tflite` |
| Chat partner "Alex" | `ChatAPI` (remote) | see §1.2 — API-only |

### Layers

```
View/                SwiftUI. Talks only to ConversationStore.
Store/               ConversationStore — @MainActor ObservableObject, all app state.
Engine/ChatService   Feature orchestration: which model, which prompt, which params.
Engine/SLMEngine     Async wrapper serializing one ChatLM on its own queue.
ChatLM.{h,mm}        Obj-C++ over llama.cpp's C API. KV cache, LoRA, sampling, cancel.
Frameworks/          Prebuilt llama.cpp + TFLite runtime.
```

`ChatViewModel.swift` **does not exist** — the old README claimed it did. State
lives in `Store/ConversationStore.swift`, feature logic in
`Engine/ChatService.swift`.

---

## 4. How inference works — read this before touching `ChatLM.mm`

No llama.cpp background assumed. This is the section to read if the engine looks
like a black box.

### 4.1 The one mental model that matters

**llama.cpp is not a chat SDK. It is a C library that predicts one token at a
time.** There is no `sendMessage()`. What it gives you is:

> given this list of integers, what integer comes next?

Everything that makes it feel like a chat — roles, system prompts, streaming,
stopping at the right place — is **our code in `ChatLM.mm`**. That's why the file
is 877 lines to expose four methods.

A *token* is roughly ¾ of a word ("unbelievable" → `un|bel|iev|able`). Text goes
in as text, becomes an array of `int32`, and comes back as text. The model only
ever sees integers.

### 4.2 The five llama.cpp objects

Everything in the C API hangs off these. iOS analogies are loose but useful:

| Object | What it is | Lifetime | Cost |
|---|---|---|---|
| `llama_model` | the weights, memory-mapped from the `.gguf` | one per loaded file | ~291 MB, **read-only, shareable** |
| `llama_context` | one *inference session* — the KV cache, compute buffers, position counters | one per model here | hundreds of MB, **mutable, not thread-safe** |
| `llama_vocab` | token ↔ text mapping, owned by the model | borrowed from model | free |
| `llama_batch` | scratch buffer of tokens to feed in one call | reused every call | sized by `n_batch` |
| `llama_sampler` | picks the next token from the model's output | rebuilt per generation | free |

**The distinction that causes bugs:** the *model* is immutable and cheap to
share; the *context* holds all mutable state and can serve exactly one
generation at a time. Every "why is the output garbage" problem in this codebase
traces to context state, never to the model.

We hold two independent `ChatLM` instances (`chat` and `task`), each with its own
model and context — see `ChatService.swift:14`.

### 4.3 The pipeline, in order

Load (`ChatLM.mm:302`), once per model:

1. **`llama_backend_init()`** — global, once per process, guarded by
   `dispatch_once` in `EnsureBackend()`.
2. **`llama_model_load_from_file`** — mmaps the weights. `mp.n_gpu_layers`
   decides CPU vs Metal; `mp.use_extra_bufts = false` is a RAM fix (§6).
3. **`llama_init_from_model`** — allocates the context: `n_ctx = 1024`,
   `n_seq_max = 1`, `n_ubatch = 128`.
4. **LoRA adapter**, if one was requested — `llama_adapter_lora_init` then
   `llama_set_adapters_lora` (§5).

Generate (`ChatLM.mm:633`, `runSystem:history:user:…`) — **every call runs all of
this**; the engine is logically stateless per call:

5. **Build the conversation** → `tokenizeSystem:history:user:` (`:473`).
   Roles and texts become `llama_chat_message[]`, then
   **`llama_chat_apply_template`** renders them using the template *embedded in
   the .gguf* — so Gemma's `<start_of_turn>` and Qwen's ChatML both work without
   us knowing which. Then `llama_tokenize` → `vector<llama_token>`.
6. **Clear the KV cache** → `llama_memory_clear`, see §4.5.
7. **Build the sampler chain** (`:716`). `temp <= 0` → greedy; otherwise
   `top_k(40)` → `top_p(0.95)` → `temp` → `dist`. (Greedy is the path that
   returns `nil` — see §6.)
8. **Prefill** — one `llama_decode` over the whole prompt at once.
9. **Decode loop** (`:738`), until `maxTokens` or end-of-generation:
   - `llama_sampler_sample(smpl, _ctx, -1)` → next token
   - `llama_vocab_is_eog(...)` → the model asked to stop
   - `llama_token_to_piece(...)` → that token's text, handed to `onChunk`
     (this is what makes streaming)
   - feed that one token back in and repeat
   - stop early if `-cancel` was called (checked every token, and during the
     prefill through llama.cpp's abort callback)
10. **Metrics** into `ChatLMTurnMetrics`: TTFT, decode tok/s, total, stop reason.

Step 9 is the whole trick: **the model's own output becomes its next input.**
That loop is the entire "AI" part.

### 4.4 Prefill vs decode — the one performance fact

Steps 8 and 9 have completely different cost profiles, and confusing them wastes
optimization effort:

| | Prefill (step 8) | Decode (step 9) |
|---|---|---|
| Processes | the whole prompt at once | exactly one token |
| Bound by | **compute** | **memory bandwidth** |
| Scales with | prompt length | reply length |
| Our share of a turn | **~73%** | ~27% |

**Prefill dominates.** A 480-token suggest prompt costs seconds; the 50-token
reply costs far less. So: to make a feature faster, **shorten the prompt** — do
not touch sampling params. This is also why the 3-second auto-suggest budget is
a prompt-engineering problem.

Read it off any turn in the log:

```
[SLMPerf] task/suggest qwen3.5-0.8b(gpu) | ttft 0.600s | decode 43.2 tok/s | TOTAL 1.572s | stop=eos | cap=160 temp=0.80
```

`ttft` is what the user waits before any text appears, and it is mostly prefill.
`stop=max_tokens` on a feature means its replies are being cut off.

### 4.5 The KV cache — one slot, cleared every call

While generating, the model caches an intermediate result (the "K/V") for every
token it has seen. That cache is what makes token #200 as cheap as token #2 —
without it every step would reprocess the whole prompt.

On the `integration` branch the context holds **one sequence of 1024 tokens**
(prompt + reply), and `runSystem:` **clears it at the start of every call**. No
prompt prefix is carried from one call to the next.

- **Cost:** every call pays a full prefill. A ~480-token suggest prompt that the
  old design mostly served from cache is now evaluated whole, so suggestions are
  slower than on `main`.
- **What it buys:** there are no reuse rules to get wrong. In particular, K/V
  computed under one LoRA adapter can never be read under another (§5), the KV
  cache is a third of its old size (1024 tokens instead of 3 × 1024), and the
  engine is about 600 lines shorter.
- **If latency needs it back:** the multi-slot, longest-prefix design is on
  `main` and `dev-q2-lora` (`pickSlotFor:` / `claimSlotFor:`, keyed by tokens
  *and* adapter). Bring the adapter key with it, or §5's silent corruption
  returns.

`SLMFeatureLabel` (the `label:` every Swift call passes) now only tags the
`[SLMPerf]` line. It no longer decides anything about caching.

### 4.6 The Swift stack above it

```
 SwiftUI View
      │  @Published, @MainActor
 ConversationStore            all app state, persistence, UI flags
      │
 ChatService                  which model + which prompt + which params
      │
 SLMEngine (chat) (task)      async/await facade; one serial queue each
      │  withCheckedContinuation
 ChatLM  (Obj-C++)            ← §4.3 lives here
      │
 llama.cpp C API
```

**`SLMEngine` (`Engine/SLMEngine.swift:107`) is the concurrency boundary and the
only reason this is safe.** A `llama_context` cannot serve two generations at
once. Each engine owns one private serial `DispatchQueue` and wraps every bridge
call in `withCheckedContinuation`, so from Swift you write:

```swift
let reply = await taskEngine.complete(system: sys, history: turns, user: text,
                                      maxTokens: 72, temperature: 0.5,
                                      label: .suggest)    // → String?
```

…and concurrent callers queue up instead of corrupting each other. **Never call
`ChatLM` directly from Swift.** It is not thread-safe and nothing will stop you.
The one exception is `SLMEngine.cancelCurrentGeneration()`, which deliberately
skips the queue: the queue is blocked inside the very call it cancels.

`ChatService` holds the per-feature params (`suggestTemp`, `translateTokens`, …)
and picks the prompt. `ConversationStore` is `@MainActor` and owns everything the
UI observes.

### 4.7 Adding a feature — the actual recipe

1. Add a prompt builder + output parser to `Engine/Prompts.swift`. **Ping me** —
   this is the file coupled to the trained weights (§2). Builders take
   `[any SLMMessage]`, so they work with the host app's own message type.
2. Add token/temp constants to `ChatService.swift` next to the existing ones.
3. Add a case to `SLMFeatureLabel` and pass it as `label:` — it tags your
   `[SLMPerf]` lines.
4. Expose it through `ConversationStore`, not straight to the view.
5. Check the `[SLMPerf]` line for your feature against the 3 s budget.

### 4.8 Glossary

- **Token** — ~¾ of a word; the integer unit the model consumes and emits.
- **GGUF** — llama.cpp's model file format. Self-contained: weights, vocab,
  *and* the chat template.
- **Quantization** — storing weights at reduced precision. Our base is `Q8_0`
  (8-bit): ~291 MB instead of ~1 GB, with a small quality cost.
- **Context window** — how many tokens fit at once. 1024 per call here, prompt
  and reply together (`ChatLM.contextTokens()`).
- **Prefill / decode** — §4.4.
- **KV cache** — §4.5.
- **EOS/EOG** — the token by which the model says "I'm done".
- **Temperature** — randomness. Low (0.2) for translation, higher (0.8) for
  suggestion variety. **Never exactly 0** (§6).
- **top_k / top_p** — restrict sampling to the k most likely tokens / the
  smallest set summing to probability p. Fixed at 40 / 0.95.
- **LoRA adapter** — a small file of weight *deltas* applied on top of the base
  model. §5.

---

## 5. The LoRA language-pack design

This is the part most likely to be misread, so here it is explicitly — and note
that **the central rule changed**. If you find a summary saying adapters are
"applied at load and never swapped", it is out of date.

**The design:** one LoRA adapter per (base model, language), shipped as a
separate 15–43 MB file from the 291–774 MB base. An adapter is bound to the
exact checkpoint it was trained against — the same `.gguf` weights, not merely
the same architecture — so it always resolves by **(base model, language)**,
never by language alone. Applying a Qwen3 adapter to Gemma-3 does not fail
loudly; it produces fluent nonsense. The sets are not interchangeable even
within the family: the Qwen3.5 packs target hybrid `attn_qkv` / `ssm_*` blocks
that the Qwen3 packs have never heard of.

**Adapters are a per-CALL choice.**

- The engine loads with a *default* adapter: the pack for (task model, native
  language). Every generation uses it unless it names another.
- `generate…adapterPath:` overrides it for one call. **Auto-suggest is the only
  caller that does**, and deliberately: a suggestion is text the user is about to
  *send*, so it must be in the conversation's language and written by a model
  adapted to it. Translate, summarize and event extraction produce text the user
  *reads*, so they stay on the native pack.
- Changing the **native** language still reloads the model (7–8 s cold), because
  that changes the default.

**Why swapping used to be forbidden — and what makes it safe now.**
`llama_set_adapters_lora` operates on the *context*, not on a sequence. Applying
an adapter does **not** clear the KV cache, and llama.cpp will not warn you.
Every cached slot would then hold K/V computed under the old weights, and the
model produces fluent, confident, *wrong* output. No error, no crash, no log
line.

What makes it safe now is that **the KV cache is cleared at the start of every
generation** (§4.5), so no K/V outlives the adapter it was computed under. If
someone adds prefix reuse back for speed, the cache key must include the adapter
(as the design on `main` does). Otherwise the silent corruption comes straight
back, and no test that checks for exceptions will catch it.

**What a switch actually costs.** Measured on the booted simulator, CPU backend,
from the app's own `[SLMPerf]` lines (~70 samples):

| | when | cost |
|---|---|---|
| cold init (`llama_adapter_lora_init`) | first touch of a pack, per app session | **50–110 ms**, ≈2 ms per MB; worst seen 245 ms |
| warm switch | every later switch to that pack | **0.000 s** — rewrites a map on the context |
| same adapter | the common case | no-op, llama.cpp returns early |

The parse is the small half *and* the free half. `prewarmAdapter` moves it off
the budgeted call (a conversation prewarms during the 1.2 s auto-suggest
debounce), and the log tags `prewarm` vs `in-call` so the two are never averaged
together. A failed init is negative-cached, so a broken pack is not re-read tens
of MB on every call.

Two traps in reading those numbers:

- **The adapter time is not inside `TOTAL`.** In a cold-switch line, prefill +
  decode + overhead accounts for all of `TOTAL`, and the adapter work surfaces
  only as `callback +0.106s` and in the `TURN` line. Judge a cold switch by
  `TURN`, or you will under-count exactly the case that threatens the budget.

**Why a separate file and not merged weights:** merging means every fine-tune
iteration is a full 291–774 MB re-download plus ~1.2 GB transient disk, because
quantized GGUFs have no useful binary delta. As an adapter it is 15–43 MB —
about 5% of its base at every size. Per-token overhead is quoted at ~6%:
arithmetic from parameter counts, **not** measured on this app (§8).

**What the adapter is trained for:** general in-language chat style and
register. Task behaviour (translate/summarize/suggest) stays in the
`Prompts.swift` templates, not in the adapter — so adding a language does not
require retraining task behaviour. The exception is the Gemma-3 pack, which is
task-trained: retuning a prompt template is a reason to retrain *that* pack.

**An uninstalled pack blocks; it does not fall back.**
`ChatService.suggestAdapter` returns three outcomes, and it is deliberately not a
`String?` — nil would collapse two of them, and `ChatLM` turns every nil into the
load-time default:

- `.pack(path)` — installed, generate under it.
- `.loaded` — the conversation is in the native language, or no pack for this
  pair was ever published. Generate under whatever the engine loaded; there is
  nothing for the user to install.
- `.missing(pack)` — published but **not installed. Not a generation input.**
  The caller stops and surfaces it, because generating under the native pack is
  the wrong-language mush this whole design exists to prevent. Both the auto and
  the manual suggest paths stop. The chat shows a notice with an inline **Get**
  button, because Settings only offers a Get for the *native* language's pack —
  without it a Vietnamese-native user has no route to the Spanish pack at all.

Flow for a native-language change: `SettingsView` →
`ConversationStore.setNativeLanguage` → `LoRAStore.download` →
`ConversationStore.applyAdapterChange` →
`ChatService.reloadTaskEngineForAdapterChange` → `ChatLM` load with `loraPath:`.

### 5.1 Language detection — the step LoRA makes necessary

A plain multilingual model needs no language decision: it continues in the
language of its context. Splitting language into swappable weights forces that
decision **before the first token**, which is what detection is for. The
guardrail needs it for the same reason (eight per-language classifiers), and both
run the same code.

`Prompts.identify` is one pass over **`LangDetect`** — MediaPipe's Language
Detector (PRADO/SGNN, 111 BCP-47 labels), extracted from its `.tflite` and ported
to Swift in `SLMChat/Engine/LangDetect.swift`. Weights are a 303 KB blob in
`Resources/`; there is no interpreter and no TFLite dependency, because the graph
is three matmuls over a hashed-character-n-gram embedding and reimplementing it
was smaller than hosting it.

The ranked hypotheses are walked **unfiltered**: narrowing the model's output to
`TranslateLanguage` would renormalize probability mass over the allowed set, so an
Italian message would be forced onto whichever of ours it least dislikes.
Unfiltered it answers "Italian", which maps to nothing and abstains. The first
hypothesis ≥ **0.40** that maps to a `TranslateLanguage` wins; the floor is below
half on purpose — `"До завтра"` scores Ukrainian 0.465 / Russian 0.429, and
Ukrainian has no pack, so the loop falls through to Russian. Otherwise nil:
abstain, and let the caller widen the text.

**This replaced `NLLanguageRecognizer` + `vietnameseRatio` in Sep 2026.** The
old second stage was a VI-vs-EN word heuristic that ran whenever Apple's model
declined to name a language — and `viDiacritics` contains the plain acute and
grave vowels, so it claimed every accented Latin-script language the recognizer
had abstained on. `"Olá, vejo-te amanhã no escritório"` scored 0.75 Vietnamese
and was screened by the **Vietnamese** guardrail pack. Demoting the heuristic
below the recognizer, which an earlier fix did, had only moved that failure to the
languages the recognizer stays quiet about.

Measured over 1,204 chat-length cases:

| | old (Apple + ratio) | new (`LangDetect`) |
|---|---|---|
| real chat messages, right | 98.3% | **99.6%** |
| Latin-script, no pack — correctly abstained | 86.4% | **100%** |
| 1–8 word fragments, right | **93.6%** | 89.2% |
| wrong-pack routings, total | 14 | **6** |
| ms per message | 0.56 | **0.029** |

The fragment row is the real cost and it is the right trade: the smaller model
spreads probability across close neighbours (`"Mein Laptop"` → nl 0.45 / de 0.32)
and drops below the floor, so it **abstains** where the old path **asserted**. An
abstention widens to the thread window and then to native — a designed path. A
confident wrong answer routes the message to a classifier that never saw the
language and nothing logs.

`LangDetectConformanceTests` pins the port against reference vectors captured
from the NumPy implementation, with an **exact integer** assertion on the n-gram
ids. That is the load-bearing test: the arithmetic is easy, but the preprocessing
depends on Plan9 Unicode tables, on iterating unicode scalars rather than
Characters, and on faithfully reproducing a buffer over-read in MediaPipe when
lowercasing shortens the text. Each is a plausible cleanup that silently changes
the answer. Regenerate fixtures with `scripts/build_langdetect_assets.py`, which
also pins the source `.tflite`'s SHA-256.

The material that script reads — Google's `.tflite`, the extracted weights, the
NumPy reference, the Plan9 `runetypebody.h` and MediaPipe's C++ sources — lives
in `language-detection/` and is **gitignored**, like every other model asset
here: nothing at build or run time touches it, so it ships via `$SLMCHAT_ASSETS`
rather than in the tree. Fetch it before regenerating anything, and before
editing `LangDetectFixtures.swift` — those vectors are ground truth captured
from the reference, and rebuilding them from the Swift port would turn the
conformance test into a tautology.

Callers cascade rather than accepting a weak answer, because chat turns are too
short to identify ("ok" tops out at Polish 0.29):

- **Suggestions** — the latest *incoming* message → the last 8 messages joined →
  the user's native language.
- **Guardrail** — the message being screened → the conversation's language.

The end of the cascade is never nil. A misdetection costs the wrong pack and
nothing else: the guardrail's regex rules run every set the region ships
regardless, precisely so coverage never depends on this.

`Prompts.suggestReplies` takes `language` as a **parameter** rather than
computing it, so the prompt's target language and the adapter come from the same
resolved value and cannot disagree. Keep it that way.

---

## 6. Load-bearing lines — do not "clean up"

Each of these looks arbitrary and fails silently or expensively if changed.

- **`ChatLM.mm:476` — `cp.n_ubatch = 128`.** The scheduler reserves a worst-case
  graph sized by `n_ubatch`, dominated by an `[n_ubatch × n_vocab]` f32 logits
  tensor. Gemma-3's vocab is 262k, so the default 512 reserved **~516 MiB**.
  128 brings it to ~130 MiB.
- **`ChatLM.mm:394` — `mp.use_extra_bufts = false`.** Repacking doubled RAM
  (~800 MB peak). Together with the line above this is the difference between
  running and being jetsammed on an older device.
- **`kCtx = 1024`.** Prompt plus reply for one call. `ChatService` derives its
  chunk and output budgets from `ChatLM.contextTokens()`, so change it there,
  never as a second literal.
- **`llama_memory_clear` at the start of `runSystem:`.** The only thing making
  per-call adapters safe. Remove it and a prefix computed under one language
  gets reused under another: fluent, confident, wrong, and completely silent
  (§5).
- **Temperature ≥ 0.1 everywhere.** At exactly 0 the greedy sampler path returns
  `nil` instantly. No error — just an empty result.
- **`GuardrailTests/` fixtures.** These are reference outputs from the *original*
  Hugging Face tokenizer and BERT-Tiny model. A Swift tokenizer tweak that
  drifts from them misclassifies text **without failing**. They run on Cmd+U.
  Keep them green; they're the only thing catching silent safety drift.
- **`SafetyLabel.modelClassCount = 8`, and `fromModelIndex`'s table.**
  `GuardEngine` identifies the class head *by width*. A stale count matches no
  head and every message comes back Clean; a raw `SafetyLabel(rawValue:)` in
  place of the table relabels every class from `child` upward. Neither throws
  (§9).
- **`Prompts.stripListMarker`.** Anchored regex, deliberately not
  `trimmingCharacters`. The old version trimmed from both ends with a set
  containing every digit and `.`, so `"see you at 3:30."` became
  `"see you at 3:"`.
- **`LoRAStore.reportProgress`'s `guard case .downloading`.** Progress ticks
  arrive from a background hop and the final one can land *after* `.installed`,
  reverting the row to `downloading(1.0)` — stuck at 100% forever.

---

## 7. Demo scaffolding — must not ship

The download experience is **faked**. This is the item most likely to be
mistaken for finished work.

- **`LoRACatalog.remoteBaseURL = nil`.** There is no CDN. Real downloads are
  written and hash-verified (`fetchAndInstall`) but have never run against a
  server. Someone needs to host the packs and set this.
- **`simulatedDownload: true` on every pack except the four Vietnamese ones.**
  It fakes a progress animation and then copies the file out of the app bundle.
  Each model's Vietnamese pack stays a plain bundled pack so the default language
  works on first launch and offline. Note `LoRACatalog.installedPath`
  deliberately refuses the bundled copy of a simulated pack until an "install"
  has run — that stand-in is playing the part of a server, not of an installed
  file.
- **The two Gemma-3 packs are the same file.** `lora-gemma3-270m-vietnamese.gguf`
  and `-english.gguf` are byte-identical copies of `multitask-6tasks-v5.gguf`
  (same SHA-256 in the catalog), so on Gemma-3 the two languages behave
  identically. Only the *plumbing* is demonstrated there. The three Qwen sets are
  genuinely distinct — 21 packs, different checksums, one training run each.
- **Language support is per (model, language).** `isSupported(forBaseModel:)`
  treats Vietnamese and English as the baseline every checkpoint handles and
  derives everything else from `LoRACatalog`. So Arabic … Chinese are live on all
  three Qwen checkpoints and greyed out on Gemma-3, and switching model can drop
  the language in use (`ConversationStore.demoteLanguageIfUnsupported`).
- **A German Qwen3.5 pack exists and is not in the catalog.** Its comment says a
  row can't compile without a `.german` case, an `NLLanguage` mapping and German
  rule routing — **all three now exist**, added for the guardrail. Only the
  `LoRAAdapter` row is missing, so the file sits unused in
  `qwen3.5-multilingual/lora/`.
- ~~**Only Vietnamese and English are prompt-detectable.**~~ **Fixed.** This used
  to be a VI-vs-EN word heuristic that answered "English" for Spanish, and the
  note here said fixing it meant a real language-ID step rather than another word
  list. That step is now in: `LangDetect` runs first and the word
  heuristic is only the fallback beneath it (§5.1). All seven languages resolve.
- **~3 GB of models bundled in the app.** `Resources/` holds four checkpoints
  (291–774 MB each) plus all 23 LoRA packs plus the guardrail and image-tagging
  models. Over-the-air install limits make this a non-starter for the App Store
  several times over. Needs on-demand resources or a first-run download, and it
  is the single largest gap between this demo and a release.
- **`ModelCatalog` lists 4 models**, each bundled with language packs trained
  against it. The stock Qwen rows that rendered greyed out on every machine were
  removed; their weights are in `legacy/`. Every listed row earns its place by
  having packs.
- **Legacy guardrail files are still in `Resources/`.** `guardrail-bert-tiny-*.tflite`
  and `guardrail-tokenizer-*.json` are the v15 assets; nothing loads them any
  more (`GuardPackCatalog` names `guardrail-v16ls-*`). Safe to delete, and worth
  it — they are the files someone will read as current.
- **`bundleIdPrefix: com.example.slmchat`** and an empty `DEVELOPMENT_TEAM`.
- **Device deploy is unverified.** Simulator only so far. Metal is slower than
  CPU below A13 and needs A14 for fast prefill — an iPhone XS should stay on
  CPU. The backend picker in Settings already falls back, and the logs record
  `backend=cpu (asked gpu)` when it does.

### Verified since, and what is still open

The tap-Get → progress → installed → reload flow has now been exercised on the
simulator, and generation with a per-conversation adapter applied has been
watched running: the Spanish and French Qwen3.5 packs install, load
(`lora init … → ok`), switch warm, and produce chips. What that run also showed
is in §5 — the measured switch costs, and the `TURN`-vs-`TOTAL` reporting gap.

Still unverified:

- **Device deploy.** Simulator only. Every number in this document is a simulator
  number.
- **Real downloads.** `fetchAndInstall` is written and hash-verified but has never
  run against a server.
- **Generation *quality*** per language, as opposed to plumbing. Nobody has read
  the Arabic, Russian or Chinese output and judged it.

---

## 8. Working with me

**Reporting a quality problem.** I need the exact input, the full log line, and
which language pack was active. The logs are already tagged — filter on
`[SLMPerf]`, `[LoRA]`, `[Suggest]`, `[ChatLM]`. A `[SLMPerf] load` line tells you
model, backend and adapter in one:

```
[SLMPerf] load task/model backend=cpu (asked gpu) lora=lora-...-vietnamese.gguf → ok in 7.413s
```

`lora=none` means bare weights; `lora=FAILED(x)` means the adapter was requested
and rejected, and the model is running unadapted. The app degrades to bare
weights rather than refusing to start — deliberate, but it means a broken
adapter looks like a quality regression, not an error.

**Performance budget.** Auto-suggest has a 3-second per-turn target. Prefill,
not decode, is ~73% of a turn — optimize prompt length before sampling params.

**Benchmarking was removed on the `integration` branch** — `BenchmarkView`,
`BenchmarkHarness` and the throughput probes. The tooling and the measurements
it produced are on `main` (see PERFORMANCE.md). On this branch, read the
`[SLMPerf]` line: TTFT, decode tok/s, total and stop reason per call.
`releaseSLMs()` / `restoreSLMs()` remain for quitting and switching backend;
**the app's SLM features are dead between those two calls**, so they must stay
paired.

Still unmeasured: the *per-token* cost of an adapter (the ~6% figure is
arithmetic) and a format-adherence rate (what % of outputs parse).

**Adding a language** is on me, not you: train the pack, quantize to GGUF, send
you the file plus a `LoRAAdapter` entry with its real SHA-256. Settings picks it
up from there — `isSupported(forBaseModel:)` reads the catalog, so there's no
second switch to flip. A `case` in `TranslateLanguage` is needed only for a
language the enum doesn't list yet, and adding one is not just a catalog change:
it also wants an `NLLanguage` mapping (§5.1), a guardrail pack
(`GuardPackCatalog`) and, if a region should act on it, rule routing
(`GuardRegion.ruleLanguages`). German is the worked example — all of that landed
for the guardrail, and only the LoRA row is outstanding (§7).

**Reporting a wrong-language suggestion.** Send the `[LoRA] suggest lang=…` line
with it. It distinguishes the four outcomes that are otherwise indistinguishable
downstream: resolved to a pack, ran under the loaded one because the conversation
is native, ran under the loaded one because no pack was ever published, or
BLOCKED because the pack isn't installed. The first tells me it's a training
problem; the others tell me it's a catalog, detection or install problem.

---

## 9. The text guardrail

It stopped being "one BERT-Tiny model" and became two detectors behind one
verdict. `Engine/Guard*.swift` owns all of it; nothing here touches the SLM.

**Layer 1, regex rules.** Runs first, and a hit short-circuits — the tokenizer
and the interpreter never execute. Rules are keyed by **region** (jurisdiction is
policy) and, within a region, by the language they were *authored in*. Every set
the region ships is run, not just the one matching the detected language: a
pattern only matches text in its own language anyway, so the language key
organises authoring, not dispatch. Making it dispatch would make rule coverage
depend on language detection, which abstains on exactly the short messages rules
are best at. Rules can only ever *raise* a verdict.

**Layer 2, the classifier.** Eight per-language BERT-Tiny `.tflite` packs
(`guardrail-v16ls-<lang>`), each with the tokenizer it was trained against, loaded
lazily and held eight-resident at ~3 MB each. The v16 decision rule:

```
p_unsafe = 1 − softmax(logits)[clean]     ← total unsafe mass, not the winning class
flag when p_unsafe >= 0.95                ← swept against the labelled demo lines
class = argmax(logits)
```

Long text is screened in windows — the whole message *plus* one window per
sentence, capped at 16 — because the model judges a window by its dominant
content, so a threat packed behind one benign sentence comes back Clean.

**Three things that will surprise you:**

1. **v16 has no severity head.** A model verdict carries `severity: nil`, and
   since blurring is `severity == .critical`, **only hand-authored rule hits blur
   anything**. The model still labels the bubble and makes the message
   reportable. This is deliberate — reconstructing severity from the class would
   put a guess where a measurement used to be — and it cuts both ways (see
   `ChatMessage.isSensitive`).
2. **`political` and `hate` are rules-only.** They left the model's taxonomy at
   v16 and did not leave the product. Whether political speech is actionable is a
   jurisdiction question, which is also why `GuardRegion.actionableClasses`
   masks it outside Vietnam — for the *model* as well as the rules, since a
   classifier has no notion of jurisdiction.
3. **"Clean" and "not screened" are different findings.** `SafetySource`
   distinguishes them. The guardrail fails open, so a missing pack or a failed
   load produces a clean-looking verdict; counting that as "screened and found
   nothing" is how a coverage gap stays invisible.
4. **The current checkpoint has false positives the previous one did not, and
   the threshold cannot remove them.** `v16ls` is a label-smoothed retrain of
   `v16` (same data, config, architecture). Smoothing at 0.1 caps the achievable
   `p_unsafe` near 0.9875 and compresses everything below it, so the scores are
   no longer separable the way the old ones were. Swept over the 285 labelled
   demo lines (`scripts/sweep_guard_threshold.py`), the old checkpoint had
   **zero** false positives anywhere in 0.75–0.95; this one has **four at 0.95**,
   its best point. Two are the ones to know about: the Spanish victim reply
   "Para ya." (*stop it*) scores 0.9837 `suicide`, up from 0.712, and a
   Vietnamese first-person distress line scores 0.9113 — the exact shape v16 was
   brought in to stop flagging. Against that it **recovers 7–8 genuinely unsafe
   lines** the old pack could not reach at any threshold.

   The other cost of 0.95 is on the recall side: **"I will kill you" scores
   0.8399 and matches no regex pattern**, so an explicit death threat is now
   missed end-to-end in English. Catching it needs the gate at ≤0.83, which buys
   back the distress false positive. There is no setting that avoids both; if
   this trade is judged wrong, roll the checkpoint back to `2026-08-26` rather
   than moving the threshold, and re-run the sweep either way.

Screening is scheduled by `MessageScreener`, not one task per message: a single
drain loop over a pending set, always taking the row nearest the viewport, ±10
either side, shedding past 48. A shed message is forgotten so scrolling back
re-requests it.
