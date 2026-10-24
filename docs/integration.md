# Integration handover — on-device SLM + guardrail

For the mobile (iOS) and desktop (macOS) teams integrating the on-device AI
features into the production chat app. Branch: **`integration`**.

The demo app (SLMChat) is **not** the thing to transfer. Transfer the engine
files listed in §2, wire them to your own app (§3), and work through the open
items (§4–§5). [HANDOFF.md](HANDOFF.md) explains how the engine works inside;
[PERFORMANCE.md](PERFORMANCE.md) has the device measurements.

Test status on the `integration` branch (macOS, Apple Silicon): **243 tests,
0 failures, 2 skipped.** Nothing has been built or run on iOS yet (§5).

---

## 1. What changed for integration

Compared with `main`, the `integration` branch:

- **KV cache:** one 1024-token slot, cleared at the start of every call. No prompt
  prefix is reused between calls. Simpler and safe with per-call LoRA adapters;
  every call pays a full prefill, so suggestions are slower than on `main`.
- **Benchmark tooling removed:** `BenchmarkHarness`, `BenchmarkView`, the
  benchmark load path, and the throughput/model-swap/quant probes.
- **`ChatLM` API trimmed:** one `load(modelPath:backend:loraPath:)`; no
  `cacheDir`, `batchThreads`, `cacheable:` or `prefill()`.
- **Metrics cut to four content-free fields:** `timeToFirstTokenSeconds`,
  `decodeTokensPerSecond`, `totalSeconds`, `stopReason`.
- **Cancel added:** `SLMEngine.cancelCurrentGeneration()` stops a running
  generation from any thread (between tokens, or mid-prefill).
- **Privacy:** logs that carry chat content go through `contentLog` /
  `contentPrint`, which do nothing in release builds.
- **Semantic search (EmbeddingGemma) removed.** Message search is exact text only.
- **`SLMMessage` protocol:** `Prompts`, `MessageScreener` and `GuardReport` take
  any conforming message type instead of the demo's `ChatMessage`.
- **Translate streaming fixed:** the `[[LANG:…]]` preamble no longer flashes on
  screen and then disappears.

---

## 2. What to transfer

### SLM (llama.cpp)

| File | Role |
|---|---|
| `SLMChat/ChatLM.h`, `SLMChat/ChatLM.mm` | Obj-C++ engine over llama.cpp: load, `stream` / `generate`, `cancel`, LoRA, metrics |
| `SLMChat/Engine/SLMEngine.swift` | Async Swift wrapper; one serial queue per engine; `[SLMPerf]` log line |
| `SLMChat/Engine/Prompts.swift` | Prompt builders and output parsers for suggest, translate, summarize, compose, event; `TranslateLanguage` |
| `SLMChat/Engine/SLMMessage.swift` | The protocol your message type conforms to |
| `SLMChat/Engine/ContentLog.swift` | Debug-only logging for lines with chat content |
| `SLMChat/Engine/ModelCatalog.swift`, `SLMChat/Engine/DesktopModels.swift` | Which checkpoints exist; trim to the one you ship |
| `SLMChat/Engine/LoRACatalog.swift`, `SLMChat/Engine/LoRAStore.swift` | Language packs (only if you ship LoRA packs) |
| `SLMChat/Model/DetectedEvent.swift` | Parser for the event-extraction output |
| `SLMChat/ResourceLookup.swift` | Finds bundled model files |

### Guardrail (TFLite via LiteRT-LM)

| File | Role |
|---|---|
| `SLMChat/GuardLM.h`, `SLMChat/GuardLM.mm`, `SLMChat/TFLiteC.h` | Obj-C++ classifier over the TFLite C API |
| `SLMChat/Engine/GuardEngine.swift` | Classify a message: rules + per-language model, on its own queue |
| `SLMChat/Engine/GuardPackCatalog.swift`, `GuardRules.swift`, `GuardRegion.swift`, `GuardTokenizer.swift` | Language packs, regional rule sets, tokenizer |
| `SLMChat/Model/SafetyLabel.swift` | `SafetyLabel`, `Severity`, `SafetySource`, `SafetyResult` |
| `SLMChat/Engine/LangDetect.swift` | Language identification (picks the guardrail pack and reply language) |
| `SLMChat/Engine/MessageScreener.swift` | Screens what's on screen first; sheds backlog on a fast scroll |
| `SLMChat/Engine/GuardReport.swift` | User report payload — never contains message text |

### Resources and build

| Item | Notes |
|---|---|
| One `.gguf` base model + its `lora-<model>-<lang>.gguf` packs | Decide which model first (§4) |
| `guardrail-v16ls-<lang>.tflite`, `guardrail-v16-tokenizer-<lang>.json` | 8 languages |
| `guardrail-rules-*.json` | One per region plus `universal` |
| `langdetect-weights.bin`, `langdetect-weights.json`, `langdetect-labels.txt` | Language detection |
| `scripts/build_llama_xcframework.sh` | Builds llama.cpp; `macos ios` adds the iPhone and simulator slices |
| `SLMChat/SLMChat-Bridging-Header.h` | Exposes `ChatLM` / `GuardLM` to Swift |
| LiteRT-LM Swift package (0.18.0) | Provides the TFLite runtime the guardrail links against. Bumping it changes the guardrail: re-run `GuardrailTests` |
| `GuardrailTests/` (engine, guardrail, prompt tests) | Bring the relevant ones; the guardrail fixtures catch silent tokenizer drift |

---

## 3. Wiring it into your app

1. **Conform your message type.**

   ```swift
   extension Message: SLMMessage {
       var isFromUser: Bool { sender == .me }
       // id, text, authorName, safety, severity — map from your fields
   }
   ```

2. **Own one `SLMEngine` and one `GuardEngine`** for the app's lifetime. Load
   once: `await slm.warmUp(modelName:adapterPath:)`,
   `await guardEngine.warmUp(language:region:)`.
3. **Generate through `SLMEngine` only** (`complete` / `stream`), never `ChatLM`
   directly — it is not thread-safe. Build prompts with `Prompts.*`, parse with
   the matching `Prompts.parse*`.
4. **Screen incoming messages** with a `MessageScreener<Message>`: call
   `screenNow` for a message that just arrived, `viewportDidShow(index:in:)` as
   rows appear.
5. **Write your own small service** over the two engines. Do not port
   `ChatService` (see §6).
6. **Call `cancelCurrentGeneration()`** when its result is no longer wanted:
   Stop button, chat closed, or a newer request replacing an older one.
7. **Watch `[SLMPerf]`** in debug logs: `ttft`, `decode tok/s`, `TOTAL`, `stop`.

---

## 4. Checklist — decisions and backend work

- [ ] **Choose the shipping model:** Qwen3.5 0.8B (current default) or Qwen2.5
      0.5B. Bundle only that model and its LoRA packs.
- [ ] **LoRA pack delivery:** downloading is off (`LoRACatalog.remoteBaseURL =
      nil`), so packs are side-load only. Decide bundling vs hosting.
- [ ] **Report submission backend:** `GuardReport.submit` only logs today. A real
      backend needs message franking (proof the reported message was really
      sent), a content-disclosure design and rate limiting — see the comments in
      `GuardReport.swift`.
- [ ] **Cancel UX:** decide where cancel is called (Stop button, chat closed,
      superseded request). The engine side is done and tested.
- [ ] **Guardrail regions and languages:** confirm which regions' rule sets and
      which of the 8 language packs ship.
- [ ] **Accept the latency trade-off** of the cleared KV cache, or plan to bring
      back prefix reuse from `main` (keyed by tokens *and* adapter — HANDOFF §4.5).

## 5. Checklist — work before shipping

- [ ] **iOS build:** run `scripts/build_llama_xcframework.sh macos ios`
      (`Frameworks/llama.xcframework` has only `macos-arm64` today).
- [ ] **Run on a real iPhone:** engine + guardrail tests, then measure memory,
      TTFT and decode speed. Nothing has run on iOS yet — this is the biggest
      open risk. Tune context size and GPU use for phones; GPU is declined below
      A13. (Bonsai 2 was evaluated as macOS-only.)
- [ ] **Add a lock around `SLMEngine.bridge`:** `tokenCount`, `activeBackend` and
      `cancelCurrentGeneration` read it off the engine queue. Low risk, but a
      reload at the same moment is a data race.
- [ ] **Release build settings:** App Sandbox and Hardened Runtime are off in the
      demo. Never bundle `APIConfig.plist`.
- [ ] **Bundle only what you ship.** The demo bundles all of `Resources/`
      (3.4 GB, including leftover embedding files).
- [ ] **Encrypt any on-device derived data at rest** (summaries, verdicts) with
      a device-only key, per the E2EE design.

---

## 6. Do not transfer (demo only)

| Part | Why |
|---|---|
| `Engine/ChatService.swift` | Demo controller holding everything (SLM, guardrail, image tags, cloud chat). Write your own |
| `Engine/ChatAPI.swift` | Sends chat history to the Gemini cloud API for the demo friend's replies. Breaks E2EE |
| `Model/GuardrailDemo.swift` | Scripted conversations for demoing the guardrail |
| `View/*`, `Model/Conversation.swift`, `Model/GroupChat.swift`, `Store/*` | The demo app's UI, state and persistence |
| `ImageTagLM`, `ImageTagEngine`, image search | Out of the SLM + guardrail scope |
| `Model/ChatMessage.swift` | Use your own message type via `SLMMessage` |

---

## 7. Privacy rules the code already follows

- All inference is on device. Nothing derived from chat content is sent anywhere.
- Logs with chat content are Debug-only (`contentLog` / `contentPrint`). Keep it
  that way when adding features.
- Guardrail verdicts stay on the device. Only a report a **person** chooses to
  send leaves it, and the report carries no message text.
- `ChatLMTurnMetrics` holds timings and a stop reason only.
