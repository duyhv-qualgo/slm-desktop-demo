# SLMChat for macOS

macOS (Apple Silicon) port of the iOS on-device SLM chat demo (`../demo`).
Every intelligent feature runs locally: the SLM through **llama.cpp / GGUF**
(Metal); semantic search through **LiteRT-LM** (EmbeddingGemma 2); the
guardrail and image tagging through the **TFLite C API** bundled in that same
LiteRT-LM library (as on iOS); language detection in plain Swift. The chat partner "Alex" replies through the remote API.

Read the iOS repo's [HANDOFF.md](docs/HANDOFF.md) for how the engine works and
which files are coupled to the trained weights. That applies unchanged here.

## Build & run

```bash
brew install xcodegen cmake             # once
./scripts/bootstrap.sh                  # clones models from ../demo, builds llama.xcframework, runs xcodegen
open SLMChatMac.xcodeproj               # scheme SLMChat, run on "My Mac"
```

`bootstrap.sh` takes assets from the iOS checkout (`SLMCHAT_IOS_ROOT`, default
`../demo`) using APFS clones, so they take no extra disk space and the iOS tree is
never written. It lists anything missing and exits non-zero.

`llama.xcframework` is built from source by `scripts/build_llama_xcframework.sh`:
llama.cpp at a pinned release (v0.7.0; the iOS app is still on b9721 from June) plus any
`scripts/llama-patches/*.patch`, arm64 only. Re-run it to rebuild; pass `ios` to
add iPhone + simulator slices, or `LLAMA_COMMIT=<sha>` to try another llama.cpp.

`SLMChatMac.xcodeproj` is generated; `project.yml` is the source of truth.

Tests (guardrail tokenizer/model parity, same fixtures as iOS):

```bash
xcodebuild test -project SLMChatMac.xcodeproj -scheme SLMChat -destination 'platform=macOS'
```

## What differs from iOS

Sources in `SLMChat/` and `GuardrailTests/` are a copy of the iOS app. Changes:

| File | Change |
|---|---|
| `SLMChatApp.swift` | single `Window` + `Settings` scene (⌘,) + ⌘N command; unloads models on quit |
| `View/ConversationListView.swift` | icon rail + sidebar (search, filters, Group/Direct) instead of a push stack; message search results |
| `View/Theme.swift`, `View/SuggestionTray.swift` | new — desktop look; reply suggestions as one scrolling row (1:1 only, never in the group) |
| `View/Platform.swift` | new — `PlatformImage`, `Clipboard`, toolbar placements, keyboard dismissal |
| other `View/*` | restyled; iOS-only modifiers routed through `Platform.swift` |
| `Engine/DesktopModels.swift` | new — ships **Qwen2.5 0.5B and Qwen3.5 0.8B only** (default Qwen3.5); Gemma-3 270M and Qwen3 0.6B and their packs are left out |
| `Engine/MessageEmbedder.swift`, `Store/MessageSearch.swift` | new — semantic text + photo search (EmbeddingGemma 2 via LiteRT-LM), index on device only |
| `Engine/ImageTagEngine.swift` | macOS decode via ImageIO (EXIF orientation baked in) |
| `Model/Conversation.swift` | suggestions withdrawn while anyone is typing (`isBotTyping`, `isDrafting`); `markSeen` for read-on-view |

Everything else under `Engine/`, `Model/`, `Store/` and the `.mm` wrappers is
unchanged. **Prompts, catalogs and generation params are owned by
the AI engineer** (HANDOFF §2): take changes from the iOS repo, don't edit here.
