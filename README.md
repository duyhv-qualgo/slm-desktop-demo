# SLMChat for macOS

macOS (Apple Silicon) port of the iOS on-device SLM chat demo (`../demo`).
Every intelligent feature runs locally: the SLM through **llama.cpp / GGUF**
(Metal), and the guardrail, language detection and image tagging through the
**TFLite C API**. The chat partner "Alex" replies through the remote API.

Read the iOS repo's [HANDOFF.md](docs/HANDOFF.md) for how the engine works and
which files are coupled to the trained weights. That applies unchanged here.

## Build & run

```bash
brew install xcodegen bazelisk          # once
./scripts/build_tflite_macos.sh         # once; builds Frameworks/tflite/libtensorflowlite_c.dylib
./scripts/bootstrap.sh                  # clones models + llama.xcframework from ../demo, runs xcodegen
open SLMChatMac.xcodeproj               # scheme SLMChat, run on "My Mac"
```

`bootstrap.sh` takes assets from the iOS checkout (`SLMCHAT_IOS_ROOT`, default
`../demo`) using APFS clones, so they take no extra disk space and the iOS tree is
never written. It lists anything missing and exits non-zero.

`SLMChatMac.xcodeproj` is generated; `project.yml` is the source of truth.

Tests (guardrail tokenizer/model parity, same fixtures as iOS):

```bash
xcodebuild test -project SLMChatMac.xcodeproj -scheme SLMChat -destination 'platform=macOS'
```

## What differs from iOS

Sources in `SLMChat/` and `GuardrailTests/` are a copy of the iOS app. Changes:

| File | Change |
|---|---|
| `SLMChatApp.swift` | single `Window` + `Settings` scene (⌘,) + ⌘N command |
| `View/ConversationListView.swift` | `NavigationSplitView` sidebar instead of a push stack |
| `View/Platform.swift` | new — `PlatformImage`, `Clipboard`, toolbar placements, keyboard dismissal |
| other `View/*` | iOS-only modifiers routed through `Platform.swift` |
| `Engine/ImageTagEngine.swift` | macOS decode via ImageIO (EXIF orientation baked in) |

Everything under `Engine/` other than that, `Model/`, `Store/` and the `.mm`
wrappers is unchanged. **Prompts, catalogs and generation params are owned by
the AI engineer** (HANDOFF §2): take changes from the iOS repo, don't edit here.
