#!/usr/bin/env bash
# =============================================================================
#  build_llama_macos.sh — build the macOS-only llama.xcframework the desktop app
#  links, from official llama.cpp at a pinned tag plus the patches in patches/.
#
#  Why not the release's prebuilt xcframework: Ternary Bonsai 2 folds a Hadamard
#  rotation into its weights, and official llama.cpp refuses LoRA adapters on
#  those weights (the LoRA delta would read the rotated input). The patch feeds
#  the delta the unrotated input instead — the PrismML fork's approach — so the
#  per-language LoRA packs work on Bonsai. Drop it once that lands upstream.
#
#  The desktop app no longer takes llama.xcframework from the iOS checkout: iOS
#  keeps its own (older) build. Bonsai needs b11556+ (prism.hadamard runtime).
#
#    LLAMA_CPP_DIR   llama.cpp checkout (default ../llama.cpp, cloned if absent)
#    LLAMA_CPP_TAG   tag to build (default below)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
TAG="${LLAMA_CPP_TAG:-b11669}"
SRC="${LLAMA_CPP_DIR:-$ROOT/../llama.cpp}"
PATCH="$ROOT/patches/llama.cpp-$TAG-hadamard-lora.patch"

[[ -f "$PATCH" ]] || { echo "ERROR: no patch for $TAG at $PATCH"; exit 1; }

if [[ ! -d "$SRC/.git" ]]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
fi
cd "$SRC"
[[ "$(git rev-parse HEAD)" == "$(git rev-parse "$TAG^{commit}")" ]] \
  || { echo "ERROR: $SRC is not at $TAG"; exit 1; }

# Apply the patch unless it's already in the tree; refuse anything else dirty.
if git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "patch already applied"
elif [[ -z "$(git status --porcelain --untracked-files=no)" ]]; then
  git apply "$PATCH"
else
  echo "ERROR: $SRC has local changes that are not $PATCH"; exit 1
fi

./build-xcframework.sh macos

OUT="$ROOT/Frameworks/llama.xcframework"
if [[ -e "$OUT" && ! -e "$OUT.pre-$TAG" ]]; then
  mv "$OUT" "$OUT.pre-$TAG"   # keep the previous build until the port is verified
fi
rm -rf "$OUT"
cp -R build-apple/llama.xcframework "$OUT"
echo "OK — $OUT (llama.cpp $TAG + $(basename "$PATCH"))"
