#!/usr/bin/env bash
# =============================================================================
#  bootstrap.sh — gather the binaries SLMChat for macOS needs, then run xcodegen.
#
#  Models, frameworks and the API config are not in git. They are pulled from a
#  checkout of the iOS demo (default ../demo, override with SLMCHAT_IOS_ROOT)
#  using APFS clones (`cp -c`): no extra disk space, and the iOS tree is only
#  read, never written.
#
#  The iOS repo's own Resources/ is NOT authoritative — it holds leftovers from
#  other branches (q4_0, v18). The list below is exactly what the catalogs in
#  SLMChat/Engine reference, and where each file actually lives over there.
#
#  The TFLite C library is built separately: scripts/build_tflite_macos.sh.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
IOS="${SLMCHAT_IOS_ROOT:-$ROOT/../demo}"
[[ -d "$IOS/SLMChat" ]] || { echo "ERROR: iOS demo not found at $IOS (set SLMCHAT_IOS_ROOT)"; exit 1; }

mkdir -p Resources Frameworks
missing=()

clone() {  # clone <src relative to iOS root> <dest dir>
  local src="$IOS/$1" dest="$2/$(basename "$1")"
  if [[ ! -e "$src" ]]; then missing+=("$1"); return; fi
  [[ -e "$dest" ]] && return
  cp -cR "$src" "$dest" 2>/dev/null || cp -R "$src" "$dest"
}

# --- Base checkpoints (ModelCatalog) -----------------------------------------
clone Resources/model.gguf                 Resources
clone models-lab/qwen2.5-0.5b-v6.1.gguf    Resources
clone models-lab/qwen3-0.6b-v6.gguf        Resources
clone models-lab/qwen3.5-0.8b.gguf         Resources

# --- LoRA language packs (LoRACatalog) ---------------------------------------
clone Resources/lora-gemma3-270m-english.gguf    Resources
clone Resources/lora-gemma3-270m-vietnamese.gguf Resources
for base in qwen2.5-0.5b-v6.1 qwen3-0.6b-v6 qwen3.5-0.8b; do
  clone "Resources/lora-$base-english.gguf" Resources
  for lang in vietnamese arabic spanish french russian chinese; do
    clone "models-lab/lora-$base-$lang.gguf" Resources
  done
done

# --- Guardrail (GuardPackCatalog + GuardRegion) ------------------------------
for l in vi en ar es fr ru zh de; do
  clone "guardrails/2026_08_25_v16_multilingual_label_smooth/$l/guardrail-v16ls-$l.tflite" Resources
  clone "Resources/guardrail-v16-tokenizer-$l.json" Resources
done
for f in "$IOS"/Resources/guardrail-rules-*.json; do clone "Resources/$(basename "$f")" Resources; done

# --- Language detection + image tagging --------------------------------------
for f in langdetect.tflite langdetect-labels.txt langdetect-weights.json \
         image-tagging.tflite image-tag-list.txt; do
  clone "Resources/$f" Resources
done

# langdetect-weights.bin is gitignored on iOS and exists nowhere — derive it.
if [[ ! -e Resources/langdetect-weights.bin ]]; then
  python3 scripts/build_langdetect_weights.py "$IOS" \
    || missing+=("Resources/langdetect-weights.bin (scripts/build_langdetect_weights.py failed)")
fi

# --- Text + photo embeddings for semantic search (not in the iOS repo; Hugging Face) --
EMB=embeddinggemma-2-text-vision-440m.litertlm
EMB_SHA=92dcbea108899e5d6e30d919b0744f90d9967e80c67a4ab5503ac16d54f62eb0
if [[ ! -e "Resources/$EMB" ]]; then
  echo "downloading $EMB (388 MB)…"
  curl -fL --progress-bar -o "Resources/$EMB.part" \
    "https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm/resolve/main/$EMB" \
    && echo "$EMB_SHA  Resources/$EMB.part" | shasum -a 256 -c --quiet \
    && mv "Resources/$EMB.part" "Resources/$EMB" \
    || { rm -f "Resources/$EMB.part"; missing+=("Resources/$EMB (download or checksum failed)"); }
fi

# --- Remote chat partner config (secret, gitignored) -------------------------
[[ -e "$IOS/Resources/APIConfig.plist" ]] && clone Resources/APIConfig.plist Resources \
  || echo "note: no APIConfig.plist — the chat partner will be mute (see APIConfig.example.plist)"

# --- llama.cpp: built here, NOT taken from iOS -------------------------------
# Official llama.cpp at a pinned tag + patches/ (Bonsai 2 needs b11556+ and the
# Hadamard LoRA patch); iOS keeps its own older build. See build_llama_macos.sh.
[[ -e Frameworks/llama.xcframework ]] || ./scripts/build_llama_macos.sh \
  || missing+=("Frameworks/llama.xcframework (scripts/build_llama_macos.sh failed)")

# --- TFLite C library (built locally) -----------------------------------------
[[ -e Frameworks/tflite/libtensorflowlite_c.dylib ]] \
  || missing+=("Frameworks/tflite/libtensorflowlite_c.dylib (run scripts/build_tflite_macos.sh)")

if (( ${#missing[@]} )); then
  echo "ERROR: missing:"; printf '  %s\n' "${missing[@]}"; exit 1
fi

command -v xcodegen >/dev/null || { echo "ERROR: brew install xcodegen"; exit 1; }
xcodegen generate
echo "OK — open SLMChatMac.xcodeproj"
