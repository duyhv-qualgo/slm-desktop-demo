#!/usr/bin/env bash
# =============================================================================
#  run_parity.sh — prove the Hadamard LoRA patch (patches/) is exact on Ternary
#  Bonsai 2, with llama.cpp's own KL-divergence tooling.
#
#  Synthetic adapters (make_adapters.py) on every folded weight, then:
#    R1  base + zero adapter    vs base             KLD must be ~0
#    R2  base + adapter         vs base             KLD must be clearly > 0 (sensitivity)
#    R3  merged(A·D·H)          vs base + adapter   KLD must be ~0   <- the claim
#    R4  merged(A, unrotated)   vs base + adapter   KLD must be large (what unpatched
#                                                    llama.cpp would compute)
#  plus: official llama.cpp (if on PATH) must refuse the adapter.
#
#    PYTHON          python with numpy >= 2.2.6, tqdm, pyyaml, requests
#    LLAMA_CPP_DIR   patched llama.cpp checkout with build-tools/bin (default ../../../llama.cpp)
#    usage: run_parity.sh BASE.gguf TEXT.txt WORKDIR
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA="${LLAMA_CPP_DIR:-$HERE/../../../llama.cpp}"
PYTHON="${PYTHON:-python3}"
BASE="$1"; TEXT="$2"; W="$3"; mkdir -p "$W"
BIN="$LLAMA/build-tools/bin"
C=(-f "$TEXT" -c 512 --chunks 16 -ngl 99 -fa on)

"$PYTHON" -I "$HERE/make_adapters.py" "$BASE" "$W/adapters"
"$BIN/llama-export-lora" -m "$BASE" --lora "$W/adapters/folded.gguf" -o "$W/merged_folded.gguf" >/dev/null 2>&1
"$BIN/llama-export-lora" -m "$BASE" --lora "$W/adapters/primal.gguf" -o "$W/merged_wrong.gguf"  >/dev/null 2>&1

if command -v llama-perplexity >/dev/null; then
  echo "=== official llama.cpp on PATH ($(llama-perplexity --version 2>&1 | grep -o 'build [0-9]*' | head -1)):"
  refusal="$(llama-perplexity -m "$BASE" --lora "$W/adapters/primal.gguf" "${C[@]}" --chunks 1 2>&1 \
               | grep -o "LoRA tensor .*not supported" | head -1 || true)"
  echo "    ${refusal:-(loaded — no refusal)}"
fi

kld() { "$BIN/llama-perplexity" "$@" "${C[@]}" 2>&1 \
          | grep -E "Mean +KLD|Maximum KLD|Same top p" | sed 's/^[0-9.]* [IEW] /    /'; }
"$BIN/llama-perplexity" -m "$BASE" "${C[@]}" --kl-divergence-base "$W/base.kld" >/dev/null 2>&1
"$BIN/llama-perplexity" -m "$BASE" --lora "$W/adapters/primal.gguf" "${C[@]}" --kl-divergence-base "$W/lora.kld" >/dev/null 2>&1
echo "=== R1 base + zero adapter vs base (expect ~0)";      kld -m "$BASE" --lora "$W/adapters/zero.gguf" --kl-divergence-base "$W/base.kld" --kl-divergence
echo "=== R2 base + adapter vs base (expect > 0)";          kld -m "$BASE" --lora "$W/adapters/primal.gguf" --kl-divergence-base "$W/base.kld" --kl-divergence
echo "=== R3 merged A·D·H vs base + adapter (expect ~0)";   kld -m "$W/merged_folded.gguf" --kl-divergence-base "$W/lora.kld" --kl-divergence
echo "=== R4 merged unrotated vs base + adapter (expect large)"; kld -m "$W/merged_wrong.gguf" --kl-divergence-base "$W/lora.kld" --kl-divergence
