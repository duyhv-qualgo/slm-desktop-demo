#!/usr/bin/env bash
# =============================================================================
#  build_tflite_macos.sh — build Frameworks/tflite/libtensorflowlite_c.dylib
#  (macOS arm64) from google-ai-edge/LiteRT source.
#
#  GuardLM and ImageTagLM call only the classic TFLite interpreter C API
#  (TfLiteModelCreateFromFile, TfLiteInterpreter*, TfLiteTensor* — see
#  SLMChat/TFLiteC.h). On iOS those symbols come re-exported from CLiteRTLM;
#  there is no macOS slice of that, and the ai-edge-litert wheel's libLiteRt.dylib
#  does not export them. So we build LiteRT's own //tflite/c:tensorflowlite_c.
#
#  Prereqs: Xcode, `brew install bazelisk`. First build takes several minutes.
#
#  HERMETIC_PYTHON_VERSION is required: Bazel otherwise defaults to 3.9, for
#  which LiteRT v2.2.0 ships no requirements lock and fails at repo mapping.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
APP_ROOT="$(pwd)"

LITERT_VERSION="${LITERT_VERSION:-v2.2.0}"
SRC="$APP_ROOT/build/litert-src/LiteRT-$LITERT_VERSION"
PYVER="${HERMETIC_PYTHON_VERSION:-3.12}"

if [[ ! -d "$SRC" ]]; then
  echo "=== Cloning LiteRT ${LITERT_VERSION}…"
  mkdir -p "$(dirname "$SRC")"
  git clone -q --depth 1 --branch "$LITERT_VERSION" \
    https://github.com/google-ai-edge/LiteRT.git "$SRC"
fi

echo "=== bazel build //tflite/c:tensorflowlite_c (macos_arm64, python $PYVER)…"
( cd "$SRC" && HERMETIC_PYTHON_VERSION="$PYVER" bazelisk build -c opt \
    --config=macos_arm64 --repo_env=HERMETIC_PYTHON_VERSION="$PYVER" \
    //tflite/c:tensorflowlite_c )

OUT="$APP_ROOT/Frameworks/tflite"
mkdir -p "$OUT"
cp -f "$SRC/bazel-bin/tflite/c/libtensorflowlite_c.dylib" "$OUT/"
chmod u+w "$OUT/libtensorflowlite_c.dylib"
# Load from the app bundle's Frameworks/ (embedded by Xcode).
install_name_tool -id @rpath/libtensorflowlite_c.dylib "$OUT/libtensorflowlite_c.dylib"

n=$(nm -gU "$OUT/libtensorflowlite_c.dylib" | grep -cE ' _TfLite(Interpreter|Model|Tensor)')
echo "=== OK: $OUT/libtensorflowlite_c.dylib ($n TfLite C symbols exported)"
