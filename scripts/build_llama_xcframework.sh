#!/usr/bin/env bash
# =============================================================================
#  build_llama_xcframework.sh — build Frameworks/llama.xcframework (llama.cpp +
#  Metal) from a pinned llama.cpp commit, plus any scripts/llama-patches/*.patch.
#
#    scripts/build_llama_xcframework.sh            # macOS slice (this app)
#    scripts/build_llama_xcframework.sh macos ios  # + iPhone and simulator
#    LLAMA_COMMIT=<sha> scripts/build_llama_xcframework.sh   # try another llama.cpp
#    LLAMA_SRC=~/llama.cpp scripts/build_llama_xcframework.sh
#                       # build a local checkout as-is: no fetch, no patches
#
#  A few minutes per slice on an M-series Mac. Needs cmake (brew install cmake).
#  Output replaces Frameworks/llama.xcframework (the old one is kept in
#  build/llama/previous/); provenance goes to Frameworks/llama-build.txt.
#
#  The pin is release v0.7.0 (2026-10-15). llama.cpp has versioned releases
#  (vX.Y.Z) since Aug 2026 — the bNNNN tags are per-merge CI builds — so bump
#  to the next release, reading its "API changes" notes first. Before this the
#  app ran b9721 (2026-06-19), what the iOS demo's framework is built from; the
#  only port it needed was use_mmap → load_mode in ChatLM.mm.
#
#  Layout follows llama.cpp's own build-xcframework.sh — static libs folded
#  into one dynamic framework, Metal shaders embedded, headers + module map —
#  with two deliberate differences:
#   - arm64 only. A fat arm64+x86_64 build leaves ggml unable to pick a CPU
#     architecture, so it falls back to generic C kernels (-DGGML_CPU_GENERIC)
#     for BOTH slices, slowing every CPU-side op. The app is arm64-only anyway.
#   - minimum OS versions match project.yml (macOS 14.0) and the iOS app (16.0).
#
#  Changing LLAMA_COMMIT or adding a patch changes the chat model's runtime:
#  re-run the full test suite and PrefillThroughputProbe (CPU and GPU).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_COMMIT="${LLAMA_COMMIT:-620d2225cc436ab0b001e3200e590e3b286ef9ab}"   # v0.7.0 (= b11669), 2026-10-15
MACOS_MIN=14.0
IOS_MIN=16.0

WORK="$ROOT/build/llama"
OUT="${OUT:-$ROOT/Frameworks/llama.xcframework}"
JOBS="$(sysctl -n hw.logicalcpu)"

SLICES=("$@"); (( ${#SLICES[@]} )) || SLICES=(macos)
for s in "${SLICES[@]}"; do
  [[ "$s" == macos || "$s" == ios ]] || { echo "ERROR: unknown slice '$s' (macos, ios)"; exit 1; }
done
command -v cmake >/dev/null || { echo "ERROR: brew install cmake"; exit 1; }
mkdir -p "$WORK"

# --- 1. Source ----------------------------------------------------------------
if [[ -n "${LLAMA_SRC:-}" ]]; then
  SRC="$(cd "$LLAMA_SRC" && pwd)"
  echo "=== Source: $SRC (as-is, patches NOT applied)"
  SOURCE_DESC="local $SRC @ $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo '?')"
  [[ -n "$(git -C "$SRC" status --porcelain --untracked-files=no 2>/dev/null)" ]] \
    && SOURCE_DESC+=" + uncommitted changes"
else
  SRC="$WORK/src"
  echo "=== Source: $LLAMA_REPO @ ${LLAMA_COMMIT:0:12}"
  if ! git -C "$SRC" cat-file -e "$LLAMA_COMMIT^{commit}" 2>/dev/null; then
    rm -rf "$SRC"; git init -q "$SRC"
    git -C "$SRC" fetch -q --depth 1 "$LLAMA_REPO" "$LLAMA_COMMIT"
  fi
  git -C "$SRC" checkout -q --force "$LLAMA_COMMIT"
  git -C "$SRC" clean -fdxq
  SOURCE_DESC="$LLAMA_REPO @ $LLAMA_COMMIT"
  for p in "$ROOT"/scripts/llama-patches/*.patch; do
    [[ -e "$p" ]] || continue
    echo "    patch $(basename "$p")"
    git -C "$SRC" apply "$p"
    SOURCE_DESC+=$'\n'"  + scripts/llama-patches/$(basename "$p")"
  done
fi

# --- 2. Build each slice as static libraries -------------------------------------
build_slice() {  # build_slice <name> <cmake args…>
  local name="$1"; shift
  local dir="$WORK/$name"
  echo "=== Building $name (log: build/llama/$name.log)"
  rm -rf "$dir"
  if ! cmake -S "$SRC" -B "$dir" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-g" -DCMAKE_CXX_FLAGS="-g" \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF \
        -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS_DEFAULT=ON \
        -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
        -DLLAMA_BUILD_MTMD=ON -DMTMD_VIDEO=OFF -DLLAMA_OPENSSL=OFF \
        "$@" > "$WORK/$name.log" 2>&1 \
     || ! cmake --build "$dir" -j "$JOBS" >> "$WORK/$name.log" 2>&1; then
    tail -30 "$WORK/$name.log"; echo "ERROR: $name failed"; exit 1
  fi
  # The fallback this script exists to avoid — never ship it silently.
  if grep -q "GGML_CPU_GENERIC" "$WORK/$name.log"; then
    echo "ERROR: $name fell back to generic CPU kernels (see log)"; exit 1
  fi
}

# --- 3. Fold into a dynamic llama.framework ------------------------------------
make_framework() {  # make_framework <name> <sdk> <min-version flag> <macos|ios>
  local name="$1" sdk="$2" minflag="$3" kind="$4"
  local dir="$WORK/$name" fw="$WORK/$name/out/llama.framework"
  local res hdr mods bin install_name
  rm -rf "$WORK/$name/out"
  if [[ "$kind" == macos ]]; then   # versioned bundle
    mkdir -p "$fw/Versions/A"/{Headers,Modules,Resources}
    ln -s A "$fw/Versions/Current"
    for l in Headers Modules Resources; do ln -s "Versions/Current/$l" "$fw/$l"; done
    ln -s Versions/Current/llama "$fw/llama"
    hdr="$fw/Versions/A/Headers"; mods="$fw/Versions/A/Modules"; res="$fw/Versions/A/Resources"
    bin="$fw/Versions/A/llama"; install_name="@rpath/llama.framework/Versions/Current/llama"
  else                              # flat bundle
    mkdir -p "$fw"/{Headers,Modules}
    hdr="$fw/Headers"; mods="$fw/Modules"; res="$fw"
    bin="$fw/llama"; install_name="@rpath/llama.framework/llama"
  fi

  cp "$SRC"/include/llama.h \
     "$SRC"/ggml/include/{ggml.h,ggml-opt.h,ggml-alloc.h,ggml-backend.h,ggml-metal.h,ggml-cpu.h,ggml-blas.h,gguf.h} \
     "$hdr/"
  # mtmd (multimodal) only where this llama.cpp builds it — no headers without the code.
  if [[ -n "$(find "$dir" -name libmtmd.a -not -path '*/out/*')" ]]; then
    cp "$SRC"/tools/mtmd/{mtmd.h,mtmd-helper.h} "$hdr/"
  fi
  cat > "$mods/module.modulemap" <<'EOF'
framework module llama {
    umbrella "Headers"

    link "c++"
    link framework "Accelerate"
    link framework "Metal"
    link framework "Foundation"

    export *
}
EOF
  local platforms=MacOSX; [[ "$kind" == ios ]] && platforms="$([[ $sdk == iphoneos ]] && echo iPhoneOS || echo iPhoneSimulator)"
  cat > "$res/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleExecutable</key><string>llama</string>
    <key>CFBundleIdentifier</key><string>org.ggml.llama</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>llama</string>
    <key>CFBundlePackageType</key><string>FMWK</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>MinimumOSVersion</key><string>${minflag##*=}</string>
    <key>CFBundleSupportedPlatforms</key><array><string>$platforms</string></array>
</dict>
</plist>
EOF

  # Every static lib the build produced (llama, ggml-*, mtmd, vendor) → one dylib.
  local libs=(); while IFS= read -r l; do libs+=("$l"); done < <(find "$dir" -name '*.a' -not -path '*/out/*' | sort)
  xcrun libtool -static -o "$dir/combined.a" "${libs[@]}" 2>/dev/null
  xcrun -sdk "$sdk" clang++ -dynamiclib -arch arm64 "$minflag" \
    -isysroot "$(xcrun --sdk "$sdk" --show-sdk-path)" \
    -Wl,-force_load,"$dir/combined.a" \
    -framework Foundation -framework Metal -framework Accelerate \
    -install_name "$install_name" -o "$bin"
  rm -f "$dir/combined.a"

  # Debug info → dSYM, then strip it from the shipped binary.
  xcrun dsymutil "$bin" -o "$WORK/$name/out/llama.dSYM" 2>/dev/null
  xcrun strip -S "$bin" -o "$bin.stripped" && mv "$bin.stripped" "$bin"
  XCFW_ARGS+=(-framework "$fw" -debug-symbols "$WORK/$name/out/llama.dSYM")
}

XCFW_ARGS=()
for s in "${SLICES[@]}"; do
  case "$s" in
    macos)
      build_slice macos -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=$MACOS_MIN
      make_framework macos macosx "-mmacosx-version-min=$MACOS_MIN" macos ;;
    ios)
      for v in iphoneos iphonesimulator; do
        build_slice "$v" -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=$v -DGGML_METAL_TARGET_OS=ios \
          -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=$IOS_MIN
        if [[ $v == iphoneos ]]; then flag="-mios-version-min=$IOS_MIN"; else flag="-mios-simulator-version-min=$IOS_MIN"; fi
        make_framework "$v" "$v" "$flag" ios
      done ;;
  esac
done

# --- 4. xcframework, then swap it in -----------------------------------------------
echo "=== Creating llama.xcframework"
rm -rf "$WORK/llama.xcframework"
xcrun xcodebuild -create-xcframework "${XCFW_ARGS[@]}" -output "$WORK/llama.xcframework" >/dev/null

if [[ -e "$OUT" ]]; then
  rm -rf "$WORK/previous"; mkdir -p "$WORK/previous"
  mv "$OUT" "$WORK/previous/"
  [[ -e "$(dirname "$OUT")/llama-build.txt" ]] && mv "$(dirname "$OUT")/llama-build.txt" "$WORK/previous/"
fi
mkdir -p "$(dirname "$OUT")"
mv "$WORK/llama.xcframework" "$OUT"
cat > "$(dirname "$OUT")/llama-build.txt" <<EOF
llama.xcframework — built $(date -u +%Y-%m-%dT%H:%M:%SZ) by scripts/build_llama_xcframework.sh
source: $SOURCE_DESC
slices: ${SLICES[*]} (arm64; macOS $MACOS_MIN, iOS $IOS_MIN)
toolchain: $(xcrun xcodebuild -version | tr '\n' ' ')
EOF
echo "OK — $OUT"
cat "$(dirname "$OUT")/llama-build.txt"
