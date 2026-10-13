#!/usr/bin/env python3
"""Generate Resources/langdetect-weights.{bin,json} and langdetect-labels.txt.

The .bin is gitignored in the iOS repo and not shipped with it, so it has to be
derived from the iOS checkout's language-detection/langdet_weights.npz. This runs
the iOS script's own `write_weights` (so the layout can't drift from what
LangDetect.swift reads), with its output redirected into THIS repo — the iOS
tree is only read.

    python3 scripts/build_langdetect_weights.py [path/to/ios/demo]
"""
import importlib.util, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IOS = os.path.abspath(sys.argv[1] if len(sys.argv) > 1
                      else os.environ.get("SLMCHAT_IOS_ROOT", os.path.join(HERE, "..", "demo")))

spec = importlib.util.spec_from_file_location(
    "build_langdetect_assets", os.path.join(IOS, "scripts", "build_langdetect_assets.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)          # defines functions; inputs resolve under IOS

os.makedirs(os.path.join(HERE, "Resources"), exist_ok=True)
for name in ("langdetect-weights.json", "langdetect-labels.txt"):
    p = os.path.join(HERE, "Resources", name)
    if os.path.exists(p):             # may be an APFS clone of the iOS file; replace, don't write through
        os.remove(p)
mod.ROOT = HERE                       # outputs only: write_weights writes f"{ROOT}/Resources/..."
mod.write_weights()
