"""Write synthetic LoRA adapters for Ternary Bonsai 2 to test the Hadamard LoRA patch.

For every folded target W_f = W·D·H (prism.hadamard metadata in the base GGUF):
  primal.gguf  — random (A, B) in the primal basis: what a normally trained pack is.
                 The patched runtime computes W_f·(H·D·x) + B·A·x.
  folded.gguf  — the same delta with A_f = A·D·H, for merging into W_f with
                 llama-export-lora: (W_f + B·A_f)·(H·D·x) == W·x + B·A·x exactly.
  zero.gguf    — primal A, B = 0: must leave the output unchanged.
Merging primal.gguf unfolded (W_f + B·A) is the negative control: the delta then
reads the rotated input, which is what unpatched llama.cpp would compute.

usage: python make_adapters.py BASE.gguf OUTDIR [--rank 8] [--sigma 0.02] [--only attn_q,ffn_up]
"""
import argparse, os, sys
from pathlib import Path

# gguf-py from the llama.cpp checkout scripts/build_llama_macos.sh builds from
sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP_DIR",
                str(Path(__file__).resolve().parents[3] / "llama.cpp")), "gguf-py"))
import numpy as np
import gguf

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("outdir")
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--sigma", type=float, default=0.02, help="std of B")
ap.add_argument("--only", default="", help="comma list of tensor kinds to target (default: all folded)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

r = gguf.GGUFReader(a.base)
def kv(name):
    f = r.fields[name]
    if f.types[0] == gguf.GGUFValueType.ARRAY:
        if f.types[-1] == gguf.GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode() for i in f.data]
        return [f.parts[i][0].item() for i in f.data]
    if f.types[0] == gguf.GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode()
    return f.parts[f.data[0]][0].item()

arch = kv("general.architecture")
bs = kv("prism.hadamard.block_size")
names = kv("prism.hadamard.weight_names")
assert kv("prism.hadamard.sign_mode") == "explicit"
assert "prism.hadamard.gdn_v_grouped" not in r.fields, "gdn_v_grouped needs the ssm_out permutation — not handled"
widths, values = kv("prism.hadamard.sign_widths"), kv("prism.hadamard.sign_values")
signs, off = {}, 0
for w in widths:
    signs[w] = np.array(values[off:off + w], dtype=np.float32); off += w

# normalized Sylvester Walsh-Hadamard block, built exactly as llama.cpp does
i = np.arange(bs)
par = np.bitwise_and(i[:, None], i[None, :])
par = np.array([[bin(v).count("1") & 1 for v in row] for row in par])
Hb = np.where(par == 1, -1.0, 1.0).astype(np.float64) / np.sqrt(bs)
assert np.allclose(Hb @ Hb, np.eye(bs))

shapes = {t.name: [int(x) for x in t.shape] for t in r.tensors}   # gguf ne order: [n_in, n_out]
only = set(filter(None, a.only.split(",")))
targets = [n for n in names if n in shapes and n != "token_embd.weight"
           and (not only or any(f".{k}.weight" in n for k in only))]
print(f"{arch}: H{bs}, {len(targets)} folded targets, rank {a.rank}, sigma {a.sigma}")

rng = np.random.default_rng(a.seed)
def fold(A, n_in):
    """A·D·H, H block-diagonal over the input dim."""
    Ad = A.astype(np.float64) * signs[n_in][None, :]
    return (Ad.reshape(A.shape[0], n_in // bs, bs) @ Hb).reshape(A.shape[0], n_in).astype(np.float32)

out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
writers = {k: gguf.GGUFWriter(str(out / f"{k}.gguf"), arch) for k in ("primal", "folded", "zero")}
for w in writers.values():
    w.add_string("general.type", "adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", float(a.rank))   # scale = alpha / rank = 1

for n in targets:
    n_in, n_out = shapes[n]
    A = (rng.standard_normal((a.rank, n_in)) / np.sqrt(n_in)).astype(np.float32)
    B = (rng.standard_normal((n_out, a.rank)) * a.sigma).astype(np.float32)
    # numpy (rows, cols) == gguf ne reversed: lora_a ne=[n_in, r], lora_b ne=[r, n_out]
    for k, (aa, bb) in {"primal": (A, B), "folded": (fold(A, n_in), B),
                        "zero": (A, np.zeros_like(B))}.items():
        writers[k].add_tensor(n + ".lora_a", aa)
        writers[k].add_tensor(n + ".lora_b", bb)

for k, w in writers.items():
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", out / f"{k}.gguf")
