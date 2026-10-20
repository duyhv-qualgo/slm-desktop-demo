#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter for Ternary Bonsai 2 to a llama.cpp GGUF adapter.

Why not llama.cpp's convert_lora_to_gguf.py: it needs the base model's HF folder
(torch + transformers), and for Bonsai 2B it would produce the same tensors — the
qwen35 converter only reorders linear-attention rows/columns when the key and
value head counts differ, and in the 2B they're both 16. This script maps names
and copies the matrices, and refuses a base where that reorder would apply.

The adapter stays in the primal basis: the patched llama.cpp (patches/) feeds a
folded weight's LoRA delta the unrotated input, so a normally trained PEFT pack
applies as is. `--folded` instead writes A·D·H (the delta in the rotated basis),
for merging with llama-export-lora in the parity check (scripts/lora_parity/).

  python peft_lora_to_gguf.py ADAPTER_DIR_OR_SAFETENSORS BASE.gguf OUT.gguf [--alpha 64] [--folded]
                               [--pad-rank 64]

`--pad-rank` zero-pads A's rows and B's columns up to that rank and scales alpha
with it, so B·A and alpha/rank — the delta — are unchanged: the same pack with
another rank's matmul shapes, for measuring what rank costs on Metal.

alpha comes from adapter_config.json next to the safetensors unless given.
"""
import argparse, json, os, re, struct, sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP_DIR",
                str(Path(__file__).resolve().parents[2] / "llama.cpp")), "gguf-py"))
import numpy as np
import gguf

# PEFT module (Qwen3.5 HF names) -> llama.cpp qwen35 tensor kind
MODULES = {
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z":   "attn_gate",
    "linear_attn.out_proj":    "ssm_out",
    "self_attn.q_proj":        "attn_q",
    "self_attn.k_proj":        "attn_k",
    "self_attn.v_proj":        "attn_v",
    "self_attn.o_proj":        "attn_output",
    "mlp.gate_proj":           "ffn_gate",
    "mlp.up_proj":             "ffn_up",
    "mlp.down_proj":           "ffn_down",
}
NAME = re.compile(r"layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")
DTYPES = {"F16": np.float16, "BF16": None, "F32": np.float32}


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        header.pop("__metadata__", None)
        out = {}
        for name, info in header.items():
            s, e = info["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if info["dtype"] == "BF16":   # widen bf16 -> f32 via the high half
                arr = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                arr = np.frombuffer(raw, DTYPES[info["dtype"]])
            out[name] = arr.reshape(info["shape"])
        return out


def kv(reader, name, default=None):
    if name not in reader.fields:
        return default
    f = reader.fields[name]
    if f.types[0] == gguf.GGUFValueType.ARRAY:
        if f.types[-1] == gguf.GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode() for i in f.data]
        return [f.parts[i][0].item() for i in f.data]
    if f.types[0] == gguf.GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode()
    return f.parts[f.data[0]][0].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter"); ap.add_argument("base"); ap.add_argument("out")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--folded", action="store_true", help="write A·D·H (parity check only)")
    ap.add_argument("--pad-rank", type=int, help="zero-pad to this rank (same delta, for speed tests)")
    a = ap.parse_args()

    src = Path(a.adapter)
    st = src / "adapter_model.safetensors" if src.is_dir() else src
    cfg_path = st.parent / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    alpha = a.alpha if a.alpha is not None else cfg.get("lora_alpha")
    if alpha is None:
        sys.exit("no lora_alpha: pass --alpha or put adapter_config.json next to the safetensors")
    if cfg.get("fan_in_fan_out"):
        sys.exit("fan_in_fan_out adapters are not supported")

    r = gguf.GGUFReader(a.base)
    arch = kv(r, "general.architecture")
    if arch != "qwen35":
        sys.exit(f"base arch is {arch}, expected qwen35")
    n_k, n_v = kv(r, "qwen35.ssm.group_count"), kv(r, "qwen35.ssm.time_step_rank")
    if n_k != n_v:
        sys.exit(f"linear-attention heads k={n_k} v={n_v} differ: the V-head reorder applies, "
                 "use llama.cpp's convert_lora_to_gguf.py instead")
    shapes = {t.name: [int(x) for x in t.shape] for t in r.tensors}   # ne: [n_in, n_out]
    refused = set(kv(r, "prism.hadamard.inverse_weight_names", []))     # token_embd (+ tied head)

    signs, bs = {}, kv(r, "prism.hadamard.block_size")
    if a.folded:
        off = 0
        values = kv(r, "prism.hadamard.sign_values")
        for w in kv(r, "prism.hadamard.sign_widths"):
            signs[w] = np.array(values[off:off + w], np.float64); off += w
        i = np.arange(bs)
        par = np.vectorize(lambda x: bin(x).count("1") & 1)(i[:, None] & i[None, :])
        Hb = np.where(par == 1, -1.0, 1.0) / np.sqrt(bs)

    pairs = {}
    for name, arr in read_safetensors(st).items():
        m = NAME.search(name)
        if not m or m.group(2) not in MODULES:
            sys.exit(f"unexpected adapter tensor: {name}")
        target = f"blk.{m.group(1)}.{MODULES[m.group(2)]}.weight"
        if target in refused:
            sys.exit(f"{target} is a Hadamard latent lookup table; LoRA on it is not supported")
        pairs.setdefault(target, {})[m.group(3)] = arr

    if a.pad_rank:
        rank = {p["A"].shape[0] for p in pairs.values()}
        if len(rank) != 1 or a.pad_rank < next(iter(rank)):
            sys.exit(f"--pad-rank {a.pad_rank} needs one rank <= it, adapter has {sorted(rank)}")
        r0 = next(iter(rank))
        alpha = alpha * a.pad_rank / r0          # keeps alpha / rank, so the scale
        for p in pairs.values():
            p["A"] = np.concatenate([p["A"], np.zeros((a.pad_rank - r0, p["A"].shape[1]), p["A"].dtype)], 0)
            p["B"] = np.concatenate([p["B"], np.zeros((p["B"].shape[0], a.pad_rank - r0), p["B"].dtype)], 1)

    w = gguf.GGUFWriter(a.out, arch)
    w.add_string("general.type", "adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", float(alpha))
    ranks = set()
    for target in sorted(pairs, key=lambda t: (int(t.split(".")[1]), t)):
        A, B = pairs[target]["A"], pairs[target]["B"]          # PEFT: A (r, n_in), B (n_out, r)
        n_in, n_out = shapes[target]
        if A.shape != (A.shape[0], n_in) or B.shape != (n_out, A.shape[0]):
            sys.exit(f"{target}: adapter A{A.shape} B{B.shape} don't fit base [{n_in} -> {n_out}]")
        ranks.add(A.shape[0])
        if a.folded:
            Ad = A.astype(np.float64) * signs[n_in][None, :]
            A = (Ad.reshape(A.shape[0], n_in // bs, bs) @ Hb).reshape(A.shape[0], n_in)
            A, B = A.astype(np.float32), B.astype(np.float32)
        w.add_tensor(target + ".lora_a", np.ascontiguousarray(A))
        w.add_tensor(target + ".lora_b", np.ascontiguousarray(B))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.out}: {len(pairs)} targets, rank {sorted(ranks)}, alpha {alpha}"
          f" (scale {alpha / max(ranks):g}){' [folded A·D·H]' if a.folded else ''}")


if __name__ == "__main__":
    main()
