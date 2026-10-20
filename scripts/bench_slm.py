#!/usr/bin/env python3
"""Feature-level benchmark for candidate task models, outside the app.

Starts `llama-server` for one model, replays the app's real task prompts
(SLMChat/Engine/Prompts.swift) at the app's generation params (ChatService),
and writes per-case latency, throughput, peak RSS and raw outputs to JSONL plus
a markdown summary. Compare two runs to decide whether a model is worth
integrating — the app itself only loads what the bundled llama.xcframework
supports.

  # current default, exactly as the app runs it (base + per-task language pack)
  python3 -I scripts/bench_slm.py --name qwen3.5-0.8b \
      --model Resources/qwen3.5-0.8b.gguf \
      --lora-dir Resources --lora-base qwen3.5-0.8b

  # candidate with no finetune: --format-hints adds a system prompt describing
  # each task's output format, since the bare model never learned it
  python3 -I scripts/bench_slm.py --name bonsai2-2b-q2_0 --format-hints \
      --server ../llama.cpp-prism/build/bin/llama-server \
      --model ../demo/models-lab/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf

Prompts are copied from Prompts.swift, which the AI engineer owns — re-sync
them here when they change there.
"""
import argparse, json, os, re, socket, subprocess, sys, time, urllib.request

USER, FRIEND = "Duy", "Alex"

# --- Prompt builders (mirror Prompts.swift) ----------------------------------

def handles(conv):
    return "\n".join(f"<{USER if who == 'me' else FRIEND}>: {t}" for who, t in conv)

def named(conv):
    return "\n".join(f"{USER if who == 'me' else FRIEND}: {t}" for who, t in conv)

def p_suggest(conv, lang):
    return (f"Suggest a few short smart-reply options in {lang} for <{USER}>, "
            f"who replies next in the conversation below.\n\nConversation:\n{handles(conv)}")

def p_summarize(conv, lang):
    return f"Summarize the following conversation in {lang}:\n\n{named(conv)}"

def p_translate(text, src, dst):
    return f"Translate the following {src} sentence to {dst}:\n{text.strip()}"

def p_event(text, lang):
    return ("Extract event information from the following conversation.\n"
            "Return a JSON object with exactly these fields:\n"
            "- intent\n- event_name\n- date\n- time\n- location\n"
            "If a field is missing, return an empty string.\n"
            f"Write event_name and location in {lang}. Keep date and time exactly as written.\n\n"
            f"Conversation:\n{text}")

def p_compose(conv, instruction, lang, draft=""):
    parts = [f"Write a reply in {lang} for {USER} in the conversation below, "
             f"following this instruction: {instruction}",
             f"Conversation:\n{handles(conv)}"]
    if draft:
        parts.append(f"Draft to rewrite:\n{draft}")
    return "\n\n".join(parts)

# Output contracts — the forms the app's parsers accept (parseSuggestions,
# parseSummary, parseTranslation, DetectedEvent, parseCompose), as the finetune
# actually emits them. Only sent with --format-hints.
HINTS = {
    "suggest": "Give 3 short, distinct replies. Format each as a line [[INTENT]] followed by "
               "the reply on the next line, e.g.\n[[ACCEPT]]\nSure, sounds good!\n"
               "Output nothing else.",
    "summarize": "Write the summary as 1-3 plain sentences. Output nothing else.",
    "translate": "Output only the translation, nothing else.",
    "event": 'Output only the JSON object. intent is "has_event" if an event is being '
             'planned, otherwise "no_event".',
    "compose": "Output exactly: [[REPLY]] on the first line, then only the message text.",
}

# App generation params (ChatService.swift).
PARAMS = {
    "suggest":   dict(max_tokens=160, temperature=0.8),
    "summarize": dict(max_tokens=96,  temperature=0.3),
    "translate": dict(max_tokens=160, temperature=0.2),
    "event":     dict(max_tokens=96,  temperature=0.1),
    "compose":   dict(max_tokens=200, temperature=0.5),
}

# --- Cases --------------------------------------------------------------------

C_DINNER = [("friend", "Hey, are you free this weekend?"),
            ("me", "Mostly, why?"),
            ("friend", "Want to grab dinner Saturday at 7? That new pho place on Le Loi."),]
C_VI = [("friend", "Ê mai đi cà phê không?"),
        ("me", "Mấy giờ?"),
        ("friend", "Tầm 9h sáng ở Highlands Nguyễn Huệ nhé"),]
C_WORK = [("friend", "Did you finish the slides for the client?"),
          ("me", "Almost, just the pricing page left"),
          ("friend", "They moved the meeting to tomorrow 10am, can you send it tonight?"),]
C_TRIP = [("friend", "So I checked flights to Da Nang"),
          ("friend", "Vietjet has a sale, 1.2M return if we book today"),
          ("me", "Nice, which dates?"),
          ("friend", "Leave Friday 14th evening, back Sunday 16th"),
          ("friend", "I can book for both of us, just send me your passport name"),
          ("me", "Ok sending now. Hotel?"),
          ("friend", "Found one near My Khe beach, 800k/night, free breakfast"),]

CASES = [
    ("suggest", "en-dinner",  p_suggest(C_DINNER, "English"), None),
    ("suggest", "vi-coffee",  p_suggest(C_VI, "Vietnamese"), None),
    ("suggest", "en-work",    p_suggest(C_WORK, "English"), None),
    ("summarize", "trip-vi",  p_summarize(C_TRIP, "Vietnamese"), None),
    ("summarize", "trip-en",  p_summarize(C_TRIP, "English"), None),
    ("summarize", "work-en",  p_summarize(C_WORK, "English"), None),
    ("translate", "en-vi",    p_translate("They moved the meeting to tomorrow 10am, can you send the slides tonight?", "English", "Vietnamese"), None),
    ("translate", "vi-en",    p_translate("Tầm 9h sáng mai mình gặp nhau ở quán cà phê gần công ty nhé", "Vietnamese", "English"), None),
    ("translate", "en-vi-2",  p_translate("Vietjet has a sale, 1.2M return if we book today", "English", "Vietnamese"), None),
    ("event", "dinner",  p_event(named(C_DINNER), "English"),
        dict(intent="CREATE_EVENT", date="Saturday", time="7", location="Le Loi")),
    ("event", "coffee-vi", p_event(named(C_VI), "Vietnamese"),
        dict(intent="CREATE_EVENT", time="9h", location="Highlands")),
    ("event", "meeting", p_event(named(C_WORK), "English"),
        dict(intent="CREATE_EVENT", date="tomorrow", time="10am")),
    ("event", "none", p_event(named([("friend", "lol that video was hilarious"), ("me", "haha right")]), "English"),
        dict(intent="NONE")),
    ("compose", "agree",    p_compose(C_DINNER, "Agree / accept", "English"), None),
    ("compose", "decline",  p_compose(C_WORK, "Politely decline", "English"), None),
    ("compose", "pro-rw",   p_compose(C_WORK, "Make it more professional", "English",
                                      draft="ya ill send it tonite np"), None),
    ("compose", "vi-fix",   p_compose(C_VI, "Fix spelling and diacritics", "Vietnamese",
                                      draft="ok minh se den dung gio"), None),
]

def target_lang(task, prompt):
    """Language the output must be in, read back from the prompt itself so the
    case table can't disagree with it. Events are free-form fields: unchecked."""
    m = {"suggest": r"options in (\w+)", "summarize": r"conversation in (\w+)",
         "translate": r"sentence to (\w+)", "compose": r"reply in (\w+)"}.get(task)
    return re.search(m, prompt).group(1) if m else None

# --- Scoring (format validity + light correctness; read outputs for the rest) --

def strip_think(s):
    s = re.sub(r"<think>[\s\S]*?</think>", "", s)
    return s.split("<think>")[0].strip()

def first_json(s):
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

VI_CHARS = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)

def lang_ok(text, lang):
    """Coarse target-language check: Vietnamese text carries diacritics, English
    doesn't. Enough to catch an untranslated echo or a reply in the wrong language."""
    if not lang:
        return None
    has_vi = bool(VI_CHARS.search(text))
    return has_vi if lang == "Vietnamese" else not has_vi

def body(out):
    """Text with the finetune's [[TAG]] markers removed, as Prompts.clean does."""
    return re.sub(r"\[\[[^\]]*\]\]", "", out).strip()

def score(task, out, expect, lang):
    o = first_json(out)
    if task == "suggest":
        n = len(o["suggestions"]) if o and isinstance(o.get("suggestions"), list) else \
            len(re.findall(r"\[\[[A-Z_]+\]\]\s*\S", out))
        return dict(format_ok=n >= 2, n=n, lang_ok=lang_ok(body(out), lang))
    if task == "summarize":
        text = (o or {}).get("summary") or body(out)
        return dict(format_ok=bool(text) and "[[" not in out, lang_ok=lang_ok(text, lang))
    if task == "translate":
        text = (o or {}).get("translation") or body(out)
        return dict(format_ok=bool(text), lang_ok=lang_ok(text, lang))
    if task == "compose":
        text = body(out)
        return dict(format_ok=out.lstrip().startswith("[[REPLY]]") and bool(text),
                    lang_ok=lang_ok(text, lang))
    if task == "event":
        if not o:
            return dict(format_ok=False, fields_ok=0, fields=len(expect))
        hit = 0
        for k, v in expect.items():
            got = str(o.get(k, "")).lower()
            if k == "intent":
                hit += (v == "NONE") == (got in ("", "none", "no_event"))
            else:
                hit += v.lower() in got
        return dict(format_ok=all(k in o for k in ("intent", "event_name", "date", "time", "location")),
                    fields_ok=hit, fields=len(expect))
    return {}

# --- Server -------------------------------------------------------------------

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def post(port, path, body, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def rss_mb(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
    return int(out.strip() or 0) / 1024

def pack_language(task, lang, native, packs):
    """The LoRA pack ChatService runs this task under: suggestions and compose
    follow the conversation (`suggestAdapter`), everything else runs on the
    native-language pack the engine was loaded with — which is also what a
    conversation language with no published pack falls back to."""
    want = lang if task in ("suggest", "compose") else native
    return want if want in packs else native

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora-dir", help="dir of lora-<base>-<language>.gguf packs; each "
                    "request runs under the pack the app would pick (see pack_language)")
    ap.add_argument("--lora-base", help="<base> in the pack file names, e.g. qwen3.5-0.8b")
    ap.add_argument("--native", default="Vietnamese")
    ap.add_argument("--server", default="llama-server")
    ap.add_argument("--format-hints", action="store_true")
    ap.add_argument("--repeats", type=int, default=3, help="runs per case; latency is the median")
    ap.add_argument("--out", default="bench-results")
    a = ap.parse_args()

    port = free_port()
    cmd = [a.server, "-m", a.model, "-ngl", "99", "-fa", "on", "-c", "4096", "-np", "1",
           "--port", str(port), "--no-webui", "--jinja"]
    packs = []  # language names, in server adapter-id order
    if a.lora_dir:
        for lang in ("Vietnamese", "English"):
            path = os.path.join(a.lora_dir, f"lora-{a.lora_base}-{lang.lower()}.gguf")
            if os.path.exists(path):
                cmd += ["--lora", path]
                packs.append(lang)
        cmd += ["--lora-init-without-apply"]
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, f"{a.name}.server.log"), "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                if srv.poll() is not None or time.time() - t0 > 120:
                    sys.exit(f"server failed to start, see {log.name}")
                time.sleep(0.5)
        load_s = time.time() - t0
        rss_idle = rss_mb(srv.pid)

        rows, peak = [], rss_idle
        for task, cid, prompt, expect in CASES:
            lang = target_lang(task, prompt)
            msgs = ([{"role": "system", "content": HINTS[task]}] if a.format_hints else []) \
                 + [{"role": "user", "content": prompt}]
            runs = []
            for _ in range(a.repeats):
                body = dict(messages=msgs, top_k=20, top_p=0.95, cache_prompt=False,
                            chat_template_kwargs={"enable_thinking": False}, **PARAMS[task])
                if packs:
                    want = pack_language(task, lang, a.native, packs)
                    body["lora"] = [{"id": i, "scale": 1.0 if l == want else 0.0}
                                    for i, l in enumerate(packs)]
                t = time.time()
                r = post(port, "/v1/chat/completions", body)
                wall = time.time() - t
                tm = r.get("timings", {})
                runs.append(dict(wall_s=wall, prompt_n=tm.get("prompt_n"), pp_tps=tm.get("prompt_per_second"),
                                 gen_n=tm.get("predicted_n"), tg_tps=tm.get("predicted_per_second"),
                                 ttft_s=(tm.get("prompt_ms") or 0) / 1000,
                                 out=r["choices"][0]["message"]["content"] or ""))
                peak = max(peak, rss_mb(srv.pid))
            runs.sort(key=lambda x: x["wall_s"])
            med = runs[len(runs) // 2]
            out = strip_think(med["out"])
            rows.append(dict(task=task, case=cid, **{k: v for k, v in med.items() if k != "out"},
                             out=out, **score(task, out, expect or {}, lang)))
            print(f"{task:9s} {cid:10s} {med['wall_s']:5.2f}s  {out[:90]!r}", flush=True)

        with open(os.path.join(a.out, f"{a.name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_summary(a, rows, load_s, rss_idle, peak)
    finally:
        srv.terminate()
        srv.wait()

def write_summary(a, rows, load_s, rss_idle, peak):
    md = [f"# {a.name}", "", f"- model: `{os.path.basename(a.model)}` ({os.path.getsize(a.model)/2**20:.0f} MiB)",
          f"- lora: {f'per-task packs from {a.lora_base} (native {a.native})' if a.lora_dir else 'none'}"
          f"  · format hints: {a.format_hints}",
          f"- load {load_s:.1f}s · RSS idle {rss_idle:.0f} MB · peak {peak:.0f} MB", "",
          "| task | cases | median wall s | median TTFT s | gen tok/s | format ok | language ok | event fields |",
          "|---|---|---|---|---|---|---|---|"]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
    for task in PARAMS:
        rs = [r for r in rows if r["task"] == task]
        lr = [r for r in rs if r.get("lang_ok") is not None]
        lk = f"{sum(r['lang_ok'] for r in lr)}/{len(lr)}" if lr else ""
        ev = (f"{sum(r['fields_ok'] for r in rs)}/{sum(r['fields'] for r in rs)}" if task == "event" else "")
        md.append(f"| {task} | {len(rs)} | {med([r['wall_s'] for r in rs]):.2f} | "
                  f"{med([r['ttft_s'] for r in rs]):.3f} | {med([r['tg_tps'] or 0 for r in rs]):.1f} | "
                  f"{sum(r['format_ok'] for r in rs)}/{len(rs)} | {lk} | {ev} |")
    md += ["", "## Outputs", ""]
    for r in rows:
        md.append(f"**{r['task']} / {r['case']}** — {r['wall_s']:.2f}s\n\n```\n{r['out']}\n```\n")
    with open(os.path.join(a.out, f"{a.name}.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {a.out}/{a.name}.md")

if __name__ == "__main__":
    main()
