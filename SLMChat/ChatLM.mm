#import "ChatLM.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/sysctl.h>

#if __has_include(<llama/llama.h>)
#import <llama/llama.h>          // Xcode: resolved from the embedded llama.xcframework
#import <llama/ggml-backend.h>   // ggml_backend_dev_* (CPU-only device selection)
#import <llama/ggml-metal.h>     // ggml_backend_metal_supports_family (GPU pre-flight)
#else
#include "llama.h"               // standalone syntax check: -I .../Headers
#include "ggml-backend.h"
#include "ggml-metal.h"
#endif

// ---------------------------------------------------------------------------
// ChatLM is the app's single SLM bridge. It now runs on llama.cpp (GGUF models)
// instead of LiteRT-LM, but keeps the exact same Obj-C surface (see ChatLM.h) so
// the Swift side (SLMEngine and every caller) is untouched.
//
// Unlike the reference LlamaBridge — which tokenizes a single user turn — ChatLM
// builds a full role-tagged conversation (optional system + prior turns + the
// latest user turn) and renders it through the model's embedded chat template
// via llama_chat_apply_template. So the model sees proper multi-turn structure,
// matching the behaviour the LiteRT-LM ChatLM had. Single-turn tasks (translate /
// suggest / summarize) just pass system:nil, history:[].
// ---------------------------------------------------------------------------

namespace {

// --- KV cache -----------------------------------------------------------------
// ONE llama sequence, cleared at the start of every generation. No prompt prefix
// is carried between calls, so every call pays a full prefill — the price of a
// cache with no reuse rules to get wrong (in particular, K/V computed under one
// LoRA adapter can never be read under another).
constexpr int kCtx = 1024;  // prompt + generated tokens for one call

// llama.cpp abort callback: polled during llama_decode, so -cancel can stop a
// long prompt pass part-way instead of waiting for it to finish.
bool ChatLMShouldAbort(void *data) {
    return static_cast<std::atomic<bool> *>(data)->load(std::memory_order_relaxed);
}

// One-time global backend init (ggml/Metal registration).
void EnsureBackend() {
    static dispatch_once_t once;
    dispatch_once(&once, ^{ llama_backend_init(); });
}

// Would Metal actually be FASTER than the CPU on this device?
//
// "Metal is available" is a different question. ggml's Metal backend gates its
// kernels on GPU family, and there are three tiers:
//
//   has_simdgroup_reduction = supportsFamily:Apple7 || Metal3   (A13+)
//   has_simdgroup_mm        = supportsFamily:Apple7             (A14+)
//
//   A14+ (Apple7)  every op on the GPU + batched matmul kernels — the full win
//   A13  (Metal3)  every op on the GPU, no matmul kernels — a smaller win
//   A12- (Apple5)  SOFT_MAX, NORM, RMS_NORM, MEAN, GROUP_NORM all UNSUPPORTED
//
// The bottom tier is the one to avoid: those ops run in every transformer layer,
// so the scheduler splits the graph back to the CPU dozens of times per forward
// pass, copying tensors at each boundary — reliably slower than staying on the
// CPU throughout. An iPhone XS (A12) sits there. This check says so before a
// model is loaded, rather than after someone reports "GPU mode is slower".
//
// (Aside: those same devices fail on iOS 18 residency sets — ggml's Metal buffer
// alloc returns NULL and the load fails outright. The escape hatch is the
// GGML_METAL_NO_RESIDENCY env var, read once at Metal device init. It isn't set
// here because the family gate already keeps them off the GPU, and disabling
// residency sets would penalise the devices where Metal does work.)

// ggml's own value for MTLGPUFamilyMetal3 (A13 / iPhone 11 and later).
constexpr int kMTLGPUFamilyMetal3 = 5001;

bool MetalIsWorthUsing() {
    ggml_backend_t b = ggml_backend_metal_init();
    if (!b) return false;
    // Mirror ggml's own has_simdgroup_reduction test — Apple7 OR Metal3 — because
    // that flag is what decides whether the core ops run on the GPU at all. A13
    // passes via Metal3: every op is supported, only the batched matmul kernels
    // (Apple7-only) are missing, so it's a smaller win rather than a loss.
    const bool ok = ggml_backend_metal_supports_family(b, 7) ||
                    ggml_backend_metal_supports_family(b, kMTLGPUFamilyMetal3);
    ggml_backend_free(b);
    return ok;
}

int SysctlInt(const char *name) {
    int v = 0; size_t sz = sizeof(v);
    if (sysctlbyname(name, &v, &sz, nullptr, 0) != 0) return -1;
    return v;
}

// Log gate for +[ChatLM setLogLevel:]. A ggml message prints only if its level
// is >= this threshold. Defaults to INFO so ordinary runs stay as they were.
std::atomic<int> gGgmlMinLevel{GGML_LOG_LEVEL_INFO};

void ChatLMLogCallback(ggml_log_level level, const char *text, void * /*ud*/) {
    // CONT continues the previous line; let it through so multi-part lines aren't
    // split. Otherwise gate on the configured minimum level.
    if (level == GGML_LOG_LEVEL_CONT || (int)level >= gGgmlMinLevel.load()) {
        if (text) fputs(text, stderr);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Per-generation metrics (see ChatLM.h). Readonly in the header; redeclared
// readwrite here so the bridge can fill them.
@interface ChatLMTurnMetrics ()
@property (nonatomic, readwrite) double timeToFirstTokenSeconds;
@property (nonatomic, readwrite) double decodeTokensPerSecond;
@property (nonatomic, readwrite) double totalSeconds;
@property (nonatomic, readwrite, copy) NSString *stopReason;
@end
@implementation ChatLMTurnMetrics
@end

// ---------------------------------------------------------------------------

@implementation ChatLM {
    llama_model       *_model;
    llama_context     *_ctx;
    const llama_vocab *_vocab;
    std::string        _chatTmpl;   // model's embedded chat template (Gemma / Qwen)
    std::string        _lastError;
    bool               _isThinking; // Qwen3/Bonsai-style reasoning model
    int                _nThreads;
    ChatLMTurnMetrics *_lastTurnMetrics;   // metrics of the last run
    llama_batch        _batch;             // reusable batch (capacity = n_batch)
    int                _batchCapacity;     // how many tokens _batch can hold
    bool               _batchAllocated;
    std::atomic<bool>  _cancelRequested;   // set by -cancel from any thread
    NSString          *_activeBackend;     // what the load actually got
    // LoRA adapters, keyed by filesystem path and owned by this engine (each is
    // tied to _model's lifetime and freed in -dealloc).
    //
    // Plural because the adapter is now a per-CALL choice, not a per-load one:
    // auto-suggest asks for the pack matching the language the conversation is
    // being held in, while translate / summarize / event stay on the user's own
    // language. Loading is lazy — an adapter enters this map the first time some
    // call asks for it — so a session that never leaves one language pays for one
    // adapter, and the map grows only with languages actually used.
    //
    // Swapping is legitimate BECAUSE it goes through llama_set_adapters_lora,
    // which acts on the context and is honoured at graph-build time, and because
    // the KV cache is cleared per call, so no K/V outlives its adapter.
    std::unordered_map<std::string, llama_adapter_lora *> _adapters;
    NSString          *_defaultLoRAPath;   // adapter from the load; nil = bare
    NSString          *_activeLoRAPath;    // what the load actually got, nil = bare
    NSString          *_currentLoRAPath;   // what is set on the context right now
}

@synthesize lastTurnMetrics = _lastTurnMetrics;

+ (void)setLogLevel:(int)level {
    // Map the LiteRT-LM level convention onto ggml's, then install a gating
    // callback. Lower LiteRT level = more verbose.
    int minLevel;
    if      (level <= 1) minLevel = GGML_LOG_LEVEL_DEBUG; // VERBOSE/DEBUG → all
    else if (level == 2) minLevel = GGML_LOG_LEVEL_INFO;
    else if (level == 3) minLevel = GGML_LOG_LEVEL_WARN;
    else if (level == 4) minLevel = GGML_LOG_LEVEL_ERROR;
    else                 minLevel = 100;                  // FATAL/SILENT → none
    gGgmlMinLevel.store(minLevel);
    llama_log_set(ChatLMLogCallback, nullptr);
}

- (instancetype)init {
    if ((self = [super init])) {
        _model = nullptr;
        _ctx   = nullptr;
        _vocab = nullptr;
        _isThinking = false;
        _batchAllocated = false;
        _cancelRequested.store(false);
        // Pin inference to the PERFORMANCE-core cluster only. Batch-1 decode is
        // memory-bound; spilling onto efficiency cores barely raises tok/s but
        // pushes CPU usage very high and wastes power/thermal. perflevel0 = the
        // P-core cluster. Fall back to ~half the logical cores if absent.
        int perf = SysctlInt("hw.perflevel0.logicalcpu");
        if (perf <= 0) {
            unsigned hw = std::thread::hardware_concurrency();
            perf = (int)std::max(1u, std::min(4u, (hw ? hw : 4u) / 2u));
        }
        _nThreads = std::max(1, perf);
    }
    return self;
}

- (void)dealloc {
    if (_batchAllocated) llama_batch_free(_batch);
    if (_ctx)   llama_free(_ctx);
    // Free the adapters BEFORE the model: each is only valid while the model it
    // was initialised from is alive (llama.cpp would otherwise free them for us
    // with the model, but doing it explicitly keeps the ordering obvious).
    // Entries may be null — -adapterAtPath: caches failures so a broken pack
    // isn't re-parsed on every call.
    for (auto &kv : _adapters) { if (kv.second) llama_adapter_lora_free(kv.second); }
    _adapters.clear();
    if (_model) llama_model_free(_model);
}

- (BOOL)loadWithModelPath:(NSString *)modelPath
                  backend:(NSString *)backend
                 loraPath:(nullable NSString *)loraPath {
    EnsureBackend();
    _activeLoRAPath     = nil;
    _lastError.clear();

    const bool wantGpu = [backend isEqualToString:@"gpu"];
    const bool canGpu  = llama_supports_gpu_offload();
    // ggml's Metal backend aborts (SIGABRT) on init failure rather than returning
    // an error, so a forced attempt can't be caught. Force CPU on the simulator
    // (no real Metal compute device) and gate device use on gpu-offload support.
#if TARGET_OS_SIMULATOR
    const bool useGpu = false;
#else
    // Ask whether the GPU is worth using, not merely whether it exists — see
    // MetalIsWorthUsing(). Only probed when the GPU was actually requested, so a
    // CPU load never spins up Metal.
    bool useGpu = wantGpu && canGpu;
    if (useGpu && !MetalIsWorthUsing()) {
        NSLog(@"[ChatLM] GPU declined: this GPU is below MTLGPUFamilyMetal3 (A13), "
              @"so ggml reports softmax/RMS-norm as unsupported and would split "
              @"the graph back to the CPU every layer — Metal would be SLOWER "
              @"here. Using CPU.");
        useGpu = false;
    }
#endif
    _activeBackend = useGpu ? @"gpu" : @"cpu";

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = useGpu ? 999 : 0;   // 999 = offload all transformer layers
    mp.load_mode    = LLAMA_LOAD_MODE_MMAP; // mmap weights (keeps phys_footprint low)
    // Disable the CPU "repack" extra buffer. By default llama.cpp re-lays-out the
    // quantized weights into an ARM-optimized layout in a SECOND, dirty ~model-
    // sized allocation on top of the mmap'd originals — nearly DOUBLING resident
    // RAM (e.g. 272 MB mmap + 272 MB repack). That was the big regression vs
    // LiteRT-LM. Turning it off keeps only the mmap'd (clean, pageable) copy. The
    // repacked GEMM kernels also need dotprod/i8mm, which the A12 (iPhone XS)
    // target lacks anyway (see build_llama_xcframework.sh), so little is lost.
    mp.use_extra_bufts = false;

    // CPU mode: run on the CPU device ALONE — no Metal, and no BLAS/Accelerate.
    //
    // Excluding BLAS is counter-intuitive (it was kept here specifically "for the
    // batched prefill matmul") but measured, on the shipping Q8_0 checkpoint:
    //
    //     BLAS on    prefill 105 tok/s   ratio 1.5x   graph splits 282
    //     BLAS off   prefill 366 tok/s   ratio 4.5x   graph splits 1
    //
    // Accelerate implements matmul only, and only above a batch threshold. Every
    // other op — norms, softmax, RoPE, activations, and each small LoRA matmul —
    // stays on the CPU backend, so at prefill batch sizes the scheduler alternates
    // CPU->BLAS->CPU through all 28 layers: 282 splits, each a sync point plus
    // tensor copies. BLAS also needs f32, so quantized weights are dequantized
    // into scratch on every matmul, discarding the point of the quantization.
    //
    // At bs=1 the batch is under BLAS's threshold, so it never engages and decode
    // sees 1 split — which is exactly why this hurt prefill only, and why the
    // prefill:decode ratio (not the raw rate) is what exposes it.
    //
    // The same argument is why Metal is excluded here; the original comment named
    // "graph-split thrash" as the reason without noticing BLAS causes it too.
    // (devs must stay alive through llama_model_load_from_file below.)
    std::vector<ggml_backend_dev_t> devs;
    if (!useGpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            auto t = ggml_backend_dev_type(d);
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU)
                continue;  // skip Metal
            if (t == GGML_BACKEND_DEVICE_TYPE_ACCEL)
                continue;  // skip BLAS/Accelerate — see above
            devs.push_back(d);
        }
        devs.push_back(nullptr);  // NULL-terminate
        mp.devices = devs.data();
    }

    // Which SIMD paths this build actually has, ONCE per process. Printed because
    // prefill throughput is decided by them and nothing else in the log reveals
    // them: DOTPROD (SDOT, 16 int8 MACs/instruction) and MATMUL_INT8 (SMMLA, i8mm)
    // are what make a batched int8 GEMM fast. A "1" means the fast path exists on
    // this device; a "0" means every int8 dot product is widened to int16 and run
    // several instructions per MAC, which costs prefill roughly 4x and costs
    // decode nothing — decode never becomes compute-bound.
    //
    // Do NOT reason about a device's prefill ceiling without reading this line
    // first. It is the difference between measuring and guessing.
    static dispatch_once_t sysInfoOnce;
    dispatch_once(&sysInfoOnce, ^{
        NSLog(@"[SLMPerf] llama system info: %s", llama_print_system_info());
    });

    _model = llama_model_load_from_file(modelPath.UTF8String, mp);
    if (!_model) { _lastError = "llama_model_load_from_file failed"; return NO; }

    llama_context_params cp = llama_context_default_params();
    // Cap context to keep the KV cache small on 4 GB devices (iPhone XS). The
    // feature prompts chunk their input to stay well under this budget.
    // Only the KV cache scales with this; the compute buffer follows n_ubatch.
    cp.n_ctx           = kCtx;
    cp.n_seq_max       = 1;
    // Split logical vs physical batch to bound the compute buffer. The ggml
    // scheduler reserves a worst-case graph sized by n_ubatch, whose dominant
    // term is the [n_ubatch × n_vocab] f32 logits tensor. With Gemma-3's 262k
    // vocab, n_ubatch=512 reserved ~516 MiB. Dropping n_ubatch to 128 gives a
    // ~130 MiB compute buffer; prefill just runs a few more passes, and decode
    // (1 token) is unaffected.
    //
    // n_batch = n_ctx, so a full prompt always fits one llama_decode call (split
    // into ubatches internally). -decode:count: refuses anything over the batch
    // capacity, so a batch smaller than the context would reject prompts the
    // context happily allows.
    cp.n_batch         = cp.n_ctx;
    cp.n_ubatch        = 128;
    cp.n_threads       = _nThreads;
    cp.n_threads_batch = _nThreads;
    // Keep llama.cpp's perf counters ON in every build:
    // they give ChatLMTurnMetrics its decode rate. The cost is a couple of clock
    // reads per decode.
    cp.no_perf         = false;

    _ctx = llama_init_from_model(_model, cp);
    if (!_ctx) {
        _lastError = "llama_init_from_model failed";
        llama_model_free(_model);
        _model = nullptr;
        return NO;
    }
    // _cancelRequested lives as long as self, which outlives _ctx (-dealloc).
    llama_set_abort_callback(_ctx, ChatLMShouldAbort, &_cancelRequested);

    // --- LoRA adapter -------------------------------------------------------
    // Applied once, here, while the KV cache is still empty — the only moment
    // it's free. See the loraPath: doc in ChatLM.h for why it never changes
    // again.
    //
    // A bad adapter degrades to the bare base rather than killing the load: the
    // model is perfectly usable without it (that IS the shipping configuration
    // for languages the base already handles), and a language pack that fails to
    // parse should cost quality, not the whole feature set. Callers read
    // -activeLoRAPath to learn which they got.
    _defaultLoRAPath = loraPath.length > 0 ? [loraPath copy] : nil;
    if (_defaultLoRAPath && [self activateAdapterPath:_defaultLoRAPath]) {
        _activeLoRAPath = _defaultLoRAPath;
    } else {
        // Asked for nothing, or the pack wouldn't load — either way the engine
        // runs bare and -activeLoRAPath reports the truth.
        _defaultLoRAPath = nil;
    }

    // A batch big enough for a whole prompt pass, reused for every decode.
    _batch = llama_batch_init((int32_t)cp.n_batch, /*embd=*/0, /*n_seq_max=*/1);
    _batchCapacity  = (int)cp.n_batch;
    _batchAllocated = true;

    // Don't trust the requested size — the header warns the context may differ
    // from what was asked. A budget smaller than a feature's prompt would
    // silently truncate conversations, so surface it.
    const uint32_t ctx = llama_n_ctx(_ctx);
    if (ctx < cp.n_ctx) {
        NSLog(@"[ChatLM] context is %u tokens, expected %u — long prompts may not fit",
              ctx, cp.n_ctx);
    }

    _vocab = llama_model_get_vocab(_model);
    const char *tmpl = llama_model_chat_template(_model, nullptr);
    if (tmpl) _chatTmpl = tmpl;
    // Reasoning models (Qwen3/Bonsai) wrap answers in <think>...</think>. Their
    // jinja template ends the generation prompt with an empty think block to skip
    // reasoning, but llama_chat_apply_template (non-jinja) drops it — so the model
    // reasons anyway. Detect such models so the tokenizer can re-add it.
    // Keyed on `enable_thinking`, the switch that emits that block, not on
    // `<think>` alone: LFM2.5's template mentions `<think>` only to strip it from
    // past turns, and appending the block makes it emit EOS immediately.
    _isThinking = _chatTmpl.find("enable_thinking") != std::string::npos;
    return YES;
}

// Length of the longest prefix of `s` that does not end inside a UTF-8 sequence.
// Byte-level BPE vocabularies (LFM2, Qwen) can split one character — a
// Vietnamese diacritic, say — across two tokens; converting such a piece on its
// own makes -stringWithUTF8String: return nil.
static size_t CompleteUTF8Prefix(const std::string &s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        const unsigned char c = (unsigned char)s[n - back];
        if ((c & 0xC0) == 0x80) continue;           // continuation byte: keep looking
        const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > back ? n - back : n;          // lead byte: is its sequence whole?
    }
    return n;
}

// Map the app's role convention to what chat templates expect. History dicts use
// "model" (LiteRT-LM/Gemma naming); llama.cpp templates use "assistant".
static std::string NormalizeRole(NSString *role) {
    std::string r = role.UTF8String ?: "user";
    if (r == "model") return "assistant";
    return r;   // "user", "system", "assistant" pass through
}

// Build the full conversation (system + history + latest user turn), render it
// through the model's chat template, and tokenize. Mirrors the LiteRT-LM ChatLM's
// multi-turn conversation construction.
+ (int)contextTokens { return kCtx; }

- (int)tokenCountOf:(NSString *)text {
    if (!_vocab || text.length == 0) return 0;
    const char *utf8 = text.UTF8String;
    if (!utf8) return 0;
    const int len = (int)strlen(utf8);
    // Negative return is the required buffer size; ask with a zero-capacity buffer.
    return -llama_tokenize(_vocab, utf8, len, nullptr, 0, /*add_special=*/false,
                           /*parse_special=*/true);
}

- (std::vector<llama_token>)tokenizeSystem:(nullable NSString *)system
                                   history:(NSArray<NSDictionary<NSString *, NSString *> *> *)history
                                      user:(NSString *)user {
    // Backing storage kept alive for the char* pointers in llama_chat_message.
    std::vector<std::string> roles, texts;
    roles.reserve(history.count + 2);
    texts.reserve(history.count + 2);

    if (system.length > 0) {
        roles.emplace_back("system");
        texts.emplace_back(system.UTF8String ?: "");
    }
    for (NSDictionary<NSString *, NSString *> *m in history) {
        roles.push_back(NormalizeRole(m[@"role"] ?: @"user"));
        texts.emplace_back([(m[@"text"] ?: @"") UTF8String] ?: "");
    }
    roles.emplace_back("user");
    texts.emplace_back(user.UTF8String ?: "");

    std::vector<llama_chat_message> msgs(roles.size());
    size_t total = 0;
    for (size_t i = 0; i < roles.size(); ++i) {
        msgs[i] = llama_chat_message{ roles[i].c_str(), texts[i].c_str() };
        total += texts[i].size();
    }

    std::string formatted;
    if (!_chatTmpl.empty()) {
        std::vector<char> buf(total * 2 + 1024);
        int n = llama_chat_apply_template(_chatTmpl.c_str(), msgs.data(), msgs.size(),
                                          /*add_ass=*/true, buf.data(), (int)buf.size());
        if (n > (int)buf.size()) {                 // buffer too small — grow once
            buf.resize(n);
            n = llama_chat_apply_template(_chatTmpl.c_str(), msgs.data(), msgs.size(),
                                          true, buf.data(), (int)buf.size());
        }
        if (n > 0) formatted.assign(buf.data(), (size_t)n);
    }
    if (formatted.empty()) {                        // no template → plain concat
        for (size_t i = 0; i < roles.size(); ++i)
            formatted += roles[i] + ": " + texts[i] + "\n";
    }

    // Disable reasoning: replicate the template's no-think behavior by closing an
    // empty think block right after the assistant header. Idempotent.
    if (_isThinking && formatted.find("</think>") == std::string::npos) {
        formatted += "<think>\n\n</think>\n\n";
    }

    int need = -llama_tokenize(_vocab, formatted.c_str(), (int)formatted.size(),
                               nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
    std::vector<llama_token> toks(need > 0 ? need : 0);
    int n = llama_tokenize(_vocab, formatted.c_str(), (int)formatted.size(),
                           toks.data(), (int)toks.size(), true, true);
    if (n < 0) toks.clear(); else toks.resize((size_t)n);
    return toks;
}

// MARK: - LoRA adapters

// The adapter at `path`, loading it on first use. nullptr on failure — callers
// treat that as "run bare" rather than as an error, matching the load-time
// behaviour: a language pack that won't parse should cost quality, not the
// feature.
//
// Failures are cached as nullptr deliberately. A broken pack would otherwise be
// re-parsed (tens of MB off disk) on every single call that asks for it.
- (llama_adapter_lora *)adapterAtPath:(NSString *)path {
    return [self adapterAtPath:path reason:@"in-call"];
}

// `reason` only tags the log line, and it is the whole point of the line: an
// init that happens under "prewarm" cost the user nothing, while the same
// milliseconds under "in-call" came straight out of a latency budget. Averaging
// the two together hides exactly the thing worth watching.
- (llama_adapter_lora *)adapterAtPath:(NSString *)path reason:(NSString *)reason {
    const std::string key = path.UTF8String;
    auto it = _adapters.find(key);
    if (it != _adapters.end()) return it->second;

    // Timed because this is the ONE expensive step in adapter switching, it runs
    // on the engine queue inside whatever call first asked for the language, and
    // auto-suggest — the only caller that ever switches — has a 3 s budget. Every
    // later switch to the same pack is a map hit.
    const auto t0 = std::chrono::steady_clock::now();
    llama_adapter_lora *a = llama_adapter_lora_init(_model, key.c_str());
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!a) {
        NSLog(@"[ChatLM] LoRA adapter failed to load, will run bare: %@",
              path.lastPathComponent);
    }
    NSLog(@"[SLMPerf] lora init %@ → %@ in %.3fs (%@, first use this session)",
          path.lastPathComponent, a ? @"ok" : @"FAILED", seconds, reason);
    _adapters[key] = a;   // negative caching: don't retry a bad file every call
    return a;
}

// Make `path` the adapter in force on the context (nil = bare weights).
// Returns whether an adapter is now applied.
//
// Cheap to call on every generation: llama_set_adapters_lora no-ops when the set
// already matches, and the adapter itself is parsed once by -adapterAtPath:.
// This does NOT touch the KV cache — -runSystem: clears it before every call,
// so no prefix computed under a different adapter is ever reused.
- (BOOL)activateAdapterPath:(nullable NSString *)path {
    if (path.length == 0) {
        if (_currentLoRAPath == nil) return NO;
        llama_set_adapters_lora(_ctx, nullptr, 0, nullptr);
        _currentLoRAPath = nil;
        return NO;
    }
    if ([path isEqualToString:_currentLoRAPath]) return YES;

    const auto t0 = std::chrono::steady_clock::now();
    const bool cached = _adapters.find(path.UTF8String) != _adapters.end();
    llama_adapter_lora *a = [self adapterAtPath:path];
    if (!a) return [self activateAdapterPath:nil];

    llama_adapter_lora *adapters[1] = { a };
    float scales[1] = { 1.0f };
    if (llama_set_adapters_lora(_ctx, adapters, 1, scales) != 0) {
        NSLog(@"[ChatLM] llama_set_adapters_lora failed for %@ — running bare",
              path.lastPathComponent);
        return [self activateAdapterPath:nil];
    }
    _currentLoRAPath = [path copy];
    // Split the two costs in the log: a warm switch should be microseconds (it
    // only rewrites a map on the context), a cold one carries the init above.
    NSLog(@"[SLMPerf] lora switch → %@ in %.3fs (%@)", path.lastPathComponent,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
          cached ? @"warm" : @"COLD, included init");
    return YES;
}

// Load a language pack into this engine WITHOUT making it current.
//
// Adapter switching is cheap except the first time a pack is touched, and that
// one parse lands inside whichever generation asked for it — for auto-suggest,
// inside a 3 s budget. Calling this when a conversation opens moves the cost off
// that path: by the time suggestions run, -activateAdapterPath: is a map hit.
//
// Safe to call repeatedly and for a pack already loaded (it returns immediately),
// and it does not disturb the KV cache or the adapter currently in force.
- (void)prewarmAdapterAtPath:(NSString *)path {
    if (!_model || path.length == 0) return;
    if (_adapters.find(path.UTF8String) != _adapters.end()) return;
    [self adapterAtPath:path reason:@"prewarm"];
}

// Decode `count` tokens starting at position `pos`, asking for logits on the
// last one only (that's all the sampler reads).
- (bool)decode:(const llama_token *)tokens
         count:(int)count
         atPos:(llama_pos)pos {
    // Refuse rather than overrun. _batch holds _batchCapacity tokens; writing
    // past it corrupts four heap allocations at once (token/pos/n_seq_id/logits)
    // and the process dies somewhere unrelated, with no hint of the cause. That
    // is not theoretical — a 1120-token prompt against a 1024 batch killed the
    // test runner silently while measuring prefill throughput.
    //
    // Callers are supposed to keep prompts under kCtx (Prompts chunks and
    // windows for exactly this reason), so reaching here means an upstream bug.
    // Failing loudly is what makes that bug findable.
    if (count > _batchCapacity || count <= 0) {
        _lastError = "batch overflow: " + std::to_string(count) +
                     " tokens exceeds capacity " + std::to_string(_batchCapacity);
        NSLog(@"[ChatLM] %s", _lastError.c_str());
        return false;
    }
    _batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        _batch.token[i]     = tokens[i];
        _batch.pos[i]       = pos + i;
        _batch.n_seq_id[i]  = 1;
        _batch.seq_id[i][0] = 0;
        _batch.logits[i]    = (i == count - 1);
    }
    return llama_decode(_ctx, _batch) == 0;
}

// Shared decode loop for both blocking and streaming generation. Stateless per
// call: the conversation is rebuilt from scratch and the KV cache is cleared
// every time.
- (std::string)runSystem:(nullable NSString *)system
                 history:(NSArray<NSDictionary<NSString *, NSString *> *> *)history
                    user:(NSString *)user
               maxTokens:(int)maxTokens
                    temp:(float)temp
             adapterPath:(nullable NSString *)adapterPath
                 onChunk:(void (^ _Nullable)(const std::string &, bool))onChunk {
    if (!_ctx) { _lastError = "engine not loaded"; if (onChunk) onChunk("", true); return ""; }

    // Put the requested adapter in force before anything is tokenized or decoded.
    // nil means "whatever the load asked for" — the overwhelmingly common case,
    // and the one every feature except auto-suggest takes. A pack that fails to
    // parse leaves the context bare.
    [self activateAdapterPath:adapterPath ?: _defaultLoRAPath];

    llama_perf_context_reset(_ctx);   // metrics reflect only this run
    _lastTurnMetrics = nil;
    // A cancel only applies to the generation running when it was requested.
    _cancelRequested.store(false);
    _lastError.clear();

    // t0 starts before templating/tokenizing so totalSeconds and TTFT are the
    // latency the caller actually observes, not just the model's share of it.
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [t0] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    std::vector<llama_token> prompt_tokens = [self tokenizeSystem:system
                                                          history:history
                                                             user:user];
    const int promptBudget = kCtx;
    if (!prompt_tokens.empty() && (int)prompt_tokens.size() > promptBudget) {
        // Caught here rather than at -decode: so the message names the prompt,
        // not the batch. A prompt this long means Prompts' chunking or history
        // window let something through; truncating would hide that and produce
        // quietly wrong output instead.
        NSLog(@"[ChatLM] prompt is %zu tokens, over the %d-token context "
              @"— refusing. Chunk or shorten it upstream.",
              prompt_tokens.size(), promptBudget);
        prompt_tokens.clear();
    }
    if (prompt_tokens.empty()) {
        _lastError = "tokenize failed or prompt too long";
        ChatLMTurnMetrics *m = [[ChatLMTurnMetrics alloc] init];
        m.totalSeconds = elapsed();
        m.stopReason = @"error";
        _lastTurnMetrics = m;
        if (onChunk) onChunk("", true);
        return "";
    }

    // Start from an empty KV cache: the whole prompt is evaluated every call.
    llama_memory_clear(llama_get_memory(_ctx), /*data=*/true);

    // Sampler chain: greedy when temp<=0, else top_k -> top_p -> temp -> dist.
    // top_k/top_p match the values the LiteRT-LM ChatLM used (40 / 0.95).
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler *smpl = llama_sampler_chain_init(sp);
    if (temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    std::string full;
    std::string pendingUTF8;   // bytes of a character not yet completed by a later token
    char piece[256];
    llama_token cur = 0;
    // The first pass evaluates the whole prompt from position 0.
    const llama_token *next = prompt_tokens.data();
    int   nextCount         = (int)prompt_tokens.size();
    llama_pos nextPos       = 0;
    int generated = 0;
    bool firstToken = true;
    NSString *stopReason = @"max_tokens";
    double ttftSeconds = 0;
    // The first loop iteration is the prompt pass (nextCount = the whole prompt);
    // every later iteration decodes one token.
    while (generated < maxTokens) {
        if (_cancelRequested.load()) { stopReason = @"cancelled"; break; }
        if (![self decode:next count:nextCount atPos:nextPos]) {
            // An abort from -cancel surfaces as a failed decode; it is not an
            // error. The half-written KV cache is cleared by the next call.
            if (_cancelRequested.load()) { stopReason = @"cancelled"; break; }
            _lastError = "llama_decode failed";
            stopReason = @"error";
            break;
        }
        nextPos += nextCount;
        cur = llama_sampler_sample(smpl, _ctx, -1);
        if (llama_vocab_is_eog(_vocab, cur)) { stopReason = @"eos"; break; }

        int np = llama_token_to_piece(_vocab, cur, piece, sizeof(piece),
                                      /*lstrip=*/0, /*special=*/false);
        if (np > 0) {
            if (firstToken) {  // wall time to first token = prefill + first decode
                ttftSeconds = elapsed();
                firstToken = false;
            }
            pendingUTF8.append(piece, (size_t)np);
            const size_t whole = CompleteUTF8Prefix(pendingUTF8);
            if (whole > 0) {
                std::string chunk = pendingUTF8.substr(0, whole);
                pendingUTF8.erase(0, whole);
                full += chunk;
                if (onChunk) onChunk(chunk, false);
            }
        }
        ++generated;
        next      = &cur;    // &cur stays valid through the next decode
        nextCount = 1;
    }

    // Decode rate from llama.cpp's counters: n_eval / t_eval_ms cover the
    // single-token decode steps only (the first token falls out of the prompt
    // pass), so the prompt doesn't dilute it.
    llama_perf_context_data d = llama_perf_context(_ctx);
    ChatLMTurnMetrics *m = [[ChatLMTurnMetrics alloc] init];
    m.timeToFirstTokenSeconds = ttftSeconds;
    m.decodeTokensPerSecond   = d.t_eval_ms > 0 ? d.n_eval / (d.t_eval_ms / 1000.0) : 0;
    m.totalSeconds            = elapsed();
    m.stopReason              = stopReason;
    _lastTurnMetrics = m;

    if (onChunk) onChunk("", true);
    llama_sampler_free(smpl);
    return full;
}

- (void)streamSystem:(nullable NSString *)system
             history:(NSArray<NSDictionary<NSString *, NSString *> *> *)history
                user:(NSString *)user
           maxTokens:(int)maxTokens
         temperature:(float)temperature
             onChunk:(void (^)(NSString *, BOOL))onChunk {
    [self runSystem:system history:history user:user maxTokens:maxTokens temp:temperature
          adapterPath:nil
            onChunk:^(const std::string &chunk, bool done) {
                NSString *s = [NSString stringWithUTF8String:chunk.c_str()];
                onChunk(s ?: @"", done ? YES : NO);
            }];
}

- (nullable NSString *)generateSystem:(nullable NSString *)system
                              history:(NSArray<NSDictionary<NSString *, NSString *> *> *)history
                                 user:(NSString *)user
                            maxTokens:(int)maxTokens
                          temperature:(float)temperature
                          adapterPath:(nullable NSString *)adapterPath {
    if (!_ctx) return nil;
    std::string out = [self runSystem:system history:history user:user
                            maxTokens:maxTokens temp:temperature
                            adapterPath:adapterPath onChunk:nil];
    if (!_lastError.empty()) return nil;
    return [NSString stringWithUTF8String:out.c_str()];
}


- (void)cancel {
    _cancelRequested.store(true);
}

- (NSString *)lastError {
    return _lastError.empty() ? @"" : [NSString stringWithUTF8String:_lastError.c_str()];
}

- (NSString *)activeBackend {
    return _activeBackend ?: @"-";
}

@end
