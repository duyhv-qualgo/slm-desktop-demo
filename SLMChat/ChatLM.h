#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Full timing/token breakdown for ONE generation, recorded on EVERY call
/// (`-generate…`, `-stream…`, `-benchmarkStream…`) — not just in the benchmark
/// harness. This is what answers "why did this turn take N seconds": how much of
/// it was prompt prefill vs token-by-token decode, how many tokens went in and
/// came out, and how much time was spent outside the model (templating,
/// tokenizing, sampling, callbacks).
///
/// Token counts and the prefill/decode split come from llama.cpp's own perf
/// counters (`llama_perf_context`), so they are exact — a streamed chunk is not
/// necessarily one token. `ttft`/`total` are wall clock around the whole call.
///
/// Read it off `ChatLM.lastTurnMetrics` right after a generation returns.
@interface ChatLMTurnMetrics : NSObject
/// Tokens fed to the model: the whole rendered conversation (system + history +
/// user turn after the chat template), NOT just the user's message.
/// Every call evaluates all of them: the KV cache is cleared per call.
@property (nonatomic, readonly) int    promptTokenCount;
/// Time inside llama_decode for the prompt pass (the batched prefill).
@property (nonatomic, readonly) double prefillSeconds;
@property (nonatomic, readonly) double prefillTokensPerSecond;
/// Wall time from the call starting to the first token being emitted —
/// templating + tokenize + prefill + first sample. What the user waits for
/// before any text appears.
@property (nonatomic, readonly) double timeToFirstTokenSeconds;
/// Tokens actually generated this turn.
@property (nonatomic, readonly) int    decodeTokenCount;
/// Time inside llama_decode for the single-token decode steps (excludes the
/// prefill pass, which already produced the first token).
@property (nonatomic, readonly) double decodeSeconds;
/// Generation speed, decode phase only — the tok/s figure worth quoting.
@property (nonatomic, readonly) double decodeTokensPerSecond;
/// Wall time for the whole call, end to end.
@property (nonatomic, readonly) double totalSeconds;
/// `totalSeconds` minus the two model phases: chat-template rendering,
/// tokenization, sampling and the caller's onChunk work. Large values here mean
/// the cost is NOT in the model.
@property (nonatomic, readonly) double overheadSeconds;
/// Characters of text produced (streamed chunks concatenated).
@property (nonatomic, readonly) int    outputCharCount;
/// Why generation stopped: `@"eos"`, `@"max_tokens"`, or `@"error"`.
@property (nonatomic, readonly, copy) NSString *stopReason;

/// Filename of the LoRA adapter this generation ran under, or `@"none"` for bare
/// weights. Not the engine's default — the adapter for THIS call, which for
/// auto-suggest is the conversation's language pack.
@property (nonatomic, readonly, copy) NSString *adapterName;
/// What it cost to put that adapter in force, in seconds. Part of
/// `overheadSeconds`, broken out because it is the one overhead term that varies
/// by three orders of magnitude between calls.
@property (nonatomic, readonly) double adapterSeconds;
/// How the adapter got there — the field that explains `adapterSeconds`:
///   `@"same"` — already in force. Costs nothing. The overwhelming majority.
///   `@"warm"` — switched to a pack already parsed this session. Microseconds.
///   `@"cold"` — parsed off disk first (tens of ms). Once per pack per session,
///               and `-prewarmAdapterAtPath:` exists to keep it off this path.
@property (nonatomic, readonly, copy) NSString *adapterState;

/// One-line, greppable summary of every field above.
@property (nonatomic, readonly, copy) NSString *summaryLine;
@end

/// Metrics llama.cpp itself recorded for one generation, read back via
/// `llama_perf_context`. These are measured INSIDE the runtime, so unlike
/// anything timed from Swift they are exact: real token counts (not decoded-
/// chunk counts) and a decode rate that excludes prefill/init.
///
/// Only populated when the engine was loaded with `enableBenchmark:YES`.
/// For per-turn app instrumentation use `ChatLMTurnMetrics` instead — it is
/// always recorded and carries strictly more detail.
@interface ChatLMBenchmarkInfo : NSObject
/// Wall time from decode start to the first emitted token (prefill + first
/// decode) — the latency the user actually waits before text appears.
@property (nonatomic, readonly) double timeToFirstTokenSeconds;
@property (nonatomic, readonly) int prefillTokenCount;
@property (nonatomic, readonly) double prefillTokensPerSecond;
/// Exact number of tokens decoded — the ground truth the streaming callback
/// can't give, since one chunk is not necessarily one token.
@property (nonatomic, readonly) int decodeTokenCount;
@property (nonatomic, readonly) double decodeTokensPerSecond;
@end

/// Activation precision knob carried over from the LiteRT-LM bridge so the
/// benchmark harness compiles unchanged. llama.cpp has no equivalent per-tensor
/// activation-precision setting, so on this runtime the value is ACCEPTED AND
/// IGNORED — every load behaves as `Auto`. Kept only for signature parity.
typedef NS_ENUM(NSInteger, ChatLMActivationType) {
    ChatLMActivationTypeAuto = -1,
    ChatLMActivationTypeF32  = 0,
    ChatLMActivationTypeF16  = 1,
    ChatLMActivationTypeI16  = 2,
    ChatLMActivationTypeI8   = 3,
};

/// Direct wrapper over llama.cpp's C API (llama.h) that supports a real system
/// prompt + role-tagged multi-turn history. Each call rebuilds the conversation
/// from an optional system message, a windowed array of prior turns, and the
/// latest user message, then renders it through the model's embedded chat
/// template via `llama_chat_apply_template`.
///
/// Single-turn tasks (translate / suggest / summarize) just pass a nil system
/// and an empty history.
@interface ChatLM : NSObject

/// Set llama.cpp's log verbosity. Uses the LiteRT-LM level convention the app
/// already speaks: 0=VERBOSE, 1=DEBUG, 2=INFO, 3=WARNING, 4=ERROR, 5=FATAL,
/// 1000=SILENT — mapped onto ggml's log levels. Process-wide.
+ (void)setLogLevel:(int)level NS_SWIFT_NAME(setLogLevel(_:));

/// Load a `.gguf` model. backend is @"cpu" or @"gpu" (Metal). cacheDir is
/// accepted for signature parity but unused — llama.cpp has no external kernel
/// cache. Returns NO on failure (-lastError for details).
- (BOOL)loadWithModelPath:(NSString *)modelPath
                  backend:(NSString *)backend
                 cacheDir:(nullable NSString *)cacheDir
    NS_SWIFT_NAME(load(modelPath:backend:cacheDir:));

/// Load a model together with a LoRA adapter (`loraPath`, a `.gguf` produced by
/// llama.cpp's `convert_lora_to_gguf.py`). Pass nil for the bare base.
///
/// This is the engine's DEFAULT adapter: the one every generation uses unless it
/// names another via `generate…adapterPath:`.
///
/// Per-call adapters are safe because the KV cache is cleared at the start of
/// every generation: no K/V computed under one adapter is ever read under
/// another.
///
/// A `loraPath` that fails to load is NOT fatal — the model stays usable on its
/// bare weights and `-activeLoRAPath` reports nil. Read that rather than
/// assuming the adapter took, exactly as with `-activeBackend`.
- (BOOL)loadWithModelPath:(NSString *)modelPath
                  backend:(NSString *)backend
                 cacheDir:(nullable NSString *)cacheDir
                 loraPath:(nullable NSString *)loraPath
    NS_SWIFT_NAME(load(modelPath:backend:cacheDir:loraPath:));

/// Load with benchmark instrumentation enabled — the benchmark harness only,
/// never the app's normal inference path.
///
/// `enableBenchmark` turns on llama.cpp's own perf timing, making
/// `-benchmarkStream…` return a `ChatLMBenchmarkInfo`.
///
/// `forcedDecodeTokens` > 0 pins the decode length: generation runs for EXACTLY
/// that many tokens and does not stop early on an EOS/stop token, so the decode
/// window is identical across models. Pass 0 to leave decoding natural (EOS
/// respected) while still collecting metrics.
///
/// `activation` is ignored on llama.cpp (see `ChatLMActivationType`).
///
/// `loraPath` behaves as in `-loadWithModelPath:backend:cacheDir:loraPath:` —
/// nil for the bare base. The benchmark sweep passes nil on the runs that
/// measure a checkpoint on its own, and an adapter path on the runs that measure
/// what the adapter costs.
- (BOOL)loadWithModelPath:(NSString *)modelPath
                  backend:(NSString *)backend
                 cacheDir:(nullable NSString *)cacheDir
                 loraPath:(nullable NSString *)loraPath
          enableBenchmark:(BOOL)enableBenchmark
       forcedDecodeTokens:(int)forcedDecodeTokens
               activation:(ChatLMActivationType)activation
    NS_SWIFT_NAME(load(modelPath:backend:cacheDir:loraPath:enableBenchmark:forcedDecodeTokens:activation:));

/// Streaming multi-turn generation. `history` is an ordered array of
/// @{@"role": @"user"|@"model", @"text": @"..."} dicts. `onChunk` fires per
/// token on the calling thread; the final call has done=YES. Blocks until
/// generation completes.
- (void)streamSystem:(nullable NSString *)system
             history:(NSArray<NSDictionary<NSString *, NSString *> *> *)history
                user:(NSString *)user
           maxTokens:(int)maxTokens
         temperature:(float)temperature
             onChunk:(void (^)(NSString *chunk, BOOL done))onChunk
    NS_SWIFT_NAME(stream(system:history:user:maxTokens:temperature:onChunk:));

/// Streaming generation that also returns the runtime's own metrics for the
/// turn. Identical to `-stream…` otherwise. Returns nil if the engine wasn't
/// loaded with `enableBenchmark:YES`.
- (nullable ChatLMBenchmarkInfo *)benchmarkStreamSystem:(nullable NSString *)system
                                                history:(NSArray<NSDictionary<NSString *, NSString *> *> *)history
                                                   user:(NSString *)user
                                              maxTokens:(int)maxTokens
                                            temperature:(float)temperature
                                                onChunk:(void (^)(NSString *chunk, BOOL done))onChunk
    NS_SWIFT_NAME(benchmarkStream(system:history:user:maxTokens:temperature:onChunk:));

/// Blocking multi-turn generation. Returns the full reply, or nil on error.
///
/// `adapterPath` selects the LoRA adapter for THIS call; nil means the one the
/// engine was loaded with, which is what every feature except auto-suggest wants.
/// Auto-suggest passes the pack for the language the conversation is being held
/// in, so its replies are written by a model adapted to that language while
/// translate / summarize / event stay on the user's own.
///
/// Switching is cheap: adapters are parsed once and cached, and
/// `llama_set_adapters_lora` no-ops when the set already matches.
///
/// An adapter that fails to load degrades to the bare base for that call rather
/// than failing it, exactly as the load-time `loraPath:` does.
- (nullable NSString *)generateSystem:(nullable NSString *)system
                              history:(NSArray<NSDictionary<NSString *, NSString *> *> *)history
                                 user:(NSString *)user
                            maxTokens:(int)maxTokens
                          temperature:(float)temperature
                          adapterPath:(nullable NSString *)adapterPath
    NS_SWIFT_NAME(generate(system:history:user:maxTokens:temperature:adapterPath:));

/// Load a language pack into this engine without making it current, so the one
/// expensive part of adapter switching — parsing the `.gguf` — happens off the
/// critical path instead of inside the generation that first asks for it.
///
/// A no-op for a pack already loaded, and it does not change the adapter
/// currently in force. Safe to call speculatively.
- (void)prewarmAdapterAtPath:(NSString *)path
    NS_SWIFT_NAME(prewarmAdapter(path:));

/// Tokens available to ONE generation: prompt plus everything it generates must
/// fit inside this. Exposed so callers size their prompts from the engine's real
/// budget instead of duplicating the number and letting it drift.
+ (int)contextTokens NS_SWIFT_NAME(contextTokens());

/// Token count of `text` as this model's vocabulary sees it, with no chat
/// template applied — the raw cost of a string.
///
/// Exists for prompt work: output tokens are the dominant term in a turn's decode
/// time, and "how many tokens is this phrasing?" is otherwise unanswerable without
/// guessing from character counts (which mislead badly — Vietnamese diacritics and
/// bracket sequences like `[[INTENT]]` both tokenize far worse than plain ASCII).
/// Returns 0 before a model is loaded.
- (int)tokenCountOf:(NSString *)text NS_SWIFT_NAME(tokenCount(of:));

/// Threads for the PROMPT pass (llama.cpp `n_threads_batch`). Decode keeps the
/// P-core count: batch-1 decode is memory-bound and gains nothing from E-cores.
/// Prefill is compute-bound, so the answer may differ — this exists to measure
/// that (`PrefillThreadsProbe`). Read-back is the value in effect; 0 restores
/// the default. Takes effect at the next load.
@property (nonatomic) int batchThreads;

@property (nonatomic, readonly, copy) NSString *lastError;

/// The backend actually in use after a successful load: `@"gpu"` or `@"cpu"`.
/// A `@"gpu"` request is served on CPU when the platform can't honour it — the
/// simulator (no Metal compute device) or a build without gpu-offload support —
/// so read this rather than assuming what was asked for. `@"-"` before a load.
@property (nonatomic, readonly, copy) NSString *activeBackend;

/// Filesystem path of this engine's DEFAULT LoRA adapter, or nil when it runs on
/// the bare base — either because no adapter was asked for, or because the one
/// that was asked for failed to load. Fixed for the lifetime of the load.
///
/// Deliberately not "the adapter currently set on the context": that changes per
/// call now (see `generate…adapterPath:`), and callers use this to decide whether
/// a reload is needed after a pack is installed or removed. Reporting the
/// transient value would make an auto-suggest in a foreign language look like a
/// pending reload.
@property (nonatomic, readonly, copy, nullable) NSString *activeLoRAPath;

/// Timing/token breakdown of the most recent generation on this engine, or nil
/// if none has run. Recorded for every call, benchmark or not — see
/// `ChatLMTurnMetrics`. Valid until the next generation starts; the engine is
/// driven from one serial queue (`SLMEngine`), so reading it right after a call
/// returns is safe.
@property (nonatomic, readonly, nullable) ChatLMTurnMetrics *lastTurnMetrics;

@end

NS_ASSUME_NONNULL_END
