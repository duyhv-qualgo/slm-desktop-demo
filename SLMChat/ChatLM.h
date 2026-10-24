#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// What one generation cost, recorded on every call (`-generate…`, `-stream…`).
/// Four numbers, all content-free: how long the user waited, how fast text came,
/// the whole call, and why it stopped. Read it off `ChatLM.lastTurnMetrics`
/// right after a generation returns.
@interface ChatLMTurnMetrics : NSObject
/// Wall time from the call starting to the first token being emitted —
/// templating + tokenize + prefill + first sample. What the user waits for
/// before any text appears. 0 if no token was produced.
@property (nonatomic, readonly) double timeToFirstTokenSeconds;
/// Generation speed, decode phase only (excludes the prompt pass), from
/// llama.cpp's own counters — the tok/s figure worth quoting.
@property (nonatomic, readonly) double decodeTokensPerSecond;
/// Wall time for the whole call, end to end.
@property (nonatomic, readonly) double totalSeconds;
/// Why generation stopped: `@"eos"`, `@"max_tokens"`, `@"cancelled"` (see
/// `-cancel`), or `@"error"`. Frequent `max_tokens` means replies are being cut
/// off.
@property (nonatomic, readonly, copy) NSString *stopReason;
@end

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

/// Load a `.gguf` model, optionally with a LoRA adapter (`loraPath`, a `.gguf`
/// produced by llama.cpp's `convert_lora_to_gguf.py`; nil for the bare base).
/// `backend` is @"cpu" or @"gpu" (Metal). Returns NO on failure (-lastError for
/// details).
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
                 loraPath:(nullable NSString *)loraPath
    NS_SWIFT_NAME(load(modelPath:backend:loraPath:));

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

/// Stop the generation that is running right now, as soon as possible: between
/// tokens, or part-way through the prompt pass. Safe to call from ANY thread —
/// it exists to be called while the engine's own thread is blocked inside
/// `-generate…` / `-stream…`.
///
/// The cancelled call returns normally: `onChunk` still gets its final
/// `done=YES`, `-generate…` returns the text produced so far (not nil), and
/// `lastTurnMetrics.stopReason` is `@"cancelled"`. With nothing running it does
/// nothing — it never cancels a generation that starts later.
- (void)cancel;

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

/// Metrics of the most recent generation on this engine, or nil if none has
/// run — see `ChatLMTurnMetrics`. Valid until the next generation starts; the engine is
/// driven from one serial queue (`SLMEngine`), so reading it right after a call
/// returns is safe.
@property (nonatomic, readonly, nullable) ChatLMTurnMetrics *lastTurnMetrics;

@end

NS_ASSUME_NONNULL_END
