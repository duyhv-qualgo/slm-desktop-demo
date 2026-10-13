#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Obj-C++ bridge that runs the text-guardrail BERT-Tiny classifier through the
/// TensorFlow Lite / LiteRT **C** interpreter API. Mirrors `ChatLM`, but for a
/// fixed-shape classifier instead of the generative engine: load once, then run
/// `classify` per message.
///
/// The TFLite symbols are provided by the already-embedded CLiteRTLM.framework
/// (see TFLiteC.h) — no separate TensorFlowLiteC framework is linked.
///
/// The model has two int32 inputs (`input_ids`, `attention_mask`, each [1,128])
/// and two float outputs (a 10-way class head and a 4-way severity head). This
/// bridge is dtype/shape agnostic: it routes inputs by tensor name and returns
/// every output head as a plain array of floats, so the Swift layer maps heads
/// by length (10 = class, 4 = severity).
@interface GuardLM : NSObject

/// Load the `.tflite` model and allocate tensors. Returns NO on failure
/// (`-lastError` for details). Safe to call once per engine instance.
- (BOOL)loadWithModelPath:(NSString *)modelPath
    NS_SWIFT_NAME(load(modelPath:));

/// Run one classification. `inputIds` and `attentionMask` are little-endian
/// int32 buffers, each 128 elements. Returns one inner array of floats per
/// output tensor (in tensor order), or nil on error. The caller maps heads by
/// count (10 → class logits, 4 → severity logits).
- (nullable NSArray<NSArray<NSNumber *> *> *)
    classifyWithInputIds:(NSData *)inputIds
           attentionMask:(NSData *)attentionMask
    NS_SWIFT_NAME(classify(inputIds:attentionMask:));

@property (nonatomic, readonly, copy) NSString *lastError;

@end

NS_ASSUME_NONNULL_END
