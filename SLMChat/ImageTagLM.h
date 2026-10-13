#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Obj-C++ bridge that runs the EdgeNeXt image-tagging model through the
/// TensorFlow Lite / LiteRT **C** interpreter API. Mirrors `GuardLM`, but for a
/// vision classifier: load once, then run `tag` per image.
///
/// The TFLite symbols are provided by the already-embedded CLiteRTLM.framework
/// (see TFLiteC.h) — no separate TensorFlowLiteC framework is linked.
///
/// The model has a single float32 image input (`1×224×224×3` NHWC or `1×3×224×224`
/// NCHW, depending on the export) and a single float32 output of per-tag scores
/// (~4585, one per RAM tag). Depending on how the model was exported, those scores
/// are either raw logits or already sigmoid probabilities; the Swift layer detects
/// which from the value range. This bridge is layout-agnostic: it exposes the input
/// tensor's shape via `-inputShape` so the Swift layer can lay the pixel buffer out
/// correctly, copies that buffer straight into the input tensor, and returns the
/// raw output scores unchanged.
@interface ImageTagLM : NSObject

/// Load the `.tflite` model and allocate tensors. Returns NO on failure
/// (`-lastError` for details). Safe to call once per engine instance.
- (BOOL)loadWithModelPath:(NSString *)modelPath
    NS_SWIFT_NAME(load(modelPath:));

/// The input tensor's dimensions (e.g. `[1, 224, 224, 3]`), resolved at load.
/// Empty until `load` succeeds. Lets the Swift layer detect NHWC vs NCHW and the
/// spatial size without hard-coding them.
@property (nonatomic, readonly, copy) NSArray<NSNumber *> *inputShape;

/// Number of output logits (tags) the model emits, resolved at load. 0 until loaded.
@property (nonatomic, readonly) NSInteger tagCount;

/// Run one forward pass. `pixels` is a little-endian float32 buffer whose length
/// must equal the product of `-inputShape` (i.e. `1*H*W*3`), already normalized and
/// laid out to match the input tensor. Returns the raw output **logits** as floats,
/// or nil on error. The caller applies `sigmoid` + thresholding.
- (nullable NSArray<NSNumber *> *)tagWithPixels:(NSData *)pixels
    NS_SWIFT_NAME(tag(pixels:));

@property (nonatomic, readonly, copy) NSString *lastError;

@end

NS_ASSUME_NONNULL_END
