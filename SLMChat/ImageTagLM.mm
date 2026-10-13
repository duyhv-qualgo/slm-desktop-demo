#import "ImageTagLM.h"
#import "TFLiteC.h"

#include <string>

// ---------------------------------------------------------------------------
// ImageTagLM: a thin Obj-C++ wrapper over the TFLite C interpreter API, matching
// GuardLM's shape. The model is loaded once and the interpreter is reused for
// every `tag` call. The TFLite interpreter is NOT thread-safe; callers
// (ImageTagEngine) serialize access on a single queue.
// ---------------------------------------------------------------------------

@implementation ImageTagLM {
    TfLiteModel*       _model;
    TfLiteInterpreter* _interpreter;
    std::string        _lastError;
    NSArray<NSNumber *>* _inputShape;
    NSInteger          _tagCount;
}

- (instancetype)init {
    if ((self = [super init])) {
        _model = nullptr;
        _interpreter = nullptr;
        _inputShape = @[];
        _tagCount = 0;
    }
    return self;
}

- (void)dealloc {
    if (_interpreter) TfLiteInterpreterDelete(_interpreter);
    if (_model) TfLiteModelDelete(_model);
}

- (BOOL)loadWithModelPath:(NSString *)modelPath {
    if (_interpreter) return YES;  // already loaded

    _model = TfLiteModelCreateFromFile(modelPath.UTF8String);
    if (!_model) {
        _lastError = "TfLiteModelCreateFromFile failed (bad path or corrupt .tflite)";
        return NO;
    }

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    // EdgeNeXt-small is ~6.7M params; 2 threads keeps CPU pressure low so tagging
    // never contends the UI (see ImageTagEngine's .utility queue).
    TfLiteInterpreterOptionsSetNumThreads(options, 2);
    _interpreter = TfLiteInterpreterCreate(_model, options);
    TfLiteInterpreterOptionsDelete(options);
    if (!_interpreter) {
        _lastError = "TfLiteInterpreterCreate failed";
        return NO;
    }
    if (TfLiteInterpreterAllocateTensors(_interpreter) != kTfLiteOk) {
        _lastError = "TfLiteInterpreterAllocateTensors failed";
        return NO;
    }

    // One float32 image input.
    if (TfLiteInterpreterGetInputTensorCount(_interpreter) < 1) {
        _lastError = "expected 1 input tensor (image)";
        return NO;
    }
    TfLiteTensor* in = TfLiteInterpreterGetInputTensor(_interpreter, 0);
    if (TfLiteTensorType(in) != kTfLiteFloat32) {
        _lastError = "image input tensor is not float32 — ImageTagEngine sends float32";
        return NO;
    }
    int nd = TfLiteTensorNumDims(in);
    NSMutableArray<NSNumber *>* dims = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [dims addObject:@(TfLiteTensorDim(in, i))];
    _inputShape = dims;
    NSLog(@"[ImageTagLM] input name=%s type=float32 shape=%@",
          TfLiteTensorName(in), dims);

    // One float32 output of per-tag logits.
    if (TfLiteInterpreterGetOutputTensorCount(_interpreter) < 1) {
        _lastError = "expected 1 output tensor (tag logits)";
        return NO;
    }
    const TfLiteTensor* out = TfLiteInterpreterGetOutputTensor(_interpreter, 0);
    if (TfLiteTensorType(out) != kTfLiteFloat32) {
        _lastError = "output tensor is not float32";
        return NO;
    }
    _tagCount = (NSInteger)(TfLiteTensorByteSize(out) / sizeof(float));
    NSLog(@"[ImageTagLM] output name=%s tagCount=%ld",
          TfLiteTensorName(out), (long)_tagCount);
    return YES;
}

- (nullable NSArray<NSNumber *> *)tagWithPixels:(NSData *)pixels {
    if (!_interpreter) { _lastError = "not loaded"; return nil; }

    TfLiteTensor* in = TfLiteInterpreterGetInputTensor(_interpreter, 0);
    if (TfLiteTensorCopyFromBuffer(in, pixels.bytes, pixels.length) != kTfLiteOk) {
        _lastError = "copy image input failed (wrong length/dtype)";
        return nil;
    }
    if (TfLiteInterpreterInvoke(_interpreter) != kTfLiteOk) {
        _lastError = "TfLiteInterpreterInvoke failed";
        return nil;
    }

    const TfLiteTensor* out = TfLiteInterpreterGetOutputTensor(_interpreter, 0);
    size_t n = TfLiteTensorByteSize(out) / sizeof(float);
    const float* data = static_cast<const float*>(TfLiteTensorData(out));
    if (!data) { _lastError = "null output data"; return nil; }

    NSMutableArray<NSNumber *>* logits = [NSMutableArray arrayWithCapacity:n];
    for (size_t k = 0; k < n; ++k) [logits addObject:@(data[k])];
    return logits;
}

- (NSString *)lastError {
    return [NSString stringWithUTF8String:_lastError.c_str()];
}

@end
