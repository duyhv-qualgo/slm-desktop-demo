#import "GuardLM.h"
#import "TFLiteC.h"

#include <string>

// ---------------------------------------------------------------------------
// GuardLM: a thin Obj-C++ wrapper over the TFLite C interpreter API. The model
// is loaded once and the interpreter is reused for every `classify` call. The
// TFLite interpreter is NOT thread-safe; callers (GuardEngine) serialize access
// on a single queue.
// ---------------------------------------------------------------------------

@implementation GuardLM {
    TfLiteModel*       _model;
    TfLiteInterpreter* _interpreter;
    std::string        _lastError;
    // Input tensor indices resolved at load, routed by name (args_0 = input_ids,
    // args_1 = attention_mask). -1 until loaded.
    int                _idsIndex;
    int                _maskIndex;
}

- (instancetype)init {
    if ((self = [super init])) {
        _model = nullptr;
        _interpreter = nullptr;
        _idsIndex = -1;
        _maskIndex = -1;
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
    // BERT-Tiny is tiny; 2 threads is plenty and keeps CPU pressure low so the
    // guardrail never contends the UI (see GuardEngine's .utility queue).
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

    // Resolve which input tensor is input_ids vs attention_mask by name. The
    // converted graph names them serving_default_args_0 / _args_1, where args_0
    // is the first positional arg (input_ids). Fall back to positional order.
    int inCount = TfLiteInterpreterGetInputTensorCount(_interpreter);
    if (inCount < 2) {
        _lastError = "expected 2 input tensors (input_ids, attention_mask)";
        return NO;
    }
    _idsIndex = 0;
    _maskIndex = 1;
    for (int i = 0; i < inCount; ++i) {
        TfLiteTensor* t = TfLiteInterpreterGetInputTensor(_interpreter, i);
        const char* name = TfLiteTensorName(t);
        std::string n = name ? name : "";
        if (n.find("args_1") != std::string::npos ||
            n.find("mask")   != std::string::npos) {
            _maskIndex = i;
        } else if (n.find("args_0") != std::string::npos ||
                   n.find("ids")    != std::string::npos) {
            _idsIndex = i;
        }
        NSLog(@"[GuardLM] input[%d] name=%s type=%d dims=%d", i, n.c_str(),
              (int)TfLiteTensorType(t), TfLiteTensorNumDims(t));

        if (TfLiteTensorType(t) != kTfLiteInt32) {
            _lastError = "input tensor is not int32 — GuardEngine sends int32";
            return NO;
        }
    }
    return YES;
}

- (nullable NSArray<NSArray<NSNumber *> *> *)
    classifyWithInputIds:(NSData *)inputIds
           attentionMask:(NSData *)attentionMask {
    if (!_interpreter) { _lastError = "not loaded"; return nil; }

    TfLiteTensor* idsT  = TfLiteInterpreterGetInputTensor(_interpreter, _idsIndex);
    TfLiteTensor* maskT = TfLiteInterpreterGetInputTensor(_interpreter, _maskIndex);

    if (TfLiteTensorCopyFromBuffer(idsT, inputIds.bytes, inputIds.length) != kTfLiteOk) {
        _lastError = "copy input_ids failed (wrong length/dtype)";
        return nil;
    }
    if (TfLiteTensorCopyFromBuffer(maskT, attentionMask.bytes, attentionMask.length) != kTfLiteOk) {
        _lastError = "copy attention_mask failed (wrong length/dtype)";
        return nil;
    }
    if (TfLiteInterpreterInvoke(_interpreter) != kTfLiteOk) {
        _lastError = "TfLiteInterpreterInvoke failed";
        return nil;
    }

    int outCount = TfLiteInterpreterGetOutputTensorCount(_interpreter);
    NSMutableArray<NSArray<NSNumber *> *> *heads =
        [NSMutableArray arrayWithCapacity:outCount];
    for (int i = 0; i < outCount; ++i) {
        const TfLiteTensor* ot = TfLiteInterpreterGetOutputTensor(_interpreter, i);
        if (TfLiteTensorType(ot) != kTfLiteFloat32) {
            _lastError = "output tensor is not float32";
            return nil;
        }
        size_t n = TfLiteTensorByteSize(ot) / sizeof(float);
        const float* data = static_cast<const float*>(TfLiteTensorData(ot));
        if (!data) { _lastError = "null output data"; return nil; }
        NSMutableArray<NSNumber *> *head = [NSMutableArray arrayWithCapacity:n];
        for (size_t k = 0; k < n; ++k) [head addObject:@(data[k])];
        [heads addObject:head];
    }
    return heads;
}

- (NSString *)lastError {
    return [NSString stringWithUTF8String:_lastError.c_str()];
}

@end
