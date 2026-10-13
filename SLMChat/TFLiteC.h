// Minimal, self-contained declarations for the TensorFlow Lite / LiteRT **C**
// interpreter API.
//
// We do NOT link a separate TensorFlowLiteC framework. The embedded
// CLiteRTLM.framework (LiteRT-LM) statically bundles the TFLite runtime and
// re-exports its C API as public symbols (verified with `nm -gU`), so these
// prototypes resolve at link time against the framework the app already embeds
// and signs. Only opaque handles + function prototypes are declared here — no
// struct internals are touched (every value is read through an accessor), so
// there is no ABI dependency on the exact bundled TFLite version.
//
// Source of truth for signatures: tensorflow/lite/core/c/c_api.h and
// c_api_types.h (Apache-2.0). Keep this list minimal — add a prototype only
// when GuardLM.mm needs it, and re-verify the symbol is exported.

#ifndef SLMCHAT_TFLITE_C_H
#define SLMCHAT_TFLITE_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles — only ever used as pointers.
typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteTensor TfLiteTensor;

typedef enum TfLiteStatus {
  kTfLiteOk = 0,
  kTfLiteError = 1,
} TfLiteStatus;

// Subset of TfLiteType we care about (values match the TFLite ABI).
typedef enum TfLiteType {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
} TfLiteType;

TfLiteModel* TfLiteModelCreateFromFile(const char* model_path);
void TfLiteModelDelete(TfLiteModel* model);

TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate(void);
void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions* options,
                                           int32_t num_threads);
void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* options);

TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteModel* model,
    const TfLiteInterpreterOptions* optional_options);
void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter);
TfLiteStatus TfLiteInterpreterAllocateTensors(TfLiteInterpreter* interpreter);

int32_t TfLiteInterpreterGetInputTensorCount(const TfLiteInterpreter* interpreter);
TfLiteTensor* TfLiteInterpreterGetInputTensor(const TfLiteInterpreter* interpreter,
                                              int32_t input_index);
int32_t TfLiteInterpreterGetOutputTensorCount(const TfLiteInterpreter* interpreter);
const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    const TfLiteInterpreter* interpreter, int32_t output_index);

TfLiteStatus TfLiteInterpreterInvoke(TfLiteInterpreter* interpreter);

TfLiteStatus TfLiteTensorCopyFromBuffer(TfLiteTensor* tensor,
                                        const void* input_data,
                                        size_t input_data_size);
TfLiteStatus TfLiteTensorCopyToBuffer(const TfLiteTensor* output_tensor,
                                      void* output_data,
                                      size_t output_data_size);

TfLiteType TfLiteTensorType(const TfLiteTensor* tensor);
int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor);
int32_t TfLiteTensorDim(const TfLiteTensor* tensor, int32_t dim_index);
size_t TfLiteTensorByteSize(const TfLiteTensor* tensor);
void* TfLiteTensorData(const TfLiteTensor* tensor);
const char* TfLiteTensorName(const TfLiteTensor* tensor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SLMCHAT_TFLITE_C_H
