#include "beamsearch_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <iostream>
#include <fstream>
#include <filesystem>

#include <vector>
#include <cmath>
#include <mutex>
#include <ctime>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

static const char* c_OpDomain = "test.beamsearchop";

struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi* ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain* domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi* ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain* domain, const OrtApi* ort_api) {
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct CustomBearmsearchKernel {
  CustomBearmsearchKernel(OrtApi api)
      : api_(api),
        ort_(api_) {
    std::cout << "CustomBearmsearchKernel constructor is invoked" << std::endl;
      if (&api_ == nullptr) {
        std::cout <<"api is nullptr"<<std::endl;
      }
 }

  void Compute(OrtKernelContext* context) {
   // Setup inputs
   if (&api_ != nullptr) {
     std::cout << "api_ exists" << std::endl;
   }

   std::cout << "compute is invoked" << std::endl;
   const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
   const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
   const float* X = ort_.GetTensorData<float>(input_X);
   const float* Y = ort_.GetTensorData<float>(input_Y);

   std::cout << "Trying to create session" << std::endl;
   OrtEnv* env;
   api_.CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "customOp", &env);
   std::cout << "end session creation" << std::endl;

   OrtSessionOptions* sessionoptions;
   api_.CreateSessionOptions(&sessionoptions);

   OrtSession* session;
   std::filesystem::path model_path = "D:\\ai\\onnxruntime\\onnxruntime\\bart_mlp_megatron_basic_test.onnx";
   std::wstring model_path_wstring = model_path.wstring();

   std::time_t start_time = std::time(0);
   try {
     api_.CreateSession(env, model_path_wstring.data(), sessionoptions, &session);
   } catch (Ort::Exception& e) {
     std::cout << e.what() << std::endl;
   }

   std::array<int64_t, 3> inputShape = {1, 2, 4};
   std::array<int64_t, 3> outputShape = {1, 2, 4};
   std::array<float, 1 * 2 * 4> input1 = {1.0f, -1.2f, 1.0f, 0.0f, -1.2f, 1.0f, 1.0f, 1.0f};
   std::array<float, 1 * 2 * 4> output1;
   std::array<const char*, 1> inputNames = {"input"};
   std::array<const char*, 1> outputNames = {"output"};

    OrtMemoryInfo *ortmemoryinfo;
    // Must be freed explicitly
    api_.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeCPU, &ortmemoryinfo);

    OrtValue* inputvalue;
    OrtValue* outputvalue;
    api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, input1.data(), 4*input1.size(), inputShape.data(),
        inputShape.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,  &inputvalue);
    api_.CreateTensorWithDataAsOrtValue(ortmemoryinfo, output1.data(), 4*output1.size(), outputShape.data(),
        outputShape.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &outputvalue);

    api_.Run(session, nullptr, inputNames.data(), &inputvalue, 1, outputNames.data(), 1, &outputvalue);

    std::time_t end_time = std::time(0);
    std::cout<<"Time elapsed for creating a session:"<<end_time-start_time<<std::endl;

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    int* out = ort_.GetTensorMutableData<int>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    for (int64_t i = 0; i < size; i++) {
      out[i] = static_cast<int>(X[i] + (*Y));
    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

struct CustomBeamSearchOP : Ort::CustomOpBase<CustomBeamSearchOP, CustomBearmsearchKernel> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {

    return new CustomBearmsearchKernel(api);
  };

  const char* GetName() const { return "CustomBeamsearchOp"; };
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };

  size_t GetInputTypeCount() const {
    // TODO Vish, how to count these?
    // There are many optional inputs
    return 2; };

  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    // TODO vish each index has a different type
    // There are some optional inputs as well, how to verify that node actually has these inputs? 
    // Is it up to the caller
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const {
    //TODO vish
    // what is the reason for this. Might change in the future.
    return 1; };
  
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    // TODO vish, same as GetInputType.
    // Optional outputs exist, how to verify that output actually exists. 
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };

} c_CustomBeamSearchOP;

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomBeamSearchOP)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}