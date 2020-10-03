#include "OpenclHelpers.h"
#include <vector>
#include "cl2.hpp"
#include <filesystem>
#include "FileHelpers.h"

std::string OpenclErrorCodeToString(const cl_int code)
{
  switch(code)
  {
  case CL_SUCCESS: return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
  case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
  case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
  case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
  case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
  case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
  case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
  case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
  case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
  case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
  case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
  case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
  case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
  case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
  case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
  case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
  case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
  case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
  case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
  case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
  case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
  case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
  case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
  case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
  case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
  case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
  case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
  case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
  case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
  case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
  case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
  case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
  case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";

  default: return "UNKNOWN CL ERROR CODE";
  }
}

void CheckOpenclCall(const cl_int status, std::string operation)
{
  if(status != CL_SUCCESS)
  {
    const std::string error = operation + " returned error: " + OpenclErrorCodeToString(status);
    throw std::exception(error.c_str());
  }
}

cl::Platform GetOpenclPlatform(cl_uint platformId)
{
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if(platformId >= platforms.size())
  {
    throw std::exception("Invalid platform id");
  }

  cl::Platform result = platforms[platformId];

  std::string str = result.getInfo<CL_PLATFORM_NAME>();
  printf("Selected platform with name: %s\n", str.c_str());

  return result;
}

cl::Device GetOpenclDevice(cl::Platform &platform, cl_uint deviceId)
{

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

  if(deviceId >= devices.size())
  {
    throw std::exception("Invalid device id");
  }

  cl::Device result = devices[deviceId];

  std::string str = result.getInfo<CL_DEVICE_NAME>();
  printf("Selected device with name: %s\n", str.c_str());

  return result;
}

cl::Context CreateOpenclContext(cl::Platform &platform, cl::Device &device)
{
  cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, cl_context_properties(platform()), 0 };

  cl_int status;
  cl::Context result = cl::Context(device, contextProperties, nullptr, nullptr, &status);
  CheckOpenclCall(status, "CreateOpenclContext");

  return result;
}

cl::Program CreateOpenclProgramFromCode(std::filesystem::path filePath, cl::Context &context, cl::Device &device)
{
  const std::string programSource = LoadTextFile(filePath.string());
  cl::Program result(context, programSource);

  if(result.build("-cl-std=CL2.0") != CL_SUCCESS)
  {
    std::string errorLog;
    // CL_PROGRAM_BUILD_LOG
    auto status = result.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &errorLog);
    printf("CreateOpenclProgramFromCode program build log: %s\n", errorLog.c_str());
    throw std::exception("Failed to build opencl program from code");
  }

  SaveOpenclProgram(result, filePath.stem().string() + ".bin");

  return result;
}

cl::Program CreateOpenclProgramFromBinary(std::filesystem::path filePath, cl::Context& context, cl::Device& device)
{
  cl_int status;
  std::vector<uint8_t> binfaryFile = LoadBinaryFile(filePath.string());
  cl::Program result(context, {device}, {binfaryFile}, nullptr, &status);

  if(result.build() != CL_SUCCESS)
  {
    std::string errorLog;
    // CL_PROGRAM_BUILD_LOG
    auto status = result.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &errorLog);
    printf("CreateOpenclProgramFromCode program build log: %s\n", errorLog.c_str());
    throw std::exception("Failed to build opencl program from binary");
  }
  return result;
}

cl::Kernel CreateOpenclKernel(cl::Program &program, const std::string &kernelName)
{
  cl::Kernel result(program, kernelName.c_str());
  return result;
}

cl::CommandQueue CreateOpenclCommandQueue(cl::Context &context, cl::Device &device)
{
  cl_int status;
  const cl_command_queue_properties props = 0;
  cl::CommandQueue result(context, device, props, &status);
  CheckOpenclCall(status, "CreateOpenclCommandQueue");
  return result;
}

void SaveOpenclProgram(cl::Program& program, std::string binaryFileName)
{
  cl_int programNumberOfDevices;
  cl_int status = program.getInfo(CL_PROGRAM_NUM_DEVICES, &programNumberOfDevices);
  CheckOpenclCall(status, "clGetProgramInfo get number of devices error");

  std::vector<size_t> programSizes(programNumberOfDevices);
  status = program.getInfo(CL_PROGRAM_BINARY_SIZES, &programSizes);
  CheckOpenclCall(status, "clGetProgramInfo get program sizes per device");

  std::vector<std::vector<uint8_t>> binaries;
  binaries.reserve(programSizes.size());
  for(auto programSize : programSizes)
  {
    binaries.emplace_back(programSize);
  }

  status = program.getInfo(CL_PROGRAM_BINARIES, &binaries);
  CheckOpenclCall(status, "clGetProgramInfo get program binaries");

  for(auto&& binary : binaries)
  {
    // This is unlikely that we would have more than 1 valid binary. But just to be safe
    if(!binary.empty() && binary[0] != '\0')
    {
      if(SaveBinaryFile(std::filesystem::current_path().string(), binaryFileName, binary))
      {
        break;
      }
    }
  }
}