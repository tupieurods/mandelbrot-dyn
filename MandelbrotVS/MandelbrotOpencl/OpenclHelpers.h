#pragma once
#pragma warning(disable:4996)
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <string>
#include <filesystem>
#include "cl2.hpp"

std::string OpenclErrorCodeToString(const cl_int code);
void CheckOpenclCall(const cl_int status, std::string operation);

cl::Platform GetOpenclPlatform(cl_uint platformId);
cl::Device GetOpenclDevice(cl::Platform &platform, cl_uint deviceId);
cl::Context CreateOpenclContext(cl::Platform& platform, cl::Device& device);
cl::Program CreateOpenclProgramFromCode(std::filesystem::path filePath, cl::Context& context, cl::Device& device);
cl::Program CreateOpenclProgramFromBinary(std::filesystem::path filePath, cl::Context& context, cl::Device& device);
cl::Kernel CreateOpenclKernel(cl::Program &program, const std::string &kernelName);
cl::CommandQueue CreateOpenclCommandQueue(cl::Context& context, cl::Device& device);

void SaveOpenclProgram(cl::Program& program, std::string binaryFileName);