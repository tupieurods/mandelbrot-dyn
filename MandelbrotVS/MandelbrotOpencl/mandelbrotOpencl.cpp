#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include "mandelbrotOpencl.h"

#include <omp.h>

#include "cl2.hpp"
#include "OpenclHelpers.h"

static const cl_uint PLATFORM_ID = 0;
static const cl_uint DEVICE_ID = 0;

std::vector<int> mandelbrotHostEnqueueOpencl(int w, int h, double* gpuTime)
{
  std::vector<int> dwellsHost(w * h);
  const std::string openclFilename = "ocl_kernel/mandelbrot.cl";
  const auto openclFileFullPath = std::filesystem::current_path().append(openclFilename);

  try
  {
    cl::Platform platform = GetOpenclPlatform(PLATFORM_ID);
    cl::Device device = GetOpenclDevice(platform, DEVICE_ID);
    cl::Context context = CreateOpenclContext(platform, device);
    cl::CommandQueue commandQueue = CreateOpenclCommandQueue(context, device);
    cl::Program program = CreateOpenclProgramFromCode(openclFileFullPath, context, device);
    const cl::Kernel kernel = CreateOpenclKernel(program, "mandelbrot");

    cl_int status = CL_SUCCESS;
    const cl::Buffer dwellsBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * w * h, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer dwellsBuffer");

    const double t1 = omp_get_wtime();
    cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_float2, cl_float2> kernelFunctor(kernel);
    kernelFunctor(
      cl::EnqueueArgs(
        commandQueue,
        cl::NDRange(w, h, 1),
        cl::NDRange(64, 4, 1)
      ),
      dwellsBuffer,
      w,
      h,
      { -1.5, -1.0 },
      { 0.5, 1.0 },
      status
    );
    status = commandQueue.finish();
    const double t2 = omp_get_wtime();
    *gpuTime = t2 - t1;

    CheckOpenclCall(cl::copy(commandQueue, dwellsBuffer, dwellsHost.begin(), dwellsHost.end()), "copy from dwellsBuffer to host");
  }
  catch(std::exception& e)
  {
    printf("Error during execution: %s\n", e.what());
  }

  return dwellsHost;
}

std::vector<int> mandelbrotDeviceEnqueueOpencl(int w, int h, double* gpuTime)
{
  std::vector<int> dwellsHost(w * h);
  const std::string openclFilename = "ocl_kernel/mandelbrot_dynamic.cl";
  const auto openclFileFullPath = std::filesystem::current_path().append(openclFilename);
  int initSubdiv = 4;
  int localWorksizeX = 64;
  int localWorksizeY = 4;

  try
  {
    cl::Platform platform = GetOpenclPlatform(PLATFORM_ID);
    cl::Device device = GetOpenclDevice(platform, DEVICE_ID);
    cl::Context context = CreateOpenclContext(platform, device);
    cl::CommandQueue commandQueue = CreateOpenclCommandQueue(context, device);
    cl::DeviceCommandQueue deviceCommandQueue =  CreateOpenclDeviceCommandQueue(context, device, std::optional<cl_uint>());
    cl::Program program = CreateOpenclProgramFromCode(openclFileFullPath, context, device);
    const cl::Kernel kernel = CreateOpenclKernel(program, "mandelbrot");

    cl_int status = CL_SUCCESS;
    const cl::Buffer dwellsBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * w * h, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer dwellsBuffer");

    const double t1 = omp_get_wtime();
    cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_float2, cl_float2, cl_int, cl_int, cl_int, cl_int, cl::DeviceCommandQueue> kernelFunctor(kernel);
    kernelFunctor(
      cl::EnqueueArgs(
        commandQueue,
        cl::NDRange(localWorksizeX * initSubdiv, localWorksizeY * initSubdiv, 1),
        cl::NDRange(localWorksizeX, localWorksizeY, 1)
      ),
      dwellsBuffer,
      w,
      h,
      { -1.5, -1.0 }, // cmin
      { 0.5, 1.0 }, // cmax
      0, // x0
      0, // y0
      w / initSubdiv, // d
      1, // depth
      deviceCommandQueue,
      status
    );
    CheckOpenclCall(status, "mandelbrotDeviceEnqueueOpencl kernel call");
    status = commandQueue.finish();
    const double t2 = omp_get_wtime();
    *gpuTime = t2 - t1;

    CheckOpenclCall(cl::copy(commandQueue, dwellsBuffer, dwellsHost.begin(), dwellsHost.end()), "copy from dwellsBuffer to host");
  }
  catch(std::exception& e)
  {
    printf("Error during execution: %s\n", e.what());
  }

  return dwellsHost;
}
