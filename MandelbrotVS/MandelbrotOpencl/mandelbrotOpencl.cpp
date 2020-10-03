#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include "mandelbrotOpencl.h"
#include "cl2.hpp"
#include "OpenclHelpers.h"

static const cl_uint PLATFORM_ID = 0;
static const cl_uint DEVICE_ID = 0;

std::vector<int> mandelbrotHostEnqueueOpencl(int w, int h, double* gpuTime)
{
  std::vector<int> dwellsHost(w * h);

  try
  {
    cl::Platform platform = GetOpenclPlatform(PLATFORM_ID);
    cl::Device device = GetOpenclDevice(platform, DEVICE_ID);
    cl::Context context = CreateOpenclContext(platform, device);
    const cl::CommandQueue commandQueue = CreateOpenclCommandQueue(context, device);
    cl::Program program;
  }
  catch(std::exception& e)
  {
    printf("Error during execution: %s\n", e.what());
  }

  return dwellsHost;
}
