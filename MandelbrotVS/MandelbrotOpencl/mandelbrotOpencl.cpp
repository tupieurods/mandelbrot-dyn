#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include "mandelbrotOpencl.h"

#include <omp.h>

#include "cl2.hpp"
#include "OpenclHelpers.h"

static const cl_uint PLATFORM_ID = 0;
static const cl_uint DEVICE_ID = 0;

std::vector<int> mandelbrotHostEnqueueOpencl(int w, int h, double *gpuTime)
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
      {-1.5, -1.0},
      {0.5, 1.0},
      status
    );
    status = commandQueue.finish();
    const double t2 = omp_get_wtime();
    *gpuTime = t2 - t1;

    CheckOpenclCall(cl::copy(commandQueue, dwellsBuffer, dwellsHost.begin(), dwellsHost.end()), "copy from dwellsBuffer to host");
  }
  catch(std::exception &e)
  {
    printf("Error during execution: %s\n", e.what());
  }

  return dwellsHost;
}

std::vector<int> mandelbrotDeviceEnqueueOpencl(int w, int h, double *gpuTime, int numberOfRuns)
{
  std::vector<int> dwellsHost(w * h);
  const std::string openclFilename = "ocl_kernel/mandelbrot_dynamic.cl";
  const auto openclFileFullPath = std::filesystem::current_path().append(openclFilename);
  int initSubdiv = 32;
  int maxDepth = 4;
  int subdiv = 4;
  int localWorksizeX = 64;
  int localWorksizeY = 4;

  try
  {
    cl::Platform platform = GetOpenclPlatform(PLATFORM_ID);
    cl::Device device = GetOpenclDevice(platform, DEVICE_ID);
    cl::Context context = CreateOpenclContext(platform, device);
    cl::CommandQueue commandQueue = CreateOpenclCommandQueue(context, device);
    cl::DeviceCommandQueue deviceCommandQueue = CreateOpenclDeviceCommandQueue(context, device, std::optional<cl_uint>());
    cl::Program program = CreateOpenclProgramFromCode(openclFileFullPath, context, device);
    const cl::Kernel kernel = CreateOpenclKernel(program, "mandelbrot");

    cl_int status = CL_SUCCESS;
    const cl::Buffer dwellsBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * w * h, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer dwellsBuffer");

    cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_float2, cl_float2, cl_int, cl_int, cl_int, cl_int, cl::DeviceCommandQueue> kernelFunctor(kernel);

    for(int T = 0; T < numberOfRuns; T++)
    {
      commandQueue.enqueueFillBuffer<cl_int>(dwellsBuffer, 0, 0, sizeof(cl_int) * w * h, nullptr, nullptr);
      status = commandQueue.finish();

      int d = w / initSubdiv;
      cl_float2 cmin = {-1.5, -1.0};
      cl_float2 cmax = {0.5, 1.0};

      const double t1 = omp_get_wtime();

      for(int depth = 1; depth <= maxDepth; depth++)
      {
        cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_float2, cl_float2, cl_int, cl_int, cl_int, cl_int> kernelFunctor(kernel);
        kernelFunctor(
          cl::EnqueueArgs(
            commandQueue,
            cl::NDRange(localWorksizeX * initSubdiv, localWorksizeY * initSubdiv, 1),
            cl::NDRange(localWorksizeX, localWorksizeY, 1)
          ),
          dwellsBuffer,
          w,
          h,
          cmin,
          cmax,
          0, // x0
          0, // y0
          d,
          1, // depth
          status
        );
        CheckOpenclCall(status, "mandelbrotDeviceEnqueueOpencl kernel call");
        status = commandQueue.finish();
        const double t2 = omp_get_wtime();
        gpuTime[T] = t2 - t1;
      }

      CheckOpenclCall(cl::copy(commandQueue, dwellsBuffer, dwellsHost.begin(), dwellsHost.end()), "copy from dwellsBuffer to host");
    }
  }
  catch(std::exception &e)
  {
    printf("Error during execution: %s\n", e.what());
  }

  return dwellsHost;
}

std::vector<int> mandelbrotDeviceEnqueueWithHostOpencl(int w, int h, double *gpuTime, int numberOfRuns)
{
  std::vector<int> dwellsHost(w * h);
  const std::string openclFilename = "ocl_kernel/mandelbrot_dynamic_with_host.cl";
  const auto openclFileFullPath = std::filesystem::current_path().append(openclFilename);
  int initSubdiv = 32;
  int maxDepth = 4;
  int subdiv = 4;
  int localWorksizeX = 64;
  int localWorksizeY = 4;

  try
  {
    cl::Platform platform = GetOpenclPlatform(PLATFORM_ID);
    cl::Device device = GetOpenclDevice(platform, DEVICE_ID);
    cl::Context context = CreateOpenclContext(platform, device);
    cl::CommandQueue commandQueue = CreateOpenclCommandQueue(context, device);
    cl::DeviceCommandQueue deviceCommandQueue = CreateOpenclDeviceCommandQueue(context, device, std::optional<cl_uint>());
    cl::Program program = CreateOpenclProgramFromCode(openclFileFullPath, context, device);
    //const cl::Kernel kernel = CreateOpenclKernel(program, "mandelbrot");
    const cl::Kernel getBorderDwellKernel = CreateOpenclKernel(program, "getBorderDwellKernel");
    const cl::Kernel fillCommonDwellKernel = CreateOpenclKernel(program, "fillCommonDwellKernel");
    const cl::Kernel mandelbrotPerPixelKernel = CreateOpenclKernel(program, "mandelbrotPerPixelKernel");
    const cl::Kernel getBorderDwellDeviceEnqueueKernel = CreateOpenclKernel(program, "getBorderDwellDeviceEnqueueKernel");

    std::array<int, 3> countsHost0 = {0, 0, 0};

    cl_int status = CL_SUCCESS;
    const cl::Buffer dwellsBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * w * h, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer dwellsBuffer");

    const cl::Buffer countsBuffer0(context, CL_MEM_READ_WRITE, sizeof(cl_int) * 3, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer countsBuffer0");

    const cl::Buffer countsBuffer1(context, CL_MEM_READ_WRITE, sizeof(cl_int) * 3, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer countsBuffer1");

    const cl::Buffer commonFillBuffer0(context, CL_MEM_READ_WRITE, sizeof(cl_int4) * initSubdiv * initSubdiv * maxDepth * maxDepth, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer commonFillBuffer0");

    const cl::Buffer perPixelBuffer0(context, CL_MEM_READ_WRITE, sizeof(cl_int2) * initSubdiv * initSubdiv * maxDepth * maxDepth, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer perPixelBuffer0");

    const cl::Buffer borderDwellBuffer0(context, CL_MEM_READ_WRITE, sizeof(cl_int2) * initSubdiv * initSubdiv * maxDepth * maxDepth, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer borderDwellBuffer0");

    const cl::Buffer borderDwellBuffer1(context, CL_MEM_READ_WRITE, sizeof(cl_int2) * initSubdiv * initSubdiv * maxDepth * maxDepth, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer borderDwellBuffer1");

    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_float2, cl_float2, cl_int, cl_int, cl_int, cl_int> getBorderDwellKernelFunctor(getBorderDwellKernel);
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_float2, cl_float2, cl_int, cl_int> getBorderDwellDeviceEnqueueKernelFunctor(
      getBorderDwellDeviceEnqueueKernel);
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int> fillCommonDwellKernelFunctor(fillCommonDwellKernel);
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int, cl_float2, cl_float2, cl_int> mandelbrotPerPixelKernelFunctor(mandelbrotPerPixelKernel);

    for(int T = 0; T < numberOfRuns; T++)
    {
      commandQueue.enqueueFillBuffer<cl_int>(dwellsBuffer, 0, 0, sizeof(cl_int) * w * h, nullptr, nullptr);
      commandQueue.enqueueFillBuffer<cl_int>(countsBuffer0, 0, 0, sizeof(cl_int) * 3, nullptr, nullptr);
      status = commandQueue.finish();

      int d = w / initSubdiv;
      cl_float2 cmin = {-1.5, -1.0};
      cl_float2 cmax = {0.5, 1.0};

      const double t1 = omp_get_wtime();

      for(int depth = 1; depth <= maxDepth; depth++)
      {
        if(depth == 1)
        {
          getBorderDwellKernelFunctor(
            cl::EnqueueArgs(
              commandQueue,
              cl::NDRange(localWorksizeX * initSubdiv, localWorksizeY * initSubdiv, 1),
              cl::NDRange(localWorksizeX, localWorksizeY, 1)
            ),
            commonFillBuffer0,
            perPixelBuffer0,
            borderDwellBuffer0,
            countsBuffer0,
            w,
            h,
            cmin,
            cmax,
            0,
            0,
            d,
            depth,
            status
          );
          CheckOpenclCall(status, "getBorderDwellKernel enqueue call");
          CheckOpenclCall(cl::copy(commandQueue, countsBuffer0, countsHost0.begin(), countsHost0.end()), "copy from countsBuffer0 to host");
        }
        else
        {
          const auto currentCountsBuffer = depth % 2 == 0 ? countsBuffer1 : countsBuffer0;
          const auto currentBorderDwellBuffer = depth % 2 == 0 ? borderDwellBuffer1 : borderDwellBuffer0;
          const auto previousBorderDwellBuffer = depth % 2 == 0 ? borderDwellBuffer0 : borderDwellBuffer1;
          commandQueue.enqueueFillBuffer<cl_int>(currentCountsBuffer, 0, 0, sizeof(cl_int) * 3, nullptr, nullptr);
          d /= subdiv;

          if(countsHost0[2] != 0)
          {
            getBorderDwellDeviceEnqueueKernelFunctor(
              cl::EnqueueArgs(
                commandQueue,
                cl::NDRange(countsHost0[2], 1, 1),
                cl::NDRange(localWorksizeX, 1, 1)
              ),
              previousBorderDwellBuffer,
              commonFillBuffer0,
              perPixelBuffer0,
              borderDwellBuffer0,
              currentCountsBuffer,
              w,
              h,
              cmin,
              cmax,
              d,
              depth,
              status
            );
            CheckOpenclCall(status, "getBorderDwellDeviceEnqueueKernelFunctor enqueue call");
            CheckOpenclCall(cl::copy(commandQueue, currentCountsBuffer, countsHost0.begin(), countsHost0.end()), "copy from currentCountsBuffer to host");
          }
          else
          {
            break;
          }
        }

        if(countsHost0[0] != 0)
        {
          fillCommonDwellKernelFunctor(
            cl::EnqueueArgs(
              commandQueue,
              cl::NDRange(countsHost0[0], 1, 1),
              cl::NDRange(localWorksizeX, 1, 1)
            ),
            commonFillBuffer0,
            dwellsBuffer,
            w,
            d
          );
        }

        if(countsHost0[1] != 0)
        {
          mandelbrotPerPixelKernelFunctor(
            cl::EnqueueArgs(
              commandQueue,
              cl::NDRange(countsHost0[1], 1, 1),
              cl::NDRange(localWorksizeX, 1, 1)
            ),
            perPixelBuffer0,
            dwellsBuffer,
            w,
            h,
            cmin,
            cmax,
            d
          );
        }
      }

      status = commandQueue.finish();
      const double t2 = omp_get_wtime();
      gpuTime[T] = t2 - t1;
    }

    CheckOpenclCall(cl::copy(commandQueue, dwellsBuffer, dwellsHost.begin(), dwellsHost.end()), "copy from dwellsBuffer to host");
  }
  catch(std::exception &e)
  {
    printf("Error during execution: %s\n", e.what());
  }

  return dwellsHost;
}


std::vector<int> mandelbrotDeviceEnqueueOpencl2(int w, int h, double *gpuTime, int numberOfRuns)
{
  std::vector<int> dwellsHost(w * h);
  const std::string openclFilename = "ocl_kernel/mandelbrot_dynamic_test2.cl";
  const auto openclFileFullPath = std::filesystem::current_path().append(openclFilename);

  try
  {
    cl::Platform platform = GetOpenclPlatform(PLATFORM_ID);
    cl::Device device = GetOpenclDevice(platform, DEVICE_ID);
    cl::Context context = CreateOpenclContext(platform, device);
    cl::CommandQueue commandQueue = CreateOpenclCommandQueue(context, device);
    cl::DeviceCommandQueue deviceCommandQueue = CreateOpenclDeviceCommandQueue(context, device, std::optional<cl_uint>());
    cl::Program program = CreateOpenclProgramFromCode(openclFileFullPath, context, device);
    const cl::Kernel kernel = CreateOpenclKernel(program, "mandelbrotDevice");

    cl_int status = CL_SUCCESS;
    const cl::Buffer dwellsBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * w * h, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer dwellsBuffer");

    for(int T = 0; T < numberOfRuns; T++)
    {
      const double t1 = omp_get_wtime();
      cl::KernelFunctor<cl::Buffer, cl_int, cl_int, cl_float2, cl_float2> kernelFunctor(kernel);
      kernelFunctor(
        cl::EnqueueArgs(
          commandQueue,
          cl::NDRange(1, 1, 1),
          cl::NDRange(1, 1, 1)
        ),
        dwellsBuffer,
        w,
        h,
        {-1.5, -1.0},
        {0.5, 1.0},
        status
      );
      CheckOpenclCall(status, "mandelbrotDeviceEnqueueOpencl2 kernel call");
      status = commandQueue.finish();
      const double t2 = omp_get_wtime();
      gpuTime[T] = t2 - t1;
    }

    CheckOpenclCall(cl::copy(commandQueue, dwellsBuffer, dwellsHost.begin(), dwellsHost.end()), "copy from dwellsBuffer to host");
  }
  catch(std::exception &e)
  {
    printf("Error during execution: %s\n", e.what());
  }

  return dwellsHost;
}
