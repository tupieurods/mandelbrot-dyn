#include <omp.h>

#include "mandelbrotOpencl.h"
#include "OpenclHelpers.h"

std::vector<int> mandelbrotDeviceEnqueueWithHostOpencl(int w, int h, double* gpuTime, int numberOfRuns, cl_uint platformId, cl_int deviceId)
{
  //printf("Inside mandelbrotDeviceEnqueueWithHostOpencl\n");

  std::vector<int> dwellsHost(w * h);
  const std::string openclFilename = "kernels/mandelbrot_dynamic_with_host.cl";
  const auto openclFileFullPath = std::filesystem::current_path().append(openclFilename);
  const auto openclIncludeDir = std::filesystem::current_path().append("kernels");
  int initSubdiv = 32;
  const int maxDepth = 4;
  int subdiv = 4;
  int localWorksizeX = 64;
  int localWorksizeY = 4;

  try
  {
    cl::Platform platform = GetOpenclPlatform(platformId);
    cl::Device device = GetOpenclDevice(platform, deviceId);
    cl::Context context = CreateOpenclContext(platform, device);
    cl::CommandQueue commandQueue = CreateOpenclCommandQueue(context, device);
    cl::DeviceCommandQueue deviceCommandQueue = CreateOpenclDeviceCommandQueue(context, device, std::optional<cl_uint>());
    cl::Program program = CreateOpenclProgramFromCode(openclFileFullPath, openclIncludeDir, context, device);

    const cl::Kernel getBorderDwellKernel = CreateOpenclKernel(program, "getBorderDwellKernel");
    const cl::Kernel fillCommonDwellKernel = CreateOpenclKernel(program, "fillCommonDwellKernel");
    const cl::Kernel mandelbrotPerPixelKernel = CreateOpenclKernel(program, "mandelbrotPerPixelKernel");
    const cl::Kernel getBorderDwellDeviceEnqueueKernel = CreateOpenclKernel(program, "getBorderDwellDeviceEnqueueKernel");

    cl_int status = CL_SUCCESS;

    std::array<int, 3> countsHost = { 0, 0, 0 };

    size_t bufferSize = sizeof(cl_int) * w * h;
    //printf("Allocating dwellsBuffer. Size: %d bytes\n", static_cast<int>(bufferSize));
    const cl::Buffer dwellsBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &status);
    CheckOpenclCall(status, "clCreateBuffer dwellsBuffer");

    std::array<cl::Buffer, maxDepth - 1> countsBuffer;
    for(int i = 0; i < maxDepth - 1; i++)
    {
      bufferSize = sizeof(cl_int) * 3;
      //printf("Allocating countsBuffer[%d]. Size: %d bytes\n", i, static_cast<int>(bufferSize));
      countsBuffer[i] = cl::Buffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, &status);
      CheckOpenclCall(status, "clCreateBuffer countsBuffer[" + std::to_string(i) + "]");
    }

    std::array<cl::Buffer, maxDepth - 1> commonFillBuffer;
    bufferSize = sizeof(cl_int4) * initSubdiv * initSubdiv;
    for(int i = 0; i < maxDepth - 1; i++)
    {
      //printf("Allocating commonFillBuffer[%d]. Size: %d bytes\n", i, static_cast<int>(bufferSize));
      commonFillBuffer[i] = cl::Buffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, &status);
      CheckOpenclCall(status, "clCreateBuffer commonFillBuffer[" + std::to_string(i) + "]");
      bufferSize *= maxDepth * maxDepth;
    }

    std::array<cl::Buffer, maxDepth - 1> perPixelBuffer;
    bufferSize = sizeof(cl_int2) * initSubdiv * initSubdiv;
    for(int i = 0; i < maxDepth - 1; i++)
    {
      //printf("Allocating perPixelBuffer[%d]. Size: %d bytes\n", i, static_cast<int>(bufferSize));
      perPixelBuffer[i] = cl::Buffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, &status);
      CheckOpenclCall(status, "clCreateBuffer perPixelBuffer[" + std::to_string(i) + "]");
      bufferSize *= maxDepth * maxDepth;
    }

    std::array<cl::Buffer, maxDepth - 1> borderDwellBuffer;
    bufferSize = sizeof(cl_int2) * initSubdiv * initSubdiv;
    for(int i = 0; i < maxDepth - 1; i++)
    {
      //printf("Allocating borderDwellBuffer[%d]. Size: %d bytes\n", i, static_cast<int>(bufferSize));
      borderDwellBuffer[i] = cl::Buffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, &status);
      CheckOpenclCall(status, "clCreateBuffer borderDwellBuffer[" + std::to_string(i) + "]");
      bufferSize *= maxDepth * maxDepth;
    }

    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_float2, cl_float2, cl_int, cl_int, cl_int, cl_int> getBorderDwellKernelFunctor(getBorderDwellKernel);
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl_int, cl_int, cl_float2, cl_float2, cl_int, cl_int> getBorderDwellDeviceEnqueueKernelFunctor(
      getBorderDwellDeviceEnqueueKernel);
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int> fillCommonDwellKernelFunctor(fillCommonDwellKernel);
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int, cl_int, cl_float2, cl_float2, cl_int> mandelbrotPerPixelKernelFunctor(mandelbrotPerPixelKernel);

    for(int T = 0; T < numberOfRuns; T++)
    {
      commandQueue.enqueueFillBuffer<cl_int>(dwellsBuffer, 0, 0, sizeof(cl_int) * w * h, nullptr, nullptr);
      for(auto& buffer : countsBuffer)
      {
        commandQueue.enqueueFillBuffer<cl_int>(buffer, 0, 0, sizeof(cl_int) * 3, nullptr, nullptr);
      }
      status = commandQueue.finish();

      int d = w / initSubdiv;
      cl_float2 cmin = { -1.5, -1.0 };
      cl_float2 cmax = { 0.5, 1.0 };

      const double t1 = omp_get_wtime();

      getBorderDwellKernelFunctor(
        cl::EnqueueArgs(
          commandQueue,
          cl::NDRange(localWorksizeX * initSubdiv, localWorksizeY * initSubdiv, 1),
          cl::NDRange(localWorksizeX, localWorksizeY, 1)
        ),
        commonFillBuffer[0],
        perPixelBuffer[0],
        borderDwellBuffer[0],
        countsBuffer[0],
        w,
        h,
        cmin,
        cmax,
        0,
        0,
        d,
        1, // depth
        status
      );
      CheckOpenclCall(status, "getBorderDwellKernel enqueue call");

      for(int depth = 2; depth <= maxDepth; depth++)
      {
        size_t previousBufferIndex = depth - 2;
        size_t currentBufferIndex = depth - 1;
        CheckOpenclCall(cl::copy(commandQueue, countsBuffer[previousBufferIndex], countsHost.begin(), countsHost.end()), "copy from countsBuffer[previousBufferIndex] to host");
        //printf("%d: %d %d %d\n", d, countsHost[0], countsHost[1], countsHost[2]);

        if(countsHost[0] != 0)
        {
          fillCommonDwellKernelFunctor(
            cl::EnqueueArgs(
              commandQueue,
              cl::NDRange(countsHost[0], 1, 1),
              cl::NDRange(localWorksizeX, 1, 1)
            ),
            commonFillBuffer[previousBufferIndex],
            dwellsBuffer,
            w,
            d,
            status
          );
          CheckOpenclCall(status, "fillCommonDwellKernel enqueue call");
        }

        if(countsHost[1] != 0)
        {
          mandelbrotPerPixelKernelFunctor(
            cl::EnqueueArgs(
              commandQueue,
              cl::NDRange(countsHost[1], 1, 1),
              cl::NDRange(localWorksizeX, 1, 1)
            ),
            perPixelBuffer[previousBufferIndex],
            dwellsBuffer,
            w,
            h,
            cmin,
            cmax,
            d,
            status
          );
          CheckOpenclCall(status, "mandelbrotPerPixelKernel enqueue call");
        }

        d /= subdiv;

        if(countsHost[2] != 0)
        {
          getBorderDwellDeviceEnqueueKernelFunctor(
            cl::EnqueueArgs(
              commandQueue,
              cl::NDRange(countsHost[2], 1, 1),
              cl::NDRange(localWorksizeX, 1, 1)
            ),
            borderDwellBuffer[previousBufferIndex],
            commonFillBuffer[currentBufferIndex],
            perPixelBuffer[currentBufferIndex],
            borderDwellBuffer[currentBufferIndex],
            countsBuffer[currentBufferIndex],
            w,
            h,
            cmin,
            cmax,
            d,
            depth,
            status
          );
          CheckOpenclCall(status, "getBorderDwellDeviceEnqueueKernel enqueue call");
        }
        else
        {
          break;
        }
      }

      status = commandQueue.finish();
      const double t2 = omp_get_wtime();
      gpuTime[T] = t2 - t1;
    }

    CheckOpenclCall(cl::copy(commandQueue, dwellsBuffer, dwellsHost.begin(), dwellsHost.end()), "copy from dwellsBuffer to host");
  }
  catch(std::exception& e)
  {
    printf("Error during execution: %s\n", e.what());
  }

  return dwellsHost;
}