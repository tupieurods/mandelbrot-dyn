#include <omp.h>

#include "mandelbrotOpencl.h"
#include "OpenclHelpers.h"

std::vector<int> mandelbrotDeviceEnqueueOpencl(int w, int h, double* gpuTime, int numberOfRuns, cl_uint platformId, cl_int deviceId)
{
  std::vector<int> dwellsHost(w * h);
  const std::string openclFilename = "kernels/mandelbrot_dynamic.cl";
  const auto openclFileFullPath = std::filesystem::current_path().append(openclFilename);
  const auto openclIncludeDir = std::filesystem::current_path().append("kernels");
  int initSubdiv = 32;
  int maxDepth = 4;
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
      cl_float2 cmin = { -1.5, -1.0 };
      cl_float2 cmax = { 0.5, 1.0 };

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
  catch(std::exception& e)
  {
    printf("Error during execution: %s\n", e.what());
  }

  return dwellsHost;
}