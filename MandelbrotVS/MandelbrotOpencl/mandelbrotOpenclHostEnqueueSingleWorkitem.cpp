#include <omp.h>

#include "mandelbrotOpencl.h"
#include "OpenclHelpers.h"

std::vector<int> mandelbrotHostEnqueueSingleWorkitemOpencl(int w, int h, double* gpuTime, int numberOfRuns, cl_uint platformId, cl_int deviceId)
{
  std::vector<int> dwellsHost(w * h);
  const std::string openclFilename = "kernels/mandelbrot_single_workitem.cl";
  const auto openclFileFullPath = std::filesystem::current_path().append(openclFilename);
  const auto openclIncludeDir = std::filesystem::current_path().append("kernels");

  try
  {
    cl::Platform platform = GetOpenclPlatform(platformId);
    cl::Device device = GetOpenclDevice(platform, deviceId);
    cl::Context context = CreateOpenclContext(platform, device);
    cl::CommandQueue commandQueue = CreateOpenclCommandQueue(context, device);
    cl::DeviceCommandQueue deviceCommandQueue = CreateOpenclDeviceCommandQueue(context, device, std::optional<cl_uint>());
    cl::Program program = CreateOpenclProgramFromCode(openclFileFullPath, openclIncludeDir, context, device);
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
        { -1.5, -1.0 },
        { 0.5, 1.0 },
        status
      );
      CheckOpenclCall(status, "mandelbrotHostEnqueueSingleWorkitemOpencl kernel call");
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