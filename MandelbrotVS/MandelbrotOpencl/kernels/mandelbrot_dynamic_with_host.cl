#ifndef MANDELBROT_DYNAMIC_WITH_HOST_CL
#define MANDELBROT_DYNAMIC_WITH_HOST_CL

#include "mandelbrot_core.cl"

/** maximum recursion depth */
#define MAX_DEPTH 4
/** region below which do per-pixel */
#define MIN_SIZE 32
/** subdivision factor along each axis */
#define SUBDIV 4

__attribute__((reqd_work_group_size(WORKSIZE_X, WORKSIZE_Y, 1)))
__kernel void mandelbrotFillCommon(
  __global int *dwells,
  int w,
  int x0,
  int y0,
  int d,
  int commonDwell
)
{
  int xPos = get_global_id(0) - get_global_offset(0) + x0;
  int yPos = get_global_id(1) - get_global_offset(1) + y0;
  dwells[yPos * w + xPos] = commonDwell;
}

__attribute__((reqd_work_group_size(WORKSIZE_X, WORKSIZE_Y, 1)))
__kernel void mandelbrotPerPixel(
  __global int *dwells,
  int w,
  int h,
  float2 cmin,
  float2 cmax,
  int x0,
  int y0
)
{
  int xPos = get_global_id(0) - get_global_offset(0) + x0;
  int yPos = get_global_id(1) - get_global_offset(1) + y0;

  int dwell = pixelDwell(w, h, cmin, cmax, xPos, yPos);
  dwells[yPos * w + xPos] = dwell;
}

__attribute__((reqd_work_group_size(WORKSIZE_X, WORKSIZE_Y, 1)))
__kernel void getBorderDwellKernel(
  __global int4 *commonFillBuffer,
  __global int2 *perPixelBuffer,
  __global int2 *borderBuffer,
  __global int *taskCountsBuffer,
  int w,
  int h,
  float2 cmin,
  float2 cmax,
  int x0,
  int y0,
  int d,
  int depth
)
{
  x0 += d * get_group_id(0);
  y0 += d * get_group_id(1);
  int commonDwell = getBorderDwell(w, h, cmin, cmax, x0, y0, d);
  if((get_local_id(0) == 0) && (get_local_id(1) == 0))
  {
    if(commonDwell != DIFF_DWELL)
    {
      int idx = atomic_inc(taskCountsBuffer);
      commonFillBuffer[idx] = (int4)(x0, y0, commonDwell, 0);
    }
    else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE)
    {
      int idx = atomic_inc(taskCountsBuffer + 0x02);
      borderBuffer[idx] = (int2)(x0, y0);
    }
    else
    {
      int idx = atomic_inc(taskCountsBuffer + 0x01);
      perPixelBuffer[idx] = (int2)(x0, y0);
    }
  }
}

__attribute__((reqd_work_group_size(WORKSIZE_X, 1, 1)))
__kernel void getBorderDwellDeviceEnqueueKernel(
  __global int2 *borderBufferOld,

  __global int4 *commonFillBuffer,
  __global int2 *perPixelBuffer,
  __global int2 *borderBufferNew,
  __global int *taskCountsBuffer,
  int w,
  int h,
  float2 cmin,
  float2 cmax,
  int d,
  int depth
)
{
  int tid = get_global_id(0) - get_global_offset(0);
  int2 conf = borderBufferOld[tid];
  queue_t defQ = get_default_queue();

  size_t globalWorkSize[2] = {WORKSIZE_X * SUBDIV, WORKSIZE_Y * SUBDIV};
  size_t localWorkSize[2] = {WORKSIZE_X, WORKSIZE_Y};

  void (^getBorderDwellKernelBLK)(void) = ^{getBorderDwellKernel(
    commonFillBuffer,
    perPixelBuffer,
    borderBufferNew,
    taskCountsBuffer,
    w,
    h,
    cmin,
    cmax,
    conf.s0,
    conf.s1,
    d,
    depth
  );};
  ndrange_t ndrange = ndrange_2D(globalWorkSize, localWorkSize);
  enqueue_kernel(
    defQ,
    CLK_ENQUEUE_FLAGS_NO_WAIT,
    ndrange,
    getBorderDwellKernelBLK
  );
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void getBorderDwellDeviceEnqueueKernelLauncher(
  __global int *taskCountsBufferOld,
  __global int2 *borderBufferOld,

  __global int4 *commonFillBuffer,
  __global int2 *perPixelBuffer,
  __global int2 *borderBufferNew,
  __global int *taskCountsBuffer,
  int w,
  int h,
  float2 cmin,
  float2 cmax,
  int d,
  int depth
)
{
  if(taskCountsBufferOld[2] == 0) return;
  queue_t defQ = get_default_queue();

  void (^launcherBLK)(void) = ^{getBorderDwellDeviceEnqueueKernel(
    borderBufferOld,
    commonFillBuffer,
    perPixelBuffer,
    borderBufferNew,
    taskCountsBuffer,
    w,
    h,
    cmin,
    cmax,
    d,
    depth
  );};
  ndrange_t ndrange = ndrange_1D(taskCountsBufferOld[2], WORKSIZE_X);
  enqueue_kernel(
    defQ,
    CLK_ENQUEUE_FLAGS_NO_WAIT,
    ndrange,
    launcherBLK
  );
}

__attribute__((reqd_work_group_size(WORKSIZE_X, 1, 1)))
__kernel void fillCommonDwellKernel(__global int4 *commonFillBuffer, __global int *dwells, int w, int d)
{
  int tid = get_global_id(0) - get_global_offset(0);
  int4 conf = commonFillBuffer[tid];
  queue_t defQ = get_default_queue();

  size_t globalWorkSize[2] = {d, d};
  size_t localWorkSize[2] = {WORKSIZE_X, WORKSIZE_Y};

  void (^mandelbrotFillCommonBLK)(void) = ^{mandelbrotFillCommon(dwells, w, conf.s0, conf.s1, d, conf.s2);};
  ndrange_t ndrange = ndrange_2D(globalWorkSize, localWorkSize);
  enqueue_kernel(
    defQ,
    CLK_ENQUEUE_FLAGS_NO_WAIT,
    ndrange,
    mandelbrotFillCommonBLK
  );
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void fillCommonDwellKernelLauncher(
  __global int *taskCountsBuffer,
  __global int4 *commonFillBuffer,
  __global int *dwells,
  int w,
  int d
)
{
  if(taskCountsBuffer[0] == 0) return;
  queue_t defQ = get_default_queue();

  void (^launcherBLK)(void) = ^{fillCommonDwellKernel(commonFillBuffer, dwells, w, d);};
  ndrange_t ndrange = ndrange_1D(taskCountsBuffer[0], WORKSIZE_X);
  enqueue_kernel(
    defQ,
    CLK_ENQUEUE_FLAGS_NO_WAIT,
    ndrange,
    launcherBLK
  );
}

__attribute__((reqd_work_group_size(WORKSIZE_X, 1, 1)))
__kernel void mandelbrotPerPixelKernel(
  __global int2 *perPixelBuffer,
  __global int *dwells,
  int w,
  int h,
  float2 cmin,
  float2 cmax,
  int d
)
{
  int tid = get_global_id(0) - get_global_offset(0);
  int2 conf = perPixelBuffer[tid];
  queue_t defQ = get_default_queue();

  size_t globalWorkSize[2] = {d, d};
  size_t localWorkSize[2] = {WORKSIZE_X, WORKSIZE_Y};
  void (^mandelbrotPerPixelBLK)(void) = ^{mandelbrotPerPixel(dwells, w, h, cmin, cmax, conf.s0, conf.s1);};
  ndrange_t ndrange = ndrange_2D(globalWorkSize, localWorkSize);
  enqueue_kernel(
    defQ,
    CLK_ENQUEUE_FLAGS_NO_WAIT,
    ndrange,
    mandelbrotPerPixelBLK
  );
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void mandelbrotPerPixelKernelLauncher(
  __global int *taskCountsBuffer,
  __global int2 *perPixelBuffer,
  __global int *dwells,
  int w,
  int h,
  float2 cmin,
  float2 cmax,
  int d
)
{
  if(taskCountsBuffer[1] == 0) return;
  queue_t defQ = get_default_queue();

  void (^launcherBLK)(void) = ^{mandelbrotPerPixelKernel(perPixelBuffer, dwells, w, h, cmin, cmax, d);};
  ndrange_t ndrange = ndrange_1D(taskCountsBuffer[1], WORKSIZE_X);
  enqueue_kernel(
    defQ,
    CLK_ENQUEUE_FLAGS_NO_WAIT,
    ndrange,
    launcherBLK
  );
}

#endif // MANDELBROT_DYNAMIC_WITH_HOST_CL