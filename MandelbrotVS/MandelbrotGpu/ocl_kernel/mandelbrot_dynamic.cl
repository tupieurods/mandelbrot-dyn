#ifndef MANDELBROT_DYNAMIC_CL
#define MANDELBROT_DYNAMIC_CL

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
__kernel void mandelbrot(
  __global int *dwells,
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
  queue_t defQ = get_default_queue();
  x0 += d * get_group_id(0);
  y0 += d * get_group_id(1);
  int commonDwell = getBorderDwell(w, h, cmin, cmax, x0, y0, d);

  if((get_local_id(0) == 0) && (get_local_id(1) == 0))
  {
    if(commonDwell != DIFF_DWELL)
    {
      size_t globalWorkSize[2] = {d, d};
      size_t localWorkSize[2] = {WORKSIZE_X, WORKSIZE_Y};
      void (^mandelbrotFillCommonBLK)(void) = ^{mandelbrotFillCommon(dwells, w, x0, y0, d, commonDwell);};
      ndrange_t ndrange = ndrange_2D(globalWorkSize, localWorkSize);
      enqueue_kernel(
        defQ,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange,
        mandelbrotFillCommonBLK
      );
    }
    else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE)
    {
      size_t globalWorkSize[2] = {WORKSIZE_X * SUBDIV, WORKSIZE_Y * SUBDIV};
      size_t localWorkSize[2] = {WORKSIZE_X, WORKSIZE_Y};
      void (^mandelbrotBLK)(void) = ^{mandelbrot(dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth + 1);};
      ndrange_t ndrange = ndrange_2D(globalWorkSize, localWorkSize);
      enqueue_kernel(
        defQ,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange,
        mandelbrotBLK
      );
    }
    else
    {
      size_t globalWorkSize[2] = {d, d};
      size_t localWorkSize[2] = {WORKSIZE_X, WORKSIZE_Y};
      void (^mandelbrotPerPixelBLK)(void) = ^{mandelbrotPerPixel(dwells, w, h, cmin, cmax, x0, y0);};
      ndrange_t ndrange = ndrange_2D(globalWorkSize, localWorkSize);
      enqueue_kernel(
        defQ,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange,
        mandelbrotPerPixelBLK
      );
    }
  }
}

/*__attribute__((reqd_work_group_size(WORKSIZE_X, WORKSIZE_Y, 1)))
__kernel void mandelbrot(
  __global int *dwells,
  int w,
  int h,
  float2 cmin,
  float2 cmax,
  int x0,
  int y0,
  int d,
  int depth,
  queue_t queue
)
{
  x0 += d * get_group_id(0);
  y0 += d * get_group_id(1);
  int commonDwell = getBorderDwell(w, h, cmin, cmax, x0, y0, d);

  if((get_local_id(0) == 0) && (get_local_id(1) == 0))
  {
    clk_event_t finishKernelEvent, enqueueMarkerEvent;
    if(commonDwell != DIFF_DWELL)
    {
      size_t globalWorkSize[2] = {d, d};
      size_t localWorkSize[2] = {WORKSIZE_X, WORKSIZE_Y};
      void (^mandelbrotFillCommonBLK)(void) = ^{mandelbrotFillCommon(dwells, w, x0, y0, d, commonDwell);};
      ndrange_t ndrange = ndrange_2D(globalWorkSize, localWorkSize);
      enqueue_kernel(
        queue,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange,
        0,
        NULL,
        &finishKernelEvent,
        mandelbrotFillCommonBLK
      );
    }
    else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE)
    {
      size_t globalWorkSize[2] = {WORKSIZE_X * SUBDIV, WORKSIZE_Y * SUBDIV};
      size_t localWorkSize[2] = {WORKSIZE_X, WORKSIZE_Y};
      void (^mandelbrotBLK)(void) = ^{mandelbrot(dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth + 1, queue);};
      ndrange_t ndrange = ndrange_2D(globalWorkSize, localWorkSize);
      enqueue_kernel(
        queue,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange,
        0,
        NULL,
        &finishKernelEvent,
        mandelbrotBLK
      );
    }
    else
    {
      size_t globalWorkSize[2] = {d, d};
      size_t localWorkSize[2] = {WORKSIZE_X, WORKSIZE_Y};
      void (^mandelbrotPerPixelBLK)(void) = ^{mandelbrotPerPixel(dwells, w, h, cmin, cmax, x0, y0);};
      ndrange_t ndrange = ndrange_2D(globalWorkSize, localWorkSize);
      enqueue_kernel(
        queue,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange,
        0,
        NULL,
        &finishKernelEvent,
        mandelbrotPerPixelBLK
      );
    }

    enqueue_marker(queue, 1, &finishKernelEvent, &enqueueMarkerEvent);

    release_event(finishKernelEvent);
    release_event(enqueueMarkerEvent);
  }
}*/

#endif // MANDELBROT_DYNAMIC_CL