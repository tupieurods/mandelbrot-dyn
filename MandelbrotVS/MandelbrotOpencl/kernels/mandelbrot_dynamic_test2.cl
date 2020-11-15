#ifndef MANDELBROT_CL
#define MANDELBROT_CL

#include "mandelbrot_core.cl"

__attribute__((reqd_work_group_size(64, 4, 1)))
__kernel void mandelbrot(__global int *dwells, int w, int h, float2 cmin, float2 cmax, int x0, int y0)
{
  int xPos = get_global_id(0) - get_global_offset(0);
  int yPos = get_global_id(1) - get_global_offset(1);

  int dwell = pixelDwell(w, h, cmin, cmax, xPos, yPos);
  dwells[yPos * w + xPos] = dwell;
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void mandelbrotDevice(__global int *dwells, int w, int h, float2 cmin, float2 cmax)
{
  queue_t defQ = get_default_queue();
  size_t globalWorkSize[2] = {w, h};
  size_t localWorkSize[2] = {64, 4};
  void (^mandelbrotBLK)(void) = ^{mandelbrot(dwells, w, h, cmin, cmax, 0, 0);};
  ndrange_t ndrange = ndrange_2D(globalWorkSize, localWorkSize);

  enqueue_kernel(
    defQ,
    CLK_ENQUEUE_FLAGS_NO_WAIT,
    ndrange,
    mandelbrotBLK
  );
}

#endif // MANDELBROT_CL