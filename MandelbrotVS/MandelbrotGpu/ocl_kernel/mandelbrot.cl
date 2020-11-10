#ifndef MANDELBROT_CL
#define MANDELBROT_CL

#include "mandelbrot_core.cl"

__attribute__((reqd_work_group_size(64, 4, 1)))
__kernel void mandelbrot(__global int *dwells, int w, int h, float2 cmin, float2 cmax)
{
  int xPos = get_global_id(0) - get_global_offset(0);
  int yPos = get_global_id(1) - get_global_offset(1);

  int dwell = pixelDwell(w, h, cmin, cmax, xPos, yPos);
  dwells[yPos * w + xPos] = dwell;
}

#endif // MANDELBROT_CL