#ifndef MANDELBROT_CL
#define MANDELBROT_CL

#define MAX_DWELL 256
#define RE(x) x.s0
#define IM(x) x.s1

float2 complexMul(float2 a, float2 b)
{
  return (float2)(RE(a) * RE(b) - IM(a) * IM(b), IM(a) * RE(b) + RE(a) * IM(b));
}

float complexAbs2(float2 a)
{
  return RE(a) * RE(a) + IM(a) * IM(a);
}

int pixelDwell(int w, int h, float2 cmin, float2 cmax, int xPos, int yPos)
{
  float2 dc = cmax - cmin;
  float fx = (float)xPos / (float)w;
  float fy = (float)yPos / (float)h;
  float2 c = cmin + (float2)(fx * dc.x, fy * dc.y);
  int dwell = 0;
  float2 z = c;
  while(dwell < MAX_DWELL && complexAbs2(z) < 2 * 2)
  {
    z = complexMul(z, z) + c;
    dwell++;
  }
  return dwell;
}

__attribute__((reqd_work_group_size(64, 4, 1)))
__kernel void mandelbrot(__global int *dwells, int w, int h, float2 cmin, float2 cmax)
{
  int xPos = get_global_id(0) - get_global_offset(0);
  int yPos = get_global_id(1) - get_global_offset(1);

  //printf((__constant char *)"%d %d\n", xPos, yPos);

  int dwell = pixelDwell(w, h, cmin, cmax, xPos, yPos);
  dwells[yPos * w + xPos] = dwell;
}

#endif // MANDELBROT_CL