#ifndef MANDELBROT_CORE_CL
#define MANDELBROT_CORE_CL

#define WORKSIZE_X 64
#define WORKSIZE_Y 4

#define MAX_DWELL 256
#define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (-1)
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

int getSameDwell(int d1, int d2)
{
  if(d1 == d2)
  {
    return d1;
  }
  else if(d1 == NEUT_DWELL || d2 == NEUT_DWELL)
  {
    return min(d1,d2);
  }
  else
  {
    return DIFF_DWELL;
  }
}

int getBorderDwell(int w, int h, float2 cmin, float2 cmax, int x0, int y0, int d)
{
  int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
  int groupSize = get_local_size(0) * get_local_size(1);
  int commonDwell = NEUT_DWELL;

  for(int r = tid; r < d; r += groupSize)
  {
    for(int b = 0; b < 4; b++)
    {
      int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
      int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
      int dwell = pixelDwell(w, h, cmin, cmax, x, y);
      commonDwell = getSameDwell(commonDwell, dwell);
    }
  }

  __local int localDwells[WORKSIZE_X * WORKSIZE_Y];
  int nt = min(d, groupSize);
  if(tid < nt)
  {
    localDwells[tid] = commonDwell;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for(; nt > 1; nt /= 2)
  {
    int ntHalf = nt / 2;
    if(tid < ntHalf)
    {
      localDwells[tid] = getSameDwell(localDwells[tid], localDwells[tid + ntHalf]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  return localDwells[0];
}

#endif // MANDELBROT_CORE_CL