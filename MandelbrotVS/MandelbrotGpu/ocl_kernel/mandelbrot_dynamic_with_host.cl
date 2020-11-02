#ifndef MANDELBROT_DYNAMIC_WITH_HOST_CL
#define MANDELBROT_DYNAMIC_WITH_HOST_CL

#define WORKSIZE_X 64
#define WORKSIZE_Y 4


#define MAX_DWELL 256
#define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (-1)
#define RE(x) x.s0
#define IM(x) x.s1

/** maximum recursion depth */
#define MAX_DEPTH 4
/** region below which do per-pixel */
#define MIN_SIZE 32
/** subdivision factor along each axis */
#define SUBDIV 4

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

#endif // MANDELBROT_DYNAMIC_WITH_HOST_CL