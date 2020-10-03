#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "myComplex.h"
#include "mandelbrotDevice.h"
#include "utils.h"
#include "conf.h"

/** computes the dwell for a single pixel */
__device__ int pixel_dwell(int w, int h, complex cmin, complex cmax, int x, int y)
{
  complex dc = cmax - cmin;
  float fx = static_cast<float>(x) / w, fy = static_cast<float>(y) / h;
  complex c = cmin + complex(fx * dc.re, fy * dc.im);
  int dwell = 0;
  complex z = c;
  while(dwell < MAX_DWELL && abs2(z) < 2 * 2)
  {
    z = z * z + c;
    dwell++;
  }
  return dwell;
} // pixel_dwell

/** computes the dwells for Mandelbrot image
    @param dwells the output array
    @param w the width of the output image
    @param h the height of the output image
    @param cmin the complex value associated with the left-bottom corner of the
    image
    @param cmax the complex value associated with the right-top corner of the
    image
 */
__global__ void mandelbrot_k(int *dwells, int w, int h, complex cmin, complex cmax)
{
  // complex value to start iteration (c)
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int dwell = pixel_dwell(w, h, cmin, cmax, x, y);
  dwells[y * w + x] = dwell;
} // mandelbrot_k


/** binary operation for common dwell "reduction": MAX_DWELL + 1 = neutral
    element, -1 = dwells are different */
#define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (-1)
__device__ int same_dwell(int d1, int d2) {
  if(d1 == d2)
    return d1;
  else if(d1 == NEUT_DWELL || d2 == NEUT_DWELL)
    return min(d1, d2);
  else
    return DIFF_DWELL;
}  // same_dwell

/** evaluates the common border dwell, if it exists */
__device__ int border_dwell
(int w, int h, complex cmin, complex cmax, int x0, int y0, int d) {
  // check whether all boundary pixels have the same dwell
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int bs = blockDim.x * blockDim.y;
  int comm_dwell = NEUT_DWELL;
  // for all boundary pixels, distributed across threads
  for(int r = tid; r < d; r += bs) {
    // for each boundary: b = 0 is east, then counter-clockwise
    for(int b = 0; b < 4; b++) {
      int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
      int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
      int dwell = pixel_dwell(w, h, cmin, cmax, x, y);
      comm_dwell = same_dwell(comm_dwell, dwell);
    }
  }  // for all boundary pixels
  // reduce across threads in the block
  __shared__ int ldwells[BSX * BSY];
  int nt = min(d, BSX * BSY);
  if(tid < nt)
    ldwells[tid] = comm_dwell;
  __syncthreads();
  for(; nt > 1; nt /= 2) {
    if(tid < nt / 2)
      ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2]);
    __syncthreads();
  }
  return ldwells[0];
}  // border_dwell

/** the kernel to fill the image region with a specific dwell value */
__global__ void dwell_fill_k
(int *dwells, int w, int x0, int y0, int d, int dwell)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if(x < d && y < d)
  {
    x += x0, y += y0;
    dwells[y * w + x] = dwell;
  }
} // dwell_fill_k

/** the kernel to fill in per-pixel values of the portion of the Mandelbrot set
		*/
__global__ void mandelbrot_pixel_k
(int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d)
{
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if(x < d && y < d)
  {
    x += x0, y += y0;
    dwells[y * w + x] = pixel_dwell(w, h, cmin, cmax, x, y);
  }
} // mandelbrot_pixel_k

/** checking for an error */
__device__ void check_error(int x0, int y0, int d)
{
  int err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    printf("error launching kernel for region (%d..%d, %d..%d)\n",
           x0, x0 + d, y0, y0 + d);
    assert(0);
  }
}

/** computes the dwells for Mandelbrot image using dynamic parallelism; one
		block is launched per pixel
		@param dwells the output array
		@param w the width of the output image
		@param h the height of the output image
		@param cmin the complex value associated with the left-bottom corner of the
		image
		@param cmax the complex value associated with the right-top corner of the
		image
		@param x0 the starting x coordinate of the portion to compute
		@param y0 the starting y coordinate of the portion to compute
		@param d the size of the portion to compute (the portion is always a square)
		@param depth kernel invocation depth
		@remarks the algorithm reverts to per-pixel Mandelbrot evaluation once
		either maximum depth or minimum size is reached
 */
__global__ void mandelbrot_block_k(int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int depth)
{
  x0 += d * blockIdx.x, y0 += d * blockIdx.y;
  int comm_dwell = border_dwell(w, h, cmin, cmax, x0, y0, d);
  if(threadIdx.x == 0 && threadIdx.y == 0)
  {
    if(comm_dwell != DIFF_DWELL)
    {
      // uniform dwell, just fill
      dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
      dwell_fill_k<<<grid, bs>>>(dwells, w, x0, y0, d, comm_dwell);
    }
    else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE)
    {
      // subdivide recursively
      dim3 bs(blockDim.x, blockDim.y), grid(SUBDIV, SUBDIV);
      mandelbrot_block_k <<<grid, bs >>>(dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth + 1);
    }
    else
    {
      // leaf, per-pixel kernel
      dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
      mandelbrot_pixel_k<<<grid, bs >>>(dwells, w, h, cmin, cmax, x0, y0, d);
    }
    cucheck_dev(cudaGetLastError());
    //check_error(x0, y0, d);
  }
} // mandelbrot_block_k
