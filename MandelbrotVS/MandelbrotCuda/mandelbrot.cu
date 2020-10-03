#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <vector>

#include "myComplex.h"
#include "utils.h"
#include "mandelbrotDevice.h"
#include "mandelbrot.h"
#include "conf.h"

__host__ std::vector<int> mandelbrotHostEnqueue(int w, int h, double *gpuTime)
{
  // allocate memory
  const size_t dwellsSize = w * h * sizeof(int);
  int* dwellsDevice;
  cucheck(cudaMalloc(reinterpret_cast<void**>(&dwellsDevice), dwellsSize))
  std::vector<int> dwellsHost(w * h);

  // compute the dwells, copy them back
  const double t1 = omp_get_wtime();
  dim3 bs(64, 4), grid(divup(w, static_cast<int>(bs.x)), divup(h, static_cast<int>(bs.y)));
  mandelbrot_k<<<grid, bs>>>(dwellsDevice, w, h, complex(-1.5, -1), complex(0.5, 1));
  cucheck(cudaDeviceSynchronize())
  const double t2 = omp_get_wtime();
  cucheck(cudaMemcpy(dwellsHost.data(), dwellsDevice, dwellsSize, cudaMemcpyDeviceToHost))
  *gpuTime = t2 - t1;

  // free data
  cudaFree(dwellsDevice);

  return dwellsHost;
}

__host__ std::vector<int> mandelbrotDeviceEnqueue(int w, int h, double* gpuTime)
{
  // allocate memory
  const size_t dwellsSize = w * h * sizeof(int);
  int* dwellsDevice;
  cucheck(cudaMalloc(reinterpret_cast<void**>(&dwellsDevice), dwellsSize))
  std::vector<int> dwellsHost(w * h);

  // compute the dwells, copy them back
  const double t1 = omp_get_wtime();
  dim3 bs(BSX, BSY), grid(INIT_SUBDIV, INIT_SUBDIV);
  mandelbrot_block_k<<<grid, bs>>>(dwellsDevice, w, h, complex(-1.5, -1), complex(0.5, 1), 0, 0, w / INIT_SUBDIV, 1);
  cucheck(cudaDeviceSynchronize())
  const double t2 = omp_get_wtime();
  cucheck(cudaMemcpy(dwellsHost.data(), dwellsDevice, dwellsSize, cudaMemcpyDeviceToHost))
  *gpuTime = t2 - t1;

  // free data
  cudaFree(dwellsDevice);
  return dwellsHost;
}