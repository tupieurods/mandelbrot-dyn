#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <omp.h>

#include "myComplex.h"
#include "utils.h"
#include "conf.h"
#include "image.h"
#include "kernels.h"
#include "mandelbrot.h"

__host__ void mandelbrotHostEnqueue()
{
  double gpu_time = 0;
  const char imagePath[] = "./mandelbrot.png";

  // allocate memory
  int w = W, h = H;
  size_t dwell_sz = w * h * sizeof(int);
  int* h_dwells, * d_dwells;
  cucheck(cudaMalloc((void**)&d_dwells, dwell_sz));
  h_dwells = static_cast<int*>(malloc(dwell_sz));

  // compute the dwells, copy them back
  double t1 = omp_get_wtime();
  dim3 bs(64, 4), grid(divup(w, bs.x), divup(h, bs.y));
  mandelbrot_k <<<grid, bs>>> (d_dwells, w, h, complex(-1.5, -1), complex(0.5, 1));
  cucheck(cudaDeviceSynchronize());
  double t2 = omp_get_wtime();
  cucheck(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
  gpu_time = t2 - t1;

  // save the image to PNG 
  save_image(imagePath, h_dwells, w, h);

  // print performance
  printf("Mandelbrot set(host enqueue) computed in %.3lf s, at %.3lf Mpix/s\n", gpu_time, h * w * 1e-6 / gpu_time);

  // free data
  cudaFree(d_dwells);
  free(h_dwells);
}

__host__ void mandelbrotDeviceEnqueue()
{
  double gpu_time = 0;
  const char imagePath[] = "./mandelbrotDeviceEnqueue.png";

  // allocate memory
  int w = W, h = H;
  size_t dwell_sz = w * h * sizeof(int);
  int* h_dwells, * d_dwells;
  cucheck(cudaMalloc((void**)&d_dwells, dwell_sz));
  h_dwells = (int*)malloc(dwell_sz);

  // compute the dwells, copy them back
  double t1 = omp_get_wtime();
  dim3 bs(BSX, BSY), grid(INIT_SUBDIV, INIT_SUBDIV);
  mandelbrot_block_k<<<grid, bs>>>(d_dwells, w, h, complex(-1.5, -1), complex(0.5, 1), 0, 0, W / INIT_SUBDIV, 1);
  cucheck(cudaDeviceSynchronize());
  double t2 = omp_get_wtime();
  cucheck(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
  gpu_time = t2 - t1;

  // save the image to PNG file
  save_image(imagePath, h_dwells, w, h);

  // print performance
  printf("Mandelbrot set(device enqueue) computed in %.3lf s, at %.3lf Mpix/s\n", gpu_time, h * w * 1e-6 / gpu_time);

  // free data
  cudaFree(d_dwells);
  free(h_dwells);
}