#include <vector>

#include "mandelbrot.h"
#include "mandelbrotOpencl.h"
#include "image.h"

/** data size */
#define H (16 * 1024)
#define W (16 * 1024)

void mandelbrotCudaStaticEnqueueTest()
{
  double gpuTime = 0;
  const char imagePath[] = "./mandelbrot.png";
  int w = W, h = H;

  std::vector<int> dwells =  mandelbrotHostEnqueueCuda(w, h, &gpuTime);

  // save the image to PNG
  save_image(imagePath, dwells.data(), w, h);

  // print performance
  printf("Nvidia CUDA. Mandelbrot set(host enqueue) computed in %.3lf s, at %.3lf Mpix/s\n\n", gpuTime, w * h * 1e-6 / gpuTime);
}

void mandelbrotCudaDynamicEnqueueTest()
{
  double gpuTime = 0;
  const char imagePath[] = "./mandelbrotDeviceEnqueue.png";
  int w = W, h = H;

  std::vector<int> dwells = mandelbrotDeviceEnqueueCuda(w, h, &gpuTime);

  // save the image to PNG
  save_image(imagePath, dwells.data(), w, h);

  // print performance
  printf("Nvidia CUDA. Mandelbrot set(device enqueue) computed in %.3lf s, at %.3lf Mpix/s\n\n", gpuTime, w * h * 1e-6 / gpuTime);
}

void mandelbrotOpenclHostEnqueueTest()
{
  double gpuTime = 0;
  const char imagePath[] = "./mandelbrot_opencl.png";
  int w = W, h = H;

  std::vector<int> dwells = mandelbrotHostEnqueueOpencl(w, h, &gpuTime);

  // save the image to PNG
  save_image(imagePath, dwells.data(), w, h);

  // print performance
  printf("AMD OPENCL. Mandelbrot set(host enqueue) computed in %.3lf s, at %.3lf Mpix/s\n\n", gpuTime, gpuTime != 0.0 ? w * h * 1e-6 / gpuTime : NAN);
}

void mandelbrotOpenclDynamicEnqueueTest()
{
  const int numberOfRuns = 2;
  double gpuTime[numberOfRuns];
  memset(gpuTime, 0, sizeof(double) * numberOfRuns);
  const char imagePath[] = "./mandelbrot_opencl_dynamic.png";
  int w = W, h = H;

  std::vector<int> dwells = mandelbrotDeviceEnqueueOpencl(w, h, gpuTime, numberOfRuns);

  // save the image to PNG
  save_image(imagePath, dwells.data(), w, h);

  // print performance
  for(int i = 0; i < numberOfRuns; i++)
  {
    printf("AMD OPENCL. Mandelbrot set(device enqueue) RUN #%d computed in %.3lf s, at %.3lf Mpix/s\n", i, gpuTime[i], gpuTime[i] != 0.0 ? w * h * 1e-6 / gpuTime[i] : NAN);
  }
  printf("\n");
}

void mandelbrotOpenclDynamicEnqueueWithHostTest()
{
  const int numberOfRuns = 2;
  double gpuTime[numberOfRuns];
  memset(gpuTime, 0, sizeof(double) * numberOfRuns);
  const char imagePath[] = "./mandelbrot_opencl_dynamic_with_host.png";
  int w = W, h = H;

  std::vector<int> dwells = mandelbrotDeviceEnqueueWithHostOpencl(w, h, gpuTime, numberOfRuns);

  // save the image to PNG
  save_image(imagePath, dwells.data(), w, h);

  // print performance
  for(int i = 0; i < numberOfRuns; i++)
  {
    printf("AMD OPENCL. Mandelbrot set(device enqueue with host) RUN #%d computed in %.3lf s, at %.3lf Mpix/s\n", i, gpuTime[i], gpuTime[i] != 0.0 ? w * h * 1e-6 / gpuTime[i] : NAN);
  }
  printf("\n");
}

void mandelbrotOpenclDynamicEnqueueTest2()
{
  const int numberOfRuns = 2;
  double gpuTime[numberOfRuns];
  memset(gpuTime, 0, sizeof(double) * numberOfRuns);
  const char imagePath[] = "./mandelbrot_opencl_dynamic_test2.png";
  int w = W, h = H;

  std::vector<int> dwells = mandelbrotDeviceEnqueueOpencl2(w, h, gpuTime, numberOfRuns);

  // save the image to PNG
  save_image(imagePath, dwells.data(), w, h);

  // print performance
  for(int i = 0; i < numberOfRuns; i++)
  {
    printf("AMD OPENCL. Second test. Mandelbrot set(device enqueue) RUN #%d computed in %.3lf s, at %.3lf Mpix/s\n", i, gpuTime[i], gpuTime[i] != 0.0 ? w * h * 1e-6 / gpuTime[i] : NAN);
  }
  printf("\n");
}

int main()
{
  mandelbrotCudaStaticEnqueueTest();
  mandelbrotCudaDynamicEnqueueTest();
  mandelbrotOpenclHostEnqueueTest();
  mandelbrotOpenclDynamicEnqueueTest();
  mandelbrotOpenclDynamicEnqueueWithHostTest();
  mandelbrotOpenclDynamicEnqueueTest2();
  return 0;
}