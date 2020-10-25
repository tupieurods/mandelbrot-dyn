#include <vector>

#include "mandelbrot.h"
#include "mandelbrotOpencl.h"
#include "image.h"

/** data size */
#define H (8 * 1024)
#define W (8 * 1024)

void mandelbrotCudaStaticEnqueueTest()
{
  double gpuTime = 0;
  const char imagePath[] = "./mandelbrot.png";
  int w = W, h = H;

  std::vector<int> dwells =  mandelbrotHostEnqueueCuda(w, h, &gpuTime);

  // save the image to PNG
  save_image(imagePath, dwells.data(), w, h);

  // print performance
  printf("Nvidia Mandelbrot set(host enqueue) computed in %.3lf s, at %.3lf Mpix/s\n", gpuTime, w * h * 1e-6 / gpuTime);
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
  printf("Nvidia Mandelbrot set(device enqueue) computed in %.3lf s, at %.3lf Mpix/s\n", gpuTime, w * h * 1e-6 / gpuTime);
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
  printf("AMD OPENCL. Mandelbrot set(host enqueue) computed in %.3lf s, at %.3lf Mpix/s\n", gpuTime, gpuTime != 0.0 ? w * h * 1e-6 / gpuTime : NAN);
}

void mandelbrotOpenclDynamicEnqueueTest()
{
  double gpuTime = 0;
  const char imagePath[] = "./mandelbrot_opencl_dynamic.png";
  int w = W, h = H;

  std::vector<int> dwells = mandelbrotDeviceEnqueueOpencl(w, h, &gpuTime);

  // save the image to PNG
  save_image(imagePath, dwells.data(), w, h);

  // print performance
  printf("AMD OPENCL. Mandelbrot set(device enqueue) computed in %.3lf s, at %.3lf Mpix/s\n", gpuTime, gpuTime != 0.0 ? w * h * 1e-6 / gpuTime : NAN);
}

int main()
{
  mandelbrotCudaStaticEnqueueTest();
  mandelbrotCudaDynamicEnqueueTest();
  mandelbrotOpenclHostEnqueueTest();
  mandelbrotOpenclDynamicEnqueueTest();
  return 0;
}