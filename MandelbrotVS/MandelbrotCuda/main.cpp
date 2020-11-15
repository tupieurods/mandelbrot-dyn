#include <vector>

#include "conf.h"
#include "mandelbrot.h"
#include "image.h"

void mandelbrotCudaStaticEnqueueTest()
{
  double gpuTime = 0;
  const char imagePath[] = "./mandelbrot.png";
  int w = W, h = H;

  std::vector<int> dwells = mandelbrotHostEnqueueCuda(w, h, &gpuTime);

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

int main()
{
  mandelbrotCudaStaticEnqueueTest();
  mandelbrotCudaDynamicEnqueueTest();
  return 0;
}