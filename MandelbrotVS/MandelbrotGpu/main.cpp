#include <vector>

#include "mandelbrot.h"
#include "image.h"

/** data size */
#define H (8 * 1024)
#define W (8 * 1024)

void mandelbrotCudaHostEnqueue()
{
  double gpuTime = 0;
  const char imagePath[] = "./mandelbrot.png";
  int w = W, h = H;

  std::vector<int> dwells =  mandelbrotHostEnqueue(w, h, &gpuTime);

  // save the image to PNG
  save_image(imagePath, dwells.data(), w, h);

  // print performance
  printf("Mandelbrot set(host enqueue) computed in %.3lf s, at %.3lf Mpix/s\n", gpuTime, w * h * 1e-6 / gpuTime);
}

void mandelbrotCudaDeviceEnqueue()
{
  double gpuTime = 0;
  const char imagePath[] = "./mandelbrotDeviceEnqueue.png";
  int w = W, h = H;

  std::vector<int> dwells = mandelbrotDeviceEnqueue(w, h, &gpuTime);

  // save the image to PNG
  save_image(imagePath, dwells.data(), w, h);

  // print performance
  printf("Mandelbrot set(device enqueue) computed in %.3lf s, at %.3lf Mpix/s\n", gpuTime, w * h * 1e-6 / gpuTime);
}

int main()
{
  mandelbrotCudaHostEnqueue();
  mandelbrotCudaDeviceEnqueue();
  return 0;
}