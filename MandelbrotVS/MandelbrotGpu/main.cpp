#include "mandelbrot.h"

int main(int argc, char **argv)
{
  mandelbrotHostEnqueue();
  mandelbrotDeviceEnqueue();
  return 0;
}
