#pragma once

#include "myComplex.h"

__global__ void mandelbrot_k(int* dwells, int w, int h, complex cmin, complex cmax);
__global__ void mandelbrot_block_k(int* dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int depth);