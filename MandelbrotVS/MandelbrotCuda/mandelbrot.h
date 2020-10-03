#pragma once

#include <vector>

std::vector<int> mandelbrotHostEnqueueCuda(int w, int h, double* gpuTime);
std::vector<int> mandelbrotDeviceEnqueueCuda(int w, int h, double* gpuTime);