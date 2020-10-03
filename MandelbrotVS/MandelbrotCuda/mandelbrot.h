#pragma once

#include <vector>

std::vector<int> mandelbrotHostEnqueue(int w, int h, double* gpuTime);
std::vector<int> mandelbrotDeviceEnqueue(int w, int h, double* gpuTime);