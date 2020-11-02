#pragma once

#include <vector>

std::vector<int> mandelbrotHostEnqueueOpencl(int w, int h, double* gpuTime);
std::vector<int> mandelbrotDeviceEnqueueOpencl(int w, int h, double* gpuTime, int numberOfRuns);
std::vector<int> mandelbrotDeviceEnqueueWithHostOpencl(int w, int h, double* gpuTime, int numberOfRuns);
std::vector<int> mandelbrotDeviceEnqueueOpencl2(int w, int h, double* gpuTime, int numberOfRuns);