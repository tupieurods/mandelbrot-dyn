#pragma once

#include <vector>
#include <CL/cl_platform.h>

std::vector<int> mandelbrotHostEnqueueOpencl(int w, int h, double* gpuTime, cl_uint platformId, cl_int deviceId);
std::vector<int> mandelbrotDeviceEnqueueOpencl(int w, int h, double* gpuTime, int numberOfRuns, cl_uint platformId, cl_int deviceId);
std::vector<int> mandelbrotDeviceEnqueueWithHostOpencl(int w, int h, double* gpuTime, int numberOfRuns, cl_uint platformId, cl_int deviceId);
std::vector<int> mandelbrotDeviceEnqueueOpencl2(int w, int h, double* gpuTime, int numberOfRuns, cl_uint platformId, cl_int deviceId);