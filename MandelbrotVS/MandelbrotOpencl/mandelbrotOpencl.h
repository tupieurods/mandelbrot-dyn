#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <vector>
#include <CL/cl_platform.h>

std::vector<int> mandelbrotHostEnqueueOpencl(int w, int h, double* gpuTime, cl_uint platformId, cl_int deviceId);
std::vector<int> mandelbrotDeviceEnqueueOpencl(int w, int h, double* gpuTime, double* gpuTimeByEvents, int numberOfRuns, cl_uint platformId, cl_int deviceId);
std::vector<int> mandelbrotDeviceEnqueueWithHostOpencl(int w, int h, double* gpuTime, int numberOfRuns, cl_uint platformId, cl_int deviceId);
std::vector<int> mandelbrotHostEnqueueSingleWorkitemOpencl(int w, int h, double* gpuTime, int numberOfRuns, cl_uint platformId, cl_int deviceId);