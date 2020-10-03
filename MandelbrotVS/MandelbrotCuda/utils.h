#pragma once

/** CUDA check macro */
#define cucheck(call) \
{\
  cudaError_t res = (call); \
  if(res != cudaSuccess) \
  { \
    const char* err_str = cudaGetErrorString(res); \
    fprintf(stderr, "%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call); \
    exit(-1); \
  } \
}

#define cucheck_dev(call) \
{ \
  cudaError_t res = (call); \
  if(res != cudaSuccess) \
  { \
    const char* err_str = cudaGetErrorString(res); \
    printf("%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call); \
    assert(0); \
  } \
}

/** a useful function to compute the number of threads */
__host__ __device__ inline int divup(int x, int y)
{
  return x / y + (x % y ? 1 : 0);
}