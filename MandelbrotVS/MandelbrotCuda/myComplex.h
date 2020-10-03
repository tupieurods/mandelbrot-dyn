#pragma once

/** a simple complex type */
struct complex
{
  __host__ __device__ complex(float re, float im = 0)
  {
    this->re = re;
    this->im = im;
  }

  /** real and imaginary part */
  float re, im;
}; // struct complex

// operator overloads for complex numbers
inline __host__ __device__ complex operator+(const complex& a, const complex& b)
{
  return {a.re + b.re, a.im + b.im};
}

inline __host__ __device__ complex operator-(const complex& a)
{
  return {-a.re, -a.im};
}

inline __host__ __device__ complex operator-(const complex& a, const complex& b)
{
  return {a.re - b.re, a.im - b.im};
}

inline __host__ __device__ complex operator*(const complex& a, const complex& b)
{
  return {a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im};
}

inline __host__ __device__ float abs2(const complex& a)
{
  return a.re * a.re + a.im * a.im;
}

inline __host__ __device__ complex operator/(const complex& a, const complex& b)
{
  const float invAbs2 = 1 / abs2(b);
  return {
    (a.re * b.re + a.im * b.im) * invAbs2,
    (a.im * b.re - b.im * a.re) * invAbs2
  };
} // operator/
