#include <stdio.h>

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void CustomSquareKernel(const int* in, const int N, int* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
     out[i] = in[i] + 1;
  }
}

void CustomSquareKernelLauncher(const int* in, const int N, int* out) {
  printf("> debug: c++ gpu-launcher\n");
  CustomSquareKernel<<<32, 256>>>(in, N, out);
}

#endif
