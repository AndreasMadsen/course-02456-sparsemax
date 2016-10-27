#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <string>
#include <cstdlib>
#include <cstdio>

static void debugprint(std::string msg) {
    if (std::getenv("DEBUG")) {
      std::printf("> debug: %s\n", msg.c_str());
    }
}

__global__ void CustomSquareKernel(const int* in, const int N, int* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
     out[i] = in[i] + 1;
  }
}

void CustomSquareKernelLauncher(const int* in, const int N, int* out) {
  debugprint("c++ gpu-launcher");
  CustomSquareKernel<<<32, 256>>>(in, N, out);
}

#endif
