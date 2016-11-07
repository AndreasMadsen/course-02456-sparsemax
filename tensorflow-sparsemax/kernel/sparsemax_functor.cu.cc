
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "sparsemax_functor.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

#include <cmath>
#include <type_traits>
#include <math_constants.h>

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
__global__ void SparsemaxKernel(const T* in,
                                const int num_rows,
                                const int num_cols,
                                T* out) {
  int row_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_id < num_rows) {
    // define integers {0, 1} in matching template type
    T zero = static_cast<T>(0);
    T one = static_cast<T>(1);
    T inf = static_cast<T>(CUDART_INF);

    // get the row array. This assumes row major data ordering, but that
    // is the default in tensorflow anyway.
    const T* in_row = &in[row_id * num_cols];
    T* out_row = &out[row_id * num_cols];

    // calculate k(z), by simultaneously sorting and computing cumsum
    
    // temporary variables used for online sorting
    T sorted_z_at_c_prev = inf;
    int sorted_i_at_c_prev = 0;
    
    // support variables
    T cumsum = zero; // cumsum use for finding support k
    T support = zero; // k
    T cumsum_support = zero; // cumsum for support i <= k

    for (int c = 0; c < num_cols; c++) {
      const T k = static_cast<T>(c) + one; // the 1-indexed index

      // online bubble sort, get the next item in the sorted vector z
      T sorted_z_at_c = -inf;
      int sorted_i_at_c = -1;
      for (int i = 0; i < num_cols; i++) {
        // if the two values are equal value:
        // we only need to increase the index
        if (in_row[i] == sorted_z_at_c_prev && i > sorted_i_at_c_prev) {
          sorted_z_at_c = in_row[i];
          sorted_i_at_c = i;
          // also stop early, to prevent the index from increasing more
          // than required.
          break;
        }

        // if the value decreased, consider it. If it's greater than what
        // was previously considered, update.
        if (in_row[i] < sorted_z_at_c_prev && in_row[i] > sorted_z_at_c) {
          sorted_z_at_c = in_row[i];
          sorted_i_at_c = i;
        }
      }
      // prepare for next iteration
      sorted_z_at_c_prev = sorted_z_at_c;
      sorted_i_at_c_prev = sorted_i_at_c;

      // calculate k(z), the sorted support index
      cumsum += sorted_z_at_c;
      if (one + k * sorted_z_at_c > cumsum) {
        support = k;
        cumsum_support = cumsum;
      } else {
        // All the remaining cases will be false, thus we break to save
        // computation time.
        break;
      }
    }

    // calculate tau(z)
    const T tau = (cumsum_support - one) / support;

    // calculate probability and copy to output
    for (int c = 0; c < num_cols; c++) {
      out_row[c] = max(in_row[c] - tau, zero);
    }
  }
}

template <typename T>
struct Sparsemax<GPUDevice, T> {
  void operator()(typename TTypes<T>::ConstMatrix input,
                  typename TTypes<T>::Matrix output) {

    const int num_rows = input.dimension(0); // batch_size
    const int num_cols = input.dimension(1);

    const int threads_pr_block = 256;
    const int blocks = static_cast<int>(std::ceil(
      static_cast<double>(num_rows) / static_cast<double>(threads_pr_block)
    ));

    SparsemaxKernel<T><<<blocks, threads_pr_block>>>(input.data(),
                                                     num_rows,
                                                     num_cols,
                                                     output.data());
  }
};

template struct Sparsemax<GPUDevice, Eigen::half>;
template struct Sparsemax<GPUDevice, float>;
template struct Sparsemax<GPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow

#endif
