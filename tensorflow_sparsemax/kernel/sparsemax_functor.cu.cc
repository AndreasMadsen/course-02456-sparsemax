
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "sparsemax_functor.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
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
__global__ void even_sort_kernel(T *sorted,
                                 const int num_rows,
                                 const int num_cols) {
  const int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_index = blockIdx.y;

  T* sorted_row = &sorted[row_index * num_cols];

  if (!(col_index & 1) && col_index < (num_cols - 1)) {
      if (sorted_row[col_index] < sorted_row[col_index + 1]) {
        T temp = sorted_row[col_index];
        sorted_row[col_index] = sorted_row[col_index + 1];
        sorted_row[col_index + 1] = temp;
      }
  }
}

template <typename T>
__global__ void odd_sort_kernel(T *sorted,
                                const int num_rows,
                                const int num_cols) {
  const int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_index = blockIdx.y;

  T* sorted_row = &sorted[row_index * num_cols];

  if ((col_index & 1) && col_index < (num_cols - 1)) {
      if (sorted_row[col_index] < sorted_row[col_index + 1]) {
        T temp = sorted_row[col_index];
        sorted_row[col_index] = sorted_row[col_index + 1];
        sorted_row[col_index + 1] = temp;
      }
  }
}

template <typename T>
void odd_even_sort(typename TTypes<T>::Matrix sorted,
                   const int num_rows,
                   const int num_cols) {
  // calculate paramization constants
  const int col_threads_per_block = 256;
  const int col_blocks = static_cast<int>(std::ceil(
   static_cast<double>(num_cols) / static_cast<double>(col_threads_per_block)
  ));

  dim3 threads_per_block(col_threads_per_block, 1, 1);
  dim3 blocks(col_blocks, num_rows, 1);

  // calculate number of odd-even iterations
  const int iterations = static_cast<int>(std::ceil(
   static_cast<double>(num_cols) / 2.0
  ));

  // Launch the even and odd kernels separately to get a global syncronization.
  // This is a very naive approach, as global syncronization is expensive.
  for(int i = 0; i < iterations; i++) {
    // launch even kernel
    even_sort_kernel<T><<<blocks, threads_per_block>>>(
      sorted.data(), num_rows, num_cols
    );

    // launch odd kernel
    odd_sort_kernel<T><<<blocks, threads_per_block>>>(
      sorted.data(), num_rows, num_cols
    );
  }
}

template <typename T>
__global__ void support_threshold_kernel(const T* cumsum,
                                         const int num_rows,
                                         const int num_cols,
                                         T* sorted) {
  const int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_index = blockIdx.y;

  const T* cumsum_row = &cumsum[row_index * num_cols];
  T* sorted_row = &sorted[row_index * num_cols];

  const T one = static_cast<T>(1);

  if (col_index < num_cols) {
    const T k = static_cast<T>(col_index + 1);
    sorted_row[col_index] = static_cast<T>(
      one + k * sorted_row[col_index] > cumsum_row[col_index]
    );
  }
}

template <typename T>
void support_threshold(typename TTypes<T>::Matrix cumsum,
                       const int num_rows,
                       const int num_cols,
                       typename TTypes<T>::Matrix sorted) {
   // calculate paramization constants
   const int col_threads_per_block = 256;
   const int col_blocks = static_cast<int>(std::ceil(
    static_cast<double>(num_cols) / static_cast<double>(col_threads_per_block)
   ));

   dim3 threads_per_block(col_threads_per_block, 1, 1);
   dim3 blocks(col_blocks, num_rows, 1);

   // launch kernel
   support_threshold_kernel<T><<<blocks, threads_per_block>>>(
     cumsum.data(), num_rows, num_cols, sorted.data()
   );
}

template <typename T>
__global__ void calculate_tau_kernel(const T* cumsum,
                                     const int num_rows,
                                     const int num_cols,
                                     T* support) {
  const int row_index = blockIdx.x * blockDim.x + threadIdx.x;

  const T one = static_cast<T>(1);

  if (row_index < num_rows) {
    const int support_index = static_cast<int>(support[row_index]) - 1;
    const T cumsum_value = cumsum[row_index * num_cols + support_index];
    support[row_index] = (cumsum_value - one) / support[row_index];
  }
}

template <typename T>
void calculate_tau(typename TTypes<T>::Matrix cumsum,
                   const int num_rows,
                   const int num_cols,
                   typename TTypes<T>::Matrix support) {
   // calculate paramization constants
   const int threads_per_block = 256;
   const int blocks = static_cast<int>(std::ceil(
    static_cast<double>(num_rows) / static_cast<double>(threads_per_block)
   ));

   // launch kernel
   calculate_tau_kernel<T><<<blocks, threads_per_block>>>(
     cumsum.data(), num_rows, num_cols, support.data()
   );
}

template <typename T>
__global__ void calculate_properbility_kernel(const T* input,
                                              const T* tau,
                                              const int num_rows,
                                              const int num_cols,
                                              T* output) {
  const int col_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_index = blockIdx.y;

  T zero = static_cast<T>(0);

  if (col_index < num_cols) {
    const int flat_index = row_index * num_cols + col_index;
    output[flat_index] = max(input[flat_index] - tau[row_index], zero);
  }
}

template <typename T>
void calculate_properbility(typename TTypes<T>::ConstMatrix input,
                            typename TTypes<T>::Matrix tau,
                            const int num_rows,
                            const int num_cols,
                            typename TTypes<T>::Matrix output) {
  // calculate paramization constants
  const int col_threads_per_block = 256;
  const int col_blocks = static_cast<int>(std::ceil(
   static_cast<double>(num_cols) / static_cast<double>(col_threads_per_block)
  ));

  dim3 threads_per_block(col_threads_per_block, 1, 1);
  dim3 blocks(col_blocks, num_rows, 1);

  // launch kernel
  calculate_properbility_kernel<T><<<blocks, threads_per_block>>>(
    input.data(), tau.data(), num_rows, num_cols, output.data()
  );
}

template <typename T>
struct Sparsemax<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T>::ConstMatrix input,
                  typename TTypes<T>::Matrix temp,
                  typename TTypes<T>::Matrix output) {

    const int num_rows = input.dimension(0); // batch_size
    const int num_cols = input.dimension(1);

    // define class axis
    const int kClassDim = 1;
    #if !defined(EIGEN_HAS_INDEX_LIST)
        Eigen::DSizes<int, 1> along_class(kClassDim);
    #else
        Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
    #endif

    // move input to sorted (temp), and sort inplace
    cudaMemcpy(temp.data(), input.data(),
               num_rows * num_cols * sizeof(T),
               cudaMemcpyDeviceToDevice);
    odd_even_sort<T>(temp, num_rows, num_cols);

    // Cumsum the sorted matrix along axis 1.
    // Put results in output as the the sorted and cumsum needs to be used
    // together.
    Eigen::internal::SumReducer<T> reducer;
    output.device(d) = temp.scan(1, reducer, false);

    // Calculate threshold used in support calculation.
    // Replace sorted matrix, with the threshold booleans.
    support_threshold<T>(output, num_rows, num_cols, temp);

    // Sum each row, to get the support index.
    // This will reuse the temporary matrix, which is larger than required,
    // but the results will just be stored as if it was a flat vector.
    temp.device(d) = temp.sum(along_class);

    // Calculate tau
    // Overwrites temp results with tau(z), again this just uses temp
    // as a flat vector.
    calculate_tau<T>(output, num_rows, num_cols, temp);

    // Calculate properbility
    // Use temp and input, and put results in output
    calculate_properbility<T>(input, temp, num_rows, num_cols, output);
  }
};

template struct Sparsemax<GPUDevice, Eigen::half>;
template struct Sparsemax<GPUDevice, float>;
template struct Sparsemax<GPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow

#endif
