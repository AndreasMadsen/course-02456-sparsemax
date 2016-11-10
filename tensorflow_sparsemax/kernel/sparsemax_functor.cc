
#include "sparsemax_functor.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

#define UNUSED(x) (void)(x)

template <typename T>
struct Sparsemax<CPUDevice, T> {
  void operator()(typename TTypes<T>::ConstMatrix input,
                  typename TTypes<T>::Matrix sorted_unused, //only used for GPU
                  typename TTypes<T>::Matrix output) {
    UNUSED(sorted_unused); // remove unused warning

    // define integers {0, 1} in matching template type
    T zero = static_cast<T>(0);
    T one = static_cast<T>(1);
    // get input size
    const int num_rows = input.dimension(0); // batch_size
    const int num_cols = input.dimension(1);

    // create temporary vector used for sorting
    std::vector<T> sorted_temp(num_cols);
    // calculate sparsemax for each row
    for (int r = 0; r < num_rows; r++) {

      // copy input to temporary vector for sorting
      for (int c = 0; c < num_cols; c++) {
        sorted_temp[c] = input(r, c);
      }

      // sort vector
      std::sort(sorted_temp.begin(), sorted_temp.end(), std::greater<T>());

      // calculate k(z), the sorted support index
      T cumsum = zero; // cumsum use for finding support k
      T support = zero; // k
      T cumsum_support = zero; // cumsum for support i <= k
      for (int c = 0; c < num_cols; c++) {
        const T k = static_cast<T>(c) + one; // the 1-indexed index

        cumsum += sorted_temp[c];
        if (one + k * sorted_temp[c] > cumsum) {
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

      // calculate sparse probability and copy it to the output
      for (int c = 0; c < num_cols; c++) {
        output(r, c) = std::max(input(r, c) - tau, zero);
      }
    }
  }
};

template struct Sparsemax<CPUDevice, Eigen::half>;
template struct Sparsemax<CPUDevice, float>;
template struct Sparsemax<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow
