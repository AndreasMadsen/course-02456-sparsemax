
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "sparsemax_loss.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Compile the Eigen code for GPUDevice
template struct functor::SparsemaxLoss<GPUDevice, Eigen::half>;
template struct functor::SparsemaxLoss<GPUDevice, float>;
template struct functor::SparsemaxLoss<GPUDevice, double>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
