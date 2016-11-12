#pragma once

// The CPU Op must be compiled by gcc and the GPU op must be compiled
// by nvcc. However because Eigen is used the code is exactly the same.
// To reuse the Eigen code, the code is defined in a header. This
// header is included in both the .cc and .cu.cc files. In the
// .cu.cc file Device = GPUDevice in the .cc file Device = CPUDevice.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct SparsemaxLoss {
  void operator()(const Device& d,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::ConstMatrix sparsemax,
                  typename TTypes<T>::ConstMatrix labels,
                  typename TTypes<T>::Vec losses) {

    // define class axis
    const int kClassDim = 1;
    #if !defined(EIGEN_HAS_INDEX_LIST)
        Eigen::DSizes<int, 1> along_class(kClassDim);
    #else
        Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
    #endif

    T zero = static_cast<T>(0);
    T half = static_cast<T>(0.5);

    // sum over support
    auto support = (sparsemax > zero).template cast<T>();
    auto sum_s = support * sparsemax * (logits - half * sparsemax);

    // - z_k + ||q||^2
    auto q_part = labels * (half * labels - logits);

    losses.device(d) = (sum_s + q_part)
      .sum(along_class)
      .eval();
  }
};

}  // namespace functor
}  // namespace tensorflow
