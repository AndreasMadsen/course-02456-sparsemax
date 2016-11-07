#pragma once

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct Sparsemax {
  void operator()(typename TTypes<T>::ConstMatrix input,
                  typename TTypes<T>::Matrix sorted,
                  typename TTypes<T>::Matrix output);
};

}  // namespace functor
}  // namespace tensorflow
