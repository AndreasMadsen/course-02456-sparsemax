
#define EIGEN_USE_THREADS

#include "sparsemax_loss.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("SparsemaxLoss")
  .Input("logits: T")
  .Input("sparsemax: T")
  .Input("labels: T")
  .Output("loss: T")
  .Attr("T: {half, float, double}")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    // Similar to SoftmaxCrossEntropyWithLogits
    // Ensure that input has rank 2, and they all have the same size
    shape_inference::ShapeHandle input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
    TF_RETURN_IF_ERROR(c->Merge(input, c->input(1), &input));
    TF_RETURN_IF_ERROR(c->Merge(input, c->input(2), &input));
    // Output is a vector with the loss for each observation
    shape_inference::DimensionHandle batch_size = c->Dim(input, 0);
    c->set_output(0, c->Vector(batch_size));
    return Status::OK();
  })
  .Doc(R"doc(
Computes sparsemax loss function [1].

[1]: https://arxiv.org/abs/1602.02068

)doc");

// The code here and in sparsemax_loss.h is rather template heavy, you should
// refer to http://stackoverflow.com/questions/610245/where-and-why-do-i-have-to-put-the-template-and-typename-keywords
// if there is some syntax you don't understand.

template <typename Device, typename T>
class SparsemaxLossOp : public OpKernel {
 public:
  explicit SparsemaxLossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& logits_in = context->input(0);
    const Tensor& sparsemax_in = context->input(1);
    const Tensor& labels_in = context->input(2);

    OP_REQUIRES(context,
                logits_in.IsSameSize(sparsemax_in),
                errors::InvalidArgument(
                    "logits and sparsemax must be same size: logits_size=",
                    logits_in.shape().DebugString(), " labels_size=",
                    sparsemax_in.shape().DebugString()));

    OP_REQUIRES(context,
                logits_in.IsSameSize(labels_in),
                errors::InvalidArgument(
                    "logits and labels must be same size: logits_size=",
                    logits_in.shape().DebugString(), " labels_size=",
                    labels_in.shape().DebugString()));

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));

    // Create an output tensor (vector with batch_size elements)
    Tensor* loss_out = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({logits_in.dim_size(0)}),
                   &loss_out));

    // Setup data view
    typename TTypes<T>::ConstMatrix logits = logits_in.matrix<T>();
    typename TTypes<T>::ConstMatrix sparsemax = sparsemax_in.matrix<T>();
    typename TTypes<T>::ConstMatrix labels = labels_in.matrix<T>();
    typename TTypes<T>::Vec losses = loss_out->flat<T>();

    // This will call the Eigen code. Note that this file doesn't need to
    // compile functor::SparsemaxLoss (it does in CPU case, but not in the GPU
    // case). It just needs to know the "symbol signature" such that linker
    // can put it all together later.
    const Device& eigen_device = context->eigen_device<Device>();
    functor::SparsemaxLoss<Device, T>()(
      eigen_device, logits, sparsemax, labels, losses
    );
  }
};

//
// A part of this design pattern was taken from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/batch_norm_op.cc
//

// This will compile the Op for CPUDevice and compile the corresponding
// Eigen code.
#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(                     \
    Name("SparsemaxLoss").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    SparsemaxLossOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU

// This will compile the Op for the GPUDevice, but it doesn't know the actual
// Eigen code. However the compiler will need to know that this is compiled
// in another file (.cu.cc by nvcc). To do this the `extern` keyword is used.
#if GOOGLE_CUDA

// This will specify the symbol signature and tell the compiler that
// SparsemaxLoss<GPUDevice, T> will be compiled from another file.
namespace functor {
#define DECLARE_GPU_SPEC(T)                           \
  template <>                                         \
  void SparsemaxLoss<GPUDevice, T>::operator()(       \
    const GPUDevice& d,                               \
    typename TTypes<T>::ConstMatrix logits,           \
    typename TTypes<T>::ConstMatrix sparsemax,        \
    typename TTypes<T>::ConstMatrix labels,           \
    typename TTypes<T>::Vec losses);                  \
  extern template struct SparsemaxLoss<GPUDevice, T>;

TF_CALL_half(DECLARE_GPU_SPEC);
TF_CALL_float(DECLARE_GPU_SPEC);
TF_CALL_double(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// This will compile the Op for GPUDevice but **not** compile the corresponding
// Eigen code.
#define REGISTER_GPU(T) REGISTER_KERNEL_BUILDER(                     \
    Name("SparsemaxLoss").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
    SparsemaxLossOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
