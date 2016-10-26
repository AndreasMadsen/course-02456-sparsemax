
#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

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
    auto logits = logits_in.matrix<T>();
    auto sparsemax = sparsemax_in.matrix<T>();
    auto labels = labels_in.matrix<T>();
    auto losses = loss_out->flat<T>();

    // define class axis
    const int kClassDim = 1;
    #if !defined(EIGEN_HAS_INDEX_LIST)
        Eigen::DSizes<int, 1> along_class(kClassDim);
    #else
        Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
    #endif

    T zero = static_cast<T>(0);
    T half = static_cast<T>(0.5);

    // z_k
    auto z_k = labels * logits;

    // sum over support
    auto support = (sparsemax > zero).template cast<T>();
    auto sum_s = support * sparsemax * (logits - half * sparsemax);

    // q norm
    auto q_norm = half * (labels * labels);

    const Device& eigen_device = context->eigen_device<Device>();
    losses.device(eigen_device) = (-z_k + sum_s + q_norm)
      .sum(along_class)
      .eval();
  }
};

#define REGISTER(Dev, T) REGISTER_KERNEL_BUILDER(                 \
    Name("SparsemaxLoss").Device(DEVICE_##Dev).TypeConstraint<T>("T"), \
    SparsemaxLossOp<Dev##Device, T>);

REGISTER(CPU, Eigen::half);
REGISTER(CPU, float);
REGISTER(CPU, double);

#undef REGISTER
