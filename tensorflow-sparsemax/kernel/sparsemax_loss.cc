#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

using namespace tensorflow;

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

template <typename T>
class SparsemaxLossOp : public OpKernel {
 public:
  explicit SparsemaxLossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& logits_in = context->input(0);
    const Tensor& sparsemax_in = context->input(1);
    const Tensor& labels_in = context->input(2);

    OP_REQUIRES(context, logits_in.IsSameSize(sparsemax_in),
                errors::InvalidArgument(
                    "logits and sparsemax must be same size: logits_size=",
                    logits_in.shape().DebugString(), " labels_size=",
                    sparsemax_in.shape().DebugString()));

    OP_REQUIRES(context, logits_in.IsSameSize(labels_in),
                errors::InvalidArgument(
                    "logits and labels must be same size: logits_size=",
                    logits_in.shape().DebugString(), " labels_size=",
                    labels_in.shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));

    // Create an output tensor (vector with batch_size elements)
    Tensor* loss_out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({logits_in.dim_size(0)}), &loss_out
    ));

    // define 0 and 0.5 in matching type
    T zero = static_cast<T>(0);
    T half = static_cast<T>(0.5);

    // Setup data view
    auto logits = logits_in.matrix<T>();
    auto sparsemax = sparsemax_in.matrix<T>();
    auto labels = labels_in.matrix<T>();
    auto losses = loss_out->flat<T>();

    // get input size
    const int num_rows = logits.dimension(0); // batch_size
    const int num_cols = logits.dimension(1);

    // calculate sparsemax for each row
    for (int r = 0; r < num_rows; r++) {

      T loss = zero;

      for (int c = 0; c < num_cols; c++) {
        // -q^T z
        loss += - labels(r, c) * logits(r, c);

        // 0.5 * sum(z_j^2 - tau(z)^2, forall S(z))
        // note that z_i^2 - tau(z)^2 = p_i (2 * z_i - p_i) forall i in S(z)
        // also that 0.5 * p_i (2 * z_i - p_i) = p_i * (z_i - 0.5 * p_i)
        if (sparsemax(r, c) > zero) {
          loss += sparsemax(r, c) * (logits(r, c) - half * sparsemax(r, c));
        }

        // 0.5 * ||q||^2
        loss += half * labels(r, c) * labels(r, c);
      }

      losses(r) = loss;
    }
  }
};

#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(                 \
    Name("SparsemaxLoss").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    SparsemaxLossOp<T>);

TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU
