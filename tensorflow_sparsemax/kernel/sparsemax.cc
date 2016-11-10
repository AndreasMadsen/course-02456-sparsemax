
#include "sparsemax_functor.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("Sparsemax")
  .Input("logits: T")
  .Output("sparsemax: T")
  .Attr("T: {half, float, double}")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    // This implements:
    //   return shape_inference::UnchangedShapeWithRank(c, 2);
    // which is defined in tensorflow/core/framework/common_shape_fns.h
    // but that is not yet in the 0.11 build.
    // Softmax uses UnchangedShapeWithRankAtLeast(c, 1) in tensorflow,
    // but UnchangedShapeWithRank(c, 2) in it's corresponding loss function,
    // which takes the same input. The strict version was chosen here.
    shape_inference::ShapeHandle input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
    c->set_output(0, input);
    return Status::OK();
  })
  .Doc(R"doc(
Computes sparsemax activations [1].

For each batch `i` and class `j` we have

    sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)

[1]: https://arxiv.org/abs/1602.02068

)doc");

template <typename Device, typename T>
class SparsemaxOp : public OpKernel {
 public:
  explicit SparsemaxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& logits_in = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));

    // Create an output tensor
    Tensor* probability_out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, logits_in.shape(),
                                                     &probability_out));

    // Create temporary tensor used for storing bitonic sort
    Tensor temp_sorted;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   logits_in.shape(), &temp_sorted));

    // Setup data view
    auto input = logits_in.matrix<T>();
    auto sorted = temp_sorted.matrix<T>();
    auto output = probability_out->matrix<T>();

    const Device& eigen_device = context->eigen_device<Device>();
    functor::Sparsemax<Device, T>()(eigen_device, input, sorted, output);
  }
};

#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(                 \
    Name("Sparsemax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    SparsemaxOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU

#if GOOGLE_CUDA

#define REGISTER_GPU(T) REGISTER_KERNEL_BUILDER(                 \
    Name("Sparsemax").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
    SparsemaxOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA
