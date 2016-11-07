#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <string>
#include <cstdlib>
#include <cstdio>

static void debugprint(std::string msg) {
    if (std::getenv("DEBUG")) {
      std::printf("> debug: %s\n", msg.c_str());
    }
}

using namespace tensorflow;

// NOTE: I think there is a buildin Op called Square. Thus calling it just
// Square causes it to fail.
REGISTER_OP("CustomSquare")
    .Input("source: int32")
    .Output("squared: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      debugprint("c++ shape");
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class CustomSquareCPUOp : public OpKernel {
 public:
  explicit CustomSquareCPUOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<int32>();

    // output = input * input
    debugprint("c++ cpu-op");
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output(i) = input(i) * input(i);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("CustomSquare").Device(DEVICE_CPU), CustomSquareCPUOp);

#if GOOGLE_CUDA
void CustomSquareKernelLauncher(const int* in, const int N, int* out);

class CustomSquareGPUOp : public OpKernel {
 public:
  explicit CustomSquareGPUOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32>();

    const int N = input.size();
    debugprint("c++ gpu-op");
    CustomSquareKernelLauncher(input.data(), N, output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("CustomSquare").Device(DEVICE_GPU), CustomSquareGPUOp);
#endif
