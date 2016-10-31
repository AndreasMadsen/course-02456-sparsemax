#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

using namespace tensorflow;

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

template <typename T>
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

    // define integers {0, 1} in matching template type 
    T zero = static_cast<T>(0);
    T one = static_cast<T>(1);

    // Setup data view
    auto input = logits_in.matrix<T>();
    auto output = probability_out->matrix<T>();

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
          // all the remaning cases will be false - thus we break to save computation time.
          break;
        }
      }

      // calculate tau(z)
      const T tau = (cumsum_support - one) / support;

      // calculate properbility and copy to output
      for (int c = 0; c < num_cols; c++) {
        output(r, c) = std::max(input(r, c) - tau, zero);
      }
    }
  }
};

#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(                 \
    Name("Sparsemax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    SparsemaxOp<T>);

TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU
