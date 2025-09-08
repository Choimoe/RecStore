#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "framework/op.h"
#include "base/tensor.h"

namespace tf = tensorflow;

REGISTER_OP("RecstoreEmbRead")
    .Input("keys: uint64")
    .Output("values: float")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        tf::shape_inference::ShapeHandle keys_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &keys_shape));
        
        const int64_t emb_dim = base::EMBEDDING_DIMENSION_D;

        tf::shape_inference::DimensionHandle L = c->Dim(keys_shape, 0);
        tf::shape_inference::ShapeHandle values_shape = c->Matrix(L, emb_dim);
        c->set_output(0, values_shape);
        return tf::OkStatus();
    })
    .Doc(R"doc(
Reads embedding vectors from Recstore.
keys: A uint64 ID tensor of length L.
values: An embedding tensor with shape [L, D].
)doc");

REGISTER_OP("RecstoreEmbUpdate")
    .Input("keys: uint64")
    .Input("grads: float")
    .SetShapeFn([](tf::shape_inference::InferenceContext* c) {
        return tf::OkStatus();
    })
    .Doc(R"doc(
Updates embedding vectors in Recstore.
keys: A uint64 ID tensor of length L.
grads: A gradient tensor with shape [L, D].
)doc");


class RecstoreEmbReadOp : public tf::OpKernel {
public:
    explicit RecstoreEmbReadOp(tf::OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(tf::OpKernelContext* context) override {
        const tf::Tensor& keys_tensor = context->input(0);
        OP_REQUIRES(context, tf::TensorShapeUtils::IsVector(keys_tensor.shape()),
                    tf::errors::InvalidArgument("Keys must be a 1-D vector."));

        const int64_t L = keys_tensor.dim_size(0);
        const int64_t D = base::EMBEDDING_DIMENSION_D;

        tf::Tensor* values_tensor = nullptr;
        tf::TensorShape values_shape({L, D});
        OP_REQUIRES_OK(context, context->allocate_output(0, values_shape, &values_tensor));

        base::RecTensor rec_keys(
            (void*)keys_tensor.flat<tensorflow::uint64>().data(),
            {L},
            base::DataType::UINT64);

        base::RecTensor rec_values(
            (void*)values_tensor->flat<float>().data(),
            {L, D},
            base::DataType::FLOAT32);

        try {
            auto op = recstore::GetKVClientOp();
            op->EmbRead(rec_keys, rec_values);
        } catch (const std::exception& e) {
            context->SetStatus(tf::errors::Internal("Recstore EmbRead failed: ", e.what()));
        }
    }
};


class RecstoreEmbUpdateOp : public tf::OpKernel {
public:
    explicit RecstoreEmbUpdateOp(tf::OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(tf::OpKernelContext* context) override {
        const tf::Tensor& keys_tensor = context->input(0);
        const tf::Tensor& grads_tensor = context->input(1);

        OP_REQUIRES(context, tf::TensorShapeUtils::IsVector(keys_tensor.shape()),
                    tf::errors::InvalidArgument("Keys must be a 1-D vector."));
        OP_REQUIRES(context, tf::TensorShapeUtils::IsMatrix(grads_tensor.shape()),
                    tf::errors::InvalidArgument("Grads must be a 2-D matrix."));
        OP_REQUIRES(context, keys_tensor.dim_size(0) == grads_tensor.dim_size(0),
                    tf::errors::InvalidArgument("Keys and Grads must have the same size in dimension 0."));
        OP_REQUIRES(context, grads_tensor.dim_size(1) == base::EMBEDDING_DIMENSION_D,
                    tf::errors::InvalidArgument("Grads has wrong embedding dimension."));

        const int64_t L = keys_tensor.dim_size(0);
        const int64_t D = grads_tensor.dim_size(1);
        
        base::RecTensor rec_keys(
            (void*)keys_tensor.flat<tensorflow::uint64>().data(),
            {L},
            base::DataType::UINT64);

        base::RecTensor rec_grads(
            (void*)grads_tensor.flat<float>().data(),
            {L, D},
            base::DataType::FLOAT32);
        
        try {
            auto op = recstore::GetKVClientOp();
            op->EmbUpdate(rec_keys, rec_grads);
        } catch (const std::exception& e) {
            context->SetStatus(tf::errors::Internal("Recstore EmbUpdate failed: ", e.what()));
        }
    }
};


REGISTER_KERNEL_BUILDER(Name("RecstoreEmbRead").Device(tf::DEVICE_CPU), RecstoreEmbReadOp);
REGISTER_KERNEL_BUILDER(Name("RecstoreEmbUpdate").Device(tf::DEVICE_CPU), RecstoreEmbUpdateOp);
