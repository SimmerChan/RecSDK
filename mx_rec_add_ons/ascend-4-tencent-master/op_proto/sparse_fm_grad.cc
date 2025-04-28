#include "sparse_fm_grad.h"

#include <iostream>

namespace ge {

IMPLEMT_COMMON_INFERFUNC(SparseFMGradInferShape)
{
    std::cout << "[SparseFMGrad] Infer shape ..." << std::endl;
    TensorDesc weight_desc = op.GetInputDescByName("weight");
    TensorDesc grad_desc = op.GetInputDescByName("grad");
    Shape weight_shape = weight_desc.GetShape();
    DataType data_type = weight_desc.GetDataType();
    Format format = weight_desc.GetFormat();

    std::cout << "\tsample_feature_size: " << weight_desc.GetShape().GetDim(0) << std::endl;
    std::cout << "\tembedding_size: " << weight_desc.GetShape().GetDim(1) << std::endl;
    std::cout << "\tbatch_size: " << grad_desc.GetShape().GetDim(0) << std::endl;

    TensorDesc output_desc = op.GetOutputDescByName("output");
    output_desc.SetShape(weight_shape);
    output_desc.SetDataType(data_type);
    output_desc.SetFormat(format);
    (void)op.UpdateOutputDesc("output", output_desc);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseFMGrad, SparseFMGradVerify)
{
    Shape weight_shape = op.GetInputDescByName("weight").GetShape();
    Shape index_shape = op.GetInputDescByName("index").GetShape();
    Shape grad_shape = op.GetInputDescByName("grad").GetShape();

    if (weight_shape.GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    if (grad_shape.GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    if (weight_shape.GetDim(0) != index_shape.GetDim(0) - 1) {
        return GRAPH_FAILED;
    }
    if (weight_shape.GetDim(1) != grad_shape.GetDim(1)) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseFMGrad, SparseFMGradInferShape);
VERIFY_FUNC_REG(SparseFMGrad, SparseFMGradVerify);

}  // namespace ge
