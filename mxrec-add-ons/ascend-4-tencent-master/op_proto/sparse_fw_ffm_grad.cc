#include "sparse_fw_ffm_grad.h"

#include <iostream>

namespace ge {

namespace {
    void printInfo(const TensorDesc& desc, const char* name) {
        std::cout << "[" << name << "]:" << std::endl;

        Shape shape = desc.GetShape();
        std::cout << "\tShape: (";
        for (size_t d = 0; d < shape.GetDimNum(); d++) {
            std::cout << shape.GetDim(d) << ",";
        }
        std::cout << ")" << std::endl;

        // std::cout << "\tData type: " << desc.GetDataType() << std::endl;
        // std::cout << "\tData format: " << desc.GetFormat() << std::endl;
        // std::cout << "\tSize: " << desc.GetSize() << std::endl;
        // std::cout << "\tRealDimCnt: " << desc.GetRealDimCnt() << std::endl;
        
        // std::cout << "\tShape range: [";
        // std::vector<std::pair<int64_t, int64_t>> shape_range;
        // desc.GetShapeRange(shape_range);
        // for (const auto& range: shape_range) {
        //     std::cout << "(" << range.first << "," << range.second << "), ";
        // }
        // std::cout << "]" << std::endl;
    }
}

IMPLEMT_COMMON_INFERFUNC(SparseFwFFMGradInferShape)
{
    std::cout << "[SparseFwFFMGrad] Infer shape ..." << std::endl;
    TensorDesc weight_desc = op.GetInputDescByName("weight");
    TensorDesc fw_weight_desc = op.GetInputDescByName("fw_weight");
    DataType data_type = weight_desc.GetDataType();
    Format format = weight_desc.GetFormat();

    TensorDesc output_desc = op.GetOutputDescByName("output");
    output_desc.SetShape(weight_desc.GetShape());
    output_desc.SetDataType(data_type);
    output_desc.SetFormat(format);
    (void)op.UpdateOutputDesc("output", output_desc);

    TensorDesc fw_output_desc = op.GetOutputDescByName("fw_output");
    fw_output_desc.SetShape(fw_weight_desc.GetShape());
    fw_output_desc.SetDataType(data_type);
    fw_output_desc.SetFormat(format);
    (void)op.UpdateOutputDesc("fw_output", fw_output_desc);

    printInfo(weight_desc, "weight");
    printInfo(fw_weight_desc, "fw_weight");
    printInfo(op.GetInputDescByName("field"), "field");
    printInfo(op.GetInputDescByName("grad"), "grad");
    // printInfo(output_desc, "output");
    // printInfo(fw_output_desc, "fw_output");

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseFwFFMGrad, SparseFwFFMGradVerify)
{
    Shape weight_shape =  op.GetInputDescByName("weight").GetShape();
    Shape fw_weight_shape = op.GetInputDescByName("fw_weight").GetShape();
    Shape field_shape = op.GetInputDescByName("field").GetShape();
    Shape index_shape = op.GetInputDescByName("index").GetShape();
    Shape cross_mean_sum_shape = op.GetInputDescByName("cross_mean_sum").GetShape();
    Shape cross_mean_square_sum_shape = op.GetInputDescByName("cross_mean_square_sum").GetShape();
    Shape fw_field_map_shape = op.GetInputDescByName("fw_field_map").GetShape();
    Shape grad_shape = op.GetInputDescByName("grad").GetShape();

    if (weight_shape.GetDimNum() != 3) {
        return GRAPH_FAILED;
    }
    if (fw_weight_shape.GetDimNum() < 1 || fw_weight_shape.GetDimNum() > 2) {
        return GRAPH_FAILED;
    }
    if (field_shape.GetDimNum() < 2) {
        return GRAPH_FAILED;
    }
    if (index_shape.GetDimNum() < 1) {
        return GRAPH_FAILED;
    }
    if (grad_shape.GetDimNum() != 2) {
        return GRAPH_FAILED;
    }

    if (weight_shape.GetDim(0) != field_shape.GetDim(0)) {
        return GRAPH_FAILED;
    }
    if (weight_shape.GetDim(0) != index_shape.GetDim(0) - 1) {
         return GRAPH_FAILED;
    }
    if (weight_shape.GetDim(2) != grad_shape.GetDim(1)) {
        return GRAPH_FAILED;
    }

    if (field_shape.GetDim(1) != 2) {
        return GRAPH_FAILED;
    }
    for (size_t i = 3; i < field_shape.GetDimNum(); i++) {
        if (field_shape.GetDim(i) != 1) {
            return GRAPH_FAILED;
        }
    }
    for (size_t i = 2; i < index_shape.GetDimNum(); i++) {
        if (index_shape.GetDim(i) != 1) {
            return GRAPH_FAILED;
        }
    }

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseFwFFMGrad, SparseFwFFMGradInferShape);
VERIFY_FUNC_REG(SparseFwFFMGrad, SparseFwFFMGradVerify);

}  // namespace ge
