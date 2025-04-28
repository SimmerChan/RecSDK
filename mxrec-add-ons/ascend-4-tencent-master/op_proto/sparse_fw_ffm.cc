#include "sparse_fw_ffm.h"

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

namespace ge {

namespace {
    const int32_t MAX_BATCH_SIZE_RANGE = 65536;
}

namespace {
    void printInfo(const TensorDesc& desc, const char* name) {
        std::cout << "[" << name << "]:" << std::endl;

        Shape shape = desc.GetShape();
        std::cout << "\tShape: (";
        for (size_t d = 0; d < shape.GetDimNum(); d++) {
            std::cout << shape.GetDim(d) << ",";
        }
        std::cout << ")" << std::endl;

        std::cout << "\tData type: " << desc.GetDataType() << std::endl;
        std::cout << "\tData format: " << desc.GetFormat() << std::endl;
        std::cout << "\tSize: " << desc.GetSize() << std::endl;
        std::cout << "\tRealDimCnt: " << desc.GetRealDimCnt() << std::endl;
        
        std::cout << "\tShape range: [";
        std::vector<std::pair<int64_t, int64_t>> shape_range;
        desc.GetShapeRange(shape_range);
        for (const auto& range: shape_range) {
            std::cout << "(" << range.first << "," << range.second << "), ";
        }
        std::cout << "]" << std::endl;
    }
}

IMPLEMT_COMMON_INFERFUNC(SparseFwFFMInferShape)
{
    std::cout << "[SparseFwFFM] Infer shape ..." << std::endl;
    TensorDesc weight_desc = op.GetInputDescByName("weight");
    Shape weight_shape = weight_desc.GetShape();
    Shape fw_weight_shape = op.GetInputDescByName("fw_weight").GetShape();
    Shape field_shape = op.GetInputDescByName("field").GetShape();
    Shape index_shape = op.GetInputDescByName("index").GetShape();

    printInfo(op.GetInputDescByName("weight"), "weight");
    printInfo(op.GetInputDescByName("fw_weight"), "fw_weight");
    printInfo(op.GetInputDescByName("field"), "field");
    printInfo(op.GetInputDescByName("index"), "index");

    TensorDesc output_desc = op.GetOutputDescByName("output");
    TensorDesc cross_mean_sum_desc = op.GetOutputDescByName("cross_mean_sum");
    TensorDesc fw_field_map_desc = op.GetOutputDescByName("fw_field_map");

    DataType data_type = weight_desc.GetDataType();
    Format format = weight_desc.GetFormat();
    int field_num = weight_shape.GetDim(1);
    int embedding_size = weight_shape.GetDim(2);

    output_desc.SetShape(Shape({UNKNOWN_DIM, embedding_size}));
    output_desc.SetShapeRange({{0, MAX_BATCH_SIZE_RANGE}, {embedding_size, embedding_size}});
    output_desc.SetDataType(data_type);
    output_desc.SetFormat(format);
    (void)op.UpdateOutputDesc("output", output_desc);

    const int32_t fw_field_num =
        floor(sqrt(2 * fw_weight_shape.GetDim(fw_weight_shape.GetDimNum() - 1)));
    
    cross_mean_sum_desc.SetShapeRange({{0, MAX_BATCH_SIZE_RANGE}, {field_num, field_num},
        {fw_field_num, fw_field_num}, {embedding_size, embedding_size}});
    cross_mean_sum_desc.SetShape(Shape({UNKNOWN_DIM, field_num, fw_field_num, embedding_size}));
    cross_mean_sum_desc.SetDataType(data_type);
    cross_mean_sum_desc.SetFormat(format);
    (void)op.UpdateOutputDesc("cross_mean_sum", cross_mean_sum_desc);
    (void)op.UpdateOutputDesc("cross_mean_square_sum", cross_mean_sum_desc);

    fw_field_map_desc.SetShape(Shape({UNKNOWN_DIM, fw_field_num}));
    fw_field_map_desc.SetShapeRange({{0, MAX_BATCH_SIZE_RANGE}, {fw_field_num, fw_field_num}});
    fw_field_map_desc.SetDataType(DT_INT32);
    fw_field_map_desc.SetFormat(FORMAT_ND);
    (void)op.UpdateOutputDesc("fw_field_map", fw_field_map_desc);

    // printInfo(op.GetOutputDescByName("output"), "output");
    // printInfo(op.GetOutputDescByName("cross_mean_sum"), "cross_mean_sum");
    // printInfo(op.GetOutputDescByName("cross_mean_square_sum"), "cross_mean_square_sum");
    // printInfo(op.GetOutputDescByName("fw_field_map"), "fw_field_map");

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseFwFFM, SparseFwFFMVerify)
{
    Shape weight_shape =  op.GetInputDescByName("weight").GetShape();
    Shape fw_weight_shape = op.GetInputDescByName("fw_weight").GetShape();
    Shape field_shape = op.GetInputDescByName("field").GetShape();
    Shape index_shape = op.GetInputDescByName("index").GetShape();

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

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseFwFFM, SparseFwFFMInferShape);
VERIFY_FUNC_REG(SparseFwFFM, SparseFwFFMVerify);

}  // namespace ge
