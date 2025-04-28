#include "sparse_fw_ffm_part2.h"

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

IMPLEMT_COMMON_INFERFUNC(SparseFwFFMPart2InferShape)
{
    std::cout << "[SparseFwFFMPart2] Infer shape begin " << std::endl;
    TensorDesc cross_mean_sum_desc = op.GetInputDescByName("cross_mean_sum");
    Shape fw_weight_shape = op.GetInputDescByName("fw_weight").GetShape();
    Shape cross_mean_sum_shape = cross_mean_sum_desc.GetShape();
    Shape fw_field_map_shape  = op.GetInputDescByName("fw_field_map").GetShape();

    TensorDesc output_desc = op.GetOutputDescByName("output");

    DataType data_type = cross_mean_sum_desc.GetDataType();
    Format format = cross_mean_sum_desc.GetFormat();
    int batch_size = cross_mean_sum_shape.GetDim(0);
    int embedding_size = cross_mean_sum_shape.GetDim(3);

    output_desc.SetShape(Shape({batch_size, embedding_size}));
    output_desc.SetShapeRange({{0, MAX_BATCH_SIZE_RANGE}, {embedding_size, embedding_size}});
    output_desc.SetDataType(data_type);
    output_desc.SetFormat(format);
    (void)op.UpdateOutputDesc("output", output_desc);

    printInfo(op.GetOutputDescByName("output"), "output");
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseFwFFMPart2, SparseFwFFMPart2Verify)
{
    Shape fw_weight_shape = op.GetInputDescByName("fw_weight").GetShape();
    Shape cross_mean_sum_shape = op.GetInputDescByName("cross_mean_sum").GetShape();
    Shape cross_mean_square_sum_shape = op.GetInputDescByName("cross_mean_square_sum").GetShape();
    Shape fw_field_map_shape = op.GetInputDescByName("fw_field_map").GetShape();

    if (fw_weight_shape.GetDimNum() < 1 || fw_weight_shape.GetDimNum() > 2) {
        return GRAPH_FAILED;
    }
    if (cross_mean_sum_shape.GetDimNum() != 4) {
        return GRAPH_FAILED;
    }
    if (cross_mean_square_sum_shape.GetDimNum() != 4) {
        return GRAPH_FAILED;
    }
    if (fw_field_map_shape.GetDimNum() != 2) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseFwFFMPart2, SparseFwFFMPart2InferShape);
VERIFY_FUNC_REG(SparseFwFFMPart2, SparseFwFFMPart2Verify);

}  // namespace ge
