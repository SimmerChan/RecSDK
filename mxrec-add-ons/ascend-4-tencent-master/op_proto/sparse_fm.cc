#include "sparse_fm.h"

#include <iostream>

namespace {
    const int32_t MAX_SAMPLE_FEATURE_SIZE = 65536;
}

namespace ge {

IMPLEMT_COMMON_INFERFUNC(SparseFMInferShape)
{
    std::cout << "[SparseFM] Infer shape ..." << std::endl;
    TensorDesc weight_desc = op.GetInputDescByName("weight");
    const int64_t sample_feature_size = weight_desc.GetShape().GetDim(0);
    const int64_t embedding_size = weight_desc.GetShape().GetDim(1);
    std::cout << "\tsample_feature_size: " << sample_feature_size << std::endl;
    std::cout << "\tembedding_size: " << embedding_size << std::endl;

    DataType data_type = weight_desc.GetDataType();
    Format format = weight_desc.GetFormat();

    auto output_desc = op.GetOutputDescByName("output");
    output_desc.SetShape(Shape({UNKNOWN_DIM, embedding_size}));
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    output_shape_range.emplace_back(std::make_pair(0, MAX_SAMPLE_FEATURE_SIZE));
    output_shape_range.emplace_back(std::make_pair(embedding_size, embedding_size));
    output_desc.SetShapeRange(output_shape_range);
    output_desc.SetDataType(data_type);
    output_desc.SetFormat(format);

    (void)op.UpdateOutputDesc("output", output_desc);
    (void)op.UpdateOutputDesc("cross_mean_sum", output_desc);
    (void)op.UpdateOutputDesc("cross_mean_square_sum", output_desc);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseFM, SparseFMVerify)
{
    TensorDesc weight_desc = op.GetInputDescByName("weight");
    TensorDesc index_desc = op.GetInputDescByName("index");

    if (weight_desc.GetShape().GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    if (index_desc.GetShape().GetDimNum() != 1) {
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseFM, SparseFMInferShape);
VERIFY_FUNC_REG(SparseFM, SparseFMVerify);

}  // namespace ge
