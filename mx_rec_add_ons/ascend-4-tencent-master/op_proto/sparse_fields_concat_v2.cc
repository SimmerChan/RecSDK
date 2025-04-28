#include "graph/utils/op_desc_utils.h"
#include "sparse_fields_concat_v2.h"
#include "iostream"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(SparseFieldsConcatV2InferShape)
{
    TensorDesc weight_desc = op.GetInputDesc(0);
    TensorDesc field_desc = op.GetInputDesc(1);
    TensorDesc index_desc = op.GetInputDesc(2);

    const int64_t weight = weight_desc.GetShape().GetDim(0);
    const int64_t field_num = weight_desc.GetShape().GetDim(1);
    const int64_t embedding_size = weight_desc.GetShape().GetDim(2);
    DataType data_type = weight_desc.GetDataType();
    Format format = weight_desc.GetFormat();
    DataType data_type_int = field_desc.GetDataType();
    std::vector<int64_t> index_num = index_desc.GetShape().GetDims();
    std::int32_t index_rank = index_num.size();
    const int32_t index_shape = static_cast<int32_t>(index_desc.GetShape().GetDim(0));

    int64_t fw_field_num;
    op.GetAttr("fw_field_num", fw_field_num);
    std::vector<int32_t> part1_fields_num;
    op.GetAttr("part1_fields", part1_fields_num);
    std::vector<int32_t> part2_fields_num;
    op.GetAttr("part2_fields", part2_fields_num);

    int32_t part1_fields_size = part1_fields_num.size();
    int32_t part2_fields_size = part2_fields_num.size();

    TensorDesc output_part1_desc = op.GetOutputDescByName("output_part1");
    TensorDesc output_part2_desc = op.GetOutputDescByName("output_part2");
    TensorDesc keys_per_field_desc = op.GetOutputDescByName("keys_per_field");

    int32_t batch_size=-1;

    const vector<string> depend_names = {"index"};
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    op_desc->SetOpInferDepends(depend_names);

    Tensor index_tensor;
    if (op.GetInputConstData("index", index_tensor) == GRAPH_SUCCESS) {
        auto size_data = reinterpret_cast<const int32_t *>(index_tensor.GetData());
        batch_size = static_cast<int32_t>(size_data[index_shape-1]);
    }

    std::cout <<"\n batch_size: " << batch_size << std::endl;
    output_part1_desc.SetShape(Shape({batch_size, part1_fields_size*field_num*embedding_size}));
    if (index_rank == 2) {
        output_part2_desc.SetShape(Shape({batch_size, batch_size, part2_fields_size*field_num*embedding_size}));
        keys_per_field_desc.SetShape(Shape({batch_size, fw_field_num, batch_size}));
    } else {
        output_part2_desc.SetShape(Shape({batch_size, part2_fields_size*field_num*embedding_size}));
        keys_per_field_desc.SetShape(Shape({batch_size, fw_field_num}));
    }
    std::cout << "\noutput_part1_desc.GetShape.GetDim(1): " <<  output_part1_desc.GetShape().GetDim(1) << std::endl;

    output_part1_desc.SetDataType(data_type);
    output_part1_desc.SetFormat(format);
    output_part2_desc.SetDataType(data_type);
    output_part2_desc.SetFormat(format);
    keys_per_field_desc.SetDataType(data_type_int);
    keys_per_field_desc.SetFormat(format);

    (void)op.UpdateOutputDesc("output_part1", output_part1_desc);
    (void)op.UpdateOutputDesc("output_part2", output_part2_desc);
    (void)op.UpdateOutputDesc("keys_per_field", keys_per_field_desc);
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseFieldsConcatV2, SparseFieldsConcatV2Verify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseFieldsConcatV2, SparseFieldsConcatV2InferShape);
VERIFY_FUNC_REG(SparseFieldsConcatV2, SparseFieldsConcatV2Verify);

}  // namespace ge
