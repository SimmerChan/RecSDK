#include "dense_select_input_v2.h"
#include <string>
#include <vector>


namespace ge
{
  IMPLEMT_VERIFIER(DenseSelectInputV2, DenseSelectInputV2Verify)
  {
    return GRAPH_SUCCESS;
  }

IMPLEMT_COMMON_INFERFUNC(DenseSelectInputV2InferShape){

    TensorDesc embeddings = op.GetInputDescByName("embeddings");
    DataType data_type = embeddings.GetDataType();
    Format format = embeddings.GetFormat();
    Shape embeddings_shape = embeddings.GetShape();
    std::int32_t batch_size = embeddings_shape.GetDim(0);
    std::vector<int32_t> select_parts_;
    std::vector<int32_t> embedding_sizes_;
    int32_t output_part1_embedding_size = 0;
    int32_t output_part2_embedding_size = 0;
    op.GetAttr("select_parts", select_parts_);
    op.GetAttr("embedding_sizes", embedding_sizes_);
    for (size_t i = 0; i < select_parts_.size(); ++i){
        if (select_parts_[i] == 2){
            output_part2_embedding_size += embedding_sizes_[i];
        } else{
            output_part1_embedding_size += embedding_sizes_[i];
        }
    }
    auto output1_desc = op.GetOutputDescByName("output_part1");
    if (output_part1_embedding_size > 0){
        output1_desc.SetShape(Shape({batch_size, output_part1_embedding_size}));
    } else{
        output1_desc.SetShape(Shape({1}));
    }
    auto output2_desc = op.GetOutputDescByName("output_part2");
    if (output_part2_embedding_size > 0){
        output2_desc.SetShape(Shape({batch_size, output_part2_embedding_size}));
    } else{
        output2_desc.SetShape(Shape({1}));
    }
     output1_desc.SetDataType(data_type);
     output2_desc.SetDataType(data_type);
     output1_desc.SetFormat(format);
     output2_desc.SetFormat(format);
     (void)op.UpdateOutputDesc("output_part1", output1_desc);
     (void)op.UpdateOutputDesc("output_part2", output2_desc);

     return GRAPH_SUCCESS;
}

  COMMON_INFER_FUNC_REG(DenseSelectInputV2, DenseSelectInputV2InferShape);
  VERIFY_FUNC_REG(DenseSelectInputV2, DenseSelectInputV2Verify);
}