#include "dense_select_input.h"

#include <vector>
#include <numeric>
#include <algorithm>

namespace ge
{
  IMPLEMT_VERIFIER(DenseSelectInput, DenseSelectInputVerify)
  {
    return GRAPH_SUCCESS;
  }

  IMPLEMT_COMMON_INFERFUNC(DenseSelectInputInferShape)
  {
    TensorDesc input_desc =  op.GetInputDesc(0);
    Shape embeddings_shape = input_desc.GetShape();
    DataType data_type = input_desc.GetDataType();
    std::int32_t batch_size = embeddings_shape.GetDim(0);
    
    
    std::vector<int64_t> embedding_sizes;

    int64_t output_embedding_size = 0;
    if (op.GetAttr("embedding_sizes", embedding_sizes) == GRAPH_SUCCESS)
    {
      output_embedding_size = std::accumulate(embedding_sizes.begin(), embedding_sizes.end(), 0);
    }

    auto output_desc =  op.GetOutputDescByName("output");
    Shape output_shape({batch_size,  output_embedding_size});
    output_desc.SetShape(output_shape);
    output_desc.SetDataType(data_type);
    
/*     std::vector<std::pair<int64_t, int64_t>> embeddings_range;
    
    std::vector<std::pair<int64_t, int64_t>> output_range;

    input_desc.GetShapeRange(embeddings_range);
    output_range.push_back(embeddings_range[0]);
    output_range.push_back(std::pair<int64_t, int64_t>{output_embedding_size, output_embedding_size});
    output_desc.SetShapeRange(output_range); */

    return op.UpdateOutputDesc("output", output_desc);
  }

  COMMON_INFER_FUNC_REG(DenseSelectInput, DenseSelectInputInferShape);
  VERIFY_FUNC_REG(DenseSelectInput, DenseSelectInputVerify);

} // namespace ge
