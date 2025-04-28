#include "sparse_fields_concat_v2_grad.h"

namespace ge {

    IMPLEMT_COMMON_INFERFUNC(SparseFieldsConcatV2GradInferShape)
    {
        TensorDesc td_output = op.GetOutputDescByName("output");
        td_output.SetShape(op.GetInputDescByName("weight").GetShape());
        td_output.SetDataType(op.GetInputDescByName("weight").GetDataType());

        (void)op.UpdateOutputDesc("output", td_output);
        return GRAPH_SUCCESS;
    }

    IMPLEMT_VERIFIER(SparseFieldsConcatV2Grad, SparseFieldsConcatV2GradVerify)
    {
        // todo 待网络打通后再撰写verify函数
 /*     OP_REQUIRES(context, weight_tensor.dims() == 3,
        errors::InvalidArgument("weight_tensor must be 3-dimensional"));
        OP_REQUIRES(context, field_tensor.dims() >= 2,
        errors::InvalidArgument("field_tensor must be at least 2-dimensional"));
        OP_REQUIRES(context, grad_part1_tensor.dims() == 2,
        errors::InvalidArgument("grad_part1_tensor must be 2-dimensional"));
        OP_REQUIRES(context, grad_part2_tensor.dims() == 2 || grad_part2_tensor.dims() == 3,
        errors::InvalidArgument("grad_part2_tensor must be 2-dimensional or 3-dimensional"));
        OP_REQUIRES(context, weight_tensor.dim_size(0) == field_tensor.dim_size(0),
        errors::InvalidArgument("sample fuature size must be equal between weight "
        "tensor dim-0 (%d) and field tensor dim-0 (%d)",
        weight_tensor.dim_size(0), field_tensor.dim_size(0)));

        OP_REQUIRES(context, weight_tensor.dim_size(0) == index_tensor.dim_size(0) - 1,
        errors::InvalidArgument("sample feature size must be equal between weight "
        "tensor dim-0 (%d) and index tensor dim-0 (%d) - 1",
        weight_tensor.dim_size(0), index_tensor.dim_size(0)));
        OP_REQUIRES(context, field_tensor.dim_size(1) == 2,
        errors::InvalidArgument("field tensor dim-1 (%d) is not equal to 2",
        field_tensor.dim_size(1)));
        OP_REQUIRES(context, keys_per_field_tensor.dim_size(0) == grad_part1_tensor.dim_size(0),
        errors::InvalidArgument("keys_per_field_tensor dim-0 (%d) is not match with "
        "grad_part1_tensor dim-0 (%d)",
        keys_per_field_tensor.dim_size(0),
        grad_part1_tensor.dim_size(0)));
        OP_REQUIRES(context, keys_per_field_tensor.dim_size(0) == grad_part2_tensor.dim_size(0),
        errors::InvalidArgument("keys_per_field_tensor dim-0 (%d) is not match with "
        "grad_part2_tensor dim-0 (%d)",
        keys_per_field_tensor.dim_size(0),
        grad_part2_tensor.dim_size(0)));
        OP_REQUIRES(context, keys_per_field_tensor.dim_size(1) == fw_field_num_,
        errors::InvalidArgument("keys_per_field_tensor dim-1 (%d) is not match with "
        "fw_field_num (%d)",
        keys_per_field_tensor.dim_size(1), fw_field_num_));

        for (int32_t i = 3; i < field_tensor.dims(); ++i)
        {
            OP_REQUIRES(context, field_tensor.dim_size(i) == 1,
            errors::InvalidArgument("field tensor dim-%d (%d) is not equal to 1",
            i, field_tensor.dim_size(i)));
        }
        OP_REQUIRES(context, index_tensor.dims() == 1 || index_tensor.dims() == 2,
        errors::InvalidArgument("index tensor must be 1-dimensional or 2-dimensional"));
        if (index_tensor.dims() == 2)
        {
            OP_REQUIRES(context, index_tensor.dim_size(1) == 2,
            errors::InvalidArgument("index tensor dim-1 (%d) is not equal to 2",
            index_tensor.dim_size(1)));
            OP_REQUIRES(context, keys_per_field_tensor.dim_size(2) == grad_part2_tensor.dim_size(1),
            errors::InvalidArgument("keys_per_field_tensor dim-2 (%d) is not match with "
            "grad_part2_tensor dim-1 (%d)",
            keys_per_field_tensor.dim_size(2),
            grad_part2_tensor.dim_size(1)));
        }*/
        return GRAPH_SUCCESS;
    }

    COMMON_INFER_FUNC_REG(SparseFieldsConcatV2Grad, SparseFieldsConcatV2GradInferShape);
    VERIFY_FUNC_REG(SparseFieldsConcatV2Grad, SparseFieldsConcatV2GradVerify);

}
// namespace ge
