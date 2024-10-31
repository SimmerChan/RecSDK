#include <assert.h>
#include "lccl_all_to_all_tiling.h"
#include "register/op_def_registry.h"

static int magic=9;
static int first=1;
namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        LcclAllToAllTilingData tiling;

        auto sendBuff = context->GetInputTensor(0);
        auto* attrs = context->GetAttrs();
        const auto* rank_ = attrs->GetAttrPointer<int64_t>(0);
        const auto* rank_Size_ = attrs->GetAttrPointer<int64_t>(1);
        int rank = static_cast<int>(*rank_);
        int rankSize = static_cast<int>(*rank_Size_);

        int ipcBufferSize = 200 * 1024 * 1024;
        const char *enviIpcBufferSize = getenv("LCCL_BUFFSIZE");
        if (enviIpcBufferSize != nullptr) {
            assert(std::stoi(enviIpcBufferSize) > 50 && "IPC Buffer size must be greater than 50");
            ipcBufferSize = (std::stoi(enviIpcBufferSize) - 4) / 2 * 1024 * 1024;
        }

        tiling.set_rank(rank);
        tiling.set_rankSize(rankSize);

        tiling.set_magic(magic);
        tiling.set_ipc(ipcBufferSize);

        if (rankSize <= 16) {
            context->SetBlockDim(rankSize * 2);
        } else{
            context->SetBlockDim(32);
        }
        uint32_t sysWorkspaceSize = 16 * 1024 * 1024;
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = sysWorkspaceSize;

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        if(first==1){
            std::cout << "all2all ; magic = " <<magic<< std::endl;
            first=0;
        }
        return ge::GRAPH_SUCCESS;
    }
}


namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        const gert::Shape* x1_shape = context->GetInputShape(0);
        const gert::Shape* x3_shape = context->GetInputShape(2);
        gert::Shape* y_shape = context->GetOutputShape(0);

        y_shape->SetDim(0, x3_shape->GetDim(0));
        y_shape->SetDim(1, x1_shape->GetDim(1));
        y_shape->SetDim(2, 1);

        return GRAPH_SUCCESS;
    }
}


namespace ops {
    class LcclAllToAll : public OpDef {
    public:
        explicit LcclAllToAll(const char* name) : OpDef(name)
        {
            this->Input("send_data")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("send_count_matrix")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT64})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("shape_vec")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("peer_mem")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT64})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("rev_data")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Attr("rank").Int();
            this->Attr("rank_size").Int();
            this->Attr("dim").Int();

            this->SetInferShape(ge::InferShape);

            this->AICore()
                    .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
        }
    };

    OP_ADD(LcclAllToAll);
}
