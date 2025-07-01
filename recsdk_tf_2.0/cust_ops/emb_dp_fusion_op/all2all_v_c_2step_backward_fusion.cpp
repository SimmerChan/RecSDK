#include "op_def.h"
#include "collectives.h"
#include "all2all_v_c_2step_backward_fusion.h"

#define ALL2ALL_BACKWARD_FUSION_CLASS_OP_LAUNCH(name, type) \
    do {                                                    \
        name<type> opKernel(rank, rankSize, extraFlag);     \
        opKernel.Init(EMB_BACKWARD_FUSION_ARGS_CALL());     \
        opKernel.Process();                                 \
    } while (0)

#define LCCL_ALL2ALLVC_BACKWARD_FUSION_FUNC_AUTO_DEF(type)                                                     \
    extern "C" __global__ __aicore__ void LcalAll2AllVC_BACKWARD_FUSION_##type(EMB_BACKWARD_FUSION_ARGS_FUN()) \
    {                                                                                                          \
        GET_COMM_ARGS;                                                                                         \
        ALL2ALL_BACKWARD_FUSION_CLASS_OP_LAUNCH(All2AllVC2StepBackwardFusion, type);                           \
    }

#if defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__) || defined(__DAV_L311__)
LCCL_910_95_TYPE_FUNC(LCCL_ALL2ALLVC_BACKWARD_FUSION_FUNC_AUTO_DEF);
#endif