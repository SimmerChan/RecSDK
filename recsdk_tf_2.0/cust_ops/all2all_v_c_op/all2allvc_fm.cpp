#include "kernel_operator.h"
#include "op_def.h"
#include "collectives.h"

#include "all2allvc_fm.h"

#define ALL2ALLVC_MESH_CLASS_OP_LAUNCH(name, type)         \
    do {                                                \
        name<type> opKernel(rank, rankSize, extraFlag); \
        opKernel.Init(KERNELS_ARGS_CALL());          \
        opKernel.Process();                             \
    } while (0)

#define LCCL_ALL2ALLVC_MESH_FUNC_AUTO_DEF(type)                                            \
    extern "C" __global__ __aicore__ void LcalAll2AllVC_Mesh_##type(KERNELS_ARGS_FUN()) \
    {                                                                                     \
        GET_COMM_ARGS;                                                                    \
        ALL2ALLVC_MESH_CLASS_OP_LAUNCH(All2AllVCFM, type);                             \
    }

#if defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__) || defined(__DAV_L311__)
LCCL_910_95_TYPE_FUNC(LCCL_ALL2ALLVC_MESH_FUNC_AUTO_DEF);
#endif