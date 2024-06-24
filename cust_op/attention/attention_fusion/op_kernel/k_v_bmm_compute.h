#ifndef KV_BMMM_COMPUTE__H
#define KV_BMMM_COMPUTE__H

#include "kernel_operator.h"
using namespace AscendC;


struct KVBmmArgs {
    GM_ADDR softmaxOut;
    GM_ADDR value;
    GM_ADDR out;
    
    int M;
    int N;
    int K;

    int batchNum;
    int batchOffset;
    int batchLen;
};

struct KVBmmPipeArgs {
    TPipe* pipe;
};

template<typename sType, typename vType>
class KVBmmCompute {
public:
    __aicore__ inline KVBmmCompute(){}

    __aicore__ inline void Init(KVBmmArgs kvBmmArgs, KVBmmPipeArgs pipeArgs)
    {
        this->kvBmmArgs = kvBmmArgs;
        this->pipeArgs = pipeArgs;

        // kernel batch offset
        sGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ sType*>(kvBmmArgs.softmaxOut), kvBmmArgs.M * kvBmmArgs.K);
        sGlobal = sGlobal[kvBmmArgs.batchOffset * kvBmmArgs.M * kvBmmArgs.K];
        vGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ vType*>(kvBmmArgs.value),  kvBmmArgs.N * kvBmmArgs.K);
        vGlobal = vGlobal[kvBmmArgs.batchOffset * kvBmmArgs.N * kvBmmArgs.K];
        outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ vType*>(kvBmmArgs.out), kvBmmArgs.M * kvBmmArgs.N);
        outGlobal = outGlobal[kvBmmArgs.batchOffset * kvBmmArgs.M * kvBmmArgs.N];
    }

    __aicore__ inline void ComputeOneBatch(int batchI)
    {
        if (batchI != 0) {
            mm.WaitIterateAll();
            mm.End();
        }

        event_t evenId = static_cast<event_t>(pipeArgs.pipe->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(evenId);
        WaitFlag<HardEvent::MTE3_MTE2>(evenId);
        mm.SetTensorA(sGlobal[batchI * kvBmmArgs.M * kvBmmArgs.K]);
        mm.SetTensorB(vGlobal[batchI * kvBmmArgs.N * kvBmmArgs.K]);

        mm.template IterateAll<false>(outGlobal[batchI * kvBmmArgs.M * kvBmmArgs.N], 0, false, true);
    }
    
    matmul::Matmul<
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, sType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, vType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, vType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, vType>
        > mm;
private:
    KVBmmArgs kvBmmArgs;
    KVBmmPipeArgs pipeArgs;
    GlobalTensor<sType> sGlobal;
    GlobalTensor<vType> vGlobal;
    GlobalTensor<vType> outGlobal;
};
#endif