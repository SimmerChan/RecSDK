#ifndef QK_BMMM_COMPUTE__H
#define QK_BMMM_COMPUTE__H
#include <cstdint>
#include "attention_fusion_kernel.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

struct QKBmmArgs {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR out;
    
    int dimM;
    int dimN;
    int dimK;

    int batchNum;
    int batchOffset;
    int batchLen;
    
    const TCubeTiling* qkMatmulTiling;
};

struct QKBmmPipeArgs {
    TPipe* pipe;
};

template<typename qType, typename kType>
class QKBmmCompute {
public:
    __aicore__ inline QKBmmCompute() {}

    __aicore__ inline void Init(QKBmmArgs qKBmmArgs, QKBmmPipeArgs pipeArgs)
    {
        this->qKBmmArgs = qKBmmArgs;

        // kernel batch offset
        qGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(qKBmmArgs.query),
                                qKBmmArgs.batchLen * qKBmmArgs.dimM * qKBmmArgs.dimK);
        qGlobal = qGlobal[qKBmmArgs.batchOffset * qKBmmArgs.dimM * qKBmmArgs.dimK];

        kGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ kType*>(qKBmmArgs.key),
                                qKBmmArgs.batchLen * qKBmmArgs.dimN * qKBmmArgs.dimK);
        kGlobal = kGlobal[qKBmmArgs.batchOffset * qKBmmArgs.dimN * qKBmmArgs.dimK];

        outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ kType*>(qKBmmArgs.out),
                                qKBmmArgs.batchLen * qKBmmArgs.dimM * qKBmmArgs.dimN);
        outGlobal = outGlobal[qKBmmArgs.batchOffset * qKBmmArgs.dimM * qKBmmArgs.dimN];
    }

    __aicore__ inline void Process()
    {
        for (int thisBatch = 0 ; thisBatch < qKBmmArgs.batchLen; thisBatch++) {
            mm.SetTensorA(qGlobal[thisBatch * qKBmmArgs.dimM * qKBmmArgs.dimK]);
            mm.SetTensorB(kGlobal[thisBatch * qKBmmArgs.dimN * qKBmmArgs.dimK], true);

            mm.IterateAll(outGlobal[thisBatch * qKBmmArgs.dimM * qKBmmArgs.dimN], 0, false);
        }
        mm.End();
    }
    
    matmul::Matmul<
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType <matmul::TPosition::GM, CubeFormat::ND, kType, true>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType>
        > mm;
private:
    QKBmmArgs qKBmmArgs;
    GlobalTensor<qType> qGlobal;
    GlobalTensor<kType> kGlobal;
    GlobalTensor<kType> outGlobal;
};
#endif