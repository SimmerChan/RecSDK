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
    
    int M;
    int N;
    int K;

    int batchNum;
    int batchOffset;
    int batchLen;
    
    const TCubeTiling* qkMatmulTiling;
};

struct QKBmmPipeArgs {
    TPipe* pipe;
};

// template<typename qType, typename kType>
// class QKBmmCompute {
// public:
//     __aicore__ inline QKBmmCompute(){}
//     __aicore__ inline void Compute(QKBmmArgs qKBmmArgs, QKBmmPipeArgs pipeArgs) {
//         // batch offset
//         uint32_t blockLenPerCore = CeilDive(qKBmmArgs.batchNum, (GetBlockNum()*2));
//         uint32_t batchOffset = blockLenPerCore*GetBlockIdx();
//         uint32_t batchLen = blockLenPerCore;
//         qKBmmArgs.qkMatmulTiling->BatchNum = 1;
//         if (batchOffset+batchLen > qKBmmArgs.batchNum) {
//             batchLen = qKBmmArgs.batchNum - batchOffset;
//         }
//         matmul::Matmul<matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType, false, LayoutMode::NORMAL>, 
//             matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, kType, true, LayoutMode::NORMAL>, 
//             matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType, false, LayoutMode::NORMAL>, 
//             matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType>, MM_CFG_MATMUL> 
//             mm;

//         REGIST_MATMUL_OBJ(pipeArgs.pipe, GetSysWorkSpacePtr(), mm, qKBmmArgs.qkMatmulTiling);


//         GlobalTensor<qType> qGlobal;
//         GlobalTensor<kType> kGlobal;
//         GlobalTensor<kType> outGlobal;

//         qGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(qKBmmArgs.query), batchLen * qKBmmArgs.M * qKBmmArgs.K);
//         qGlobal = qGlobal[batchOffset * qKBmmArgs.M * qKBmmArgs.K];
//         kGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ kType*>(qKBmmArgs.key), batchLen * qKBmmArgs.N * qKBmmArgs.K);
//         kGlobal = kGlobal[batchOffset * qKBmmArgs.N * qKBmmArgs.K];
//         outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ kType*>(qKBmmArgs.out), batchLen * qKBmmArgs.M * qKBmmArgs.N);
//         outGlobal = outGlobal[batchOffset * qKBmmArgs.M * qKBmmArgs.N];

//         mm.SetTensorA(qGlobal);
//         mm.SetTensorB(kGlobal, true);
//         mm.IterateBatch(outGlobal, batchLen, batchLen, false);
//         mm.End();
//     }


// };

template<typename qType, typename kType>
class QKBmmCompute {
public:
    __aicore__ inline QKBmmCompute(){}

    __aicore__ inline void Init(QKBmmArgs qKBmmArgs, QKBmmPipeArgs pipeArgs){
        this->qKBmmArgs = qKBmmArgs;

        // kernel batch offset
        qGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ qType*>(qKBmmArgs.query), qKBmmArgs.batchLen * qKBmmArgs.M * qKBmmArgs.K);
        qGlobal = qGlobal[qKBmmArgs.batchOffset * qKBmmArgs.M * qKBmmArgs.K];
        kGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ kType*>(qKBmmArgs.key), qKBmmArgs.batchLen * qKBmmArgs.N * qKBmmArgs.K);
        kGlobal = kGlobal[qKBmmArgs.batchOffset * qKBmmArgs.N * qKBmmArgs.K];
        outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ kType*>(qKBmmArgs.out), qKBmmArgs.batchLen * qKBmmArgs.M * qKBmmArgs.N);
        outGlobal = outGlobal[qKBmmArgs.batchOffset * qKBmmArgs.M * qKBmmArgs.N];

    }

    __aicore__ inline void Process(){
        for (int thisBatch = 0 ; thisBatch < qKBmmArgs.batchLen; thisBatch++) {
            mm.SetTensorA(qGlobal[thisBatch * qKBmmArgs.M * qKBmmArgs.K]);
            mm.SetTensorB(kGlobal[thisBatch * qKBmmArgs.N * qKBmmArgs.K], true);
            // mm.SetWorkspace(outGlobal[thisBatch * qKBmmArgs.M * qKBmmArgs.N].GetPhyAddr(), qKBmmArgs.M * qKBmmArgs.N);
            // while(mm.template Iterate<false>()) {
            //     mm.template GetTensorC<false>();
            // }
            mm.IterateAll(outGlobal[thisBatch * qKBmmArgs.M * qKBmmArgs.N], 0, false);
        }
        mm.End();
    }
    
    matmul::Matmul<
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType, false>, 
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, kType, true>, 
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType, false>, 
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, qType>
        > 
        mm;
private:
    QKBmmArgs qKBmmArgs;
    GlobalTensor<qType> qGlobal;
    GlobalTensor<kType> kGlobal;
    GlobalTensor<kType> outGlobal; 
};
#endif