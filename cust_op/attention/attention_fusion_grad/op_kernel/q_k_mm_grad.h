#ifndef QK_MM_GRAD_H
#define QK_MM_GRAD_H
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "utils.h"
using namespace AscendC;

struct QKMmGradArgs {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR gradSoftmax;
    GM_ADDR gradQuery;
    GM_ADDR gradKey;
    
    int queryDim1;
    int queryDim2;

    int keyDim1;
    int keyDim2;  


    int batchNum;
    int batchOffset;
    int batchLen;
};

template<typename tType>
class QKMmGradCompute {
public:
    __aicore__ inline QKMmGradCompute(){}

    __aicore__ inline void Init(QKMmGradArgs mmArgs)
    {
        this->mmArgs = mmArgs;
        query.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.query), mmArgs.batchNum * mmArgs.queryDim1 * mmArgs.queryDim2);
        query = query[mmArgs.batchOffset * mmArgs.queryDim1 * mmArgs.queryDim2];

        key.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.key), mmArgs.batchNum * mmArgs.keyDim1 * mmArgs.keyDim2);
        key = key[mmArgs.batchOffset * mmArgs.keyDim1 * mmArgs.keyDim2];

        gradSoftmax.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.gradSoftmax), mmArgs.batchNum * mmArgs.queryDim1 * mmArgs.keyDim1);
        gradSoftmax = gradSoftmax[mmArgs.batchOffset * mmArgs.queryDim1 * mmArgs.keyDim1];

        gradQuery.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.gradQuery), mmArgs.batchNum * mmArgs.queryDim1 * mmArgs.queryDim2);
        gradQuery = gradQuery[mmArgs.batchOffset * mmArgs.queryDim1 * mmArgs.queryDim2];

        gradKey.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.gradKey), mmArgs.batchNum * mmArgs.keyDim1 * mmArgs.keyDim2);
        gradKey = gradKey[mmArgs.batchOffset * mmArgs.keyDim1 * mmArgs.keyDim2];
    }


    
    __aicore__ inline void ProcessDQ(uint32_t batchI)
    {
        if (batchI != 0) {
            mmGradQ.WaitIterateAll();
            mmGradQ.End();
        }
        mmGradQ.SetTensorA(gradSoftmax[batchI * mmArgs.queryDim1 * mmArgs.keyDim1]);
        mmGradQ.SetTensorB(key[batchI * mmArgs.keyDim1 * mmArgs.keyDim2]);

        mmGradQ.template IterateAll<false>(gradQuery[batchI * mmArgs.queryDim1 * mmArgs.queryDim2], 0, false, true);
    }

    __aicore__ inline void ProcessDK(uint32_t batchI)
    {
        if (batchI != 0) {
            mmGradK.WaitIterateAll();
            mmGradK.End();
        }
        mmGradK.SetTensorA(gradSoftmax[batchI * mmArgs.queryDim1 * mmArgs.keyDim1], true);
        mmGradK.SetTensorB(query[batchI * mmArgs.queryDim1 * mmArgs.queryDim2]);

        mmGradK.template IterateAll<false>(gradKey[batchI * mmArgs.keyDim1 * mmArgs.keyDim2], 0, false, true);
    }

    matmul::Matmul<
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType>
        > mmGradQ;

    matmul::Matmul<
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, true>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>,
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType>
        > mmGradK;

    private:
        QKMmGradArgs mmArgs;
        // C = A*BT dA=C*B (vec) dB=CT*A (Cube)
        GlobalTensor<tType> query;
        GlobalTensor<tType> key;
        GlobalTensor<tType> gradSoftmax;
        GlobalTensor<tType> gradQuery;
        GlobalTensor<tType> gradKey;
};
#endif