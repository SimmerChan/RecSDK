#ifndef VS_MM_GRAD_H
#define VS_MM_GRAD_H
#include <cstdint>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "utils.h"
using namespace AscendC;

struct VSMmGradArgs {
    GM_ADDR softmaxOut;
    GM_ADDR value;
    GM_ADDR dout;
    GM_ADDR gradValue;
    GM_ADDR gradSoftmax;
    
    int sDim1;
    int sDim2;

    int vDim1;
    int vDim2;  


    int batchNum;
    int batchOffset;
    int batchLen;
};

struct VSMmGradPipeArgs {
    TPipe* pipe;
};

template<typename tType>
class VSMmGradCompute {
public:
    __aicore__ inline VSMmGradCompute(){}

    __aicore__ inline void Init(VSMmGradArgs mmArgs, VSMmGradPipeArgs pipeArgs){
        this->mmArgs = mmArgs;
        softmaxOut.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.softmaxOut), mmArgs.batchNum * mmArgs.sDim1 * mmArgs.sDim2);
        softmaxOut = softmaxOut[mmArgs.batchOffset * mmArgs.sDim1 * mmArgs.sDim2];

        value.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.value), mmArgs.batchNum * mmArgs.vDim1 * mmArgs.vDim2);
        value = value[mmArgs.batchOffset * mmArgs.vDim1 * mmArgs.vDim2];

        dout.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.dout), mmArgs.batchNum * mmArgs.sDim1 * mmArgs.vDim2);
        dout = dout[mmArgs.batchOffset * mmArgs.sDim1 * mmArgs.vDim2];

        gradS.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.gradSoftmax), mmArgs.batchNum * mmArgs.sDim1 * mmArgs.sDim2);
        gradS = gradS[mmArgs.batchOffset * mmArgs.sDim1 * mmArgs.sDim2];

        gradValue.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(mmArgs.gradValue), mmArgs.batchNum * mmArgs.vDim1 * mmArgs.vDim2);
        gradValue = gradValue[mmArgs.batchOffset * mmArgs.vDim1 * mmArgs.vDim2];
    }

    __aicore__ inline void Compute(){
        for (int thisBatch = 0 ; thisBatch < mmArgs.batchLen; thisBatch++) {
            ProcessDV(thisBatch);
            ProcessDS(thisBatch);
        }
    }
    
    __aicore__ inline void ProcessDV(uint32_t batchI){
        if (batchI != 0) {
            mmGradV.WaitIterateAll();
            mmGradV.End();
        }
        mmGradV.SetTensorA(softmaxOut[batchI * mmArgs.sDim1 * mmArgs.sDim2], true);
        mmGradV.SetTensorB(dout[batchI * mmArgs.sDim1 * mmArgs.vDim2]);

        mmGradV.template IterateAll<false>(gradValue[batchI * mmArgs.vDim1 * mmArgs.vDim2], 0, false, true);
        // mm.IterateAll(dB[batchI * mmArgs.vDim1 * mmArgs.vDim2], 0, false);
    }

    __aicore__ inline void ProcessDS(uint32_t batchI){
        if (batchI != 0) {
            mmGradS.WaitIterateAll();
            mmGradS.End();
        }
        mmGradS.SetTensorA(dout[batchI * mmArgs.sDim1 * mmArgs.vDim2]);
        mmGradS.SetTensorB(value[batchI * mmArgs.vDim1 * mmArgs.vDim2], true);

        mmGradS.template IterateAll<false>(gradS[batchI * mmArgs.sDim1 * mmArgs.sDim2], 0, false, true);
    }

    matmul::Matmul<
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, true>, 
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>, 
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>, 
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType>
        > 
        mmGradV;

    matmul::Matmul<
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>, 
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, true>, 
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType, false>, 
        matmul::MatmulType<matmul::TPosition::GM, CubeFormat::ND, tType>
        > 
        mmGradS;

    private:
        VSMmGradArgs mmArgs;
        VSMmGradPipeArgs pipeArg;
        // C = A*B dA=C*BT (vec) dB=AT*C (Cube)
        GlobalTensor<tType> softmaxOut;
        GlobalTensor<tType> value;
        GlobalTensor<tType> dout;
        GlobalTensor<tType> gradS;
        GlobalTensor<tType> gradValue;
};
#endif