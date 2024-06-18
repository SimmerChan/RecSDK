#ifndef ATTENTION_FUSION_GRAD_KERNEL_H
#define ATTENTION_FUSION_GRAD_KERNEL_H
#include <cstdint>
#include "kernel_operator.h"
#include "v_s_mm_grad.h"
#include "q_k_mm_grad.h"
#include "normalize_grad.h"
#include "utils.h"
using namespace AscendC;

struct AttentionFusionGradArgs {
    GM_ADDR dout;
    GM_ADDR softmaxOut;
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;

    GM_ADDR gradQuery;
    GM_ADDR gradKey;
    GM_ADDR gradValue;
    GM_ADDR gradSoftmax;
    
    int queryDim1;
    int queryDim2;
    int keyDim1;
    int keyDim2;
    int valueDim1;
    int valueDim2;
    int batchNum;

    int numOfNormalnizeOnce;
    int paddingKeyDim1;
    float attenDimSqrt;
    int keyDimAlign;
    
    const TCubeTiling* gardVMatmulTiling;
    const TCubeTiling* gardSMatmulTiling;
    const TCubeTiling* gardQMatmulTiling;
    const TCubeTiling* gardKMatmulTiling;
    const SoftMaxTiling* softmaxtiling;

    const ConfusionTransposeTiling* confusionTransposeTilingData;
    const ConfusionTransposeTiling* confusionTransposeTilingData1;
    const ConfusionTransposeTiling* confusionTransposeTilingData2;
    const ConfusionTransposeTiling* confusionTransposeTilingData3;
};

struct AttentionFusionGradPipe {
    TPipe* pipe;
};

template<typename tType>
class AttentionFusionGradKernel {
    public:
        __aicore__ inline AttentionFusionGradKernel() {};

        __aicore__ inline void Compute(AttentionFusionGradArgs args) {
            // Args
            this->args = args;
            GetBatchOffsetAndLen(args.batchNum, batchOffsetThisCore, batchLenThisCore);
            // Matmul Register
            REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), 
            vSmm.mmGradV, args.gardVMatmulTiling, vSmm.mmGradS, args.gardSMatmulTiling,
            qKmm.mmGradQ, args.gardQMatmulTiling, qKmm.mmGradK, args.gardKMatmulTiling);

            // VSmm Initialize 
            VSMmGradArgs vSmmArgs {
                args.softmaxOut, args.value, args.dout, args.gradValue, args.gradSoftmax, 
                args.queryDim1, args.keyDim1, args.valueDim1, args.valueDim2, args.batchNum, 
                batchOffsetThisCore, batchLenThisCore

            };
            VSMmGradPipeArgs vSmmPieArgs {&pipe};
            
            vSmm.Init(vSmmArgs, vSmmPieArgs);

            // VSmm Initialize 
            QKMmGradArgs QKmmArgs {
                args.query, args.key, args.gradSoftmax, args.gradQuery, args.gradKey, 
                args.queryDim1, args.queryDim2, args.keyDim1, args.keyDim2, args.batchNum, 
                batchOffsetThisCore, batchLenThisCore
            };
            
            qKmm.Init(QKmmArgs); 
            // Start compute
            Process();
        }

        __aicore__ inline void Process() {
            NormalNizeMatmulFusion();

        }
    private:

        __aicore__ inline void NormalNizeMatmulFusion() {

            NormGradArgs normGradArgs {args.softmaxOut, args.gradSoftmax, args.queryDim1, args.keyDim1, args.batchNum, batchOffsetThisCore , batchLenThisCore,
            args.numOfNormalnizeOnce, args.paddingKeyDim1, args.attenDimSqrt, args.keyDimAlign, args.softmaxtiling, args.confusionTransposeTilingData, args.confusionTransposeTilingData1, args.confusionTransposeTilingData2,  
                 args.confusionTransposeTilingData3};
            NormGradPipeArgs normGradPipe {&pipe};

            NormalGradCompute<tType> normalCompute;
            normalCompute.Init(normGradArgs, normGradPipe);


            for (int thisBatch = 0 ; thisBatch < normGradArgs.batchLen; thisBatch++) {
                vSmm.ProcessDS(thisBatch);
            }
            for (int thisBatch = 0 ; thisBatch < normGradArgs.batchLen; thisBatch++) {
                vSmm.ProcessDV(thisBatch);
                normalCompute.ProcessOneBatch(thisBatch);
            }
            for (int thisBatch = 0 ; thisBatch < normGradArgs.batchLen; thisBatch++) {
                qKmm.ProcessDQ(thisBatch);
                qKmm.ProcessDK(thisBatch);
            }
        }

        __aicore__ inline void GetBatchOffsetAndLen(int batchNum, int& batchOffset, int& batchLen) {
            // batch offset
            int blockLenPerCoreBase = batchNum / (GetBlockNum()*2);
            int remain = batchNum % (GetBlockNum()*2);
            if (GetBlockIdx() < remain) {
                batchLen = blockLenPerCoreBase + 1;
                batchOffset = GetBlockIdx() * batchLen;
            } else {
                batchLen = blockLenPerCoreBase;
                batchOffset = GetBlockIdx() * blockLenPerCoreBase + remain;
            }
        }

        AttentionFusionGradArgs args;
        VSMmGradCompute<tType> vSmm;
        QKMmGradCompute<tType> qKmm;
        TPipe pipe;

        int batchOffsetThisCore;
        int batchLenThisCore;

};
#endif