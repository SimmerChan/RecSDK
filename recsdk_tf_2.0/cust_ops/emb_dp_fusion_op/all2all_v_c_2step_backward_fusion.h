#ifndef LCCL_ALL2ALLVC_2STEP_H
#define LCCL_ALL2ALLVC_2STEP_H
#include "kernel_operator.h"
#include "collectives.h"
#include "ipc_queue.h"
#include "lcal_910_95_const.h"

using namespace AscendC;

template <typename T>
class All2AllVC2StepBackwardFusion : public Collectives {
    constexpr static int INVALID_RANK_NUM = 0xFFFFFFFF;  // 非法rank
    constexpr static int64_t CORE_NUMS_PER_STAGE = 16;   // 每个阶段提供的最大核数
    constexpr static int64_t SHARE_QUE_DEPTH = 8;        // 单个共享队列深度
    constexpr static int64_t DATA_SLICE = 2;
    constexpr static int64_t STAGE_NUM = 2;
    constexpr static int64_t MAX_FLAG_OFFSET = 256;
    constexpr static int64_t SINGLE_RANK_MAX_NUM = CORE_NUMS_PER_STAGE;
    constexpr static int64_t MULTI_RANK_SIZE = (LCAL_MAX_RANK_SIZE + SINGLE_RANK_MAX_NUM - 1) / SINGLE_RANK_MAX_NUM;

    constexpr static int64_t IDLER_CORE = 0;       // 闲置的核
    constexpr static int64_t PRODUCER_CORE_X = 1;  // 生产组，负责向共享内存写入数据，input->ipc
    constexpr static int64_t PRODUCER_CORE_Y = 2;  // 生产组，负责向共享内存写入数据，input->ipc
    constexpr static int64_t CONSUMER_CORE_X = 3;  // 消费组X
    constexpr static int64_t CONSUMER_CORE_Y = 4;  // 消费组Y

    constexpr static int32_t STEP_NUM = 2;
    constexpr static int32_t FIRST_STEP = 0;
    constexpr static int32_t SECOND_STEP = 1;
    constexpr static int32_t CORE_NUM_PER_RANK = 2;
    constexpr static int32_t INPUT_X_CORE_NUM = 1;
    constexpr static int32_t INPUT_Y_CORE_NUM = 1;
    constexpr static int32_t INPUT_CORE_NUM = INPUT_X_CORE_NUM + INPUT_Y_CORE_NUM;

    constexpr static int32_t MAX_RANK_SIZE = 8;
    constexpr static int32_t STEP1_X_FLAG_OFFSET = 64 * MAX_RANK_SIZE * CORE_NUM_PER_RANK;
    constexpr static int32_t STEP1_Y_FLAG_OFFSET = 65 * MAX_RANK_SIZE * CORE_NUM_PER_RANK;
    constexpr static int32_t STEP2_X_FLAG_OFFSET = 66 * MAX_RANK_SIZE * CORE_NUM_PER_RANK;
    constexpr static int32_t STEP2_Y_FLAG_OFFSET = 67 * MAX_RANK_SIZE * CORE_NUM_PER_RANK;

    constexpr static int COPY_OP = Op::COPYONLY;

public:
    FORCE_INLINE_AICORE All2AllVC2StepBackwardFusion(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag)
    {
    }

    FORCE_INLINE_AICORE void Init(EMB_BACKWARD_FUSION_ARGS_FUN())
    {
        Collectives::Init(EMB_BACKWARD_FUSION_ARGS_CALL());
        atomOp = op;

        sendCountMatrixGm.SetGlobalBuffer((__gm__ int64_t*)sendCountMatrix, rankSize * rankSize);
        // 初始化共享内存信息
        InitShare();
        // 初始化核分组
        InitCoreGroup();
        // 初始化数据切片
        InitDataSlice();

        // 初始化输入输出
        for (int j = 0; j < rankSize; j++) {
            sendLen += sendCountMatrixGm.GetValue(rank * rankSize + j);
        }
        allGatherLen = outShape * dim;

        if (xRankSize == 1) {
            xLen = 0;
        } else if (yRankSize == 1) {
            xLen = allGatherLen;
        } else {
            xLen = allGatherLen * (yRankSize + 1) / (xRankSize + 1 + yRankSize + 1);
        }
        yLen = allGatherLen - xLen;

        xReduceCoreNum = (xRankSize - 1);
        yReduceCoreNum = (yRankSize - 1);

        blockNum = (INPUT_CORE_NUM + xReduceCoreNum + yReduceCoreNum) * CORE_NUM_PER_RANK;
        blockGroupIdx = blockIdx / CORE_NUM_PER_RANK;
        blockItemIdx = blockIdx % CORE_NUM_PER_RANK;
        isLastBlockInGroup = (blockIdx % CORE_NUM_PER_RANK == CORE_NUM_PER_RANK - 1);

        if (blockIdx >= blockNum) {
            return;
        }
        this->tempTensor2 = tempTensor2;
        this->output = output;
        this->input = input;
        this->outShape = outShape;
        initAllGather();

        tempTensorOneGt.SetGlobalBuffer((__gm__ T*)tempTensor1);
        tempTensorTwoGt.SetGlobalBuffer((__gm__ T*)tempTensor2);
        ipcXGt.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET));
        ipcYGt.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET) + xLen * xRankSize);
        inputGt.SetGlobalBuffer((__gm__ T*)input);
        outputGt.SetGlobalBuffer((__gm__ T*)output);
        setp2OutputGt.SetGlobalBuffer((__gm__ T*)output);
    }

    FORCE_INLINE_AICORE void initAllGather()
    {
        if (blockGroupIdx < INPUT_X_CORE_NUM) {
            srcInputGt.SetGlobalBuffer((__gm__ T*)tempTensor2);
            input2OutputGt.SetGlobalBuffer((__gm__ T*)output);
            const int64_t totalDataNum = xLen;
            const int64_t perDataNum = totalDataNum / CORE_NUM_PER_RANK;
            const int64_t lastCoreDataNum = totalDataNum - perDataNum * (CORE_NUM_PER_RANK - 1);
            remainDataNum[FIRST_STEP] = isLastBlockInGroup ? lastCoreDataNum : perDataNum;
            blockDataOffsetNum[FIRST_STEP] = perDataNum * blockItemIdx;
        } else if (blockGroupIdx < INPUT_CORE_NUM) {
            srcInputGt.SetGlobalBuffer((__gm__ T*)tempTensor2 + xLen);
            input2OutputGt.SetGlobalBuffer((__gm__ T*)output + xLen);
            const int64_t totalDataNum = yLen;
            const int64_t perDataNum = totalDataNum / CORE_NUM_PER_RANK;
            const int64_t lastCoreDataNum = totalDataNum - perDataNum * (CORE_NUM_PER_RANK - 1);
            remainDataNum[FIRST_STEP] = isLastBlockInGroup ? lastCoreDataNum : perDataNum;
            blockDataOffsetNum[FIRST_STEP] = perDataNum * blockItemIdx;
        } else {
            InitStep1();
            InitStep2();
        }
    }

    FORCE_INLINE_AICORE void Process()
    {
        if (blockIdx >= blockNum) {
            return;
        }
        processGather();

        sync.SetInnerFlag(magic, 1, rank, blockIdx + MAX_FLAG_OFFSET * STEP_NUM);
        for (int i = 0; i < blockNum; i++) {
            sync.WaitInnerFlag(magic, 1, rank, i + MAX_FLAG_OFFSET * STEP_NUM);
        }

        if (coreGroup == PRODUCER_CORE_X || coreGroup == PRODUCER_CORE_Y) {
            ProducerStage();
        } else if (coreGroup == CONSUMER_CORE_X || coreGroup == CONSUMER_CORE_Y) {
            ConsumerStage();
        }

        sync.SetInnerFlag(magic, 1, rank, blockIdx + MAX_FLAG_OFFSET);
        for (int i = 0; i < blockNum; i++) {
            sync.WaitInnerFlag(magic, 1, rank, i + MAX_FLAG_OFFSET);
        }

        if (blockGroupIdx < INPUT_CORE_NUM) {
            Input2Ipc();
            Input2Output();
            Ipc2Output();
        } else {
            Step1();
            Step2();
        }
    }

private:
    GlobalTensor<T> inputGt;
    GlobalTensor<T> tempTensorOneGt;
    GlobalTensor<T> tempTensorTwoGt;
    GlobalTensor<T> outputGtInit;
    GlobalTensor<T> readGt;
    GlobalTensor<T> writeGt;
    GlobalTensor<int64_t> sendCountMatrixGm;

    int targetXRankIdx;
    int targetYRankIdx;

    int64_t maxSliceNum;
    int64_t revLen = 0;
    int64_t sendLen = 0;
    int64_t sendOffset[MULTI_RANK_SIZE];
    int64_t revOffset[MULTI_RANK_SIZE];
    int64_t inputDataLen[MULTI_RANK_SIZE];
    int64_t outputDataLen[MULTI_RANK_SIZE];
    int64_t ipcOffsetList[MULTI_RANK_SIZE];

    int waitRankListForWrite[MULTI_RANK_SIZE][1];  // 写共享内存时，需要等待的rank列表
    int waitNumForWrite[MULTI_RANK_SIZE];          // 写共享内存时，需要等待的数量
    int waitBlockForWrite[MULTI_RANK_SIZE];        // 写共享内存时，需要等待的标志位

    int64_t queLen;
    int64_t queSize;
    int64_t coreNumPerStage;  // 每个阶段使用的核数
    int64_t coreNumPerRank;   // 每个rank数据分配的核数
    int64_t rankNumThisCore;
    int64_t rankNumPerCore;  // 每个核负责的rank数
    int64_t coreGroup;       // 当前核的功能分组
    int64_t coreStep;
    int64_t groupCoreIdx[MULTI_RANK_SIZE];  // 当前核在组内的索引，可以为等效核索引
    int64_t targetRank[MULTI_RANK_SIZE];    // 当前核负责的rank

    IpcQueue<T> readQue[MULTI_RANK_SIZE];   // 读端共享内存队列
    IpcQueue<T> writeQue[MULTI_RANK_SIZE];  // 写端共享内存队列
    int64_t queElemLen;                     // 共享内存队列里每个元素大小（以T计）

    int64_t sliceNum[MULTI_RANK_SIZE];      // 当前核负责的数据切片总数
    int64_t copyLen;                        // 当前拷贝数据片的长度（以T计）
    int64_t inputOffset[MULTI_RANK_SIZE];   // 当前核负责的input偏移（以T计）
    int64_t inputLen[MULTI_RANK_SIZE];      // 当前核负责的input长度（以T计）
    int64_t outputOffset[MULTI_RANK_SIZE];  // 当前核负责的output偏移（以T计）
    int64_t outputLen[MULTI_RANK_SIZE];     // 当前核负责的output长度（以T计）
    GlobalTensor<T> srcInputGt;
    GlobalTensor<T> outputGt;
    GlobalTensor<T> input2OutputGt;
    GlobalTensor<T> setp2OutputGt;

    GlobalTensor<T> ipcXGt;
    GlobalTensor<T> ipcYGt;

    GlobalTensor<T> srcIpcGt[STEP_NUM];
    GlobalTensor<T> dstIpcGt[STEP_NUM];

    int atomOp;

    int32_t peerRankId[STEP_NUM] = {0};

    int32_t xReduceCoreNum = 0;
    int32_t yReduceCoreNum = 0;

    int64_t xLen;
    int64_t yLen;
    int64_t remainDataNum[STEP_NUM];
    int64_t blockDataOffsetNum[STEP_NUM];
    int64_t outShape;
    int64_t allGatherLen;

    int32_t blockGroupIdx = 0;
    int32_t blockItemIdx = 0;
    bool isLastBlockInGroup = false;

    GM_ADDR input;
    GM_ADDR output;
    GM_ADDR tempTensor2;

    FORCE_INLINE_AICORE void processGather()
    {
        if (blockIdx == 0) {
            readGt = inputGt;
            __gm__ T* output = const_cast<__gm__ T*>(tempTensorOneGt.GetPhyAddr());
            int32_t ub_head_offset = 96;
            __ubuf__ T* inputUB[2] = {(__ubuf__ T*)(ub_head_offset),
                                      (__ubuf__ T*)(ub_head_offset + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + UB_ALIGN_SIZE)};
            int64_t remain = sendLen * sizeof(T);
            int64_t offset = 0;
            int64_t outOffset = 0;
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2等MTE3
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // MTE2等MTE3
            int loop = 0;
            SetAtomic<T>(COPYONLY);
            while (remain > 0) {
                event_t eventId = (loop & 1) ? EVENT_ID0 : EVENT_ID1;
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
                // emb数量
                int64_t totalNum = 0;
                totalNum = remain < UB_SINGLE_DMA_SIZE_MAX / PING_PONG_SIZE
                               ? remain / dim / sizeof(T)
                               : UB_SINGLE_DMA_SIZE_MAX / PING_PONG_SIZE / dim / sizeof(T);

                __ubuf__ T* buffer = (loop & 1) ? inputUB[0] : inputUB[1];
                CpGM2UB(buffer, (__gm__ T*)readGt[offset].GetPhyAddr(), totalNum * dim * sizeof(T));
                offset += totalNum * dim;
                AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
                AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
                for (int j = 0; j < totalNum; j++) {
                    int64_t outIdx = *((__gm__ int64_t*)gatherIdxPtr + outOffset + j);
                    CpUB2GM((output + outIdx * dim), buffer + j * dim, dim * sizeof(T));
                }
                AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
                remain -= UB_SINGLE_DMA_SIZE_MAX / PING_PONG_SIZE;
                outOffset += totalNum;
                loop += 1;
            }
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2等MTE3
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // MTE2等MTE3
            AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID3);      // Scalar等MTE3
            AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
            UnsetAtomic(COPYONLY);
        }
    }
    // 计算rank数量较大时的queNum 以及  每个队列里单块可放入的元素数量queElemLen
    __aicore__ inline void InitShare()
    {
        int64_t queNum = xRankSize + yRankSize;  // 两个方向各一个队列
        int64_t queElemNum = xRankSize * yRankSize * 2;  // x方向 x个队列， 每个队列的每个element可以放y个数据
        queElemLen = IPC_BUFF_MAX_SIZE / sizeof(T) / queElemNum / SHARE_QUE_DEPTH;  // 计算共享队列元素大小
    }

    __aicore__ inline void InitCoreGroup()
    {
        coreNumPerRank = 1;
        rankNumPerCore = 1;
        // 多卡下为CORE_NUMS_PER_STAGE
        coreNumPerStage = xRankSize + yRankSize;
        rankNumThisCore = 1;
        // 负责x方向 input2ipc的core
        if (blockIdx < 2 * coreNumPerStage) {
            if (blockIdx < coreNumPerStage) {
                coreGroup = blockIdx % coreNumPerStage < xRankSize ? PRODUCER_CORE_X : PRODUCER_CORE_Y;
                targetRank[0] = (coreGroup == PRODUCER_CORE_X) ? yRankIdx * xRankSize + blockIdx
                                                               : (blockIdx - xRankSize) * xRankSize + xRankIdx;
            } else {
                coreGroup = blockIdx % coreNumPerStage < xRankSize ? CONSUMER_CORE_X : CONSUMER_CORE_Y;
                targetRank[0] = (coreGroup == CONSUMER_CORE_X)
                                    ? yRankIdx * xRankSize + blockIdx % coreNumPerStage
                                    : (blockIdx % coreNumPerStage - xRankSize) * xRankSize + xRankIdx;
            }
            targetXRankIdx = targetRank[0] % xRankSize;
            targetYRankIdx = targetRank[0] / xRankSize;
        } else {
            coreGroup = IDLER_CORE;
        }
    }

    __aicore__ inline void InitDataSlice()
    {
        queLen = IPC_BUFF_MAX_SIZE / sizeof(T) / (xRankSize * yRankSize * STAGE_NUM);  // 一个que可放入的元素数量
        queSize = queLen * sizeof(T);

        // 生产者负责搬运本rank的输入数据至共享内存，input-->share
        if ((coreGroup == PRODUCER_CORE_X) || (coreGroup == PRODUCER_CORE_Y)) {
            ProducerDataSlice();
        } else if ((coreGroup == CONSUMER_CORE_X) || (coreGroup == CONSUMER_CORE_Y)) {
            ConsumerDataSlice();
        }
    }

    __aicore__ inline void ProducerDataSlice()
    {
        maxSliceNum = 0;
        if (coreGroup == PRODUCER_CORE_X) {
            for (int i = 0; i < yRankSize; i++) {
                writeQue[i].Init(&sync, magic,
                                 shareAddrs[targetRank[0]] + IPC_DATA_OFFSET + (xRankIdx * yRankSize + i) * queSize,
                                 queLen, queElemLen);
                sendOffset[i] = 0;
                for (int j = 0; j < (targetXRankIdx + i * xRankSize); j++) {
                    sendOffset[i] += sendCountMatrixGm.GetValue(rank * rankSize + j);
                }
                inputDataLen[i] =
                    sendCountMatrixGm.GetValue(rank * rankSize + (targetXRankIdx + i * xRankSize)) / DATA_SLICE;
                SplitData(inputDataLen[i], 0, inputOffset[i], inputLen[i], sendOffset[i]);
                // 当前核负责的数据切片数，能分成一个que中的多少小块
                sliceNum[i] = CeilDiv(inputLen[i], queElemLen);
                if (sliceNum[i] > maxSliceNum) {
                    maxSliceNum = sliceNum[i];
                }
            }
        } else {
            for (int i = 0; i < xRankSize; i++) {
                writeQue[i].Init(&sync, magic,
                                 shareAddrs[targetRank[0]] + IPC_DATA_OFFSET +
                                     (xRankSize * yRankSize + yRankIdx * xRankSize + i) * queSize,
                                 queLen, queElemLen);
                sendOffset[i] = 0;
                for (int j = 0; j < (targetYRankIdx * xRankSize + i); j++) {
                    sendOffset[i] += sendCountMatrixGm.GetValue(rank * rankSize + j);
                }
                int inputDataLenTmp =
                    sendCountMatrixGm.GetValue(rank * rankSize + (targetYRankIdx * xRankSize + i)) / DATA_SLICE;
                sendOffset[i] += inputDataLenTmp;
                inputDataLen[i] =
                    sendCountMatrixGm.GetValue(rank * rankSize + (targetYRankIdx * xRankSize + i)) - inputDataLenTmp;
                SplitData(inputDataLen[i], 0, inputOffset[i], inputLen[i], sendOffset[i]);
                // 当前核负责的数据切片数，能分成一个que中的多少小块
                sliceNum[i] = CeilDiv(inputLen[i], queElemLen);
                if (sliceNum[i] > maxSliceNum) {
                    maxSliceNum = sliceNum[i];
                }
            }
        }
    }

    __aicore__ inline void ConsumerDataSlice()
    {
        maxSliceNum = 0;
        if (coreGroup == CONSUMER_CORE_X) {
            for (int k = 0; k < yRankSize; k++) {
                readQue[k].Init(&sync, magic,
                                shareAddrs[targetRank[0]] + IPC_DATA_OFFSET +
                                    (xRankSize * yRankSize + k * xRankSize + xRankIdx) * queSize,
                                queLen, queElemLen);
                revOffset[k] = 0;
                for (int j = 0; j < (k * xRankSize + targetXRankIdx); j++) {
                    revOffset[k] += sendCountMatrixGm.GetValue(j * rankSize + rank);
                }
                int inputDataLenTmp =
                    sendCountMatrixGm.GetValue((k * xRankSize + targetXRankIdx) * rankSize + rank) / DATA_SLICE;
                revOffset[k] += inputDataLenTmp;
                outputDataLen[k] =
                    sendCountMatrixGm.GetValue((k * xRankSize + targetXRankIdx) * rankSize + rank) - inputDataLenTmp;
                SplitData(outputDataLen[k], 0, outputOffset[k], outputLen[k], revOffset[k]);
                // 当前核负责的数据切片数，能分成一个que中的多少小块
                sliceNum[k] = CeilDiv(outputLen[k], queElemLen);
                if (sliceNum[k] > maxSliceNum) {
                    maxSliceNum = sliceNum[k];
                }
            }
        } else {
            for (int k = 0; k < xRankSize; k++) {
                readQue[k].Init(&sync, magic,
                                shareAddrs[targetRank[0]] + IPC_DATA_OFFSET + (k * yRankSize + yRankIdx) * queSize,
                                queLen, queElemLen);
                revOffset[k] = 0;
                for (int j = 0; j < (xRankSize * targetYRankIdx + k); j++) {
                    revOffset[k] += sendCountMatrixGm.GetValue(j * rankSize + rank);
                }
                outputDataLen[k] =
                    sendCountMatrixGm.GetValue((xRankSize * targetYRankIdx + k) * rankSize + rank) / DATA_SLICE;
                SplitData(outputDataLen[k], 0, outputOffset[k], outputLen[k], revOffset[k]);
                // 当前核负责的数据切片数，能分成一个que中的多少小块
                sliceNum[k] = CeilDiv(outputLen[k], queElemLen);
                if (sliceNum[k] > maxSliceNum) {
                    maxSliceNum = sliceNum[k];
                }
            }
        }
    }

    __aicore__ inline void SplitData(const int64_t totalLen, const int64_t useCoreIdx, int64_t& dataOffset,
                                     int64_t& dataLen, int startOffset)
    {
        // 向上整除获取每个core切分的数据个数
        dataLen = totalLen;
        // 数据量极小或略微超过核数的情况，后面若干个core数据量为0
        dataOffset = useCoreIdx * dataLen + startOffset;  // 使用当前block在useBlock里的相对索引来计算偏移
        // 非整除情况，最后一个core数据量为剩余数据量
        if (dataOffset + dataLen - startOffset > totalLen) {
            dataLen = totalLen;
        }
    }

    __aicore__ inline void ProducerStage()
    {
        for (auto i = 0; i < rankNumThisCore; ++i) {
            if (targetRank[i] == INVALID_RANK_NUM) {
                continue;
            }
            // 写共享内存队列时，需要等待当前rank
            waitRankListForWrite[i][0] = targetRank[i];
            waitNumForWrite[i] = 1;
        }
        InputToSharePipeline();
    }

    __aicore__ inline void InputToSharePipeline()
    {
        int64_t flagValue[MULTI_RANK_SIZE];  // 要等待标志位的存储值
        for (auto i = 0; i < rankNumThisCore; ++i) {
            flagValue[i] = -1;  // 统一赋值为-1，方便后续判断
        }
        // 以最多切片sliceNum[0]为切片进行循环，切片数不足不拷贝
        for (auto sliceIdx = 0; sliceIdx < maxSliceNum; ++sliceIdx) {
            for (auto i = 0; i < rankNumThisCore; ++i) {
                if (targetRank[i] == INVALID_RANK_NUM) {
                    continue;
                }
                InputToShareSlice(i, sliceIdx, flagValue[i]);
            }
        }
    }

    __aicore__ inline void InputToShareSlice(int64_t idx, int64_t sliceIdx, int64_t& flagValue)
    {
        // 计算当前切片拷贝数据量，数据量不为0时不拷贝
        int tmpRankSize = (coreGroup == PRODUCER_CORE_X) ? yRankSize : xRankSize;
        for (int i = 0; i < tmpRankSize; i++) {
            copyLen = inputLen[i] - queElemLen * sliceIdx;
            if (copyLen > queElemLen) {
                copyLen = queElemLen;
            } else if (copyLen <= 0) {
                copyLen = 0;
                continue;
            }
            readGt = tempTensorOneGt[sliceIdx * queElemLen + inputOffset[i]];
            // 这里一共定义了xranksize*yranksize *2 个标志位，两个方向各使用一半，前一半以[x,y]为shape考虑
            // [j,k]表示x方向第j个队列发给y方向第k个rank的；y方向反之
            int tmpRankIdx = (coreGroup == PRODUCER_CORE_X) ? xRankIdx * yRankSize + i
                                                            : xRankSize * yRankSize + yRankIdx * xRankSize + i;
            writeQue[i].DeQue(waitRankListForWrite[idx], 1, tmpRankIdx + xRankSize * yRankSize * STAGE_NUM);
            writeGt = writeQue[i].EnQue();
            if (copyLen > 0) {
                CpGM2GMPingPong<T>(copyLen * sizeof(T), readGt, writeGt, COPYONLY);
            }
            sync.SetInnerFlag(magic, sliceIdx, targetRank[idx], tmpRankIdx);
        }
    }

    __aicore__ inline void ConsumerStage()
    {
        int64_t flagValue[MULTI_RANK_SIZE];  // 需等待标志位的存储值
        int tmpRankSize = (coreGroup == CONSUMER_CORE_X) ? yRankSize : xRankSize;
        for (auto i = 0; i < tmpRankSize; ++i) {
            flagValue[i] = -1;
        }
        // 以最多切片sliceNum[0]为切片数进行循环，切片数不足的不拷贝
        for (auto sliceIdx = 0; sliceIdx < maxSliceNum; ++sliceIdx) {
            for (auto i = 0; i < rankNumPerCore; ++i) {
                if (targetRank[i] == INVALID_RANK_NUM) {
                    continue;
                }
                ShareToOutputSlice(i, sliceIdx, flagValue);
            }
        }
    }
    __aicore__ inline void calTotalNum(int64_t remain, int64_t& totalNum)
    {
        if (copyLen < dim) {
            totalNum = 1;
        } else {
            totalNum = remain < UB_SINGLE_DMA_SIZE_MAX / PING_PONG_SIZE
                           ? remain / dim / sizeof(T)
                           : UB_SINGLE_DMA_SIZE_MAX / PING_PONG_SIZE / dim / sizeof(T);
        }
    }

    __aicore__ inline void ShareToOutputSlice(int64_t idx, int64_t sliceIdx, int64_t* flagValue)
    {
        int tmpRankSize = (coreGroup == CONSUMER_CORE_X) ? yRankSize : xRankSize;
        for (int i = 0; i < tmpRankSize; i++) {
            copyLen = outputLen[i] - queElemLen * sliceIdx;
            if (copyLen > queElemLen) {
                copyLen = queElemLen;
            } else if (copyLen <= 0) {
                copyLen = 0;
                continue;
            }
            // consume的x方向，查看target上的flag后半部分[ysize,xsize],来判断上一步y方向上的第tmpRankSize个rank是否发给target了
            int tmpRankIdx = (coreGroup == CONSUMER_CORE_X) ? xRankSize * yRankSize + i * xRankSize + xRankIdx
                                                            : i * yRankSize + yRankIdx;
            if (flagValue[i] < sliceIdx) {
                sync.WaitInnerFlag(magic, sliceIdx, targetRank[idx], tmpRankIdx);
            }
            readGt = readQue[i].ReadFront();
            __gm__ T* output1 = const_cast<__gm__ T*>(tempTensorTwoGt.GetPhyAddr());
            if (copyLen > 0) {
                flagValue[i] = sync.GetInnerFlag(targetRank[idx], tmpRankIdx) & EVENT_ID_MASK;
                uss(output1, copyLen, i, sliceIdx);
            }
            sync.SetInnerFlag(magic, sliceIdx, targetRank[idx], tmpRankIdx + xRankSize * yRankSize * STAGE_NUM);
        }
        if (sliceIdx == maxSliceNum - 1) {
            for (int i = 0; i < tmpRankSize; i++) {
                int tmpRankIdx = (coreGroup == CONSUMER_CORE_X) ? xRankSize * yRankSize + i * xRankSize + xRankIdx
                                                                : i * yRankSize + yRankIdx;
                sync.SetInnerFlag(0, 0, targetRank[idx], tmpRankIdx + xRankSize * yRankSize * STAGE_NUM);
                sync.SetInnerFlag(0, 0, targetRank[idx], tmpRankIdx);
            }
        }
    }

    FORCE_INLINE_AICORE void uss(__gm__ T* output1, int64_t copyLen, int i, int64_t sliceIdx)
    {
        int32_t ub_head_offset = 96;
        __ubuf__ T* inputUB[2] = {(__ubuf__ T*)(ub_head_offset),
                                  (__ubuf__ T*)(ub_head_offset + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + UB_ALIGN_SIZE)};
        int64_t remain = copyLen * sizeof(T);
        int64_t offset = 0;
        int64_t outOffset = 0;
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2等MTE3
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // MTE2等MTE3
        int loop = 0;
        SetAtomic<T>(ADD);
        while (remain > 0) {
            event_t eventId = (loop & 1) ? EVENT_ID0 : EVENT_ID1;
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            // emb数量
            int64_t totalNum = 0;
            calTotalNum(remain, totalNum);

            __ubuf__ T* buffer = (loop & 1) ? inputUB[0] : inputUB[1];
            CpGM2UB(buffer, (__gm__ T*)readGt[offset].GetPhyAddr(), totalNum * dim * sizeof(T));
            offset += totalNum * dim;
            AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            for (int j = 0; j < totalNum; j++) {
                int64_t outIdx = *((__gm__ int64_t*)restorePtr + sliceIdx * queElemLen / dim + outputOffset[i] / dim +
                                   outOffset + j);
                if (copyLen < dim && coreGroup == CONSUMER_CORE_Y) {
                    CpUB2GM((output1 + outIdx * dim), buffer + j * dim, dim * sizeof(T) / DATA_SLICE);
                } else if (copyLen < dim && coreGroup == CONSUMER_CORE_X) {
                    CpUB2GM((output1 + outIdx * dim + dim / DATA_SLICE), buffer + j * dim,
                            dim * sizeof(T) / DATA_SLICE);
                } else {
                    CpUB2GM((output1 + outIdx * dim), buffer + j * dim, dim * sizeof(T));
                }
            }
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
            remain -= UB_SINGLE_DMA_SIZE_MAX / PING_PONG_SIZE;
            outOffset += totalNum;
            loop += 1;
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2等MTE3
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // MTE2等MTE3
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID3);      // Scalar等MTE3
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
        UnsetAtomic(ADD);
    }

    FORCE_INLINE_AICORE void InitStep1()
    {
        if (blockGroupIdx < INPUT_CORE_NUM + xReduceCoreNum) {
            const int32_t localGroupId = (blockGroupIdx - INPUT_CORE_NUM);

            peerRankId[FIRST_STEP] = yRankIdx * xRankSize + (xRankIdx + 1 + localGroupId) % xRankSize;
            const int64_t totalDataNum = xLen;
            const int64_t perDataNum = totalDataNum / CORE_NUM_PER_RANK;
            const int64_t lastCoreDataNum = totalDataNum - perDataNum * (CORE_NUM_PER_RANK - 1);
            remainDataNum[FIRST_STEP] = isLastBlockInGroup ? lastCoreDataNum : perDataNum;
            const int64_t dataOffsetNum = xLen * (peerRankId[FIRST_STEP] % xRankSize) + perDataNum * blockItemIdx;
            __gm__ T* srcIpcGm = (__gm__ T*)(shareAddrs[peerRankId[FIRST_STEP]] + IPC_DATA_OFFSET);
            __gm__ T* dstIpcGm = (__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET);
            srcIpcGt[FIRST_STEP].SetGlobalBuffer(srcIpcGm + dataOffsetNum);
            dstIpcGt[FIRST_STEP].SetGlobalBuffer(dstIpcGm + dataOffsetNum);
        } else {
            const int32_t localGroupId = (blockGroupIdx - (INPUT_CORE_NUM + xReduceCoreNum));

            peerRankId[FIRST_STEP] = (yRankIdx + 1 + localGroupId) % yRankSize * xRankSize + xRankIdx;
            const int64_t totalDataNum = yLen;
            const int64_t perDataNum = totalDataNum / CORE_NUM_PER_RANK;
            const int64_t lastCoreDataNum = totalDataNum - perDataNum * (CORE_NUM_PER_RANK - 1);
            remainDataNum[FIRST_STEP] = isLastBlockInGroup ? lastCoreDataNum : perDataNum;
            const int64_t dataOffsetNum =
                xLen * xRankSize + yLen * (peerRankId[FIRST_STEP] / xRankSize) + perDataNum * blockItemIdx;

            __gm__ T* srcIpcGm = (__gm__ T*)(shareAddrs[peerRankId[FIRST_STEP]] + IPC_DATA_OFFSET);
            __gm__ T* dstIpcGm = (__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET);
            srcIpcGt[FIRST_STEP].SetGlobalBuffer(srcIpcGm + dataOffsetNum);
            dstIpcGt[FIRST_STEP].SetGlobalBuffer(dstIpcGm + dataOffsetNum);
        }
    }

    FORCE_INLINE_AICORE void InitStep2()
    {
        if (blockGroupIdx < INPUT_CORE_NUM + yReduceCoreNum) {
            const int32_t localGroupId = (blockGroupIdx - INPUT_CORE_NUM);

            peerRankId[SECOND_STEP] = (yRankIdx + 1 + localGroupId) % yRankSize * xRankSize + xRankIdx;
            const int64_t totalDataNum = xLen;
            const int64_t perDataNum = totalDataNum / CORE_NUM_PER_RANK;
            const int64_t lastCoreDataNum = totalDataNum - perDataNum * (CORE_NUM_PER_RANK - 1);
            remainDataNum[SECOND_STEP] = isLastBlockInGroup ? lastCoreDataNum : perDataNum;
            const int64_t dataOffsetNum = perDataNum * blockItemIdx;
            __gm__ T* srcIpcGm = (__gm__ T*)(shareAddrs[peerRankId[SECOND_STEP]] + IPC_DATA_OFFSET);
            srcIpcGt[SECOND_STEP].SetGlobalBuffer(srcIpcGm + dataOffsetNum);
        } else {
            const int32_t localGroupId = (blockGroupIdx - (INPUT_CORE_NUM + yReduceCoreNum));

            peerRankId[SECOND_STEP] = yRankIdx * xRankSize + (xRankIdx + 1 + localGroupId) % xRankSize;
            const int64_t totalDataNum = yLen;
            const int64_t perDataNum = totalDataNum / CORE_NUM_PER_RANK;
            const int64_t lastCoreDataNum = totalDataNum - perDataNum * (CORE_NUM_PER_RANK - 1);
            remainDataNum[SECOND_STEP] = isLastBlockInGroup ? lastCoreDataNum : perDataNum;
            const int64_t dataOffsetNum = xLen * xRankSize + perDataNum * blockItemIdx;
            __gm__ T* srcIpcGm = (__gm__ T*)(shareAddrs[peerRankId[SECOND_STEP]] + IPC_DATA_OFFSET);
            srcIpcGt[SECOND_STEP].SetGlobalBuffer(srcIpcGm + dataOffsetNum);
        }
    }

    FORCE_INLINE_AICORE void Input2Ipc()
    {
        if (blockGroupIdx < INPUT_X_CORE_NUM) {
            const int64_t targetOffset = xRankIdx * xLen;
            const int64_t curBlockOffset = blockDataOffsetNum[FIRST_STEP];
            CpGM2GMPingPong(remainDataNum[FIRST_STEP] * sizeof(T), srcInputGt[curBlockOffset],
                            ipcXGt[targetOffset + curBlockOffset], COPY_OP);
            sync.SetSyncFlag(magic, 0, STEP1_X_FLAG_OFFSET + xRankIdx * CORE_NUM_PER_RANK + blockItemIdx, rank);
        } else {
            const int64_t targetOffset = yRankIdx * yLen;
            const int64_t curBlockOffset = blockDataOffsetNum[FIRST_STEP];
            CpGM2GMPingPong(remainDataNum[FIRST_STEP] * sizeof(T), srcInputGt[curBlockOffset],
                            ipcYGt[targetOffset + curBlockOffset], COPY_OP);
            sync.SetSyncFlag(magic, 0, STEP1_Y_FLAG_OFFSET + yRankIdx * CORE_NUM_PER_RANK + blockItemIdx, rank);
        }
    }

    FORCE_INLINE_AICORE void Input2Output()
    {
        const int64_t curBlockOffset = blockDataOffsetNum[FIRST_STEP];
        const int64_t cpDataSize = remainDataNum[FIRST_STEP] * sizeof(T);
        CpGM2GMPingPong(cpDataSize, srcInputGt[curBlockOffset], input2OutputGt[rank * allGatherLen + curBlockOffset],
                        COPY_OP);
    }

    FORCE_INLINE_AICORE void Ipc2Output()
    {
        if (blockGroupIdx < INPUT_X_CORE_NUM) {
            const int64_t curBlockOffset = blockDataOffsetNum[FIRST_STEP];
            for (int32_t i = 1; i < xRankSize; ++i) {
                const int32_t targetXId = (xRankIdx + i) % xRankSize;
                const int32_t targetRankId = targetXId + yRankIdx * xRankSize;

                const int32_t flagIdx = targetXId * CORE_NUM_PER_RANK + blockItemIdx;
                sync.WaitSyncFlag(magic, 0, STEP1_X_FLAG_OFFSET + flagIdx, rank);

                const int64_t cpDataSize = remainDataNum[FIRST_STEP] * sizeof(T);
                CpGM2GMPingPong(cpDataSize, ipcXGt[targetXId * xLen + curBlockOffset],
                                outputGt[targetRankId * allGatherLen + curBlockOffset], COPY_OP);
            }
        } else {
            const int64_t curBlockOffset = blockDataOffsetNum[FIRST_STEP];
            for (int32_t i = 1; i < yRankSize; ++i) {
                const int32_t targetYId = (yRankIdx + i) % yRankSize;
                const int32_t targetRankId = xRankIdx + targetYId * xRankSize;

                const int32_t flagIdx = targetYId * CORE_NUM_PER_RANK + blockItemIdx;
                sync.WaitSyncFlag(magic, 0, STEP1_Y_FLAG_OFFSET + flagIdx, rank);

                const int64_t cpDataSize = remainDataNum[FIRST_STEP] * sizeof(T);
                CpGM2GMPingPong(cpDataSize, ipcYGt[targetYId * yLen + curBlockOffset],
                                outputGt[targetRankId * allGatherLen + xLen + curBlockOffset], COPY_OP);
            }
        }
    }

    FORCE_INLINE_AICORE void Step1()
    {
        if (blockGroupIdx < INPUT_CORE_NUM + xReduceCoreNum) {
            const int32_t flagIdx = (peerRankId[FIRST_STEP] % xRankSize) * CORE_NUM_PER_RANK + blockItemIdx;
            sync.WaitSyncFlag(magic, 0, STEP1_X_FLAG_OFFSET + flagIdx, peerRankId[FIRST_STEP]);
            CpGM2GMPingPong(remainDataNum[FIRST_STEP] * sizeof(T), srcIpcGt[FIRST_STEP], dstIpcGt[FIRST_STEP], COPY_OP);
            sync.SetSyncFlag(magic, 0, STEP1_X_FLAG_OFFSET + flagIdx, rank);
        } else {
            const int32_t flagIdx = (peerRankId[FIRST_STEP] / xRankSize) * CORE_NUM_PER_RANK + blockItemIdx;
            sync.WaitSyncFlag(magic, 0, STEP1_Y_FLAG_OFFSET + flagIdx, peerRankId[FIRST_STEP]);
            CpGM2GMPingPong(remainDataNum[FIRST_STEP] * sizeof(T), srcIpcGt[FIRST_STEP], dstIpcGt[FIRST_STEP], COPY_OP);
            sync.SetSyncFlag(magic, 0, STEP1_Y_FLAG_OFFSET + flagIdx, rank);
        }
    }

    FORCE_INLINE_AICORE void Step2()
    {
        if (blockGroupIdx < INPUT_CORE_NUM + yReduceCoreNum) {
            const int32_t peerXId = peerRankId[SECOND_STEP] % xRankSize;
            const int32_t peerYId = peerRankId[SECOND_STEP] / xRankSize;
            for (int32_t i = 0; i < xRankSize; ++i) {
                const int32_t targetXId = (peerXId + i) % xRankSize;
                const int32_t targetRankId = targetXId + peerYId * xRankSize;

                const int32_t flagIdx = targetXId * CORE_NUM_PER_RANK + blockItemIdx;
                sync.WaitSyncFlag(magic, 0, STEP1_X_FLAG_OFFSET + flagIdx, peerRankId[SECOND_STEP]);

                const int64_t cpDataSize = remainDataNum[SECOND_STEP] * sizeof(T);
                CpGM2GMPingPong(cpDataSize, srcIpcGt[SECOND_STEP][targetXId * xLen],
                                outputGt[targetRankId * allGatherLen + xLen / CORE_NUM_PER_RANK * blockItemIdx],
                                COPY_OP);
            }
        } else {
            const int32_t peerXId = peerRankId[SECOND_STEP] % xRankSize;
            const int32_t peerYId = peerRankId[SECOND_STEP] / xRankSize;
            for (int32_t i = 0; i < yRankSize; ++i) {
                const int32_t targetYId = (peerYId + i) % yRankSize;
                const int32_t targetRankId = peerXId + targetYId * xRankSize;

                const int32_t flagIdx = targetYId * CORE_NUM_PER_RANK + blockItemIdx;
                sync.WaitSyncFlag(magic, 0, STEP1_Y_FLAG_OFFSET + flagIdx, peerRankId[SECOND_STEP]);

                const int64_t cpDataSize = remainDataNum[SECOND_STEP] * sizeof(T);
                CpGM2GMPingPong(cpDataSize, srcIpcGt[SECOND_STEP][targetYId * yLen],
                                outputGt[targetRankId * allGatherLen + xLen + yLen / CORE_NUM_PER_RANK * blockItemIdx],
                                COPY_OP);
            }
        }
    }
};

#endif  // LCCL_ALL2ALLVC_2STEP_H