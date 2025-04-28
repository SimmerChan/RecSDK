/*
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include "acl/acl.h"
#include "aclnnop/aclnn_mm.h"

#include <string>

void CALL_RT(int x)
{
    if (auto ret = (x != 0)) {
        std::error << "[ERROR] Failed to exec acl api " << #x << ", result: " << ret << std::endl; \
        throw 1;
    }
}

void CreateAclTensorShape(std::vector<int64_t> &shape,
                          std::vector<int64_t> &strides,
                          int nRow, int nCol,
                          int transpose)
{
    shape.clear();
    shape.push_back(nRow);
    shape.push_back(nCol);
    
    strides.clear();
    if (transpose) {
        strides.push_back(1);
        strides.push_back(nRow);
    } else {
        strides.push_back(nCol);
        strides.push_back(1);
    }
}

void CreateAclTensor2d(const std::vector<int64_t> &shape,
                       const std::vector<int64_t> &strides,
                       __fp16 *deviceAddr, aclTensor **tensor, int format)
{
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        ACL_FLOAT16,
        strides.data(),
        0,
        format ==0 ? aclFormat::ACL_FORMAT_ND: aclFormat::ACL_FORMAT_FRACTAL_NZ,
        shape.data(),
        shape.size(),
        deviceAddr);
}

int main(int argc, char **argv)
{
    int64_t verifyLevel = 0;
    int64_t deviceId = 3;
    int deviceIdIndex = 8;
    deviceId = std::atoi(argv[device_id_index]);
    CALL_RT(aclInit(nullptr));
    CALL_RT(aclrtSetDevice(deviceId));

    aclrtStream stream;
    CALL_RT(aclrtCreateStream(&stream));

    int transA;
    int transB;
    int formatA;
    int formatB;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;

    // 打印当前行的数据
    int index = 1;
    m = std::atoi(argv[index++]);
    k = std::atoi(argv[index++]);
    n = std::atoi(argv[index++]);
    transA = std::atoi(argv[index++]);
    transB = std::atoi(argv[index++]);
    formatA = std::atoi(argv[index++]);
    formatB = std::atoi(argv[index++]);

    printf("M: %d, K: %d, N: %d, transA: %d, transB: %d, formatA: %d, formatB: %d\n",
        m,
        k,
        n,
        transA,
        transB,
        formatA,
        formatB);

    RunMatmul(m, n, k, transA, transB, formatA, formatB);
    
    CALL_RT(aclrtResetDevice(deviceId));
    CALL_RT(aclFinalize());

    return 0;
}

void RunMatmul(int64_t m,
               int64_t n,
               int64_t k,
               int transA,
               int transB,
               int formatA,
               int formatB,)
{
    int64_t A_size = sizeof(__fp16) * m * k;
    int cubeSize = 16;
    if (formatA == 1) {
        A_size = sizeof(__fp16) *
                ((m + cubeSize - 1) / cubeSize * cubeSize) *
                ((k + cubeSize - 1) / cubeSize * cubeSize);
    }
    int64_t B_size = sizeof(__fp16) * n * k;
    if (formatB == 1) {
        B_size = sizeof(__fp16) *
                ((n + cubeSize - 1) / cubeSize * cubeSize) *
                ((K + cubeSize - 1) / cubeSize * cubeSize);
    }
    int64_t C_size = sizeof(__fp16) * m * n;

    __fp16 *dA = nullptr;
    __fp16 *dB = nullptr;
    __fp16 *dC = nullptr;

    aclTensor *tensorA = nullptr;
    aclTensor *tensorB = nullptr;
    aclTensor *tensorC = nullptr;

    CALL_RT(aclrtMalloc((void **)(&dA), A_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&dB), B_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&dC), C_size, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

    CreateAclTensorShape(shape, strides, m, k, transA);
    CreateAclTensor2d(shape, strides, dA, &tensorA, formatA);

    CreateAclTensorShape(shape, strides, k, n, transB);
    CreateAclTensor2d(shape, strides, dB, &tensorB, formatB);

    CreateAclTensorShape(shape, strides, m, n, 0);
    CreateAclTensor2d(shape, strides, dC, &tensorC, 0);

    int8_t cubeMathType = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnMm第一段接口
    CALL_RT(aclnnMmGetWorkspaceSize(tensorA, tensorB, tensorC, cubeMathType, &workspaceSize, &executor));

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        CALL_RT(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    // 调用aclnnMm第二段接口
    CALL_RT(aclnnMm(workspaceAddr, workspaceSize, executor, stream));

    // 算子结束后需要同步，才能测数据
    CALL_RT(aclrtSynchronizeStream(stream));

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclDestroyTensor(tensorA);
    aclDestroyTensor(tensorB);
    aclDestroyTensor(tensorC);

    CALL_RT(aclrtFree(dA));
    CALL_RT(aclrtFree(dB));
    CALL_RT(aclrtFree(dC));
}
