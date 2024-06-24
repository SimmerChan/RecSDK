/**
* @file main.cpp
*
* Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "acl/acl.h"
#include "op_runner.h"

#include "common.h"

bool g_isDevice = false;
int deviceId = 0;

OperatorDesc CreateOpDesc()
{
    // define operator
    std::vector<int64_t> dout { 1024, 1000, 80 };
    std::vector<int64_t> softmax_out { 1024, 1000, 50 };
    std::vector<int64_t> query { 1024, 1000, 80};
    std::vector<int64_t> key  { 1024, 50, 80  };
    std::vector<int64_t> value { 1024, 50, 80  };

    std::vector<int64_t> grad_query { 1024, 1000, 80};
    std::vector<int64_t> grad_key { 1024, 50, 80 };
    std::vector<int64_t> grad_value { 1024, 50, 80 };




    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(ACL_FLOAT, dout.size(), dout.data(), format);
    opDesc.AddInputTensorDesc(ACL_FLOAT, softmax_out.size(), softmax_out.data(), format);
    opDesc.AddInputTensorDesc(ACL_FLOAT, query.size(), query.data(), format);
    opDesc.AddInputTensorDesc(ACL_FLOAT, key.size(), key.data(), format);
    opDesc.AddInputTensorDesc(ACL_FLOAT, value.size(), value.data(), format);

    opDesc.AddOutputTensorDesc(ACL_FLOAT, grad_query.size(), grad_query.data(), format);
    opDesc.AddOutputTensorDesc(ACL_FLOAT, grad_key.size(), grad_key.data(), format);
    opDesc.AddOutputTensorDesc(ACL_FLOAT, grad_value.size(), grad_value.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    ReadFile("../input/dout.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    ReadFile("../input/softmax_out.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    ReadFile("../input/query.bin", fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2));
    ReadFile("../input/key.bin", fileSize, runner.GetInputBuffer<void>(3), runner.GetInputSize(3));
    ReadFile("../input/value.bin", fileSize, runner.GetInputBuffer<void>(4), runner.GetInputSize(4));
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    WriteFile("../output/grad_query.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));
    WriteFile("../output/grad_key.bin", runner.GetOutputBuffer<void>(1), runner.GetOutputSize(1));
    WriteFile("../output/grad_value.bin", runner.GetOutputBuffer<void>(2), runner.GetOutputSize(2));
    INFO_LOG("Write output success");
    return true;
}

void DestoryResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destory resource failed");
    } else {
        INFO_LOG("Destory resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        }
        else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    // acl.json is dump or profiling config file
    if (aclInit(NULL) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestoryResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp()
{
    // create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    // process output data
    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestoryResource();
        return FAILED;
    }

    DestoryResource();

    return SUCCESS;
}
