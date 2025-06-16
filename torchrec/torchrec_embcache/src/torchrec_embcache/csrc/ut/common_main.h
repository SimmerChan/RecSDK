/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
* Copyright (c) huawei Platforms, Inc. and affiliates.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>
#include <glog/logging.h>

int common_main(int argc, char *argv[])
{
    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);

    ::testing::InitGoogleTest(&argc, argv);

    int result = 0;
    result = RUN_ALL_TESTS();
    return result;
}
