/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <random>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <limits>
#include <mpi.h>

#include "utils/common.h"
#include "utils/logger.h"
#include "utils/error.h"
#include "emb_table/embedding_dynamic.h"

using namespace std;
using namespace MxRec;
using namespace testing;
using namespace tensorflow;

class EmbeddingDynamicTest : public testing::Test {
protected:
    EmbeddingDynamicTest()
    {
        int embSize = 1000;
        int extEmbSize = 2000;
        struct EmbInfoParams embParam(string("test1"), 0, embSize, extEmbSize, true, true, false, false);
        std::vector<size_t> vocabsize = {10000, 0, 0};
        vector<EmbCache::InitializerInfo> initializeInfos = {};
        std::vector<std::string> ssdDataPath = {""};
        std::vector<int64_t> paddingKeys = {1};
        vector<int> maxStep = {1000};
        embInfo_ = EmbInfo(embParam, vocabsize, initializeInfos, ssdDataPath, paddingKeys);
        int rankId;
        MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
        rankInfo_ = RankInfo(rankId, 0, 0, 1, maxStep);
        rankInfo_.useDynamicExpansion = true;
        savePath << "test_dir/device" << rankInfo_.rankId;
    }

    void SetUp() {
    }
    void TearDown() {
    }

    static void SetupTestCase()
    {
        if (access("test_dir", F_OK) == 0) {
            system("rm -rf test_dir");
        }
    }

    static void TearDownTestCase()
    {
        if (access("test_dir", F_OK) == 0) {
            system("rm -rf test_dir");
        }
    }

    EmbInfo embInfo_;
    RankInfo rankInfo_;
    stringstream savePath;
};

TEST_F(EmbeddingDynamicTest, TestMallocEmbeddingBlockShouldThrowErrorWhenNoUseDynamicExpansion)
{
    rankInfo_.useDynamicExpansion = false;
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    vector<emb_key_t> testKeys = {0,1,2,3};
    EXPECT_THROW(table->Key2Offset(testKeys, TRAIN_CHANNEL_ID), std::bad_alloc);
}

TEST_F(EmbeddingDynamicTest, TestKey2OffsetShouldReturnInvaildAddrWhenKeyOffsetMapEmpty)
{
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    const size_t keySize = 100;
    int invalidCount = 20;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < keySize; ++i) {
        if (i < invalidCount) {
            testKeys.push_back(INVALID_KEY_VALUE);
        } else {
            testKeys.push_back(i);
        }
    }

    table->Key2Offset(testKeys, EVAL_CHANNEL_ID);
    for (size_t i = 0; i < testKeys.size(); ++i) {
        EXPECT_EQ(testKeys[i], 0);
    }
}

TEST_F(EmbeddingDynamicTest, TestKey2OffsetShouldMaxOffsetEqualKeySizeWhenUseTrainChannelAndNoError)
{
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    const string tableName = "test1";
    const size_t keySize = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < keySize; ++i) {
        testKeys.push_back(i);
    }

    table->Key2Offset(testKeys, TRAIN_CHANNEL_ID);
    MxRec::KeyOffsetMemT kom;
    kom[tableName] = table->keyOffsetMap;
    map<EmbNameT, string> tmp;
    for (auto it = kom.begin(); it != kom.end(); ++it) {
        tmp.insert(pair<EmbNameT, string>(it->first, MapToString(it->second).c_str()));
    }
    LOG_INFO("test Key2Offset: lookupKeys: {}, keyOffsetMap: {}",
             VectorToString(testKeys), MapToString(tmp));

    EXPECT_EQ(table->capacity(), table->BLOCK_EMB_NUM);
    EXPECT_EQ(table->maxOffset, keySize);
}

TEST_F(EmbeddingDynamicTest, TestKey2OffsetForDpShouldReturnInvaildAddrWhenKeyOffsetMapEmpty)
{
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    const size_t keySize = 100;
    int invalidCount = 20;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < keySize; ++i) {
        if (i < invalidCount) {
            testKeys.push_back(INVALID_KEY_VALUE);
        } else {
            testKeys.push_back(i);
        }
    }

    table->Key2OffsetForDp(testKeys, EVAL_CHANNEL_ID);
    for (size_t i = 0; i < testKeys.size(); ++i) {
        EXPECT_EQ(testKeys[i], 0);
    }
}

TEST_F(EmbeddingDynamicTest, TestKey2OffsetForDpThrowErrorWhenUseTrainCannel)
{
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    const size_t keySize = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < keySize; ++i) {
        testKeys.push_back(i);
    }
    EXPECT_THROW(table->Key2OffsetForDp(testKeys, TRAIN_CHANNEL_ID), std::runtime_error);
}

TEST_F(EmbeddingDynamicTest, TestSaveWhenSaveDeltaIsFalse)
{
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    const string tableName = "test1";
    const size_t keySize = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < keySize; ++i) {
        testKeys.push_back(i);
    }
    table->Key2Offset(testKeys, TRAIN_CHANNEL_ID);

    OptimizerInfo info;
    info.optimName = "Adam";
    info.optimParams = {"momentum", "velocity"};
    table->SetOptimizerInfo(info);

    map<emb_key_t, KeyInfo> keyInfo;
    table->SetFileSystemPtr(savePath.str());
    table->Save(savePath.str(), 1, false, keyInfo);

    stringstream saveKeyPath;
    saveKeyPath << savePath.str() << "/" << tableName << "/key";
    stringstream saveEmbeddingPath;
    saveEmbeddingPath << savePath.str() << "/" << tableName << "/embedding";
    stringstream saveMomentumPath;
    saveMomentumPath << savePath.str() << "/" << tableName << "/" << info.optimName + "_" + info.optimParams[0];
    stringstream saveVelocityPath;
    saveVelocityPath << savePath.str() << "/" << tableName << "/" << info.optimName + "_" + info.optimParams[1];
    EXPECT_EQ(access(saveKeyPath.str().c_str(), F_OK), 0);
    EXPECT_EQ(access(saveEmbeddingPath.str().c_str(), F_OK), 0);
    EXPECT_EQ(access(saveMomentumPath.str().c_str(), F_OK), 0);
    EXPECT_EQ(access(saveVelocityPath.str().c_str(), F_OK), 0);
}

TEST_F(EmbeddingDynamicTest, TestLoadWhenSaveDeltaIsFalse)
{
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    const string tableName = "test1";
    const size_t keySize = 100;
    OptimizerInfo info;
    info.optimName = "Adam";
    info.optimParams = {"momentum", "velocity"};
    table->SetOptimizerInfo(info);

    stringstream fileKeyPath;
    fileKeyPath << savePath.str() << "/" << tableName << "/key/slice_" << rankInfo_.rankId << ".data";
    stringstream newFileKeyPath;
    newFileKeyPath << savePath.str() << "/" << tableName << "/key/slice.data";
    RenameFilePath(fileKeyPath.str(),newFileKeyPath.str());

    stringstream fileEmbeddingPath;
    fileEmbeddingPath << savePath.str() << "/" << tableName << "/embedding/slice_" << rankInfo_.rankId << ".data";
    stringstream newFileEmbeddingPath;
    newFileEmbeddingPath << savePath.str() << "/" << tableName << "/embedding/slice.data";
    RenameFilePath(fileEmbeddingPath.str(),newFileEmbeddingPath.str());

    stringstream fileMomentumPath;
    fileMomentumPath << savePath.str() << "/" << tableName << "/" << info.optimName + "_" + info.optimParams[0] << "/" << "slice_" << rankInfo_.rankId << ".data";
    stringstream newFileMomentumPath;
    newFileMomentumPath << savePath.str() << "/" << tableName << "/" << info.optimName + "_" + info.optimParams[0] << "/" << "slice.data";
    RenameFilePath(fileMomentumPath.str(), newFileMomentumPath.str());

    stringstream fileVelocityPath;
    fileVelocityPath << savePath.str() << "/" << tableName << "/" << info.optimName + "_" + info.optimParams[1] << "/" << "slice_" << rankInfo_.rankId << ".data";
    stringstream newFileVelocityPath;
    newFileVelocityPath << savePath.str() << "/" << tableName << "/" << info.optimName + "_" + info.optimParams[1] << "/" << "slice.data";
    RenameFilePath(fileVelocityPath.str(), newFileVelocityPath.str());

    map<string, unordered_set<emb_cache_key_t>> trainKeySet;
    vector<string> warmStartTables;
    table->SetFileSystemPtr(savePath.str());
    table->Load(savePath.str(), trainKeySet, warmStartTables);

    EXPECT_EQ(table->loadOffset.size(), keySize / table->rankSize_);
    EXPECT_EQ(table->deviceKey.size(), keySize / table->rankSize_);
    EXPECT_EQ(table->maxOffset, keySize / table->rankSize_);
}

TEST_F(EmbeddingDynamicTest, TestSaveKeyWhenSaveDeltaIsTrue)
{
    shared_ptr<EmbeddingDynamic> table = std::make_shared<EmbeddingDynamic>(embInfo_, rankInfo_, 0);
    const string tableName = "test1";
    const size_t keySize = 100;
    vector<emb_key_t> testKeys;
    for (size_t i = 0; i < keySize; ++i) {
        testKeys.push_back(i);
    }
    table->Key2Offset(testKeys, TRAIN_CHANNEL_ID);

    map<emb_key_t, KeyInfo> keyInfo;
    KeyInfo info;
    keyInfo[1] = info;
    stringstream savePathOne;
    savePathOne << "test_dir/SaveDeltaTrue" << rankInfo_.rankId;;
    table->SetFileSystemPtr(savePathOne.str());
    table->SaveKey(savePathOne.str(), true, keyInfo);

    stringstream saveKeyPathOne;
    saveKeyPathOne << savePathOne.str() << "/" << tableName << "/key";
    EXPECT_EQ(access(saveKeyPathOne.str().c_str(), F_OK), 0);
}