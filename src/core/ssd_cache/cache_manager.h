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

#ifndef MXREC_CACHE_MANAGER_H
#define MXREC_CACHE_MANAGER_H

#include <string>
#include <list>
#include <unordered_map>
#include <vector>
#include <unordered_set>

#include "hd_transfer/hd_transfer.h"
#include "host_emb/host_emb.h"
#include "lfu_cache.h"
#include "ssd_engine/ssd_engine.h"
#include "utils/common.h"
#include "preprocess_mapper.h"
#include "ock_ctr_common/include/factory.h"

namespace MxRec {

    struct TableInfo {
        std::string name;
        size_t hostVocabSize;
        size_t devVocabSize;
        size_t& maxOffset;
        absl::flat_hash_map<emb_key_t, int64_t>& keyOffsetMap;
        std::vector<int64_t>& evictDevPos;     // 记录HBM内被淘汰的key
        std::vector<int64_t>& evictHostPos; // 记录Host内淘汰列表
    };

    enum class TransferRet {
        TRANSFER_OK = 0, // 转移成功或无需处理
        TRANSFER_ERROR,
        SSD_SPACE_NOT_ENOUGH,
        DDR_SPACE_NOT_ENOUGH,
    };

    enum class TransferType {
        DDR_2_HBM = 0,
        DDR_2_EVICT,
        HBM_2_DDR,
        HBM_2_EVICT,
    };

    enum class RecordType {
        DDR = 0,
        NOT_DDR,
    };

    class CacheManager {
    public:
        CacheManager() = default;
        CacheManager(const CacheManager&) = delete;
        CacheManager& operator=(const CacheManager&) = delete;

        ~CacheManager();

        void Init(HostEmb* hostEmbPtr, vector<EmbInfo>& mgmtEmbInfo);

        void Init(ock::ctr::EmbCacheManagerPtr embCachePtr, vector<EmbInfo>& mgmtEmbInfo);

        void Load(unordered_map<std::string, unordered_map<emb_cache_key_t, freq_num_t>>& ddrFreqInitMap,
                  unordered_map<std::string, unordered_map<emb_cache_key_t, freq_num_t>>& excludeDdrFreqInitMap,
                  int step, int rankSize, int rankId);

        void Load(const std::vector<EmbInfo> &mgmtEmbInfo, int step);

        void SaveSSDEngine(int step);

        // 转换DDR和SSD数据
        TransferRet TransferDDREmbWithSSD(TableInfo& table,
                                          const vector<emb_key_t>& originalKeys, int channelId);

        /* HBM与DDR换入换出时刷新频次信息 */
        void RefreshFreqInfoCommon(const string& embTableName, vector<emb_key_t>& keys,
                                   TransferType type);

        bool IsKeyInSSD(const string& embTableName, emb_key_t key);

        void EvictSSDEmbedding(const string& embTableName, vector<emb_key_t>& keys);

        void EvictSSDEmbedding(const string& embTableName, vector<emb_cache_key_t>& keys);

        void PutKey(const string& embTableName, const emb_key_t& key, RecordType type);

        void ProcessSwapOutKeys(const string& tableName, const vector<emb_cache_key_t> &swapOutKeys,
                                vector<emb_cache_key_t> &swapOutDDRKeys,
                                vector<emb_cache_key_t> &swapOutDDRAddrOffs,
                                vector<emb_cache_key_t> &swapOutSSDKeys,
                                vector<emb_cache_key_t> &swapOutSSDAddrOffs);

        void ProcessSwapInKeys(const string& tableName, const vector<emb_cache_key_t> &swapInKeys,
                               vector<emb_cache_key_t> &DDRToSSDKeys, vector<emb_cache_key_t> &SSDToDDRKeys);

        void UpdateSSDEmb(string tableName, float *embPtr, uint32_t extEmbeddingSize,
                          vector<emb_cache_key_t> &keys, const vector<uint64_t> &swapOutSSDAddrOffs);

        void TransferDDR2SSD(string tableName, uint32_t extEmbeddingSize, vector<emb_cache_key_t> &keys,
                             vector<float *> &addrs);

        void FetchSSDEmb2DDR(string tableName, uint32_t extEmbeddingSize, vector<emb_cache_key_t> &keys,
                             const vector<float *> &addrs);

        int64_t GetTableEmbeddingSize(const string& tableName);

        // DDR内每个表中emb数据频次缓存；map<embTableName, 频次缓存>
        unordered_map<std::string, LFUCache> ddrKeyFreqMap;
        // 每张表中非DDR内key的出现次数
        unordered_map<std::string, unordered_map<emb_cache_key_t, freq_num_t>> excludeDDRKeyCountMap;

        // 每一个table对应一个PreProcessMapper，预先推演HBM->DDR的情况
        std::unordered_map<std::string, PreProcessMapper> preProcessMapper;

        int preProcessStep = 0;
        int embeddingTaskStep = 0;
        std::mutex evictWaitMut;
        std::condition_variable evictWaitCond;
        std::vector<std::thread> ssdEvictThreads;

    private:
        struct EmbBaseInfo {
            uint64_t maxTableSize;
            vector<std::string> savePath;
            bool isExist;
        };

        void GetDDREmbInfo(vector<emb_cache_key_t>& keys,
                           TableInfo& table,
                           vector<size_t>& ddrTransferPos, vector<vector<float>>& ddrEmbData) const;

        void UpdateDDREmbInfo(const std::string& embTableName,
                              vector<size_t>& ddrTransferPos,
                              vector<vector<float>>& ssdEmbData) const;

        void RefreshRelateInfoWithDDR2SSD(TableInfo& table,
                                          vector<emb_cache_key_t>& ddrSwapOutKeys,
                                          vector<freq_num_t>& ddrSwapOutCounts);

        void RefreshRelateInfoWithSSD2DDR(TableInfo& table,
                                          vector<emb_cache_key_t>& externalSSDKeys, vector<size_t>& ddrTransferPos);

        void GetSSDKeys(const std::string& embTableName, vector<emb_key_t>& externalKeys,
                        vector<emb_cache_key_t>& externalSSDKeys);

        TransferRet TransferDDREmb2SSD(TableInfo& table,
                                       int64_t ddrSwapOutSize, const vector<emb_cache_key_t>& keys,
                                       vector<size_t>& ddrTransferPos);

        TransferRet TransferSSDEmb2DDR(TableInfo& table,
                                       vector<emb_cache_key_t>& externalSSDKeys, vector<size_t>& ddrTransferPos,
                                       vector<vector<float>>& ssdEmbData);

        void CreateSSDTableIfNotExist(const std::string& embTableName);

        void RestoreLeastFreqInfo(const std::string& embTableName, vector<emb_cache_key_t>& ddrSwapOutKeys,
                                  vector<freq_num_t>& ddrSwapOutCounts);

        static void HandleDDRTransferPos(vector<size_t>& ddrTransferPos, vector<emb_cache_key_t>& externalSSDKeys,
                                         TableInfo& table);

        inline void GetExternalKeys(const absl::flat_hash_map<emb_key_t, int64_t> &keyOffsetMap,
                                    vector<emb_key_t>& externalKeys,
                                    vector<emb_cache_key_t>& internalKeys, const vector<emb_key_t>& keys) const;

        void AddDebugAndTraceLog(size_t batchKeySize, vector<emb_key_t>& externalKeys,
                                 vector<emb_cache_key_t>& externalSSDKeys) const;

        void HandleRepeatAndInvalidKey(const vector<emb_key_t>& originalKeys, vector<emb_key_t>& keys) const;

        unordered_map<std::string, EmbBaseInfo> embBaseInfos;

    GTEST_PRIVATE:
        shared_ptr<SSDEngine> ssdEngine = std::make_shared<SSDEngine>();
        HostEmb* hostEmbs {};
        ock::ctr::EmbCacheManagerPtr embCache {};
    };
}

#endif // MXREC_CACHE_MANAGER_H
