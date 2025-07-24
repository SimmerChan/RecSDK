/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_EMB_TABLE_INITIALIZER_H
#define EMBEDDING_CACHE_EMB_TABLE_INITIALIZER_H

#include <cstddef>
#include <cstring>
#include <random>
#include <algorithm>
#include <vector>

#include "common/common.h"

namespace Embcache {
struct WeightInitParam {
    float mean;
    float stddev;
    float minVal;
    float maxVal;
};

class Initializer {
public:
    static void GenUniform(float* array, size_t size, float minVal, float maxVal)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib(minVal, maxVal);
        std::generate(array, array + size, [&]() { return distrib(gen); });
    }

    static void GenLinear(float* array, size_t size, float minVal, float maxVal)
    {
        if (size == 0) {
            return;
        }
        if (size == 1) {
            array[0] = minVal;
            return;
        }
        for (size_t i = 0; i < size; ++i) {
            array[i] = minVal + 1.0 * i / (size - 1) * (maxVal - minVal);
        }
    }

    static void GenTruncatedNormal(float* array, size_t size, WeightInitParam weightParam,
                                   unsigned int seed = std::random_device{}())
    {
        if (array == nullptr || size == 0 || weightParam.stddev <= 0.0f || weightParam.minVal >= weightParam.maxVal) {
            return;
        }

        std::mt19937 gen(seed);
        
        std::normal_distribution<float> distrib(weightParam.mean, weightParam.stddev);

        std::generate(array, array + size, [&]() {
            float val = distrib(gen);
            while (val < weightParam.minVal || val > weightParam.maxVal) {
                val = distrib(gen);
            }
            return val;
        });
    }

    static void InitEmbeddingWeights(float* embeddingAddr, const EmbConfig& cfg)
    {
        if (cfg.initializerType == InitializerType::LINEAR) {
            Initializer::GenLinear(embeddingAddr, cfg.embDim, cfg.weightInitMin, cfg.weightInitMax);
        } else if (cfg.initializerType == InitializerType::TRUNCATED_NORMAL) {
            WeightInitParam param = {cfg.weightInitMean, cfg.weightInitStddev,
                                     cfg.weightInitMin, cfg.weightInitMax};
            Initializer::GenTruncatedNormal(embeddingAddr, cfg.embDim, param);
        } else {
            Initializer::GenUniform(embeddingAddr, cfg.embDim, cfg.weightInitMin, cfg.weightInitMax);
        }
    }

    static void InitEmbeddingWeightsLimitPool(float* embeddingAddr, const EmbConfig& cfg,
                                   unsigned int seed = std::random_device{}())
    {
        static std::vector<std::vector<float>> staticPool;
        static std::default_random_engine engine;
        if (staticPool.empty()) {
            engine.seed(abs(cfg.seed));
            staticPool = std::vector<std::vector<float>>(cfg.initializerRadomPoolSize, std::vector<float>(cfg.embDim));
            for(size_t i = 0; i < cfg.initializerRadomPoolSize; i++) {
                if (cfg.initializerType == InitializerType::LINEAR) {
                    Initializer::GenLinear(staticPool[i].data(), cfg.embDim, cfg.weightInitMin, cfg.weightInitMax);
                } else if (cfg.initializerType == InitializerType::TRUNCATED_NORMAL) {
                    WeightInitParam param = {cfg.weightInitMean, cfg.weightInitStddev,
                                            cfg.weightInitMin, cfg.weightInitMax};
                    Initializer::GenTruncatedNormal(staticPool[i].data(), cfg.embDim, param);
                } else {
                    Initializer::GenUniform(staticPool[i].data(), cfg.embDim, cfg.weightInitMin, cfg.weightInitMax);
                }
            }
        } else {
            std::uniform_int_distribution<int> uDistribution(0, cfg.initializerRadomPoolSize-1);
            int randIndex = uDistribution(engine);
            std::memcpy(embeddingAddr, staticPool[randIndex].data(), cfg.embDim*sizeof(float));
        }
    }
};

}  // namespace Embcache
#endif  // EMBEDDING_CACHE_EMB_TABLE_INITIALIZER_H
