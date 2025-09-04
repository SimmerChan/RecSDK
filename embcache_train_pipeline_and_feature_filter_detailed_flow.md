 # EmbCache训练流水线和准入淘汰机制详细流程图

## 1. 整体架构图

```mermaid
graph TD
    A[PyTorch训练任务] --> B[训练数据]
    B --> C[EmbCacheTrainPipelineSparseDist]
    C --> D[特征处理与分发]
    D --> E[模型前向传播]
    E --> F[损失计算]
    F --> G[反向传播]
    G --> H[参数更新]
    H --> I[特征淘汰]
    I --> J[训练完成/下一轮]

    subgraph 训练流水线核心组件
        C
        D
        E
        F
        G
        H
        I
    end

    subgraph 准入淘汰机制
        K[特征统计]
        L[准入过滤]
        M[时间戳记录]
        N[特征淘汰]
        K --> L --> M --> N
    end

    C -.-> K
    C -.-> L
    C -.-> M
    C -.-> N
```

## 2. 训练流水线详细流程图

```mermaid
graph TD
    A[训练开始] --> B[fill_pipeline阶段]
    B --> C[progress核心处理阶段]
    C --> D{是否还有数据}
    D -->|是| E[处理当前batch]
    E --> F[推进流水线]
    F --> D
    D -->|否| G[训练结束]
    
    subgraph fill_pipeline阶段
        B1[Batch i: 初始化和预处理]
        B2[Batch i+1: 数据分发准备]
        B3[Batch i+2: 特征处理]
        B4[Batch i+3: 等待处理]
        B5[Batch i+4~i+n: 并行处理]
        B --> B1
        B1 --> B2
        B2 --> B3
        B3 --> B4
        B4 --> B5
    end
    
    subgraph progress核心处理阶段
        C1[前向传播阶段]
        C2[反向传播阶段]
        C3[参数更新阶段]
        C --> C1
        C1 --> C2
        C2 --> C3
        C3 --> E
    end
```

## 3. fill_pipeline阶段详细流程图

```mermaid
graph TD
    FP[fill_pipeline开始] --> B0{是否已填充}
    B0 -->|是| FP_END[结束]
    B0 -->|否| B1[检查是否执行最后批次]
    B1 -->|是| FP_END
    B1 -->|否| B2[入队Batch i]
    
    B2 --> B3[初始化流水线模块]
    B3 --> B4[等待稀疏数据分发完成]
    B4 --> B5[执行后处理输入分发]
    B5 --> B6[等待batch数据准备完成]
    B6 --> B7[开始计算swap信息]
    B7 --> B8[等待并获取swap信息]
    B8 --> B9[异步主机embedding查找]
    B9 --> B10[异步恢复操作]
    B10 --> B11[记录swapout事件]
    B11 --> B12[执行swapout操作]
    
    B12 --> B13[入队Batch i+1]
    B13 --> B14[开始稀疏数据分发]
    B14 --> B15[融合输入分发拆分]
    B15 --> B16[等待稀疏数据分发完成]
    B16 --> B17[执行后处理输入分发]
    B17 --> B18[等待batch数据准备完成]
    B18 --> B19[开始计算swap信息]
    
    B19 --> B20[入队Batch i+2]
    B20 --> B21[开始稀疏数据分发]
    B21 --> B22[融合输入分发拆分]
    B22 --> B23[等待稀疏数据分发完成]
    B23 --> B24[执行后处理输入分发]
    B24 --> B25[等待batch数据准备完成]
    
    B25 --> B26[入队Batch i+3]
    B26 --> B27[开始稀疏数据分发]
    B27 --> B28[融合输入分发拆分]
    
    B28 --> B29[入队Batch i+4 ~ i+4+n]
    B29 --> B30[开始稀疏数据分发]
    B30 --> FP_END
```

## 4. progress核心处理阶段详细流程图

```mermaid
graph TD
    P[progress开始] --> P1[增加全局步骤计数]
    P1 --> P2[检查模型是否已附加]
    P2 --> P3[填充流水线]
    P3 --> P4[检查是否有batch数据]
    P4 -->|否| P_ERR[抛出StopIteration异常]
    P4 -->|是| P5[设置模块上下文]
    
    P5 --> P6{模型是否训练模式}
    P6 -->|是| P7[梯度清零]
    P6 -->|否| P8[跳过梯度清零]
    P7 --> P8
    
    P8 --> P9{流水线长度>=4}
    P9 -->|是| P10[等待batch i+2数据分发完成]
    P9 -->|否| P11[跳过等待]
    P10 --> P11
    
    P11 --> P12{流水线长度>=5}
    P12 -->|是| P13[融合输入分发拆分]
    P12 -->|否| P14[跳过融合]
    P13 --> P14
    
    P14 --> P15[入队下一个batch]
    
    P15 --> P16{流水线长度>=5+n}
    P16 -->|是| P17[开始稀疏数据分发]
    P16 -->|否| P18[跳过数据分发]
    P17 --> P18
    
    P18 --> P19[同步等待swapout事件]
    P19 --> P20[异步更新主机侧Embedding]
    P20 --> P21[等待主机更新完成]
    
    P21 --> P22[将swapin张量拷贝到NPU]
    P22 --> P23[等待并执行swapin操作]
    
    P23 --> FW[前向传播阶段]
```

## 5. 前向传播阶段详细流程图

```mermaid
graph TD
    FW[前向传播阶段开始] --> FW1[执行模型前向传播]
    FW1 --> FW2[计算损失和输出]
    
    FW2 --> FW3{流水线长度>=2}
    FW3 -->|是| FW4[等待i+1 swap参数计算完成]
    FW3 -->|否| FW5[跳过等待]
    FW4 --> FW5
    
    FW5 --> FW6{是否需要执行特征淘汰}
    FW6 -->|是| FW7[执行特征淘汰]
    FW6 -->|否| FW8[跳过特征淘汰]
    FW7 --> FW8
    
    FW8 --> FW9{流水线长度>=3}
    FW9 -->|是| FW10[计算i+2 batch的swap信息]
    FW9 -->|否| FW11[跳过计算]
    FW10 --> FW11
    
    FW11 --> FW12{流水线长度>=2}
    FW12 -->|是| FW13[记录swapout事件]
    FW13 --> FW14[异步主机embedding查找]
    FW14 --> FW15[异步恢复操作]
    FW12 -->|否| FW16[跳过操作]
    FW15 --> FW16
    
    FW16 --> FW17{流水线长度>=2}
    FW17 -->|是| FW18[执行swapout操作]
    FW17 -->|否| FW_END[前向传播阶段结束]
    FW18 --> FW_END
```

## 6. 反向传播阶段详细流程图

```mermaid
graph TD
    BW[反向传播阶段开始] --> BW1{模型是否训练模式}
    BW1 -->|否| BW_END[跳过反向传播]
    BW1 -->|是| BW2[执行反向传播计算梯度]
    BW2 --> BW3[执行优化器更新参数]
    BW3 --> BW_END[反向传播阶段结束]
```

## 7. 参数更新阶段详细流程图

```mermaid
graph TD
    PU[参数更新阶段开始] --> PU1{流水线长度>=4}
    PU1 -->|是| PU2[执行后处理输入分发]
    PU1 -->|否| PU3[跳过处理]
    PU2 --> PU3
    
    PU3 --> PU4{流水线长度>=2}
    PU4 -->|是| PU5[执行swapout操作]
    PU4 -->|否| PU6[跳过操作]
    PU5 --> PU6
    
    PU6 --> PU7[等待主机更新完成]
    PU7 --> PU8[出队已完成的batch]
    PU8 --> PU_END[参数更新阶段结束]
```

## 8. 准入淘汰机制在流水线中的位置图

```mermaid
graph TD
    A[训练流水线] --> B[特征统计阶段]
    B --> C[准入过滤阶段]
    C --> D[时间戳记录阶段]
    D --> E[特征淘汰阶段]
    
    subgraph 特征统计阶段
        B1[第一次All2All后]
        B2[统计特征访问次数]
        B3[更新featureRecordMap]
        B1 --> B2 --> B3
    end
    
    subgraph 准入过滤阶段
        C1[前向传播前]
        C2[应用准入过滤]
        C3[过滤未达标特征]
        C1 --> C2 --> C3
    end
    
    subgraph 时间戳记录阶段
        D1[第四次All2All后]
        D2[记录特征时间戳]
        D3[更新timestampRecordMap]
        D1 --> D2 --> D3
    end
    
    subgraph 特征淘汰阶段
        E1[定期检查淘汰条件]
        E2[执行FeatureEvict]
        E3[标记待淘汰特征]
        E4[实际移除特征]
        E1 --> E2 --> E3 --> E4
    end
```

## 9. 数据准确性保障机制图

```mermaid
graph TD
    A[数据准确性保障] --> B[特征统计准确性]
    A --> C[时间戳记录准确性]
    A --> D[准入淘汰同步]
    
    subgraph 特征统计准确性
        B1[流水线顺序执行]
        B2[featureRecordMap存储]
        B3[并发安全保证]
        B1 --> B2 --> B3
    end
    
    subgraph 时间戳记录准确性
        C1[定期记录时间戳]
        C2[淘汰时机控制]
        C3[淘汰同步机制]
        C1 --> C2 --> C3
    end
    
    subgraph 准入淘汰同步
        D1[流水线同步]
        D2[事件同步]
        D3[计数同步]
        D1 --> D2 --> D3
    end
```

## 10. 四次All2All操作与准入淘汰交互图

```mermaid
graph TD
    A[训练流程开始] --> B[第一次All2All]
    B --> C[特征统计和准入控制]
    C --> D[第二次All2All]
    D --> E[特征聚合]
    E --> F[前向传播]
    F --> G[第三次All2All]
    G --> H[梯度分发]
    H --> I[反向传播]
    I --> J[第四次All2All]
    J --> K[时间戳记录和特征淘汰]
    K --> L[训练流程结束]
    
    subgraph 第一次All2All
        B1[特征分发]
        B2[EmbCacheRwSparseFeaturesDist._dist]
        B1 --> B2
    end
    
    subgraph 特征统计和准入控制
        C1[StatisticsKeyCount]
        C2[CountFilter]
        C1 --> C2
    end
    
    subgraph 第二次All2All
        D1[特征聚合]
        D2[SparseFeaturesPostDist._dist]
        D1 --> D2
    end
    
    subgraph 第三次All2All
        G1[梯度分发]
        G2[反向传播计算梯度]
        G1 --> G2
    end
    
    subgraph 第四次All2All
        J1[梯度聚合]
        J2[参数更新]
        J1 --> J2
    end
    
    subgraph 时间戳记录和特征淘汰
        K1[RecordTimestamp]
        K2[FeatureEvict]
        K3[EvictFeatures]
        K1 --> K2 --> K3
    end
```

## 11. 准入淘汰机制详细流程图

### 11.1 特征统计流程图

```mermaid
graph TD
    A[训练数据输入] --> B[创建KeyedJaggedTensorWithCount]
    B --> C[RwSparseFeaturesDist._forward_func]
    C --> D[bucketize_kjt_before_all2all]
    D --> E{是否开启local unique和准入}
    E -->|是| F[返回KeyedJaggedTensorWithCount]
    E -->|否| G[返回KeyedJaggedTensor]
    F --> H[第一次All2All]
    G --> H
    H --> I[PostInputDist处理]
    I --> J[do_unique_hash_out]
    J --> K[HashMapBase.statistic_key_count]
    K --> L[EmbcacheManager.StatisticsKeyCount]
    L --> M[FeatureFilter.StatisticsKeyCount]
    M --> N[更新featureRecordMap]
```

### 11.2 准入过滤流程图

```mermaid
graph TD
    A[模型前向传播] --> B[EmbeddingCollection计算]
    B --> C[lookup操作]
    C --> D{是否启用准入控制}
    D -->|否| E[正常lookup]
    D -->|是| F[检查特征准入状态]
    F --> G[FeatureFilter.CountFilter]
    G --> H{特征访问次数>=阈值}
    H -->|是| I[允许特征准入]
    H -->|否| J[拒绝特征准入并标记为无效]
    I --> K[正常lookup]
    J --> K
    K --> L[返回lookup结果]
```

### 11.3 时间戳记录流程图

```mermaid
graph TD
    A[输入分发阶段] --> B[EmbCacheShardedEmbeddingCollection.input_dist]
    B --> C{是否有时间戳数据}
    C -->|否| D[跳过时间戳记录]
    C -->|是| E[_record_timestamp_data]
    E --> F{是否启用特征淘汰}
    F -->|否| D
    F -->|是| G[EmbcacheManager.record_timestamp]
    G --> H[FeatureFilter.RecordTimestamp]
    H --> I[更新timestampRecordMap]
    I --> J{是否需要执行淘汰}
    J -->|是| K[FeatureFilter.FeatureEvict]
    J -->|否| L[继续执行]
    K --> M[标记待淘汰特征]
```

### 11.4 特征淘汰流程图

```mermaid
graph TD
    A[训练步骤计数] --> B{是否达到淘汰间隔}
    B -->|否| C[跳过淘汰]
    B -->|是| D[_start_feature_evict]
    D --> E[EmbCacheShardedEmbeddingCollection.host_embedding_evict]
    E --> F[EmbcacheManager.evict_features]
    F --> G[FeatureFilter.evictFeatureRecord.GetEvictKeys]
    G --> H[SwapManager.RemoveKeys]
    H --> I[FeatureFilter.evictFeatureRecord.SetSwapCount]
    I --> J[EmbcacheManager.RecordEmbeddingUpdateTimes]
    J --> K{是否需要移除Embedding}
    K -->|否| L[继续执行]
    K -->|是| M[EmbcacheManager.RemoveEmbeddingTableInfo]
    M --> N[EmbTable.RemoveEmbedding]
    N --> O[EvictFeatureRecord.ClearEvictInfo]
```

## 12. 准入淘汰机制与训练流水线集成图

```mermaid
graph TD
    A[训练流水线progress方法] --> B[前向传播]
    B --> C[特征统计]
    C --> D[准入过滤]
    D --> E[模型计算]
    E --> F[反向传播]
    F --> G[参数更新]
    G --> H{是否达到淘汰步数}
    H -->|否| I[继续下一轮]
    H -->|是| J[执行特征淘汰]
    J --> K[记录时间戳]
    K --> L[检查超时特征]
    L --> M[标记待淘汰特征]
    M --> N[实际移除特征]
    N --> I
```