# Feature Filter功能设计文档

## 1. 概述

该设计文档详细描述了在Embedding Cache中新增Feature Filter功能的完整实现方案。Feature Filter功能主要用于控制特征的准入和淘汰策略，通过设置准入阈值和淘汰阈值来优化Embedding Cache的使用效率。该功能包含两个核心组件：FeatureFilter（特征过滤器）和EvictFeatureRecord（特征淘汰记录），它们协同工作以实现特征的准入控制和淘汰机制。

## 2. 功能需求

1. 支持基于访问频率的特征准入控制
2. 支持基于时间戳的特征淘汰机制
3. 提供可配置的准入阈值和淘汰阈值
4. 支持特征淘汰记录和管理
5. 支持特征记录的加载和持久化
6. 支持特征访问计数统计
7. 支持特征时间戳记录和处理

## 3. 系统架构

### 3.1 详细架构图

```mermaid
graph TD
    A[PyTorch/TorchRec训练任务] --> B{输入数据类型}
    B -->|带计数数据| C[KeyedJaggedTensorWithCount]
    B -->|带时间戳数据| D[KeyedJaggedTensorWithTimestamp]
    
    subgraph 数据预处理阶段
        C --> E[PostInputDist]
        D --> E
        E --> F[HashMap::do_unique_hash_out]
        F --> G[FeatureFilter::StatisticsKeyCount]
        G --> H[featureRecordMap记录访问次数]
    end
    
    subgraph 特征处理与时间戳记录阶段
        C --> I[Embedding::forward]
        D --> I
        I --> J[EmbCacheManager::_record_timestamp_data]
        J --> K[FeatureFilter::RecordTimestamp]
        K --> L[timestampRecordMap记录时间戳]
        K --> M{是否需要执行淘汰检查}
        M -->|是| N[FeatureFilter::FeatureEvict]
        N --> O[检查超时特征]
        O --> P[evictFeatureRecord记录待淘汰特征]
        M -->|否| Q[继续执行]
    end
    
    subgraph 准入控制阶段
        I --> R[EmbCacheManager::GetSwapInfo]
        R --> S{是否启用准入控制}
        S -->|是| T[FeatureFilter::CountFilter]
        T --> U[检查特征访问次数]
        U --> V{访问次数 >= 准入阈值}
        V -->|是| W[允许特征准入]
        V -->|否| X[拒绝特征准入]
        S -->|否| Y[直接加载特征]
    end
    
    subgraph 缓存交换阶段
        R --> Z[SwapManager::GetSwapInfo]
        Z --> AA[SwapInfo]
    end
    
    subgraph 异步更新与特征淘汰阶段
        I --> AB[EmbCacheManager::EmbeddingUpdateAsync]
        AB --> AC[异步执行EmbeddingUpdate]
        AC --> AD[EmbCacheManager::RecordEmbeddingUpdateTimes]
        AD --> AE[embUpdateCount_递增]
        AE --> AF{是否需要淘汰特征}
        AF -->|是| AG[EmbCacheManager::RemoveEmbeddingTableInfo]
        AG --> AH[从EmbeddingTable移除特征]
        AF -->|是| AI[EmbCacheManager::EvictFeatures]
        AI --> AJ[SwapManager::RemoveKeys]
    end
    
    style A fill:#FFFF00,stroke:#333
    style C fill:#FFFF00,stroke:#333
    style D fill:#FFFF00,stroke:#333
    style E fill:#FFFF00,stroke:#333
    style I fill:#FFFF00,stroke:#333
    style J fill:#FFFF00,stroke:#333
    style K fill:#FFFF00,stroke:#333
    style T fill:#90EE90,stroke:#333
    style N fill:#90EE90,stroke:#333
    style W fill:#90EE90,stroke:#333
    style X fill:#FFB6C1,stroke:#333
```

### 3.2 架构说明

#### 3.2.1 数据输入层
- 支持两种输入数据类型：`KeyedJaggedTensorWithCount` 和 `KeyedJaggedTensorWithTimestamp`
- 数据经过 `PostInputDist` 预处理

#### 3.2.2 特征统计与记录
- 通过 `FeatureFilter::StatisticsKeyCount` 统计特征访问次数并记录到 `featureRecordMap`
- 通过 `FeatureFilter::RecordTimestamp` 记录时间戳到 `timestampRecordMap`

#### 3.2.3 准入控制机制
- 可配置的准入控制开关
- 基于访问次数的准入策略，通过 `FeatureFilter::CountFilter` 实现
- 未达到阈值的特征将被拒绝准入

#### 3.2.4 特征淘汰机制
- 定期检查是否需要执行特征淘汰
- 通过 `FeatureFilter::FeatureEvict` 检查超时特征
- 记录待淘汰特征到 `evictFeatureRecord`

#### 3.2.5 缓存管理
- 通过 `SwapManager` 管理缓存交换信息
- 异步更新机制保证系统性能

#### 3.2.6 异步更新与清理
- 异步执行 `EmbeddingUpdate`
- 定期清理和淘汰不需要的特征
- 从 `EmbeddingTable` 和 `SwapManager` 中移除淘汰的特征

## 4. 设计方案

### 4.1 核心组件

#### 4.1.1 EvictFeatureRecord（特征淘汰记录）

EvictFeatureRecord类用于记录和管理待淘汰的特征信息：

```cpp
class EvictFeatureRecord {
public:
    EvictFeatureRecord() = default;
    bool CanRemoveFromEmbTable(uint64_t embUpdateCount) const;
    void ClearEvictInfo();
    void SetSwapCount(uint64_t swapCount);
    std::vector<int64_t>& GetEvictKeys();

private:
    uint64_t executeSwapCount = 0;
    std::vector<int64_t> evictKeys;
};
```

主要功能：
- CanRemoveFromEmbTable: 判断是否可以从Embedding表中移除特征
- ClearEvictInfo: 清除淘汰信息
- SetSwapCount: 设置交换计数
- GetEvictKeys: 获取待淘汰的键列表

#### 4.1.2 FeatureFilter（特征过滤器）

FeatureFilter类实现了具体的特征准入和淘汰逻辑：

```cpp
class FeatureFilter {
public:
    FeatureFilter(const std::string& tableName, int32_t admitThreshold, uint64_t evictThreshold,
                  uint64_t evictStepInterval);

    void StatisticsKeyCount(const int64_t* featureDataPtr, const int64_t* countDataPtr, int64_t startIndex,
                            int64_t endIndex, bool isCountDataEmpty);

    void CountFilter(int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex);

    void RecordTimestamp(const int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex,
                         const int64_t* timestampDataPtr);

    void FeatureEvict();

    const std::unordered_map<int64_t, FeatureRecord>& GetFeatureCountMap();
    const std::unordered_map<int64_t, std::time_t>& GetFeatureTimestampMap();

    void LoadFeatureRecords(const std::vector<int64_t>& keys, std::vector<uint64_t>& counts);
    void LoadTimestampRecords(const std::vector<int64_t>& keys, std::vector<int64_t>& timestamps);

private:
    std::string tableName;
    int32_t admitThreshold_ = -1;
    std::unordered_map<int64_t, FeatureRecord> featureRecordMap;
    uint64_t evictThreshold = 0;
    uint64_t evictStepInterval = 0;
    uint64_t recordTsBatchId = 0;
    std::time_t latestTimestamp = 0;
    std::unordered_map<int64_t, std::time_t> timestampRecordMap;
};
```

主要功能：
- StatisticsKeyCount: 统计特征访问次数
- CountFilter: 根据访问次数过滤特征
- RecordTimestamp: 记录特征时间戳
- FeatureEvict: 执行特征淘汰
- LoadFeatureRecords: 加载特征记录
- LoadTimestampRecords: 加载时间戳记录

### 4.2 类图

```mermaid
classDiagram
    class EmbcacheManager {
        -embNum_ int32_t
        -embConfigs_ std::vector~EmbConfig~
        -swapManagers_ std::vector~SwapManager~
        -embeddingTables_ std::vector~std::unique_ptr~EmbTable~~
        -featureFilters std::vector~FeatureFilter~
        -swapCount_ uint64_t
        -embUpdateCount_ uint64_t
        -enableFastHashMap_ bool
        -optimNum_ int32_t
        -needAccumulateOffset_ bool
        +ComputeSwapInfoAsync() AsyncTask~SwapInfo~
        +EmbeddingLookupAsync() AsyncTask~SwapinTensor~
        +EmbeddingUpdateAsync() AsyncTask~void~
        +EvictFeatures() void
        +RecordTimestamp() void
        +StatisticsKeyCount() void
        +RecordEmbeddingUpdateTimes() void
        -ComputeSwapInfo() SwapInfo
        -EmbeddingLookup() SwapinTensor
        -EmbeddingUpdate() void
        -NeedEvictEmbeddingTable() bool
        -RemoveEmbeddingTableInfo() void
    }
    
    class FeatureFilter {
        -evictFeatureRecord_ EvictFeatureRecord
        -tableName_ std::string
        -admitThreshold_ int32_t
        -featureRecordMap_ std::unordered_map~int64_t, FeatureRecord~
        -evictThreshold_ uint64_t
        -evictStepInterval_ uint64_t
        -recordTsBatchId_ uint64_t
        -latestTimestamp_ std::time_t
        -timestampRecordMap_ std::unordered_map~int64_t, std::time_t~
        +FeatureFilter() 
        +StatisticsKeyCount() void
        +CountFilter() void
        +RecordTimestamp() void
        +FeatureEvict() void
        +GetFeatureCountMap() const std::unordered_map~int64_t, FeatureRecord~&
        +GetFeatureTimestampMap() const std::unordered_map~int64_t, std::time_t~&
        +LoadFeatureRecords() void
        +LoadTimestampRecords() void
    }
    
    class EvictFeatureRecord {
        -executeSwapCount_ uint64_t
        -evictKeys_ std::vector~int64_t~
        +EvictFeatureRecord()
        +CanRemoveFromEmbTable() bool
        +ClearEvictInfo() void
        +SetSwapCount() void
        +GetEvictKeys() std::vector~int64_t~&
    }
    
    class FeatureRecord {
        +count uint64_t
    }
    
    class SwapManager {
        -swapinKeys_ std::vector~std::vector~int64_t~~
        -swapoutKeys_ std::vector~std::vector~int64_t~~
        +SwapInKeys() std::vector~std::vector~int64_t~~
        +SwapOutKeys() std::vector~std::vector~int64_t~~
        +RemoveKeys() void
    }
    
    class EmbTable {
        +RemoveEmbedding() void
    }
    
    class SwapInfo {
        +swapoutKeys std::vector~std::vector~int64_t~~
        +swapoutOffs at::Tensor
        +swapinKeys std::vector~std::vector~int64_t~~
        +swapinOffs at::Tensor
        +batchOffs at::Tensor
        +swapinKeysLengthPreSum std::vector~int64_t~
        +swapoutKeysLengthPreSum std::vector~int64_t~
    }
    class JaggedTensorWithCount {
        +JaggedTensorWithCount()
        +counts torch.Tensor
    }
    
    class KeyedJaggedTensorWithCount {
        +KeyedJaggedTensorWithCount()
        +counts torch.Tensor
        +from_jt_dict()
        +split()
    }
    
    class JaggedTensorWithTimestamp {
        +JaggedTensorWithTimestamp()
        +timestamps torch.Tensor
    }
    
    class KeyedJaggedTensorWithTimestamp {
        +KeyedJaggedTensorWithTimestamp()
        +timestamps torch.Tensor
        +from_jt_dict()
        +split()
    }
    
    FeatureFilter --> EvictFeatureRecord : uses
    FeatureFilter --> FeatureRecord : uses
    KeyedJaggedTensorWithCount --> JaggedTensorWithCount : contains
    KeyedJaggedTensorWithCount --> KeyedJaggedTensor : extends
    JaggedTensorWithCount --> JaggedTensor : extends
    KeyedJaggedTensorWithTimestamp --> JaggedTensorWithTimestamp : contains
    KeyedJaggedTensorWithTimestamp --> KeyedJaggedTensor : extends
    JaggedTensorWithTimestamp --> JaggedTensor : extends
    EmbcacheManager --> "1" FeatureFilter : manages
    EmbcacheManager --> "1" SwapManager : uses
    EmbcacheManager --> "1" EmbTable : manages
    EmbcacheManager --> "creates" SwapInfo : creates
```

### 4.3 特征准入控制

#### 4.3.1 特征准入流程

```mermaid
graph TD
    A[PyTorch训练任务开始] --> B[创建KeyedJaggedTensorWithCount]
    B --> C[调用Embedding::forward]
    C --> D[EmbCacheManager::StatisticsKeyCount]
    D --> E[FeatureFilter::StatisticsKeyCount]
    E --> F[更新featureRecordMap_访问计数]
    C --> G[EmbCacheManager::ComputeSwapInfo]
    G --> H{是否启用准入控制?}
    H -->|是| I[FeatureFilter::CountFilter]
    H -->|否| J[跳过准入控制]
    I --> K[查询featureRecordMap_中的访问次数]
    K --> L{访问次数 >= 准入阈值?}
    L -->|是| M[特征准入到Embedding Cache]
    L -->|否| N[特征被标记为无效]
    J --> O[直接加载特征]
    M --> P[继续处理]
    N --> P
    O --> P
    
    style A fill:#FFFF00,stroke:#333
    style B fill:#FFFF00,stroke:#333
    style C fill:#FFFF00,stroke:#333
    style D fill:#FFFF00,stroke:#333
    style E fill:#90EE90,stroke:#333
    style I fill:#90EE90,stroke:#333
    style M fill:#90EE90,stroke:#333
    style N fill:#FFB6C1,stroke:#333
```

#### 4.3.2 特征准入时序图

```mermaid
sequenceDiagram
    participant Train as PyTorch训练任务
    participant KJT as KeyedJaggedTensorWithCount
    participant E as Embedding
    participant EM as EmbCacheManager
    participant FF as FeatureFilter
    participant FRM as featureRecordMap_
    
    Train->>KJT: 创建带计数的特征数据
    KJT->>E: 提供特征数据和计数信息
    E->>EM: StatisticsKeyCount(batchKeys, offset, counts, tableIndex)
    EM->>FF: StatisticsKeyCount(featureDataPtr, countDataPtr, startIndex, endIndex, isCountDataEmpty)
    FF->>FRM: 更新特征访问次数
    FRM-->>FF: 更新完成
    
    E->>EM: ComputeSwapInfo(batchKeys, offsetPerKey, tableIndices)
    EM->>EM: 检查是否启用准入控制
    alt 启用准入控制
        EM->>FF: CountFilter(featureDataPtr, startIndex, endIndex)
        FF->>FRM: 查询特征访问次数
        FRM-->>FF: 返回特征访问次数
        FF->>FF: 比较次数与准入阈值
        alt 访问次数 < 准入阈值
            FF->>E: 将特征标记为无效(-1)
        else 访问次数 >= 准入阈值
            FF->>E: 保留特征值
        end
    else 不启用准入控制
        EM->>E: 直接加载特征
    end
```

### 4.4 特征淘汰机制

#### 4.4.1 特征淘汰详细流程

```mermaid
graph TD
    A[PyTorch训练任务开始] --> B[创建KeyedJaggedTensorWithTimestamp]
    B --> C[调用Embedding::forward]
    C --> D[EmbCacheManager::_record_timestamp_data]
    D --> E[EmbCacheManager::RecordTimestamp]
    E --> F[FeatureFilter::RecordTimestamp]
    F --> G[更新timestampRecordMap_时间戳]
    F --> H{是否需要执行淘汰检查?}
    H -->|是| I[FeatureFilter::FeatureEvict]
    H -->|否| J[继续执行]
    I --> K[遍历timestampRecordMap_检查超时]
    K --> L{特征时间戳超阈值?}
    L -->|是| M[记录到EvictFeatureRecord]
    L -->|否| N[保留特征]
    M --> O[EmbCacheManager::EvictFeatures]
    O --> P[SwapManager::RemoveKeys]
    P --> Q[EmbTable::RemoveEmbedding]
    N --> R[结束]
    J --> R
    
    style A fill:#FFFF00,stroke:#333
    style B fill:#FFFF00,stroke:#333
    style C fill:#FFFF00,stroke:#333
    style D fill:#FFFF00,stroke:#333
    style E fill:#FFFF00,stroke:#333
    style F fill:#90EE90,stroke:#333
    style I fill:#90EE90,stroke:#333
    style M fill:#90EE90,stroke:#333
```

#### 4.4.2 特征淘汰时序图

```mermaid
sequenceDiagram
    participant Train as PyTorch训练任务
    participant KJT as KeyedJaggedTensorWithTimestamp
    participant E as Embedding
    participant EM as EmbCacheManager
    participant FF as FeatureFilter
    participant TRM as timestampRecordMap_
    participant EFR as EvictFeatureRecord
    
    Train->>KJT: 创建带时间戳的特征数据
    KJT->>E: 提供特征数据和时间戳信息
    E->>EM: _record_timestamp_data(features)
    EM->>EM: 检查features是否有_timestamps属性
    EM->>EM: RecordTimestamp(batchKeys, offsetPerKey, timestamps, tableIndices)
    EM->>FF: RecordTimestamp(featureDataPtr, startIndex, endIndex, timestampsPtr)
    FF->>TRM: 记录特征时间戳
    TRM-->>FF: 记录完成
    FF->>FF: 检查是否需要执行淘汰
    alt 需要执行淘汰
        FF->>FF: FeatureEvict()
        FF->>TRM: 遍历时间戳记录
        loop 遍历所有特征
            TRM->>TRM: 检查时间戳是否超阈值
            alt 时间戳超阈值
                TRM-->>FF: 返回超阈值特征
                FF->>EFR: 记录待淘汰特征
            end
        end
        FF->>TRM: 移除淘汰特征记录
        TRM-->>FF: 移除完成
    end
    
    E->>EM: EmbeddingUpdateAsync
    EM->>EM: 异步执行EmbeddingUpdate
    EM->>EM: RecordEmbeddingUpdateTimes
    EM->>EM: embUpdateCount_递增
    EM->>EM: NeedEvictEmbeddingTable检查
    alt 需要淘汰特征
        EM->>EM: RemoveEmbeddingTableInfo
        EM->>ET: RemoveEmbedding(keys)
        EM->>EM: EvictFeatures
        EM->>SM: RemoveKeys(evictFeatures)
        SM-->>EM: 移除完成
    end
```

## 5. 接口设计

### 5.1 C++接口

#### FeatureFilter类接口

```cpp
class FeatureFilter {
public:
    FeatureFilter(const std::string& tableName, int32_t admitThreshold, uint64_t evictThreshold,
                  uint64_t evictStepInterval);
    
    void StatisticsKeyCount(const int64_t* featureDataPtr, const int64_t* countDataPtr, int64_t startIndex,
                            int64_t endIndex, bool isCountDataEmpty);

    void CountFilter(int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex);

    void RecordTimestamp(const int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex,
                         const int64_t* timestampDataPtr);

    void FeatureEvict();

    const std::unordered_map<int64_t, FeatureRecord>& GetFeatureCountMap();
    const std::unordered_map<int64_t, std::time_t>& GetFeatureTimestampMap();

    void LoadFeatureRecords(const std::vector<int64_t>& keys, std::vector<uint64_t>& counts);
    void LoadTimestampRecords(const std::vector<int64_t>& keys, std::vector<int64_t>& timestamps);
};
```

#### EvictFeatureRecord类接口

```cpp
class EvictFeatureRecord {
public:
    EvictFeatureRecord() = default;
    bool CanRemoveFromEmbTable(uint64_t embUpdateCount) const;
    void ClearEvictInfo();
    void SetSwapCount(uint64_t swapCount);
    std::vector<int64_t>& GetEvictKeys();
};
```

### 5.2 Python接口

在Python层，通过扩展KeyedJaggedTensor，添加了对特征计数和时间戳的支持，以支持基于访问频率的特征过滤和基于时间的特征淘汰功能：

```python
class KeyedJaggedTensorWithCount(KeyedJaggedTensor):
    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        counts: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        length_per_key: Optional[List[int]] = None,
        lengths_offset_per_key: Optional[List[int]] = None,
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[List[int]] = None,
    ) -> None:
```

该类扩展了标准的KeyedJaggedTensor，增加了[counts]属性，用于记录每个特征ID的出现次数。这些计数信息会被传递给FeatureFilter的StatisticsKeyCount方法，用于统计特征访问频率，进而实现基于访问频率的准入控制。

```python
class KeyedJaggedTensorWithTimestamp(KeyedJaggedTensor):
    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        length_per_key: Optional[List[int]] = None,
        lengths_offset_per_key: Optional[List[int]] = None,
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[List[int]] = None,
    ) -> None:
```

该类扩展了标准的KeyedJaggedTensor，增加了[timestamps]属性，用于记录每个特征ID的时间戳。这些时间戳信息会被传递给FeatureFilter的RecordTimestamp方法，用于实现基于时间的特征淘汰机制。

这些扩展类使得Embedding Cache能够同时支持基于访问频率的准入控制和基于时间戳的特征淘汰功能。

## 6. 用例设计

### 6.1 特征准入控制用例

#### 用例名称
特征准入控制

#### 用例描述
系统根据特征的历史访问次数决定是否将特征加载到Embedding Cache中

#### 前置条件
1. FeatureFilter已初始化并配置了准入阈值
2. 存在待加载的特征数据

#### 主要流程
1. 系统接收特征加载请求
2. FeatureFilter统计特征访问次数
3. FeatureFilter检查特征是否满足准入条件
4. 如果满足准入条件，特征被加载到Embedding Cache
5. 如果不满足准入条件，特征被标记为无效

#### 后置条件
特征根据准入策略被正确处理

### 6.2 特征淘汰用例

#### 用例名称
特征淘汰

#### 用例描述
系统根据特征的时间戳信息淘汰长时间未使用的特征

#### 前置条件
1. FeatureFilter已初始化并配置了淘汰阈值
2. 系统中存在已加载的特征

#### 主要流程
1. 系统定期检查特征使用情况
2. FeatureFilter记录特征时间戳
3. FeatureFilter识别需要淘汰的特征
4. 将待淘汰特征记录到EvictFeatureRecord
5. 在适当时机从Embedding Cache中移除特征

#### 后置条件
长时间未使用的特征被正确淘汰，释放内存空间

### 6.3 特征记录加载用例

#### 用例名称
特征记录加载

#### 用例描述
系统支持从持久化存储中加载特征访问记录和时间戳记录

#### 前置条件
1. FeatureFilter已初始化
2. 存在持久化的特征记录数据

#### 主要流程
1. 系统启动或恢复时加载特征记录
2. FeatureFilter加载特征访问次数记录
3. FeatureFilter加载特征时间戳记录
4. 系统基于加载的记录继续特征过滤操作

#### 后置条件
特征记录被成功加载并可用于后续的过滤操作

### 6.4 特征计数统计用例

#### 用例名称
特征计数统计

#### 用例描述
系统统计特征的访问次数，用于准入控制决策

#### 前置条件
1. FeatureFilter已初始化并配置了准入阈值
2. 存在携带计数信息的KeyedJaggedTensorWithCount数据

#### 主要流程
1. 系统接收携带特征计数信息的KeyedJaggedTensorWithCount数据
2. FeatureFilter通过StatisticsKeyCount方法统计特征访问次数
3. 访问次数信息被存储在featureRecordMap中
4. 后续的准入控制将基于这些统计信息进行决策

#### 后置条件
特征访问次数被正确统计并存储，可用于后续的准入控制

### 6.5 时间戳处理用例

#### 用例名称
时间戳处理

#### 用例描述
系统记录特征的时间戳信息，用于淘汰长时间未使用的特征

#### 前置条件
1. FeatureFilter已初始化并配置了淘汰阈值
2. 存在携带时间戳信息的KeyedJaggedTensorWithTimestamp数据

#### 主要流程
1. 系统接收携带时间戳信息的KeyedJaggedTensorWithTimestamp数据
2. FeatureFilter通过RecordTimestamp方法记录特征时间戳
3. 时间戳信息被存储在timestampRecordMap中
4. 在适当的时机，系统根据时间戳判断是否需要淘汰特征

#### 后置条件
特征时间戳被正确记录并存储，可用于后续的淘汰策略

## 7. 测试用例图

```mermaid
graph TD
    A[特征过滤测试] --> B{测试类型}
    B -->|准入控制测试| C[验证特征访问次数统计]
    B -->|淘汰机制测试| D[验证特征时间淘汰功能]
    B -->|时间戳处理测试| E[验证时间戳相关操作]
    
    C --> F[创建带计数数据的KeyedJaggedTensorWithCount]
    C --> G[运行StatisticsKeyCount统计特征访问次数]
    C --> H[手动统计特征访问次数]
    C --> I[读取featureRecordMap_中的统计数据]
    C --> J[对比手动统计与系统统计结果]
    
    D --> K[创建带时间戳数据的KeyedJaggedTensorWithTimestamp]
    D --> L[运行RecordTimestamp记录特征时间戳]
    D --> M[模拟时间流逝超过evictThreshold_]
    D --> N[运行FeatureEvict执行特征淘汰]
    D --> O[检查evictFeatureRecord_中的待淘汰特征]
    D --> P[验证特征是否从timestampRecordMap_中移除]
    
    E --> Q[创建带时间戳的KeyedJaggedTensorWithTimestamp]
    E --> R[执行permute操作]
    E --> S[验证时间戳是否正确跟随特征移动]
    E --> T[执行split操作]
    E --> U[验证分割后的时间戳是否正确]

    style A fill:#FFFF00,stroke:#333
    style F fill:#90EE90,stroke:#333
    style G fill:#90EE90,stroke:#333
    style K fill:#90EE90,stroke:#333
    style Q fill:#90EE90,stroke:#333
    style H fill:#90EE90,stroke:#333
    style I fill:#90EE90,stroke:#333
    style L fill:#90EE90,stroke:#333
    style R fill:#90EE90,stroke:#333
    style T fill:#90EE90,stroke:#333
```

## 8. 测试方案

基于对测试文件[test_feature_filter.py](file://c:\zengxiong\RecSDK\torchrec\torchrec_embcache\tests\acc_test\test_feature_filter.py)和[test_kjt_with_time.py](file://c:\zengxiong\RecSDK\torchrec\torchrec_embcache\tests\acc_test\test_kjt_with_time.py)的分析，重新设计测试方案如下：

### 8.1 测试用例详情表

| 用例编号 | 用例名称 | 测试目标 | 测试步骤 | 预期结果 | 优先级 |
|---------|---------|---------|---------|---------|--------|
| TC001 | 特征访问次数统计测试 | 验证StatisticsKeyCount方法能正确统计特征访问次数 | 1. 创建带计数数据的KeyedJaggedTensorWithCount<br>2. 调用FeatureFilter::StatisticsKeyCount方法<br>3. 手动统计特征访问次数<br>4. 从FeatureFilter::GetFeatureCountMap获取统计数据<br>5. 对比手动统计与系统统计结果 | featureRecordMap_中的特征计数与手动统计结果一致 | 高 |
| TC002 | 特征准入控制测试 | 验证CountFilter方法能正确过滤未达到访问次数阈值的特征 | 1. 设置admitThreshold_为特定值<br>2. 调用StatisticsKeyCount统计特征访问次数<br>3. 调用CountFilter进行准入过滤<br>4. 检查未达到阈值的特征是否被设置为INVALID_KEY | 未达到准入阈值的特征被正确设置为INVALID_KEY | 高 |
| TC003 | 特征时间戳记录测试 | 验证RecordTimestamp方法能正确记录特征时间戳 | 1. 创建带时间戳数据的KeyedJaggedTensorWithTimestamp<br>2. 调用FeatureFilter::RecordTimestamp方法<br>3. 从FeatureFilter::GetFeatureTimestampMap获取时间戳数据 | timestampRecordMap_中的时间戳与输入数据一致 | 高 |
| TC004 | 特征淘汰测试 | 验证FeatureEvict方法能正确识别并记录超时特征 | 1. 调用RecordTimestamp记录特征时间戳<br>2. 模拟时间流逝超过evictThreshold_<br>3. 调用FeatureEvict执行特征淘汰<br>4. 检查evictFeatureRecord_中的待淘汰特征 | 超时特征被正确识别并记录到evictFeatureRecord_中 | 高 |
| TC005 | 时间戳permute操作测试 | 验证KeyedJaggedTensorWithTimestamp的permute操作能正确处理时间戳 | 1. 创建带时间戳的KeyedJaggedTensorWithTimestamp<br>2. 执行permute操作<br>3. 验证时间戳是否正确跟随特征移动 | permute操作后时间戳与特征保持对应关系 | 中 |
| TC006 | 时间戳split操作测试 | 验证KeyedJaggedTensorWithTimestamp的split操作能正确处理时间戳 | 1. 创建带时间戳的KeyedJaggedTensorWithTimestamp<br>2. 执行split操作<br>3. 验证分割后的时间戳是否正确 | split操作后各部分时间戳与特征保持对应关系 | 中 |
| TC007 | 特征记录加载测试 | 验证LoadFeatureRecords和LoadTimestampRecords方法能正确加载特征记录 | 1. 准备特征记录数据<br>2. 调用LoadFeatureRecords加载访问次数记录<br>3. 调用LoadTimestampRecords加载时间戳记录<br>4. 验证featureRecordMap_和timestampRecordMap_中的数据 | 特征记录被正确加载到对应的数据结构中 | 中 |
| TC008 | 准入淘汰组合测试 | 验证同时启用准入控制和淘汰机制时功能正确性 | 1. 同时设置admitThreshold_和evictThreshold_<br>2. 运行完整的特征处理流程<br>3. 验证准入和淘汰机制都正常工作 | 准入和淘汰机制同时正常工作，互不干扰 | 高 |

### 8.2 补充测试用例

基于对代码的全面分析，建议增加以下测试用例以提高测试覆盖率：

| 用例编号 | 用例名称 | 测试目标 | 测试步骤 | 预期结果 | 优先级 |
|---------|---------|---------|---------|---------|--------|
| TC009 | 未启用准入控制测试 | 验证未设置准入阈值时所有特征都能通过准入检查 | 1. 不设置admitThreshold_(-1为默认值)<br>2. 执行CountFilter操作<br>3. 检查特征是否都被保留 | 所有特征都被保留，不进行准入过滤 | 中 |
| TC010 | 未启用淘汰机制测试 | 验证未设置淘汰阈值时不会执行特征淘汰 | 1. 设置evictThreshold_为0<br>2. 执行FeatureEvict操作<br>3. 检查是否有特征被标记为待淘汰 | 不执行特征淘汰操作 | 中 |
| TC011 | 空数据处理测试 | 验证各方法能正确处理空数据输入 | 1. 传入空数据给各个方法<br>2. 检查是否能正常处理不发生崩溃 | 各方法能正确处理空数据输入 | 中 |
| TC012 | 边界值测试 | 验证特征ID为-1等边界值的处理 | 1. 输入特征ID为-1的数据<br>2. 执行各个处理方法<br>3. 检查处理结果 | 特征ID为-1的数据被正确处理（通常被忽略） | 中 |
| TC013 | 大数据量测试 | 验证系统在处理大量特征数据时的性能和正确性 | 1. 构造大量特征数据<br>2. 执行完整的特征处理流程<br>3. 检查处理结果和性能表现 | 系统能正确处理大量数据且性能在可接受范围内 | 低 |

## 9. 性能优化

1. 使用std::unordered_map存储特征信息，提高查询效率
2. 通过批量处理特征统计和过滤操作，减少系统调用次数
3. 异步淘汰机制，避免阻塞主流程
4. 只在必要时执行淘汰检查，减少计算开销

## 10. 安全性考虑

1. 对特征记录数据进行定期清理，防止内存泄漏
2. 添加边界检查，防止数组越界访问
3. 使用安全的字符串处理函数
4. 对输入参数进行有效性验证

## 11. 测试方案

1. 单元测试：对FeatureFilter和EvictFeatureRecord类进行独立测试
2. 集成测试：测试特征准入和淘汰功能在完整流程中的表现
3. 压力测试：测试在高并发场景下的性能表现
4. 边界测试：测试各种边界条件下的正确性

## 12. 部署方案

1. 通过配置文件设置准入阈值和淘汰阈值
2. 提供监控指标，用于观察特征过滤效果
3. 支持动态调整阈值参数，无需重启服务

这个设计方案实现了基于访问频率和时间戳的特征准入和淘汰机制，能够有效提高Embedding Cache的利用率和推荐系统的整体性能.