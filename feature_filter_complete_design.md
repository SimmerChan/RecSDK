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

## 3. 设计方案

### 3.1 核心组件

#### 3.1.1 EvictFeatureRecord（特征淘汰记录）

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

#### 3.1.2 FeatureFilter（特征过滤器）

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

### 3.2 类图

```mermaid
classDiagram
    class EvictFeatureRecord {
        +EvictFeatureRecord()
        +CanRemoveFromEmbTable(uint64_t embUpdateCount) bool
        +ClearEvictInfo()
        +SetSwapCount(uint64_t swapCount)
        +GetEvictKeys() std::vector<int64_t>&
        -executeSwapCount uint64_t
        -evictKeys std::vector<int64_t>
    }
    
    class FeatureFilter {
        +FeatureFilter(const std::string& tableName, int32_t admitThreshold, uint64_t evictThreshold, uint64_t evictStepInterval)
        +StatisticsKeyCount(const int64_t* featureDataPtr, const int64_t* countDataPtr, int64_t startIndex, int64_t endIndex, bool isCountDataEmpty)
        +CountFilter(int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex)
        +RecordTimestamp(const int64_t* featureDataPtr, int64_t startIndex, int64_t endIndex, const int64_t* timestampDataPtr)
        +FeatureEvict()
        +GetFeatureCountMap() const std::unordered_map<int64_t, FeatureRecord>&
        +GetFeatureTimestampMap() const std::unordered_map<int64_t, std::time_t>&
        +LoadFeatureRecords(const std::vector<int64_t>& keys, std::vector<uint64_t>& counts)
        +LoadTimestampRecords(const std::vector<int64_t>& keys, std::vector<int64_t>& timestamps)
        -tableName std::string
        -admitThreshold_ int32_t
        -featureRecordMap std::unordered_map<int64_t, FeatureRecord>
        -evictThreshold uint64_t
        -evictStepInterval uint64_t
        -recordTsBatchId uint64_t
        -latestTimestamp std::time_t
        -timestampRecordMap std::unordered_map<int64_t, std::time_t>
        +evictFeatureRecord EvictFeatureRecord
    }
    
    class FeatureRecord {
        +count uint64_t
    }
    
    class JaggedTensorWithCount {
        +JaggedTensorWithCount(values, weights, lengths, offsets, counts)
        +counts torch.Tensor
    }
    
    class KeyedJaggedTensorWithCount {
        +KeyedJaggedTensorWithCount(keys, values, counts, weights, lengths, offsets, stride, stride_per_key_per_rank, length_per_key, lengths_offset_per_key, offset_per_key, index_per_key, jt_dict, inverse_indices)
        +counts torch.Tensor
        +from_jt_dict(jt_dict) KeyedJaggedTensorWithCount
        +split(segments) List[KeyedJaggedTensorWithCount]
    }
    
    class JaggedTensorWithTimestamp {
        +JaggedTensorWithTimestamp(values, weights, lengths, offsets, timestamps)
        +timestamps torch.Tensor
    }
    
    class KeyedJaggedTensorWithTimestamp {
        +KeyedJaggedTensorWithTimestamp(keys, values, timestamps, weights, lengths, offsets, stride, stride_per_key_per_rank, length_per_key, lengths_offset_per_key, offset_per_key, index_per_key, jt_dict, inverse_indices)
        +timestamps torch.Tensor
        +from_jt_dict(jt_dict) KeyedJaggedTensorWithTimestamp
        +split(segments) List[KeyedJaggedTensorWithTimestamp]
    }
    
    FeatureFilter --> EvictFeatureRecord : uses
    FeatureFilter --> FeatureRecord : uses
    KeyedJaggedTensorWithCount --> JaggedTensorWithCount : contains
    KeyedJaggedTensorWithCount --> KeyedJaggedTensor : extends
    JaggedTensorWithCount --> JaggedTensor : extends
    KeyedJaggedTensorWithTimestamp --> JaggedTensorWithTimestamp : contains
    KeyedJaggedTensorWithTimestamp --> KeyedJaggedTensor : extends
    JaggedTensorWithTimestamp --> JaggedTensor : extends
```

### 3.3 特征准入控制

#### 3.3.1 特征准入流程

```mermaid
graph TD
    A[新特征请求加载] --> B{是否启用准入控制?}
    B -->|否| C[直接加载特征]
    B -->|是| D[调用CountFilter]
    D --> E[查询特征历史访问次数]
    E --> F{访问次数 >= 准入阈值?}
    F -->|是| G[允许特征准入]
    G --> H[加载特征到Embedding Cache]
    F -->|否| I[拒绝特征准入]
    I --> J[将特征标记为无效]
    C --> K[结束]
    H --> K
    J --> K
```

#### 3.3.2 特征准入时序图

```mermaid
sequenceDiagram
    participant EC as EmbeddingCache
    participant FF as FeatureFilter
    participant FRM as featureRecordMap
    
    EC->>FF: StatisticsKeyCount(featureData, countData, start, end, isEmpty)
    FF->>FRM: 更新特征访问次数
    FRM-->>FF: 更新完成
    EC->>FF: CountFilter(featureData, start, end)
    FF->>FRM: 查询特征访问次数
    FRM-->>FF: 返回特征访问次数
    FF->>FF: 比较次数与准入阈值
    alt 访问次数 < 准入阈值
        FF->>EC: 将特征标记为无效(-1)
    else 访问次数 >= 准入阈值
        FF->>EC: 保留特征值
    end
```

#### 3.3.3 特征计数统计时序图

```mermaid
sequenceDiagram
    participant KJT as KeyedJaggedTensorWithCount
    participant PI as PostInputDist
    participant HM as HashMap
    participant FF as FeatureFilter
    participant FRM as featureRecordMap
    
    KJT->>PI: 提供特征数据和计数信息
    PI->>HM: do_unique_hash_out(origin_kjt, ...)
    HM->>HM: 检查origin_kjt是否有counts属性
    HM->>FF: statistic_key_count(ids, offsets, counts, table_i)
    FF->>FRM: StatisticsKeyCount处理计数信息
    FRM-->>FF: 更新特征访问次数
    FF->>FF: 存储计数信息用于后续准入控制
```

### 3.4 特征淘汰机制

#### 3.4.1 特征淘汰流程

```mermaid
graph TD
    A[触发淘汰步骤] --> B{是否启用淘汰机制?}
    B -->|否| C[跳过淘汰]
    B -->|是| D[调用FeatureEvict]
    D --> E[检查特征时间戳]
    E --> F{当前时间-特征时间 > 淘汰阈值?}
    F -->|是| G[标记为待淘汰特征]
    G --> H[记录到EvictFeatureRecord]
    F -->|否| I[保留在缓存中]
    H --> J[从记录中移除淘汰特征]
    J --> K[结束]
    I --> K
    C --> K
```

#### 3.4.2 特征淘汰时序图

```mermaid
sequenceDiagram
    participant EC as EmbeddingCache
    participant FF as FeatureFilter
    participant TRM as timestampRecordMap
    participant EFR as EvictFeatureRecord
    
    EC->>FF: RecordTimestamp(featureData, start, end, timestampData)
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
        FF->>EFR: 获取待淘汰特征列表
        EFR-->>FF: 返回待淘汰特征列表
        FF->>TRM: 移除淘汰特征记录
        TRM-->>FF: 移除完成
        FF->>EC: 返回淘汰特征列表
    end
```

#### 3.4.3 时间戳处理时序图

```mermaid
sequenceDiagram
    participant KJT as KeyedJaggedTensorWithTimestamp
    participant E as Embedding
    participant EM as EmbcacheManager
    participant FF as FeatureFilter
    participant TRM as timestampRecordMap
    participant EFR as EvictFeatureRecord
    
    KJT->>E: 提供特征数据和时间戳信息
    E->>EM: _record_timestamp_data(features)
    EM->>EM: 检查features是否有_timestamps属性
    EM->>FF: RecordTimestamp(keyPtr, startIndex, endIndex, timestampsPtr)
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
```

## 4. 接口设计

### 4.1 C++接口

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

### 4.2 Python接口

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

## 5. 用例设计

### 5.1 特征准入控制用例

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

### 5.2 特征淘汰用例

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

### 5.3 特征记录加载用例

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

### 5.4 特征计数统计用例

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

### 5.5 时间戳处理用例

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

## 6. 性能优化

1. 使用std::unordered_map存储特征信息，提高查询效率
2. 通过批量处理特征统计和过滤操作，减少系统调用次数
3. 异步淘汰机制，避免阻塞主流程
4. 只在必要时执行淘汰检查，减少计算开销

## 7. 安全性考虑

1. 对特征记录数据进行定期清理，防止内存泄漏
2. 添加边界检查，防止数组越界访问
3. 使用安全的字符串处理函数
4. 对输入参数进行有效性验证

## 8. 测试方案

1. 单元测试：对FeatureFilter和EvictFeatureRecord类进行独立测试
2. 集成测试：测试特征准入和淘汰功能在完整流程中的表现
3. 压力测试：测试在高并发场景下的性能表现
4. 边界测试：测试各种边界条件下的正确性

## 9. 部署方案

1. 通过配置文件设置准入阈值和淘汰阈值
2. 提供监控指标，用于观察特征过滤效果
3. 支持动态调整阈值参数，无需重启服务

这个设计方案实现了基于访问频率和时间戳的特征准入和淘汰机制，能够有效提高Embedding Cache的利用率和推荐系统的整体性能。