# dense_to_jagged算子详细设计方案

## 1. 算子概述

dense_to_jagged算子是FBGEMM库中用于将padded dense张量转换为jagged张量的关键算子。该算子在推荐系统中广泛应用于处理变长序列数据，将填充后的固定维度张量转换为紧凑的不规则张量格式。

### 1.1 基本概念
- **Dense张量**：经过填充的固定维度张量，形状为[B, MaxT, D]，其中B是批次大小，MaxT是最大序列长度，D是特征维度
- **Offset张量**：记录每个序列实际长度的一维张量，长度为B+1，从0开始递增
- **Jagged张量**：紧凑存储的不规则张量，形状为[Total_L, D]，其中Total_L是所有序列实际长度之和

### 1.2 转换逻辑示意图

```mermaid
graph TD
    A[Dense Tensor] --> C[dense_to_jagged算子]
    B[Offset Tensor] --> C
    C --> D[Jagged Tensor]
```

### 1.3 算子核心算法

根据README中的伪代码，算子的核心逻辑如下：

```python
def dense_to_jagged(dense, offset, jagged_dim0):
    jagged_dense = torch.zeros(jagged_dim0, dense.shape[2], dtype=dense.dtype, device=dense.device)
    
    for i in range(offset.shape[0] - 1):
        copyLen = offset[i + 1] - offset[i]
        jagged_dense[offset[i]:offset[i + 1], :] = dense[i, 0:copyLen, :]
    
    return jagged_dense
```

## 2. Ascend C算子实现架构

### 2.1 整体架构图

```mermaid
graph TD
    A[Host端<br/>op_host] --> B[Kernel端<br/>op_kernel]
    A --> C[Tiling配置]
    B --> D[AI Core执行]
    C --> D
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e8
```

### 2.2 Host端实现（op_host）

Host端主要负责：
1. **形状推导**：根据输入张量形状推导输出张量形状
2. **数据类型推导**：确定输出张量的数据类型
3. **Tiling配置**：根据硬件资源和数据规模生成分片配置

核心代码分析：
```cpp
// 输入约束检查
OPS_CHECK(denseShape.GetDim(DIM0) != offsetShape.GetDim(DIM0) - 1,
    OPS_LOG_E("[ERROR]", "dense shape[0] != offset shape[0] - 1"), return ge::GRAPH_FAILED);

// 分片配置计算
int singleCoreBatch = (offsetShape.GetDim(DIM0) - 1) / coreNum;
int left = (offsetShape.GetDim(DIM0) - 1) % coreNum;
int singleLoopSize = (ubSize - RESERVER_UB_SIZE) / 2 / ALIGN_512 * ALIGN_512;
```

## 3. Kernel端实现详解（op_kernel）

Kernel端使用Ascend C编程模型实现具体的计算逻辑，这是整个算子的核心部分。

### 3.1 数据结构定义

```cpp
struct DenseToJaggedArgs {
    GM_ADDR dense;          // 输入dense张量的全局内存地址
    GM_ADDR offset;         // 输入offset张量的全局内存地址
    GM_ADDR jagged_dense;   // 输出jagged_dense张量的全局内存地址

    int32_t denseDim1;      // dense张量的第1维大小(MaxT)
    int32_t denseDim2;      // dense张量的第2维大小(D)
    int32_t left;           // 分片处理中余下的batch数
    int32_t singleCoreBatch;// 每个AI Core处理的batch数
    int32_t singleLoopSize; // 单次循环处理的数据大小
    int64_t denseTotal;     // dense张量总元素数
    int64_t jaggedTotal;    // jagged_dense张量总元素数
};
```

### 3.2 核心类实现

```cpp
template<typename dType, typename tType>
class DenseToJagged {
public:
    __aicore__ inline DenseToJagged() {};

    __aicore__ inline void init(DenseToJaggedArgs *args, TPipe *pipe)
    {
        this->args = args;
        this->pipe = pipe;

        thisId = GetBlockIdx();
        if (thisId < args->left) {
            args->singleCoreBatch += 1;
            offsetStartPos = thisId * args->singleCoreBatch;
        } else {
            offsetStartPos = (args->singleCoreBatch + 1) * args->left + 
                             (thisId - args->left) * args->singleCoreBatch;
        }

        align = sizeof(dType);
        denseGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->dense), 
                                args->denseTotal * align);
        jaggedDenseGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->jagged_dense), 
                                      args->jaggedTotal * align);

        pipe->InitBuffer(inQueue, 1, args->singleLoopSize);
        pipe->InitBuffer(outQueue, 1, args->singleLoopSize);
    }

    __aicore__ inline void Compute()
    {
        ComputeEachBatch();
    }
};
```

### 3.3 核心计算逻辑详解

Kernel端的核心计算过程可以分为以下几个步骤：

#### 3.3.1 多核并行处理策略
```mermaid
graph TD
    A[输入数据] --> B[分片到多个AI Core]
    B --> C[AI Core 0处理Batch 0-N0]
    B --> D[AI Core 1处理Batch N0-N1]
    B --> E[AI Core 2处理Batch N1-N2]
    B --> F[...]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e8
    style E fill:#fce4ec
    style F fill:#fff8e1
```
#### 3.3.2 单个Batch处理流程

对于每个AI Core处理的每个Batch，核心计算流程如下：

```mermaid
graph TD
    A[读取offset信息] --> B[计算copyRows]
    B --> C[计算数据拷贝地址]
    C --> D[循环处理数据块]
    D --> E[数据拷贝到LocalTensor]
    E --> F[数据处理]
    F --> G[数据写回GlobalTensor]
    G --> H{还有剩余数据?}
    H -->|是| D
    H -->|否| I[处理下一个Batch]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e8
    style E fill:#fce4ec
    style F fill:#fff8e1
    style G fill:#f1f8e9
    style H fill:#bbdefb
    style I fill:#c8e6c9
```

#### 3.3.3 详细计算过程

1. **获取offset信息**：
   ```cpp
   jaggedPos = *(oPtr + offsetStartPos + i);
   jaggedPosNext = *(oPtr + offsetStartPos + i + 1);
   int copyRows = jaggedPosNext - jaggedPos;
   ```

2. **计算数据拷贝地址**：
   ```cpp
   GlobalTensor<uint8_t> jaggedDenseCopyGb = jaggedDenseGb[jaggedPos * args->denseDim2 * align];
   GlobalTensor<uint8_t> denseCopyGb = denseGb[(offsetStartPos + i) * args->denseDim2 * args->denseDim1 * align];
   ```

3. **循环处理数据块**：
   - 计算本次需要处理的数据长度
   - 分配LocalTensor缓冲区
   - 进行数据对齐处理
   - 执行数据拷贝操作
   - 处理未对齐的数据尾部

4. **数据拷贝与处理**：
   ```cpp
   // 拷贝超对齐大小数据以避免处理未对齐尾部
   DataCopy(localIn, denseCopyGb, overAlignLen);
   
   // 数据在输入输出队列间传递
   DataCopy(localOut, localInCopy, overAlignLen);
   
   // 拷贝对齐部分数据
   DataCopy(jaggedDenseCopyGb, localOutCopy, alignLen);
   
   // 处理未对齐尾部数据
   if (unAlignLen != 0) {
       const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
       DataCopyPad(jaggedDenseCopyGb[alignLen], localOutCopy[alignLen], dataCopyExtParams);
   }
   ```

### 3.4 内存管理与优化

Kernel端使用了以下内存优化策略：

1. **LocalTensor缓冲区**：
   - 使用TQue队列管理LocalTensor
   - inQueue用于输入数据缓冲
   - outQueue用于输出数据缓冲

2. **数据对齐处理**：
   - 使用ALIGN_32进行32字节对齐
   - 分别处理对齐部分和未对齐部分数据

3. **流水线处理**：
   - 使用Ascend C的TPipe进行流水线操作
   - 提高数据处理效率

### 3.5 类型支持

Kernel支持多种数据类型组合：

```cpp
// 修改前的实现方式（使用多个if-else分支）
if (tiling_data.denseType == floatType && tiling_data.offsetType == int32Type) {
    DenseToJagged_Kernel::DenseToJagged<float, int32_t> kernel;
    kernel.init(&args, &pipe);
    kernel.Compute();
} else if (tiling_data.denseType == floatType && tiling_data.offsetType == int64Type) {
    DenseToJagged_Kernel::DenseToJagged<float, int64_t> kernel;
    kernel.init(&args, &pipe);
    kernel.Compute();
} else if (tiling_data.denseType == int64Type && tiling_data.offsetType == int64Type) {
    DenseToJagged_Kernel::DenseToJagged<int64_t, int64_t> kernel;
    kernel.init(&args, &pipe);
    kernel.Compute();
} else if (tiling_data.denseType == int64Type && tiling_data.offsetType == int32Type) {
    DenseToJagged_Kernel::DenseToJagged<int64_t, int32_t> kernel;
    kernel.init(&args, &pipe);
    kernel.Compute();
}

// 修改后的实现方式（使用宏定义方式）
extern "C" __global__ __aicore__ void dense_to_jagged(GM_ADDR dense, GM_ADDR offset, GM_ADDR jagged_dense,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    DenseToJagged_Kernel::DenseToJaggedArgs args {
        dense, offset, jagged_dense, tiling_data.denseDim1, tiling_data.denseDim2, tiling_data.left,
        tiling_data.singleCoreBatch, tiling_data.singleLoopSize, tiling_data.denseTotal, tiling_data.jaggedTotal
    };

    TPipe pipe;
    DenseToJagged_Kernel::DenseToJagged<DTYPE_DENSE, DTYPE_OFFSET> kernel;
    kernel.init(&args, &pipe);
    kernel.Compute();
}
```

## 4. PTA适配层实现

PTA（PyTorch Adapter）层负责将PyTorch的调用转换为Ascend算子调用：

### 4.1 接口定义
```cpp
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("dense_to_jagged_forward(Tensor dense, "
          "                        Tensor[] offsets, "
          "                        SymInt? total_L=None) -> Tensor");

    m.def("dense_to_jagged(Tensor dense, "
          "                Tensor[] offsets, "
          "                SymInt? total_L=None) -> (Tensor, Tensor[])");
}
```

### 4.2 核心实现逻辑
```cpp
// 修改前的函数签名
at::Tensor dense_to_jagged_forward_npu(const at::Tensor& dense,
                                       const tensor_list& offsets,
                                       const c10::optional<int64_t> total_L)
{
    // 参数检查
    TORCH_CHECK(dense.dim() == 3, "dense must be 3-dimensional");
    TORCH_CHECK(offsets.size() == 1, "Only single-dimension jagged tensors supported");
    
    // 计算输出大小
    int64_t expected_total_L = offsets.back()[-1].item<int64_t>();
    int64_t totalLength = total_L.value_or(expected_total_L);
    
    // 支持BF16和FP16类型，需要先转换为FP32进行处理，然后转回原类型
    if (dense.dtype() == at::kBFloat16 || dense.dtype() == at::kHalf) {
        auto dense_float = dense_contin.to(at::kFloat);
        EXEC_NPU_CMD(aclnnDenseToJagged, dense_float, offsets[0], totalLength, output);
        return output.to(dense.dtype());
    } else {
        EXEC_NPU_CMD(aclnnDenseToJagged, dense_contin, offsets[0], totalLength, output);
        return output;
    }
};

// 修改后的函数签名
std::tuple<at::Tensor, tensor_list> dense_to_jagged_forward_npu(const at::Tensor& dense,
                                       const tensor_list& offsets,
                                       const c10::optional<int64_t> total_L)
{
    // 参数检查
    TORCH_CHECK(dense.dim() == 3, "dense must be 3-dimensional");
    TORCH_CHECK(offsets.size() == 1, "Only single-dimension jagged tensors supported");
    
    // 计算输出大小
    int64_t expected_total_L = offsets.back()[-1].item<int64_t>();
    int64_t totalLength = total_L.value_or(expected_total_L);
    
    // 支持BF16和FP16类型，需要先转换为FP32进行处理，然后转回原类型
    if (dense.dtype() == at::kBFloat16 || dense.dtype() == at::kHalf) {
        auto dense_float = dense_contin.to(at::kFloat);
        EXEC_NPU_CMD(aclnnDenseToJagged, dense_float, offsets[0], totalLength, output);
        return {output.to(dense.dtype()), offsets};
    } else {
        EXEC_NPU_CMD(aclnnDenseToJagged, dense_contin, offsets[0], totalLength, output);
        return {output, offsets};
    }
};
```

### 4.3 函数返回值修改

```cpp
// 修改前的dense_to_jagged_npu函数实现
std::tuple<at::Tensor, tensor_list> dense_to_jagged_npu(const at::Tensor& dense,
                                                        const tensor_list& offsets,
                                                        const c10::optional<int64_t> total_L)
{
    return {dense_to_jagged_forward_npu(dense, offsets, total_L), offsets};
};

// 修改后的dense_to_jagged_npu函数实现
std::tuple<at::Tensor, tensor_list> dense_to_jagged_npu(const at::Tensor& dense,
                                                        const tensor_list& offsets,
                                                        const c10::optional<int64_t> total_L)
{
    return dense_to_jagged_forward_npu(dense, offsets, total_L);
};
```

### 4.4 与开源算子对齐说明

为了与开源FBGEMM算子保持接口一致性，我们对PTA层的函数返回值进行了重要修改：

1. **返回类型变更**：
   - 原始实现：`at::Tensor dense_to_jagged_forward_npu(...)`
   - 修改后实现：`std::tuple<at::Tensor, tensor_list> dense_to_jagged_forward_npu(...)`
   
   这一修改使我们的实现与开源FBGEMM的接口保持一致，确保了在PyTorch生态中的兼容性。

2. **返回值内容**：
   - 修改后的函数不仅返回转换后的jagged张量，还同时返回offsets张量列表
   - 这种设计与FBGEMM开源实现保持一致，便于上层代码处理

3. **接口一致性优势**：
   - 保持与FBGEMM开源算子的接口一致性
   - 简化上层代码的调用逻辑
   - 提高代码的可移植性和兼容性

## 5. 测试用例分析

测试用例位于`mxrec_add_ons/rec_for_torch/torch_plugin/torch_demo/dense_to_jagged/test_dense_to_jagged.py`，主要覆盖以下方面：

### 5.1 功能测试
- 不同数据类型组合测试（float32/int64/bfloat16/float16/int32作为dense，int32/int64作为offset）
- 不同维度参数测试
- output_size参数的有无测试

### 5.2 精度测试
- 与CPU参考实现结果对比，确保精度误差在容差范围内

#### 5.2.1 精度验证标准

根据项目规范，不同数据类型的精度容差标准如下：

| 数据类型 | 容差值 | 说明 |
|---------|--------|------|
| float16 | 1e-3 | 双千分之一 |
| bfloat16 | 5e-3 | 双千分之五 |
| int32 | 1e-4 | 双万分之一 |
| float32 | 1e-4 | 双万分之一 |
| int64 | 1e-4 | 双万分之一 |

#### 5.2.2 精度验证实现

测试代码中通过`get_tolerance`函数根据数据类型动态获取相应的容差值：

```python
def get_tolerance(dense_dtype):
    """根据数据类型获取相应的容差值"""
    if dense_dtype == torch.float16:
        return 1e-3  # float16: 双千分之一
    elif dense_dtype == torch.bfloat16:
        return 5e-3  # bfloat16: 双千分之五
    elif dense_dtype in [torch.int32, torch.float32, torch.int64]:
        return 1e-4  # int32、float32和int64: 双万分之一
    else:
        raise ValueError(f"Unsupported data type: {dense_dtype}")
```

在测试过程中，通过`compare_results`函数进行精度验证：

```python
def compare_results(golden_result, npu_result, tolerance=1e-4):
    """比较CPU和NPU的结果"""
    # 检查两个结果的形状是否相同
    assert golden_result.shape == npu_result.shape, \
        f"Shape mismatch: golden {golden_result.shape} vs npu {npu_result.shape}"

    # 对所有张量进行数值比较（包括空张量）
    if golden_result.numel() > 0:
        result_forward = torch.abs(golden_result - npu_result) < tolerance
        assert result_forward.all().item(), "Result values do not match within tolerance"
    else:
        # 空张量直接通过检查（形状已验证）
        assert torch.equal(golden_result, npu_result), "Empty tensors should be equal"
```

### 5.3 自动求导测试
- 前向传播结果验证
- 反向传播梯度验证

测试用例核心代码：
```python
@pytest.mark.parametrize("dims", DIM_LIST)
@pytest.mark.parametrize("types", TYPE_LIST)
@pytest.mark.parametrize("use_output_size", [True, False])
def test_dense_to_jagged(dims, types, use_output_size):
    dense_dim0, dense_dim1, dense_dim2 = dims
    # 1. 生成随机输入数据
    dense_datatype, _ = types
    denses, offsets = generate_test_data(dense_dim0, dense_dim1, dense_dim2, dense_datatype)

    run_test(denses, offsets, types, use_output_size)
```

## 6. 与CUDA实现对比

| 特性 | Ascend实现 | CUDA实现 | 差异分析 |
|------|------------|----------|----------|
| 编程模型 | Ascend C | CUDA C | 语法和API不同，但核心算法一致 |
| 内存管理 | GM/L1/L0三级缓存 | Global/Shared/Register | 架构差异导致内存层级不同 |
| 并行策略 | 多AI Core并行 | 多线程块并行 | 并行粒度和调度机制不同 |
| 数据对齐 | 32字节对齐 | 通常16字节对齐 | 对齐要求不同 |
| 精度 | fp32/fp16/bf16/int64/int32 | fp32/fp16/bf16/int64/int32 | 精度支持基本一致 |

## 7. 本次修改点总结

### 7.1 主要修改内容

1. **JSON配置文件修改**：
   - 增加了对int32、bf16、fp16数据类型的支持
   - 简化了format字段定义方式，使用format_list替代原来的重复定义

2. **Host端代码修改**：
   - 使用DataTypeList替代DataType方式定义支持的数据类型
   - 增加了对int32、bf16和fp16类型的支持
   - 输出张量使用Follow方式继承输入张量的数据类型，简化了配置

3. **Kernel端代码修改**：
   - 采用宏定义方式(DTYPE_DENSE, DTYPE_OFFSET)替代复杂的if-else分支判断
   - 移除了手动类型判断代码，使用AscendC的模板机制自动推导类型
   - 代码结构更加简洁，提高了可维护性

4. **PTA适配层修改**：
   - 修改了函数返回值类型，从单一Tensor改为tuple形式，与fbgemm接口保持一致
   - 修改了dense_to_jagged_npu函数的实现，避免重复添加offsets到返回值中
   - 增加了与开源算子对齐的说明，确保接口一致性

5. **测试用例修改**：
   - 增加了对新增数据类型的支持测试
   - 增加了边界情况和特殊场景测试
   - 完善了测试框架，提高了代码复用性
   - 增加了精度验证相关内容，明确不同数据类型的容差标准

### 7.2 优化点

1. **代码结构优化**：
   - Kernel端使用宏定义方式替代大量if-else分支，代码更简洁
   - Host端使用DataTypeList方式定义类型，更符合规范

2. **测试完善**：
   - 增加了全面的数据类型测试
   - 增加了边界和特殊场景测试
   - 重构了测试框架，提高了代码复用性和可维护性
   - 明确了精度验证标准，确保不同数据类型的测试准确性

3. **文档更新**：
   - 更新了README文档中的支持数据类型说明
   - 提供了完整的设计文档
   - 增加了精度验证相关内容