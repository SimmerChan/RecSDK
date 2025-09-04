# dense_to_jagged算子BF16/FP16支持详细设计

## 1. 设计目标

为dense_to_jagged算子增加BF16和FP16数据类型支持，以提升计算效率和降低内存占用。

## 2. 设计方案

### 2.1 JSON配置文件修改

修改dense_to_jagged.json文件，增加BF16和FP16支持，扩展类型组合以支持FP32+INT64->FP16/BF16等转换：

```json
{
    "op": "DenseToJagged",
    "language": "cpp",
    "input_desc": [
        {
            "name": "dense",
            "param_type": "required",
            "format": [
                "ND", "ND", "ND", "ND", "ND", "ND",
                "ND", "ND", "ND", "ND", "ND", "ND"
            ],
            "type": [
                "fp32", "fp32", "int64", "int64", "bf16", "fp16",
                "fp32", "fp32", "fp16", "bf16", "fp16", "bf16"
            ]
        },
        {
            "name": "offset",
            "param_type": "required",
            "format": [
                "ND", "ND", "ND", "ND", "ND", "ND",
                "ND", "ND", "ND", "ND", "ND", "ND"
            ],
            "type": [
                "int64", "int32", "int32", "int64", "int64", "int64",
                "int64", "int64", "int64", "int64", "int32", "int32"
            ]
        }
    ],
    "output_desc": [
        {
            "name": "jagged_dense",
            "param_type": "required",
            "format": [
                "ND", "ND", "ND", "ND", "ND", "ND",
                "ND", "ND", "ND", "ND", "ND", "ND"
            ],
            "type": [
                "fp32", "fp32", "int64", "int64", "bf16", "fp16",
                "fp16", "bf16", "fp16", "bf16", "fp32", "fp32"
            ]
        }
    ],
    "attr": [
        {
            "name": "jagged_dim",
            "param_type": "optional",
            "type": "int",
            "default_value": 0
        }
    ]
}
```

### 2.2 Host端修改

修改[op_host/dense_to_jagged.cpp](file:///c%3A/zengxiong/RecSDK/mxrec_add_ons/rec_for_torch/operators/dense_to_jagged/op_host/dense_to_jagged.cpp)文件，增加BF16和FP16类型支持：

```cpp
// 增加新的类型常量定义
constexpr int32_t TYPE_BF16 = 15;
constexpr int32_t TYPE_FP16 = 14;

// 修改OpDef注册，扩展类型支持
this->Input("dense")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, ge::DT_BF16, ge::DT_FLOAT16,
              ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
    .FormatList({ge::FORMAT_ND});
this->Input("offset")
    .ParamType(REQUIRED)
    .DataType({ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
              ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32})
    .FormatList({ge::FORMAT_ND});
this->Output("jagged_dense")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, ge::DT_BF16, ge::DT_FLOAT16,
              ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
    .FormatList({ge::FORMAT_ND});
```

### 2.3 Kernel端修改

修改[op_kernel/dense_to_jagged.cpp](file:///c%3A/zengxiong/RecSDK/mxrec_add_ons/rec_for_torch/operators/dense_to_jagged/op_kernel/dense_to_jagged.cpp)文件，扩展模板参数支持：

```cpp
// 增加新的类型常量
constexpr int32_t TYPE_BF16 = 15;
constexpr int32_t TYPE_FP16 = 14;

// 修改类型定义部分
namespace DenseToJagged_Kernel {
constexpr int32_t ALIGN_32 = 32;
constexpr int32_t ALIGN_16 = 16;
constexpr int32_t TYPE_FLOAT = 0;
constexpr int32_t TYPE_INT32 = 3;
constexpr int32_t TYPE_INT64 = 9;
constexpr int32_t TYPE_BF16 = 15;
constexpr int32_t TYPE_FP16 = 14;

template <typename dType, typename oType>
class DenseToJagged {
public:
    struct DenseToJaggedArgs {
        GM_ADDR dense;
        GM_ADDR offset;
        GM_ADDR jagged_dense;
        int32_t denseDim1;
        int32_t denseDim2;
        int32_t left;
        int32_t singleCoreBatch;
        int32_t singleLoopSize;
        int64_t denseTotal;
        int64_t jaggedTotal;
    };

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
            offsetStartPos = (args->singleCoreBatch + 1) * args->left + (thisId - args->left) * args->singleCoreBatch;
        }

        align = sizeof(dType);
        denseGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->dense), args->denseTotal * align);
        jaggedDenseGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->jagged_dense), args->jaggedTotal * align);

        pipe->InitBuffer(inQueue, 1, args->singleLoopSize);
        pipe->InitBuffer(outQueue, 1, args->singleLoopSize);
    }

    __aicore__ inline void Compute()
    {
        ComputeEachBatch();
    }

private:
    __aicore__ inline void DataCopyPadLocal2Gm(const GlobalTensor<uint8_t>& gt, const LocalTensor<uint8_t>& lt,
                                               uint32_t unAlignLen)
    {
        GlobalTensor<uint16_t> uint16Gt;
        uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), unAlignLen/2);
        LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

        uint32_t len = unAlignLen / 2;
        SetAtomicAdd<uint16_t>();
        uint64_t mask0 = (1uL << 16) - (1uL << len);
        uint64_t mask[2] = {mask0, 0};
        Duplicate<uint16_t>(uint16Lt, 0, mask, 1, 1, 1);
        pipe_barrier(PIPE_ALL);

        DataCopy(uint16Gt, uint16Lt, ALIGN_16);
        SetAtomicNone();
    }

    __aicore__ inline void ComputeEachBatch()
    {
        oType jaggedPos;
        oType jaggedPosNext;

        __gm__ oType *oPtr = (__gm__ oType *)args->offset;

        for (int i = 0; i < args->singleCoreBatch; i++) {
            // Get information form offset tensor to jag dense
            jaggedPos = *(oPtr + offsetStartPos + i);
            jaggedPosNext = *(oPtr + offsetStartPos + i + 1);
            int copyRows = jaggedPosNext - jaggedPos;

            // Get jagged Global tensor with offset
            GlobalTensor<uint8_t> jaggedDenseCopyGb = jaggedDenseGb[jaggedPos * args->denseDim2 * align];
            GlobalTensor<uint8_t> denseCopyGb =
                denseGb[(offsetStartPos + i) * args->denseDim2 * args->denseDim1 * align];

            // When offset[n] - offset[n + 1] > dense dim1, only need to copy dense dim1 * dim2
            // otherwise, copy (offset[n] - offset[n + 1]) * dense dim2
            int64_t remainLen = copyRows > args->denseDim1 ? (args->denseDim1 * args->denseDim2 * align) :
                (copyRows * args->denseDim2 * align);
            while (remainLen > 0) {
                // args->singleLoopSize - ALIGN_32 to avoid overAlignLen exceed singleLoopSize
                int64_t thisLen = args->singleLoopSize - ALIGN_32;
                if (remainLen < (args->singleLoopSize - ALIGN_32)) {
                    thisLen = remainLen;
                }

                LocalTensor<uint8_t> localIn = inQueue.AllocTensor<uint8_t>();
                LocalTensor<uint8_t> localOut = outQueue.AllocTensor<uint8_t>();

                uint32_t overAlignLen = (thisLen + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32;
                uint32_t alignLen = thisLen / ALIGN_32 * ALIGN_32;
                uint32_t unAlignLen = thisLen - alignLen;

                // Copy over aligned size to avoid dealing unaligned tail
                DataCopy(localIn, denseCopyGb, overAlignLen);
                inQueue.EnQue(localIn);

                LocalTensor<uint8_t> localInCopy = inQueue.DeQue<uint8_t>();

                // Copy from input to output queue
                DataCopy(localOut, localInCopy, overAlignLen);
                outQueue.EnQue(localOut);

                LocalTensor<uint8_t> localOutCopy = outQueue.DeQue<uint8_t>();

                // Copy aligned size, left unaligned tail for DataCopyPad to deal with
                if (alignLen != 0) {
                    DataCopy(jaggedDenseCopyGb, localOutCopy, alignLen);
                }

                if (unAlignLen != 0) {
#ifndef SUPPORT_V200
                    const DataCopyExtParams dataCopyExtParams{1, unAlignLen, 0, 0, 0};
                    DataCopyPad(jaggedDenseCopyGb[alignLen], localOutCopy[alignLen], dataCopyExtParams);
#else
                    DataCopyPadLocal2Gm(jaggedDenseCopyGb[alignLen], localOutCopy[alignLen], unAlignLen);
#endif
                }

                jaggedDenseCopyGb = jaggedDenseCopyGb[thisLen];
                denseCopyGb = denseCopyGb[thisLen];
                inQueue.FreeTensor(localInCopy);
                outQueue.FreeTensor(localOutCopy);
                remainLen = remainLen - thisLen;
            }
        }
    }

    TPipe *pipe;
    int32_t align;
    int32_t thisId;
    int32_t offsetStartPos;
    DenseToJaggedArgs *args;

    GlobalTensor<uint8_t> denseGb;
    GlobalTensor<uint8_t> jaggedDenseGb;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
};

// 修改kernel调用部分，建议使用查表法重构
extern "C" __global__ __aicore__ void dense_to_jagged(GM_ADDR dense, GM_ADDR offset, GM_ADDR jagged_dense,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    DenseToJagged_Kernel::DenseToJaggedArgs args {
        dense, offset, jagged_dense, tiling_data.denseDim1, tiling_data.denseDim2, tiling_data.left,
        tiling_data.singleCoreBatch, tiling_data.singleLoopSize, tiling_data.denseTotal, tiling_data.jaggedTotal
    };

    TPipe pipe;
    int32_t floatType = DenseToJagged_Kernel::TYPE_FLOAT; // 0
    int32_t bf16Type = DenseToJagged_Kernel::TYPE_BF16;   // 15
    int32_t fp16Type = DenseToJagged_Kernel::TYPE_FP16;   // 14
    int32_t int32Type = DenseToJagged_Kernel::TYPE_INT32; // 3
    int32_t int64Type = DenseToJagged_Kernel::TYPE_INT64; // 9

    // Init DenseToJagged class with different data type according to dense and offset data type.
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
    } else if (tiling_data.denseType == bf16Type && tiling_data.offsetType == int32Type) {
        // BF16类型处理分支
        DenseToJagged_Kernel::DenseToJagged<bfloat16_t, int32_t> kernel;
        kernel.init(&args, &pipe);
        kernel.Compute();
    } else if (tiling_data.denseType == bf16Type && tiling_data.offsetType == int64Type) {
        // BF16类型处理分支
        DenseToJagged_Kernel::DenseToJagged<bfloat16_t, int64_t> kernel;
        kernel.init(&args, &pipe);
        kernel.Compute();
    } else if (tiling_data.denseType == fp16Type && tiling_data.offsetType == int32Type) {
        // FP16类型处理分支
        DenseToJagged_Kernel::DenseToJagged<float16_t, int32_t> kernel;
        kernel.init(&args, &pipe);
        kernel.Compute();
    } else if (tiling_data.denseType == fp16Type && tiling_data.offsetType == int64Type) {
        // FP16类型处理分支
        DenseToJagged_Kernel::DenseToJagged<float16_t, int64_t> kernel;
        kernel.init(&args, &pipe);
        kernel.Compute();
    }
}
```

#### 2.3.1 Kernel内部处理BF16/FP16类型的细节

由于BF16和FP16都是16位数据类型，它们在Kernel中的处理与FP32类似，但需要注意以下几点：

1. **内存对齐处理**：
   ```cpp
   align = sizeof(dType); // 对于BF16/FP16，sizeof返回2
   denseGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->dense), args->denseTotal * align);
   ```

2. **数据拷贝处理**：
   数据拷贝操作对所有数据类型都是一致的，Ascend C的DataCopy函数会根据实际数据大小进行处理。

3. **未对齐数据处理**：
   对于BF16/FP16类型的未对齐数据处理，需要使用专门的处理函数：
   ```cpp
   __aicore__ inline void DataCopyPadLocal2Gm(const GlobalTensor<uint8_t>& gt, const LocalTensor<uint8_t>& lt,
                                              uint32_t unAlignLen)
   {
       GlobalTensor<uint16_t> uint16Gt;
       uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), unAlignLen/2);
       LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

       uint32_t len = unAlignLen / 2;
       SetAtomicAdd<uint16_t>();
       uint64_t mask0 = (1uL << 16) - (1uL << len);
       uint64_t mask[2] = {mask0, 0};
       Duplicate<uint16_t>(uint16Lt, 0, mask, 1, 1, 1);
       pipe_barrier(PIPE_ALL);

       DataCopy(uint16Gt, uint16Lt, ALIGN_16);
       SetAtomicNone();
   }
   ```

4. **模板实例化**：
   BF16和FP16类型使用专门的模板实例化：
   ```cpp
   // BF16类型实例化
   DenseToJagged_Kernel::DenseToJagged<bfloat16_t, int32_t> kernel;
   
   // FP16类型实例化
   DenseToJagged_Kernel::DenseToJagged<float16_t, int64_t> kernel;
   ```

### 2.4 PTA适配层修改

修改PTA层代码以支持新数据类型，通过类型转换方式支持FP32+INT64->FP16/BF16等组合：

```cpp
// 在dense_to_jagged_forward_npu函数中增加BF16/FP16支持
at::Tensor dense_to_jagged_forward_npu(const at::Tensor& dense,
                                       const tensor_list& offsets,
                                       const c10::optional<int64_t> total_L)
{
    TORCH_CHECK(dense.dim() == 3,
        "dense must be 3-dimensional (B, MaxT, D), but got ", dense.dim(), "D input");
    TORCH_CHECK(offsets.size() == 1,
        "Only single-dimension jagged tensors supported (offsets.size() must be 1)");

    const at::OptionalDeviceGuard guard(device_of(dense));
    auto D = dense.size(-1);
    auto dense_contin = dense.contiguous();

    // 从offsets计算预期的total_L
    int64_t expected_total_L = offsets.back()[-1].item<int64_t>();

    // 校验输入的total_L
    if (total_L.has_value()) {
        TORCH_CHECK(
            total_L.value() == expected_total_L,
            "total_L (", total_L.value(), ") does not match the value calculated from offsets (",
            expected_total_L, ")"
        );
    }

    int64_t totalLength = total_L.value_or(expected_total_L);
    auto output = at::empty({totalLength, D}, dense.options());
    
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
```

## 3. 测试方案

### 3.1 功能测试
增加BF16和FP16数据类型的测试用例：

```python
@pytest.mark.parametrize("dense_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_dense_to_jagged_new_dtypes(dense_dtype, offset_dtype):
    # 测试BF16和FP16类型支持
    # 使用与现有测试相同的逻辑，但扩展数据类型范围
    pass
```

修改现有的测试用例以包含BF16和FP16类型：

```python
# 在test_dense_to_jagged.py中修改
DENSE_DATATYPE = [torch.float32, torch.int64, torch.bfloat16, torch.float16] # 增加新数据类型
```

### 3.2 边界测试
增加边界情况测试用例：

```python
# 边界测试用例
EDGE_CASE_DIMS = [
    (1, 10, 1),      # 最小batch和特征维度
    (10, 1, 16),     # 最小序列长度
    (1, 1, 1),       # 所有维度都最小
    (256, 500, 32),  # 较大的batch和特征维度
]

@pytest.mark.parametrize("dims", EDGE_CASE_DIMS)
@pytest.mark.parametrize("dense_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_dense_to_jagged_edge_cases(dims, dense_dtype, offset_dtype):
    """边界情况测试：测试各种极端维度组合"""
    # 实现边界测试逻辑
    pass
```

### 3.3 特殊场景测试
增加特殊场景测试：

```python
def test_dense_to_jagged_empty_offsets():
    """测试空偏移量的情况"""
    # 实现空偏移量测试逻辑
    pass

def test_dense_to_jagged_large_offsets():
    """测试大偏移量的情况"""
    # 实现大偏移量测试逻辑
    pass

@pytest.mark.parametrize("dense_dtype", [torch.bfloat16, torch.float16])
def test_bf16_fp16_precision(dense_dtype):
    """专门测试BF16和FP16精度"""
    # 实现BF16/FP16精度测试逻辑
    pass
```

### 3.4 性能测试
对比不同数据类型的性能表现：

1. 内存占用对比
2. 计算速度对比
3. 精度损失评估

### 3.5 精度测试
验证BF16/FP16与FP32的精度差异在可接受范围内。

## 4. 性能优化建议

### 4.1 查表法重构建议

当前[dense_to_jagged](file:///c%3A/zengxiong/RecSDK/mxrec_add_ons/rec_for_torch/operators/dense_to_jagged/op_kernel/dense_to_jagged.cpp#L180-L180)函数中使用了多个if-else分支来处理不同的数据类型组合，可以考虑使用查表法进行重构以提高性能：

```cpp
// 定义函数指针类型
typedef void (*KernelFunc)(DenseToJagged_Kernel::DenseToJaggedArgs*, TPipe*);

// 定义内核函数模板实例化
template<typename DT, typename OT>
void LaunchKernel(DenseToJagged_Kernel::DenseToJaggedArgs* args, TPipe* pipe) {
    DenseToJagged_Kernel::DenseToJagged<DT, OT> kernel;
    kernel.init(args, pipe);
    kernel.Compute();
}

// 定义查表结构
struct KernelEntry {
    int denseType;
    int offsetType;
    KernelFunc func;
};

// 构建内核函数查找表
static const KernelEntry kernelTable[] = {
    {DenseToJagged_Kernel::TYPE_FLOAT, DenseToJagged_Kernel::TYPE_INT32, LaunchKernel<float, int32_t>},
    {DenseToJagged_Kernel::TYPE_FLOAT, DenseToJagged_Kernel::TYPE_INT64, LaunchKernel<float, int64_t>},
    {DenseToJagged_Kernel::TYPE_INT64, DenseToJagged_Kernel::TYPE_INT64, LaunchKernel<int64_t, int64_t>},
    {DenseToJagged_Kernel::TYPE_INT64, DenseToJagged_Kernel::TYPE_INT32, LaunchKernel<int64_t, int32_t>},
    {DenseToJagged_Kernel::TYPE_BF16, DenseToJagged_Kernel::TYPE_INT32, LaunchKernel<bfloat16_t, int32_t>},
    {DenseToJagged_Kernel::TYPE_BF16, DenseToJagged_Kernel::TYPE_INT64, LaunchKernel<bfloat16_t, int64_t>},
    {DenseToJagged_Kernel::TYPE_FP16, DenseToJagged_Kernel::TYPE_INT32, LaunchKernel<float16_t, int32_t>},
    {DenseToJagged_Kernel::TYPE_FP16, DenseToJagged_Kernel::TYPE_INT64, LaunchKernel<float16_t, int64_t>}
};

// 使用查表法调用内核函数
extern "C" __global__ __aicore__ void dense_to_jagged(GM_ADDR dense, GM_ADDR offset, GM_ADDR jagged_dense,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    DenseToJagged_Kernel::DenseToJaggedArgs args {
        dense, offset, jagged_dense, tiling_data.denseDim1, tiling_data.denseDim2, tiling_data.left,
        tiling_data.singleCoreBatch, tiling_data.singleLoopSize, tiling_data.denseTotal, tiling_data.jaggedTotal
    };

    TPipe pipe;
    
    // 查找并调用对应的内核函数
    for (const auto& entry : kernelTable) {
        if (entry.denseType == tiling_data.denseType && entry.offsetType == tiling_data.offsetType) {
            entry.func(&args, &pipe);
            return;
        }
    }
    
    // 如果没有找到匹配的类型组合，可以抛出错误或使用默认处理
}
```

通过这种方式，可以将原来的多个if-else分支简化为一个查表过程，不仅提高了代码的可读性，也可能在某些编译器优化下获得更好的性能。不过需要注意的是，这种优化的实际效果还需要通过性能测试来验证。

## 5. 实施计划

1. **第一阶段**：修改JSON配置文件和Host端代码，扩展类型支持
2. **第二阶段**：修改Kernel端代码，增加BF16/FP16支持
3. **第三阶段**：修改PTA适配层，支持新数据类型
4. **第四阶段**：编写测试用例，验证功能正确性
5. **第五阶段**：性能测试和精度评估

## 6. 风险评估

1. **兼容性风险**：新数据类型可能与现有代码不兼容，已通过PTA层类型转换解决
2. **精度风险**：BF16/FP16精度可能影响模型训练效果，通过测试验证精度在可接受范围内
3. **性能风险**：新数据类型可能在某些场景下性能不如FP32，通过性能测试评估

通过以上详细设计方案，可以为dense_to_jagged算子增加BF16和FP16支持，提升算子在实际应用中的性能表现。