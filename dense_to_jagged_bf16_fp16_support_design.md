# dense_to_jagged算子BF16/FP16支持详细设计

## 1. 设计目标

为dense_to_jagged算子增加BF16和FP16数据类型支持，以提升计算效率和降低内存占用。

## 2. 设计方案

### 2.1 JSON配置文件修改

修改dense_to_jagged.json文件，增加BF16和FP16支持：

```json
{
    "op": "DenseToJagged",
    "language": "cpp",
    "input_desc": [
        {
            "name": "dense",
            "param_type": "required",
            "format": [
                "ND", "ND", "ND", "ND", "ND", "ND"
            ],
            "type": [
                "fp32", "fp32", "int64", "int64", "bf16", "fp16"
            ]
        },
        {
            "name": "offset",
            "param_type": "required",
            "format": [
                "ND", "ND", "ND", "ND"
            ],
            "type": [
                "int64", "int32", "int32", "int64"
            ]
        }
    ],
    "output_desc": [
        {
            "name": "jagged_dense",
            "param_type": "required",
            "format": [
                "ND", "ND", "ND", "ND", "ND", "ND"
            ],
            "type": [
                "fp32", "fp32", "int64", "int64", "bf16", "fp16"
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

``cpp
// 增加新的类型常量定义
constexpr int32_t TYPE_BF16 = 15;
constexpr int32_t TYPE_FP16 = 14;

// 修改OpDef注册
this->Input("dense")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, ge::DT_BF16, ge::DT_FLOAT16})
    .FormatList({ge::FORMAT_ND});
this->Output("jagged_dense")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, ge::DT_BF16, ge::DT_FLOAT16})
    .FormatList({ge::FORMAT_ND});
```

### 2.3 Kernel端修改

修改[op_kernel/dense_to_jagged.cpp](file:///c%3A/zengxiong/RecSDK/mxrec_add_ons/rec_for_torch/operators/dense_to_jagged/op_kernel/dense_to_jagged.cpp)文件，扩展模板参数支持：

```
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
        int32_t denseTotal;
        int32_t jaggedTotal;
    };

    void init(DenseToJaggedArgs* args, TPipe* pipe) {
        this->args = args;
        this->pipe = pipe;
    }

    void Compute() {
        int32_t align = sizeof(dType);
        denseGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->dense), args->denseTotal * align);
        offsetGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->offset), args->jaggedTotal * sizeof(oType));
        jaggedGb.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(args->jagged_dense), args->jaggedTotal * align);

        int32_t loopNum = args->denseDim1 / args->singleCoreBatch;
        int32_t loopRemain = args->denseDim1 % args->singleCoreBatch;

        for (int32_t loop = 0; loop < loopNum; loop++) {
            int32_t start = loop * args->singleCoreBatch;
            int32_t end = start + args->singleCoreBatch;

            ComputeLoop(start, end);
        }

        if (loopRemain > 0) {
            int32_t start = loopNum * args->singleCoreBatch;
            int32_t end = start + loopRemain;

            ComputeLoop(start, end);
        }
    }

private:
    void ComputeLoop(int32_t start, int32_t end) {
        int32_t align = sizeof(dType);
        LocalTensor<uint8_t> denseLt = LocalTensor<uint8_t>(args->singleLoopSize * args->denseDim2 * align);
        LocalTensor<uint8_t> offsetLt = LocalTensor<uint8_t>(args->singleLoopSize * sizeof(oType));
        LocalTensor<uint8_t> jaggedLt = LocalTensor<uint8_t>(args->singleLoopSize * args->denseDim2 * align);

        for (int32_t i = start; i < end; i++) {
            int32_t offset = i * args->denseDim2 * align;
            DataCopy(denseLt, denseGb, offset, args->denseDim2 * align);

            offset = i * sizeof(oType);
            DataCopy(offsetLt, offsetGb, offset, sizeof(oType));

            ComputeSingle(denseLt, offsetLt, jaggedLt);

            offset = i * args->denseDim2 * align;
            DataCopy(jaggedGb, jaggedLt, offset, args->denseDim2 * align);
        }
    }

    void ComputeSingle(LocalTensor<uint8_t>& denseLt, LocalTensor<uint8_t>& offsetLt, LocalTensor<uint8_t>& jaggedLt) {
        int32_t align = sizeof(dType);
        LocalTensor<uint8_t> denseLtPad = LocalTensor<uint8_t>(args->singleLoopSize * args->denseDim2 * align);
        LocalTensor<uint8_t> offsetLtPad = LocalTensor<uint8_t>(args->singleLoopSize * sizeof(oType));
        LocalTensor<uint8_t> jaggedLtPad = LocalTensor<uint8_t>(args->singleLoopSize * args->denseDim2 * align);

        DataCopyPadLocal2Local(denseLtPad, denseLt, args->left);
        DataCopyPadLocal2Local(offsetLtPad, offsetLt, args->left);
        DataCopyPadLocal2Local(jaggedLtPad, jaggedLt, args->left);

        ComputeSinglePad(denseLtPad, offsetLtPad, jaggedLtPad);

        DataCopyPadLocal2Local(denseLt, denseLtPad, args->left);
        DataCopyPadLocal2Local(offsetLt, offsetLtPad, args->left);
        DataCopyPadLocal2Local(jaggedLt, jaggedLtPad, args->left);
    }

    void ComputeSinglePad(LocalTensor<uint8_t>& denseLt, LocalTensor<uint8_t>& offsetLt, LocalTensor<uint8_t>& jaggedLt) {
        int32_t align = sizeof(dType);
        LocalTensor<uint8_t> denseLtPad = LocalTensor<uint8_t>(args->singleLoopSize * args->denseDim2 * align);
        LocalTensor<uint8_t> offsetLtPad = LocalTensor<uint8_t>(args->singleLoopSize * sizeof(oType));
        LocalTensor<uint8_t> jaggedLtPad = LocalTensor<uint8_t>(args->singleLoopSize * args->denseDim2 * align);

        DataCopyPadLocal2Local(denseLtPad, denseLt, args->left);
        DataCopyPadLocal2Local(offsetLtPad, offsetLt, args->left);
        DataCopyPadLocal2Local(jaggedLtPad, jaggedLt, args->left);

        ComputeSinglePad(denseLtPad, offsetLtPad, jaggedLtPad);

        DataCopyPadLocal2Local(denseLt, denseLtPad, args->left);
        DataCopyPadLocal2Local(offsetLt, offsetLtPad, args->left);
        DataCopyPadLocal2Local(jaggedLt, jaggedLtPad, args->left);
    }

    void DataCopyPadLocal2Local(LocalTensor<uint8_t>& ltPad, LocalTensor<uint8_t>& lt, int32_t left) {
        int32_t align = sizeof(dType);
        LocalTensor<uint8_t> ltPadLeft = LocalTensor<uint8_t>(left * args->denseDim2 * align);
        LocalTensor<uint8_t> ltPadRight = LocalTensor<uint8_t>(left * args->denseDim2 * align);

        DataCopy(ltPadLeft, ltPad, 0, left * args->denseDim2 * align);
        DataCopy(ltPadRight, ltPad, args->singleLoopSize * args->denseDim2 * align - left * args->denseDim2 * align,
                 left * args->denseDim2 * align);

        DataCopy(ltPad, lt, left * args->denseDim2 * align, args->singleLoopSize * args->denseDim2 * align - 2 * left * args->denseDim2 * align);
    }

    void DataCopyPadLocal2Gm(const GlobalTensor<uint8_t>& gt, const LocalTensor<uint8_t>& lt,
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

    DenseToJaggedArgs* args;
    TPipe* pipe;
    GlobalTensor<uint8_t> denseGb;
    GlobalTensor<uint8_t> offsetGb;
    GlobalTensor<uint8_t> jaggedGb;
};

// 修改kernel调用部分
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

    // 扩展类型支持
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
        DenseToJagged_Kernel::DenseToJagged<bfloat16, int32_t> kernel;
        kernel.init(&args, &pipe);
        kernel.Compute();
    } else if (tiling_data.denseType == bf16Type && tiling_data.offsetType == int64Type) {
        DenseToJagged_Kernel::DenseToJagged<bfloat16, int64_t> kernel;
        kernel.init(&args, &pipe);
        kernel.Compute();
    } else if (tiling_data.denseType == fp16Type && tiling_data.offsetType == int32Type) {
        DenseToJagged_Kernel::DenseToJagged<float16, int32_t> kernel;
        kernel.init(&args, &pipe);
        kernel.Compute();
    } else if (tiling_data.denseType == fp16Type && tiling_data.offsetType == int64Type) {
        DenseToJagged_Kernel::DenseToJagged<float16, int64_t> kernel;
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
   DenseToJagged_Kernel::DenseToJagged<bfloat16, int32_t> kernel;
   
   // FP16类型实例化
   DenseToJagged_Kernel::DenseToJagged<float16, int64_t> kernel;
   ```

### 2.4 PTA适配层修改

修改PTA层代码以支持新数据类型：

``cpp
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
    
    // 支持BF16和FP16类型
    if (dense.dtype() == at::kBFloat16 || dense.dtype() == at::kHalf) {
        EXEC_NPU_CMD(aclnnDenseToJagged, dense_contin.to(at::kFloat), offsets[0], totalLength, output);
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

``python
@pytest.mark.parametrize("dense_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_dense_to_jagged_new_dtypes(dense_dtype, offset_dtype):
    # 测试BF16和FP16类型支持
    # 使用与现有测试相同的逻辑，但扩展数据类型范围
    pass
```

修改现有的测试用例以包含BF16和FP16类型：

``python
# 在test_dense_to_jagged.py中修改
DENSE_DATATYPE = [torch.float32, torch.int64, torch.bfloat16, torch.float16] # 增加新数据类型
```

### 3.2 性能测试
对比不同数据类型的性能表现：

1. 内存占用对比
2. 计算速度对比
3. 精度损失评估

### 3.3 精度测试
验证BF16/FP16与FP32的精度差异在可接受范围内。

## 4. 实施计划

1. **第一阶段**：修改JSON配置文件和Host端代码
2. **第二阶段**：修改Kernel端代码，增加BF16/FP16支持
3. **第三阶段**：修改PTA适配层，支持新数据类型
4. **第四阶段**：编写测试用例，验证功能正确性
5. **第五阶段**：性能测试和精度评估

## 5. 风险评估

1. **兼容性风险**：新数据类型可能与现有代码不兼容
2. **精度风险**：BF16/FP16精度可能影响模型训练效果
3. **性能风险**：新数据类型可能在某些场景下性能不如FP32

通过以上详细设计方案，可以为dense_to_jagged算子增加BF16和FP16支持，提升算子在实际应用中的性能表现。