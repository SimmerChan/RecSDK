# 使用pytorch的方式调用算子工程
该样例脚本基于python3.11,torch支持2.1.0、2.5.1、2.6.0版本。调用算子前需完成配套软件的安装和所需算子的安装。目前支持的torch版本配套关系如下：

| torch版本      | 配套关系                                                                                          |
|--------------|-----------------------------------------------------------------------------------------------|
| torch==2.1.0 | torch_npu==2.1.0<br/>fbgemm+gpu==0.5.0+cpu<br/>torchrec==0.5.0+npu<br/>hybrid_torchrec==0.5.0 |
| torch==2.5.1 | torch_npu==2.5.1<br/>fbgemm+gpu==1.0.0+cpu<br/>torchrec==1.0.0+npu<br/>hybrid_torchrec==1.0.0 |
| torch==2.6.0 | torch_npu==2.6.0<br/>fbgemm+gpu==1.1.0+cpu<br/>torchrec==1.1.0+npu<br/>hybrid_torchrec==1.1.0 |

## 运行样例算子
### 1.安装自定义算子
- 下载mindxsdk-mxrec-add-ons-poc软件包,在mindxsdk-mxrec-add-ons-poc/mxrec_ops/目录下安装所需执行的算子，
例如：

```bash
    # 按需安装算子
    cd mindxsdk-mxrec-add-ons-poc/mxrec_ops/
    bash mxrec_opp_permute2d_sparse_data.run
    bash mxrec_opp_bounds_check_indices.run
    bash mxrec_opp_asynchronous_complete_cumsum.run
    bash mxrec_opp_split_embedding_codegen_forward_unweighted.run
    bash mxrec_opp_backward_codegen_adagrad_unweighted_exact.run
```

### 2.安装算子适配层

  - 以torch版本2.6.0为例,进入到样例目录,执行如下命令。
    ```bash
    cd mindxsdk-mxrec-add-ons-poc/load_library/2.6.0/common/
    bash build_ops.sh
    ```
  - 执行命令后会在common目录下生成libfbgemm_npu_api.so文件，并同时在python默认的site-packages路径下存放编译好的libfbgemm_npu_api.so方便使用。

### 3.样例执行

  - 样例执行时通过torch.ops.load_library的方式加载适配层文件。以下样例以python默认路径为例。
    ```bash
    import sysconfig
    import torch
    torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
    ```
  - 说明:common目录下的编译会将版本目录的所有算子适配层编译成一个.so文件,方便使用。如果只想编译单个算子的.so,只需进入具体算子适配层目录进行编译。例如：
    ```bash
    # 编译
    cd mindxsdk-mxrec-add-ons-poc/load_library/2.6.0/permute2d_sparse_data/
    bash build_ops.sh
    
    # 加载
    import sysconfig
    import torch
    torch.ops.load_library("path/to/build/libpermute2d_sparse_data.so") #.so文件的绝对路径
    ```

## 更新说明
| 时间         | 更新事项     |
|------------|----------|
| 2024/05/22 | 新增本readme |
| 2025/04/22 | 更新算子调用方式 |