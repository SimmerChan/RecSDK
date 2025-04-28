# Ascend4Tencent

#### 介绍
托管贡献给腾讯的算子

#### 软件架构
```
.
|-- CMakeLists.txt
|-- README.md
|-- SparseFieldsConcatV2Grad.iml
|-- build.sh
|-- cmake
|   |-- config.cmake
|   |-- external                                    //第三方依赖库
|   |   |-- eigen.cmake
|   |   `-- secure_c.cmake
|   `-- util                                        //框架自动生成                                     
|       |-- ...
|-- cpukernel
|   |-- CMakeLists.txt
|   |-- impl
|   |   |-- sparse_fields_concat_v2_grad_kernels.cc   //算子实现文件
|   |   `-- sparse_fields_concat_v2_grad_kernels.h    //算子实现头文件
|   |-- op_info_cfg
|   |   `-- aicpu_kernel
|   |       `-- sparse_fields_concat_v2_grad.ini      //算子信息库，框架自动生成
|   `-- toolchain.cmake
|-- framework
|   |-- CMakeLists.txt
|   `-- tf_plugin
|       |-- CMakeLists.txt
|       `-- tensorflow_sparse_fields_concat_v2_grad_plugin.cc   //tf插件适配
|-- op_proto
|   |-- CMakeLists.txt
|   |-- sparse_fields_concat_v2_grad.cc                    //算子shape dtype推导
|   `-- sparse_fields_concat_v2_grad.h                     //算子原型定义
|-- op_tiling
|   `-- CMakeLists.txt
|-- scripts
|   |-- help.sh
|   |-- install.sh
|   |-- uninstall.sh
|   `-- upgrade.sh
`-- testcases
    `-- st
        `-- sparse_fields_concat_v2_grad
            `-- aicpu_kernel
                `-- SparseFieldsConcatV2Grad_case_20220810104154.json    //算子测试文件
```

#### 安装教程
1. 设置环境变量
修改build.sh中ASCEND_HOME变量为自己的Ascend安装路径
```
export ASCEND_HOME=/usr/local/Ascend
```


2. 编出run包
```
bash build.sh
```

3. 安装run包
```
cd build_out/
./custom_opp_tlinux_x86_64.run
```

#### 使用说明

1.  运行st，可以验证算子功能是否正确。运行命令

```python3.7 /usr/local/Ascend/ascend-toolkit/5.1.RC2/python/site-packages/bin/msopst run -i ./testcases/st/sparse_fields_concat_v2_grad/aicpu_kernel/SparseFieldsConcatV2Grad_case_20220810104154.json -soc Ascend910A -d 6```

参数说明：
- /usr/local/Ascend/ascend-toolkit/5.1.RC2/python/site-packages/bin/msopst   st框架地址
- ./testcases/st/sparse_fields_concat_v2_grad/aicpu_kernel/SparseFieldsConcatV2Grad_case_20220810104154.json   算子用例
- -soc 指定类型
- -d   指定device
2.  运行tf测试。
    python3.7 testcases/tf_test/test_sparse_fields_concat_v2_grad.py

#### 参与贡献
1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request

