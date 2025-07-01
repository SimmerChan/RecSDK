# split_embedding_codegen_forward_unweighted算子及样例说明

## split_embedding_codegen_forward_unweighted算子文件结构

```shell
├── split_embedding_codegen_forward_unweighted.json    # 算子原型配置
├── op_host    # split_embedding_codegen_forward_unweighted算子Host侧实现
├── op_kernel  # split_embedding_codegen_forward_unweighted算子Kernel侧实现
├── README.md  # split_embedding_codegen_forward_unweighted算子说明文档
└── run.sh     # split_embedding_codegen_forward_unweighted算子安装脚本
```

## Ascend C参考设计

更多详情可以参考CANN官方的Ascend
C算子开发手册[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

## split_embedding_codegen_forward_unweighted算子使用

1. 上传split_embedding_codegen_forward_unweighted文件夹到目标环境，并进入当前目录，执行指令对split_embedding_codegen_forward_unweighted算子进行编译和部署

```shell
bash run.sh
```

注：需先在环境中设置CANN相关环境变量，再执行算子编译和安装指令。使用默认路径安装CANN时设置环境变量指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## split_embedding_codegen_forward_unweighted算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的split_embedding_codegen_forward_unweighted, 实现embedding bag的查询功能
b) 算子参数说明：

* dev_weights: 表的weights；
* uvm_weight: 预留参数不支持配置；
* lxu_cache_weight: 预留参数不支持配置；
* weights_pacements: 预留参数不支持配置;
* weights_offsets: 每张表的偏移量;
* D_offsets: 每张表embeding dim的offsets;
* indices: 查询表的indics;
* offsets: indices对应的偏移;
* lxu_cache_locations: 预留参数不支持配置;
* out: 查询的向量；
* total_D: 输出的dim之和;
* max_D: 表中最大的Embedding Dim;;
* pooling_mode: pooling的方式Sum或者Mean;
* output_dtype: 预留参数不支持配置;
* is_experimental: 预留参数不支持配置;

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.0.RC2及之后版本；
* 支持的输入数据类型：dev_weights为float32类型，weights_offsets、indices、offsets为int64，D_offsets为int32；
* ，dev_weights的dims为所有表的[embed_dim * embed_size]，weights_offsets为表的个数[ num_embed ], weights_offsets的dims为[ num_embed+1 ], D_offsets的dim为[ num_embed+1 ]。indices的dim0为offset最后一位的值。offsets为[batchsize, num_embed]。

## 算子逻辑
```
feat_cnt = weights_offsets.shape[0]
batch_size = (offsets.shape[0]-1) // feat_cnt
results = np.zeros((batch_size, total_D)).astype(np.float32)
for i in range(feat_cnt):
    embed_dim = D_offsets[i+1] - D_offsets[i]
    for b in range(batch_size):
        this_offset = offsets[i*batch_size+b]
        next_offset = offsets[i*batch_size+b+1]
        this_indics = indices[this_offset: next_offset]
        for j in this_indics:
            this_embed_index = weights_offsets[i]+j*embed_dim
            this_embed = dev_weights[this_embed_index: this_embed_index+embed_dim]
            results[b, D_offsets[i]:D_offsets[i+1]] = results[b, D_offsets[i]:D_offsets[i+1]] + this_embed/len(this_indics)
```