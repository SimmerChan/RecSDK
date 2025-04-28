# split_embedding_codegen_lookup_adagrad_function
fbgemm查表算子的npu实现，用于查找一个或多个嵌入表，适用于训练，支持嵌入表在反向传播期间进行更新。当前优化器仅支持EXACT_ADAGRAD。

## split_embedding_codegen_forward_unweighted算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的split_embedding_codegen_forward_unweighted, 实现embedding bag的查询功能

b) 算子参数说明：
| 名称 | 类型 | 数据类型 | 数据格式 | 备注 |
|----|----|----|----|----|
| dev_weights | 输入| float32 | embed_dim * embed_size | 表的weights |
| uvm_weight | 输入| float32 | N/A | 预留参数不支持配置 |
| lxu_cache_weight | 输入| float32 | N/A| 预留参数不支持配置 |
| weights_pacements | 输入 | float32 | N/A | 预留参数不支持配置 |
| weights_offsets | 输入 | int64_t | [ num_embed+1 ] | 每张表的偏移量 |
| D_offsets | 输入 | int32-t | [ num_embed+1 ] | 每张表embeding dim偏移量 |
| indices | 输入 | int64_t | 一维张量 | 查表的索引值 |
| hash_indices | 可选输入 | int64_t | 一维张量 | 稀疏表查表的索引值 |
| offsets | 输入 | int64_t | [batchSize * num_embed + 1] | 索引对应的偏移|
| lxu_cache_locations | 输入 | int64_t | N/A |预留参数不支持配置|
| out | 输出 | float32 | [batchSize, total_D] |查询的向量 |
| total_D | 属性 | int | N/A | 输出的dim之和 |
| max_D | 属性 | int | N/A| 表中最大的Embedding Dim|
| pooling_mode | 属性 | int | N/A | pooling的方式Sum或者Mean或None|
| output_dtype | 属性 | int | N/A | 预留参数不支持配置|
| is_experimental | 属性 | int | N/A | 预留参数不支持配置|

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.0.RC2及之后版本；
* embed_dim：每张表的dim长度需为8的倍数
* num_embed: 表的个数
* indices的dim0为offset最后一位的值


## backward_codegen_adagrad_unweighted_exact融合算子介绍

1. 算子分析

a) 算子的主要功能是实现fbgemm的backward_codegen_adagrad_unweighted_exact, 实现了将反向梯度计算后，使用adagrad算法，将weights和momentum1进行更新

b) 算子参数说明：
| 名称 | 类型 | 数据类型 | 数据格式 | 备注 |
|----|----|----|----|----|
| grad_output | 输入| float32 |[batchsize, total]  | 查询向量的反向梯度 |
| dev_weights | 输入| float32 | [embed_dim * embed_size] | 预留参数不支持配置 |
| uvm_weight | 输入| float32 | N/A  | 预留参数不支持配置 |
| lxu_cache_weight | 输入| float32 | N/A | 预留参数不支持配置 |
| weights_pacements | 输入 | float32 | N/A  | 预留参数不支持配置 |
| weights_offsets | 输入 | int64_t | [ num_embed+1 ] | 每张表的偏移量 |
| D_offsets | 输入 | int32_t | [ num_embed+1 ] | 每张表embeding dim的偏移量 |
| hash_size_cumsum | 输入 | int32_t | [ num_embed+1 ] | 表size的偏移 |
| indices | 输入 | int64_t | 一维张量 | 查表的索引值 |
| hash_indices | 可选输入 | int64_t | 一维张量 | 稀疏表查表的索引值 |
| unique_offsets | 可选输入 | int64_t | 一维张量 | 每张表去重后的偏移量 |
| unique_ids | 可选输入 | int64_t | 一维张量 | 稀疏表查表的索引值 |
| unique_inverse | 可选输入 | int64_t |一维张量 | 查询表的索引对应的unique_ids位置 |
| offsets | 输入 | int64_t | [batchSize * num_embed + 1] | 索引对应的偏移|
| lxu_cache_locations | 输入 | int64_t | N/A |预留参数不支持配置|
| momentum1_uvm | 输入 | float32 | N/A |预留参数不支持配置 |
| momentum1_placements | 输入 | int32_t | N/A |预留参数不支持配置 |
| momentum1_offsets | 输入 | int64_t | N/A |预留参数不支持配置 |
| momentum1_dev | 输出 | float32 | [embed_dim * embed_size] |输出值 |
| max_D | 属性 | int | N/A| 表中最大的Embedding Dim |
| total_hash_size_bits | 属性 | int | N/A| hash表size和的int值用多少位bit表示|
| pooling_mode | 属性 | int | N/A | pooling的方式Sum或者Mean或None
| BT_block_size | 属性 | int | N/A| 预留参数不支持配置|
| max_segment_length_per_warp | 属性 | int | N/A| 预留参数不支持配置|
| stochastic_rounding | 属性 | int | N/A| 预留参数不支持配置|
| info_B_num_bits | 属性 | int | N/A| 预留参数不支持配置|
| info_B_mask_int64 | 属性 | int | N/A| 预留参数不支持配置|
| use_uniq_cache_locations | 属性 | int | N/A| 预留参数不支持配置|
| use_homogeneous_placements | 属性 | int | N/A| 预留参数不支持配置|
| eps | 属性 | float | N/A | adagrad的eps|
| learning_rate | 属性 | float | N/A | 学习率|

c) 算子约束说明：

* 支持的型号：Atlas A2系列产品;
* 支持的CANN版本：8.0.RC2及之后版本；
* embed_dim：每张表的dim长度需为8的倍数
* num_embed: 表的个数
* indices的dim0为offset最后一位的值，unique_ids的dim0为unique_offsets最后一位的值。
* 配置unique_ids时，需同时配置unique_offsets, unique_inverse，配置后将使用unique信息进行参数更新。

## 使用方式

### 下载软件包并解压
tar -zxvf Ascend-mindxsdk-mxrec-add-ons-poc-linux-aarch64.tar.gz

### 部署安装算子
进入解压后的mxrec_ops目录
执行./mxrec_opps_split_embedding_codegen_forward_unweighted.run 完成算子安装部署
执行./mxrec_opps_backward_codegen_adagrad_unweighted_exact.run 完成算子安装部署


### 安装torch适配层
进入解压后的torch_library/torch_plugin目录
pip3 install torch_npu*.whl
