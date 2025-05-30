# Mmoe
本文档主要介绍如何进行Mmoe模型的数据预处理和训练

## 主要依赖
**Pytorch:** 2.6.0

## 数据集准备
1. 准备alicpp数据集
2. 对于[Ali-CPP](https://tianchi.aliyun.com/dataset/408)数据集，我们提供完整的预处理流程[参考](https://gitee.com/ascend/RecSDK/blob/develop/examples/rec_model_zoo/behaviour_and_multi_task/data/aliccp/README.md)，进入cliccp目录执行如下命令：
```commandline
cd ./aliccp
bash run.sh 
```
执行完成后数据集会生成到指定目录，本用例默认生成在aliccp_out目录。
## 训练
执行训练脚本，传入模型所需参数，参考命令如下：
```commandline
python3 mmoe.py --data_dir ./aliccp_out/   # 根据实际情况传入参数
```
2.参数说明
```commandline
通过以下命令查看参数及默认值情况
python3 mmoe.py  --help
```

