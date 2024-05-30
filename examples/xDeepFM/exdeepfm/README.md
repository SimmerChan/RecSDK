# xDeepFM迁移样例

## 模型参考开源链接

1. https://github.com/Leavingseason/xDeepFM

2. Commits on Oct 15, 2018，提交的SHA-1 hash值（提交ID）：114c4c45b1cb6144b2540f92a2b357c3f445e98e

3. 只保留执行所需要的代码及文件，其他已删除。

## 迁移NPU

请参照昇腾社区CANN商用版文档先使用迁移工具进行NPU自动迁移：https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldev/tfmigr1/tfmigr1_000009.html


## 迁移mxRec

1、修改IO/iterator.py，把第30~41行


```python
        _fm_feat_indices, _fm_feat_values,
        _fm_feat_shape, _labels, _dnn_feat_indices,
        _dnn_feat_values, _dnn_feat_weights, _dnn_feat_shape = iterator.get_next()
        self.initializer = iterator.initializer
        self.fm_feat_indices = _fm_feat_indices
        self.fm_feat_values = _fm_feat_values
        self.fm_feat_shape = _fm_feat_shape
        self.labels = _labels
        self.dnn_feat_indices = _dnn_feat_indices
        self.dnn_feat_values = _dnn_feat_values
        self.dnn_feat_weights = _dnn_feat_weights
        self.dnn_feat_shape = _dnn_feat_shape
```
` ` ` `改为：
```python
        batch = iterator.get_next()
        self.initializer = iterator.initializer
        self.fm_feat_indices = batch.get('fm_feat_indices')
        self.fm_feat_values = batch.get('fm_feat_values')
        self.fm_feat_shape = batch.get('fm_feat_shape')
        self.labels = batch.get('labels')
        self.dnn_feat_indices = batch.get('dnn_feat_indices')
        self.dnn_feat_values = batch.get('dnn_feat_values')
        self.dnn_feat_weights = batch.get('dnn_feat_weights')
        self.dnn_feat_shape = batch.get('dnn_feat_shape')
```

` ` ` `第63~65行
```python
        return fm_feat_indices, fm_feat_values,
        fm_feat_shape, labels, dnn_feat_indices,
        dnn_feat_values, dnn_feat_weights, dnn_feat_shape
```
` ` ` `改为：
```python
        return {
            'fm_feat_indices': fm_feat_indices, 'fm_feat_values': fm_feat_values, 'fm_feat_shape': fm_feat_shape,
            'labels': labels, 'dnn_feat_indices': dnn_feat_indices, 'dnn_feat_values': dnn_feat_values,
            'dnn_feat_weights': dnn_feat_weights, 'dnn_feat_shape': dnn_feat_shape
        }
```

2、修改src/base_model.py。把embedding初始化值设成tf.zeros_initializer()，把84行
```python
        return tf.truncated_normal_initializer(stddev=hparams.init_value)
```
` ` ` `改为：
```python
        return tf.zeros_initializer()
```

` ` ` `更新自动改图模式下生成新数据集中batch的label记录，把188~189行
```python
    def eval(self, sess):
        return sess.run([self.loss, self.data_loss, self.pred, self.iterator.labels], \
```
` ` ` `改为：
```python
    def eval(self, sess, eval_label):
        return sess.run([self.loss, self.data_loss, self.pred, eval_label], \
```

3、修改src/base_model.py。把embedding初始化值设成tf.zeros_initializer()，把84行
```python
        return tf.truncated_normal_initializer(stddev=hparams.init_value)
```
` ` ` `改为：
```python
        return tf.zeros_initializer()
```


### 二进制包安装

从昇腾开源社区直接获取编译打包后的产品包。解压后包含tf1和tf2两个版本的whl安装包，使用pip命令安装whl包（请根据实际需求，选取对应TensorFlow版本匹配的Wheel包）：
```shell
pip3 install mx_rec-{version}-py3-none-linux_{arch}.whl
```

Wheel包默认安装在Python的“site-packages”路径，如通过“--target”参数指定目录，在安装完成后需要将mxRec路径加入“PYTHONPATH”环境变量。

```shell
export PYTHONPATH={mxrec_install_path}:{mxrec_install_path}/mxRec:$PYTHONPATH
```

如需使用动态扩容功能，进入已解压的mxRec软件包“mindxsdk-mxrec/cust_op/cust_op_by_addr”目录中。参考以下命令编译并安装动态扩容算子包。
```shell
bash run.sh
```

### 源码编译安装

编译环境依赖：
- Python3.7.5
- GCC 7.3.0
- CMake 3.20.6

开源依赖：
- [pybind11 v2.10.3](https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip)
- [securec](https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip)
- [openmpi 4.1.5](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz): 请参考软件文档在编译环境完成安装
- tensorflow 1.15/2.6.5：根据实际需求选择对应版本

将pybind11和securec的压缩包放在与mxRec代码同级的opensource目录下，并且将其分别更名为pybind11-2.10.3.zip、huaweicloud-sdk-c-obs-3.23.9.zip。如果没有opensource目录，则需要在mxRec同级的目录下手动创建opensource目录，然后将pybind11和securec的压缩包放在opensource目录下。

为了构建多个版本的whl包，编译脚本在python虚拟环境完成对应tensorflow版本的安装。用户可以根据实际情况调整编译脚本，指定tensorflow的安装路径。编译方法：

进入mxRec代码目录：
- setup.py：执行脚本setup.py，比如：**python3.7 setup.py**完成tf1和tf2版本whl包的构建和打包，构建成功后，whl包在build/mindxsdk-mxrec/目录下，其中tf1_whl和tf2_whl目录下存在对应的whl包。执行脚本前，请参考build/build_tf1.sh、build/build_tf2.sh创建对应的虚拟环境，在虚拟环境中完成对应tensorflow版本的安装，并修改对应的激活命令。
- setup_tf1.py：执行脚本setup_tf1.py，比如：**python3.7 setup_tf1.py bdist_wheel**完成tf1版本whl包的构建，构建成功后，whl包在build/mindxsdk-mxrec/tf1_whl子目录下。执行脚本前，请参考build/build_tf1.sh创建tf1虚拟环境，在虚拟环境中完成tensorflow 1.15.0版本的安装，并修改对应的激活命令。
- setup_tf2.py：执行脚本setup_tf2.py，比如：**python3.7 setup_tf2.py bdist_wheel**完成tf2版本whl包的构建，构建成功后，whl包在build/mindxsdk-mxrec/tf2_whl子目录下。执行脚本前，请参考build/build_tf2.sh创建tf2虚拟环境，在虚拟环境中完成tensorflow 2.6.5版本的安装，并修改对应的激活命令。

如需使用动态扩容功能，进入“./cust_op/cust_op_by_addr”目录中。参考以下命令编译并安装动态扩容算子包。
```shell
bash run.sh
```

## 测试用例

### Python侧测试用例

运行Python测试用例所需依赖：

- pytest 7.1.1
- pytest-cov 4.1.0
- pytest-html

如需使用python测试用例，需要先安装上述依赖以及能够在tf1环境下进行源码编译，然后进入tests目录中。参考以下命令执行python侧测试用例：
```shell
bash run_python_dt.sh
```

### C++侧测试用例

运行C++侧测试用例所需依赖：

- [googletest 1.8.1](https://github.com/google/googletest/archive/refs/tags/release-1.8.1.zip)
- [emock 0.9.0](https://github.com/ez8-co/emock/archive/refs/tags/v0.9.0.zip)
- [pybind11 v2.10.3](https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip)
- [securec](https://github.com/huaweicloud/huaweicloud-sdk-c-obs/archive/refs/tags/v3.23.9.zip)

将googletest、emock、pybind11和securec的压缩包放在与mxRec代码同级的opensource目录下，并且将其分别更名为googletest-release-1.8.1.zip、
emock-0.9.0.zip、pybind11-2.10.3.zip、 huaweicloud-sdk-c-obs-3.23.9.zip。如果没有opensource目录，则需要在mxRec同级的目录下手动创建opensource目录，
然后将前述几个压缩包放在opensource目录下。

如需使用C++测试用例，需要按照上述描述准备需要的依赖，准备好之后，进入src目录中。参考以下命令执行C++测试用例：

tf1环境下使用如下命令：
```shell
bash test_ut.sh tf1
```

tf2环境下使用如下命令：
```shell
bash test_ut.sh tf2
```

## 使用指导

mxRec所支持的使用环境、功能特性、API接口与使用样例请参考[mxRec用户指南](https://www.hiascend.com/document/detail/zh/mind-sdk/60rc1/mxRec/mxrecug/mxrecug_0001.html)。

## 参考设计

mxRec框架基础镜像，基于TensorFlow 1.15.0、tensorflow2.6.5制作的基础镜像，安装mxRec后即可开始训练，以及样例使用介绍。

1. https://ascendhub.huawei.com/#/detail/mxrec-tf1

2. https://ascendhub.huawei.com/#/detail/mxrec-tf2
