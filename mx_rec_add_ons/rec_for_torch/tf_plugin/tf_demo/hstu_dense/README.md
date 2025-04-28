## 使用tensorflow的调用的方式调用算子工程
该样例脚本基于tensorflow2.6.5,python3.7.5运行

### 环境依赖
在[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/cfe24a13b14e465ebbcf816ad6f73c9e)下载mxrec-tf2镜像，并创建容器；

## 运行样例算子
### 1.编译算子工程
运行此样例前，请参考[部署算子工程](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0072.html)完成算子部署。
- 例如：
    ```bash
    ./custom_opp_<target os>_<target architecture>.run --install-path=<path>
    ```
### 2.编译tensorflow自定义算子库
    ```bash
    cd tf_ops
    sh build_ops.sh
    ```

### 2.tensorflow调用的方式调用样例运行
  - 样例执行

    样例执行过程中会自动生成测试数据，然后运行tensorflow样例，最后打印运行结果。详见test_hstu_dense_backward_demo.py
    ```bash
    cd ../
    python3 hstu_dense_backward_demo.py
    ```
## 更新说明
| 时间         | 更新事项     |
|------------| ------------ |
| 2024/11/08 | 新增本readme |