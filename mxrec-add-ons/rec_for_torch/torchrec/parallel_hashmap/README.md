# hybrid_torchrec hashmap替换样例
## 说明
以引用folly为例，执行bash.sh编译,在build路径下会生成libparallel_hashmap.so。在运行hybrid_torchrec时加入该环境变量，就会使用该hashmap
```shell
export PARALLEL_HASH_MAP_SO="XXXXX/build/libparallel_hashmap.so"
```


