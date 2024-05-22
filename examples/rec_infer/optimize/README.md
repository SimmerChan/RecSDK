# 链接ARM的optimized-routines库
在memcpy等接口占比较大的模型中，有性能收益，源码路径为(https://github.com/ARM-software/optimized-routines/tree/v23.01)
> unzip optimized-routines-23.01.zip<br>
> cd optimized-routines-23.01<br>

在源码基础上，修改代码，修改脚本如下：
> for m in memcmp memcpy memset memmove memrchr strcpy strchrnul strchr strcmp stpcpy strncmp strnlen strrchr; do<br>
>     for f in $(grep __${m}_aarch64 * -r |awk -F ':' '{print $1}'); do<br>
>         sed_str1="__${m}_aarch64"<br>
>         sed_str2="${m}"<br>
>         sed -i 's!'${sed_str1}'!'${sed_str2}'!g' $f<br>
>     done<br>
> done<br>

> make ARCH=aarch64 -j 8<br>
> cp build/lib/libstringlib.so /usr/local/lib/<br>

在编译tensorflow serving时链接libstringlib.so，相关修改代码参考0001-Performance-optimization-referrence
运行server时，需要配置环境变量：
> export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH<br>



# 链接jemalloc库
源码下载链接: https://github.com/jemalloc/jemalloc/archive/refs/tags/5.3.0.tar.gz
编译安装命令如下：
> tar -xzvf jemalloc-5.3.0.tar.gz<br>
> cd jemalloc-5.3.0<br>
> ./autogen.sh<br>
> make -j 8<br>
> make install<br>

安装完成后，默认安装在/usr/local/lib/，在编译tensorflow serving时链接libjemalloc.so，相关修改代码参考0001-Performance-optimization-referrence
运行server时，需要配置环境变量：
> export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH<br>

# gRPC配置优化
增加NUM_CQS,MIN_POLLERS,MAX_POLLERS这三个配置项的配置，在多线程请求推理场景可以提升性能
配置项参考gRPC官网(https://grpc.github.io/grpc/cpp/classgrpc_1_1_server_builder.html)
具体修改参考0001-Performance-optimization-referrence，配置最优值根据不同模型和机器可能有所不同；
