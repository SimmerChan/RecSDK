# 第一阶段：基础环境配置（快速完成）
FROM debian:12 AS base

WORKDIR /tmp
COPY ./version.info ./
COPY ./ascend_install.info ./

# 1. 配置证书和Debian源
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN echo "deb https://mirrors.huaweicloud.com/debian/ bookworm main contrib non-free" > ./sources.list && \
    echo "deb-src https://mirrors.huaweicloud.com/debian/ bookworm main contrib non-free" >> ./sources.list && \
    echo "deb https://mirrors.huaweicloud.com/debian-security/ bookworm-security main contrib non-free" >> ./sources.list && \
    echo "deb-src https://mirrors.huaweicloud.com/debian-security/ bookworm-security main contrib non-free" >> ./sources.list && \
    echo "deb https://mirrors.huaweicloud.com/debian/ bookworm-updates main contrib non-free" >> ./sources.list && \
    echo "deb-src https://mirrors.huaweicloud.com/debian/ bookworm-updates main contrib non-free" >> ./sources.list


# 第二阶段：安装编译工具（中等耗时）
FROM base AS build-tools

# 2. 安装编译工具链（保留全部依赖）
RUN mv /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list.d/debian.sources.bak && \
    cp -f ./sources.list /etc/apt/ && \
    apt-get update && \
    apt-get -y install net-tools make wget zlib1g-dev bzip2 libbz2-dev libssl-dev libncurses5-dev libncursesw5-dev openssh-client \
    openssh-server libsqlite3-dev libopenmpi-dev libreadline-dev tk-dev libgdbm-dev libdb-dev libpcap-dev liblzma-dev  \
    libffi-dev libhdf5-dev patch pciutils lcov vim dos2unix g++ automake libtool autoconf m4 perl git e2fsprogs && \
    apt-get clean && \
    rm -rf /var/cache/apt/


# 第三阶段：安装CMake（中等耗时）
FROM build-tools AS cmake-installer

# 3. 安装cmake-3.20.6
RUN wget https://cmake.org/files/v3.20/cmake-3.20.6.tar.gz && \
    . /etc/profile && gcc -v && tar -zxf cmake-3.20.6.tar.gz && \
    cd cmake-3.20.6 && \
    ./bootstrap && make -j 32 && make install && cd .. && \
    rm -rf cmake-3.20.6*


# 第四阶段：安装Python（中等耗时）
FROM cmake-installer AS python-installer

# 4. 安装Python 3.11.9
RUN wget https://mirrors.huaweicloud.com/python/3.11.9/Python-3.11.9.tgz && \
    . /etc/profile && gcc -v && tar -xvf Python-3.11.9.tgz && \
    cd Python-3.11.9 && \
    mkdir -p build && cd build && \
    ../configure --enable-shared --prefix=/usr/local/python3.11.9 && \
    make -j && make install && \
    cd ../../ && rm -rf Python-3.11.9* && \
    ldconfig

ENV PATH=$PATH:/usr/local/python3.11.9/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.11.9/lib

RUN ln -s /usr/local/python3.11.9/bin/python3 /usr/bin/python && \
    ln -s /usr/local/python3.11.9/bin/pip3 /usr/bin/pip

# 配置Python源
RUN mkdir ~/.pip && touch ~/.pip/pip.conf && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "trusted-host=mirrors.aliyun.com" >> ~/.pip/pip.conf && \
    echo "index-url=https://mirrors.aliyun.com/pypi/simple/" >> ~/.pip/pip.conf && \
    echo "timeout=200" >> ~/.pip/pip.conf


# 第五阶段：安装Python依赖（耗时较长）
FROM python-installer AS python-deps

# 5. 安装Python包（保留全部依赖）
RUN wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl && \
    wget https://download.pytorch.org/whl/cpu/fbgemm_gpu-1.1.0%2Bcpu-cp311-cp311-manylinux2014_x86_64.whl

RUN pip3 install -U pip && \
    pip3 install numpy==1.25.0 && \
    pip3 install wheel && \
    pip3 install decorator && \
    pip3 install sympy==1.13.3 && \
    pip3 install cffi==1.17.1 && \
    pip3 install expecttest && \
    pip3 install pybind11 && \
    pip3 install pyyaml && \
    pip3 install psutil && \
    pip3 install pathlib2 && \
    pip3 install pandas==2.2.3 && \
    pip3 install grpcio && \
    pip3 install grpcio-tools==1.67.0 && \
    pip3 install protobuf==5.28.2 && \
    pip3 install scipy==1.14.1 && \
    pip3 install requests && \
    pip3 install scikit-learn && \
    pip3 install easydict && \
    pip3 install attrs && \
    pip3 install pytest && \
    pip3 install pytest-cov && \
    pip3 install pytest-html && \
    pip3 install Cython && \
    pip3 install absl-py && \
    pip3 install gin-config && \
    pip3 install torch-2.6.0+cpu-*.whl && \
    pip3 install fbgemm_gpu-1.1.0+cpu-*.whl && \
    pip3 install torch-npu==2.6.0.rc1 && \
    pip3 install tensorboard==2.18.0 && \
    pip3 install black && \
    pip3 install cmake && \
    pip3 install hypothesis==6.70.1 && \
    pip3 install iopath && \
    pip3 install pyre-extensions && \
    pip3 install scikit-build && \
    pip3 install torchmetrics==1.0.3 && \
    pip3 install torchx && \
    pip3 install tqdm && \
    pip3 install usort && \
    rm -rf /root/.cache/pip


# 第六阶段：安装CANN工具包（耗时最长）
FROM python-deps AS cann-installer

# 6. 复制CANN安装包
COPY ./Ascend-cann-toolkit_*.run ./
COPY ./Ascend-cann-kernels-*.run ./

# 7. 设置驱动路径环境变量
ARG ASCEND_BASE=/usr/local/Ascend
ENV LD_LIBRARY_PATH=$ASCEND_BASE/driver/lib64:$ASCEND_BASE/driver/lib64/common:$ASCEND_BASE/driver/lib64/driver:$LD_LIBRARY_PATH

# 8. 安装CANN工具包
ARG TOOLKIT_PKG=Ascend-cann-toolkit*.run
ARG KERNEL_PKG=Ascend-cann-kernels*.run
RUN rm -rf $ASCEND_BASE/ascend-toolkit

RUN umask 0027 && \
   chmod +x $TOOLKIT_PKG && \
   bash $TOOLKIT_PKG --quiet --install --install-path=$ASCEND_BASE && \
   . $ASCEND_BASE/ascend-toolkit/set_env.sh && \
   chmod +x $KERNEL_PKG && \
   bash $KERNEL_PKG --quiet --install && \
   rm -rf /root/.cache/pip && \
   rm -f $TOOLKIT_PKG && rm -f $KERNEL_PKG


# 最终阶段：保留全部内容
FROM debian:12

WORKDIR /app

# 复制所有必要的文件
COPY --from=base /tmp/version.info ./
COPY --from=base /tmp/ascend_install.info ./
COPY --from=build-tools /etc/apt/sources.list /etc/apt/
COPY --from=python-installer /usr/local/python3.11.9 /usr/local/python3.11.9
COPY --from=python-installer /usr/bin/python /usr/bin/python
COPY --from=python-installer /usr/bin/pip /usr/bin/pip
COPY --from=python-installer /root/.pip /root/.pip
COPY --from=cann-installer /usr/local/Ascend /usr/local/Ascend

# 设置环境变量
ENV PATH=$PATH:/usr/local/python3.11.9/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.11.9/lib:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver

# 验证安装
RUN python --version && \
    pip --version && \
    ls /usr/local/Ascend/ascend-toolkit