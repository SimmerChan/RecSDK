
#!/bin/bash

# 设置环境变量
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=0

if [ ! $ASCEND_HOME_DIR ]; then
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_DIR=$HOME/Ascend/ascend-toolkit/latest
    else
        export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

export DDK_PATH=$ASCEND_HOME_DIR
arch=$(uname -m)
export NPU_HOST_LIB=$ASCEND_HOME_DIR/${arch}-linux/lib64

WORKER_DIR=$(pwd)
PARENT_PATH=$(dirname $(pwd))
cd "$PARENT_PATH"

# 编译算子
build_op()
{
    CURRENT_DIR=$(pwd)
    bash ./creat.sh
}

op_test() {
    # 1. 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*
    rm ./input/*.bin
    rm ./output/*.bin

    # 2. 生成输入数据和真值数据
    python3 scripts/gen_data.py
    if [ $? -ne 0 ]; then
        echo "ERROR: generate input data failed!"
        return 1
    fi
    echo "INFO: generate input data success!"


    # 3. 编译acl可执行文件
    rm -rf build; mkdir -p build; cd build
    cmake ../src
    if [ $? -ne 0 ]; then
        echo "ERROR: cmake failed!"
        return 1
    fi
    echo "INFO: cmake success!"
    make
    if [ $? -ne 0 ]; then
        echo "ERROR: make failed!"
        return 1
    fi
    echo "INFO: make success!"

    # 4. 运行可执行文件
    cd ../output
    echo "INFO: execute op!"
    ./$exec_op

    if [ $? -ne 0 ]; then
        echo "ERROR: acl executable run failed! please check your project!"
        return 1
    fi
    echo "INFO: acl executable run success!"

    # 5. 比较真值文件
    cd ../
    ret=`python3 scripts/verify_result.py`
    echo $ret
    if [ "x$ret" == "xtest pass" ]; then
        echo ""
        echo "#####################################"
        echo "INFO: you have passed the Precision!"
        echo "#####################################"
        echo ""
    fi
}

to_snake_case(){
    local input_str=$1
    echo "$input_str" | sed -E 's/([A-Z])/_\1/g' | sed -E 's/^_//g' | tr 'A-Z' 'a-z'
}

prefix='aclnn_'
exec_prefix='execute_'
exec_suffix='_op'
test_all_op() {
    for dir in "$PARENT_PATH"/*; do
        cd "$PARENT_PATH"
        if [ -d "$dir" ]; then
            dir_name=$(basename "$dir")
            if [ "$dir_name" != "OpTest" ]; then
                echo "Entering directory: $dir_name"
                cd "$dir_name"
                test_dir_name="$prefix$(to_snake_case "$dir_name")"
                exec_op="$exec_prefix$(to_snake_case "$dir_name")$exec_suffix"
                echo "test_dir_name: $test_dir_name"
                build_op
                cd "$WORKER_DIR"
                cd "$test_dir_name"
                op_test
            fi
        fi
    done
}

TEST_OP_COUNT=$(($(find "$PARENT_PATH" -maxdepth 1 -type d | wc -l) - 2))
echo -e "\033[32m==============Start to test op, total op count is: $TEST_OP_COUNT=====================\033[0m"

if [ $# -ge 1 ]; then
    work_dir_name=$1
    cd "$work_dir_name"
    test_dir_name="$prefix$(to_snake_case "$work_dir_name")"
    exec_op="$exec_prefix$(to_snake_case "$work_dir_name")$exec_suffix"
    build_op
    cd "$WORKER_DIR"
    cd "$test_dir_name"
    op_test
else
    test_all_op
fi






