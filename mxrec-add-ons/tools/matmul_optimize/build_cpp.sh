
CANN_DIR=${ASCEND_HOME_PATH}

g++ -std=c++17 -o test_aclnn_msprofop test_aclnn_msprofop.cpp -I. -I${CANN_DIR}/include -L${CANN_DIR}/lib64 -lascendcl -laclnn_math -lnnopbase -lopapi
