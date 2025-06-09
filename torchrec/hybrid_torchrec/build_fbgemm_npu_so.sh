cur_path="$(pwd)"
fbgemm_npu_so_path="../../mxrec_add_ons/rec_for_torch/torch_plugin/torch_library/2.6.0/common"
cd ${fbgemm_npu_so_path}
bash build_ops.sh
echo ${cur_path}
cp ./build/*.so ${cur_path}/hybrid_torchrec/