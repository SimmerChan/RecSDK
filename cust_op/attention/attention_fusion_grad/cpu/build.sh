 
 
rm -rf build
cmake -S . -B build -G "Unix Makefiles" -Dbinary_file=attention_fusion_grad_cpu -Dtestcase=attention_fusion_grad -Dtensor_data_file=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/ascendebug/features/compiler/cpu_compiler/cpu_tensor_data.cpp -Dkernel_cpp=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/attention_fusion_grad/op_impl/ai_core/tbe/attention_fusion_grad_impl/dynamic/attention_fusion_grad.cpp -Dkernel_file=/home/whf/attentionLayer/attention-layer/attention_fusion_grad/cpu/AttentionFusionGrad/cpu/src/_gen_kernel_attention_fusion_grad.cpp -Dproduct_type=Ascend910B2 -Dinstall_path=/usr/local/Ascend/ascend-toolkit/ -Dcanndev_repo=None -Ddata_definition_file=/home/whf/attentionLayer/attention-layer/attention_fusion_grad/cpu/AttentionFusionGrad/cpu/src/data_definition.txt -Dkernel_name=attention_fusion_grad -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=on
cmake --build build
 
 
 