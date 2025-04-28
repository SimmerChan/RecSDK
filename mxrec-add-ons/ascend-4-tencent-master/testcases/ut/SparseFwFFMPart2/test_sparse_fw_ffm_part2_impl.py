from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("SparseFwFFMPart2", "impl.dynamic.sparse_fw_ffm_part2", "sparse_fw_ffm_part2")

def calc_expect_func(fw_weight, cross_mean_sum_res, cross_mean_square_sum_res, fw_field_map_res, output_res):
    fw_weight = fw_weight.get("value").reshape(-1)
    cross_mean_sum_res = cross_mean_sum_res.get("value")
    cross_mean_square_sum_res = cross_mean_square_sum_res.get("value")
    fw_field_map_res = fw_field_map_res.get("value")
    batch_size = 12 #动态shape
    field_num = 2
    fw_field_num = 75
    embedding_size = 64
    output_res = np.zeros((batch_size, embedding_size), dtype=np.float32)
    # for n in range(batch_size):
    for n in range(batch_size):
      fw_iter = 0
      for fw_field_1 in range(fw_field_num):
          field_1 = fw_field_map_res[n][fw_field_1]
          if  field_1 >= 0:
              multi_tag = False  
              if field_1 >= field_num:
                  field_1 -= field_num
                  multi_tag = True
              for fw_field_2 in range(fw_field_1):
                  field_2 = fw_field_map_res[n][fw_field_2]
                  if field_2 >=0:
                      if field_2 >= field_num:
                          field_2 -= field_num
                      for k in range(embedding_size):
                          output_res[n][k] += cross_mean_sum_res[n][field_1][fw_field_2][k] * cross_mean_sum_res[n][field_2][fw_field_1][k] * (1.0 + fw_weight[fw_iter]) 
                  fw_iter += 1
              if multi_tag:
                  for k in range(embedding_size):
                      output_res[n][k] += 0.5* (cross_mean_sum_res[n][field_1][fw_field_1][k] * cross_mean_sum_res[n][field_1][fw_field_1][k] - cross_mean_square_sum_res[n][field_1][fw_field_1][k]) * (1.0 + fw_weight[fw_iter]) 
              fw_iter += 1
          else:  
              fw_iter += fw_field_1 + 1
    return output_res

batch_size =  12
part_num = 2
fw_field_num = 75
feature_dim = 64
fw_weight_shape = (int(fw_field_num*(fw_field_num+1)/2), )
cross_mean_sum_shape = (batch_size, part_num, fw_field_num, feature_dim)
fw_field_map_shape = (batch_size, fw_field_num)

data_np_fw_weight = np.random.random(fw_weight_shape).astype(np.float32)
data_np_cross_mean_sum = np.random.uniform(0, 10, size=cross_mean_sum_shape).astype(np.float32)
data_np_cross_mean_square_sum = np.random.uniform(0, 10, size=cross_mean_sum_shape).astype(np.float32)
data_np_fw_field_map = np.random.randint(1, 3, batch_size * fw_field_num).reshape(batch_size, fw_field_num).astype(np.int32)

# precision
ut_case.add_precision_case(["Ascend910B"], {
    "params": [
        {"shape": (2850,), "dtype": "float32", "format": "ND",
         "range": [(2850, 2850)],
         "ori_shape": (2850,), "run_ori_shape":(2850,), "ori_format": "ND", "param_type": "input", "run_shape": (2850,), "value": data_np_fw_weight},
        {"shape": (-1, 2, 75, 64), "dtype": "float32", "format": "ND",
         "range": [(1, 40000), (2, 2), (75, 75), (64, 64)],
         "ori_shape": (12, 2, 75, 64), "run_ori_shape":(12, 2, 75, 6), "ori_format": "ND", "param_type": "input", "run_shape": (12, 2, 75, 64), "value": data_np_cross_mean_sum},
        {"shape": (-1, 2, 75, 64), "dtype": "float32", "format": "ND",
         "range": [(1, 40000), (2, 2), (75, 75), (64, 64)],
         "ori_shape": (12, 2, 75, 64),  "run_ori_shape":(12, 2, 75, 6),"ori_format": "ND", "param_type": "input", "run_shape": (12, 2, 75, 64), "value": data_np_cross_mean_square_sum},
        {"shape": (-1, 75), "dtype": "int32", "format": "ND",
         "range": [(1, 40000), (75, 75)],
         "ori_shape": (12, 75), "run_ori_shape":(12, 75), "ori_format": "ND", "param_type": "input", "run_shape": (12, 75), "value": data_np_fw_field_map},
        {"shape": (-1, 64), "dtype": "float32", "format": "ND",
         "range": [(1, 40000), (64, 64)],
         "ori_shape": (12, 64), "ori_format": "ND", "param_type": "output", "run_shape": (12, 64)},
    ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})


if __name__ == "__main__":
    from tbe.common.platform import set_current_compile_soc_info
    set_current_compile_soc_info("Ascend910B")
    ut_case.run("Ascend910B", simulator_mode="pv",
                simulator_lib_path="/usr/local/Ascend/CANN-1.84/x86_64-linux/simulator") 