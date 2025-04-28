import numpy as np
import math

def np_sparse_fw_ffm(weight, fw_weight, field, index):
    sample_feature_size = weight.shape[0]
    field_num = weight.shape[1]
    embedding_size = weight.shape[2]
    batch_size = 2
    feature_dim = embedding_size
    fw_field_num = math.floor(math.sqrt(2 * fw_weight.shape[-1]));
    buffer_size = field_num * fw_field_num * embedding_size
    output_res = np.zeros(shape=[batch_size, feature_dim], dtype=weight.dtype)
    cross_mean_sum_res = np.zeros(shape=[batch_size, field_num, fw_field_num, embedding_size], dtype=fw_weight.dtype)
    cross_mean_square_sum_res = np.zeros(shape=[batch_size, field_num, fw_field_num, embedding_size], dtype=fw_weight.dtype)
    fw_field_map_res = np.ones(shape=[batch_size, fw_field_num], dtype=field.dtype) * (-1)
    print("fw_field_map_res", fw_field_map_res)
    for n in range(batch_size):
        if len(fw_weight.shape) > 1:
            fw_weight = fw_weight[n]
        index_range = [s for s in range(len(index)) if index[s]==n ]
        # print(f"n:{n}, index_range:{index_range}")
        for s in index_range:                 
              field_1 = field[s][0] - 1
              if field_1 < 0 or field_1 > field_num:
                  continue
              fw_field_1 = field[s][1] - 1
              if fw_field_1 < 0 or fw_field_1 > fw_field_num:
                  continue

              for field_2 in range(field_num):
                  for k in range(embedding_size):
                      cross_mean_sum_res[n][field_2][fw_field_1][k] += weight[s][field_2][k]
                      cross_mean_square_sum_res[n][field_2][fw_field_1][k] += weight[s][field_2][k] * weight[s][field_2][k]
              if fw_field_map_res[n][fw_field_1] != -1.0:
                  print(f"fw_field_map_res[{n}][{fw_field_1}]", fw_field_map_res[n][fw_field_1])
                  if fw_field_map_res[n][fw_field_1] < field_num:
                      fw_field_map_res[n][fw_field_1] += field_num
              else:
                  print(f"fw_field_map_res[{n}][{fw_field_1}]", fw_field_map_res[n][fw_field_1])
                  fw_field_map_res[n][fw_field_1] = field[s][0] - 1
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
    return [output_res, cross_mean_sum_res, cross_mean_square_sum_res, fw_field_map_res]

def calc_expect_func(weight, fw_weight, field, index, output, cross_mean_sum, cross_mean_square_sum, fw_field_map):
    res = np_sparse_fw_ffm(weight["value"], fw_weight["value"], field["value"], index["value"])
    return res

if __name__ == "__main__":
    weight =  {'value': np.array([[[ 1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.]],

       [[ 3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.]],

       [[ 5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.]],

       [[ 7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.]],

       [[ 9.,  9.,  9.,  9.],
        [10., 10., 10., 10.]],

       [[ 1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.]],

       [[ 1.,  1.,  1.,  1.],
        [ 2.,  2.,  2.,  2.]],

       [[ 3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.]],

       [[ 5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.]],

       [[ 3.,  3.,  3.,  3.],
        [ 4.,  4.,  4.,  4.]],

       [[ 5.,  5.,  5.,  5.],
        [ 6.,  6.,  6.,  6.]],

       [[ 7.,  7.,  7.,  7.],
        [ 8.,  8.,  8.,  8.]],

       [[ 9.,  9.,  9.,  9.],
        [10., 10., 10., 10.]]], dtype=np.float), 'dtype': 'float', 'shape': [13, 2, 4], 'format': 'ND'}
    fw_weight =  {'value': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float), 'dtype': 'float', 'shape': [6], 'format': 'ND'}
    field={'value': np.array([[1, 1],
       [1, 1],
       [1, 1],
       [2, 2],
       [2, 3],
       [2, 2],
       [1, 1],
       [1, 1],
       [1, 1],
       [2, 2],
       [2, 2],
       [2, 2],
       [2, 3]], dtype=np.int32), 'dtype': 'int32', 'shape': [13, 2], 'format': 'ND'}
    index =  {'value': np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2], dtype=np.int32), 'dtype': 'int32', 'shape': [14], 'format': 'ND'}
    calc_expect_func(weight,fw_weight, field, index, None, None, None, None)