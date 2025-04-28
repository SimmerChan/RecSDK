from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import logging
import tensorflow as tf
import numpy as np
import math
# from npu_device.compat.v1 import *

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.flags.DEFINE_string("local_log_dir", "output/train_logs.txt", "Log file path")
FLAGS = tf.compat.v1.flags.FLAGS

atol = 0.001
rtol = 0.001


def config(excute_type):
    if excute_type == 'npu':
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_data_pre_proc"].b = True
        custom_op.parameter_map["mix_compile_mode"].b = True
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["min_group_size"].b = 1

    elif excute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    return session_config


def fw_weight_symmetric(fw_weight):
    cal_shape = [1, fw_weight.shape[0]] if len(fw_weight.shape) == 1 else fw_weight.shape
    n = math.floor(math.sqrt(2 * int(cal_shape[1])))
    result_index = []
    for b in range(int(cal_shape[0])):
        for i in range(n):
            for j in range(n):
                ii, jj = (i, j) if i > j else (j, i)
                result_index.append(b * int(cal_shape[1]) + ii * (ii + 1) // 2 + jj)
    symmetric_metric = tf.gather(tf.reshape(fw_weight, [-1]), result_index)
    out_shape = (n, n) if len(fw_weight.shape) == 1 else (fw_weight.shape[0], n, n)
    symmetric_metric = tf.reshape(symmetric_metric, out_shape)
    return symmetric_metric


def sparse_fw_fm(weight_tensor, fw_weight, field_tensor, index_tensor):
    fw_weight = fw_weight_symmetric(fw_weight)
    batch_size = index_tensor[index_tensor.shape[0] - 1]
    fw_field_num = int(fw_weight.shape[1])
    field_num = int(weight_tensor.shape[1])
    batch_index = index_tensor[:-1]
    fw_field_index = field_tensor[:, 1] - 1
    weight_tensor_square = tf.square(weight_tensor)

    batch_field_fwfield_index = batch_index * field_num * fw_field_num + fw_field_index
    batch_field_fwfield_index = tf.concat([tf.expand_dims(batch_field_fwfield_index, axis=-1)] * int(field_num),
                                          axis=-1)
    batch_field_fwfield_index = tf.reshape(batch_field_fwfield_index, (-1,))
    field_index = tf.constant(np.arange(field_num, dtype=np.int64))
    field_index = tf.tile(field_index, weight_tensor.shape[0:1])
    field_index = tf.reshape(field_index, batch_field_fwfield_index.shape)
    batch_field_fwfield_index += field_index * fw_field_num
    weight_tensor = tf.reshape(weight_tensor, (-1, int(weight_tensor.shape[-1])))
    weight_tensor_square = tf.reshape(weight_tensor_square, (-1, int(weight_tensor_square.shape[-1])))

    cross_mean_sum_tensor = tf.math.unsorted_segment_sum(weight_tensor,
                                                         batch_field_fwfield_index,
                                                         num_segments=batch_size * field_num * fw_field_num)
    cross_mean_square_sum_tensor = tf.math.unsorted_segment_sum(weight_tensor_square,
                                                                batch_field_fwfield_index,
                                                                num_segments=batch_size * field_num * fw_field_num)
    cross_mean_sum_tensor = tf.reshape(cross_mean_sum_tensor, (batch_size, field_num, fw_field_num, -1))
    cross_mean_square_sum_tensor = tf.reshape(cross_mean_square_sum_tensor, (batch_size, field_num, fw_field_num, -1))

    batch_fw_field_index = batch_index * fw_field_num + fw_field_index
    fw_field_map_res = tf.math.unsorted_segment_min(field_tensor[:, 0],
                                                    batch_fw_field_index,
                                                    num_segments=batch_size * fw_field_num)
    fw_field_map_res = tf.math.floormod(tf.clip_by_value(fw_field_map_res, 0, field_num + 1), field_num + 1) - 1
    batch_fw_field_to_field_ind = tf.reshape(fw_field_map_res, (batch_size, fw_field_num)) + 1
    zero_tensor = tf.zeros((batch_size, 1, fw_field_num, cross_mean_sum_tensor.shape[-1]))
    cross_mean_sum_tensor_padded_zero_dim = tf.concat([zero_tensor, cross_mean_sum_tensor], axis=1)
    fw_field_cross_mean_sum_tensor = tf.gather(cross_mean_sum_tensor_padded_zero_dim,
                                               batch_fw_field_to_field_ind,
                                               axis=1,
                                               batch_dims=1)
    fw_field_cross_mean_sum_tensor_t = tf.transpose(fw_field_cross_mean_sum_tensor, perm=[0, 2, 1, 3])
    if len(fw_weight.shape) == 2:
        fw_weight_add_one = tf.expand_dims((fw_weight + 1), axis=0)
    else:
        fw_weight_add_one = fw_weight + 1
    fw_weight_add_one = tf.expand_dims(fw_weight_add_one, axis=-1)
    ffm_fw_field_cross_mean_sum_tensor = (fw_field_cross_mean_sum_tensor *
                                          fw_field_cross_mean_sum_tensor_t) * fw_weight_add_one

    if len(fw_weight.shape) == 2:
        diag_weight = tf.reduce_sum((fw_weight + 1) * tf.eye(fw_field_num), -1)
    else:
        diag_weight = tf.reduce_sum((fw_weight + 1) * tf.expand_dims(tf.eye(fw_field_num), axis=0), -1)
    cross_mean_square_sum_tensor_padded_zero_dim = tf.concat([zero_tensor, cross_mean_square_sum_tensor], axis=1)
    fw_field_diag_square_sum_tensor = tf.gather(tf.transpose(cross_mean_square_sum_tensor_padded_zero_dim,
                                                             perm=(0, 2, 1, 3)),
                                                tf.expand_dims(batch_fw_field_to_field_ind, axis=-1),
                                                axis=2,
                                                batch_dims=2)
    diag_weight = tf.expand_dims(diag_weight, axis=0) if len(fw_weight.shape) == 2 else diag_weight
    fw_field_diag_square_sum_tensor = tf.squeeze(fw_field_diag_square_sum_tensor) * tf.expand_dims(diag_weight, axis=-1)
    output_res = 0.5 * (tf.reduce_sum(ffm_fw_field_cross_mean_sum_tensor, [1, 2]) -
                        tf.reduce_sum(fw_field_diag_square_sum_tensor, 1))

    fw_field_multi_tag = tf.math.unsorted_segment_sum(
        tf.ones_like(field_tensor[:, 0]), batch_fw_field_index, num_segments=batch_size * fw_field_num) - 1
    fw_field_multi_tag = tf.clip_by_value(fw_field_multi_tag, 0, 1)
    fw_field_map_res = fw_field_map_res + fw_field_multi_tag * field_num
    fw_field_map_res = tf.reshape(fw_field_map_res, (batch_size, fw_field_num))
    return output_res, cross_mean_sum_tensor, cross_mean_square_sum_tensor, fw_field_map_res


def testcase(weight, fw_weight, field, index):
    weight_tensor = tf.convert_to_tensor(weight)
    fw_weight_tensor = tf.convert_to_tensor(fw_weight)
    field_tensor = tf.convert_to_tensor(field)
    index_tensor = tf.convert_to_tensor(index)

    output = sparse_fw_fm(weight_tensor, fw_weight_tensor, field_tensor, index_tensor)
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session(config=config('cpu')) as session:
        session.run(init)
        out1 = session.run(output)
    out2 = np_sparse_fw_ffm(weight, fw_weight, field, index)
    print('====================================')
    print("output_tf:", out1)
    print("output_numpy:", out2)
    cmp_result = np.allclose(out1[0], out2[0], atol, rtol)
    print("compare result - output: ", cmp_result)
    print('====================================')


def np_sparse_fw_ffm(weight, fw_weight, field, index):
    field_num = weight.shape[1]
    embedding_size = weight.shape[2]
    batch_size = 2
    feature_dim = embedding_size
    fw_field_num = math.floor(math.sqrt(2 * fw_weight.shape[-1]))
    output_res = np.zeros(shape=[batch_size, feature_dim], dtype=weight.dtype)
    cross_mean_sum_res = np.zeros(shape=[batch_size, field_num, fw_field_num, embedding_size], dtype=fw_weight.dtype)
    cross_mean_square_sum_res = np.zeros(shape=[batch_size, field_num, fw_field_num, embedding_size],
                                         dtype=fw_weight.dtype)
    fw_field_map_res = np.ones(shape=[batch_size, fw_field_num], dtype=field.dtype) * (-1)
    for n in range(batch_size):
        if len(fw_weight.shape) > 1:
            fw_weight_temp = fw_weight[n]
        else:
            fw_weight_temp = fw_weight
        index_range = [s for s in range(len(index)) if index[s] == n]
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
                    cross_mean_square_sum_res[n][field_2][fw_field_1][
                        k] += weight[s][field_2][k] * weight[s][field_2][k]
            if fw_field_map_res[n][fw_field_1] != -1.0:
                if fw_field_map_res[n][fw_field_1] < field_num:
                    fw_field_map_res[n][fw_field_1] += field_num
            else:
                fw_field_map_res[n][fw_field_1] = field[s][0] - 1
        fw_iter = 0
        for fw_field_1 in range(fw_field_num):
            field_1 = fw_field_map_res[n][fw_field_1]
            if field_1 >= 0:
                multi_tag = False
                if field_1 >= field_num:
                    field_1 -= field_num
                    multi_tag = True
                for fw_field_2 in range(fw_field_1):
                    field_2 = fw_field_map_res[n][fw_field_2]
                    if field_2 >= 0:
                        if field_2 >= field_num:
                            field_2 -= field_num
                        for k in range(embedding_size):
                            output_res[n][k] += cross_mean_sum_res[n][field_1][fw_field_2][k] * cross_mean_sum_res[n][
                                field_2][fw_field_1][k] * (1.0 + fw_weight_temp[fw_iter])
                    fw_iter += 1
                if multi_tag:
                    for k in range(embedding_size):
                        output_res[n][k] += 0.5 * (cross_mean_sum_res[n][field_1][fw_field_1][k] *
                                                   cross_mean_sum_res[n][field_1][fw_field_1][k] -
                                                   cross_mean_square_sum_res[n][field_1][fw_field_1][k]) * (
                                                       1.0 + fw_weight_temp[fw_iter])
                fw_iter += 1
            else:
                fw_iter += fw_field_1 + 1
    return [output_res, cross_mean_sum_res, cross_mean_square_sum_res, fw_field_map_res]


def main(unused_argv):
    weight_data = [
        1., 1., 1., 1., 2., 2., 2., 2.,
        3., 3., 3., 3., 4., 4., 4., 4.,
        5., 5., 5., 5., 6., 6., 6., 6.,
        7., 7., 7., 7., 8., 8., 8., 8.,
        9., 9., 9., 9., 10., 10., 10., 10.,
        1., 1., 1., 1., 2., 2., 2., 2.,
        1., 1., 1., 1., 2., 2., 2., 2.,
        3., 3., 3., 3., 4., 4., 4., 4.,
        5., 5., 5., 5., 6., 6., 6., 6.,
        3., 3., 3., 3., 4., 4., 4., 4.,
        5., 5., 5., 5., 6., 6., 6., 6.,
        7., 7., 7., 7., 8., 8., 8., 8.,
        9., 9., 9., 9., 10., 10., 10., 10.,
    ]
    weight = np.array(weight_data).astype(np.float32).reshape(13, 2, 4)
    fw_weight = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32).reshape((2, 6))
    field = np.array([
        1, 1,
        1, 1,
        1, 1,
        2, 2,
        2, 3,
        2, 2,
        1, 1,
        1, 1,
        1, 1,
        2, 2,
        2, 2,
        2, 2,
        2, 3
    ], dtype=np.int64).reshape((13, 2))
    index = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2]).astype(np.int64)
    
    testcase(weight, fw_weight, field, index)


if __name__ == "__main__":
    tf.compat.v1.app.run()