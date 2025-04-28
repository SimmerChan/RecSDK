'''
REG_OP(SparseFwFFMGrad)
    .INPUT(weight, TensorType({DT_FLOAT,DT_DOUBLE}))
    .INPUT(fw_weight, TensorType({DT_FLOAT,DT_DOUBLE}))
    .INPUT(field, TensorType({DT_INT32}))
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(cross_mean_sum, TensorType({DT_FLOAT,DT_DOUBLE}))
    .INPUT(cross_mean_square_sum, TensorType({DT_FLOAT,DT_DOUBLE}))
    .INPUT(fw_field_map, TensorType({DT_INT32}))
    .INPUT(grad, TensorType({DT_FLOAT,DT_DOUBLE}))
    .OUTPUT(output, TensorType({DT_FLOAT,DT_DOUBLE}))
    .OUTPUT(fw_output, TensorType({DT_FLOAT,DT_DOUBLE}))
    .OP_END_FACTORY_REG(SparseFwFFMGrad)
'''

import numpy as np


def np_parallel_sparse_fw_ffm_grad(weight, fw_weight, field, index, cross_mean_sum, cross_mean_square_sum, fw_field_map, grad):
    # weight.shape [13, 2, 4]
    # cross_mean_sum.shape [2,2,3,4] [batch_size, field_num, fw_field_num, embedding_size]
    # output.shape = weight.shape
    # fw_output.sahpe = fw_weight.sahpe
    sample_feature_size = weight.shape[0]
    field_num = weight.shape[1]
    embedding_size = weight.shape[2]
    batch_size = grad.shape[0]
    fw_field_num = cross_mean_sum.shape[2]
    buffer_size = field_num * fw_field_num * embedding_size
    output_res = np.zeros(shape=weight.shape, dtype=weight.dtype)
    fw_output_res = np.zeros(shape=fw_weight.shape, dtype=fw_weight.dtype)
    fw_cross_mean_sum_grad = np.zeros(
        shape=[batch_size, field_num, fw_field_num, embedding_size], dtype=fw_weight.dtype)
    fw_cross_mean_square_sum_grad = np.zeros(
        shape=[batch_size, field_num, fw_field_num, embedding_size], dtype=fw_weight.dtype)
    for n in range(batch_size):
        if len(fw_weight.shape) > 1:
            fw_weight = fw_weight[n]
            fw_output_res = fw_output_res[n]
        fw_iter = 0
        for fw_field_1 in range(fw_field_num):
            multi_tag = False
            field_1 = fw_field_map[n][fw_field_1]
            if (field_1 >= 0):
                if (field_1 >= field_num):
                    multi_tag = True
                    field_1 = field_1 - field_num
                for fw_field_2 in range(fw_field_1):
                    field_2 = fw_field_map[n][fw_field_2]
                    if (field_2 >= 0):
                        if (field_2 >= field_num):
                            field_2 = field_2 - field_num
                        # for k in range(embedding_size):
                        grad_value = grad[n]
                        # index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_size + k
                        # index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_size + k

                        if (abs((1.0) + fw_weight[fw_iter]) > 0):
                            fw_cross_mean_sum_grad[n][field_1][fw_field_2] += grad_value * (
                                (1.) + fw_weight[fw_iter]) * cross_mean_sum[n][field_2][fw_field_1]
                            fw_cross_mean_sum_grad[n][field_2][fw_field_1] += grad_value * (
                                (1.) + fw_weight[fw_iter]) * cross_mean_sum[n][field_1][fw_field_2]

                        fw_output_res[fw_iter] += np.sum(grad_value * cross_mean_sum[n][field_1][fw_field_2] * cross_mean_sum[n][field_2][fw_field_1])
                    fw_iter += 1

                # for k in range(embedding_size):
                grad_value = grad[n]
                # index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_size + k
                if (abs((1.0) + fw_weight[fw_iter]) > 0):
                    fw_cross_mean_sum_grad[n][field_1][fw_field_1] += grad_value * (
                        (1.0) + fw_weight[fw_iter]) * cross_mean_sum[n][field_1][fw_field_1]
                    fw_cross_mean_square_sum_grad[n][field_1][fw_field_1] -= (
                        0.5) * grad_value * ((1.0) + fw_weight[fw_iter])

                if (multi_tag):
                    fw_output_res[fw_iter] += np.sum((0.5) * grad_value * (cross_mean_sum[n][field_1][fw_field_1] *
                                                                    cross_mean_sum[n][field_1][fw_field_1] - cross_mean_square_sum[n][field_1][fw_field_1]))
                fw_iter += 1
            else:
                fw_iter += fw_field_1 + 1
    for s in range(sample_feature_size):
        sample_id = index[s]
        field_1 = field[s][0] - 1
        fw_field_1 = field[s][1] - 1

        # T* cross_mean_sum_grad_ptr = fw_cross_mean_sum_grad.get() + sample_id * buffer_size;
        # T* cross_mean_square_sum_grad_ptr = fw_cross_mean_square_sum_grad.get() + sample_id * buffer_size;

        for field_2 in range(field_num):
            # for k in range(embedding_size):
                # int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_size + k;
            weight_value = weight[s][field_2]

            if (field_1 == field_2):
                output_res[s][field_2] += fw_cross_mean_sum_grad[sample_id][field_2][fw_field_1] + (
                    2.0) * fw_cross_mean_square_sum_grad[sample_id][field_2][fw_field_1] * weight_value
            else:
                output_res[s][field_2] += fw_cross_mean_sum_grad[sample_id][field_2][fw_field_1]

    return [output_res, fw_output_res]


def np_sparse_fw_ffm_grad(weight, fw_weight, field, index, cross_mean_sum, cross_mean_square_sum, fw_field_map, grad):
    # weight.shape [13, 2, 4]
    # cross_mean_sum.shape [2,2,3,4] [batch_size, field_num, fw_field_num, embedding_size]
    # output.shape = weight.shape
    # fw_output.sahpe = fw_weight.sahpe
    sample_feature_size = weight.shape[0]
    field_num = weight.shape[1]
    embedding_size = weight.shape[2]
    batch_size = grad.shape[0]
    fw_field_num = cross_mean_sum.shape[2]
    buffer_size = field_num * fw_field_num * embedding_size
    output_res = np.zeros(shape=weight.shape, dtype=weight.dtype)
    fw_output_res = np.zeros(shape=fw_weight.shape, dtype=fw_weight.dtype)
    fw_cross_mean_sum_grad = np.zeros(
        shape=[batch_size, field_num, fw_field_num, embedding_size], dtype=fw_weight.dtype)
    fw_cross_mean_square_sum_grad = np.zeros(
        shape=[batch_size, field_num, fw_field_num, embedding_size], dtype=fw_weight.dtype)
    for n in range(batch_size):
        if len(fw_weight.shape) > 1:
            fw_weight = fw_weight[n]
            fw_output_res = fw_output_res[n]
        fw_iter = 0
        for fw_field_1 in range(fw_field_num):
            multi_tag = False
            field_1 = fw_field_map[n][fw_field_1]
            if (field_1 >= 0):
                if (field_1 >= field_num):
                    multi_tag = True
                    field_1 = field_1 - field_num
                for fw_field_2 in range(fw_field_1):
                    field_2 = fw_field_map[n][fw_field_2]
                    if (field_2 >= 0):
                        if (field_2 >= field_num):
                            field_2 = field_2 - field_num
                        for k in range(embedding_size):
                            grad_value = grad[n][k]
                            # index_1 = (field_1 * fw_field_num + fw_field_2) * embedding_size + k
                            # index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_size + k

                            if (abs((1.0) + fw_weight[fw_iter]) > 0):
                                fw_cross_mean_sum_grad[n][field_1][fw_field_2][k] += grad_value * (
                                    (1.) + fw_weight[fw_iter]) * cross_mean_sum[n][field_2][fw_field_1][k]
                                fw_cross_mean_sum_grad[n][field_2][fw_field_1][k] += grad_value * (
                                    (1.) + fw_weight[fw_iter]) * cross_mean_sum[n][field_1][fw_field_2][k]

                            fw_output_res[fw_iter] += grad_value * cross_mean_sum[n][field_1][fw_field_2][k] * \
                                cross_mean_sum[n][field_2][fw_field_1][k]
                    fw_iter += 1

                for k in range(embedding_size):
                    grad_value = grad[n][k]
                    # index_1 = (field_1 * fw_field_num + fw_field_1) * embedding_size + k
                    if (abs((1.0) + fw_weight[fw_iter]) > 0):
                        fw_cross_mean_sum_grad[n][field_1][fw_field_1][k] += grad_value * (
                            (1.0) + fw_weight[fw_iter]) * cross_mean_sum[n][field_1][fw_field_1][k]
                        fw_cross_mean_square_sum_grad[n][field_1][fw_field_1][k] -= (
                            0.5) * grad_value * ((1.0) + fw_weight[fw_iter])

                    if (multi_tag):
                        fw_output_res[fw_iter] += (0.5) * grad_value * (cross_mean_sum[n][field_1][fw_field_1][k] *
                                                                        cross_mean_sum[n][field_1][fw_field_1][k] - cross_mean_square_sum[n][field_1][fw_field_1][k])

                fw_iter += 1
            else:
                fw_iter += fw_field_1 + 1
    for s in range(sample_feature_size):
        sample_id = index[s]
        field_1 = field[s][0] - 1
        fw_field_1 = field[s][1] - 1

        # T* cross_mean_sum_grad_ptr = fw_cross_mean_sum_grad.get() + sample_id * buffer_size;
        # T* cross_mean_square_sum_grad_ptr = fw_cross_mean_square_sum_grad.get() + sample_id * buffer_size;

        for field_2 in range(field_num):
            for k in range(embedding_size):
                # int32_t index_2 = (field_2 * fw_field_num + fw_field_1) * embedding_size + k;
                weight_value = weight[s][field_2][k]

                if (field_1 == field_2):
                    output_res[s][field_2][k] += fw_cross_mean_sum_grad[sample_id][field_2][fw_field_1][k] + (
                        2.0) * fw_cross_mean_square_sum_grad[sample_id][field_2][fw_field_1][k] * weight_value
                else:
                    output_res[s][field_2][k] += fw_cross_mean_sum_grad[sample_id][field_2][fw_field_1][k]

    return [output_res, fw_output_res]


def calc_expect_func(weight, fw_weight, field, index, cross_mean_sum, cross_mean_square_sum, fw_field_map, grad,
                     output, fw_output):
    res = np_sparse_fw_ffm_grad(weight["value"], fw_weight["value"], field["value"], index["value"],
                                cross_mean_sum["value"], cross_mean_square_sum["value"], fw_field_map["value"],
                                grad["value"])
    return res


if __name__ == "__main__":
    weight_data = [
        1., 1., 1., 1.,
        2., 2., 2., 2.,
        3., 3., 3., 3.,
        4., 4., 4., 4.,
        5., 5., 5., 5.,
        6., 6., 6., 6.,
        7., 7., 7., 7.,
        8., 8., 8., 8.,
        9., 9., 9., 9.,
        10., 10., 10., 10.,
        1., 1., 1., 1.,
        2., 2., 2., 2.,
        1., 1., 1., 1.,
        2., 2., 2., 2.,
        3., 3., 3., 3.,
        4., 4., 4., 4.,
        5., 5., 5., 5.,
        6., 6., 6., 6.,
        3., 3., 3., 3.,
        4., 4., 4., 4.,
        5., 5., 5., 5.,
        6., 6., 6., 6.,
        7., 7., 7., 7.,
        8., 8., 8., 8.,
        9., 9., 9., 9.,
        10., 10., 10., 10.,
    ]
    weight = np.array(weight_data).astype(np.float32).reshape(13, 2, 4)
    fw_weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                         dtype=np.float32).reshape((6,))
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
    ], dtype=np.int32).reshape((13, 2))
    index = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                     1, 1, 1, 2]).astype(np.int32)
    cross_mean_sum = np.array([
        0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3,
        0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6,
        0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3,
        0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9,
    ], dtype=np.float32).reshape((2, 2, 3, 4))
    cross_mean_square_sum = np.array([
        0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3,
        0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6,
        0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3,
        0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9,
    ], dtype=np.float32).reshape((2, 2, 3, 4))
    fw_field_map = np.array([
        0, 1, 1,
        0, 1, 1,
    ], dtype=np.int32).reshape((2, 3))
    grad = np.array([
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8
    ], dtype=np.float32).reshape((2, 4))
    res0,res1 = np_sparse_fw_ffm_grad(weight, fw_weight, field, index,
                                cross_mean_sum, cross_mean_square_sum, fw_field_map, grad)
    res_1_0,res_1_1 = np_parallel_sparse_fw_ffm_grad(weight, fw_weight, field, index,
                                cross_mean_sum, cross_mean_square_sum, fw_field_map, grad)
    print(np.allclose(res0, res_1_0,0.001,0.001))
    print(np.allclose(res1, res_1_1,0.001,0.001))
