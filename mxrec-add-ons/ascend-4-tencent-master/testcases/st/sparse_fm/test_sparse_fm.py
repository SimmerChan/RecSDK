import numpy as np


def fm(weight_splited, embedding):
    one_batch_output = np.zeros((embedding), dtype=float)
    one_cross_mean_sum = np.zeros((embedding), dtype=float)
    one_cross_mean_square_sum = np.zeros((embedding), dtype=float)
    for each_row in weight_splited:
        one_cross_mean_sum += each_row
        one_cross_mean_square_sum += np.power(each_row, 2)
    one_batch_output = 0.5 * (np.power(one_cross_mean_sum, 2) - one_cross_mean_square_sum)
    one_batch_output[0] = one_cross_mean_sum[0]

    return one_batch_output, one_cross_mean_sum, one_cross_mean_square_sum

def calc_expect_func(weight, index, output, cross_mean_sum, cross_mean_square_sum):
    bias_dim = 1
    weight_value = weight["value"]
    index_value = index["value"]
    tmp_batch = int(index_value[0])
    batch = int(index_value[-1])
    embedding = weight_value.shape[1]

    cross_mean_sum = np.zeros((batch, embedding), dtype=float)
    cross_mean_square_sum = np.zeros((batch, embedding), dtype=float)
    output = np.zeros((batch, embedding), dtype=float)
    output_row = 0
    pre_split_point = 0
    for i in range(1, index_value.shape[0]):
        if index_value[i] != tmp_batch or i == index_value.shape[0]-1:
            split_point = i
            weight_splited = np.split(weight, [pre_split_point, split_point])
            weight_splited_pre = weight_splited[1]
            one_batch_output, one_cross_mean_sum, one_cross_mean_square_sum = \
                fm(weight_splited_pre, embedding)
            output[output_row] = one_batch_output
            cross_mean_sum[output_row] = one_cross_mean_sum
            cross_mean_square_sum[output_row] = one_cross_mean_square_sum
            pre_split_point = split_point
            tmp_batch = index_value[i]
            output_row += 1

    return output, cross_mean_sum, cross_mean_square_sum


if __name__ == "__main__":
    weight_data = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [-0.9, -1.0, -1.1, -1.2],
        [1.3, 1.4, 1.5, 1.6],
        [1.7, 1.8, 1.9, 2.0],
        [2.1, 2.2, 2.3, 2.4],
        [-2.5, -2.6, -2.7, -2.8],
        [2.9, 3.0, 3.1, 3.2],
        [3.3, 3.4, 3.5, 3.6],
        [3.7, 3.8, 3.9, 4.0],
        [-4.1, -4.2, -4.3, -4.4]
    ]
    weight = {"value":np.array(weight_data).astype(np.float32).reshape(11,4),
              "dtype": "float", "shape": [11, 4], "format": "ND"}
    index_data = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 5]
    index = {"value":np.array(index_data).astype(np.int32),
             "dtype": "int", "shape": [12], "format": "ND"}
    output = None
    cross_mean_sum = None
    cross_mean_square_sum = None
    output, cross_mean_sum, cross_mean_square_sum = calc_expect_func(weight, index, output,
                                                                     cross_mean_sum,
                                                                     cross_mean_square_sum)