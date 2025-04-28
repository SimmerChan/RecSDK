import tensorflow as tf


def sparse_fm(weight_tensor, index_tensor, bias_dim):
    batch_size = index_tensor[index_tensor.shape[0] - 1]
    weight_tensor_square = tf.square(weight_tensor)
    cross_mean_sum_tensor = tf.math.unsorted_segment_sum(weight_tensor, index_tensor[:-1], num_segments=batch_size)
    cross_mean_square_sum_tensor = tf.math.unsorted_segment_sum(weight_tensor_square, index_tensor[:-1],
                                                                num_segments=batch_size)
    output_tensor_without_bias = 0.5 * (tf.square(cross_mean_sum_tensor) - cross_mean_square_sum_tensor)
    output_tensor = tf.concat([cross_mean_sum_tensor[:, :bias_dim], output_tensor_without_bias[:, bias_dim:]], axis=1)
    return output_tensor, cross_mean_sum_tensor, cross_mean_square_sum_tensor


if __name__ == "__main__":
    w = tf.constant(2, shape=[10, 7])
    index_tensor = tf.constant([
        0, 1, 0, 0, 0,
        0, 1, 0, 1, 0, 3
    ])
    bias_dim = 1
    sparse_fm_op = sparse_fm(w, index_tensor, bias_dim)
    with tf.Session() as sess:
        print(sess.run(sparse_fm_op))
