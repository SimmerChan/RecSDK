import tensorflow as tf


# Create the tensors
def batch_norm(x, is_training, decay=0.99, epsilon=1e-3):
    """
    实现 Batch Normalization。

    Args:
        x: 输入张量 (例如：[batch_size, height, width, channels])。
        is_training: 一个布尔值，指示当前是训练阶段还是推理阶段。
        decay: 移动平均的衰减率。
        epsilon: 为了数值稳定性而添加到方差上的一个小常数。

    Returns:
        经过 Batch Normalization 处理后的张量。
    """

    return tf.nn.batch_normalization(x, moving_mean, moving_variance, offset, scale, epsilon)


# Enable XLA by wrapping the function with tf.function
@tf.function(jit_compile=True)
def compute_operations(input_tensor):
    # Step 1: 先 reshape 为 (128, 192, 1, 256)
    reshaped_tensor = tf.reshape(input_tensor, (128, 192, 1, 256))

    # Step 2: BatchNormalization (假设使用标准的 BN 操作)
    # 注意：BatchNormalization 在训练模式下会有不同的行为，因此需要指定训练模式
    batch_norm_tensor = batch_norm(reshaped_tensor, is_training=False)

    # Step 3: 再次 reshape 为 (128, 192, 2, 256)
    reshaped_bn_tensor = tf.reshape(batch_norm_tensor, (128, 192, 256))

    # Step 4: 创建一个常量张量，形状为 (128, 192, 2, 256)
    constant_tensor = tf.constant(1.0, shape=(128, 192, 256))

    # Step 5: 做逐元素相减操作
    sub_output = tf.subtract(reshaped_bn_tensor, constant_tensor)

    # 最后，reshape 为 (128, 192, 256)
    final_result = tf.reshape(sub_output, (128, 192, 256))

    return final_result


input_tensor = tf.random.normal((128, 192, 256))
# reshaped_tensor = tf.reshape(input_tensor, (128, 192, 1, 256))
# batch_norm_tensor = batch_norm(reshaped_tensor, is_training=False)
scale = tf.Variable(tf.ones([256]))
offset = tf.Variable(tf.zeros([256]))

# 使用指数移动平均更新全局均值和方差
moving_mean = tf.Variable(tf.zeros([256]), trainable=False)
moving_variance = tf.Variable(tf.ones([256]), trainable=False)

# Call the function
final_output = compute_operations(input_tensor)

# Print the final output shape
print("Final output shape:", final_output.shape)
