import tensorflow as tf


@tf.function(jit_compile=True) # 启用 XLA JIT 编译
def process_tensors(tensors, clip_min, clip_max):
    # Step 1: 对每个 tensor 应用 tf.clip_by_value
    clipped_tensors = [tf.clip_by_value(t, clip_min, clip_max) for t in tensors]

    # Step 2: 使用 tf.concat 将所有 tensor 沿 axis=1 拼接
    output_tensor = tf.concat(clipped_tensors, axis=1)

    return output_tensor


# 创建 201 个形状为 (128, 1) 的随机 tensor
tensors = [tf.random.normal((128, 1)) for _ in range(201)]

# 设置 clip 的最小值和最大值
clip_min = -1.0
clip_max = 1.0

# 调用函数处理这些 tensors
output_tensor = process_tensors(tensors, clip_min, clip_max)

# 打印输出张量的形状，确保它是 (128, 201)
print("Output shape:", output_tensor.shape)  # 应该输出 (128, 201)
