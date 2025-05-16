import tensorflow as tf

@tf.function(jit_compile=True)  # 启用 XLA JIT 编译
def process_tensors(input0, input1):
    # Step 1: 执行逐元素减法（sub）
    sub_output = tf.subtract(input0, input1, name="sub_output")

    # Step 2: 执行逐元素乘法（mul）
    mul_output = tf.multiply(input0, input1, name="mul_output")

    # Step 3: 在 axis=2 上进行 concat
    output = tf.concat([input0, input1, sub_output, mul_output], axis=2)

    return output

# 创建两个形状为 (128, 128, 144) 的输入张量
input0 = tf.random.normal((128, 128, 144))
input1 = tf.random.normal((128, 128, 144))

# 调用处理函数
output_tensor = process_tensors(input0, input1)

# 打印输出张量的形状，确保它是 (128, 128, 576)
print("Output shape:", output_tensor.shape)  # 应该输出 (128, 128, 576)
