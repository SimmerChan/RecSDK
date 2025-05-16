import tensorflow as tf

# Create the tensors
input1 = tf.random.normal((128, 150, 148))  # Shape (128, 150, 148)
input2 = tf.random.normal((128, 150, 148))  # Shape (128, 150, 148)
input3 = tf.random.normal((128, 150, 1))


# Enable XLA by wrapping the function with tf.function
@tf.function(jit_compile=True)
def compute_operations(input1, input2, input3):
    add_output = input1 + input2
    reduce_output = tf.reduce_mean(add_output, axis=-1, keepdims=True)
    sub_output = add_output - reduce_output
    square_output = tf.square(add_output - reduce_output)
    reduce2_output = tf.reduce_mean(square_output, axis=-1, keepdims=True)
    add2_output = reduce2_output + input3
    rsqrt_output = tf.math.rsqrt(add2_output)
    mul_output = sub_output * rsqrt_output
    const_tensor1 = tf.constant(1.0, shape=(148,))
    const_tensor2 = tf.constant(2.0, shape=(148,))
    mul2_output = mul_output * const_tensor1
    add3_output = mul2_output + const_tensor2
    return add3_output


# Call the function
final_output = compute_operations(input1, input2, input3)

# Print the final output shape
print("Final output shape:", final_output.shape)
