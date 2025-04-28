from onnx import TensorProto
from onnx.helper import make_tensor, make_tensor_value_info, make_attribute, make_model, make_node, make_graph
from onnx.checker import check_model
import onnx

casual = 1
batch_size = 1
max_seq_len = 768
num_heads = 4
attention_dim = 32
siluScale = 1 / max_seq_len
maskType = 3
layout = "normal"

q = make_tensor_value_info("q", TensorProto.FLOAT16, [batch_size, max_seq_len, num_heads, attention_dim])
k = make_tensor_value_info("k", TensorProto.FLOAT16, [batch_size, max_seq_len, num_heads, attention_dim])
v = make_tensor_value_info("v", TensorProto.FLOAT16, [batch_size, max_seq_len, num_heads, attention_dim])
attn_bias = make_tensor_value_info("attn_bias", TensorProto.FLOAT16, [batch_size, num_heads, max_seq_len, max_seq_len])
mask = make_tensor_value_info("mask", TensorProto.FLOAT16, [batch_size, 1, max_seq_len, max_seq_len])

attn_output = make_tensor_value_info("attn_output", TensorProto.FLOAT16, [batch_size, max_seq_len, num_heads, attention_dim])

node = make_node("HstuDenseForward", ["q", "k", "v", "mask", "attn_bias"], ["attn_output"])
node.attribute.append(make_attribute("siluScale", siluScale))
node.attribute.append(make_attribute("maxSeqLen", max_seq_len))
node.attribute.append(make_attribute("maskType", maskType))
node.attribute.append(make_attribute("casual", casual))
node.attribute.append(make_attribute("layout", layout))

graph = make_graph([node], "hstu", [q, k, v, mask, attn_bias], [attn_output])

model_def = make_model(graph, producer_name="hstu-onnx")

model_def.opset_import[0].version = 11

onnx.save(model_def, "hstu.onnx")

print('The model is:\n{}'.format(model_def))