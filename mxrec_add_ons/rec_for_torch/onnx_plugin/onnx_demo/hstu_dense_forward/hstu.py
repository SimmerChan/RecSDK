import onnx
from onnx import TensorProto
from onnx.helper import make_tensor_value_info, make_attribute, make_model, make_node, make_graph

CASUAL = 1
BATCH_SIZE = 1
MAX_SEQ_LEN = 768
NUM_HEADS = 4
ATTENTION_DIM = 32
SILUSCALE = 1 / MAX_SEQ_LEN
MASKTYPE = 3
LAYOUT = "normal"

q = make_tensor_value_info("q", TensorProto.FLOAT16, [BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, ATTENTION_DIM])
k = make_tensor_value_info("k", TensorProto.FLOAT16, [BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, ATTENTION_DIM])
v = make_tensor_value_info("v", TensorProto.FLOAT16, [BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, ATTENTION_DIM])
attn_bias = make_tensor_value_info("attn_bias", TensorProto.FLOAT16, [BATCH_SIZE, NUM_HEADS, MAX_SEQ_LEN, MAX_SEQ_LEN])
mask = make_tensor_value_info("mask", TensorProto.FLOAT16, [BATCH_SIZE, 1, MAX_SEQ_LEN, MAX_SEQ_LEN])

attn_output = make_tensor_value_info("attn_output", TensorProto.FLOAT16,
                                     [BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, ATTENTION_DIM])

node = make_node("HstuDenseForward", ["q", "k", "v", "mask", "attn_bias"], ["attn_output"])
node.attribute.append(make_attribute("siluScale", SILUSCALE))
node.attribute.append(make_attribute("maxSeqLen", MAX_SEQ_LEN))
node.attribute.append(make_attribute("maskType", MASKTYPE))
node.attribute.append(make_attribute("casual", CASUAL))
node.attribute.append(make_attribute("layout", LAYOUT))

graph = make_graph([node], "hstu", [q, k, v, mask, attn_bias], [attn_output])

model_def = make_model(graph, producer_name="hstu-onnx")

model_def.opset_import[0].version = 11

onnx.save(model_def, "hstu.onnx")
