import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from dataclasses import dataclass


if tf.__version__.startswith("1"):
    from npu_bridge.hccl import hccl_ops
else:
    from npu_device.compat.v1.hccl import hccl_ops


@dataclass
class ModelConfig:
    user_range: int = 1e5
    item_range: int = 1e5

    user_feat_cnt: int = 16
    item_feat_cnt: int = 16

    user_table_dim: int = 128
    item_table_dim: int = 128

    batch_size: int = 1024
    learning_rate: float = 1e-3


def get_sess_config(dump_data=False, dump_path="./dump_output", dump_steps="0|1|2", use_deterministic=0):
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)

    session_config.gpu_options.allow_growth = True
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["mix_compile_mode"].b = False
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["min_group_size"].b = 1
    custom_op.parameter_map["HCCL_algorithm"].s = tf.compat.as_bytes("level0:pairwise;level1:pairwise")
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["iterations_per_loop"].i = 1
    if use_deterministic:
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("must_keep_origin_dtype")
        custom_op.parameter_map["deterministic"].i = 1
    else:
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["hcom_parallel"].b = False
    custom_op.parameter_map["op_precision_mode"].s = tf.compat.as_bytes("op_impl_mode.ini")
    custom_op.parameter_map["op_execute_timeout"].i = 2000
    if dump_data:
        """
        To see the details, please refer to the descriptions at official web site
        """
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(dump_steps)
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    return session_config
