import argparse
import multiprocessing

DataType = [
    'UNDEFINED',
    'FLOAT',
    'UINT8',
    'INT8',
    'UINT16',
    'INT16',
    'INT32',
    'INT64',
    'STRING',
    'BOOL',
    'FLOAT16',
    'DOUBLE',
    'UINT32',
    'UINT64',
    'COMPLEX64',
    'COMPLEX128',
    'BFLOAT16',
]


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        type=int,
                        default=0,
                        help="0:onnx , default:0")
    parser.add_argument("--onnx", help="onnx path")
    parser.add_argument(
        "--optype",
        default="NULL",
        help="search bottom and top optype of optype from Network")
    parser.add_argument("--optype_location", default="A", choices=["A", "S", "M", "E"],  \
    help="A:All,S:Start,M:Middle,E:End; Search for operators before or after the search operator with the search operator as ends or starts.")
    parser.add_argument("--optype_before_n_layers",
                        default=1,
                        type=int,
                        help="search n layers of optype from Network")
    parser.add_argument("--optype_after_n_layers",
                        default=1,
                        type=int,
                        help="search n layers of optype from Network")
    parser.add_argument("--onnxoptimizer", default=0, type=int, help="onnxoptimizer on or off")
    parser.add_argument("--debug", default=0, type=int, help="debug on or off")
    #parser.add_argument("--processes", default=int(multiprocessing.cpu_count()/3), type=int, help="multi processes freeze pb")
    args = parser.parse_args()
    return args

import onnx
import onnxoptimizer
def optimize(model: onnx.ModelProto ) -> onnx.ModelProto:
    """
    :model参数: 待优化的ONXX模型.
    :return: 优化之后的ONNX模型.
    简化之前, 使用这个方法产生会在'forward_all'用到的ValueInfo
    简化之后，使用这个方法去折叠前一步产生的常量到initializer中并且消除没被使用的常量
    """
    onnx.checker.check_model(model)
    onnx.helper.strip_doc_string(model)
    optimizers_list = [
        'eliminate_deadend',
        'eliminate_nop_dropout',
        'eliminate_nop_cast',
        'eliminate_nop_monotone_argmax', 'eliminate_nop_pad',
        'extract_constant_to_initializer', 'eliminate_unused_initializer',
        'eliminate_nop_transpose',
        'eliminate_nop_flatten', 'eliminate_identity',
        'fuse_add_bias_into_conv',
        'fuse_consecutive_concats',
        'fuse_consecutive_log_softmax',
        'fuse_consecutive_reduce_unsqueeze', 'fuse_consecutive_squeezes',
        'fuse_consecutive_transposes', 'fuse_matmul_add_bias_into_gemm',
        'fuse_pad_into_conv', 'fuse_transpose_into_gemm', 'eliminate_duplicate_initializer'
    ]
    model = onnxoptimizer.optimize(model, optimizers_list, fixed_point=True)
    onnx.checker.check_model(model)
    return model


