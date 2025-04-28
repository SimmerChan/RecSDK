import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_context
from tbe.common.platform import set_current_compile_soc_info
import numpy as np


class Constant:
    """
    The class for constant
    """
    # tiling arg num
    TILING_ARG_NUM = 4
    # MAX INT 64
    MAX_INT64 = 2**64 - 1
    # MAX BURST LEN
    MAX_BURST_LEN = 65535
    # size of float32
    SIZE_OF_FLOAT = 4
    # one block size
    BLOCK_SIZE = 32
    # max  float32 (1, 64) float16 (1, 128) float64 (1, 32)
    MAX_FLOAT32_REPEAT_TIME = 64


class SparseFwFFMPart2:

    def __init__(self,
                 fw_weight,
                 cross_mean_sum,
                 cross_mean_square_sum,
                 fw_field_map,
                 output,
                 kernel_name="sparse_fw_ffm_part2"):
        self.debug = False
        self.tik_instance = tik.Tik(disable_debug=(not self.debug))
        self.tik_profiling = tik.Dprofile()

        if self.debug: 
            self.fw_weight_gm = self.tik_instance.Tensor(fw_weight.get('dtype'), (Constant.MAX_INT64,),
                                                        name="fw_weight_gm",
                                                        scope=tik.scope_gm)
            self.cross_mean_sum_gm = self.tik_instance.Tensor(cross_mean_sum.get('dtype'), (Constant.MAX_INT64,),
                                                              name="cross_mean_sum_gm",
                                                              scope=tik.scope_gm)
            self.cross_mean_square_sum_gm = self.tik_instance.Tensor(cross_mean_square_sum.get('dtype'), (Constant.MAX_INT64,),
                                                                    name="cross_mean_square_sum_gm",
                                                                    scope=tik.scope_gm)
            self.fw_field_map_gm = self.tik_instance.Tensor(fw_field_map.get('dtype'), (Constant.MAX_INT64,),
                                                            name="fw_field_map_gm",
                                                            scope=tik.scope_gm)
            self.output_gm = self.tik_instance.Tensor(output.get('dtype'), (Constant.MAX_INT64,),
                                                      name="output",
                                                      scope=tik.scope_gm)
        else:
            self.fw_weight_gm = self.tik_instance.Tensor(fw_weight.get('dtype'), (Constant.MAX_INT64,),
                                                        name="fw_weight_gm",
                                                        scope=tik.scope_gm)
            self.cross_mean_sum_gm = self.tik_instance.Tensor(cross_mean_sum.get('dtype'), (Constant.MAX_INT64,),
                                                              name="cross_mean_sum_gm",
                                                              scope=tik.scope_gm)
            self.cross_mean_square_sum_gm = self.tik_instance.Tensor(cross_mean_square_sum.get('dtype'), (Constant.MAX_INT64,),
                                                                    name="cross_mean_square_sum",
                                                                    scope=tik.scope_gm)
            self.fw_field_map_gm = self.tik_instance.Tensor("int32", (Constant.MAX_INT64,),
                                                            name="fw_field_map",
                                                            scope=tik.scope_gm)
            self.output_gm = self.tik_instance.Tensor("float32", (Constant.MAX_INT64,),
                                                      name="output",
                                                      scope=tik.scope_gm)

            self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                      name="tiling_gm",
                                                      scope=tik.scope_gm)
          

        self.fw_weight_shape = fw_weight.get("shape")
        self.cross_mean_sum_shape = cross_mean_sum.get("shape")
        self.part_num, self.fw_field_num, self.embedding_size = self.cross_mean_sum_shape[1], self.cross_mean_sum_shape[
            2], self.cross_mean_sum_shape[3]

        self.kernel_name = kernel_name

    def _tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from tensor_move tiling

        Returns
        -------
        None
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_ub",
                                                  scope=tik.scope_ubuf)

        if self.debug:
            self.tiling_act_core_num = self.tik_instance.Scalar("int64", name="tiling_act_core_num", init_value=1)
            self.tiling_batch_each_core = self.tik_instance.Scalar("int64", name="tiling_batch_each_core", init_value=1)
            self.tiling_batch_last_core = self.tik_instance.Scalar("int64", name="tiling_batch_last_core", init_value=1)

            self.tiling_act_core_num.set_as(48)
            self.tiling_batch_each_core.set_as(21)
            self.tiling_batch_last_core.set_as(self.tiling_ub[2])
        else:
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)

            self.tiling_act_core_num = self.tik_instance.Scalar("int64", name="tiling_act_core_num", init_value=1)
            self.tiling_batch_each_core = self.tik_instance.Scalar("int64", name="tiling_batch_each_core", init_value=1)
            self.tiling_batch_last_core = self.tik_instance.Scalar("int64", name="tiling_batch_last_core", init_value=1)

            self.tiling_act_core_num.set_as(self.tiling_ub[0])
            self.tiling_batch_each_core.set_as(self.tiling_ub[1])
            self.tiling_batch_last_core.set_as(self.tiling_ub[2])

    def compute_each_batch(self, batch_index):
        """
        compute each batch

        Parameters
        ----------
        batch_index: scalar, batch index

        Returns
        -------
        None
        """
        # fw_weight_num = fw_weight_num * (fw_weight_num + 1) / 2
        cross_mean_sum_num = self.part_num * self.fw_field_num * self.embedding_size
        fw_weight_num = self.fw_weight_shape[0]
        # (1,  fw_field_num*(fw_field_num+1)/2 )
        self.fw_weight_ub = self.tik_instance.Tensor("float32", (fw_weight_num,), scope=tik.scope_ubuf, name="fw_weight_ub")
        self.cross_mean_sum_ub = self.tik_instance.Tensor("float32", (cross_mean_sum_num,), scope=tik.scope_ubuf, name="cross_mean_sum_ub")
        self.step_one_temp_ub = self.tik_instance.Tensor("float32", (cross_mean_sum_num,), scope=tik.scope_ubuf, name="step_one_temp_ub")
        self.cross_mean_square_sum_ub = self.tik_instance.Tensor("float32", (cross_mean_sum_num,), scope=tik.scope_ubuf, name="cross_mean_square_sum_ub")
        self.fw_field_map_ub = self.tik_instance.Tensor("int32", (self.fw_field_num,), scope=tik.scope_ubuf, name="fw_field_map_ub")
        self.output_ub = self.tik_instance.Tensor("float32", (self.embedding_size,), scope=tik.scope_ubuf, name="output_ub")

        # move all data from fw_weight_num to ub
        fw_weight_burst_len = int(fw_weight_num * Constant.SIZE_OF_FLOAT + Constant.BLOCK_SIZE - 1) // Constant.BLOCK_SIZE
        self.tik_instance.data_move(self.fw_weight_ub, self.fw_weight_gm, 0, 1, fw_weight_burst_len, 0, 0)

        # move all data from cross_mean_sum_num to ub
        cross_mean_sum_burst_len = (cross_mean_sum_num * Constant.SIZE_OF_FLOAT + Constant.BLOCK_SIZE -
                                    1) // Constant.BLOCK_SIZE
        cross_mean_sum_index = batch_index * cross_mean_sum_num
        self.tik_instance.data_move(self.cross_mean_sum_ub, self.cross_mean_sum_gm[cross_mean_sum_index], 0, 1, cross_mean_sum_burst_len, 0,
                                    0)
        self.tik_instance.data_move(self.cross_mean_square_sum_ub, self.cross_mean_square_sum_gm[cross_mean_sum_index], 0, 1, cross_mean_sum_burst_len, 0,
                                    0)
        fw_field_map_burst_len = (self.fw_field_num * Constant.SIZE_OF_FLOAT + Constant.BLOCK_SIZE -
                                  1) // Constant.BLOCK_SIZE
        fw_field_map_index = batch_index * self.fw_field_num
        self.tik_instance.data_move(self.fw_field_map_ub, self.fw_field_map_gm[fw_field_map_index], 0, 1, fw_field_map_burst_len, 0, 0)

        field_1 = self.tik_instance.Scalar("int32", name="field_1", init_value=0)
        field_2 = self.tik_instance.Scalar("int32", name="field_2", init_value=0)
        scalar_one = self.tik_instance.Scalar(dtype="float32", init_value=1.0)
        repeat_times = fw_weight_num // Constant.MAX_FLOAT32_REPEAT_TIME
        last_repeat_num = fw_weight_num % Constant.MAX_FLOAT32_REPEAT_TIME
        # fw_weight = 1.0 + fw_weight
        self.tik_instance.vec_adds(Constant.MAX_FLOAT32_REPEAT_TIME, self.fw_weight_ub, self.fw_weight_ub, scalar_one,
                                  repeat_times, 8, 8)
        if last_repeat_num > 0:
            last_index = repeat_times * Constant.MAX_FLOAT32_REPEAT_TIME
            self.tik_instance.vec_adds(last_repeat_num, self.fw_weight_ub[last_index], self.fw_weight_ub[last_index], scalar_one, 1, 8, 8)
        
        self.tik_instance.vec_dup(self.embedding_size, self.output_ub, 0.0, 1, 8)
        with self.tik_instance.for_range(0, self.fw_field_num) as fw_field_1:
            field_1.set_as(self.fw_field_map_ub[fw_field_1])
            with self.tik_instance.if_scope(field_1 > self.part_num - 1):
                field_1_update = field_1 - self.part_num
                with self.tik_instance.for_range(0, fw_field_1) as fw_field_2:
                    field_2.set_as(self.fw_field_map_ub[fw_field_2])
                    with self.tik_instance.if_scope(field_2 > self.part_num - 1):
                        field_2_update = field_2 - self.part_num
                        index_1 = field_1_update * self.fw_field_num * self.embedding_size + fw_field_2 * self.embedding_size
                        index_2 = field_2_update * self.fw_field_num * self.embedding_size + fw_field_1 * self.embedding_size
                        self.tik_instance.vec_mul(self.embedding_size, self.step_one_temp_ub,
                                                  self.cross_mean_sum_ub[index_1], self.cross_mean_sum_ub[index_2], 1,
                                                  8, 8, 8)

                        # calculate fw_weight[fw_iter] + 1
                        scalar_fw_weight_add_one = self.tik_instance.Scalar(dtype="float32", init_value=0.0)
                        fw_iter = fw_field_1 * self.fw_field_num + fw_field_2 - (149 - fw_field_1) * fw_field_1 / 2
                        scalar_fw_weight_add_one.set_as(self.fw_weight_ub[fw_iter])
                        
                        self.tik_instance.vec_muls(self.embedding_size, self.step_one_temp_ub,
                                                   self.step_one_temp_ub, scalar_fw_weight_add_one, 1,
                                                   8, 8)
                        self.tik_instance.vec_add(self.embedding_size, self.output_ub, self.output_ub,
                                                  self.step_one_temp_ub, 1, 8, 8, 8)
                    with self.tik_instance.elif_scope(field_2 > -1):
                        index_1 = field_1_update * self.fw_field_num * self.embedding_size + fw_field_2 * self.embedding_size
                        index_2 = field_2 * self.fw_field_num * self.embedding_size + fw_field_1 * self.embedding_size
                        self.tik_instance.vec_mul(self.embedding_size, self.step_one_temp_ub,
                                                  self.cross_mean_sum_ub[index_1], self.cross_mean_sum_ub[index_2], 1,
                                                  8, 8, 8)

                        # calculate fw_weight[fw_iter] + 1
                        scalar_fw_weight_add_one = self.tik_instance.Scalar(dtype="float32", init_value=0.0)
                        fw_iter = fw_field_1 * self.fw_field_num + fw_field_2 - (149 - fw_field_1) * fw_field_1 / 2
                        scalar_fw_weight_add_one.set_as(self.fw_weight_ub[fw_iter])
                        
                        self.tik_instance.vec_muls(self.embedding_size, self.step_one_temp_ub,
                                                   self.step_one_temp_ub, scalar_fw_weight_add_one, 1,
                                                   8, 8)
                        self.tik_instance.vec_add(self.embedding_size, self.output_ub, self.output_ub,
                                                  self.step_one_temp_ub, 1, 8, 8, 8)
                index_3 = field_1_update * self.fw_field_num * self.embedding_size + fw_field_1 * self.embedding_size
                self.tik_instance.vec_mul(self.embedding_size,  self.step_one_temp_ub, self.cross_mean_sum_ub[index_3],  self.cross_mean_sum_ub[index_3], 1, 8, 8, 8 )
                self.tik_instance.vec_sub(self.embedding_size,  self.step_one_temp_ub, self.step_one_temp_ub,  self.cross_mean_square_sum_ub[index_3], 1, 8, 8, 8 )
                
                fw_iter = fw_field_1 * (fw_field_1 + 3) // 2
                scalar_fw_weight_add_one = self.tik_instance.Scalar(dtype="float32", init_value=0.0)
                scalar_fw_weight_add_one.set_as(self.fw_weight_ub[fw_iter])
                scalar_fw_weight_add_one = scalar_fw_weight_add_one * 0.5
                self.tik_instance.vec_muls(self.embedding_size, self.step_one_temp_ub, self.step_one_temp_ub, scalar_fw_weight_add_one, 1, 8, 8)
                
                self.tik_instance.vec_add(self.embedding_size, self.output_ub, self.output_ub, self.step_one_temp_ub, 1, 8, 8, 8)
            with self.tik_instance.elif_scope(field_1 > -1):
                with self.tik_instance.for_range(0, fw_field_1) as fw_field_2:
                    field_2 = self.tik_instance.Scalar(dtype="int32", init_value=0.0)
                    field_2.set_as(self.fw_field_map_ub[fw_field_2])
                    with self.tik_instance.if_scope(field_2 > self.part_num - 1):
                        field_2_update = field_2 - self.part_num
                        index_1 = field_1 * self.fw_field_num * self.embedding_size + fw_field_2 * self.embedding_size
                        index_2 = field_2_update * self.fw_field_num * self.embedding_size + fw_field_1 * self.embedding_size
                        self.tik_instance.vec_mul(self.embedding_size, self.step_one_temp_ub,
                                                  self.cross_mean_sum_ub[index_1], self.cross_mean_sum_ub[index_2], 1,
                                                  8, 8, 8)

                        # calculate fw_weight[fw_iter] + 1
                        scalar_fw_weight_add_one = self.tik_instance.Scalar(dtype="float32", init_value=0.0)
                        fw_iter = fw_field_1 * self.fw_field_num + fw_field_2 - (149 - fw_field_1) * fw_field_1 / 2
                        scalar_fw_weight_add_one.set_as(self.fw_weight_ub[fw_iter])
                        
                        self.tik_instance.vec_muls(self.embedding_size, self.step_one_temp_ub,
                                                   self.step_one_temp_ub, scalar_fw_weight_add_one, 1,
                                                   8, 8)
                        self.tik_instance.vec_add(self.embedding_size, self.output_ub, self.output_ub,
                                                  self.step_one_temp_ub, 1, 8, 8, 8)
                    with self.tik_instance.elif_scope(field_2 > -1):
                        index_1 = field_1 * self.fw_field_num * self.embedding_size + fw_field_2 * self.embedding_size
                        index_2 = field_2 * self.fw_field_num * self.embedding_size + fw_field_1 * self.embedding_size
                        self.tik_instance.vec_mul(self.embedding_size, self.step_one_temp_ub,
                                                  self.cross_mean_sum_ub[index_1], self.cross_mean_sum_ub[index_2], 1,
                                                  8, 8, 8)

                        # calculate fw_weight[fw_iter] + 1
                        scalar_fw_weight_add_one = self.tik_instance.Scalar(dtype="float32", init_value=0.0)
                        fw_iter = fw_field_1 * self.fw_field_num + fw_field_2 - (149 - fw_field_1) * fw_field_1 / 2
                        scalar_fw_weight_add_one.set_as(self.fw_weight_ub[fw_iter])
                        
                        self.tik_instance.vec_muls(self.embedding_size, self.step_one_temp_ub,
                                                   self.step_one_temp_ub, scalar_fw_weight_add_one, 1,
                                                   8, 8)
                      
                        self.tik_instance.vec_add(self.embedding_size, self.output_ub, self.output_ub,
                                                  self.step_one_temp_ub, 1, 8, 8, 8)

        burst_len = self.embedding_size * Constant.SIZE_OF_FLOAT // Constant.BLOCK_SIZE
        self.tik_instance.data_move(self.output_gm[batch_index * self.embedding_size], self.output_ub, 0, 1, burst_len, 0, 0)  
    def sparse_fw_ffm_part2_tik(self):
        self._tiling_args()
        
        with self.tik_instance.for_range(0, self.tiling_act_core_num, block_num=self.tiling_act_core_num) as core_index:
            with self.tik_instance.if_scope(core_index < (self.tiling_act_core_num - 1)):
                with self.tik_instance.for_range(0, self.tiling_batch_each_core) as i:
                    batch_index = core_index * self.tiling_batch_each_core + i
                    self.compute_each_batch(batch_index)
            with self.tik_instance.if_scope(core_index == (self.tiling_act_core_num - 1)):
                with self.tik_instance.for_range(0, self.tiling_batch_last_core) as i:
                    batch_index = core_index * self.tiling_batch_each_core + i
                    self.compute_each_batch(batch_index)
        if not self.debug:
            tbe_context.get_context().add_compile_info("vars", {"core_num": self.tik_profiling.get_aicore_num()})

        opt_config = {"out_of_bound_sync_check": True, "save_temp_cce_file": True}
        if self.debug:
            self.tik_instance.BuildCCE(
                kernel_name=self.kernel_name,
                inputs=[self.fw_weight_gm, self.cross_mean_sum_gm, self.cross_mean_square_sum_gm, self.fw_field_map_gm],
                outputs=[self.output_gm],
                config=opt_config)
        else:
            self.tik_instance.BuildCCE(
                kernel_name=self.kernel_name,
                inputs=[self.fw_weight_gm, self.cross_mean_sum_gm, self.cross_mean_square_sum_gm, self.fw_field_map_gm],
                outputs=[self.output_gm],
                flowtable=[self.tiling_gm],
                config=opt_config)
          

        return self.tik_instance

@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sparse_fw_ffm_part2(fw_weight,
                        cross_mean_sum,
                        cross_mean_square_sum,
                        fw_field_map,
                        output,
                        kernel_name="sparse_fw_ffm_part2"):
    obj = SparseFwFFMPart2(fw_weight, cross_mean_sum, cross_mean_square_sum, fw_field_map, output, kernel_name)
    tik_instance = obj.sparse_fw_ffm_part2_tik()
    # obj.tik_output_debug()
    return tik_instance
  
# if __name__ == '__main__':
#     set_current_compile_soc_info("Ascend910B2")
#     batch_size =  1000
#     part_num = 2
#     fw_field_num = 75
#     feature_dim = 64
#     fw_weight_shape = (1,  fw_field_num*(fw_field_num+1)/2 )
#     cross_mean_sum_shape = (batch_size, part_num, fw_field_num, feature_dim)
#     fw_field_map_shape = (batch_size, fw_field_num)
#     output_shape = (batch_size, feature_dim)
#     fw_weight = {"shape": fw_weight_shape, "dtype": "float32"}
#     cross_mean_sum = {"shape": cross_mean_sum_shape, "dtype": "float32"}
#     cross_mean_square_sum = {"shape": cross_mean_sum_shape, "dtype": "float32"}
#     fw_field_map = {"shape": fw_field_map_shape, "dtype": "float32"}
#     output = {"shape": output_shape, "dtype": "float32"}
#     kernel_name = "sparse_fw_ffm_part2"
#     obj = SparseFwFFMPart2(fw_weight, cross_mean_sum, cross_mean_square_sum, fw_field_map, output, kernel_name)
#     tik_instance = obj.sparse_fw_ffm_part2_tik()
#     obj.tik_output_debug()