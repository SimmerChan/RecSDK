# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
sparse_fw_ffm_part2_grad.py
"""
from impl import constant_util as constant
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from tbe.common.platform import get_bit_len


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # tiling param nums
    TILING_NUMS = 8
    # `1 byte = 8 bit`
    EIGHT_BIT = 8
    # BLOCK_BYTES
    BLOCK_BYTES = 32
    # int64 bytes
    INT64_BYTES = 8
    # max int64
    MAX_INT64 = 2**64 - 1
    # compute only zero axis, cut last dim
    MODE0 = 0
    # int32 or fp32 1 block has num size
    FP32_BLOCK_SIZE = 8
    # fw_weight_typical_len
    FW_WEIGHT_MAX_LEN = 2880
    # fw_field_map
    FIELD_MAP_MAX_LEN = 75
    # field_num typical_len
    FIELD_TYPICAL_NUM = 2
    RESERVED_UB_SIZE = 10240


class SparseFwFFMGradInit:
    """
    Function: class that execute sparse_fw_ffm_part2_grad
    """

    def __init__(self, grad, cross_mean_sum, cross_mean_square_sum, fw_weight, fw_field_map,
                 fw_output_res, fw_cross_mean_sum_grad, fw_cross_mean_square_sum_grad, kernel_name):
        self.tik_instance = tik.Tik()
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.unknown_max_shape = (Constant.MAX_INT64,)
        self.kernel_name = kernel_name
        self.opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True, "dynamic_tik": True}
        self.tiling_key = None
        self.tiling_dtype = "int64"
        self.tiling_shape = (Constant.TILING_NUMS,)
        self.kernel_name = kernel_name
        self.embedding_size = grad.get("shape")[-1]
        self.new_mode = True if self.embedding_size > 64 else False
        # gm init
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype,
                                                  self.tiling_shape,
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self.grad_gm = self.tik_instance.Tensor(
            grad.get("dtype"), self.unknown_max_shape, name="grad", scope=tik.scope_gm)
        self.cross_mean_sum_gm = self.tik_instance.Tensor(cross_mean_sum.get(
            "dtype"), self.unknown_max_shape, name="cross_mean_sum", scope=tik.scope_gm)
        self.cross_mean_square_sum_gm = self.tik_instance.Tensor(cross_mean_square_sum.get(
            "dtype"), self.unknown_max_shape, name="cross_mean_square_sum", scope=tik.scope_gm)
        self.fw_weight_gm = self.tik_instance.Tensor(fw_weight.get(
            "dtype"), self.unknown_max_shape, name="fw_weight", scope=tik.scope_gm)
        self.fw_field_map_gm = self.tik_instance.Tensor(fw_field_map.get(
            "dtype"), self.unknown_max_shape, name="fw_field_map", scope=tik.scope_gm)

        self.fw_output_res_gm = self.tik_instance.Tensor(fw_output_res.get(
            "dtype"), self.unknown_max_shape, name="fw_output_res", is_atomic_add=True, scope=tik.scope_gm)
        self.fw_cross_mean_sum_grad_gm = self.tik_instance.Tensor(fw_cross_mean_sum_grad.get(
            "dtype"), self.unknown_max_shape, name="fw_cross_mean_sum_grad", scope=tik.scope_gm)
        self.fw_cross_mean_square_sum_grad_gm = self.tik_instance.Tensor(fw_cross_mean_square_sum_grad.get(
            "dtype"), self.unknown_max_shape, name="fw_cross_mean_square_sum_grad", scope=tik.scope_gm)

        self.input_gm_list = [self.grad_gm, self.cross_mean_sum_gm,
                              self.cross_mean_square_sum_gm, self.fw_weight_gm, self.fw_field_map_gm]
        self.output_gm_list = [self.fw_output_res_gm,
                               self.fw_cross_mean_sum_grad_gm, self.fw_cross_mean_square_sum_grad_gm]
        # scalar init
        self.tiling_key = self.tik_instance.Scalar(self.tiling_dtype, "tiling_key", init_value=0)
        self.batch_size = self.tik_instance.Scalar(self.tiling_dtype, "batch_size", init_value=0)
        self.field_num = self.tik_instance.Scalar(self.tiling_dtype, "field_num", init_value=0)
        self.fw_field_num = self.tik_instance.Scalar(self.tiling_dtype, "fw_field_num", init_value=0)
        self.fw_weight_len = self.tik_instance.Scalar(self.tiling_dtype, "fw_weight_len", init_value=0)
        self.core_used_num = self.tik_instance.Scalar(self.tiling_dtype, "core_used_num", init_value=0)
        self.one_core_process_batch = self.tik_instance.Scalar(self.tiling_dtype, "core_process_batch", init_value=0)
        self.batch_tail = self.tik_instance.Scalar(self.tiling_dtype, "batch_tail", init_value=0)
        self.this_core_process_batch = self.tik_instance.Scalar(self.tiling_dtype, "this_process_batch", init_value=0)
        self.batch_offset = self.tik_instance.Scalar(self.tiling_dtype, "batch_offset", init_value=0)
        self.for_batch_offset = self.tik_instance.Scalar(self.tiling_dtype, "for_batch_offset", init_value=0)
        self.fw_iter = self.tik_instance.Scalar(self.tiling_dtype, "fw_iter", init_value=0)
        self.fw_weight_get_iter_plus_1 = self.tik_instance.Scalar("float32", "fw_weight_get_iter_plus_1", init_value=0)
        self.fw_weight_get_iter_plus_1_0 = self.tik_instance.Scalar("float32", "fw_weight_get_iter_plus_1_0", init_value=0)
        self.fw_weight_get_iter_plus_1_1 = self.tik_instance.Scalar("float32", "fw_weight_get_iter_plus_1_1", init_value=0)
        self.abs_fw_weight = self.tik_instance.Scalar("float32", "abs_fw_weight", init_value=0)
        self.multi_tag = self.tik_instance.Scalar(self.tiling_dtype, "multi_tag", init_value=0)
        self.field_1 = self.tik_instance.Scalar("int32", "field_1", init_value=0)
        self.field_2 = self.tik_instance.Scalar("int32", "field_2", init_value=0)
        self.field_2_0 = self.tik_instance.Scalar("int32", "field_2_0", init_value=0)
        self.field_2_1 = self.tik_instance.Scalar("int32", "field_2_1", init_value=0)
        self.start_iter = self.tik_instance.Scalar(self.tiling_dtype, "start_iter", init_value=0)
        self.cross_mean_size = self.tik_instance.Scalar(self.tiling_dtype, "cross_mean_size", init_value=0)
        # ub init 不参与DB的ub
        self.fw_weight_ub = self.tik_instance.Tensor(fw_weight.get(
            "dtype"), [Constant.FW_WEIGHT_MAX_LEN], name="fw_weight_ub", scope=tik.scope_ubuf)
        self.fw_data_ub = self.tik_instance.Tensor("float32", [Constant.FW_WEIGHT_MAX_LEN], name="fw_data_ub", scope=tik.scope_ubuf)


    def tiling_args(self):
        """
        tiling info:

        """
        tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_NUMS,), name="tiling_ub", scope=tik.scope_ubuf)
        tiling_burst = Constant.TILING_NUMS // (Constant.BLOCK_BYTES // Constant.INT64_BYTES)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, tiling_burst, 0, 0)
        self.tiling_key.set_as(tiling_ub[0])
        self.batch_size.set_as(tiling_ub[1])
        self.field_num.set_as(tiling_ub[2])
        self.fw_field_num.set_as(tiling_ub[3])
        self.fw_weight_len.set_as(tiling_ub[4])
        self.core_used_num.set_as(self.core_nums)
        with self.tik_instance.if_scope(self.batch_size < self.core_nums):
            self.core_used_num.set_as(self.batch_size)
        self.one_core_process_batch.set_as(self.batch_size // self.core_used_num)
        self.batch_tail.set_as(self.batch_size % self.core_used_num)
        self.this_core_process_batch.set_as(self.one_core_process_batch)
        self.cross_mean_size.set_as(self.embedding_size * self.fw_field_num * self.field_num)

    def _data_move(self, dst, src, len):
        nburst = 1
        burst = (len + Constant.FP32_BLOCK_SIZE - 1) // Constant.FP32_BLOCK_SIZE
        self.tik_instance.data_move(dst, src, 0, nburst, burst, 0, 0)

    def _vec_dup(self, ub_tensor, dump_value, len):
        remain_mask = len % 64
        repeat_times = (len - remain_mask) // 64
        if repeat_times != 0:
            self.tik_instance.vec_dup(64, ub_tensor, dump_value, repeat_times, 8)
        if remain_mask != 0:
            offset = repeat_times * 64
            self.tik_instance.vec_dup(remain_mask, ub_tensor[offset:], dump_value, 1, 8)

    def _vmuls(self, ub_out_tensor, ub_in_tensor, muls_value, len):
        remain_mask = len % 64
        repeat_times = (len - remain_mask) // 64
        if repeat_times != 0:
            self.tik_instance.vmuls(64, ub_out_tensor, ub_in_tensor,muls_value, repeat_times,1,1,8,8)
        if remain_mask != 0:
            offset = repeat_times * 64
            self.tik_instance.vmuls(remain_mask, ub_out_tensor[offset:], ub_in_tensor[offset:], muls_value, 1,1,1,8,8)

    def _vsub(self, ub_out_tensor, ub_in_tensor_0, ub_in_tensor_1, len):
        remain_mask = len % 64
        repeat_times = (len - remain_mask) // 64
        if repeat_times != 0:
            self.tik_instance.vsub(64, ub_out_tensor, ub_in_tensor_0,ub_in_tensor_1, repeat_times,1,1,1,8,8,8)
        if remain_mask != 0:
            offset = repeat_times * 64
            self.tik_instance.vsub(remain_mask, ub_out_tensor[offset:], ub_in_tensor_0[offset:],
                                    ub_in_tensor_1[offset:], 1,1,1,1,8,8,8)

    def _vmul(self, ub_out_tensor, ub_in_tensor_0, ub_in_tensor_1, len):
        remain_mask = len % 64
        repeat_times = (len - remain_mask) // 64
        if repeat_times != 0:
            self.tik_instance.vmul(64, ub_out_tensor, ub_in_tensor_0,ub_in_tensor_1, repeat_times,1,1,1,8,8,8)
        if remain_mask != 0:
            offset = repeat_times * 64
            self.tik_instance.vmul(remain_mask, ub_out_tensor[offset:], ub_in_tensor_0[offset:],
                                    ub_in_tensor_1[offset:], 1,1,1,1,8,8,8)

    def init_ub_tensors(self):
        self.fw_field_map_ub = self.tik_instance.Tensor("int32", [(Constant.FIELD_MAP_MAX_LEN + 7) // 8 * 8],
                                                        name="fw_field_map_ub", scope=tik.scope_ubuf) 
        self.grad_ub = self.tik_instance.Tensor("float32", [self.embedding_size], name="grad_ub", scope=tik.scope_ubuf)
        cross_mean_ub_size = self.embedding_size * Constant.FIELD_MAP_MAX_LEN * Constant.FIELD_TYPICAL_NUM
        self.cross_mean_sum_ub = self.tik_instance.Tensor("float32", [cross_mean_ub_size], name="cross_mean_sum_ub", scope=tik.scope_ubuf)
        self.cross_mean_square_sum_ub = self.tik_instance.Tensor("float32", [cross_mean_ub_size], name="cross_mean_square_sum_ub", scope=tik.scope_ubuf)
        self.grad_muls_ub = self.tik_instance.Tensor("float32", [self.embedding_size], name="grad_muls_ub", scope=tik.scope_ubuf)
        self.grad_muls_ub_0 = self.tik_instance.Tensor("float32", [self.embedding_size], name="grad_muls_ub_0", scope=tik.scope_ubuf)
        self.grad_muls_ub_1 = self.tik_instance.Tensor("float32", [self.embedding_size], name="grad_muls_ub_1", scope=tik.scope_ubuf)
        self.grad_muls_2_ub = self.tik_instance.Tensor("float32", [self.embedding_size], name="grad_muls_2_ub", scope=tik.scope_ubuf)
        self.grad_muls_3_ub = self.tik_instance.Tensor("float32", [self.embedding_size], name="grad_muls_3_ub", scope=tik.scope_ubuf)
        self.reduce_sum_ub = self.tik_instance.Tensor("float32", [Constant.FP32_BLOCK_SIZE*80], name="reduce_sum_ub", scope=tik.scope_ubuf)
        # self.reduce_sum_2_ub = self.tik_instance.Tensor("float32", [Constant.FP32_BLOCK_SIZE], name="reduce_sum_2_ub", scope=tik.scope_ubuf)
        self.out_0_0 = self.tik_instance.Tensor("float32", [self.embedding_size], name="out_0_0", scope=tik.scope_ubuf)
        self.out_1_0 = self.tik_instance.Tensor("float32", [self.embedding_size], name="out_1_0", scope=tik.scope_ubuf)
        self.out_2_0 = self.tik_instance.Tensor("float32", [self.embedding_size], name="out_2_0", scope=tik.scope_ubuf)
        self.out_0_1 = self.tik_instance.Tensor("float32", [self.embedding_size], name="out_0_1", scope=tik.scope_ubuf)
        self.out_1_1 = self.tik_instance.Tensor("float32", [self.embedding_size], name="out_1_1", scope=tik.scope_ubuf)
        self.out_2_1 = self.tik_instance.Tensor("float32", [self.embedding_size], name="out_2_1", scope=tik.scope_ubuf)

        self.out_3 = self.tik_instance.Tensor("float32", [self.embedding_size], name="out_3", scope=tik.scope_ubuf)
        self.out_4 = self.tik_instance.Tensor("float32", [self.embedding_size], name="out_4", scope=tik.scope_ubuf)
        self.out_5 = self.tik_instance.Tensor("float32", [self.embedding_size], name="out_5", scope=tik.scope_ubuf)
        self.square_ub = self.tik_instance.Tensor("float32", [self.embedding_size], name="square_ub", scope=tik.scope_ubuf)
        if not self.new_mode:
            self.mean_sum_out_ub = self.tik_instance.Tensor("float32", [cross_mean_ub_size], name="mean_sum_out_ub", scope=tik.scope_ubuf)
            self.mean_square_sum_out_ub = self.tik_instance.Tensor("float32", [cross_mean_ub_size], name="mean_square_sum_out_ub", scope=tik.scope_ubuf)
        self.fw_output_res_ub = self.tik_instance.Tensor("float32", [Constant.FW_WEIGHT_MAX_LEN], name="fw_output_res_ub", scope=tik.scope_ubuf)

    def sparse_fw_ffm_part2_grad_compute_tiling(self):
        """
        sparse_fw_ffm_part2_grad_compute_tiling
        """
        self.tiling_args()
        with self.tik_instance.for_range(0, self.core_nums, block_num=self.core_nums) as core_id:
            with self.tik_instance.if_scope(core_id < self.core_used_num):
                with self.tik_instance.if_scope(self.batch_tail > 0):
                    with self.tik_instance.if_scope(core_id < self.batch_tail):
                        self.this_core_process_batch.set_as(self.one_core_process_batch + 1)
                self.batch_offset.set_as(core_id * self.this_core_process_batch)
                with self.tik_instance.if_scope(self.batch_tail > 0):
                    with self.tik_instance.if_scope(core_id > (self.batch_tail - 1)):
                        self.batch_offset.set_as(core_id * self.one_core_process_batch + self.batch_tail)
                # 搬入fw_weight
                self._data_move(self.fw_weight_ub, self.fw_weight_gm, self.fw_weight_len)
                self.tik_instance.vadds(64, self.fw_weight_ub, self.fw_weight_ub, 1.0, (self.fw_weight_len + 63) // 64,1,1,8,8)
                self.tik_instance.vec_dup(64, self.fw_data_ub, 0.0, (self.fw_weight_len + 63) // 64,  64 // Constant.FP32_BLOCK_SIZE)
                with self.tik_instance.for_range(0, self.this_core_process_batch) as batch_cycle_id:
                    self.init_ub_tensors()
                    self.fw_iter.set_as(0)
                    self.for_batch_offset.set_as(self.batch_offset + batch_cycle_id)
                    fw_field_map_offset = self.for_batch_offset * self.fw_field_num
                    cross_mean_offset = self.for_batch_offset * self.cross_mean_size
                    grad_offset = self.for_batch_offset * self.embedding_size
                    self._data_move(self.fw_field_map_ub, self.fw_field_map_gm[fw_field_map_offset:], self.fw_field_num)
                    self._data_move(self.cross_mean_sum_ub, self.cross_mean_sum_gm[cross_mean_offset:], self.cross_mean_size)
                    self._data_move(self.cross_mean_square_sum_ub, self.cross_mean_square_sum_gm[cross_mean_offset:], self.cross_mean_size)
                    self._data_move(self.grad_ub, self.grad_gm[grad_offset:], self.embedding_size)
                    if self.new_mode:
                        self._vmuls(self.grad_muls_3_ub, self.grad_ub, 0.5, self.embedding_size)
                    else:
                        self.tik_instance.vec_dup(self.embedding_size, self.mean_sum_out_ub, 0.0, self.fw_field_num * self.field_num, self.embedding_size // Constant.FP32_BLOCK_SIZE)
                        self.tik_instance.vec_dup(self.embedding_size, self.mean_square_sum_out_ub, 0.0, self.fw_field_num * self.field_num, self.embedding_size // Constant.FP32_BLOCK_SIZE)
                        self.tik_instance.vmuls(self.embedding_size,self.grad_muls_3_ub,self.grad_ub,0.5,1,1,1,8,8)
                    self.tik_instance.vec_dup(64, self.fw_output_res_ub, 0.0, (self.fw_weight_len + 64) // 64,  64 // Constant.FP32_BLOCK_SIZE)
                    self.tik_instance.set_atomic_add(1)
                    self.do_fw_field_for()
                    self.tik_instance.set_atomic_add(0)
                    if not self.new_mode:
                        self._data_move(self.fw_cross_mean_sum_grad_gm[cross_mean_offset:], self.mean_sum_out_ub, self.cross_mean_size)
                        self._data_move(self.fw_cross_mean_square_sum_grad_gm[cross_mean_offset:], self.mean_square_sum_out_ub, self.cross_mean_size)
                    self.tik_instance.vadd(64, self.fw_data_ub, self.fw_data_ub, self.fw_output_res_ub,(self.fw_weight_len + 64) // 64,1,1,1,8,8,8)
                self.tik_instance.set_atomic_add(1)
                self._data_move(self.fw_output_res_gm, self.fw_data_ub, self.fw_weight_len)
                self.tik_instance.set_atomic_add(0)

    def do_fw_field_for(self):
        with self.tik_instance.for_range(0, self.fw_field_num) as fw_field_1:
            self.multi_tag.set_as(0)
            self.start_iter.set_as(self.fw_iter)
            self.field_1.set_as(self.fw_field_map_ub[fw_field_1])
            self.tik_instance.vec_dup(64, self.reduce_sum_ub, 0.0, 10, 8)
            with self.tik_instance.if_scope(self.field_1 > -1):
                with self.tik_instance.if_scope(self.field_1 > (self.field_num-1)):
                    self.multi_tag.set_as(1)
                    self.field_1.set_as(self.field_1 - self.field_num)
                with self.tik_instance.for_range(0, fw_field_1//2) as fw_field_2:
                    self.field_2_0.set_as(self.fw_field_map_ub[fw_field_2*2])
                    self.field_2_1.set_as(self.fw_field_map_ub[fw_field_2*2+1])
                    with self.tik_instance.if_scope(tik.all((self.field_2_0 > -1),(self.field_2_1 > -1))):
                        with self.tik_instance.if_scope(self.field_2_0 > (self.field_num-1)):
                            self.field_2_0.set_as(self.field_2_0 - self.field_num)
                        with self.tik_instance.if_scope(self.field_2_1 > (self.field_num-1)):
                            self.field_2_1.set_as(self.field_2_1 - self.field_num)
                        self.fw_weight_get_iter_plus_1_0.set_as(self.fw_weight_ub[self.fw_iter])
                        self.fw_weight_get_iter_plus_1_1.set_as(self.fw_weight_ub[self.fw_iter+1])

                        cross_mean_embedding_offset_0_0 = (self.field_2_0 * self.fw_field_num + fw_field_1) * self.embedding_size
                        cross_mean_embedding_offset_1_0 = (self.field_1 * self.fw_field_num + fw_field_2*2) * self.embedding_size

                        cross_mean_embedding_offset_0_1 = (self.field_2_1 * self.fw_field_num + fw_field_1) * self.embedding_size
                        cross_mean_embedding_offset_1_1 = (self.field_1 * self.fw_field_num + fw_field_2*2+1) * self.embedding_size
                        if self.new_mode:
                            gm_offset_0_0 = self.for_batch_offset * self.cross_mean_size + cross_mean_embedding_offset_0_0
                            gm_offset_1_0 = self.for_batch_offset * self.cross_mean_size + cross_mean_embedding_offset_1_0
                            gm_offset_0_1 = self.for_batch_offset * self.cross_mean_size + cross_mean_embedding_offset_0_1
                            gm_offset_1_1 = self.for_batch_offset * self.cross_mean_size + cross_mean_embedding_offset_1_1
                            self._vmuls(self.grad_muls_ub_0,self.grad_ub,self.fw_weight_get_iter_plus_1_0,self.embedding_size)
                            self._vmuls(self.grad_muls_ub_1,self.grad_ub,self.fw_weight_get_iter_plus_1_1,self.embedding_size)
                            self._vmul(self.out_0_0,self.grad_muls_ub_0,self.cross_mean_sum_ub[cross_mean_embedding_offset_0_0:],self.embedding_size)
                            self._vmul(self.out_1_0,self.grad_muls_ub_0,self.cross_mean_sum_ub[cross_mean_embedding_offset_1_0:],self.embedding_size)
                            self._vmul(self.out_0_1,self.grad_muls_ub_1,self.cross_mean_sum_ub[cross_mean_embedding_offset_0_1:],self.embedding_size)
                            self._vmul(self.out_1_1,self.grad_muls_ub_1,self.cross_mean_sum_ub[cross_mean_embedding_offset_1_1:],self.embedding_size)
                            self._data_move(self.fw_cross_mean_sum_grad_gm[gm_offset_1_0:], self.out_0_0, self.embedding_size)
                            self._data_move(self.fw_cross_mean_sum_grad_gm[gm_offset_0_0:], self.out_1_0, self.embedding_size)
                            self._data_move(self.fw_cross_mean_sum_grad_gm[gm_offset_1_1:], self.out_0_1, self.embedding_size)
                            self._data_move(self.fw_cross_mean_sum_grad_gm[gm_offset_0_1:], self.out_1_1, self.embedding_size)
                            self._vmul(self.out_2_0,self.cross_mean_sum_ub[cross_mean_embedding_offset_0_0:],self.cross_mean_sum_ub[cross_mean_embedding_offset_1_0:],self.embedding_size)
                            self._vmul(self.out_2_0,self.out_2_0,self.grad_ub,self.embedding_size)
                            self._vmul(self.out_2_1,self.cross_mean_sum_ub[cross_mean_embedding_offset_0_1:],self.cross_mean_sum_ub[cross_mean_embedding_offset_1_1:],self.embedding_size)
                            self._vmul(self.out_2_1,self.out_2_1,self.grad_ub,self.embedding_size)
                            self.tik_instance.vadd(self.embedding_size - 64,self.out_2_0,self.out_2_0,self.out_2_0[64:],1,1,1,1,8,8,8)
                            self.tik_instance.vadd(self.embedding_size - 64,self.out_2_1,self.out_2_1,self.out_2_1[64:],1,1,1,1,8,8,8)
                            self.tik_instance.vcadd(64,self.reduce_sum_ub[(self.fw_iter-self.start_iter)*8],self.out_2_0,1,1,1,8)
                            self.tik_instance.vcadd(64,self.reduce_sum_ub[(self.fw_iter+1-self.start_iter)*8],self.out_2_1,1,1,1,8)
                        else:
                            self.tik_instance.vmuls(self.embedding_size,self.grad_muls_ub_0,self.grad_ub,self.fw_weight_get_iter_plus_1_0,1,1,1,8,8)
                            self.tik_instance.vmuls(self.embedding_size,self.grad_muls_ub_1,self.grad_ub,self.fw_weight_get_iter_plus_1_1,1,1,1,8,8)
                            self.tik_instance.vmul(self.embedding_size,self.out_0_0,self.grad_muls_ub_0,self.cross_mean_sum_ub[cross_mean_embedding_offset_0_0:],1,1,1,1,8,8,8)
                            self.tik_instance.vmul(self.embedding_size,self.out_1_0,self.grad_muls_ub_0,self.cross_mean_sum_ub[cross_mean_embedding_offset_1_0:],1,1,1,1,8,8,8)
                            self.tik_instance.vmul(self.embedding_size,self.out_0_1,self.grad_muls_ub_1,self.cross_mean_sum_ub[cross_mean_embedding_offset_0_1:],1,1,1,1,8,8,8)
                            self.tik_instance.vmul(self.embedding_size,self.out_1_1,self.grad_muls_ub_1,self.cross_mean_sum_ub[cross_mean_embedding_offset_1_1:],1,1,1,1,8,8,8)
                            self.tik_instance.vadd(self.embedding_size, self.mean_sum_out_ub[cross_mean_embedding_offset_1_0:], self.mean_sum_out_ub[cross_mean_embedding_offset_1_0:], self.out_0_0,1,1,1,1,8,8,8)
                            self.tik_instance.vadd(self.embedding_size, self.mean_sum_out_ub[cross_mean_embedding_offset_0_0:], self.mean_sum_out_ub[cross_mean_embedding_offset_0_0:], self.out_1_0,1,1,1,1,8,8,8)
                            self.tik_instance.vadd(self.embedding_size, self.mean_sum_out_ub[cross_mean_embedding_offset_1_1:], self.mean_sum_out_ub[cross_mean_embedding_offset_1_1:], self.out_0_1,1,1,1,1,8,8,8)
                            self.tik_instance.vadd(self.embedding_size, self.mean_sum_out_ub[cross_mean_embedding_offset_0_1:], self.mean_sum_out_ub[cross_mean_embedding_offset_0_1:], self.out_1_1,1,1,1,1,8,8,8)
                            self.tik_instance.vmul(self.embedding_size,self.out_2_0,self.cross_mean_sum_ub[cross_mean_embedding_offset_0_0:],self.cross_mean_sum_ub[cross_mean_embedding_offset_1_0:],1,1,1,1,8,8,8)
                            self.tik_instance.vmul(self.embedding_size,self.out_2_0,self.out_2_0,self.grad_ub,1,1,1,1,8,8,8)
                            self.tik_instance.vmul(self.embedding_size,self.out_2_1,self.cross_mean_sum_ub[cross_mean_embedding_offset_0_1:],self.cross_mean_sum_ub[cross_mean_embedding_offset_1_1:],1,1,1,1,8,8,8)
                            self.tik_instance.vmul(self.embedding_size,self.out_2_1,self.out_2_1,self.grad_ub,1,1,1,1,8,8,8)
                            self.tik_instance.vcadd(self.embedding_size,self.reduce_sum_ub[(self.fw_iter-self.start_iter)*8],self.out_2_0,1,1,1,8)
                            self.tik_instance.vcadd(self.embedding_size,self.reduce_sum_ub[(self.fw_iter+1-self.start_iter)*8],self.out_2_1,1,1,1,8)
                        self.fw_iter.set_as(self.fw_iter+2)
                    with self.tik_instance.else_scope():
                        self.inner_for(fw_field_1, fw_field_2*2, self.out_0_0, self.out_1_0, self.out_2_0)
                        self.inner_for(fw_field_1, fw_field_2*2+1, self.out_0_0, self.out_1_0, self.out_2_0)

                with self.tik_instance.if_scope(fw_field_1 % 2 == 1):
                    self.inner_for(fw_field_1, fw_field_1 - 1, self.out_0_0, self.out_1_0, self.out_2_0)
                self.fw_weight_get_iter_plus_1.set_as(self.fw_weight_ub[self.fw_iter])
                cross_mean_embedding_offset_2 = (self.field_1 * self.fw_field_num + fw_field_1) * self.embedding_size
                if self.new_mode:
                    gm_offset_2 = self.for_batch_offset * self.cross_mean_size + cross_mean_embedding_offset_2
                    self._vmuls(self.grad_muls_2_ub,self.grad_ub,self.fw_weight_get_iter_plus_1,self.embedding_size)
                    self._vmul(self.out_3,self.grad_muls_2_ub,self.cross_mean_sum_ub[cross_mean_embedding_offset_2:],self.embedding_size,)
                    self._vmuls(self.out_4,self.grad_muls_2_ub,-0.5,self.embedding_size)
                    self._data_move(self.fw_cross_mean_sum_grad_gm[gm_offset_2:], self.out_3, self.embedding_size)
                    self._data_move(self.fw_cross_mean_square_sum_grad_gm[gm_offset_2:], self.out_4, self.embedding_size)                    
                else:
                    self.tik_instance.vmuls(self.embedding_size,self.grad_muls_2_ub,self.grad_ub,self.fw_weight_get_iter_plus_1,1,1,1,8,8)
                    self.tik_instance.vmul(self.embedding_size,self.out_3,self.grad_muls_2_ub,self.cross_mean_sum_ub[cross_mean_embedding_offset_2:],1,1,1,1,8,8,8)
                    self.tik_instance.vmuls(self.embedding_size,self.out_4,self.grad_muls_2_ub,-0.5,1,1,1,8,8)
                    self.tik_instance.vadd(self.embedding_size, self.mean_sum_out_ub[cross_mean_embedding_offset_2:], self.mean_sum_out_ub[cross_mean_embedding_offset_2:], self.out_3,1,1,1,1,8,8,8)
                    self.tik_instance.vadd(self.embedding_size, self.mean_square_sum_out_ub[cross_mean_embedding_offset_2:], self.mean_square_sum_out_ub[cross_mean_embedding_offset_2:], self.out_4,1,1,1,1,8,8,8)
                with self.tik_instance.if_scope(self.multi_tag > 0):
                    if self.new_mode:
                        self._vmul(self.square_ub,self.cross_mean_sum_ub[cross_mean_embedding_offset_2:],self.cross_mean_sum_ub[cross_mean_embedding_offset_2:],self.embedding_size)
                        self._vsub(self.out_5,self.square_ub,self.cross_mean_square_sum_ub[cross_mean_embedding_offset_2:],self.embedding_size)
                        self._vmul(self.out_5,self.out_5,self.grad_muls_3_ub,self.embedding_size)
                        self.tik_instance.vadd(self.embedding_size - 64,self.out_5,self.out_5,self.out_5[64:],1,1,1,1,8,8,8)
                        self.tik_instance.vcadd(64,self.reduce_sum_ub[(self.fw_iter-self.start_iter)*8],self.out_5,1,1,1,8)
                    else:
                        self.tik_instance.vmul(self.embedding_size,self.square_ub,self.cross_mean_sum_ub[cross_mean_embedding_offset_2:],self.cross_mean_sum_ub[cross_mean_embedding_offset_2:],1,1,1,1,8,8,8)
                        self.tik_instance.vsub(self.embedding_size,self.out_5,self.square_ub,self.cross_mean_square_sum_ub[cross_mean_embedding_offset_2:],1,1,1,1,8,8,8)
                        self.tik_instance.vmul(self.embedding_size,self.out_5,self.out_5,self.grad_muls_3_ub,1,1,1,1,8,8,8)
                        self.tik_instance.vcadd(self.embedding_size,self.reduce_sum_ub[(self.fw_iter-self.start_iter)*8],self.out_5,1,1,1,8)
                self.fw_iter.set_as(self.fw_iter+1)
            with self.tik_instance.else_scope():
                self.fw_iter.set_as(self.fw_iter+fw_field_1+1)
            self.do_fw_output_res()

    def inner_for(self, fw_field_1, fw_field_2, out_0, out_1, out_2):
        self.field_2.set_as(self.fw_field_map_ub[fw_field_2])
        with self.tik_instance.if_scope(self.field_2 > -1):
            with self.tik_instance.if_scope(self.field_2 > (self.field_num-1)):
                self.field_2.set_as(self.field_2 - self.field_num)
            self.fw_weight_get_iter_plus_1.set_as(self.fw_weight_ub[self.fw_iter])
                        # with self.tik_instance.if_scope(self.fw_weight_get_iter_plus_1 != 0):
                        # embedding size 只支持 64 以内
            cross_mean_embedding_offset_0 = (self.field_2 * self.fw_field_num + fw_field_1) * self.embedding_size
            cross_mean_embedding_offset_1 = (self.field_1 * self.fw_field_num + fw_field_2) * self.embedding_size
            if self.new_mode:
                gm_offset_0 = self.for_batch_offset * self.cross_mean_size + cross_mean_embedding_offset_0
                gm_offset_1 = self.for_batch_offset * self.cross_mean_size + cross_mean_embedding_offset_1
                self._vmuls(self.grad_muls_ub,self.grad_ub,self.fw_weight_get_iter_plus_1,self.embedding_size)
                self._vmul(out_0,self.grad_muls_ub,self.cross_mean_sum_ub[cross_mean_embedding_offset_0:],self.embedding_size)
                self._vmul(out_1,self.grad_muls_ub,self.cross_mean_sum_ub[cross_mean_embedding_offset_1:],self.embedding_size)
                self._data_move(self.fw_cross_mean_sum_grad_gm[gm_offset_1:], out_0, self.embedding_size)
                self._data_move(self.fw_cross_mean_sum_grad_gm[gm_offset_0:], out_1, self.embedding_size)
                self._vmul(out_2,self.cross_mean_sum_ub[cross_mean_embedding_offset_0:],self.cross_mean_sum_ub[cross_mean_embedding_offset_1:],self.embedding_size)
                self._vmul(out_2,out_2,self.grad_ub,self.embedding_size)
                self.tik_instance.vadd(self.embedding_size - 64,out_2,out_2,out_2[64:],1,1,1,1,8,8,8)
                self.tik_instance.vcadd(64,self.reduce_sum_ub[(self.fw_iter-self.start_iter)*8],out_2,1,1,1,8)
            else:
                self.tik_instance.vmuls(self.embedding_size,self.grad_muls_ub,self.grad_ub,self.fw_weight_get_iter_plus_1,1,1,1,8,8)
                self.tik_instance.vmul(self.embedding_size,out_0,self.grad_muls_ub,self.cross_mean_sum_ub[cross_mean_embedding_offset_0:],1,1,1,1,8,8,8)
                self.tik_instance.vmul(self.embedding_size,out_1,self.grad_muls_ub,self.cross_mean_sum_ub[cross_mean_embedding_offset_1:],1,1,1,1,8,8,8)
                self.tik_instance.vadd(self.embedding_size, self.mean_sum_out_ub[cross_mean_embedding_offset_1:], self.mean_sum_out_ub[cross_mean_embedding_offset_1:], out_0,1,1,1,1,8,8,8)
                self.tik_instance.vadd(self.embedding_size, self.mean_sum_out_ub[cross_mean_embedding_offset_0:], self.mean_sum_out_ub[cross_mean_embedding_offset_0:], out_1,1,1,1,1,8,8,8)
                self.tik_instance.vmul(self.embedding_size,out_2,self.cross_mean_sum_ub[cross_mean_embedding_offset_0:],self.cross_mean_sum_ub[cross_mean_embedding_offset_1:],1,1,1,1,8,8,8)
                self.tik_instance.vmul(self.embedding_size,out_2,out_2,self.grad_ub,1,1,1,1,8,8,8)
                self.tik_instance.vcadd(self.embedding_size,self.reduce_sum_ub[(self.fw_iter-self.start_iter)*8],out_2,1,1,1,8)
        self.fw_iter.set_as(self.fw_iter+1)

    def do_fw_output_res(self):
        with self.tik_instance.for_range(self.start_iter, self.fw_iter) as set_offset:
            self.fw_output_res_ub[set_offset].set_as(self.reduce_sum_ub[(set_offset-self.start_iter)*8])


    def sparse_fw_ffm_part2_grad_compute(self):
        """
        sparse_fw_ffm_part2_grad_compute
        """
        self.sparse_fw_ffm_part2_grad_compute_tiling()
        wr_compile_info = {
            "core_num": self.core_nums
        }
        tbe_context.get_context().add_compile_info("vars", wr_compile_info)
        flowtable_list = [self.tiling_gm]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_gm_list,
                                   outputs=self.output_gm_list,
                                   flowtable=flowtable_list,
                                   config=self.opt_config)

        return self.tik_instance


@register_operator("SparseFwFFMPart2Grad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def sparse_fw_ffm_part2_grad(grad, cross_mean_sum, cross_mean_square_sum,
                             fw_weight, fw_field_map,
                             fw_output_res, fw_cross_mean_sum_grad, fw_cross_mean_square_sum_grad,
                             kernel_name="sparse_fw_ffm_part2_grad"):
    """
    todo dtype and shape checking
    """
    src_dtype = grad.get("dtype").lower()
    supported_dtype = ("float32")
    para_check.check_dtype(src_dtype, supported_dtype, param_name="grad")
    obj = SparseFwFFMGradInit(grad, cross_mean_sum, cross_mean_square_sum,
                              fw_weight, fw_field_map,
                              fw_output_res, fw_cross_mean_sum_grad, fw_cross_mean_square_sum_grad,
                              kernel_name)

    return obj.sparse_fw_ffm_part2_grad_compute()
