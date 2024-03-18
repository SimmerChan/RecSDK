# coding=utf-8
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import glob
import os
import shutil
import tensorflow as tf

from npu_bridge.npu_init import DumpConfig, npu_hooks_append, NPUEstimator, NPURunConfig, npu_plugin
from config import Config
from hook_utils import CalcQpsHook
from input_func import input_fn
from model_func import get_model_fn
from utils import FeatureSpecIns
from mx_rec.graph.modifier import GraphModifierHook
from mx_rec.util.initialize import init
from mx_rec.util.log import logger

LOGGER_ERROR = logger.error("error cpu bind info, skipped.")

npu_plugin.set_device_sat_mode(0)
tf.logging.set_verbosity(tf.logging.INFO)

_SSD_SAVE_PATH = ["ssd_data"]


def main(params):
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    dump_config = DumpConfig(enable_dump=False,
                             dump_path='/data/dump_path/',
                             dump_step='1|2',
                             dump_mode='all')
    model_dir = f"{params.model_ckpt_dir}_rank"
    run_config = NPURunConfig(
        model_dir=model_dir,
        save_summary_steps=100000,  # tf.summary运行周期
        save_checkpoints_steps=params.save_checkpoints_steps,
        keep_checkpoint_max=1,
        dump_config=dump_config,
        session_config=session_config,
        precision_mode='allow_mix_precision',
        enable_data_pre_proc=True,
        iterations_per_loop=params.iterations_per_loop,
        op_compiler_cache_mode="enable",
        HCCL_algorithm="level0:fullmesh;level1:pairwise",
    )

    cfg = Config(params)

    est = NPUEstimator(
        model_fn=get_model_fn(),
        params=params,
        model_dir=params.model_dir,
        config=run_config
    )

    hooks_list = [GraphModifierHook(modify_graph=params.modify_graph)]
    hooks_list.append(CalcQpsHook(params.batch_size, args.iterations_per_loop))

    if params.run_mode == 'train':
        est.train(input_fn=lambda: input_fn(params, cfg, ), max_steps=args.train_steps,
                  hooks=npu_hooks_append(hooks_list))
        return
    if params.run_mode == 'predict':
        results = est.predict(input_fn=lambda: input_fn(params, cfg, is_eval=True),
                              hooks=npu_hooks_append(hooks_list=hooks_list),
                              yield_single_examples=False)
        for _ in results:
            pass
        return
    if params.run_mode == 'train_and_evaluate':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(params, cfg, ),
                                            max_steps=params.train_steps, hooks=npu_hooks_append(hooks_list=hooks_list))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(params, cfg, is_eval=True),
                                          steps=params.eval_steps, hooks=npu_hooks_append(hooks_list=hooks_list))
        tf.estimator.train_and_evaluate(
            est, train_spec=train_spec, eval_spec=eval_spec)
        return


def _del_related_dir(relate_real_path: str):
    dirs = glob.glob(relate_real_path)
    for sub_dir in dirs:
        shutil.rmtree(sub_dir, ignore_errors=True)


def _clear_saved_model():
    _del_related_dir("/root/ascend/log/*")
    _del_related_dir(os.path.join(os.getcwd(), "kernel_meta*"))
    _del_related_dir(os.path.join(os.getcwd(), "op_cache"))

    is_train_mode = str(args.run_mode).startswith("train")
    if not is_train_mode:
        return
    logger.info("train mode: saved-model will be deleted.")
    _del_related_dir(os.path.join(os.getcwd(), "model_dir_rank*"))

    if os.getenv("CACHE_MODE", "") != "SSD":
        return
    # ssd not allow overwrite file, should clear it before training
    for part_path in _SSD_SAVE_PATH:
        if not os.path.isabs(part_path):
            part_path = os.path.join(os.getcwd(), part_path)  # 相对路径拼接当前路径作为全路径
        shutil.rmtree(part_path, ignore_errors=True)
        try:
            os.mkdir(part_path)
        except FileExistsError:
            logger.warning("ssd path already exists")  # 多进程并行，忽略异常


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_mode', type=str, default='train')
    parser.add_argument('--model_ckpt_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_steps', type=int, default=-1)
    parser.add_argument('--eval_steps', type=int, default=-1)
    parser.add_argument('--line_per_sample', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--save_checkpoints_steps', type=int, default=100)
    parser.add_argument('--iterations_per_loop', type=int, default=10)
    parser.add_argument('--use_one_shot', type=int, default=0)
    parser.add_argument('--modify_graph', type=int, default=1)
    parser.add_argument('--rank_size', type=int, default=8)

    args, unknowns = parser.parse_known_args()
    if args.run_mode == "train_and_evaluate":
        args.train_steps = 50000
        args.eval_steps = 1000
        args.save_checkpoints_steps = args.train_steps
    try:
        args.use_mpi = bool(int(os.getenv("USE_MPI")))
        args.rank_id = int(os.getenv("RANK_ID")) if os.getenv("RANK_ID") else None
        args.rank_size = int(os.getenv("TRAIN_RANK_SIZE")) if os.getenv("TRAIN_RANK_SIZE") else None
        args.use_dynamic_expansion = bool(int(os.getenv("USE_DYNAMIC_EXPANSION", 0)))
        args.use_multi_lookup = bool(int(os.getenv("USE_MULTI_LOOKUP", 0)))
        args.modify_graph = bool(int(os.getenv("USE_MODIFY_GRAPH", 0)))
        args.use_one_shot = bool(int(os.getenv("USE_ONE_SHOT", 0)))
        args.use_hot = bool(int(os.getenv("USE_HOT", 1)))
        args.use_faae = bool(int(os.getenv("USE_FAAE", 0)))
        args.use_dynamic = bool(int(os.getenv("USE_DYNAMIC", 0)))
        args.cache_mode = os.getenv("CACHE_MODE")
    except ValueError as err:
        raise ValueError(f"please correctly config USE_DYNAMIC_EXPANSION or USE_MULTI_LOOKUP or USE_FAAE "
                         f"or USE_MODIFY_GRAPH or other environment variable value.") from err
    _clear_saved_model()
    # ASCEND INIT
    init(use_mpi=args.use_mpi, rank_id=args.rank_id, rank_size=args.rank_size, train_steps=args.train_steps,
         eval_steps=args.eval_steps, use_dynamic=args.use_dynamic, use_dynamic_expansion=args.use_dynamic_expansion,
         use_hot=args.use_hot)
    args.model_dir = f"{args.model_ckpt_dir}_rank"
    FeatureSpecIns.set_instance()
    main(args)
    logger.info("Demo done!")
