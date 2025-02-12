import itertools
import os
import sys
from absl import app, flags, logging
from apex import optimizers as apex_optim

from dlrm.data.feature_spec import FeatureSpec
from dlrm.model.distributed import DistributedDlrm
from dlrm.utils import distributed as dist
from dlrm.utils.checkpointing.distributed import make_distributed_checkpoint_writer, make_distributed_checkpoint_loader
from dlrm.utils.distributed import get_gpu_batch_sizes, get_device_mapping, is_main_process, is_distributed
from dlrm.scripts.main import load_feature_spec, validate_flags

import datetime
from time import time

import dllogger
import numpy as np
import torch
from absl import app, flags

import dlrm.scripts.utils as utils
from dlrm.data.data_loader import get_data_loaders
from dlrm.data.utils import prefetcher, get_embedding_sizes

import re
import torch
import torch_npu

FLAGS = flags.FLAGS

# Basic run settings
flags.DEFINE_enum("mode", default='train', enum_values=['train', 'test', 'inference_benchmark'],
                  help="Select task to be performed")
flags.DEFINE_integer("seed", 12345, "Random seed")

# Training flags
flags.DEFINE_integer("batch_size", 65536, "Batch size used for training")
flags.DEFINE_integer("test_batch_size", 65536, "Batch size used for testing/validation")
flags.DEFINE_float("lr", 24, "Base learning rate")
flags.DEFINE_integer("epochs", 1, "Number of epochs to train for")
flags.DEFINE_integer("max_steps", None, "Stop training after doing this many optimization steps")

# Learning rate schedule flags
flags.DEFINE_integer("warmup_factor", 0, "Learning rate warmup factor. Must be a non-negative integer")
flags.DEFINE_integer("warmup_steps", 8000, "Number of warmup optimization steps")
flags.DEFINE_integer("decay_steps", 24000,
                     "Polynomial learning rate decay steps. If equal to 0 will not do any decaying")
flags.DEFINE_integer("decay_start_step", 48000,
                     "Optimization step after which to start decaying the learning rate, "
                     "if None will start decaying right after the warmup phase is completed")
flags.DEFINE_integer("decay_power", 2, "Polynomial learning rate decay power")
flags.DEFINE_float("decay_end_lr", 0, "LR after the decay ends")

# Model configuration
flags.DEFINE_enum("embedding_type", "custom_cuda",
                  ["joint", "custom_cuda", "multi_table", "joint_sparse", "joint_fused"],
                  help="The type of the embedding operation to use")
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of embedding space for categorical features")
flags.DEFINE_list("top_mlp_sizes", [1024, 1024, 512, 256, 1], "Linear layer sizes for the top MLP")
flags.DEFINE_list("bottom_mlp_sizes", [512, 256, 128], "Linear layer sizes for the bottom MLP")
flags.DEFINE_enum("interaction_op", default="cuda_dot", enum_values=["cuda_dot", "dot", "cat"],
                  help="Type of interaction operation to perform.")

# Data configuration
flags.DEFINE_string("dataset", None, "Path to dataset directory")
flags.DEFINE_string("feature_spec", default="feature_spec.yaml",
                    help="Name of the feature spec file in the dataset directory")
flags.DEFINE_enum("dataset_type", default="parametric", enum_values=['synthetic_gpu', 'parametric'],
                  help='The type of the dataset to use')
flags.DEFINE_boolean("shuffle_batch_order", False, "Read batch in train dataset by random order", short_name="shuffle")

flags.DEFINE_integer("max_table_size", None,
                     "Maximum number of rows per embedding table, "
                     "by default equal to the number of unique values for each categorical variable")
flags.DEFINE_boolean("hash_indices", False,
                     "If True the model will compute `index := index % table size` "
                     "to ensure that the indices match table sizes")

# Synthetic data configuration
flags.DEFINE_integer("synthetic_dataset_num_entries", default=int(2 ** 15 * 1024),
                     help="Number of samples per epoch for the synthetic dataset")
flags.DEFINE_list("synthetic_dataset_table_sizes", default=','.join(26 * [str(10 ** 5)]),
                  help="Cardinalities of variables to use with the synthetic dataset.")
flags.DEFINE_integer("synthetic_dataset_numerical_features", default='13',
                     help="Number of numerical features to use with the synthetic dataset")
flags.DEFINE_boolean("synthetic_dataset_use_feature_spec", default=False,
                     help="Create a temporary synthetic dataset based on a real one. "
                          "Uses --dataset and --feature_spec"
                          "Overrides synthetic_dataset_table_sizes and synthetic_dataset_numerical_features."
                          "--synthetic_dataset_num_entries is still required")

# Checkpointing
flags.DEFINE_string("load_checkpoint_path", None, "Path from which to load a checkpoint")
flags.DEFINE_string("save_checkpoint_path", None, "Path to which to save the training checkpoints")

# Saving and logging flags
flags.DEFINE_string("log_path", "./log.json", "Destination for the log file with various results and statistics")
flags.DEFINE_integer("test_freq", None,
                     "Number of optimization steps between validations. If None will test after each epoch")
flags.DEFINE_float("test_after", 0, "Don't test the model unless this many epochs has been completed")
flags.DEFINE_integer("print_freq", 200, "Number of optimizations steps between printing training status to stdout")
flags.DEFINE_integer("benchmark_warmup_steps", 0,
                     "Number of initial iterations to exclude from throughput measurements")

# Machine setting flags
flags.DEFINE_string("base_device", "cuda", "Device to run the majority of the model operations")
flags.DEFINE_boolean("amp", False, "If True the script will use Automatic Mixed Precision")

flags.DEFINE_boolean("cuda_graphs", False, "Use CUDA Graphs")

# inference benchmark
flags.DEFINE_list("inference_benchmark_batch_sizes", default=[1, 64, 4096],
                  help="Batch sizes for inference throughput and latency measurements")
flags.DEFINE_integer("inference_benchmark_steps", 200,
                     "Number of steps for measuring inference latency and throughput")

# Miscellaneous
flags.DEFINE_float("auc_threshold", None, "Stop the training after achieving this AUC")
flags.DEFINE_boolean("optimized_mlp", True, "Use an optimized implementation of MLP from apex")
flags.DEFINE_enum("auc_device", default="GPU", enum_values=['GPU', 'CPU'],
                  help="Specifies where ROC AUC metric is calculated")

flags.DEFINE_string("backend", "nccl", "Backend to use for distributed training. Default nccl")
flags.DEFINE_boolean("bottom_features_ordered", False,
                     "Sort features from the bottom model, useful when using saved "
                     "checkpoint in different device configurations")
flags.DEFINE_boolean("freeze_mlps", False,
                     "For debug and benchmarking. Don't perform the weight update for MLPs.")
flags.DEFINE_boolean("freeze_embeddings", False,
                     "For debug and benchmarking. Don't perform the weight update for the embeddings.")
flags.DEFINE_boolean("Adam_embedding_optimizer", False, "Swaps embedding optimizer to Adam")
flags.DEFINE_boolean("Adam_MLP_optimizer", False, "Swaps MLP optimizer to Adam")

# test mode settings flags
flags.DEFINE_integer("total_devices", 1, "Total number of devices to run the test on")


def main(argv):
    torch.manual_seed(FLAGS.seed)
    torch.npu.set_compile_mode(jit_mode=False)

    use_gpu = "cpu" not in FLAGS.base_device.lower()
    rank, world_size, gpu = dist.init_distributed(backed=FLAGS.backend, use_gpu=use_gpu)
    device = FLAGS.base_device

    feature_spec = load_feature_spec(FLAGS)

    cat_feature_count = len(get_embedding_sizes(feature_spec, None))
    validate_flags(cat_feature_count)

    world_embedding_size = get_embedding_sizes(feature_spec, max_table_size=FLAGS.max_table_size)
    world_categorical_feature_sizes = np.asarray(world_embedding_size)
    device_mapping = get_device_mapping(world_embedding_size, num_gpus=world_size)

    categorical_feature_sizes = world_categorical_feature_sizes[device_mapping["embedding"][rank]].tolist()
    num_numerical_features = feature_spec.get_number_of_numerical_features()

    bottom_mlp_sizes = FLAGS.bottom_mlp_sizes if rank == device_mapping["bottom_mlp"] else None


    model = DistributedDlrm(
        vectors_per_gpu=device_mapping['vectors_per_gpu'],
        embedding_device_mapping=device_mapping['embedding'],
        embedding_type=FLAGS.embedding_type,
        embedding_dim=FLAGS.embedding_dim,
        world_num_categorical_features=len(world_categorical_feature_sizes),
        categorical_feature_sizes=categorical_feature_sizes,
        num_numerical_features=num_numerical_features,
        hash_indices=FLAGS.hash_indices,
        bottom_mlp_sizes=bottom_mlp_sizes,
        top_mlp_sizes=FLAGS.top_mlp_sizes,
        interaction_op=FLAGS.interaction_op,
        fp16=FLAGS.amp,
        use_cpp_mlp=FLAGS.optimized_mlp,
        bottom_features_ordered=FLAGS.bottom_features_ordered,
        device=device
    )

    dist.setup_distributed_print(is_main_process())

    checkpoint_loader = make_distributed_checkpoint_loader(device_mapping=device_mapping, rank=rank)

    if FLAGS.load_checkpoint_path:
        checkpoint_loader.load_checkpoint(model, FLAGS.load_checkpoint_path)
        model.to(device)


    def parallelize(model):
        if world_size <= 1:
            return model
        
        model.top_model = torch.nn.parallel.DistributedDataParallel(model.top_model)
        return model
    
    model = parallelize(model)
    model = model.eval()

    numerical_features_example = torch.zeros(1, 13, dtype=torch.float32)
    numerical_features_example = numerical_features_example.to(device)

    categorical_features_example = torch.zeros(1, 26, dtype=torch.int64)
    categorical_features_example = categorical_features_example.to(device)

    input_names = ["numerical_features", "categorical_features"]
    output_names = ["output"]

    dynamic_axes = {
        "numerical_features": {0: "batch_size"},
        "categorical_features": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    x = (numerical_features_example, categorical_features_example)

    torch.onnx.export(model, 
                      x, 
                      "output/dlrm.onnx", 
                      input_names=input_names, 
                      output_names=output_names, 
                      dynamic_axes=dynamic_axes,
                      opset_version=11
                    )
    

if __name__ == "__main__":
    app.run(main)