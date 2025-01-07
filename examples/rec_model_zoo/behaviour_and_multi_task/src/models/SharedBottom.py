# coding=utf-8

import os
import glob
import json
import random
import shutil
import logging
import tensorflow as tf
from functools import partial
from datetime import date, timedelta, datetime

tf.compat.v1.enable_control_flow_v2()
tf.compat.v1.enable_resource_variables()

# from hook_train import OperatorTimingHook as train_operator_hook
# from hook_infer import OperatorTimingHook as infer_operator_hook
from utils import get_third_nearest_checkpoint, count_params

from npu_bridge.npu_init import *

tf.compat.v1.set_random_seed(2024)
# np.random.seed(2024)
random.seed(2024)

MODEL_NAME = "SharedBottom"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", "512,256", "deep layers")
tf.app.flags.DEFINE_string("tower_layers", "128,64", "tower layers")
tf.app.flags.DEFINE_float("ctr_task_wgt", 0.5, "loss weight of ctr task")
tf.app.flags.DEFINE_string("data_dir", "../data/aliccp/cast50_padded/", "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", f"../checkpoint/aliccp/{MODEL_NAME}/", "code check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", f"../model/serving/{MODEL_NAME}/", "export servable code for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", "train", "task type")
tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing code or not")
tf.app.flags.DEFINE_integer("max_seq_len", 50, "max length of sequence")
# tf.app.flags.DEFINE_string("gpu", '5', "id of gpu")

# os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(asctime)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler("../logs/aliccp/" + MODEL_NAME  + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("FLAGS: " + str(FLAGS))

spec = json.load(open(FLAGS.data_dir + "spec.json"))


feature_descriptions = {}
for mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
    key_map = {
        tf.estimator.ModeKeys.TRAIN: "train",
        tf.estimator.ModeKeys.EVAL: "val",
        tf.estimator.ModeKeys.PREDICT: "test"        
    }

    feature_description = {
        'y': tf.io.FixedLenFeature([], tf.float32),
        'z': tf.io.FixedLenFeature([], tf.float32),
        'one_hot_fields': tf.io.FixedLenFeature([len(spec["one_hot_fields"])], tf.int64)
    }
    for mul_fields in spec["multi_hot_fields"]:
        feature_description[mul_fields] = tf.io.FixedLenFeature([spec[f"{key_map[mode]}_max_length"][mul_fields]], tf.int64)
    for mul_fields in spec["special_fields"]:
        feature_description[mul_fields] = tf.io.FixedLenFeature([spec[f"{key_map[mode]}_max_length"][mul_fields]], tf.int64)
    feature_descriptions[mode] = feature_description

def parse_example(mode, example):
    parsed_exapmle = tf.io.parse_example(example, feature_descriptions[mode])
    input = {}
    target = {"y": parsed_exapmle["y"], "z": parsed_exapmle["z"]}
    for index, key in enumerate(spec["one_hot_fields"]):
        input[key] = parsed_exapmle["one_hot_fields"][:, index]
    for key in spec["multi_hot_fields"]:
        input[key] = parsed_exapmle[key]
    for key in spec["special_fields"]:
        input[key] = parsed_exapmle[key]
    return input, target

# def input_fn(file_pattern, mode, batch_size=32, num_epochs=1, perform_shuffle=False):
#     file_names = glob.glob(file_pattern)
#     file_names.sort()
#     dataset = tf.data.TFRecordDataset(
#         file_names, buffer_size=batch_size * 32, num_parallel_reads=20
#     )
#     if perform_shuffle:
#         dataset = dataset.shuffle(buffer_size=batch_size * 32)
        
#     dataset = dataset.repeat(num_epochs).batch(batch_size, drop_remainder=True).map(
#         partial(
#             parse_example,
#             mode,
#         ),
#         num_parallel_calls=10
#     ).prefetch(300)
#     return dataset

def input_fn(filenames, mode, batch_size=32, num_epochs=1, perform_shuffle=False):
    
    dataset = tf.data.TFRecordDataset(filenames)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=500000)
    
    dataset = dataset.repeat(num_epochs).batch(batch_size, drop_remainder=True).map(
        partial(
            parse_example,
            mode,
        ),
        num_parallel_calls=10
    ).prefetch(100)
    
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    
    return batch_features, batch_labels


def model_fn(features, labels, mode):
    """build Estimator model"""
    
    def embedding_lookup_sparse_fake(params, ids, combiner=None, name=None):
        dense_mask = tf.expand_dims(tf.cast(ids >= 0, tf.float32), axis=-1)
        ids = tf.where(tf.equal(ids, -1), tf.zeros_like(ids), ids)
        embedding = tf.nn.embedding_lookup(params, ids, name=name+"_dense_lookup") * dense_mask
        summed_embedding = tf.reduce_sum(embedding, axis=1)
        if combiner == "sum":
            return summed_embedding
        elif combiner == "mean":
            return summed_embedding / tf.reduce_sum(dense_mask, axis=1)
        else:
            raise ValueError("combiner only supoort 'sum', 'mean'")
    
    with tf.compat.v1.variable_scope("Embedding-Layer"):
        emb_weights = {}
        for key, vocab_len in spec["vocab_length"].items():
            emb_weights[key] = tf.compat.v1.get_variable(
                name=key + "_emb_wgts",
                shape=[vocab_len + 1, FLAGS.embedding_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
            )
        
        embeddings = {}
        for key in ["101", "121", "122", "124", "125", "126", "127", "128", "129", 
                    "205", "206", "207", "216", "508", "509", "702", "301"]:
            embeddings[key] = tf.nn.embedding_lookup(emb_weights[key], features[key], name=key + "_embedding_lookup")
            embeddings[key] = tf.reshape(embeddings[key], [-1, 1, FLAGS.embedding_size])
        for key in ["109_14", "110_14", "127_14", "150_14", "210", "853"]:
            embeddings[key] = tf.expand_dims(
                embedding_lookup_sparse_fake(emb_weights[key], features[key], combiner="sum", name=key + "_embedding_lookup"), 
                axis=1
            )
    
    embedding = tf.concat(
        [embeddings[field_name] for field_name in spec["one_hot_fields"]] + 
        [embeddings[field_name] for field_name in spec["multi_hot_fields"]] +
        [embeddings[field_name] for field_name in spec["special_fields"]],
        axis=2,
    )  # None * 1 * (23 * E)

    x_deep = tf.reshape(embedding, [-1, 23 * FLAGS.embedding_size])
    
    with tf.compat.v1.variable_scope("shared-bottom"):
        bottom_units = list(map(int, FLAGS.deep_layers.strip().split(',')))
        bottom_output = x_deep
        for i in range(len(bottom_units)):
            bottom_output = tf.contrib.layers.fully_connected(inputs=bottom_output, num_outputs=bottom_units[i], 
                                                              activation_fn=tf.nn.relu, scope='bottom_mlp_%d' % i)

    with tf.compat.v1.variable_scope("tower"):
        tower_units = list(map(int, FLAGS.tower_layers.strip().split(',')))
        
        def build_tower(tower_input, name):
            y_tower = tower_input
            for i in range(len(tower_units)):
                y_tower = tf.contrib.layers.fully_connected(inputs=y_tower, num_outputs=tower_units[i], 
                                                            activation_fn=tf.nn.relu, scope=name+'_tower_mlp_%d' % i)
            return y_tower

        # CTR
        y_ctr = build_tower(bottom_output, name='ctr')
        y_ctr = tf.contrib.layers.fully_connected(inputs=y_ctr, num_outputs=1, activation_fn=None, \
                                                scope='deep_out_click')
        y_ctr = tf.reshape(y_ctr, [-1, ])
        y_ctr_prediction = tf.sigmoid(y_ctr)

        # CVR
        y_cvr = build_tower(bottom_output, name='cvr')
        y_cvr = tf.contrib.layers.fully_connected(inputs=y_cvr, num_outputs=1, activation_fn=None, \
                                                scope='deep_out_valid_play')
        y_cvr = tf.reshape(y_cvr, [-1, ])
        y_cvr_prediction = tf.sigmoid(y_cvr)
    
    y_ctcvr_prediction = y_ctr_prediction * y_cvr_prediction

    # print('-' * 60)
    # count_params()
    # print('-' * 60)
    
    predictions = {
        "ctr": y_ctr_prediction,
        "cvr": y_cvr_prediction,
        "ctcvr": y_ctcvr_prediction
    }

    export_outputs = {
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions
        )
    }
    # Estimator predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, export_outputs=export_outputs
        )

    # ------build loss function------
    with tf.compat.v1.variable_scope("loss-function-part"):
        epsilon = 1e-7
        click_weight = 0.14
        conversion_weight = 0.023
        ctr_task_wgt = FLAGS.ctr_task_wgt
        
        ctr_loss = - (1 - click_weight) / click_weight * labels['y'] * tf.math.log(y_ctr_prediction + epsilon) - \
                    (1 - labels['y']) * tf.math.log(1 - y_ctr_prediction + epsilon)
        ctr_loss = tf.reduce_mean(ctr_loss)
        
        ctcvr_loss = - (1 - conversion_weight) / conversion_weight * labels['z'] * tf.math.log(y_ctcvr_prediction + epsilon) - \
                    (1 - labels['z']) * tf.math.log(1 - y_ctcvr_prediction + epsilon)
        ctcvr_loss = tf.reduce_mean(ctcvr_loss)
        
        loss = ctr_task_wgt * ctr_loss + (1 - ctr_task_wgt) * ctcvr_loss

    # Provide an estimator spec for `ModeKeys.EVAL`
    if mode == tf.estimator.ModeKeys.EVAL:
        ctr_mask = labels["y"] > 0
        cvr_labels = tf.boolean_mask(labels["z"], ctr_mask)
        cvr_pre = tf.boolean_mask(y_cvr_prediction, ctr_mask)
        
        eval_metric_ops = {
            "auc_ctr": tf.compat.v1.metrics.auc(labels["y"], y_ctr_prediction),
            "auc_cvr": tf.compat.v1.metrics.auc(cvr_labels, cvr_pre),
            "auc_ctcvr": tf.compat.v1.metrics.auc(labels["z"], y_ctcvr_prediction)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
        )

    # ------bulid optimizer------
    if FLAGS.optimizer == "Adam":
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8
        )
    elif FLAGS.optimizer == "Adagrad":
        optimizer = tf.compat.v1.train.AdagradOptimizer(
            learning_rate=FLAGS.learning_rate, initial_accumulator_value=1e-6
        )
    elif FLAGS.optimizer == "Momentum":
        optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=FLAGS.learning_rate, momentum=0.95
        )
    elif FLAGS.optimizer == "SGD":
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

    # train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    
    gvs = optimizer.compute_gradients(loss)

    def ClipIfNotNone(grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -1, 1)

    clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(clipped_gradients, global_step=tf.compat.v1.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op
        )


def main(_):
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + (date.today() + timedelta(-1)).strftime('%Y%m%d')
    
    # ------init Envs------
    # print("dataset dir", FLAGS.data_dir)
    # tr_dir = os.path.join(FLAGS.data_dir, "train/data_train.csv.tfrecord.*")
    # print("tr_dir:", tr_dir)
    # va_dir = os.path.join(FLAGS.data_dir, "val/data_val.csv.tfrecord.*")
    # print("va_dir:", va_dir)
    # te_dir = os.path.join(FLAGS.data_dir, "test/data_test.csv.tfrecord.*")
    # print("te_dir:", te_dir)
    
    train_order = json.load(open("./order.json"))
    tr_files = ["%strain/data_train.csv.tfrecord.%s" % (FLAGS.data_dir, index) for index in train_order["reading_order"]]
    print("tr_files:", tr_files)
    va_files = glob.glob("%sval/data_val.csv.tfrecord.*" % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%stest/data_test.csv.tfrecord.*" % FLAGS.data_dir)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing code cleaned at %s" % FLAGS.model_dir)
    spec = json.load(open(FLAGS.data_dir + "spec.json"))
    
    # ------ for NPU  ------
    config = NPURunConfig(
        model_dir=FLAGS.model_dir,
        log_step_count_steps=100, save_summary_steps=100,
        save_checkpoints_steps=spec["dataset_size"]["train"] // FLAGS.batch_size + 1,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    )
    model = NPUEstimator(model_fn=model_fn, model_dir=FLAGS.model_dir, config=config)

    hook = tf.estimator.experimental.stop_if_no_increase_hook(model, "auc_ctr", max_steps_without_increase=spec["dataset_size"]["train"] // FLAGS.batch_size,
                                                         run_every_secs=None, run_every_steps=10)
    hook_stop = tf.estimator.StopAtStepHook(last_step=200)

    if FLAGS.task_type == "train":
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=None, batch_size=FLAGS.batch_size, perform_shuffle=True, mode=tf.estimator.ModeKeys.TRAIN),
            hooks=[hook]
        )

        test_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size, mode=tf.estimator.ModeKeys.EVAL),
            steps=None,
            start_delay_secs=10,
            throttle_secs=0
        )
        logger.info("start train and evaluate")
        tf.estimator.train_and_evaluate(model, train_spec, test_spec)
        logger.info("early stopped, start evaluating....")
        model.evaluate(
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size, mode=tf.estimator.ModeKeys.PREDICT),
                           checkpoint_path=get_third_nearest_checkpoint(model.model_dir))
    
    elif FLAGS.task_type == "eval":
        model.evaluate(
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size)
        )
    
    elif FLAGS.task_type == 'infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size, mode=tf.estimator.ModeKeys.PREDICT),
                               predict_keys=["ctr", "cvr", "ctcvr"], hooks=[])
        with open(FLAGS.data_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\t%f\t%f\n" % (prob['ctr'], prob['cvr'], prob['ctcvr']))
    
    elif FLAGS.task_type == 'profiling_train':
        model.train(input_fn=lambda: input_fn(tr_files, num_epochs=1, batch_size=FLAGS.batch_size, perform_shuffle=True, mode=tf.estimator.ModeKeys.TRAIN),
                        hooks=[hook_stop])
    
    elif FLAGS.task_type == 'profiling_infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size, mode=tf.estimator.ModeKeys.PREDICT),
                               predict_keys=["ctr", "cvr", "ctcvr"], hooks=[hook_stop])
        with open(FLAGS.data_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\t%f\t%f\n" % (prob['ctr'], prob['cvr'], prob['ctcvr']))


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run()
