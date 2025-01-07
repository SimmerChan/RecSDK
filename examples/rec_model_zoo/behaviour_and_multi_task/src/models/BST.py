# coding=utf-8

import os
import glob
import json
import random
import shutil
import logging
import numpy as np
import tensorflow as tf
from functools import partial
from datetime import date, timedelta, datetime

# from hook_train import OperatorTimingHook as train_operator_hook
# from hook_infer import OperatorTimingHook as infer_operator_hook
from utils import get_third_nearest_checkpoint, count_params

from npu_bridge.npu_init import *

tf.compat.v1.set_random_seed(2024)
np.random.seed(2024)
random.seed(2024)

MODEL_NAME = "BST"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_integer("transformer_layers", 1, "Number of transformer layers")
tf.app.flags.DEFINE_integer("att_embedding_size", 4, "Size of attention embedding")
tf.app.flags.DEFINE_integer("heads_num", 4, "Number of attention heads")
tf.app.flags.DEFINE_string("attention_layers", '80,40', "Attention Net mlp layers")
tf.app.flags.DEFINE_string("deep_layers", "512,256,128,64", "deep layers")
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
        dense_ids = {}
        dense_len = {}
        for key in ["101", "121", "122", "124", "125", "126", "127", 
                    "128", "129", "205", "508", "509", "702", "301"]:
            embeddings[key] = tf.nn.embedding_lookup(emb_weights[key], features[key], name=key + "_embedding_lookup")
            embeddings[key] = tf.reshape(embeddings[key], [-1, 1, FLAGS.embedding_size])
        
        embeddings["853"] = tf.expand_dims(
            embedding_lookup_sparse_fake(emb_weights["853"], features["853"], combiner="sum", name="853" + "_embedding_lookup"), 
            axis=1
        )
        
        for key in ["206", "207", "216"]:
            embeddings[key] = tf.nn.embedding_lookup(emb_weights[key], features[key], name=key + "_embedding_lookup")
        
        embeddings["210"] = embedding_lookup_sparse_fake(emb_weights["210"], features["210"], combiner="sum", name="210" + "_embedding_lookup")
        
        for key in ["109_14", "110_14", "127_14", "150_14"]:
            feature_dense = features[key]
            dense_ids[key] = feature_dense
            dense_mask = tf.expand_dims(tf.cast(feature_dense >= 0, tf.float32), axis=-1)                     # None * P * 1
            dense_len[key] = tf.reduce_sum(tf.cast(feature_dense >= 0, tf.int32), axis=1, keepdims=True)
            feature_dense = tf.where(tf.equal(feature_dense, -1), tf.zeros_like(feature_dense), feature_dense)
            emb = tf.nn.embedding_lookup(emb_weights[key], feature_dense, name=key + "_embedding_lookup")     # None * P * E
            emb = tf.multiply(emb, dense_mask)
            embeddings[key] = emb
    
    
    with tf.compat.v1.variable_scope("Transformer-layer", reuse=tf.compat.v1.AUTO_REUSE):
        transformer_num = FLAGS.transformer_layers
        heads_num = FLAGS.heads_num
        
        def positional_encoding(inputs, maxlen):
            E = inputs.get_shape().as_list()[-1]  # static
            N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # None * T

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
                for pos in range(maxlen)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # maxlen * E

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)
            
            outputs = outputs * E ** 0.5  # scale

            return outputs + inputs
                
        def positional_encoding_learn(inputs, maxlen, scope="position_encoding"):
            E = inputs.get_shape().as_list()[-1]  # static
            N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # None * T

            # First part of the PE function: sin and cos argument
            position_enc = tf.compat.v1.get_variable(scope, [maxlen, E], # W_shape
                                                     initializer=tf.contrib.layers.xavier_initializer())

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)
            
            outputs = outputs * E ** 0.5  # scale

            return outputs + inputs
        
        def layer_normalization(inputs, scope="ln"):
            epsilon = 1e-9
            gamma = tf.compat.v1.get_variable(name="gamma_%s" % scope, shape=[inputs.get_shape().as_list()[-1], ],
                                              initializer=tf.ones_initializer())
            beta = tf.compat.v1.get_variable(name="beta_%s" % scope, shape=[inputs.get_shape().as_list()[-1], ],
                                             initializer=tf.zeros_initializer())
            mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
            normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
            outputs = gamma * normalized + beta
            return outputs

        def transformer_layer(inputs, masks, att_embedding_size, heads_num, dropout_rate=0.2, layer_name="key", layer_index=0):
            num_units = att_embedding_size * heads_num
            embedding_size = int(inputs.get_shape().as_list()[-1])
            
            if num_units != embedding_size:
                raise ValueError(
                    "att_embedding_size * heads_num must equal the last dimension size of inputs,got %d * %d != %d" % (
                        att_embedding_size, heads_num, embedding_size))
            
            w_Q = tf.compat.v1.get_variable(name="weight_Q_%s_%d" % (layer_name, layer_index), shape=[embedding_size, num_units],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
            w_K = tf.compat.v1.get_variable(name="weight_K_%s_%d" % (layer_name, layer_index), shape=[embedding_size, num_units],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
            w_V = tf.compat.v1.get_variable(name="weight_V_%s_%d" % (layer_name, layer_index), shape=[embedding_size, num_units],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
            
            fw1 = tf.compat.v1.get_variable(name="fw1_%s_%d" % (layer_name, layer_index), shape=[num_units, 4 * num_units],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
            fw2 = tf.compat.v1.get_variable(name="fw2_%s_%d" % (layer_name, layer_index), shape=[4 * num_units, num_units],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
            
            queries = inputs
            keys = inputs
            query_masks = masks
            key_masks = masks

            query_masks = tf.sequence_mask(query_masks, tf.shape(inputs)[1], dtype=tf.float32)
            key_masks = tf.sequence_mask(key_masks, tf.shape(inputs)[1], dtype=tf.float32)
            query_masks = tf.squeeze(query_masks, axis=1)
            key_masks = tf.squeeze(key_masks, axis=1)
            
            Q = tf.tensordot(queries, w_Q, axes=(-1, 0))  # None * T_q * (D * h)
            K = tf.tensordot(keys, w_K, axes=(-1, 0))
            V = tf.tensordot(keys, w_V, axes=(-1, 0))

            # (h * None) * T_q * D
            Q_ = tf.concat(tf.split(Q, heads_num, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, heads_num, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, heads_num, axis=2), axis=0)
            
            # scaled_dot_product
            outputs = tf.matmul(Q_, K_, transpose_b=True)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            
            key_masks = tf.tile(key_masks, [heads_num, 1])

            # (h * None) * T_q * T_k
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)

            # (h * None) * T_q * T_k
            outputs = tf.where(tf.equal(key_masks, 1), outputs, paddings, )
            
            outputs -= tf.reduce_max(outputs, axis=-1, keepdims=True)
            outputs = tf.nn.softmax(outputs, axis=-1)
            query_masks = tf.tile(query_masks, [heads_num, 1])  # (h * None) * T_q
            # (h * None) * T_q * T_k
            query_masks = tf.tile(tf.expand_dims(
                query_masks, -1), [1, 1, tf.shape(keys)[1]])

            outputs *= query_masks
            
            if mode == tf.estimator.ModeKeys.TRAIN:
                outputs = tf.nn.dropout(outputs, rate=dropout_rate)
                
            # Weighted sum
            result = tf.matmul(outputs, V_)
            result = tf.concat(tf.split(result, heads_num, axis=0), axis=2)
            
            # res
            result += queries
            
            result = layer_normalization(result, scope="%s_%d_ln1" % (layer_name, layer_index))
            
            fw1 = tf.nn.leaky_relu(tf.tensordot(result, fw1, axes=[-1, 0]))
            if mode == tf.estimator.ModeKeys.TRAIN:
                fw1 = tf.nn.dropout(fw1, rate=dropout_rate)
            fw2 = tf.tensordot(fw1, fw2, axes=[-1, 0])
            
            result += fw2
            
            result = layer_normalization(result, scope="%s_%d_ln2" % (layer_name, layer_index))
            
            return result
        
        
        for key in ["109_14", "110_14", "127_14", "150_14"]:
            embeddings[key] = positional_encoding_learn(embeddings[key], maxlen=FLAGS.max_seq_len, scope=key + "_pos")
            for i in range(transformer_num):
                embeddings[key] = transformer_layer(inputs=embeddings[key],
                                                    masks=dense_len[key],
                                                    att_embedding_size=embeddings[key].get_shape().as_list()[-1] // heads_num,
                                                    heads_num=heads_num,
                                                    layer_name=key,
                                                    layer_index=i)

    with tf.compat.v1.variable_scope("Field-wise-Pooling-layer", reuse=tf.compat.v1.AUTO_REUSE):
        attention_layers = list(map(int, FLAGS.attention_layers.strip().split(',')))
        
        def attention_unit(a_xx_emb, ub_dense_id, ub_emb, masks, unit_name="targ_hist"):
            dense_mask = tf.sequence_mask(masks, tf.shape(ub_emb)[1])
            
            padded_dim = tf.shape(ub_dense_id)[1]
            
            ax_emb = tf.reshape(tf.tile(a_xx_emb,[1, padded_dim]), shape=[-1, padded_dim, FLAGS.embedding_size])     # None * E --> None * P * E
            x_inputs = tf.concat([ax_emb, ub_emb, ax_emb - ub_emb, ax_emb * ub_emb], axis=-1) 		                 # None * P * 4E
            for i in range(len(attention_layers)):
                x_inputs = tf.contrib.layers.fully_connected(inputs=x_inputs, num_outputs=attention_layers[i], 
                                                             activation_fn=tf.nn.leaky_relu, scope="att_fc_%s_%d" % (unit_name, i))
                
            att_wgt = tf.contrib.layers.fully_connected(inputs=x_inputs, num_outputs=1, 
                                                        activation_fn=None, scope="att_out_%s" % unit_name)  # None * P * 1
            att_wgt = tf.reshape(att_wgt, shape=[-1, 1, padded_dim])                                         # None * 1 * P
            paddings = tf.ones_like(att_wgt) * (-2 ** 32 + 1)                                                # None * 1 * P
            att_wgt = tf.where(dense_mask, att_wgt, paddings)                                                # None * 1 * P
            att_wgt = att_wgt / (ub_emb.get_shape().as_list()[-1] ** 0.5)
            att_wgt = tf.nn.softmax(att_wgt)                                                                 # None * 1 * P
            wgt_emb = tf.matmul(att_wgt, ub_emb)                                                             # None * 1 * E
        
            return wgt_emb
        
        for target_key, his_key in zip(
                ["206", "207", "216", "210"],
                ["109_14", "110_14", "127_14", "150_14"]
        ):
            embeddings[his_key] = attention_unit(a_xx_emb=embeddings[target_key],
                                                 ub_dense_id=dense_ids[his_key],
                                                 ub_emb=embeddings[his_key],
                                                 masks=dense_len[his_key],
                                                 unit_name="%s_%s" % (target_key, his_key))    # None * 1 * E
    
    for key in ["206", "207", "210", "216"]:
        embeddings[key] = tf.reshape(embeddings[key], [-1, 1, FLAGS.embedding_size])
    
    embedding = tf.concat(
        [embeddings[field_name] for field_name in spec["one_hot_fields"]] +
        [embeddings[field_name] for field_name in spec["multi_hot_fields"]] +
        [embeddings[field_name] for field_name in spec["special_fields"]],
        axis=2,
    )  # None * 1 * 23 * E)

    x_deep = tf.reshape(embedding, [-1, 23 * FLAGS.embedding_size])  # None * (23 * E)
    
    with tf.compat.v1.variable_scope("MLP-layer"):
        deep_layers = list(map(int, FLAGS.deep_layers.strip().split(',')))

        for i in range(len(deep_layers)):
            x_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=deep_layers[i],
                                                       activation_fn=tf.nn.leaky_relu, scope='mlp%d' % i)
    
    with tf.compat.v1.variable_scope("BST-out"):
        y_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=1, activation_fn=tf.identity, \
             scope='bst_out')
        y = tf.reshape(y_deep,shape=[-1])
        pred = tf.sigmoid(y)

    # print('-' * 60)
    # count_params()
    # print('-' * 60)
    
    predictions={"prob": pred}

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
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels["y"]))
        
        epsilon = 1e-7
        click_weight = 0.14
        
        ctr_loss = - (1 - click_weight) / click_weight * labels['y'] * tf.math.log(pred + epsilon) - \
                    (1 - labels['y']) * tf.math.log(1 - pred + epsilon)
        loss = tf.reduce_mean(ctr_loss)

    # Provide an estimator spec for `ModeKeys.EVAL`
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "auc_ctr": tf.compat.v1.metrics.auc(labels["y"], pred),
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
                               predict_keys="prob", hooks=[])
        with open(FLAGS.data_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    
    elif FLAGS.task_type == 'profiling_train':
        model.train(input_fn=lambda: input_fn(tr_files, num_epochs=1, batch_size=FLAGS.batch_size, perform_shuffle=True, mode=tf.estimator.ModeKeys.TRAIN),
                        hooks=[hook_stop])
    
    elif FLAGS.task_type == 'profiling_infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size, mode=tf.estimator.ModeKeys.PREDICT),
                               predict_keys="prob", hooks=[hook_stop])
        with open(FLAGS.data_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run()
