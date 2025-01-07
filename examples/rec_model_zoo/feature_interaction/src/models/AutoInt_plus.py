# coding=utf-8
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import shutil
import random
import logging
import tensorflow as tf
from datetime import date, timedelta, datetime

# from hook_train import OperatorTimingHook as train_operator_hook
# from hook_infer import OperatorTimingHook as infer_operator_hook
from utils import get_third_nearest_checkpoint, count_params

from npu_bridge.npu_init import *

MODEL_NAME = "AutoInt_plus"

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("feature_size", 2100000, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 39, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 10, "Embedding size")
tf.app.flags.DEFINE_integer("train_size", 33003326, "Number of instances in the train set")
tf.app.flags.DEFINE_integer("batch_size", 4096, "Number of batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '400,400,400', "deep layers")
tf.app.flags.DEFINE_integer("attention_layers", 3, "Number of attention layers")
tf.app.flags.DEFINE_integer("att_embedding_size", 5, "Size of attention embedding")
tf.app.flags.DEFINE_integer("heads_num", 2, "Number of attention heads")
tf.app.flags.DEFINE_string("data_dir", '../data/criteo/', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", f'../checkpoint/criteo/{MODEL_NAME}/',"model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type")
tf.app.flags.DEFINE_boolean("clear_existing_model", True, "clear existing model or not")
# tf.app.flags.DEFINE_string("gpu", '5', "id of gpu")

# os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(asctime)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler("../logs/" + FLAGS.data_dir.split('/')[-2] + '/' + MODEL_NAME + "_" + \
                         datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info("FLAGS: " + str(FLAGS))

# ------ Load tfrecord dataset ------
def input_fn(filenames, batch_size=32, field_size=39, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    def extract_fn(data_record):
        features = {
             # Extract features using the keys set during creation
            'label': tf.io.FixedLenFeature(shape=(), dtype=tf.float32),
            'ids': tf.io.FixedLenFeature(shape=(field_size,), dtype=tf.int64),
            'values': tf.io.FixedLenFeature(shape=(field_size,), dtype=tf.float32),
        }
        sample = tf.io.parse_example(data_record, features)
        sample['ids'] = tf.cast(sample['ids'], dtype=tf.int32)
        return {"feat_ids": sample['ids'], "feat_vals": sample['values']}, sample['label']
    
    dataset = tf.data.TFRecordDataset(filenames)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=500000)
    
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True).map(extract_fn, num_parallel_calls=10).prefetch(100)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    learning_rate = params["learning_rate"]
    attention_layers = params["attention_layers"]
    # att_size = params["att_embedding_size"]
    heads_number = params["heads_num"]
    att_size = embedding_size / heads_number
    layers  = list(map(int, params["deep_layers"].split(',')))

    # ------bulid weights------
    Feat_Emb_deep = tf.compat.v1.get_variable(name="emb_deep", shape=[feature_size, embedding_size],
                               initializer=tf.random_normal_initializer(stddev=0.1), )
    
    # ------build feature-------
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    # ------build f(x)------
    with tf.compat.v1.variable_scope("Embedding-Layer"):
        embeddings_origin_deep = tf.nn.embedding_lookup(Feat_Emb_deep, feat_ids)   # None * F * E
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])               # None * F * 1
        embeddings = tf.multiply(embeddings_origin_deep, feat_vals)
    
    with tf.compat.v1.variable_scope("Multihead-Attention-Layer", reuse=tf.compat.v1.AUTO_REUSE):
        def multihead_attention(x, embedding_dim=10, att_embedding_size=5, heads_num=2, layer_index=0):
            
            w_Q = tf.compat.v1.get_variable(name="weight_Q_%d" % layer_index, 
                                            shape=[embedding_dim, att_embedding_size * heads_num],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
            w_K = tf.compat.v1.get_variable(name="weight_K_%d" % layer_index, 
                                            shape=[embedding_dim, att_embedding_size * heads_num],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
            w_V = tf.compat.v1.get_variable(name="weight_V_%d" % layer_index, 
                                            shape=[embedding_dim, att_embedding_size * heads_num],
                                            initializer=tf.random_normal_initializer(stddev=0.1), )
            
            w_Res = tf.compat.v1.get_variable(name="weight_Res_%d" % layer_index, 
                                              shape=[embedding_dim, att_embedding_size * heads_num],
                                              initializer=tf.random_normal_initializer(stddev=0.1), )

            # None * F * (D * head_num)
            query = tf.tensordot(x, w_Q, axes=(-1, 0))
            key = tf.tensordot(x, w_K, axes=(-1, 0))
            value = tf.tensordot(x, w_V, axes=(-1, 0))

            # head_num * None * F * D
            query = tf.stack(tf.split(query, heads_num, axis=2))
            key = tf.stack(tf.split(key, heads_num, axis=2))
            value = tf.stack(tf.split(value, heads_num, axis=2))
            
            inner_product = tf.matmul(query, key, transpose_b=True)       # head_num * None * F * F
            inner_product /= att_embedding_size ** 0.5
            
            normalized_att_scores = tf.nn.softmax(inner_product, axis=-1)

            result = tf.matmul(normalized_att_scores, value)              # head_num * None * F * D
            result = tf.concat(tf.split(result, heads_num, ), axis=-1)
            result = tf.squeeze(result, axis=0)                           # None * F * (D * head_num)

            result += tf.tensordot(x, w_Res, axes=(-1, 0))
            result = tf.nn.relu(result)

            return result
        
        attention_part = embeddings
        
        for i in range(attention_layers):
            attention_part = multihead_attention(x=attention_part, embedding_dim=embedding_size, 
                                                 att_embedding_size=att_size, heads_num=heads_number,
                                                 layer_index=i)
                
    with tf.compat.v1.variable_scope("FC-Layer"):
        fc_inputs = tf.reshape(attention_part, shape=[-1, field_size * embedding_size])

        y = tf.contrib.layers.fully_connected(inputs=fc_inputs, num_outputs=1, activation_fn=tf.identity, \
                                                       scope='fc_out')
    
    with tf.compat.v1.variable_scope("Deep-Layer"):
        deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])    # None * (F * E)
        
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], \
                                                            scope='mlp%d' % i)
        
        y_mlp = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                                                       scope='mlp_out')
    
    y += y_mlp
    y = tf.reshape(y, shape=[-1])
    
    pred = tf.sigmoid(y)
    
    # print('-' * 60)
    # count_params()
    # print('-' * 60)

    predictions = {"prob": pred}
    export_outputs = {
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))

    # Provide an estimator spec for `ModeKeys.EVAL`
    log_loss = tf.compat.v1.losses.log_loss(labels, pred)
    auc_metric = tf.compat.v1.metrics.auc(labels, pred)
    loss_metric = tf.compat.v1.metrics.mean(log_loss)
    eval_metric_ops = {
        "auc": tf.compat.v1.metrics.auc(labels, pred),
        "logloss": tf.compat.v1.metrics.mean(log_loss),
        "stop_criterion": (auc_metric[0]-loss_metric[0], tf.group(auc_metric[1],loss_metric[1]))
    }

    # ------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.compat.v1.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            train_op=train_op)

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)

def main(_):
    # ------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('attention_layers ', FLAGS.attention_layers)
    print('att_embedding_size ', FLAGS.att_embedding_size)
    print('heads_num ', FLAGS.heads_num)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)

    # ------init Envs------
    tr_files = glob.glob("%s/tr*tfrecords" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/va*tfrecords" % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%s/te*tfrecords" % FLAGS.data_dir)
    print("te_files:", te_files)
    
    train_size = FLAGS.train_size

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    # ------bulid Tasks------
    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "deep_layers": FLAGS.deep_layers,
        "attention_layers": FLAGS.attention_layers,
        "att_embedding_size": FLAGS.att_embedding_size,
        "heads_num": FLAGS.heads_num
    }
    
    # ------ for NPU  ------
    config = NPURunConfig(
        model_dir=FLAGS.model_dir,
        log_step_count_steps=100, save_summary_steps=100,
        save_checkpoints_steps=train_size // FLAGS.batch_size + 1,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    )
    estimator = NPUEstimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)
    
    hook = tf.estimator.experimental.stop_if_no_increase_hook(estimator, "stop_criterion", max_steps_without_increase=train_size // FLAGS.batch_size,
                                                         run_every_secs=None, run_every_steps=10)
    hook_stop = tf.estimator.StopAtStepHook(last_step=200)
    os.makedirs(estimator.eval_dir())
    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=None, batch_size=FLAGS.batch_size, field_size=FLAGS.field_size, perform_shuffle=True),
            hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size, field_size=FLAGS.field_size), steps=None,
            start_delay_secs=10, throttle_secs=0)
        logger.info("start train and evaluate")
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        logger.info("Early stopped, start evaluation...")
        estimator.evaluate(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size, field_size=FLAGS.field_size),
                           checkpoint_path=get_third_nearest_checkpoint(estimator.model_dir))
    
    elif FLAGS.task_type == 'eval':
        estimator.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size, field_size=FLAGS.field_size))
    
    elif FLAGS.task_type == 'infer':
        preds = estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size, field_size=FLAGS.field_size),
                               predict_keys="prob")
        with open(FLAGS.data_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    
    elif FLAGS.task_type == 'profiling_train':
        estimator.train(input_fn=lambda: input_fn(tr_files, num_epochs=1, batch_size=FLAGS.batch_size, field_size=FLAGS.field_size, perform_shuffle=True),
                        hooks=[hook_stop])
    
    elif FLAGS.task_type == 'profiling_infer':
        preds = estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size, field_size=FLAGS.field_size),
                               predict_keys="prob", hooks=[hook_stop])
        with open(FLAGS.data_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run()