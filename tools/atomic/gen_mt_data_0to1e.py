import logging
import numpy as np
import tensorflow as tf
import random

logging.getLogger().setLevel(logging.INFO)

np.random.seed(0)

line_per_sample = 10000
samples_num = 10000 * 800  #
sparse_feat_list = ['feat_ids']
# todo
sparse_feat_len = [100]

num = 0
import sys

hot_zhanbi = sys.argv[1:][0]
hot_zhanbi = float(hot_zhanbi)/10
logging.info("%f", hot_zhanbi)

tfpath = "/home/insert/data"+str(hot_zhanbi)
import os
if not os.path.exists(tfpath):
    os.mkdir(tfpath)
    
tfpath = "/home/insert/data"+str(hot_zhanbi)+"/tf"

part1 = np.array(random.sample(range(0, 2), 1)) 

def write_records(writer, line_cnt, file_cnt):
    features = {
        'label': tf.train.Feature(
            float_list=tf.train.FloatList(value=np.random.randint(2, size=line_per_sample).tolist()))
    }

    count = 0
    for i, sparse_feat in enumerate(sparse_feat_list):
        np.random.seed(count)
        # global num
        logging.info("===sparse=%s", sparse_feat)
        part2=np.array(random.sample(range(0 + 100 * line_per_sample * (10 * file_cnt + line_cnt), 100 * line_per_sample * (10 * file_cnt + line_cnt+1)), int(100 * line_per_sample * (1- hot_zhanbi))))
        features[sparse_feat] = tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=part1.astype(np.int64).tolist() * int(100 * line_per_sample * hot_zhanbi) + part2.astype(np.int64).tolist())
        )

        count += 1
    features = tf.train.Features(feature=features)
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())


def gen_tfrecords(tf_path):
    file_cnt = 0
    line_per_file = 10
    line_cnt = 0
    writer = tf.python_io.TFRecordWriter(f"{tf_path}_{file_cnt}.tfrecord")
    sample_cnt = 0
    while True:
        write_records(writer, line_cnt, file_cnt)
        line_cnt += 1
        sample_cnt += line_per_sample
        logging.info(f">>>>>>>>>>>>count {sample_cnt} end.")
        if sample_cnt == samples_num:
            break
        if line_cnt == line_per_file:
            file_cnt += 1
            line_cnt = 0
            writer.close()
            writer = tf.python_io.TFRecordWriter(f"{tf_path}_{file_cnt}.tfrecord")
    writer.close()


if __name__ == '__main__':
    gen_tfrecords(tf_path=tfpath)
