import re
import glob
import numpy as np
import tensorflow as tf

def get_third_nearest_checkpoint(path):
    filenames = glob.glob(path + '/model.ckpt-*.index')
    pattern = re.compile(r'model.ckpt-(.*?).index', re.S)
    versions = []
    for filename in filenames:
        versions += [int(re.findall(pattern, filename)[0])]
    versions = sorted(versions)
    return path+'/model.ckpt-'+str(versions[-3])

def count_params():
    total_list = []
    emb_list = []
    for v in tf.compat.v1.trainable_variables():
        layer_name = v.name
        print(layer_name)
        layer_shape = v.get_shape().as_list()
        print(layer_shape)
        layer_prod = np.prod(layer_shape)
        # print(layer_prod)
        total_list.append(layer_prod)
        if 'emb' in layer_name:
            emb_list.append(layer_prod)
    total_num = np.sum(total_list)
    emb_num = np.sum(emb_list)
    print("Model size: %.2fM" % (total_num / 1e6))
    print("The proportion of non-Embedding: %.2f%%" % ((1 - emb_num / total_num) * 100))
