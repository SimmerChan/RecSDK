import os
import argparse
from graph_partition import GraphPartitioner
import tensorflow as tf

template = \
    '''
    platform_configs {
    key: "tensorflow"
      value {
          source_adapter_config {
            [type.googleapis.com/tensorflow.serving.SaveModelBundleSourceAdapterConfig] {
              legacy_config {
    
        }
        }
        }
      }
    }
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', type=str, default='./')
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--output_filename', type=str, default='config.cfg')
    args = parser.parse_args()

    signature_def = "serving_default"

    # 模型配置
    embedding_lookup_op_type = ["Sum"]
    heavy_load_ops = ["MatMul"]  # 必须下沉的算子（暂时没用到）
    use_whole_graph = False
    partition_to_first_heavy_load = False
    #########################################################

    out_filepath = os.path.join(args.output_path, args.output_filename)

    print("Try to load model from {}...".format(args.model_path))

    with tf.compat.v1.Session() as sess:
        try:
            meta_gragh = tf.compat.v1.saved_model.loader.load(sess, ["serve"], args.model_path)

        except Exception as e:
            print("Error when try to load model, will try to partition graph anyway!", e)

        print("Load Model Done!")
        print("Try to generate sub graph...")
        ops = sess.graph.get_operations()
        graph_partitioner = GraphPartitioner()

        graph_partitioner.graph = sess.graph
        graph_partitioner.signature_def = meta_gragh.signature_def.get(signature_def)
        graph_partitioner.set_emedding_lookup_op_type(embedding_lookup_op_type)

        inputs, outputs = graph_partitioner.get_sub_graph()

        print("Sub graph is generated!")

    # 这里后续根据输出文件生成subs
    print("Generation cfg file...")
    print(inputs, outputs)

    res_string = "[[" + inputs + "," + outputs + "]]"

    output = template.replace("#value@in_out_pair#", res_string)
    if os.path.exists(out_filepath):
        os.remove(out_filepath)

    # open text file
    text_file = open(out_filepath, "w")

    # write string to file
    n = text_file.write(output)

    # close file
    text_file.close()
    print("Finished!")



