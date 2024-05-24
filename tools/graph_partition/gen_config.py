import os
import argparse
from graph_partition import GraphPartitioner
import tensorflow as tf

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

    output_filepath = os.path.join(args.output_path, args.output_filename)

    print("Try to load model from {}...".format(args.model_path))

    with tf.compat.v1.Session() as sess:
        try:
            meta_graph = tf.compat.v1.saved_model.loader.load(sess, ["serve"], args.model_path)

        except Exception as e:
            print("Error when try to load model, will try to partition graph anyway!", e)

        print("Load Model Done!")
        print("Try to generate sub graph...")
        ops = sess.graph.get_operations()
        graph_partitioner = GraphPartitioner()

        graph_partitioner.graph = sess.graph
        graph_partitioner.signature_def = meta_graph.signature_def.get(signature_def)
        graph_partitioner.set_embedding_lookup_op_type(embedding_lookup_op_type)

        inputs, outputs = graph_partitioner.get_sub_graph()

        print("Sub graph is generated!")

    # 这里后续根据输出文件生成subs
    print("Generating cfg file...")
    print(inputs, outputs)

    res_string = "[[" + inputs + "," + outputs + "]]"

    ori_test = open("template.cfg")
    template = ori_test.read()
    output = template.replace("#value@in_out_pair#", res_string)
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    # open text file
    text_file = open(output_filepath, "w")

    # write string to file
    n = text_file.write(output)

    # close file
    text_file.close()
    print("Finished!")



