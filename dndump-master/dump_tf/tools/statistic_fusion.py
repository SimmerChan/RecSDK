import argparse
#####################################config##########################################
def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help=
        "0:pb 1:pbtxt 2:onnx , choices=[0,1,2],default:0"
    )
    parser.add_argument("--pb", default=None,help="tf pb path. --type=0")
    parser.add_argument("--pbtxt", default=None,help="tf pbtxt path . --type=1")
    parser.add_argument("--onnx", default=None,help="onnx file path . --type=2")
    parser.add_argument("--fusionop_csv",default=None,help="fusion_op_x_x_x.csv")
    args = parser.parse_args()
    return args
args = get_args()
#####################################config##########################################


op_dict={}
if args.type==0:
    model_pb=args.pb
elif args.type==1:
    model_pb=args.pbtxt
elif args.type==2:
    model_pb=args.onnx
else:
    model_pb=args.pb

#parse onnx
def parse_onnx(model_pb=None):
    if model_pb is None:
        return {}
    import onnx
    original_model = onnx.load(model_pb)
    for i in range(len(original_model.graph.node)):
        node_name = original_model.graph.node[i].name if not original_model.graph.node[i].name == '' else \
        original_model.graph.node[i].op_type + '_' + original_model.graph.node[i].output[0]
        op_dict[node_name]=original_model.graph.node[i].op_type
    for i in range(len(original_model.graph.initializer)):
        op_dict[original_model.graph.initializer[i].name] = 'Const'
    return op_dict



#parse pb or pbtxt
def parse_pb(model_pb=None):
    if model_pb is None:
        return {}
    from tensorflow.python.platform import gfile
    import tensorflow as tf
    with gfile.FastGFile(model_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) if args.type==0 else text_format.Merge(f.read(), graph_def)
        try:
            tf.import_graph_def(graph_def, name='')
        except Exception as e:
            print("tf.import_graph_def [", model_pb, "] failed!")
            print(e)
    all_ops = tf.get_default_graph().get_operations()
    for index, op in enumerate(all_ops):
        op_dict[op.node_def.name]=op.type
    return op_dict

#read fusion_op_x_x_x.csv
data=[]
update_flag=True
import csv
if args.fusionop_csv is not None:
    f = csv.reader(open(args.fusionop_csv,'r'))
    for i in f:
        type_str=None
        if len(i) == 12:
            update_flag=False
            #print('{} has found [Fusion Type]! no need add !'.format(args.fusionop_csv))
            break
        if i[4] == "Original Ops":
            op_dict=parse_onnx(model_pb) if args.type==2 else parse_pb(model_pb)
            i.append("Fusion Type")
            data.append(tuple(i))
            continue
        for j in i[4].split(','):
            tmp=None
            if j in op_dict.keys():
                tmp = op_dict[j]
            elif j.split("_")[0] in op_dict.keys():
                tmp = op_dict[j.split("_")[0]]
            else:
                for i_key in op_dict.keys():
                    if i_key in j:
                        tmp=op_dict[i_key]
                        break
                    else:
                        tmp=None
                if tmp is None:
                    tmp = j.split("/")[-1].split("_")[0]
            type_str = tmp if type_str is None else type_str + "+" + tmp
        i.append(type_str)
        data.append(tuple(i))

# write fusion_op_x_x_x.csv
if update_flag:
    f_w = open(args.fusionop_csv,'w')
    writer = csv.writer(f_w)
    for i in data:
        writer.writerow(i)
    f_w.close()
    #print('{} has added [Fusion Type]!'.format(args.fusionop_csv))
