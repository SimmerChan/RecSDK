import json
import os
import argparse
import math

parser = argparse.ArgumentParser(description='Parse arguments')
parser.add_argument("--length", type=float, default=math.inf, help="max length for sequence fields")
parser.add_argument("--proc", type=int, default=1, help="num of working processes")
parser.add_argument("--padding", type=bool, default=False, help="generate padded dataset")
args = parser.parse_args()
args.length = math.inf if args.length == -1 else args.length

obj = json.load(open("keymap_train.json", "r"))
new_dict = dict()
for key in obj.keys():
    arr = []
    for k, v in obj[key].items():
        if v <= 1:
            continue
        arr.append(int(k))
    arr.sort()
    new_dict[key] = arr
    
json_str = json.dumps(new_dict, indent=4, sort_keys=True)
with open("keymap_train_pruned.json", "w") as json_file:
    json_file.write(json_str)
    
for key in obj.keys():
    if not os.path.exists("./aliccp_out/vocab"): 
        os.makedirs("./aliccp_out/vocab")
    f= open("./aliccp_out/vocab/vocab_" + key, "w")
    f.writelines([f"{val}\n" for val in new_dict[key]])
    f.close()

