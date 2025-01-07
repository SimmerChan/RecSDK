import json
import sys
from collections import *

import argparse
import math
parser = argparse.ArgumentParser(description='Parse arguments')
parser.add_argument("--length", type=float, default=math.inf, help="max length for sequence fields")
parser.add_argument("--proc", type=int, default=1, help="num of working processes")
parser.add_argument("--padding", type=bool, default=False, help="generate padded dataset")
args = parser.parse_args()
args.length = math.inf if args.length == -1 else args.length

def parse_data(file_name, index_dict_in = None):
    print(f"begin to parse {file_name}")
    index_dict_local = dict()
    with open(file_name, "r") as f:
        rlines = f.readlines()
        for line in rlines:
            field_dict = defaultdict(int)
            line = line.strip().split(",")
            feat_len = len(line)
            # common_feature_index|feat_num|feat_list
            if feat_len == 3:
                if line[0] not in index_dict_in["common_index"]:
                    print(f"index {line[0]} not in common index")
                    continue

                shown_nums = index_dict_in["common_index"][line[0]]

                feat_strs = line[2]
                for fstr in feat_strs.split("\x01"):
                    field, feat_val = fstr.split("\x02")
                    feat, val = feat_val.split("\x03")
                    if field_dict[field] >= args.length:
                        continue
                    else:
                        field_dict[field] += 1
                    if not field in index_dict_in:
                        index_dict_in[field] = dict()
                    if not feat in index_dict_in[field]:
                        index_dict_in[field][feat] = 0
                    index_dict_in[field][feat] += shown_nums

            # sample_id|y|z|common_feature_index|feat_num|feat_list
            elif feat_len == 6:
                # y=0 & z=1 filter
                if line[1] == "0" and line[2] == "1":
                    continue

                if not "common_index" in index_dict_local:
                    index_dict_local["common_index"] = dict()
                if not line[3] in index_dict_local["common_index"]:
                    index_dict_local["common_index"][line[3]] = 0
                index_dict_local["common_index"][line[3]] += 1

                feat_strs = line[5]
                for fstr in feat_strs.split("\x01"):
                    field, feat_val = fstr.split("\x02")
                    feat, val = feat_val.split("\x03")
                    if field_dict[field] >= args.length:
                        continue
                    else:
                        field_dict[field] += 1
                    if not field in index_dict_local:
                        index_dict_local[field] = dict()
                    if not feat in index_dict_local[field]:
                        index_dict_local[field][feat] = 0
                    index_dict_local[field][feat] += 1
    return index_dict_local

if __name__ == "__main__":
    # args = args.parse_args()
    all_field_id = [
        "101",
        "109_14",
        "110_14",
        "127_14",
        "150_14",
        "121",
        "122",
        "124",
        "125",
        "126",
        "127",
        "128",
        "129",
        "205",
        "206",
        "207",
        "210",
        "216",
        "508",
        "509",
        "702",
        "853",
        "301",
    ]
    train_data_path = "."
    test_data_path = "."
    index_dict = parse_data(train_data_path + "/sample_skeleton_train.csv")
    print("parse skeleton_train.csv successfully")
    parse_data(train_data_path + "/common_features_train.csv", index_dict)
    print("parse features_train.csv successfully")
    index_dict.pop("common_index")

    json_str = json.dumps(index_dict, indent=4)
    with open("keymap_train.json", "w") as json_file:
        json_file.write(json_str)
