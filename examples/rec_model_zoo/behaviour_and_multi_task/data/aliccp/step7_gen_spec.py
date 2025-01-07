#!/usr/bin/env python
#coding=utf-8

import os
import glob
import json
flags = {
    "input_dir": ".",
    "output_dir": "./aliccp_out"
}
fields = ["101", "109_14", "110_14", "127_14", "150_14", "121", "122", "124", "125", "126", "127", "128", "129", "205", "206", "207", "210", "216", "508", "509", "702", "853", "301"]
multi_hot_fields = set(["109_14", "110_14", "127_14", "150_14"])
special_fields = set(["210", "853"])
one_hot_fields = list(filter(lambda x: x not in multi_hot_fields and x not in special_fields, fields))


def iter_count(file_name):
    from itertools import takewhile, repeat
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count("\n") for buf in buf_gen)

if __name__ == "__main__":
    file_spec = {
        "one_hot_fields": one_hot_fields,
        "multi_hot_fields": list(multi_hot_fields),
        "special_fields": list(special_fields),
        "vocab_length": {},
        "parts": {}
    }
    
    file_list = list(glob.glob(os.path.join(flags["output_dir"], "vocab", "vocab_*")))
    for file in file_list:
        key = os.path.basename(file).split("_", 1)[1]
        file_spec["vocab_length"][key] = iter_count(file)
    
    file_list = list(map(os.path.basename, glob.glob(os.path.join(flags["output_dir"], "*/*.tfrecord.*")))) 
    file_list.sort()
    for file in file_list:
        patterns = file.split(".")    
        part, index = patterns[0].split("_")[1], patterns[-1]
        if part not in file_spec["parts"]:
            file_spec["parts"][part] = []
        file_spec["parts"][part].append(int(index))
    
    for key, value in file_spec["parts"].items():
        value.sort()
        for i, v in enumerate(value):
            assert i==v, f"{part} index {i} not found"
        file_spec["parts"][key] = len(value)
    
    print(file_spec)
    print("still processing, please wait ...")
    file_spec["dataset_size"] = {
        "train":  "sample_skeleton_train_parsed.csv",
        "val":  "sample_skeleton_val_splitted_parsed.csv",
        "test":  "sample_skeleton_test_splitted_parsed.csv"
    }
    for key, value in file_spec["dataset_size"].items():
        file_spec["dataset_size"][key] = iter_count(value)

    for key in ["train", "val", "test"]:
        filename = f"data_{key}_max_length.json"
        file_spec[f"{key}_max_length"] = json.load(open(filename, "r"))

    print(file_spec)
    s = json.dump(file_spec, open(flags["output_dir"] + "/spec.json", "w"), indent=2)