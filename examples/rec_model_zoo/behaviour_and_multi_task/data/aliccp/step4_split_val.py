import argparse
import math

import numpy as np

parser = argparse.ArgumentParser(description='Parse arguments')
parser.add_argument("--length", type=float, default=math.inf, help="max length for sequence fields")
parser.add_argument("--proc", type=int, default=1, help="num of working processes")
parser.add_argument("--padding", type=bool, default=False, help="generate padded dataset")
args = parser.parse_args()
args.length = math.inf if args.length == -1 else args.length
np.random.seed(2024)


def iter_count(file_name):
    from itertools import takewhile, repeat
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count("\n") for buf in buf_gen)


def random_true_false(length: int, num_of_true: int):
    return np.random.permutation(length) < num_of_true


if __name__ == "__main__":
    line_count = iter_count("./sample_skeleton_test_parsed.csv")
    random_arr = random_true_false(line_count, int(line_count * 0.5))
    testfile = open("sample_skeleton_test_splitted_parsed.csv", "w")
    valfile = open("sample_skeleton_val_splitted_parsed.csv", "w")
    with open("sample_skeleton_test_parsed.csv") as f:
        p = 0
        while True:
            lines = f.readlines(int(1e7))
            if len(lines) == 0:
                break
            for line in lines:
                if random_arr[p]:
                    testfile.write(line)
                else:
                    valfile.write(line)
                p += 1
    testfile.close()
    valfile.close()
