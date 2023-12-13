import os
import sys
import xlsxwriter
import datetime
from google.protobuf import text_format
import argparse
import networkx as nx


#####################################config##########################################
def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_list",type=str,nargs='*',help="args: file1.json file2.json")
    args = parser.parse_args()
    return args

args = get_args()
print(args)

if len(args.json_list) != 2:
    raise ValueError("--json_list args must be 2 json file path! eg: python fusion_compare.py --json_list file1.json file2.json ")


########################################################################################

def compare_json():
    return 0

##########################################Main-start#######################################


print('---------------------------------------------------------------')

