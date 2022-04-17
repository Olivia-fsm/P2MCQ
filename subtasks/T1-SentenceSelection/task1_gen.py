import argparse
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle
import pandas as pd
import numpy as np
import random
import json
import re
import os
import warnings
warnings.filterwarnings('ignore')


### Set Input Arguement ###
parser = argparse.ArgumentParser()

### Set Training Arguments ###
parser.add_argument('--task1_out_path', required=True, type=str,
                    help="path to subtask1 output (txt).")
parser.add_argument('--output_path', required=True, type=str,
                    help="path to save splited output.")
parser.add_argument('--sent_output_path', required=True, type=str,
                    help="path to save sentences output.")


def get_subtask1(task1_path):
    res_list = []
    with open(task1_path, 'r') as trg:
        for sent in trg.readlines():
            res_list.append(sent.split("<q>"))
    return res_list


if __name__ == '__main__':
    args = parser.parse_args()
    task1_list = get_subtask1(args.task1_out_path)
    with open(args.output_path, 'wb') as trg:
        pickle.dump(task1_list, trg)
    with open(args.sent_output_path, 'w') as trg:
        for x in task1_list:
            if len(x) >= 1:
                trg.write(x[0].replace('\n', '')+'\n')
            else:
                trg.write("\n")
            if len(x) >= 2:
                trg.write(x[1].replace('\n', '')+'\n')
            else:
                trg.write("\n")
            if len(x) >= 3:
                trg.write(x[2].replace('\n', '')+'\n')
            else:
                trg.write("\n")
