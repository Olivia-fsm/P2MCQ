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
parser.add_argument('--access_out_path', required=True, type=str,
                    help="path to subtask3 output by ACCESS model (txt).")
parser.add_argument('--output_path', required=True, type=str,
                    help="path to save splited output.")


def get_subtask3(task3_path):
    res_list = []

    with open(task3_path, 'r') as trg:
        cur_list = []
        for idx, sent in enumerate(trg.readlines()):
            if idx % 3 == 0 and idx != 0:
                res_list.append(cur_list)
            else:
                if len(sent.replace("\n", '')) > 0:
                    cur_list.append(sent)
    return res_list


if __name__ == '__main__':
    args = parser.parse_args()
    task3_list = get_subtask3(args.task3_out_path)
    with open(args.output_path, 'wb') as trg:
        pickle.dump(task3_list, trg)
