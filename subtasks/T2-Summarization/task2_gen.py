import nltk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import datasets
import torch
from collections import OrderedDict, Counter
from pprint import pprint
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
parser.add_argument('--model_path', required=True, type=str,
                    help="path to pretrained model.")
parser.add_argument('--datapath', required=True, type=str,
                    help="path to section-context data (pkl/csv).")
parser.add_argument('--trg_key', default="Context", required=False, type=str,
                    help="key to input paragraphs in the dataFrame/dict.")
parser.add_argument('--output_path', required=True, type=str,
                    help="path to save summary output.")
parser.add_argument('--verbose', default=False, required=False, type=bool,
                    help="whether output the result.")
parser.add_argument('--device', default="cpu", required=False, type=str,
                    help="whether output the result.")


def get_model(model_path, device):
    MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    TOKENIZER = AutoTokenizer.from_pretrained(model_path)

    # Set model parameters or use the default
    print(MODEL.config)
    return MODEL, TOKENIZER


def generate_summary(test_samples, model, tokenizer, encoder_max_length=512, gen_max_length=128):
    inputs = tokenizer(
        test_samples["context"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(
        input_ids, attention_mask=attention_mask, max_length=gen_max_length)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


def get_realset(datapath: str, trg_key="Context", candidate_key=None):
    data_format = datapath.split('.')[-1]
    if data_format == 'csv':
        real_DF = pd.read_csv(datapath)
        passages = list(real_DF[trg_key].values)
    else:
        with open(datapath, 'rb') as trg:
            real_DF = pickle.load(trg)
            passages = real_DF[trg_key]
    real_datasets = []

    if candidate_key is not None:
        real_testset = {
            'context': [],
            'candidate': []
        }
        for idx, (passage_list, candidate) in enumerate(zip(passages, real_DF[candidate_key])):
            real_testset['context'].append('\n'.join(passage_list))
            real_testset['candidate'].append(candidate)
            if (idx+1) % 20 == 0:
                real_datasets.append(real_testset)
                real_testset = {
                    'context': [],
                    'candidate': []
                }
        real_datasets.append(real_testset)
    else:
        real_testset = {
            'context': [],
        }
        for idx, passage_list in enumerate(passages):
            real_testset['context'].append('\n'.join(passage_list))

            # real_testset['candidate'].append(candidate)
            if (idx+1) % 20 == 0:
                real_datasets.append(real_testset)
                real_testset = {
                    'context': [],
                    # 'candidate': []
                }
        real_datasets.append(real_testset)
    # print(len(real_datasets))
    return real_datasets, passages


def get_real_output(real_datasets: list, output_path: str, MODEL):
    real_output = []
    for testset in real_datasets:
        out = generate_summary(testset, MODEL, TOKENIZER)[1]
        real_output.extend(out)
    if output_path is not None:
        with open(output_path, 'wb') as trg:
            pickle.dump(real_output, trg)
    return real_output


def print_output(input_list, output_list):
    assert len(input_list) == len(output_list)
    for idx, (i, o) in enumerate(zip(input_list, output_list)):
        print(f"======== {idx} =========")
        print(f"INPUT:  {i}")
        print(f"OUTPUT: {o}")
        print()

# real_datasets = get_realset()
# real_output = get_real_output()


if __name__ == "__main__":
    args = parser.parse_args()
    MODEL, TOKENIZER = get_model(args.model_path, args.device)
    real_datasets, passages = get_realset(args.datapath, trg_key=args.trg_key)
    real_output = get_real_output(
        real_datasets, output_path=args.output_path, MODEL=MODEL)
    if args.verbose:
        print_output(passages, real_output)
