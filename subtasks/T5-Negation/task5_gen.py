import torch
from tqdm import tqdm
import pickle
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
neg_model_name = 'minwhoo/bart-base-negative-claim-generation'
neg_tokenizer = AutoTokenizer.from_pretrained(neg_model_name)
NEG_MODEL = AutoModelForSeq2SeqLM.from_pretrained(neg_model_name).to(DEVICE)
# NEG_MODEL

# examples = [
#     "Little Miss Sunshine was filmed over 30 days.",
#     "Magic Johnson did not play for the Lakers.",
#     "Claire Danes is wedded to an actor from England.",
#     "In all conditions we measure participant learning, collaboration and attitudes."
# ]

### Set Input Arguement ###
parser = argparse.ArgumentParser()

### Set Training Arguments ###
parser.add_argument('--task1_out_path', required=True, type=str,
                    help="path to subtask1 output (txt).")
parser.add_argument('--output_path', required=True, type=str,
                    help="path to save paraphrase output.")
parser.add_argument('--verbose', default=False, required=False, type=bool,
                    help="whether output the result.")


def get_negative_candidate(neg_tokenizer, NEG_MODEL, examples, mode='crossAUG', verbose=False):
    batch = neg_tokenizer(examples, max_length=1024, padding=True,
                          truncation=True, return_tensors="pt").to(DEVICE)
    out = NEG_MODEL.generate(batch['input_ids'].to(
        DEVICE), num_beams=5, max_length=128)
    negative_examples = neg_tokenizer.batch_decode(
        out, skip_special_tokens=True)
    if verbose:
        print(negative_examples)
    if type(examples) == str:
        return negative_examples[0]
    return negative_examples


def get_subtask1(task1_path):
    res_list = []
    with open(task1_path, 'r') as trg:
        for sent in trg.readlines():
            res_list.append(sent.split("<q>"))
    return res_list


def get_doc_negation(task1_path, output_path):
    subtask5_output_list = []

    for sent_list in tqdm(get_subtask1(task1_path)):
        negs = get_negative_candidate(
            neg_tokenizer=neg_tokenizer, NEG_MODEL=NEG_MODEL, examples=sent_list)
        subtask5_output_list.append(negs)
    with open(output_path, 'wb') as trg:
        pickle.dump(subtask5_output_list, trg)
    return subtask5_output_list


def print_output(input_list, output_list):
    assert len(input_list) == len(output_list)
    for idx, (in_file, out_file) in enumerate(zip(input_list, output_list)):
        print(f"======== Doc {idx} =========")
        for i, o in zip(in_file, out_file):
            print(f"INPUT:  {i}")
            print(f"OUTPUT: {o}")
            print()


if __name__ == '__main__':
    args = parser.parse_args()
    res_list = get_subtask1(args.task1_out_path)
    real_output = get_doc_negation(
        args.task1_out_path, output_path=args.output_path)
    if args.verbose:
        print_output(res_list, real_output)
