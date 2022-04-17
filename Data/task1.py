import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import spacy
import string
import re
import argparse
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
PUNCTUATIONS = string.punctuation


### Set Input Arguement ###
parser = argparse.ArgumentParser()

### Set Training Arguments ###
parser.add_argument('--input_path', required=True, type=str,
                    help="path to input section-context pairs.")
parser.add_argument('--tgt_path', default=None, required=False, type=str,
                    help="path to grount-truth extracted sentences.")
parser.add_argument('--src_write_into', required=True, type=str,
                    help="path to save source sentences.")
parser.add_argument('--tgt_write_into', default=None, required=False, type=str,
                    help="path to save target sentences.")


### Context Cleaning ###
nltk.download('stopwords')
nltk.download('wordnet')
STOPWORDS = stopwords.words('english')

nlp = spacy.load("en_core_web_sm")
lmtzr = WordNetLemmatizer()


def context_clean(text, rm_punct=True):
    text = re.sub(r'[ ]?\(.*?\)[ ]?', ' ', text)
    text = re.sub(r'[ ]?\[.*?\][ ]?', ' ', text)
    text = re.sub(r'[ ]?<.*?>[ ]?', ' ', text)
    if rm_punct:
        for punct in PUNCTUATIONS:
            text = text.replace(punct, '')
    return text


def get_sents(text: str):
    ### Split passage into sentences ###
    # Remove redundant blanks #
    text = re.sub('[\n\t\r]+', ' ', text.strip())
    text = re.sub('[ ]+', ' ', text)

    sent_tokenizer = PunktSentenceTokenizer()
    sents = sent_tokenizer.tokenize(text)
    return [x for x in sents if len(x) > 3]


def get_tokens(paragraph_sents: list):
    # before lemmatization #
    sent_tokens = []
    clean_sents = []
    for sent in paragraph_sents:
        clean_sent = context_clean(sent.lower())
        clean_sents.append(clean_sent)

        tokens = [token.text for token in nlp(clean_sent)]
        sent_tokens.append(tokens)
    return clean_sents, sent_tokens


def get_terms(paragraph_sents: list, lemmatizer=lmtzr):
    _, sent_tokens = get_tokens(paragraph_sents)

    sent_terms = []
    for tokens in sent_tokens:
        terms = [lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS]
        sent_terms.append(terms)
    return sent_terms


def get_sents_split(psgs_list):
    renew_list = []
    for psgs in psgs_list:
        renew_entry = []
        for psg in psgs:
            renew_entry += get_sents(psg)
        renew_list.append(renew_entry)
    return renew_list


def make_presumm_input(psgs_list, input_write_into: str, trg_write_into=None, trg_summ_list=None, passage_split_sents=False):
    if not passage_split_sents:
        psgs_list = [get_sents(p) for p in psgs_list]
        # print(psgs_list)
    if trg_summ_list is not None:
        with open(input_write_into, 'w') as input_trg:
            with open(trg_write_into, 'w') as trg_trg:
                for psg_sents, summ_sents in tqdm(zip(psgs_list, trg_summ_list)):
                    input_trg.write(' [CLS] [SEP] '.join(psg_sents)+'\n')
                    trg_trg.write(summ_sents.replace('\n', '')+'\n')
        print(f"Input write into => {input_write_into}")
        print(f"Target write into => {trg_write_into}")
    else:
        with open(input_write_into, 'w') as input_trg:
            for psg_sents in tqdm(psgs_list):
                input_trg.write(' [CLS] [SEP] '.join(psg_sents)+'\n')
        print(f"Input write into => {input_write_into}")


def read_context(input_path):
    input_format = input_path.split(".")[-1]
    if input_format == "csv":
        input_DF = pd.read_csv(input_path)[["SectionTitle", "Context"]]
        return input_DF['Context'].values
    elif input_format == "pkl":
        input_dict = pickle.load(input_path)
        return input_dict["Context"]
    elif input_format == "txt":
        with open(input_path, 'r')as trg:
            input_list = trg.readlines()
        return input_list
    return None


if __name__ == '__main__':
    args = parser.parse_args()
    cxt_list = read_context(args.input_path)
    if args.tgt_path is not None:
        trg_list = read_context(args.tgt_path)
    else:
        trg_list = None

    make_presumm_input(psgs_list=cxt_list,
                       input_write_into=args.src_write_into,
                       trg_summ_list=trg_list,
                       trg_write_into=args.tgt_write_into
                       )
