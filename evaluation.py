### Evaluation ###
### Evaluation Scores ###
# !pip install rouge-score
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

### Set Input Arguement ###
parser = argparse.ArgumentParser()

### Set Training Arguments ###
parser.add_argument('--input_path', required=True, type=str,
                    help="path to input sentences.")
parser.add_argument('--pred_path', required=True, type=str,
                    help="path to model prediction.")
parser.add_argument('--gold_path', required=True, type=str,
                    help="path to ground-truth target.")


def bleu_eval(pred_candidates, references: list, weights=None, verbose=True):
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    if type(pred_candidates) == list:
        scores = []
        for pred in pred_candidates:
            score = sentence_bleu(references, pred, weights=weights)
            if verbose:
                print(score)
            scores.append(score)
        return scores
    elif type(pred_candidates) == str:
        return sentence_bleu(references, pred_candidates, weights=weights)


def rouge_eval(pred: str, reference: str, metrics=None, use_stemmer=True):
    if metrics is None:
        metrics = ['rouge1', 'rouge2', 'rouge3', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=use_stemmer)
    scores_dict = scorer.score(pred, reference)
    return scores_dict


def result_lookup(input_list, gen_list, gold_list):
    score_dict = {
        'BLEU': [],
        'ROUGE-1': [],
        'ROUGE-2': [],
        'ROUGE-3': [],
        'ROUGE-L': [],
    }
    for idx, (x, y, z) in enumerate(zip(input_list, gen_list, gold_list)):
        print(f'{idx}-->')
        print(f'\tInput: {x}')
        print(f'\tGenerated:  {y}')
        print(f'\tReference:  {z}')
        b = bleu_eval(y, [z])
        print(f'\tBLUE: {b}')
        Rouge_Scores = rouge_eval(y, z)
        # print(Rouge_Scores.keys())
        print(f'\tRouge-1: {Rouge_Scores["rouge1"]}')
        print(f'\tRouge-2: {Rouge_Scores["rouge2"]}')
        print(f'\tRouge-3: {Rouge_Scores["rouge3"]}')
        print(f'\tRouge-L: {Rouge_Scores["rougeL"]}')
        # return 0
        score_dict['BLEU'].append(b)
        # print(Rouge_Scores["rouge1"].fmeasure)
        score_dict['ROUGE-1'].append(Rouge_Scores["rouge1"].fmeasure)
        score_dict['ROUGE-2'].append(Rouge_Scores["rouge2"].fmeasure)
        score_dict['ROUGE-3'].append(Rouge_Scores["rouge3"].fmeasure)
        score_dict['ROUGE-L'].append(Rouge_Scores["rougeL"].fmeasure)
    return score_dict


def get_input(file_path: str):
    """Read in txt file."""
    with open(file_path, 'r') as trg:
        X = [a.replace('\n', '') for a in trg.readlines()]
    return X


if __name__ == '__main__':
    args = parser.parse_args()
    input_list = get_input(args.input_path)
    gen_list = get_input(args.pred_path)
    gold_list = get_input(args.gold_path)

    realworld_score_dict = result_lookup(
        input_list=input_list, gen_list=gen_list, gold_list=gold_list)

    sns.distplot(realworld_score_dict['BLEU'])
    sns.distplot(realworld_score_dict['ROUGE-1'])
    sns.distplot(realworld_score_dict['ROUGE-2'])
    sns.distplot(realworld_score_dict['ROUGE-3'])
    sns.distplot(realworld_score_dict['ROUGE-L'])
    plt.legend(['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-3', 'ROUGE-L'])
    plt.title('Evaluation Score')
    plt.savefig("./eval_img.png")
    print('Average BLEU: ', np.mean(realworld_score_dict['BLEU']))
    print('Average ROUGE-1: ', np.mean(realworld_score_dict['ROUGE-1'])*100)
    print('Average ROUGE-2: ', np.mean(realworld_score_dict['ROUGE-2'])*100)
    print('Average ROUGE-3: ', np.mean(realworld_score_dict['ROUGE-3'])*100)
    print('Average ROUGE-L: ', np.mean(realworld_score_dict['ROUGE-L'])*100)
