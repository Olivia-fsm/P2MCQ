import argparse
from collections import OrderedDict, Counter
from pprint import pprint
from tqdm.auto import tqdm
import pickle
import pandas as pd
import numpy as np
import warnings
import scipdf
warnings.filterwarnings('ignore')


### Set Input Arguement ###
parser = argparse.ArgumentParser()

### Set Training Arguments ###
parser.add_argument('--pdf_path', required=True, type=str,
                    help="path to custom pdf document.")
parser.add_argument('--save_path', required=True, type=str,
                    help="path for saving processed data.")
parser.add_argument('--save_format', default="csv", required=False, type=str,
                    help="save format (pkl/csv).")
parser.add_argument('--interactive', default=False, type=bool,
                    help="Whether enable the interactive editing mode.")

### PDF Parsing ###


def parse_and_save_PDF(PDF_link: str, save_path=None):
    """Parse PDF document into a dictionary of sections"""
    paper_parsed_dict = scipdf.parse_pdf_to_dict(PDF_link)
    if save_path is not None:
        try:
            with open(save_path, 'wb') as trg:
                pickle.dump(paper_parsed_dict, trg)
        except:
            pass
    return paper_parsed_dict


def get_text(paper_parsed_dict: dict):
    content_dict = {
        'SectionTitle': ['abstract'],
        'Context': [paper_parsed_dict['abstract']],
    }
    paper_sec_list = paper_parsed_dict['sections']
    for sec in paper_sec_list:
        sec_name = sec['heading']
        sec_n_fig_ref = sec['n_figure_ref']
        sec_n_pub_ref = sec['n_publication_ref']
        sec_text = sec['text']
        if sec_text.strip() == '' or sec_name.strip() == '':
            continue
        content_dict['SectionTitle'].append(sec_name)
        content_dict['Context'].append(sec_text)
    return content_dict


def delete_keys(content_dict, redundant_key_list=None, interactive_mode=False):
    titles, contents = [], []

    if interactive_mode:
        for title, content in zip(content_dict['SectionTitle'], content_dict['Context']):
            print('Title--> ', title)
            d = input('Delete?(D): ')
            if d.lower() == 'd':
                continue
            titles.append(title)
            contents.append(content)
    else:
        for title, content in zip(content_dict['SectionTitle'], content_dict['Context']):
            if title in redundant_key_list:
                continue
            titles.append(title)
            contents.append(content)
    new_content_dict = {
        'SectionTitle': titles,
        'Context': contents,
    }
    return new_content_dict


def edit(content_DICT):
    """Select the specific section in an interactive mode."""
    keep_ids = []
    for idx, (sec, cont) in enumerate(zip(content_DICT['SectionTitle'], content_DICT['Context'])):
        print(f'{sec}-->')
        pprint({cont})
        print('===========')
        edit_or_not = input('Edit?(Y/N/D)')
        if edit_or_not.lower() == 'd':
            continue
        elif edit_or_not.lower() == 'y':
            print(cont)
            X = input('New Content:')
            content_DICT['Context'][idx] = X
        keep_ids.append(idx)
    content_DICT['SectionTitle'] = [content_DICT['SectionTitle'][x]
                                    for x in keep_ids]
    content_DICT['Context'] = [content_DICT['Context'][x] for x in keep_ids]
    return content_DICT


def save_csv(trg_dict, save_path, save_format='csv'):
    if save_format == "csv":
        trg_DF = pd.DataFrame(trg_dict)[["SectionTitle", "Context"]]
        trg_DF.to_csv(save_path, index_label=False)
    else:
        with open(save_path, 'wb') as trg:
            pickle.dump(trg_dict, trg)
    print("Saved into {} as {}.".format(save_path, save_format))


if __name__ == '__main__':
    args = parser.parse_args()

    paper_parsed_dict = scipdf.parse_pdf_to_dict(args.pdf_path)
    content_DICT = get_text(paper_parsed_dict)
    if args.interactive:
        edit(content_DICT)
    save_csv(content_DICT, args.save_path, args.save_format)
