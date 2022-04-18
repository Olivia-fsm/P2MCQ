# Towards Process-Oriented, Modular, and Versatile Question Generation that Meets Educational Needs

Codebase and pre-trained models for NAACL-2022 submission ***Towards Process-Oriented, Modular, and Versatile Question Generation that Meets Educational Needs*** by Xu Wang, Simin Fan, Jessica Houghton and Lu Wang.

---

## P2MCQ Dataset

The P2MCQ dataset archives 160 multiple-choice 307 questions with 629 question options in total (197 correct answers and 432 incorrect answers or distractors) from HCI-101 course. The dataset could be downloaded [here](https://drive.google.com/drive/folders/15UlOicIHAlU6akAJE6ngp2y_krKkpe1B?usp=sharing). 

## Data Preprocessing
### Parsing PDF Document

As for the PDF document preprocessing, we first use ***[scipdf-parser](https://github.com/titipata/scipdf_parser)*** to parse the PDF into sections in plain text format. 

To keep the parser running, make sure the GROBID is running backend by executing the following commands in your command line before processing your custom data:

```bash
pip install git+https://github.com/titipata/scipdf_parser

git clone https://github.com/titipata/scipdf_parser.git

bash /scipdf_parser/serve_grobid.sh
```

You can process your own pdf-document with the code:

```bash
python /Data/preprocessing.py --pdf_path <path2pdf_doc> --save_path <path to save processed data> --save_format <save format, default as csv>
```

The `pdf_path` could be the path on your local file directory, or a public accessible link (e.g. `https://arxiv.org/pdf/1908.08345.pdf` )

### Task1. Make input for Neural-based Sentence Selection

We follow the extractive summarization methodology introduced by ([Liu and Lapata, 2019](https://arxiv.org/pdf/1908.08345.pdf)) to select salient sentences from the give paragraph.

```bash
python /Data/task1.py --input_path <path to input passages> --src_write_into <path to save processed input> --tgt_path <path to target summary (not required)> --tgt_write_into   <path to save processed target>
```

## Modularized Automatic Models

We propose a list of on-the-shelf and fine-tuned models for the purpose of modularizing the end-to-end MCQ generation process. The subtasks include ***[T1-sentence selection]***; ***[T2-Abstractive Paragraph Summarization]***; ***[T3-Sentence Simplification]***; ***[T4-Paraphrasing]***; ***[T5-Negation Generation]***.   

| task                                               | Instruction                                          | Reference                                                    |
| -------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Sentence Selection (i.e. extractive summarization) | [BertSUMEXT](https://github.com/Olivia-fsm/P2MCQ/tree/master/subtasks/T1-SentenceSelection) | The implementation is based on the [original codebase](https://github.com/nlpyang/PreSumm) released by Liu and Lapata |
| Abstractive Summarization                          | [BertSUMEXTABS](https://github.com/Olivia-fsm/P2MCQ/tree/master/subtasks/T2-Summarization)       [Bart-HCI](https://github.com/Olivia-fsm/P2MCQ/tree/master/subtasks/T2-Summarization)                                                       |                                                              |
| Sentence Simplification                            | [ACCESS](https://github.com/Olivia-fsm/P2MCQ/tree/master/subtasks/T3-Simplification)  [MUSS](https://github.com/Olivia-fsm/P2MCQ/tree/master/subtasks/T3-Simplification)                                                             |  The implementation is based on the original codebase([ACCESS](https://github.com/facebookresearch/access) [MUSS](https://github.com/facebookresearch/muss)) released by Martin et al.                                                            |
| Paraphrasing                                       | [Bart-para-SCI](https://github.com/Olivia-fsm/P2MCQ/tree/master/subtasks/T4-Paraphrasing)                                                             |         Finetuned on [ParaSCI](https://github.com/dqxiu/ParaSCI) by Dong et al.                                                     |
| Negation                                           | [CrossAUG](https://github.com/Olivia-fsm/P2MCQ/tree/master/subtasks/T5-Negation)                                                             |             The implementation is based on the [original codebase](https://github.com/minwhoo/CrossAug) released by Lee et al.                                                 |

---

## Evaluation

The quality of the generated texts is evaluated with BLEU, ROUGE-1, ROUGE-2 and ROUGE-L scores. The references are supposed to be provided.
```
python ./evaluation.py --input_path <input_filepath (txt)> --pred_path <pred_filepath (txt)> --gold_path <gold_filepath (txt)>
```


#### Potential Pitfall

1. If you see the following error message

   >  oserror: libcublas.so.10: cannot open shared object file: no such file or directory

   Check whether your `torch` and `cuda` version is compatible with your operating system. You can check your CUDA version by `nividia-smi`.

