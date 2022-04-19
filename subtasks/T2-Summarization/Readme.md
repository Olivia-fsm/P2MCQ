# T2-Abstractive Summarization

We implement two version of abstractive summarization models: (1) ***BertSUMEXTABS***: following the methodology and proposed by Liu and Lapata, we take the released model checkpoint pretrained on CNN/DM; (2) ***BARTSUM-HCI***: fine-tuned BART model on arxiv-HCI dataset. 

## [No Domain Adaptation] BertSUMEXTABS

We follow the implementation of Liu and Lapata to generate abstractive summary. The BERTSUMEXTABS model pretrained on CNN/DM is applied, and can be downloaded [here](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ). The input data should be processed as task1.

```bash
python /subtasks/T1-SentenceSelection/PreSumm/src/train.py -task abs -mode test_text -text_src <input file for task1> -test_batch_size 8 -log_file <log file path> -test_from <pretrained model ckpt> -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 500 -alpha 0.95 -min_length 50 -result_path <output path>
```

## [HCI Domain Adaptation] BARTSUM-HCI

#### arxiv-HCI Dataset

We retrieved xxx HCI academic papers from arxiv and aligned each sentence in the abstract with two following paragraphs with maximum lexical similarity given by BM25. The dataset can be downloaded [here](https://drive.google.com/drive/folders/1BA8gwoV4d2k50dDnaurEVrIKj9Ip2C54?usp=sharing). 

#### Pretrained BARTSUM-HCI

We finetune BART model on arxiv-HCI dataset for 6 epochs. The training process is shown as below.

The pretrained checkpoint can be downloaded [here](https://drive.google.com/drive/folders/1CjcI2S0N9jN5zQNWWnVHhr7rcxIDWvcO?usp=sharing).

|  Step | Training Loss | Validation Loss |    Rouge1 |   Rouge2 |    Rougel | Rougelsum |   Gen Len |
| ----: | ------------: | --------------: | --------: | -------: | --------: | --------: | --------: |
|  2500 |      4.258400 |        4.165351 | 19.747900 | 5.102000 | 16.290200 | 16.305200 | 19.778500 |
|  5000 |      4.098000 |        4.107798 | 20.456800 | 4.921400 | 16.658200 | 16.692300 | 19.818400 |
|  7500 |      4.031400 |        4.074930 | 20.233200 | 4.771900 | 16.647100 | 16.648800 | 19.770300 |
| 10000 |      3.916400 |        4.065139 | 20.412700 | 4.887800 | 16.460400 | 16.477400 | 19.740000 |
| 12500 |      3.816600 |        4.043947 | 20.596800 | 5.341900 | 16.919000 | 16.931500 | 19.797800 |
| 15000 |      3.801300 |        4.041983 | 20.794500 | 5.283500 | 16.916300 | 16.952200 | 19.768900 |
| 17500 |      3.783100 |        4.033934 | 20.845400 | 5.570800 | 17.083900 | 17.101700 | 19.810200 |
| 20000 |      3.675500 |        4.042748 | 19.904100 | 4.757600 | 16.196200 | 16.185900 | 19.771700 |
| 22500 |      3.652300 |        4.030946 | 20.760200 | 5.536000 | 16.983300 | 17.000800 | 19.764800 |
| 25000 |      3.644600 |        4.028274 | 20.514100 | 5.295200 | 16.753900 | 16.752400 | 19.756500 |



You can generated paragraph summarization with the domain-adapted BARTSUM-HCI model with following code:

```bash
python subtasks/T2-Summarization/task2_gen.py --model_path <model_path> --datapath <input_path (csv/pkl)> --output_path <output_path> --device cuda
```

 If you want to print the input-summary pairs, you can set `--verbose True`.

