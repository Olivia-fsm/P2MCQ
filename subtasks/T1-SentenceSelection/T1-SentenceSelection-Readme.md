# T1-Sentence Selection

The implementation of this model based on the original codebase of **EMNLP 2019 paper [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)** released by Liu and Lapata. The BERTSUMEXT model pretrained on CNN/DM is used to generate extractive summary for a given paragraph, i.e. select a set of important sentences including the salient information of the the whole paragraph.

---

## Data Preprocessing

### Preprocessed Input Data

A preprocessed input dataset for one of papers used in interview is provided [here](https://github.com/Olivia-fsm/P2MCQ/blob/master/subtasks/T1-SentenceSelection/sample_input_task1.txt) as an example. You can feel free to play with that :)

### Run on the Custom Dataset

Process your own dataset into the compatible input format of Task1 with:

```bash
python /Data/task1.py --input_path <path to input passages> --src_write_into <path to save processed input> --tgt_path <path to target summary (not required)> --tgt_write_into   <path to save processed target>
```

## Model & Experiment

We apply the pretrained BERTSUMEXT model released by Liu and Lapata. The pretrained checkpoint could be downloaded [here](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ). Then run sentence selection experiment with:

```bash
python /subtasks/T1-SentenceSelection/PreSumm/src/train.py -task ext -mode test_text -text_src <input file for task1> -test_batch_size 8 -log_file <log file path> -test_from <pretrained model ckpt> -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 500 -alpha 0.95 -min_length 20 -result_path <output path>
```



