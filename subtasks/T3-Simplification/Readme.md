# T3-Simplification

We applied ***AudienCe-CEntric Sentence Simplification (ACCESS)*** and ***Multilingual Unsupervised Sentence Simplification (MUSS)*** models proposed in ([Martin et. al, 2020](https://aclanthology.org/2020.lrec-1.577/)) and ([Martin et. al, 2021](https://arxiv.org/abs/2005.00352)).

---

## ACCESS

The pretrained model are released by Martin et. al in their original codebase. The sentence simplification experiment could be run by the following code:

```bash
python subtasks/T3-Simplification/access/scripts/generate.py < <path to source sentences> > <subtask3_output>

python subtasks/T3-Simplification/task3_gen.py --task3_out_path <subtask3_output> --output_path <output_path>
```

For ACCESS model, make sure to create the compatible environment following the instruction in ACCESS original repo:

```bash
git clone https://github.com/facebookresearch/access.git
cd access
pip install -e .
pip install --force-reinstall easse@git+git://github.com/feralvam/easse.git@580ec953e4742c3ae806cc85d867c16e9f584505
pip install --force-reinstall fairseq@git+https://github.com/louismartin/fairseq.git@controllable-sentence-simplification

pip install torch==1.2
pip install sacrebleu==1.3.7
```

 To prevent mess up your local environment, we highly recommended to create a new conda environment by:

```bash
conda create -n <new env name>
```

## MUSS

The pretrained model are released by Martin et. al in their original codebase. The simplification experiment could be run by the following code:

```bash
python subtasks/T3-Simplification/muss/scripts/simplify.py <path to source sentences> --model-name muss_en_wikilarge_mined
```

#### Potential Pitfall

The MUSS model could only be run with **Python version <3.8**. The detailed issue could be referred to the [original MUSS codebase](https://github.com/facebookresearch/muss/issues/12) released by facebook research.