# Data Preprocessing

## Parsing PDF Document

As for the PDF document preprocessing, we first use ***[scipdf-parser](https://github.com/titipata/scipdf_parser)*** to parse the PDF into sections in plain text format. 

To keep the parser running, make sure the GROBID is running backend by executing the following commands in your command line before processing your custom data:

```bash
pip install git+https://github.com/titipata/scipdf_parser

git clone https://github.com/titipata/scipdf_parser.git

bash /scipdf_parser/serve_grobid.sh
```

You can process your own pdf-document with the code:

```bash
python Data/preprocessing.py --pdf_path <path2pdf_doc> --save_path <path to save processed data> --save_format <save format, default as csv>
```

The `pdf_path` could be the path on your local file directory, or a public accessible link (e.g. `https://arxiv.org/pdf/1908.08345.pdf` )

## Task1. Make input for Neural-based Sentence Selection

We follow the extractive summarization methodology introduced by ([Liu and Lapata, 2019](https://arxiv.org/pdf/1908.08345.pdf)) to select salient sentences from the give paragraph.

```bash
python /Data/task1.py --input_path <path to input passages> --src_write_into <path to save processed input> --tgt_path <path to target summary (not required)> --tgt_write_into   <path to save processed target>
```

#### Potential Pitfall

1. If you see the following error message

   >  oserror: libcublas.so.10: cannot open shared object file: no such file or directory

   Check whether your `torch` and `cuda` version is compatible with your operating system. You can check your CUDA version by `nividia-smi`.

