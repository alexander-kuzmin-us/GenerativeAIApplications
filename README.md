# Project 5: Generative AI Applications

Fine-tuning a pretrained **GPT-2 small** (Hugging Face Transformers, PyTorch training loop) to continue **Emily Dickinson's poems** from their real first lines, with an evaluation of output quality, repetition and memorization.

## Submission files

| File | What it is |
|---|---|
| `generative_model.ipynb` | Full project, executed top to bottom with outputs |
| `module_summary.pdf` | Generative AI Analysis and Ethics Report (APA 7 template) |
| `requirements.txt` | `pip freeze` of the environment that ran the notebook |
| `data/12242-0.txt`, `data/README.md` | The dataset (Project Gutenberg #12242) and its source, checksum and licence |

## Reproduce

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace generative_model.ipynb --ExecutePreprocessor.timeout=-1
```

The first run downloads the GPT-2 weights (about 550 MB) from the Hugging Face Hub. On an Apple M4 (MPS backend) the full notebook takes about 5 minutes, about 2.5 of them training. It also runs on CUDA or CPU (CPU is much slower).
