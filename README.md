# Transformer Evaluation for Code Similarity

This repository contains code for evaluating various transformer-based models on code similarity and clone detection tasks. The project implements fine-tuning and evaluation pipelines for multiple transformer architectures including BERT, jTrans, StateFormer, and their variants.

## Overview

This project evaluates transformer models on code similarity tasks using a triplet loss approach. The models are trained to learn embeddings that can identify similar code snippets, which is useful for clone detection, code search, and related tasks.

## Features

- **Multiple Model Support**: Evaluation framework for BERT, jTrans, StateFormer, and various BERT variants (JTP, CWP, DUP, GSM)
- **Triplet Loss Training**: Uses anchor-positive-negative triplets for learning discriminative code embeddings
- **MAP@R Evaluation**: Implements Mean Average Precision at R metric for similarity evaluation
- **Cross-Validation**: 10-fold cross-validation setup for robust evaluation
- **Results Analysis**: Scripts for statistical analysis and visualization of results

## Project Structure

```
.
├── BERT/                 # BERT model implementation
│   ├── config.py
│   ├── dataloader.py
│   ├── model.py
│   └── tokenizer.py
├── jTrans/              # jTrans model implementation
│   ├── config.py
│   ├── dataloader.py
│   ├── model.py
│   └── tokenizer.py
├── GPT/                 # GPT model implementation
│   └── model.py
├── stateformer/         # StateFormer model (fairseq-based)
├── UniASM/              # UniASM model implementation
├── evaluator/           # Evaluation utilities
│   ├── evaluator.py     # MAP@R evaluation
│   └── extract_answers.py
├── results/             # Evaluation results for different models
├── finetune.py          # Main fine-tuning script
├── preprocess.py        # Data preprocessing script
├── model.py             # Model wrapper for different architectures
├── drawfig.py           # Results visualization
└── test.sh              # Test script for running evaluations
```

## Requirements

### Core Dependencies

- Python 3.6+
- PyTorch
- Transformers (HuggingFace)
- NumPy
- SciPy
- Matplotlib
- tqdm


## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd transformer-evaluation
```

2. Install core dependencies:
```bash
pip install torch transformers numpy scipy matplotlib tqdm
```

3. For StateFormer, install fairseq dependencies:
```bash
cd stateformer
pip install --editable .
cd ..
```

## Data Format

The project expects data in JSONL format where each line contains:
```json
{
  "index": "unique_id",
  "label": "label_id",
  "code": [/* BasicBlocks array */]
}
```

The `code` field should contain an array of BasicBlocks extracted from the code.

## Usage

### 1. Data Preprocessing

First, preprocess your data using `preprocess.py`:
```bash
python preprocess.py
```

This script creates 10-fold cross-validation splits in `data/0/` through `data/9/` directories, each containing:
- `train.jsonl`: Training data
- `valid.jsonl`: Validation data
- `test.jsonl`: Test data

### 2. Fine-tuning Models

Fine-tune a model using `finetune.py`:

```bash
python finetune.py \
    --model_name_or_path=jTrans \
    --output_dir=./data/0 \
    --train_data_file=./data/0/train.jsonl \
    --eval_data_file=./data/0/valid.jsonl \
    --test_data_file=./data/0/test.jsonl \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --epoch=2 \
    --learning_rate=2e-5 \
    --seed=123456
```

### 3. Testing Models

Run evaluation on test set:
```bash
python finetune.py \
    --model_name_or_path=jTrans \
    --output_dir=./data/0 \
    --test_data_file=./data/0/test.jsonl \
    --do_test \
    --eval_batch_size=8
```

Or test without fine-tuning (using pretrained model only):
```bash
python finetune.py \
    --model_name_or_path=jTrans \
    --output_dir=./data/0 \
    --test_data_file=./data/0/test.jsonl \
    --do_test_only \
    --eval_batch_size=8
```

### 4. Evaluation

Extract ground truth answers:
```bash
python evaluator/extract_answers.py \
    -c data/0/test.jsonl \
    -o data/0/answers.jsonl
```

Evaluate predictions:
```bash
python evaluator/evaluator.py \
    -a data/0/answers.jsonl \
    -p data/0/predictions.jsonl \
    > results/jTrans/0_score.log
```

### 5. Running Full Evaluation

Use the provided test script for 10-fold cross-validation:
```bash
bash test.sh
```

### 6. Results Analysis

Generate visualizations and statistical comparisons:
```bash
python drawfig.py
```

This generates:
- Box plots comparing different models
- Statistical significance tests (t-tests) comparing models to BERT baseline

## Supported Models

- **BERT**: Standard BERT model
- **jTrans**: Transformer model for code analysis
- **StateFormer**: State-aware transformer model
- **BERT-JTP**: BERT with JTP task from jTrans
- **BERT-CWP**: BERT with CWP taks from PalmTree
- **BERT-DUP**: BERT with DUP taks from PalmTree
- **BERT-GSM**: BERT with GSM taks from Stateformer



## Evaluation Metric

The project uses **MAP@R** (Mean Average Precision at R):
- For each query, R is the number of relevant items (items with the same label)
- MAP@R measures the average precision of retrieving relevant items within the top R results

## Results

Evaluation results are stored in the `results/` directory, organized by model name. Each fold's results are saved as `{fold}_score.log` files.

## Configuration

Key hyperparameters can be adjusted in `finetune.py`:
- `--learning_rate`: Learning rate (default: 2e-5)
- `--epoch`: Number of training epochs (default: 2)
- `--train_batch_size`: Training batch size (default: 8)
- `--eval_batch_size`: Evaluation batch size (default: 8)
- `--weight_decay`: Weight decay for regularization (default: 0.0)
- `--max_grad_norm`: Gradient clipping (default: 1.0)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this code in your research, please cite the relevant papers for the models you use:
- BERT: [Devlin et al., 2019]
- jTrans: [Reference to jTrans paper]
- StateFormer: [Reference to StateFormer paper]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- This project builds upon the HuggingFace Transformers library
- StateFormer implementation is based on fairseq
- Evaluation metrics follow standard information retrieval practices

