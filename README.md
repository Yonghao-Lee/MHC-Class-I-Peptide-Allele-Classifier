# MHC Class I Peptide–Allele Classifier

A PyTorch MLP that predicts which of six common HLA class I alleles will recognise a given 9-mer peptide. Trained on labelled positive/negative peptide sequences, then applied to the SARS-CoV-2 Spike protein to identify candidate epitopes for vaccine design.

---

## Background

The adaptive immune system uses **HLA class I (MHC-I) molecules** to display short fragments of intracellular proteins — typically 9 amino acids long — on the cell surface. When a fragment is recognised as foreign, T-cells are triggered to destroy the cell, which is the basis of antiviral immunity and a central concept in epitope-based vaccine design.

Predicting which peptides a given HLA allele will present is therefore a high-value computational problem in immunology and vaccine development. This project builds a small neural classifier for that task and demonstrates its use on the SARS-CoV-2 Spike protein.

## Task

Given a 9-mer amino-acid peptide as input, predict one of seven classes:

| Index | Class       | Meaning                                         |
|-------|-------------|-------------------------------------------------|
| 0     | `A0101`     | Recognised by HLA-A\*01:01                      |
| 1     | `A0201`     | Recognised by HLA-A\*02:01                      |
| 2     | `A0203`     | Recognised by HLA-A\*02:03                      |
| 3     | `A0207`     | Recognised by HLA-A\*02:07                      |
| 4     | `A0301`     | Recognised by HLA-A\*03:01                      |
| 5     | `A2402`     | Recognised by HLA-A\*24:02                      |
| 6     | `Negative`  | Not recognised by any of the six alleles above  |

This multiclass framing (rather than binary detect/not-detect) lets the model answer *which* allele recognises a peptide — needed downstream for SARS-CoV-2 epitope prediction.

## Data

- **~12,891 positive** peptide sequences (across the six alleles).
- **~24,492 negative** peptide sequences.
- Stratified 90/10 train/test split.
- Inverse-frequency class weighting applied in the loss to handle the ~2:1 negative-to-positive imbalance.

> **Note.** The training data is not redistributed in this repository because it was provided as part of a university coursework dataset. See `data/README.md` for instructions on supplying your own equivalent data.

### Input representation

Each amino acid is encoded as a 20-dimensional one-hot vector over the standard alphabet `ACDEFGHIKLMNPQRSTVWY`. The full 9-mer is flattened to a single tensor of size **180** (= 9 positions × 20 amino acids).

One-hot encoding is preferred over integer indexing because integer encoding would impose a spurious ordinal relationship between chemically unrelated residues.

## Models

Three MLP variants were compared, all in PyTorch.

### (b) Baseline — Hidden size 180

```
Linear(180→180) → ReLU → Linear(180→180) → ReLU → Linear(180→7)
```

~66,000 parameters — roughly 2× the training set size. Designed to be intentionally over-parameterised, demonstrating the textbook overfitting failure mode.

### (c) Improved — Bottleneck + regularisation

```
Linear(180→128) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(128→64) → BatchNorm → ReLU → Dropout(0.3)
              → Linear(64→7)
```

~20,000 parameters. Adds batch normalisation, 30% dropout, and a `ReduceLROnPlateau` scheduler.

### (d) Linear-only — Ablation

Same shape as (c) but with all activations, BatchNorm, and Dropout removed. Equivalent to a single affine transformation `Wx + b`. Used as a controlled test of whether the task is linearly separable.

All three trained for 100 epochs with Adam (`lr=1e-3`), batch size 128, class-weighted cross-entropy.

## Results

### Baseline (Part 2b) — severe overfitting

Train loss converges to ~0.24 while test loss diverges to ~3.5. Test accuracy is barely above the majority-class baseline:

| Class       | Precision | Recall | F1   | Support |
|-------------|----------:|-------:|-----:|--------:|
| A0101       | 0.41      | 0.52   | 0.46 | 129     |
| A0201       | 0.33      | 0.39   | 0.35 | 256     |
| A0203       | 0.32      | 0.38   | 0.35 | 196     |
| A0207       | 0.36      | 0.48   | 0.41 | 343     |
| A0301       | 0.24      | 0.30   | 0.26 | 165     |
| A2402       | 0.30      | 0.48   | 0.37 | 200     |
| Negative    | 0.82      | 0.68   | 0.74 | 2450    |
| **Accuracy**     |           |        | **59.53%** | 3739 |
| Macro avg   | 0.40      | 0.46   | 0.42 | 3739    |
| Weighted avg | 0.65     | 0.60   | 0.61 | 3739    |

### Improved architecture (Part 2c) — overfitting resolved

Train and test losses now track each other. Per-allele recall improves dramatically while overall accuracy stays comparable — the right behaviour for an imbalanced multiclass problem.

| Class       | Precision | Recall | F1   | Support |
|-------------|----------:|-------:|-----:|--------:|
| A0101       | 0.47      | 0.87   | 0.61 | 129     |
| A0201       | 0.38      | 0.55   | 0.45 | 256     |
| A0203       | 0.35      | 0.67   | 0.46 | 196     |
| A0207       | 0.45      | 0.72   | 0.55 | 343     |
| A0301       | 0.28      | 0.65   | 0.39 | 165     |
| A2402       | 0.40      | 0.91   | 0.55 | 200     |
| Negative    | 0.96      | 0.53   | 0.68 | 2450    |
| **Accuracy**     |           |        | **59.45%** | 3739 |
| Macro avg   | 0.47      | 0.70   | 0.53 | 3739    |
| Weighted avg | 0.76     | 0.59   | 0.62 | 3739    |

### Baseline vs. Improved — summary

| Metric           | (b) Baseline | (c) Improved | Δ        |
|------------------|-------------:|-------------:|---------:|
| Accuracy         | 59.53%       | 59.45%       | −0.08%   |
| Macro recall     | 0.46         | 0.70         | **+0.24** |
| Macro F1         | 0.42         | 0.53         | **+0.11** |
| Weighted F1      | 0.61         | 0.62         | +0.01    |
| Final test loss  | 3.52         | ~0.8         | ↓        |
| Overfitting      | Severe       | Minimal      | resolved |

### Linear ablation (Part 2d) — depth without nonlinearity

| Metric        | (c) Nonlinear | (d) Linear |
|---------------|--------------:|-----------:|
| Accuracy      | 59.45%        | 57.74%     |
| Macro F1      | 0.53          | 0.52       |
| Macro recall  | 0.70          | 0.71       |
| Weighted F1   | 0.62          | 0.60       |

The linear model performs surprisingly well — the task is partially linearly separable in 180-dim one-hot space — but the nonlinear architecture wins consistently across aggregate metrics, confirming the practical benefit of nonlinearity even when the gap is modest.

## Application: SARS-CoV-2 Spike protein epitopes (Part 2e)

The improved model was applied to the **SARS-CoV-2 Spike protein** ([UniProt P0DTC2](https://www.uniprot.org/uniprotkb/P0DTC2/entry), 1273 amino acids) by sliding a 9-residue window across the full sequence (1265 candidate peptides) and recording the highest-scoring HLA allele for each.

### Top 3 candidate epitopes overall

| Rank | Peptide       | Position  | Allele | Score   |
|-----:|---------------|-----------|--------|--------:|
| 1    | `STEIYQAGS`   | 468–476   | A0101  | 0.9992  |
| 2    | `YNENGTITD`   | 278–286   | A0101  | 0.9987  |
| 3    | `ESEFRVYSS`   | 153–161   | A0101  | 0.9983  |

### Top peptide per HLA allele

| Allele | Peptide       | Position  | Score   |
|--------|---------------|-----------|--------:|
| A0101  | `STEIYQAGS`   | 468–476   | 0.9992  |
| A0201  | `SLLIVNNAT`   | 115–123   | 0.9688  |
| A0203  | `AQKFNGLTV`   | 851–859   | 0.9796  |
| A0207  | `ILPDPSKPS`   | 804–812   | 0.9831  |
| A0301  | `EIYQAGSTP`   | 470–478   | 0.9949  |
| A2402  | `NYKLPDDFT`   | 421–429   | 0.9975  |

These regions of the Spike protein are predicted to be among the most likely to be presented by HLA class I and recognised by CD8+ T-cells, making them candidate epitopes for vaccine design.

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Then place the data files (six `*_pos.txt` files and `negs.txt`) into `./data/` — see `data/README.md`.

## Run

```bash
python peptide_classifier.py --data-dir ./data --epochs 100 --batch-size 128
```

Or open the notebook:

```bash
jupyter notebook peptide_classifier.ipynb
```

The Spike-protein analysis automatically downloads the canonical sequence from UniProt at runtime:

```python
url = "https://rest.uniprot.org/uniprotkb/P0DTC2.fasta"
```

## Project structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── peptide_classifier.py        # or peptide_classifier.ipynb
└── data/
    └── README.md                # how to supply training data
```

## Limitations and notes

- **Course project scope.** This was built as part of an Introduction to Deep Learning assignment. The model is small (~20k params), the dataset is modest, and no held-out validation was used for hyperparameter tuning — these are quick comparisons rather than benchmark-grade results.
- **Imbalanced accuracy is a poor headline metric.** The 59% accuracy is dominated by the negative class. Macro recall (0.70 in the improved model) is the more meaningful number for downstream epitope ranking.
- **No experimental validation.** Predicted epitope candidates are computational suggestions; real vaccine work requires wet-lab confirmation (binding assays, T-cell response).
- **Allele coverage.** Only six HLA-A alleles are modelled; HLA-B and HLA-C coverage and broader population genetics are not addressed.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- SARS-CoV-2 Spike sequence: [UniProt P0DTC2](https://www.uniprot.org/uniprotkb/P0DTC2/entry).
- Built using PyTorch, scikit-learn, pandas, NumPy, and matplotlib.
- This work originated as an assignment for an Introduction to Deep Learning course.
