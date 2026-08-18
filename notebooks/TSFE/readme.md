# TSFE: Two-Stage Feature Enhancement for Remote Sensing Image Captioning

Implementation of the **TSFE** architecture for remote sensing image captioning, based on:

> Guo et al., *Remote Sensing* 2024, 16, 1843

**Notebook:** `tsfe-base-architecture.ipynb`
**Dataset:** [RSICD on Kaggle](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset)
**Environment:** Kaggle (GPU: Tesla T4, internet enabled)

## Architecture

| Module | Role |
|---|---|
| **AMFF** | Adaptive Multi-Scale Feature Fusion — Swin Transformer backbone + SENet channel attention |
| **LFSE** | Local Feature Squeeze and Enhancement — horizontal/vertical multi-head attention + local (soft-attention) refinement |
| **FID**  | Feature Interaction Decoder — LSTM decoder with an MLP that fuses global image features into the text embedding space |

## What the notebook does

1. Installs dependencies and loads the RSICD dataset from Kaggle
2. Builds a vocabulary and aligns it with pretrained GloVe (300d) embeddings
3. **Stage 1 — Fine-tuning:** trains the AMFF encoder alone to align its global feature with GloVe sentence embeddings, then freezes it
4. **Stage 2 — Main training:** trains the full TSFE model (AMFF → LFSE → FID) end-to-end
5. Evaluates with **BLEU-1/2/3/4, CIDEr, ROUGE-L, METEOR** (via `pycocoevalcap`, on decoded word sequences)
6. Generates captions at inference time with **beam search** (beam size 3)
7. Saves checkpoints, vocabulary files, and config — ready for downstream (e.g. FastAPI) deployment
8. Plots training/validation curves and qualitative caption examples
9. Lays out the ablation structure mirroring the paper's Table 4

## Notebook structure

| # | Section |
|---|---|
| 1 | Install dependencies |
| 2 | Imports & configuration |
| 3 | GloVe embeddings |
| 4 | Vocabulary & GloVe embedding matrix |
| 5 | Dataset & DataLoader |
| 6 | Model — AMFF, LFSE, FID |
| 7 | Loss functions (paper §3.4) |
| 8 | Evaluation — real BLEU / METEOR / ROUGE-L / CIDEr |
| 9 | Stage 1 — Fine-tuning task (paper §3.4, Fig. 2) |
| 10 | Stage 2 — Main training (30 epochs, paper §4.3) |
| 11 | Training curves |
| 12 | Test-set evaluation |
| 13 | Ablation study — mirrors paper Table 4 |
| 14 | Qualitative examples |
| 15 | Save all artifacts |
| 16 | Summary of corrections vs. previous notebook |

## Key hyperparameters (paper §4.3)

- Image size: 224×224
- Embedding dim: 300 (GloVe)
- LSTM hidden size: 512
- MHA heads: 8, local soft-attention maps: N=8
- Batch size: 32
- Fine-tuning epochs: 10, main training epochs: 30
- Learning rate: 3e-4 (decoder), 1e-5 (encoder)
- Beam size: 3 (inference)

## Corrections vs. a previous version of this notebook

1. `evaluate()` now decodes embeddings → words and computes real BLEU/METEOR/ROUGE-L/CIDEr via `pycocoevalcap`, instead of a proxy loss on embedding vectors
2. Fine-tuning stage is fully **decoupled** from main training; the encoder is frozen before the main loop starts
3. FID uses a proper **MLP** to align `Vglobal` (image space) with the text embedding space (Eq. 16), instead of a raw concatenation
4. LFSE implements true **horizontal + vertical squeeze** multi-head attention (Eq. 6–7), not full-sequence MHA
5. Local attention applies **N=8 soft-attention maps** to `Vs` (post-MHA) to produce `V'local` (Eq. 10–12)
6. `caption()` uses **beam search** (beam_size=3) instead of greedy decoding
7. Epochs match the paper exactly: 10 fine-tuning + 30 main training, batch_size=32

## Outputs

Running the notebook produces, under `/kaggle/working/checkpoints/`:
- `tsfe_best.pth`, `tsfe_final.pth` — model checkpoints
- `word2idx.json`, `idx2word.json` — vocabulary
- `config.json` — training configuration

And under `/kaggle/working/`:
- `training_curves.png` — loss and validation CIDEr curves
- `caption_examples.png` — qualitative predictions vs. ground truth

## Reference results

Paper (RSICD test set): **BLEU-4 = 54.86**, **CIDEr = 305.70**

The notebook prints the run's own test-set metrics alongside these paper values for comparison.

## Requirements

Installed automatically in the first cell: `timm`, `torchmetrics`, `pycocoevalcap`, `nltk`, `gdown`, `einops`, `pycocotools`, plus standard `torch`/`torchvision`/`pandas`/`matplotlib`. Requires GPU and a GloVe `6B.300d` embeddings file available as a Kaggle dataset input.