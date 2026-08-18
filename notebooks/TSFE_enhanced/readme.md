# TSFE v3: SwinV2-Base Backbone + Transformer Decoder (L=6)

Enhanced implementation of the **TSFE** architecture for remote sensing image captioning, based on:

> Guo et al., *Remote Sensing* 2024, 16, 1843

**Notebook:** `tsfe-enhanced-version.ipynb`
**Dataset:** [RSICD on Kaggle](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset)
**Environment:** Kaggle (GPU, internet enabled)

This is **v3**, building on a previous baseline notebook (v2) with two architectural enhancements and several memory-efficiency measures for training on limited GPU memory.

## Enhancements over the baseline (v2)

1. **Backbone upgrade — Swin-Base → SwinV2-Base**
   - Model: `swinv2_base_window8_256`, input resolution 256×256
   - Updated feature map channels: F2=256ch, F3=512ch, F4=1024ch
   - AMFF `concat_ch` adjusted to 256+512+1024 = 1792

2. **Decoder replacement — LSTM → Transformer Decoder (L=6)**
   - Queries = learned positional embeddings over `max_seq_len`
   - Cross-attention over `V′_local`; causal self-attention mask for autoregressive decoding
   - Same loss, same beam search, same API as v2 — drop-in replacement

## Memory-efficiency measures

- **AMP** (mixed precision) training via `torch.cuda.amp`
- **Gradient accumulation** — effective batch size 32, using 4 micro-steps of 8
- `torch.cuda.empty_cache()` + `gc.collect()` after every epoch
- Checkpoint saved after **every** epoch (both best and latest), not just at the end

## Architecture

| Module | Role |
|---|---|
| **AMFF** | Adaptive Multi-Scale Feature Fusion — SwinV2-Base backbone + SENet channel attention |
| **LFSE** | Local Feature Squeeze and Enhancement — horizontal/vertical multi-head attention + local (soft-attention) refinement |
| **FID**  | Feature Interaction Decoder — now a **6-layer Transformer decoder** (d_model=512, 8 heads) fusing global image features into the text embedding space |

## Notebook structure

| # | Section |
|---|---|
| 1 | Install dependencies |
| 2 | Imports & configuration |
| 3 | GloVe embeddings |
| 4 | Vocabulary & GloVe embedding matrix |
| 5 | Dataset & DataLoader (+ dataset diagnostic/checks) |
| 6 | Model — AMFF (SwinV2 channels), LFSE, FID (Transformer Decoder L=6) |
| 7 | Loss functions (paper §3.4) |
| 8 | Evaluation — real BLEU / METEOR / ROUGE-L / CIDEr (+ save/cleanup helper) |
| 9 | Stage 1 — Fine-tuning task (paper §3.4, Fig. 2): AMP + gradient accumulation, checkpoint every epoch |
| 10 | Stage 2 — Main training (30 epochs, paper §4.3): only LFSE + FID trained, encoder frozen; AMP + gradient accumulation, GPU cache cleared each epoch |
| 11 | Training curves |
| 12 | Test-set evaluation |
| 13 | Ablation study — mirrors paper Table 4 |
| 14 | Qualitative examples |
| 15 | Save all artifacts |
| 16 | Summary of enhancements & corrections vs. previous notebook |

## v2 → v3 comparison

| Item | v2 (original) | v3 (this notebook) |
|---|---|---|
| Backbone | Swin-Base (224×224, F2=192ch) | **SwinV2-Base** (256×256, F2=256ch) |
| Decoder | Single-layer LSTM (hidden=512) | **Transformer Decoder L=6** (d_model=512, 8 heads) |
| Precision | FP32 | **AMP (FP16/BF16 via GradScaler)** |
| Batch | 32 direct | 8 micro × 4 accum = **effective 32** |
| Memory management | None | **`empty_cache` + `gc` after every epoch** |
| Checkpointing | Best only | **Every epoch + best** |
| CIDEr/BLEU eval | pycocoevalcap ✓ | pycocoevalcap ✓ |
| Fine-tuning decoupled from main training | ✓ | ✓ |
| Encoder frozen after fine-tuning | ✓ | ✓ |
| LFSE horizontal + vertical squeeze attention | ✓ | ✓ |
| Beam search inference | ✓ | ✓ |

## Training regime

Two-stage training, matching the paper (§4.3):

1. **Stage 1 (fine-tuning):** trains the SwinV2 + AMFF encoder alone to align `Vglobal` with GloVe sentence embeddings; encoder is frozen afterward
2. **Stage 2 (main training):** only **LFSE + FID (Transformer decoder)** are trained, 30 epochs, encoder and AMFF remain frozen

## Outputs

Running the notebook produces, under the checkpoints directory:
- Best and latest checkpoints, saved after **every** epoch of both stages
- `word2idx.json`, `idx2word.json` — vocabulary
- `config.json` — training configuration

And in the working directory:
- Training/validation curve plots
- Qualitative caption examples (predictions vs. ground truth)

## Reference results

Paper (RSICD test set): **BLEU-4 = 54.86**, **CIDEr = 305.70**

The notebook prints its own test-set BLEU-1/2/3/4, CIDEr, ROUGE-L, and METEOR alongside these paper values for comparison.

## Requirements

`timm` (for SwinV2), `torchmetrics`, `pycocoevalcap`, `nltk`, `gdown`, `einops`, `pycocotools`, plus standard `torch`/`torchvision`/`pandas`/`matplotlib`. Requires a CUDA GPU (AMP-capable) and a GloVe `6B.300d` embeddings file available as a Kaggle dataset input.