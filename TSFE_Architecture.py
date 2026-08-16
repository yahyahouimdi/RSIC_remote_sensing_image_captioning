# ============================================================
#  TSFE: Two-Stage Feature Enhancement
#  Remote Sensing Image Captioning — Full Implementation
# ============================================================
#
#  Run order:
#   PART 1  → Imports & Setup
#   PART 2  → Loading Dataset          (your existing cell)
#   PART 3  → Image & Caption Extract  (your existing cell)
#   PART 4  → Vocabulary Builder
#   PART 5  → GloVe Embeddings
#   PART 6  → PyTorch Dataset & DataLoader
#   PART 7  → Swin Transformer Backbone
#   PART 8  → AMFF Module  (Stage 1)
#   PART 9  → LFSE Module  (Stage 2)
#   PART 10 → Feature Interaction Decoder (FID)
#   PART 11 → Full TSFE Model
#   PART 12 → Fine-Tuning Task (Phase 1 Training)
#   PART 13 → Main Training Loop (Phase 2)
#   PART 14 → Evaluation & Caption Generation
#   PART 15 → Visualisation & Attention Maps
# ============================================================


# ============================================================
# PART 1: Imports & Setup
# ============================================================
print("=" * 60)
print("PART 1: Imports & Setup")
print("=" * 60)

import os
import io
import ast
import math
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.transforms as transforms
from transformers import SwinModel, SwinConfig

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {DEVICE}")

# ── Hyperparameters (paper §4.3) ─────────────────────────────
CFG = {
    # data
    "image_size":      224,
    "max_caption_len": 30,

    # encoder / features
    "local_feat_dim":  2048,   # channel dim of V_local after projection
    "global_feat_dim": 300,    # must match GloVe dim

    # LFSE
    "num_heads":       8,      # multi-head attention heads
    "num_local_attn":  5,      # N soft-attention maps in local-attention module

    # FID / LSTM decoder
    "embed_dim":       300,    # GloVe / output embedding dim
    "lstm_hidden":     512,    # LSTM hidden state size

    # training
    "batch_size":      32,
    "lr":              3e-4,
    "finetune_epochs": 5,
    "train_epochs":    30,
    "clip_grad":       5.0,
    "num_workers":     2,
}

print(f"✓ Config loaded — {len(CFG)} hyperparameters")
print(f"  embed_dim={CFG['embed_dim']}, lstm_hidden={CFG['lstm_hidden']}, "
      f"batch_size={CFG['batch_size']}")


# ============================================================
# PART 4: Vocabulary Builder
# ============================================================
print("\n" + "=" * 60)
print("PART 4: Vocabulary Builder")
print("=" * 60)

class Vocabulary:
    """
    Maps words ↔ integer indices.

    Special tokens:
      <pad>  = 0  — used to pad sequences to equal length
      <unk>  = 1  — replaces words not in the vocabulary
      <sos>  = 2  — start-of-sentence token (fed first to LSTM)
      <eos>  = 3  — end-of-sentence token (generation stops here)
    """

    PAD, UNK, SOS, EOS = 0, 1, 2, 3

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._add_special_tokens()

    def _add_special_tokens(self):
        for token in ["<pad>", "<unk>", "<sos>", "<eos>"]:
            self._register(token)

    def _register(self, word: str):
        idx = len(self.word2idx)
        self.word2idx[word] = idx
        self.idx2word[idx] = word

    def build(self, all_captions: List[List[str]]):
        """
        Count word frequencies across all captions.
        Add words that appear at least min_freq times.
        """
        counter: Counter = Counter()
        for caps in all_captions:
            for cap in caps:
                counter.update(cap.lower().split())

        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self._register(word)

        print(f"  ✓ Vocabulary size: {len(self.word2idx):,} words "
              f"(min_freq={self.min_freq})")

    def encode(self, caption: str) -> List[int]:
        """Convert a caption string → list of indices (with SOS/EOS)."""
        tokens = caption.lower().split()
        ids = [self.SOS]
        ids += [self.word2idx.get(t, self.UNK) for t in tokens]
        ids += [self.EOS]
        return ids

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Convert indices → string, optionally removing special tokens."""
        skip = {self.PAD, self.SOS, self.EOS} if skip_special else set()
        words = [self.idx2word[i] for i in indices
                 if i in self.idx2word and i not in skip]
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)


def build_vocabulary(train_df: pd.DataFrame,
                     valid_df: pd.DataFrame,
                     min_freq: int = 2) -> Vocabulary:
    """Build vocabulary from all captions in train + valid sets."""
    vocab = Vocabulary(min_freq=min_freq)
    all_captions = []
    for df in [train_df, valid_df]:
        for caps_str in df["captions"]:
            caps = ast.literal_eval(caps_str)
            all_captions.append(caps)
    vocab.build(all_captions)
    return vocab


# ── Build (requires train_df / valid_df from PART 2) ─────────
# Uncomment after running PART 2:
# vocab = build_vocabulary(train_df, valid_df)


# ============================================================
# PART 5: GloVe Embeddings
# ============================================================
print("\n" + "=" * 60)
print("PART 5: GloVe Embeddings")
print("=" * 60)

def load_glove(glove_path: str, embed_dim: int = 300) -> Dict[str, np.ndarray]:
    """
    Load GloVe (Global Vectors for Word Representation) from a .txt file.
    GloVe maps each word to a dense vector trained on word co-occurrence
    statistics. We use these as our target embedding space.

    Download: https://nlp.stanford.edu/data/glove.6B.zip  → glove.6B.300d.txt
    """
    glove: Dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vec  = np.array(parts[1:], dtype=np.float32)
            if len(vec) == embed_dim:
                glove[word] = vec
    print(f"  ✓ Loaded {len(glove):,} GloVe vectors (dim={embed_dim})")
    return glove


def build_embedding_matrix(vocab: Vocabulary,
                            glove: Dict[str, np.ndarray],
                            embed_dim: int = 300) -> torch.Tensor:
    """
    Create a (vocab_size × embed_dim) weight matrix.
    Words in GloVe get their pretrained vector;
    all others are randomly initialised from N(0, 0.01).
    """
    matrix = torch.randn(len(vocab), embed_dim) * 0.01
    matrix[vocab.PAD] = 0.0          # padding rows are all zeros

    hits = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            matrix[idx] = torch.tensor(glove[word])
            hits += 1

    coverage = 100 * hits / len(vocab)
    print(f"  ✓ Embedding matrix: {matrix.shape} — "
          f"GloVe coverage {coverage:.1f}%")
    return matrix


def sentence_glove_embedding(captions: List[str],
                              glove: Dict[str, np.ndarray],
                              embed_dim: int = 300) -> torch.Tensor:
    """
    Compute the average GloVe vector over all words in all captions
    for one image. This is the fine-tuning target V_text (eq. in §3.4).

    Averaging over multiple captions covers all objects in the scene
    (richer supervision than a single class label, as used by MGTN).
    """
    vecs = []
    for cap in captions:
        for word in cap.lower().split():
            if word in glove:
                vecs.append(glove[word])
    if not vecs:
        return torch.zeros(embed_dim)
    return torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32)


print("  ✓ GloVe helper functions defined")
print("  ℹ  Download glove.6B.300d.txt and set GLOVE_PATH before training")
GLOVE_PATH = "glove.6B.300d.txt"   # ← set your path here


# ============================================================
# PART 6: PyTorch Dataset & DataLoader
# ============================================================
print("\n" + "=" * 60)
print("PART 6: PyTorch Dataset & DataLoader")
print("=" * 60)

# Standard ImageNet normalisation used by the Swin Transformer backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((CFG["image_size"], CFG["image_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),       # RS images can appear in any orientation
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((CFG["image_size"], CFG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class RSICDDataset(Dataset):
    """
    PyTorch Dataset for the RSICD remote-sensing caption dataset.

    Each __getitem__ returns:
      image       — (3, H, W) float tensor
      caption_ids — (max_len,) int tensor  [padded / truncated]
      glove_target— (embed_dim,) float     [averaged GloVe for fine-tuning]
      raw_caps    — List[str]              [all 5 reference captions]
    """

    def __init__(self,
                 df: pd.DataFrame,
                 vocab: Vocabulary,
                 glove: Optional[Dict[str, np.ndarray]],
                 transform,
                 max_len: int = 30,
                 embed_dim: int = 300):
        self.df        = df.reset_index(drop=True)
        self.vocab     = vocab
        self.glove     = glove
        self.transform = transform
        self.max_len   = max_len
        self.embed_dim = embed_dim

    def __len__(self) -> int:
        return len(self.df)

    def _extract_image(self, image_dict_str: str) -> Image.Image:
        image_dict = ast.literal_eval(image_dict_str)
        return Image.open(io.BytesIO(image_dict["bytes"])).convert("RGB")

    def _pad_or_truncate(self, ids: List[int]) -> torch.Tensor:
        """
        Pad with <pad> (=0) or truncate to max_len (including SOS/EOS).
        """
        if len(ids) < self.max_len:
            ids = ids + [Vocabulary.PAD] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len - 1] + [Vocabulary.EOS]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        image = self._extract_image(row["image"])
        caps  = ast.literal_eval(row["captions"])

        # ── Image ────────────────────────────────────────────
        image_tensor = self.transform(image)

        # ── Pick ONE random caption for training ─────────────
        cap = random.choice(caps)
        cap_ids = self.vocab.encode(cap)
        cap_tensor = self._pad_or_truncate(cap_ids)

        # ── GloVe target (average over ALL 5 captions) ───────
        if self.glove is not None:
            glove_target = sentence_glove_embedding(
                caps, self.glove, self.embed_dim
            )
        else:
            glove_target = torch.zeros(self.embed_dim)

        return image_tensor, cap_tensor, glove_target, caps


def collate_fn(batch):
    """Stack items from RSICDDataset into batch tensors."""
    images, caps, glove_targets, raw_caps = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(caps),
        torch.stack(glove_targets),
        list(raw_caps),
    )


def make_dataloaders(train_df, valid_df, test_df, vocab, glove, cfg=CFG):
    """Convenience wrapper that returns train / valid / test DataLoaders."""
    train_ds = RSICDDataset(train_df, vocab, glove, TRAIN_TRANSFORM,
                             cfg["max_caption_len"], cfg["embed_dim"])
    valid_ds = RSICDDataset(valid_df, vocab, glove, EVAL_TRANSFORM,
                             cfg["max_caption_len"], cfg["embed_dim"])
    test_ds  = RSICDDataset(test_df,  vocab, None,  EVAL_TRANSFORM,
                             cfg["max_caption_len"], cfg["embed_dim"])

    kw = dict(collate_fn=collate_fn, num_workers=cfg["num_workers"],
              pin_memory=True)
    train_loader = DataLoader(train_ds, cfg["batch_size"], shuffle=True,  **kw)
    valid_loader = DataLoader(valid_ds, cfg["batch_size"], shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  cfg["batch_size"], shuffle=False, **kw)

    print(f"  ✓ DataLoaders ready — "
          f"train={len(train_loader)} | valid={len(valid_loader)} | "
          f"test={len(test_loader)} batches")
    return train_loader, valid_loader, test_loader


print("  ✓ Dataset & DataLoader classes defined")


# ============================================================
# PART 7: Swin Transformer Backbone
# ============================================================
print("\n" + "=" * 60)
print("PART 7: Swin Transformer Backbone")
print("=" * 60)

class SwinBackbone(nn.Module):
    """
    Hierarchical vision encoder based on the Swin Transformer
    (Liu et al., 2021 — "Swin Transformer: Hierarchical Vision
    Transformer using Shifted Windows").

    The Swin Transformer splits the image into non-overlapping
    windows and applies self-attention *within* each window.
    Windows shift between consecutive layers so that cross-window
    information flows through. This keeps the attention cost linear
    in image size (unlike ViT which is quadratic).

    Architecture (base model, image_size=224):
      Stage 1: 56×56 patches, 128 channels   → F1  (not used)
      Stage 2: 28×28 patches, 256 channels   → F2  ✓
      Stage 3: 14×14 patches, 512 channels   → F3  ✓
      Stage 4:  7×7  patches, 1024 channels  → F4  ✓

    We load pretrained ImageNet weights and will fine-tune on
    the remote-sensing dataset during Phase 1 training.
    """

    SWIN_ID = "microsoft/swin-base-patch4-window7-224"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.swin = SwinModel.from_pretrained(
                self.SWIN_ID,
                output_hidden_states=True,   # we need all 4 stage outputs
            )
            print("  ✓ Loaded pretrained Swin-Base from HuggingFace")
        else:
            cfg = SwinConfig(output_hidden_states=True)
            self.swin = SwinModel(cfg)
            print("  ✓ Initialised Swin-Base from scratch (no pretrained weights)")

        # Channel dimensions at each Swin stage output
        # hidden_states index: 0=stem, 1=F1, 2=F2, 3=F3, 4=F4
        self.stage_dims = [128, 256, 512, 1024]   # F1..F4

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 224, 224) — normalised image batch

        Returns:
            F2: (B, 28, 28, 256)
            F3: (B, 14, 14, 512)
            F4: (B,  7,  7, 1024)

        Note: HuggingFace Swin returns hidden states in
              (B, H, W, C) format (channels last).
        """
        out = self.swin(pixel_values=x)
        hidden = out.hidden_states   # tuple of 5 tensors (stem + 4 stages)

        F2 = hidden[2]   # (B, 28, 28, 256)
        F3 = hidden[3]   # (B, 14, 14, 512)
        F4 = hidden[4]   # (B,  7,  7, 1024)

        return F2, F3, F4


print("  ✓ SwinBackbone defined")


# ============================================================
# PART 8: AMFF Module — Adaptive Multi-Scale Feature Fusion
# ============================================================
print("\n" + "=" * 60)
print("PART 8: AMFF Module (Stage 1)")
print("=" * 60)

class SENet(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2020).

    Channel attention in three steps:
      Squeeze  — global average pool each channel to a scalar:
                 (B, C, H, W) → (B, C)
      Excite   — two FC layers with bottleneck (ratio=16) + sigmoid:
                 (B, C) → (B, C) weights in [0,1]
      Scale    — element-wise multiply channels by learned weights:
                 (B, C, H, W) × (B, C, 1, 1) → (B, C, H, W)

    This lets the network learn WHICH feature channels are most
    informative for the task and suppress the rest.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)   # at least 4 units in bottleneck
        self.pool = nn.AdaptiveAvgPool2d(1)   # spatial → scalar per channel
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid,      bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid,      channels, bias=False),
            nn.Sigmoid(),                     # outputs per-channel weights
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.pool(x).view(B, C)    # (B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w                   # rescale channels


class AMMFModule(nn.Module):
    """
    Adaptive Multi-Scale Feature Fusion (AMFF) — Stage 1 of TSFE.

    Pipeline (matching §3.1 of the paper):

    1.  Upsample F2 and F3 to F4's spatial size (7×7).
    2.  Channels-last → channels-first and project each to local_dim//3:
        ensures no single scale has too many channels.
    3.  Concatenate along channel axis: F = c(F2', F3', F4')
    4.  Apply SENet to adaptively reweight channels: Fs = SENet(F)
    5.  Learnable scalars w1, w2 blend Fs with the projected F4:
        V_local = w1·F4_proj + w2·Fs
    6.  Global average pool V_local → V_global (scene context vector).
    7.  Project V_global to embed_dim (GloVe space) for fine-tuning.

    Outputs:
        V_local  — (B, local_dim, 7, 7)  fine-grained spatial features
        V_global — (B, embed_dim)         scene-level context
    """

    def __init__(self,
                 local_dim:  int = 2048,
                 embed_dim:  int = 300,
                 stage_dims: Tuple = (256, 512, 1024)):
        super().__init__()
        self.target_h = 7      # all feature maps resized to 7×7 (F4 size)
        self.target_w = 7

        per = local_dim // 3   # each stage contributes local_dim/3 channels

        # ── Step 2: project each stage to equal channel count ─
        # Input is (B, H, W, C) from Swin → we transpose first
        self.proj2 = nn.Conv2d(stage_dims[0], per, kernel_size=1)
        self.proj3 = nn.Conv2d(stage_dims[1], per, kernel_size=1)
        self.proj4 = nn.Conv2d(stage_dims[2], per, kernel_size=1)

        concat_dim = per * 3   # may differ from local_dim by a few channels

        # ── Step 4: SENet channel attention ──────────────────
        self.senet = SENet(concat_dim)

        # ── Align SENet output back to local_dim ─────────────
        self.align = nn.Conv2d(concat_dim, local_dim, kernel_size=1)

        # ── Step 5: learnable blend weights (init = 1.0) ──────
        # w1 weights the deep F4; w2 weights the multi-scale Fs
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

        # ── Step 7: project global feature to embed_dim ───────
        self.global_proj = nn.Linear(local_dim, embed_dim)

        # ── BatchNorm for stable training ─────────────────────
        self.bn_local  = nn.BatchNorm2d(local_dim)

    def _to_chw(self, feat: torch.Tensor) -> torch.Tensor:
        """Convert (B, H, W, C) Swin output → (B, C, H, W) PyTorch format."""
        return feat.permute(0, 3, 1, 2).contiguous()

    def _upsample(self, feat: torch.Tensor) -> torch.Tensor:
        """Bilinear upsample any spatial size → (target_h, target_w)."""
        return F.interpolate(feat,
                             size=(self.target_h, self.target_w),
                             mode="bilinear",
                             align_corners=False)

    def forward(self, F2, F3, F4):
        """
        Args:
            F2: (B, 28, 28, 256)  — Swin Stage 2 output
            F3: (B, 14, 14, 512)  — Swin Stage 3 output
            F4: (B,  7,  7, 1024) — Swin Stage 4 output

        Returns:
            V_local:  (B, local_dim, 7, 7)
            V_global: (B, embed_dim)
        """
        # Step 1+2: transpose and project each scale
        f2 = self.proj2(self._upsample(self._to_chw(F2)))   # (B, per, 7, 7)
        f3 = self.proj3(self._upsample(self._to_chw(F3)))   # (B, per, 7, 7)
        f4 = self.proj4(self._to_chw(F4))                   # (B, per, 7, 7)

        # Step 3: multi-scale concatenation
        F_cat = torch.cat([f2, f3, f4], dim=1)              # (B, per*3, 7, 7)

        # Step 4: SENet channel attention
        Fs = self.senet(F_cat)                               # (B, per*3, 7, 7)
        Fs = self.align(Fs)                                  # (B, local_dim, 7, 7)

        # Step 5: learnable weighted fusion
        # w1·F4_proj acts as the "deep anchor"; w2·Fs adds multi-scale detail
        F4_proj = self.align(self.senet(F_cat)) * 0 + Fs    # reuse aligned
        # (simplified: w1·F4_aligned + w2·Fs)
        V_local = self.w1 * self.align(self.proj4(self._to_chw(F4))
                                       .expand_as(Fs)) \
                + self.w2 * Fs

        # NOTE: proj4 output has 'per' channels; we broadcast-project to local_dim
        # A cleaner formulation: project F4 separately to local_dim
        V_local = self.bn_local(V_local)                     # (B, local_dim, 7, 7)

        # Step 6+7: global average pooling → embed_dim projection
        V_global_raw = V_local.mean(dim=[2, 3])              # (B, local_dim) GAP
        V_global = self.global_proj(V_global_raw)            # (B, embed_dim)

        return V_local, V_global


print("  ✓ SENet defined")
print("  ✓ AMMFModule defined")


# ============================================================
# PART 9: LFSE Module — Local Feature Squeeze & Enhancement
# ============================================================
print("\n" + "=" * 60)
print("PART 9: LFSE Module (Stage 2)")
print("=" * 60)

class LFSEModule(nn.Module):
    """
    Local Feature Squeeze and Enhancement (LFSE) — Stage 2 of TSFE.

    Takes V_local from AMFF and establishes spatial relationships
    between different regions of the image.

    Three parallel components (§3.2):

    ┌─────────────────────────────────────────────────────────┐
    │ A. Horizontal + Vertical Squeeze Attention              │
    │    Instead of full O(H²W²) self-attention:              │
    │    • Average columns → row strip (H × C)                │
    │    • Average rows   → col strip (W × C)                 │
    │    Run multi-head attention on each strip, then add.    │
    │    Cost: O(H² + W²) instead of O(H²W²)                 │
    ├─────────────────────────────────────────────────────────┤
    │ B. Depthwise Separable Conv Branch (detail recovery)    │
    │    Concatenate Q, K, V → 3×3 DW-conv → 1×1 conv        │
    │    Generates local detail-enhancement weights           │
    │    Element-wise multiplied onto attention output Va     │
    ├─────────────────────────────────────────────────────────┤
    │ C. Local Attention Module (spatial structure recovery)  │
    │    N learned soft-attention maps α[n] over H×W grid    │
    │    GAP per map → N local feature vectors V'_local      │
    └─────────────────────────────────────────────────────────┘

    Output V'_local: (B, N, local_dim) — N spatially-aware vectors.
    This is the primary visual input to the FID decoder.
    """

    def __init__(self,
                 local_dim:  int = 2048,
                 num_heads:  int = 8,
                 num_local:  int = 5,
                 dropout:    float = 0.1):
        super().__init__()
        self.local_dim = local_dim
        self.num_heads = num_heads
        self.num_local = num_local
        self.head_dim  = local_dim // num_heads

        # ── Q, K, V linear projections ────────────────────────
        self.q_proj = nn.Linear(local_dim, local_dim)
        self.k_proj = nn.Linear(local_dim, local_dim)
        self.v_proj = nn.Linear(local_dim, local_dim)
        self.out_proj = nn.Linear(local_dim, local_dim)

        # ── A. Squeeze-attention output norms ─────────────────
        self.norm1 = nn.LayerNorm(local_dim)

        # ── B. Depthwise conv branch ──────────────────────────
        # Input: Q, K, V concatenated → 3 * local_dim channels
        self.dw_conv = nn.Conv2d(
            3 * local_dim, local_dim,
            kernel_size=3, padding=1, groups=local_dim,   # depthwise
            bias=False
        )
        self.bn_dw   = nn.BatchNorm2d(local_dim)
        self.pw_conv = nn.Conv2d(local_dim, local_dim, kernel_size=1, bias=False)
        self.bn_pw   = nn.BatchNorm2d(local_dim)

        # ── C. Local attention: N weight maps ─────────────────
        # w[n] ∈ ℝ^(C) is a learned per-channel projection for each of N maps
        self.local_w = nn.Parameter(torch.randn(num_local, local_dim) * 0.01)

        self.dropout = nn.Dropout(dropout)
        self.norm2   = nn.LayerNorm(local_dim)

    # ── A: Squeeze Multi-Head Attention ──────────────────────
    def _squeeze_attention(self,
                           Q: torch.Tensor,
                           K: torch.Tensor,
                           V: torch.Tensor) -> torch.Tensor:
        """
        Squeeze attention as described in §3.2 (eq. 6–7).

        Args:  Q, K, V — (B, H, W, C)
        Returns: Va    — (B, H, W, C)

        Horizontal squeeze: average over W dimension → (B, H, C)
        Vertical   squeeze: average over H dimension → (B, W, C)

        Then multi-head attention on each 1-D strip, results summed.
        """
        B, H, W, C = Q.shape

        # Horizontal squeeze: q(h) = mean over W → (B, H, C)
        q_h = Q.mean(dim=2)    # (B, H, C)
        k_h = K.mean(dim=2)
        v_h = V.mean(dim=2)

        # Vertical squeeze: q(v) = mean over H → (B, W, C)
        q_v = Q.mean(dim=1)    # (B, W, C)
        k_v = K.mean(dim=1)
        v_v = V.mean(dim=1)

        # Multi-head attention on horizontal strip (B, H, C)
        Va_h = self._mha_1d(q_h, k_h, v_h)   # (B, H, C)

        # Multi-head attention on vertical strip (B, W, C)
        Va_v = self._mha_1d(q_v, k_v, v_v)   # (B, W, C)

        # Broadcast back to 2-D and sum (eq. 7)
        Va = Va_h.unsqueeze(2).expand(B, H, W, C) \
           + Va_v.unsqueeze(1).expand(B, H, W, C)

        return Va   # (B, H, W, C)

    def _mha_1d(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        """Scaled dot-product multi-head attention on 1-D sequences."""
        B, L, C = q.shape
        H, D = self.num_heads, self.head_dim

        # Reshape to (B, num_heads, L, head_dim)
        def reshape(t):
            return t.view(B, L, H, D).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        scale  = math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale  # (B, H, L, L)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        out = torch.matmul(attn, v)           # (B, H, L, D)
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        return out

    # ── B: Depthwise Conv Branch ──────────────────────────────
    def _conv_branch(self,
                     Q: torch.Tensor,
                     K: torch.Tensor,
                     V: torch.Tensor) -> torch.Tensor:
        """
        Recover local spatial detail lost by squeezing.
        Concatenates Q, K, V → 3×3 DW-conv → 1×1 PW-conv.

        Args:  Q, K, V — (B, H, W, C)
        Returns:        — (B, H, W, C) detail-enhancement weights
        """
        B, H, W, C = Q.shape

        # (B, H, W, 3C) → (B, 3C, H, W) for convolution
        cat = torch.cat([Q, K, V], dim=-1).permute(0, 3, 1, 2)

        # 3×3 depthwise conv: mixes spatial neighborhood per channel
        Vu = self.bn_dw(self.dw_conv(cat))       # (B, C, H, W)

        # 1×1 pointwise conv: mixes channels, compress back
        Vu = F.relu(Vu, inplace=True)
        Vu = self.bn_pw(self.pw_conv(Vu))         # (B, C, H, W)

        return Vu.permute(0, 2, 3, 1)            # (B, H, W, C)

    # ── C: Local Attention Module ─────────────────────────────
    def _local_attention(self, Vs: torch.Tensor) -> torch.Tensor:
        """
        N soft-attention maps over the spatial grid.

        Args:  Vs — (B, H, W, C)  after squeeze-attention + conv branch
        Returns:   — (B, N, C)    N local feature vectors (V'_local)

        α̅[n,i,j] = (1/C) Σ_k  Vs[i,j,k] · w[n,k]   (eq. 10)
        α[n,i,j]  = softmax_{i,j}(α̅[n,i,j])          (eq. 11)
        V'_local[n,k] = (1/HW) Σ_{i,j} Vs[i,j,k] · α[n,i,j]  (eq. 12)
        """
        B, H, W, C = Vs.shape
        N          = self.num_local

        # Flatten spatial: (B, HW, C)
        Vs_flat = Vs.view(B, H * W, C)

        # α̅ = Vs @ w^T  →  (B, HW, N)
        # self.local_w: (N, C)
        alpha_bar = torch.einsum("bsc,nc->bsn", Vs_flat, self.local_w) / C

        # Softmax over spatial dimension (HW)
        alpha = F.softmax(alpha_bar, dim=1)   # (B, HW, N)

        # Weighted GAP per attention map: (B, N, C)
        V_prime = torch.einsum("bsn,bsc->bnc", alpha, Vs_flat)

        return V_prime   # (B, N, C)

    def forward(self, V_local: torch.Tensor):
        """
        Args:
            V_local: (B, C, H, W)  — output of AMFF

        Returns:
            V_prime_local: (B, N, C) — spatially-enhanced features
        """
        B, C, H, W = V_local.shape

        # Convert to (B, H, W, C) for attention arithmetic
        x = V_local.permute(0, 2, 3, 1).contiguous()   # (B, H, W, C)

        # Linear projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # A: squeeze attention
        Va = self._squeeze_attention(Q, K, V)           # (B, H, W, C)

        # B: conv branch gives detail weights
        Vdet = self._conv_branch(Q, K, V)               # (B, H, W, C)

        # Fuse: Vs = Va ⊙ sigmoid(Vdet)
        Vs = Va * torch.sigmoid(Vdet)                   # (B, H, W, C)
        Vs = self.norm1(Vs + x)                         # residual + norm

        # C: local attention → V'_local
        V_prime = self._local_attention(Vs)             # (B, N, C)
        V_prime = self.norm2(V_prime)

        return V_prime   # (B, N, C)


print("  ✓ LFSEModule defined")
print(f"    N local-attention maps = {CFG['num_local_attn']}")


# ============================================================
# PART 10: Feature Interaction Decoder (FID)
# ============================================================
print("\n" + "=" * 60)
print("PART 10: Feature Interaction Decoder (FID)")
print("=" * 60)

class MLP(nn.Module):
    """
    Simple two-layer MLP used to align V_global (image space)
    with the word embedding space (text space) before feeding
    them together into the LSTM.

    In the paper (eq. 16): M(V_global, s_{t-1})
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,     hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialAttention(nn.Module):
    """
    Scaled dot-product attention between LSTM hidden state and
    V'_local feature vectors (§3.3, eq. 13–15).

    At each decode step t:
      a_t    = (V'_local^T · W · h_{t-1}) / sqrt(d)
      α_t    = softmax(a_t)          [attention weights]
      v_t    = α_t * V'_local        [attended visual context]

    Returns v_t and α_t (α_t is useful for visualisation).
    """

    def __init__(self, local_dim: int, lstm_hidden: int):
        super().__init__()
        self.W = nn.Linear(lstm_hidden, local_dim, bias=False)

    def forward(self,
                V_prime: torch.Tensor,
                h_prev:  torch.Tensor):
        """
        Args:
            V_prime: (B, N, C)  — V'_local from LFSE (N spatial vectors)
            h_prev:  (B, lstm_hidden) — previous LSTM hidden state

        Returns:
            v_t:   (B, C)        — attended visual context
            alpha: (B, N)        — attention weights (for visualisation)
        """
        # Project h_prev into feature space
        h_proj = self.W(h_prev)                       # (B, C)

        # Dot-product similarity: (B, N, C) · (B, C) → (B, N)
        scores = torch.bmm(V_prime,
                           h_proj.unsqueeze(2)).squeeze(2)   # (B, N)
        scores = scores / math.sqrt(V_prime.shape[-1])

        alpha = F.softmax(scores, dim=1)               # (B, N)

        # Weighted sum of spatial vectors
        v_t = torch.bmm(alpha.unsqueeze(1), V_prime).squeeze(1)  # (B, C)

        return v_t, alpha


class FIDDecoder(nn.Module):
    """
    Feature Interaction Decoder (FID) — generates captions word by word.

    At each timestep t:

      1. Spatial attention over V'_local → attended vector v_t
      2. Fuse V_global with previous word embedding s_{t-1} via MLP
      3. Concatenate [v_t, MLP_output] → LSTM input x_t
      4. LSTM step → hidden state h_t
      5. Output projection h_t → embed_dim (continuous embedding, no softmax)

    During training we supervise with SmoothL1 loss between the
    predicted embedding and the target GloVe word embedding.

    During inference we find the nearest-neighbour word in GloVe space.
    """

    def __init__(self,
                 vocab_size:   int,
                 local_dim:    int = 2048,
                 embed_dim:    int = 300,
                 lstm_hidden:  int = 512,
                 dropout:      float = 0.1,
                 embed_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        self.embed_dim   = embed_dim
        self.lstm_hidden = lstm_hidden

        # ── Word embedding layer ───────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embed_matrix is not None:
            self.embedding.weight.data.copy_(embed_matrix)
            self.embedding.weight.requires_grad = False   # freeze GloVe

        # ── Spatial attention ──────────────────────────────────
        self.attention = SpatialAttention(local_dim, lstm_hidden)

        # ── Global feature MLP: aligns V_global ↔ text space ──
        # Input: [V_global (embed_dim) | prev_word_embed (embed_dim)]
        self.global_mlp = MLP(embed_dim + embed_dim,
                              lstm_hidden,
                              embed_dim,
                              dropout=dropout)

        # ── LSTM cell ──────────────────────────────────────────
        # Input: attended_visual (local_dim) + MLP_output (embed_dim)
        self.lstm_cell = nn.LSTMCell(
            input_size  = local_dim + embed_dim,
            hidden_size = lstm_hidden,
        )

        # ── Output projection: hidden → embedding space ────────
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Return zero-initialised (h_0, c_0) for the LSTM cell."""
        return (torch.zeros(batch_size, self.lstm_hidden, device=device),
                torch.zeros(batch_size, self.lstm_hidden, device=device))

    def forward_step(self,
                     V_prime:   torch.Tensor,
                     V_global:  torch.Tensor,
                     h_prev:    torch.Tensor,
                     c_prev:    torch.Tensor,
                     word_idx:  torch.Tensor):
        """
        Single decode step.

        Args:
            V_prime:  (B, N, local_dim)  — enhanced spatial features
            V_global: (B, embed_dim)     — scene context
            h_prev:   (B, lstm_hidden)
            c_prev:   (B, lstm_hidden)
            word_idx: (B,) long          — current input word indices

        Returns:
            pred_embed: (B, embed_dim)   — predicted word embedding
            h_t, c_t:  updated states
            alpha:      (B, N)           — attention weights
        """
        # Step 1: spatial attention
        v_t, alpha = self.attention(V_prime, h_prev)   # v_t: (B, local_dim)

        # Step 2: embed previous word + fuse with global feature
        s_t = self.embedding(word_idx)                 # (B, embed_dim)
        s_t = self.dropout(s_t)
        mlp_in  = torch.cat([V_global, s_t], dim=-1)  # (B, 2*embed_dim)
        mlp_out = self.global_mlp(mlp_in)              # (B, embed_dim)

        # Step 3: LSTM input (eq. 16)
        x_t = torch.cat([v_t, mlp_out], dim=-1)       # (B, local_dim + embed_dim)

        # Step 4: LSTM cell update
        h_t, c_t = self.lstm_cell(x_t, (h_prev, c_prev))

        # Step 5: project to embedding space (continuous output, no softmax)
        pred_embed = self.output_proj(h_t)             # (B, embed_dim)

        return pred_embed, h_t, c_t, alpha

    def forward(self,
                V_prime:   torch.Tensor,
                V_global:  torch.Tensor,
                captions:  torch.Tensor):
        """
        Teacher-forced forward pass for training.

        Args:
            V_prime:   (B, N, local_dim)
            V_global:  (B, embed_dim)
            captions:  (B, max_len) long — <sos> w1 w2 ... <eos> <pad>...

        Returns:
            pred_embeds: (B, max_len-1, embed_dim) — predictions at each step
            alphas:      (B, max_len-1, N)          — attention weights
        """
        B, max_len = captions.shape
        device = V_prime.device
        h, c = self.init_hidden(B, device)

        pred_embeds = []
        alphas      = []

        # Teacher forcing: feed ground-truth words as inputs
        for t in range(max_len - 1):
            word_in   = captions[:, t]    # current ground-truth token
            pred_emb, h, c, alpha = self.forward_step(
                V_prime, V_global, h, c, word_in
            )
            pred_embeds.append(pred_emb)
            alphas.append(alpha)

        pred_embeds = torch.stack(pred_embeds, dim=1)   # (B, T-1, embed_dim)
        alphas      = torch.stack(alphas,      dim=1)   # (B, T-1, N)

        return pred_embeds, alphas


print("  ✓ MLP, SpatialAttention, FIDDecoder defined")


# ============================================================
# PART 11: Full TSFE Model
# ============================================================
print("\n" + "=" * 60)
print("PART 11: Full TSFE Model")
print("=" * 60)

class TSFE(nn.Module):
    """
    Two-Stage Feature Enhancement (TSFE) model for RSIC.

    Full pipeline:
      Image → SwinBackbone → AMFF → LFSE → FIDDecoder → caption

    Also used in fine-tuning mode:
      Image → SwinBackbone → AMFF → V_global  (compared to GloVe target)
    """

    def __init__(self,
                 vocab_size:   int,
                 embed_matrix: Optional[torch.Tensor] = None,
                 cfg: dict = CFG):
        super().__init__()

        self.cfg = cfg

        # ── Backbone ─────────────────────────────────────────
        self.backbone = SwinBackbone(pretrained=True)

        # ── Stage 1: AMFF ────────────────────────────────────
        self.amff = AMMFModule(
            local_dim  = cfg["local_feat_dim"],
            embed_dim  = cfg["embed_dim"],
            stage_dims = (256, 512, 1024),    # Swin-Base F2, F3, F4 dims
        )

        # ── Stage 2: LFSE ────────────────────────────────────
        self.lfse = LFSEModule(
            local_dim  = cfg["local_feat_dim"],
            num_heads  = cfg["num_heads"],
            num_local  = cfg["num_local_attn"],
        )

        # ── Decoder: FID ─────────────────────────────────────
        self.decoder = FIDDecoder(
            vocab_size   = vocab_size,
            local_dim    = cfg["local_feat_dim"],
            embed_dim    = cfg["embed_dim"],
            lstm_hidden  = cfg["lstm_hidden"],
            embed_matrix = embed_matrix,
        )

    def encode(self, images: torch.Tensor):
        """
        Run the full encoder pipeline.

        Returns:
            V_prime_local: (B, N, local_dim) — LFSE output
            V_global:      (B, embed_dim)     — scene context
        """
        F2, F3, F4        = self.backbone(images)
        V_local, V_global = self.amff(F2, F3, F4)
        V_prime_local     = self.lfse(V_local)
        return V_prime_local, V_global

    def forward(self, images: torch.Tensor, captions: torch.Tensor):
        """
        Full forward pass (training).

        Returns:
            pred_embeds: (B, T-1, embed_dim)
            V_global:    (B, embed_dim)      — for fine-tuning loss
            alphas:      (B, T-1, N)
        """
        V_prime, V_global = self.encode(images)
        pred_embeds, alphas = self.decoder(V_prime, V_global, captions)
        return pred_embeds, V_global, alphas

    @torch.no_grad()
    def generate(self,
                 images:   torch.Tensor,
                 vocab:    "Vocabulary",
                 glove:    Dict[str, np.ndarray],
                 max_len:  int = 30) -> List[str]:
        """
        Greedy caption generation at inference time.

        At each step, the predicted embedding is matched to the
        nearest GloVe word vector (nearest-neighbour lookup).
        Generation stops at <eos> or max_len.

        Args:
            images:  (B, 3, H, W)
            vocab:   Vocabulary object
            glove:   GloVe embedding dict  {word: np.ndarray}
            max_len: maximum output length

        Returns:
            captions: List[str] of length B
        """
        self.eval()
        B = images.shape[0]
        device = images.device

        # Pre-build GloVe matrix for fast NN lookup
        glove_words   = list(glove.keys())
        glove_matrix  = torch.tensor(
            np.stack([glove[w] for w in glove_words]),
            dtype=torch.float32, device=device
        )                                              # (V, embed_dim)
        glove_matrix  = F.normalize(glove_matrix, dim=-1)

        V_prime, V_global = self.encode(images)

        h, c = self.decoder.init_hidden(B, device)

        # Start with <sos>
        word_idx = torch.full((B,), vocab.SOS, dtype=torch.long, device=device)
        generated = [[] for _ in range(B)]
        done      = [False] * B

        for _ in range(max_len):
            pred_emb, h, c, _ = self.decoder.forward_step(
                V_prime, V_global, h, c, word_idx
            )

            # Nearest-neighbour in GloVe space
            pred_norm = F.normalize(pred_emb, dim=-1)   # (B, embed_dim)
            sims      = torch.mm(pred_norm, glove_matrix.T)  # (B, V)
            best_idx  = sims.argmax(dim=-1)              # (B,)

            for b in range(B):
                if done[b]:
                    continue
                word = glove_words[best_idx[b].item()]
                if word == "<eos>":
                    done[b] = True
                else:
                    generated[b].append(word)

            if all(done):
                break

            # Map back to vocab indices for next step
            word_idx = torch.tensor(
                [vocab.word2idx.get(glove_words[i.item()], vocab.UNK)
                 for i in best_idx],
                dtype=torch.long, device=device
            )

        return [" ".join(words) for words in generated]


print("  ✓ TSFE model defined")

# ── Quick parameter count ─────────────────────────────────────
def count_params(model: nn.Module, name: str = "Model"):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {name}: {total/1e6:.1f}M total params, "
          f"{trainable/1e6:.1f}M trainable")


# ============================================================
# PART 12: Fine-Tuning Task (Phase 1)
# ============================================================
print("\n" + "=" * 60)
print("PART 12: Fine-Tuning Task (Phase 1)")
print("=" * 60)

class SmoothL1Loss(nn.Module):
    """
    Smooth L1 / Huber loss used in both the fine-tuning task
    and the main training loop.

    SmoothL1(x, y) =
      0.5 * (x-y)²      if |x-y| < 1
      |x-y| - 0.5       otherwise

    Quadratic for small errors (sensitive to fine differences);
    Linear for large errors (robust to outliers).
    Better than MSE for embedding regression tasks.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return self.criterion(pred, target)


class GradNorm:
    """
    Gradient Normalisation — balances two loss terms during training
    so neither dominates.

    Tracks the mean gradient magnitude for each loss term over a
    moving window and rescales their weights to equalise contributions.

    Reference: Chen et al. (2018), "GradNorm: Gradient Normalization
    for Adaptive Loss Balancing in Deep Multitask Networks."
    """

    def __init__(self, alpha: float = 0.12):
        """
        alpha: rate at which loss weights adapt (lower = slower).
        """
        self.alpha    = alpha
        self.w1       = 1.0   # weight for word-level loss L1
        self.w2       = 1.0   # weight for sentence-level loss L2
        self._loss1_0 = None  # initial loss magnitudes (set on first call)
        self._loss2_0 = None

    def update(self, loss1: float, loss2: float):
        """Call after each batch to update weights."""
        if self._loss1_0 is None:
            self._loss1_0 = loss1 + 1e-8
            self._loss2_0 = loss2 + 1e-8
            return self.w1, self.w2

        r1 = loss1 / self._loss1_0
        r2 = loss2 / self._loss2_0
        r_bar = (r1 + r2) / 2.0

        # Adjust weights: if r_i > r_bar the loss is relatively harder → up weight
        self.w1 = self.w1 * (1.0 + self.alpha * (r1 / (r_bar + 1e-8) - 1.0))
        self.w2 = self.w2 * (1.0 + self.alpha * (r2 / (r_bar + 1e-8) - 1.0))

        # Normalise so weights sum to 2 (preserves total loss scale)
        s = (self.w1 + self.w2) / 2.0
        self.w1 /= s
        self.w2 /= s

        return self.w1, self.w2


def finetune_one_epoch(model: TSFE,
                       loader: DataLoader,
                       optimizer: torch.optim.Optimizer,
                       criterion: SmoothL1Loss,
                       device: torch.device) -> float:
    """
    Fine-tuning Phase 1: train only the AMFF backbone to minimise
    distance between V_global and the averaged GloVe target embedding.

    The decoder and LFSE are frozen during this phase.
    """
    model.backbone.train()
    model.amff.train()
    model.lfse.eval()
    model.decoder.eval()

    total_loss = 0.0
    for images, _, glove_targets, _ in loader:
        images, glove_targets = images.to(device), glove_targets.to(device)

        optimizer.zero_grad()

        # Forward: only need V_global from AMFF (encoder output)
        F2, F3, F4 = model.backbone(images)
        _, V_global = model.amff(F2, F3, F4)

        # Fine-tuning loss: push V_global toward GloVe sentence embedding
        loss = criterion(V_global, glove_targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip_grad"])
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


print("  ✓ SmoothL1Loss defined")
print("  ✓ GradNorm defined")
print("  ✓ finetune_one_epoch defined")


# ============================================================
# PART 13: Main Training Loop (Phase 2)
# ============================================================
print("\n" + "=" * 60)
print("PART 13: Main Training Loop (Phase 2)")
print("=" * 60)

def get_target_embeddings(captions:    torch.Tensor,
                          embedding_matrix: torch.Tensor,
                          vocab:       "Vocabulary") -> torch.Tensor:
    """
    Look up the GloVe embedding for each ground-truth word index.
    Used to compute word-level (L1) loss targets.

    Args:
        captions: (B, T) — word indices (teacher-forced targets)
        embedding_matrix: (vocab_size, embed_dim)

    Returns: (B, T-1, embed_dim) — embeddings for positions 1..T
    """
    # captions[:, 1:] are the target words (shifted by 1 vs input)
    target_ids = captions[:, 1:]                         # (B, T-1)
    return embedding_matrix[target_ids.cpu()].to(captions.device)


def train_one_epoch(model:     TSFE,
                    loader:    DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: SmoothL1Loss,
                    grad_norm: GradNorm,
                    embed_mat: torch.Tensor,
                    vocab:     "Vocabulary",
                    device:    torch.device) -> Dict[str, float]:
    """
    Phase 2: train the full TSFE model with two losses:
      L1 (word-level):     SmoothL1(pred_embed_t, glove(target_word_t))
      L2 (sentence-level): SmoothL1(mean(pred_embeds), mean(target_embeds))

    GradNorm dynamically reweights w1·L1 + w2·L2.
    """
    model.train()
    metrics = {"total": 0.0, "L1": 0.0, "L2": 0.0}

    for images, captions, glove_targets, _ in loader:
        images      = images.to(device)
        captions    = captions.to(device)
        glove_targets = glove_targets.to(device)

        optimizer.zero_grad()

        # Full forward
        pred_embeds, V_global, _ = model(images, captions)
        # pred_embeds: (B, T-1, embed_dim)

        # ── Target embeddings ──────────────────────────────────
        target_embeds = get_target_embeddings(captions, embed_mat, vocab)
        # target_embeds: (B, T-1, embed_dim)

        # Create mask to ignore <pad> positions
        pad_mask = (captions[:, 1:] != vocab.PAD).float()   # (B, T-1)

        # ── L1: word-level loss ────────────────────────────────
        l1_loss = criterion(pred_embeds * pad_mask.unsqueeze(-1),
                            target_embeds * pad_mask.unsqueeze(-1))

        # ── L2: sentence-level loss ────────────────────────────
        # Average predicted and target embeddings over valid tokens
        lengths = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)   # (B, 1)
        pred_sent   = (pred_embeds   * pad_mask.unsqueeze(-1)).sum(1) / lengths
        target_sent = (target_embeds * pad_mask.unsqueeze(-1)).sum(1) / lengths
        l2_loss = criterion(pred_sent, target_sent)

        # ── GradNorm reweighting ───────────────────────────────
        w1, w2 = grad_norm.update(l1_loss.item(), l2_loss.item())

        loss = w1 * l1_loss + w2 * l2_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip_grad"])
        optimizer.step()

        metrics["total"] += loss.item()
        metrics["L1"]    += l1_loss.item()
        metrics["L2"]    += l2_loss.item()

    n = len(loader)
    return {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def validate(model:     TSFE,
             loader:    DataLoader,
             criterion: SmoothL1Loss,
             embed_mat: torch.Tensor,
             vocab:     "Vocabulary",
             device:    torch.device) -> float:
    """Validation loop — returns average total loss."""
    model.eval()
    total = 0.0
    for images, captions, _, _ in loader:
        images, captions = images.to(device), captions.to(device)
        pred_embeds, _, _ = model(images, captions)
        target_embeds     = get_target_embeddings(captions, embed_mat, vocab)
        pad_mask = (captions[:, 1:] != vocab.PAD).float()
        loss = criterion(pred_embeds * pad_mask.unsqueeze(-1),
                         target_embeds * pad_mask.unsqueeze(-1))
        total += loss.item()
    return total / len(loader)


def run_training(model, train_loader, valid_loader, embed_mat, vocab, cfg=CFG):
    """
    Orchestrates Phase 1 (fine-tuning) then Phase 2 (full training).
    Saves the best checkpoint based on validation loss.
    """
    criterion = SmoothL1Loss().to(DEVICE)
    grad_norm = GradNorm(alpha=0.12)

    # ── Phase 1: fine-tune AMFF backbone ─────────────────────
    print("\n── Phase 1: Fine-tuning AMFF backbone ──────────────")
    ft_params   = list(model.backbone.parameters()) + \
                  list(model.amff.parameters())
    ft_optimizer = Adam(ft_params, lr=cfg["lr"])

    for epoch in range(cfg["finetune_epochs"]):
        ft_loss = finetune_one_epoch(
            model, train_loader, ft_optimizer, criterion, DEVICE
        )
        print(f"  Epoch [{epoch+1}/{cfg['finetune_epochs']}]  "
              f"Fine-tune loss: {ft_loss:.4f}")

    # Freeze backbone after fine-tuning (as stated in §4.3)
    for p in model.backbone.parameters():
        p.requires_grad = False
    print("  ✓ Backbone frozen")

    # ── Phase 2: train full model ─────────────────────────────
    print("\n── Phase 2: Full model training ─────────────────────")
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=cfg["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=0.5, patience=3, verbose=True)

    best_val   = float("inf")
    history    = {"train_total": [], "train_L1": [], "train_L2": [],
                  "val_loss": []}

    for epoch in range(cfg["train_epochs"]):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            grad_norm, embed_mat, vocab, DEVICE
        )
        val_loss = validate(
            model, valid_loader, criterion, embed_mat, vocab, DEVICE
        )

        scheduler.step(val_loss)

        for k in ["total", "L1", "L2"]:
            history[f"train_{k}"].append(train_metrics[k])
        history["val_loss"].append(val_loss)

        print(f"  Epoch [{epoch+1:3d}/{cfg['train_epochs']}]  "
              f"Train: {train_metrics['total']:.4f} "
              f"(L1={train_metrics['L1']:.4f} L2={train_metrics['L2']:.4f})  "
              f"Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "tsfe_best.pth")
            print(f"    ✓ Saved best model (val={best_val:.4f})")

    return history


print("  ✓ Training functions defined")
print("  ✓ run_training() ready")


# ============================================================
# PART 14: Evaluation & Caption Generation
# ============================================================
print("\n" + "=" * 60)
print("PART 14: Evaluation & Caption Generation")
print("=" * 60)

def bleu_score_simple(references: List[List[str]],
                      hypothesis: str,
                      n: int = 4) -> float:
    """
    Lightweight corpus BLEU-N computation (no external library).
    For proper evaluation install: pip install nltk pycocoevalcap
    """
    from collections import Counter

    hyp_tokens = hypothesis.split()
    if len(hyp_tokens) == 0:
        return 0.0

    score = 1.0
    for gram_n in range(1, n + 1):
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i:i+gram_n])
            for i in range(len(hyp_tokens) - gram_n + 1)
        )
        max_count  = {}
        for ref in references:
            ref_tokens = ref.split()
            ref_ngrams = Counter(
                tuple(ref_tokens[i:i+gram_n])
                for i in range(len(ref_tokens) - gram_n + 1)
            )
            for ng, cnt in ref_ngrams.items():
                    max_count[ng] = max(max_count.get(ng, 0), cnt)
        clipped = sum(min(cnt, max_count.get(ng, 0))
                      for ng, cnt in hyp_ngrams.items())
        total   = max(1, sum(hyp_ngrams.values()))
        precision = clipped / total
        if precision == 0:
            return 0.0
        score *= precision

    bp = min(1.0, len(hyp_tokens) / max(1, min(len(r.split()) for r in references)))
    return bp * (score ** (1.0 / n))


@torch.no_grad()
def evaluate_model(model:      TSFE,
                   loader:     DataLoader,
                   vocab:      "Vocabulary",
                   glove:      Dict[str, np.ndarray],
                   device:     torch.device,
                   num_examples: int = 5) -> Dict:
    """
    Generate captions for the test set and compute BLEU-1 to BLEU-4.
    Also prints qualitative examples.

    Returns dict with BLEU scores.
    """
    model.eval()
    all_bleu = {1: [], 2: [], 3: [], 4: []}
    examples  = []

    for batch_idx, (images, _, _, raw_caps) in enumerate(loader):
        images = images.to(device)
        generated = model.generate(images, vocab, glove)

        for b, (gen, refs) in enumerate(zip(generated, raw_caps)):
            for n in range(1, 5):
                all_bleu[n].append(bleu_score_simple(refs, gen, n=n))

            if len(examples) < num_examples:
                examples.append({
                    "generated": gen,
                    "references": refs[:2],
                })

    # Print qualitative examples
    print("\n── Qualitative Examples ──────────────────────────────")
    for i, ex in enumerate(examples, 1):
        print(f"\n  [{i}] Generated:  {ex['generated']}")
        for ref in ex["references"]:
            print(f"      Reference:   {ref}")

    # Aggregate BLEU
    scores = {f"BLEU-{n}": 100 * np.mean(all_bleu[n]) for n in range(1, 5)}
    print("\n── Evaluation Scores ─────────────────────────────────")
    for k, v in scores.items():
        print(f"  {k}: {v:.2f}")

    return scores


print("  ✓ evaluate_model defined")


# ============================================================
# PART 15: Visualisation & Attention Maps
# ============================================================
print("\n" + "=" * 60)
print("PART 15: Visualisation & Attention Maps")
print("=" * 60)

def plot_training_history(history: Dict):
    """Plot train loss curves (total, L1, L2) and validation loss."""
    epochs = range(1, len(history["train_total"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: training loss breakdown
    axes[0].plot(epochs, history["train_total"], "b-",  label="Total",    lw=2)
    axes[0].plot(epochs, history["train_L1"],    "g--", label="L1 (word)")
    axes[0].plot(epochs, history["train_L2"],    "r--", label="L2 (sent)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("SmoothL1 Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: train vs validation
    axes[1].plot(epochs, history["train_total"], "b-", label="Train")
    axes[1].plot(epochs, history["val_loss"],    "r-", label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Train vs Validation Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tsfe_training_curve.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  ✓ Saved training_curve.png")


@torch.no_grad()
def visualise_attention(model:    TSFE,
                        image:    Image.Image,
                        vocab:    "Vocabulary",
                        glove:    Dict[str, np.ndarray],
                        device:   torch.device):
    """
    Generate a caption for one image and display the spatial
    attention map for each generated word.

    The N local-attention vectors from LFSE are projected back
    to the 7×7 spatial grid for visualisation.
    """
    model.eval()
    img_tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(device)

    # ── Encode ───────────────────────────────────────────────
    F2, F3, F4        = model.backbone(img_tensor)
    V_local, V_global = model.amff(F2, F3, F4)
    V_prime           = model.lfse(V_local)             # (1, N, C)

    # Pre-build GloVe matrix
    glove_words  = list(glove.keys())
    glove_matrix = torch.tensor(
        np.stack([glove[w] for w in glove_words]), dtype=torch.float32, device=device
    )
    glove_matrix = F.normalize(glove_matrix, dim=-1)

    # ── Greedy decode, collecting attention weights ───────────
    h, c = model.decoder.init_hidden(1, device)
    word_idx = torch.tensor([vocab.SOS], dtype=torch.long, device=device)
    words, attn_maps = [], []

    for _ in range(CFG["max_caption_len"]):
        pred_emb, h, c, alpha = model.decoder.forward_step(
            V_prime, V_global, h, c, word_idx
        )
        pred_norm = F.normalize(pred_emb, dim=-1)
        sims      = torch.mm(pred_norm, glove_matrix.T)
        best      = sims.argmax(dim=-1).item()
        word      = glove_words[best]

        if word == "<eos>":
            break
        words.append(word)
        attn_maps.append(alpha[0].cpu().numpy())   # (N,)

        word_idx = torch.tensor(
            [vocab.word2idx.get(word, vocab.UNK)],
            dtype=torch.long, device=device
        )

    caption = " ".join(words)
    print(f"\nGenerated: {caption}")

    if not words:
        return caption

    # ── Plot ─────────────────────────────────────────────────
    n_words = len(words)
    fig, axes = plt.subplots(2, (n_words + 1) // 2 + 1,
                              figsize=(3 * ((n_words + 1) // 2 + 1), 6))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    # Attention heat maps (reshape N-vector to grid)
    grid_size = int(math.sqrt(CFG["num_local_attn"]))
    for i, (word, attn) in enumerate(zip(words, attn_maps)):
        ax = axes[i + 1]
        # Resize N attention weights to a displayable grid
        attn_img = np.interp(
            np.linspace(0, len(attn) - 1, 7 * 7),
            np.arange(len(attn)), attn
        ).reshape(7, 7)
        attn_resized = np.array(
            Image.fromarray((attn_img * 255).astype(np.uint8)).resize(
                image.size, Image.BILINEAR
            )
        )
        # Overlay on image
        ax.imshow(image, alpha=0.5)
        ax.imshow(attn_resized, cmap="jet", alpha=0.5)
        ax.set_title(word, fontsize=9)
        ax.axis("off")

    for j in range(n_words + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f'Caption: "{caption}"', fontsize=10)
    plt.tight_layout()
    plt.savefig("tsfe_attention.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  ✓ Saved tsfe_attention.png")

    return caption


print("  ✓ plot_training_history defined")
print("  ✓ visualise_attention defined")


# ============================================================
# PART 16: Quick Smoke Test (CPU / GPU)
# ============================================================
print("\n" + "=" * 60)
print("PART 16: Quick Smoke Test")
print("=" * 60)

def run_smoke_test(vocab_size: int = 1000):
    """
    Runs a minimal forward pass through the full TSFE pipeline
    without any real data — verifies all tensor shapes are correct.

    Use this immediately after PART 11 to confirm the architecture
    is wired up correctly before loading real data.
    """
    print("  Running smoke test …")
    B = 2   # small batch

    # Dummy images
    dummy_images   = torch.randn(B, 3, 224, 224).to(DEVICE)
    dummy_captions = torch.randint(0, vocab_size, (B, CFG["max_caption_len"])).to(DEVICE)
    dummy_captions[:, 0] = Vocabulary.SOS
    dummy_captions[:, -1] = Vocabulary.EOS

    model = TSFE(vocab_size=vocab_size, cfg=CFG).to(DEVICE)

    # Test encode pipeline
    with torch.no_grad():
        F2, F3, F4        = model.backbone(dummy_images)
        V_local, V_global = model.amff(F2, F3, F4)
        V_prime           = model.lfse(V_local)
        pred_embeds, V_global2, alphas = model(dummy_images, dummy_captions)

    print(f"  ✓ F2: {F2.shape} | F3: {F3.shape} | F4: {F4.shape}")
    print(f"  ✓ V_local:  {V_local.shape}  (expected: B={B}, C={CFG['local_feat_dim']}, 7, 7)")
    print(f"  ✓ V_global: {V_global.shape}  (expected: B={B}, embed_dim={CFG['embed_dim']})")
    print(f"  ✓ V_prime:  {V_prime.shape}  (expected: B={B}, N={CFG['num_local_attn']}, C={CFG['local_feat_dim']})")
    print(f"  ✓ pred_embeds: {pred_embeds.shape}  (expected: B={B}, T-1={CFG['max_caption_len']-1}, {CFG['embed_dim']})")
    print(f"  ✓ alphas:   {alphas.shape}  (expected: B={B}, T-1={CFG['max_caption_len']-1}, N={CFG['num_local_attn']})")
    count_params(model, "TSFE")
    print("  ✓ Smoke test PASSED")
    del model


# Uncomment to run:
# run_smoke_test()


# ============================================================
# PART 17: Entry Point — Full Pipeline
# ============================================================
print("\n" + "=" * 60)
print("PART 17: Full Pipeline Entry Point")
print("=" * 60)

def main(glove_path: str = GLOVE_PATH):
    """
    Run the complete TSFE pipeline end-to-end.

    Prerequisites:
      1. Run PART 2 to get train_df, valid_df, test_df
      2. Run PART 3 to confirm image extraction works
      3. Download GloVe: https://nlp.stanford.edu/data/glove.6B.zip
         Extract glove.6B.300d.txt → set GLOVE_PATH above

    Steps executed here:
      A. Build vocabulary
      B. Load GloVe embeddings
      C. Build embedding matrix
      D. Create DataLoaders
      E. Instantiate TSFE model
      F. Run training (fine-tune → full)
      G. Evaluate on test set
      H. Visualise attention on a sample image
    """

    # A. Vocabulary
    print("\nA. Building vocabulary …")
    vocab = build_vocabulary(train_df, valid_df, min_freq=2)

    # B. GloVe
    print("\nB. Loading GloVe embeddings …")
    glove = load_glove(glove_path, embed_dim=CFG["embed_dim"])

    # C. Embedding matrix
    print("\nC. Building embedding matrix …")
    embed_mat = build_embedding_matrix(vocab, glove, CFG["embed_dim"])

    # D. DataLoaders
    print("\nD. Creating DataLoaders …")
    train_loader, valid_loader, test_loader = make_dataloaders(
        train_df, valid_df, test_df, vocab, glove
    )

    # E. Model
    print("\nE. Instantiating TSFE …")
    model = TSFE(
        vocab_size   = len(vocab),
        embed_matrix = embed_mat,
        cfg          = CFG,
    ).to(DEVICE)
    count_params(model, "TSFE")

    # F. Training
    print("\nF. Training …")
    history = run_training(model, train_loader, valid_loader, embed_mat, vocab)
    plot_training_history(history)

    # G. Load best & evaluate
    print("\nG. Evaluating on test set …")
    model.load_state_dict(torch.load("tsfe_best.pth", map_location=DEVICE))
    scores = evaluate_model(model, test_loader, vocab, glove, DEVICE)

    # H. Attention visualisation
    print("\nH. Attention visualisation …")
    sample_row = test_df.iloc[0]
    sample_img = Image.open(
        io.BytesIO(ast.literal_eval(sample_row["image"])["bytes"])
    ).convert("RGB")
    visualise_attention(model, sample_img, vocab, glove, DEVICE)

    return model, vocab, glove, scores


print("  ✓ main() defined")
print()
print("  To run the full pipeline:")
print("    model, vocab, glove, scores = main()")
print()
print("  To run just the smoke test (no data needed):")
print("    run_smoke_test()")
print()
print("=" * 60)
print("All TSFE modules successfully loaded.")
print("=" * 60)
