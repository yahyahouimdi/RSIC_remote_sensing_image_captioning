# Remote Sensing Image Captioning — CNN Encoder + LSTM Decoder

This notebook trains and evaluates an image-captioning model on the **RSICD** (Remote Sensing Image Captioning Dataset). It follows a classic encoder–decoder architecture: a frozen **ResNet-50** CNN encoder extracts image features, and an **LSTM** decoder generates captions word by word. The notebook trains a baseline model, then a regularized/augmented "improved" model, and evaluates both with **BLEU** and **CIDEr**.

## Dataset

- **Source:** [`thedevastator/rsicd-image-caption-dataset`](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset) on Kaggle, downloaded automatically via `kagglehub`.
- **Splits:** `train.csv`, `valid.csv`, `test.csv`, each with `filename`, `captions` (a list of ~5 human-written captions per image), and `image` (raw image bytes).
- Requires a Kaggle account/API token configured in the Colab environment for `kagglehub` to download the data.

## Environment

- Designed to run on **Google Colab** with a GPU runtime (tested on a T4).
- Key libraries: `torch`, `torchvision`, `nltk`, `pandas`, `matplotlib`, `pycocoevalcap` (installed in-notebook), `kagglehub`.
- The first cell verifies imports, downloads required NLTK tokenizer data, and confirms GPU availability.

## Notebook structure

The notebook is organized as a sequence of build-and-test steps, followed by full training, inference, and evaluation:

| Section | What it does |
|---|---|
| Test imports | Verifies libraries, NLTK data, and GPU/device setup. |
| Loading Dataset | Downloads RSICD via `kagglehub` and loads train/valid/test CSVs. |
| Image & Caption Extraction | Helper functions to decode image bytes and parse the caption list for a row. |
| Building Vocabulary | Tokenizes all captions and builds a word→index vocabulary (`min_freq=3`), with `<PAD>`, `<START>`, `<END>`, `<UNK>` special tokens. |
| Image Transforms | Standard ImageNet-style resize/normalize pipeline for the baseline model. |
| Dataset Class | `RSICDDataset`: returns `(image_tensor, encoded_caption)` pairs (first caption per image). |
| DataLoader | Custom `collate_fn` plus `train_loader` / `valid_loader` (batch size 16). |
| Model Architecture | `EncoderCNN` (frozen ResNet-50 + trainable projection layer) and `DecoderRNN` (embedding + single-layer LSTM + linear output). One forward pass is sanity-checked. |
| Loss & Optimizer | `CrossEntropyLoss` (ignoring the `<PAD>` index) and Adam optimizer; one backward pass is sanity-checked. |
| One Training Iteration | End-to-end smoke test of a single training step before committing to full training. |
| **Full baseline training** | Trains the baseline `encoder`/`decoder` for 10 epochs over the real `train_loader`/`valid_loader`, tracking `history['train_loss']` / `history['val_loss']`, and checkpointing the best epoch. |
| **Improved model** | A second, regularized model: `DecoderRNN_Improved` (2-layer LSTM + dropout), heavier data augmentation (`RandomCrop`, flips, color jitter, rotation), an `Adam` optimizer with `ReduceLROnPlateau`, and early stopping. Trains for up to 10 epochs and plots a loss comparison against the baseline. |
| Google Drive checkpointing | Mounts Google Drive and redirects all checkpoint saves to `MyDrive/image_captioning_models/`, so trained weights survive a Colab disconnect/runtime reset. |
| Inference & Testing | Loads the best saved checkpoint and defines two decoding strategies: `generate_caption_greedy` (argmax at each step) and `generate_caption_beam_search` (beam width 5). Prints and visualizes sample generations against ground-truth captions. |
| **Evaluation: BLEU & CIDEr** | Scores greedy vs. beam-search decoding on a sample of validation images using corpus-level BLEU-1–4 (with smoothing) and CIDEr, summarized in a comparison table. |
| Download checkpoint | One-click download of a checkpoint from Drive straight to your local machine via `google.colab.files.download`. |

## Model details

**Encoder (`EncoderCNN`)**
- ResNet-50 pretrained on ImageNet, all convolutional layers frozen.
- Final pooled features (2048-d) projected to `embed_size=256` via a trainable linear layer — this is the only part of the encoder that gets fine-tuned.

**Decoder (baseline `DecoderRNN`)**
- Embedding (256-d) → single-layer LSTM (hidden size 512) → linear projection to vocabulary size.
- Image features are prepended as the first "timestep" input to the LSTM.

**Decoder (`DecoderRNN_Improved`)**
- Same idea, but 2-layer LSTM with dropout (0.5) between layers and on the embeddings/outputs, trained on stronger data augmentation and with a random-caption-per-image sampling strategy (instead of always using the first caption).

## Training

Both models are trained with `CrossEntropyLoss` (padding index ignored) and Adam. The improved model additionally uses:
- Gradient clipping (`max_norm=5.0`)
- `ReduceLROnPlateau` learning-rate scheduling
- Early stopping (patience 3)

Checkpoints (`best_model_baseline.pth`, `best_model_improved.pth`) are saved whenever validation loss improves, and are written directly to Google Drive (`/content/drive/MyDrive/image_captioning_models/`) rather than the ephemeral Colab local disk.

## Inference

Two decoding strategies are implemented:
- **Greedy search** — picks the highest-probability token at every step.
- **Beam search** (width 5) — keeps the top-k partial sequences at each step and returns the highest-scoring complete sequence.

## Evaluation

The final evaluation section scores both decoding strategies against all human reference captions for each sampled validation image:

- **BLEU-1 to BLEU-4** via `nltk.translate.bleu_score.corpus_bleu`, with smoothing to handle short generated captions.
- **CIDEr** via `pycocoevalcap.cider.Cider`.

Results are assembled into a `pandas` DataFrame indexed by method (`Greedy` vs `Beam Search (k=5)`) with one column per metric. The number of validation images scored is controlled by `NUM_EVAL_SAMPLES` (default 200) — reduce it for a quick check, or set it higher (up to the full validation set) for a more reliable estimate.

## How to run

1. Open the notebook in Google Colab and select a GPU runtime.
2. Run the cells top to bottom. The first "test" cells validate each pipeline component in isolation before full training starts — safe to keep them, they double as a smoke test.
3. When you reach the Google Drive cell, authorize the Drive mount so checkpoints save persistently.
4. Full training (baseline + improved) will take the longest; both loops print per-epoch train/val loss and save best-checkpoint messages.
5. Run the Inference and Evaluation sections to generate sample captions and compute the BLEU/CIDEr comparison table.
6. Use the final "download checkpoint" cell any time you want a local copy of a trained model.

## Known limitations

- Captions are lower-cased and tokenized with NLTK's `word_tokenize`; punctuation handling is simple and can affect BLEU scores.
- Vocabulary is built with a minimum frequency threshold (`min_freq=3`), so rare words map to `<UNK>` and may lower fluency for uncommon scene descriptions.
- The improved model's early stopping means its training history length can differ from the baseline's fixed 10 epochs — this only affects the x-axis length in the loss comparison plot, not the correctness of either run.