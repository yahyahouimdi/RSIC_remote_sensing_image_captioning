# Remote Sensing Image Captioning — CNN Encoder + LSTM Decoder

This notebook implements a **CNN-LSTM architecture** for image captioning on the **RSICD** (Remote Sensing Image Captioning Dataset). The notebook focuses on **data pipeline setup and testing**, including environment verification, dataset loading, preprocessing, vocabulary building, and DataLoader creation. It provides a solid foundation for training encoder–decoder models where a **ResNet-50** CNN encoder extracts visual features and an **LSTM** decoder generates captions word by word.

## Dataset

- **Source:** [`thedevastator/rsicd-image-caption-dataset`](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset) on Kaggle, downloaded automatically via `kagglehub`.
- **Splits:** `train.csv`, `valid.csv`, `test.csv`, each with `filename`, `captions` (a list of ~5 human-written captions per image), and `image` (raw image bytes).
- Requires a Kaggle account/API token configured in the Colab environment for `kagglehub` to download the data.

## Environment

- Designed to run on **Google Colab** with a GPU runtime (tested on a T4).
- Key libraries: `torch`, `torchvision`, `nltk`, `pandas`, `matplotlib`, `pycocoevalcap` (installed in-notebook), `kagglehub`.
- The first cell verifies imports, downloads required NLTK tokenizer data, and confirms GPU availability.

## Notebook structure

The notebook is organized as a **pipeline testing workflow**, verifying each component before model training:

| Section | What it does |
|---|---|
| **Part 1: Test imports** | Verifies all required libraries (PyTorch, torchvision, NLTK, pandas, matplotlib), downloads NLTK tokenizer data, and checks GPU/device availability. |
| **Part 2: Loading Dataset** | Downloads the RSICD dataset via `kagglehub` from Kaggle and loads `train.csv`, `valid.csv`, `test.csv`. Displays dataset statistics (number of samples per split). |
| **Part 3: Image & Caption Extraction** | Tests helper functions to extract PIL images from encoded bytes and parse caption lists. Displays a sample image and its captions. |
| **Part 4: Building Vocabulary** | Tokenizes all captions across train and valid splits using NLTK's `word_tokenize`. Creates a vocabulary with a minimum frequency threshold (`min_freq=3`), special tokens (`<PAD>`, `<START>`, `<END>`, `<UNK>`), and example encoding/decoding. |
| **Part 5: Testing Image Transforms** | Applies standard ImageNet-style preprocessing (resize to 224×224, normalize with ImageNet mean/std). Visualizes before/after transformations. |
| **Part 6: Testing Dataset Class** | Implements `RSICDDataset` (custom PyTorch Dataset) that returns `(image_tensor, encoded_caption)` pairs. Tests random sampling and batch-compatible indexing. |
| **Part 7: Testing DataLoader** | Creates batched data loaders for train and valid splits with custom collation to handle variable-length captions. |

## Key features

- **Complete data pipeline**: Load, extract, preprocess, and batch RSICD images and captions
- **Modular testing**: Each pipeline step is tested independently before use
- **Vocabulary management**: Automatic tokenization with frequency-based filtering
- **PyTorch-ready**: Custom Dataset and DataLoader classes for easy integration with training loops
- **GPU support**: Automatic GPU detection and tensor placement on available device

## Preprocessing pipeline

1. **Image loading** → Extract PIL images from encoded bytes
2. **Tokenization** → NLTK word tokenization with lowercase normalization
3. **Vocabulary encoding** → Map tokens to indices with `<UNK>` fallback for rare words
4. **Image transforms** → Resize (224×224) + ImageNet normalization (mean/std)
5. **Batching** → PyTorch DataLoader with custom collation for variable-length captions

## How to run

1. Open the notebook in **Google Colab** (or a local Jupyter environment with GPU support).
2. Select a **GPU runtime** (tested on T4 GPUs in Colab).
3. Run cells sequentially from top to bottom:
   - **Parts 1–7** test and validate each pipeline component in isolation
   - Each test prints verification messages (`✓`) and displays sample data
   - Safe to run multiple times; no checkpoints or side effects
4. After Part 7, the data pipeline is ready for model training in subsequent notebooks or scripts.

## Output and verification

Each section prints diagnostic information:
- Import success/failure and device type (CPU/GPU)
- Dataset statistics (number of splits, samples, features)
- Sample images and captions (visualized with matplotlib)
- Vocabulary statistics (size, special tokens, top frequent words)
- Transform verification (tensor shapes, value ranges)
- Dataset indexing tests (batch-compatible shapes, caption lengths)

This structured testing approach ensures **data integrity** before model training begins.

## Known considerations

- **Captions** are lower-cased and tokenized with NLTK's `word_tokenize`; punctuation handling is basic.
- **Vocabulary** uses a frequency threshold (`min_freq=3`); rare words map to `<UNK>`, which may affect caption fluency.
- **Image batch size** and **sequence length** are configurable in the Dataset and DataLoader classes.
- **GloVe embeddings** (available in the project's `glove.6B` folder) can be loaded for word representation in downstream training.
- **RSICD dataset** requires a Kaggle account with the `kagglehub` API token configured in Colab.