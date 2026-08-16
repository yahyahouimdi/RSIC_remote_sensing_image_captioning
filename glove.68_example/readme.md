# GloVe 6B Example Folder

This folder contains a lightweight local copy of the GloVe 6B word embeddings, created for project use and for uploading to GitHub without storing the full original dataset in the repository.

The original GloVe 6B embeddings are widely used in natural language processing and are especially useful for text representation, semantic similarity, and word embedding experiments. Because the full files can be large, this folder keeps a smaller, project-friendly version for demonstration, testing, and development purposes.

---

## What is included

The folder contains the following embedding files:

- glove.6B.50d.txt
- glove.6B.100d.txt
- glove.6B.200d.txt
- glove.6B.300d.txt

These files contain pre-trained word vectors where:

- each line corresponds to a word and its embedding vector;
- the number after `6B` indicates the training corpus size (6 billion tokens);
- the number after `d` indicates the vector dimension (50, 100, 200, or 300).

For example:

- `glove.6B.50d.txt` = 50-dimensional vectors
- `glove.6B.100d.txt` = 100-dimensional vectors
- `glove.6B.200d.txt` = 200-dimensional vectors
- `glove.6B.300d.txt` = 300-dimensional vectors

---

## What is GloVe?

GloVe stands for Global Vectors for Word Representation. It is a model for learning word embeddings from a large text corpus.

The main idea behind GloVe is to learn vectors such that words that appear in similar contexts have similar representations. This allows the model to capture semantic relationships, such as:

- man vs woman
- king vs queen
- Paris vs France
- dog vs animal

These embeddings are useful in many NLP tasks, including:

- text classification;
- sentiment analysis;
- semantic similarity;
- machine translation support;
- caption generation and multimodal learning;
- feature initialization for neural networks.

---

## Why this folder exists

The full GloVe 6B files are large and may not be practical to store or push to GitHub in every project copy. This folder provides a local example/version for:

- testing code locally;
- running lightweight experiments;
- presenting the project in GitHub or academic storage;
- keeping the repository manageable.

This is a practical subset or copy intended to support the project workflow without storing the full original public resource in the repository itself.

---

## Source and official download

The original GloVe embeddings were released by the Stanford NLP group.

Official source:

- Stanford NLP GloVe project: https://nlp.stanford.edu/projects/glove/

The direct download page for the 6B vectors is usually available from the project releases, where you can download the `.txt` files such as:

- glove.6B.zip
- glove.6B.50d.txt
- glove.6B.100d.txt
- glove.6B.200d.txt
- glove.6B.300d.txt

A person can download the full original version from the official Stanford page and place it in the same structure used by this project.

---

## Recommended usage

These vectors are often loaded with Python using a simple lookup dictionary. In projects such as image captioning, they can be used to initialize the word embedding layer or to encode textual tokens before feeding them into a language model.

This folder is especially useful when the full original GloVe package is too large for direct repository storage or collaboration constraints.

---

## Note

This folder is a project example copy of the original Stanford GloVe 6B embeddings. It is intended for research, testing, and local development, and it should be replaced or supplemented by the official download when full-scale experimentation is needed.

The original embeddings remain the authoritative source.
