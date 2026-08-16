# Existing Architecture Documents

This folder contains the main research architectures reviewed during the project. Each document represents a different idea used in image captioning, visual-language modeling, attention refinement, and semantic enhancement. The goal of this collection is to provide a quick and readable summary of the most relevant literature that influenced the design and comparison of the TSFE system.

---

## Quick Overview

| No. | Architecture | File | Main focus | Why it matters for this research |
|---|---|---|---|---|
| 1 | TSFE | [TSFE (Two-Stage Feature Enhancement Network).pdf](TSFE%20%28Two-Stage%20Feature%20Enhancement%20Network%29.pdf) | Two-stage visual-language enhancement | This is the direct baseline/reference model for the current work |
| 2 | CRSR | [CRSR (Cross-Modal Retrieval and Semantic Refinement).pdf](CRSR%20%28Cross-Modal%20Retrieval%20and%20Semantic%20Refinement%29.pdf) | Cross-modal alignment and semantic refinement | Shows how retrieval + refinement can improve meaning and text quality |
| 3 | Feature refinement and rethinking attention | [Feature refinement and rethinking attention.pdf](Feature%20refinement%20and%20rethinking%20attention.pdf) | Attention redesign and feature refinement | Highlights the importance of better attention mechanisms |
| 4 | Mask-Guided Transformer | [Mask-Guided Transformer.pdf](Mask-Guided%20Transformer.pdf) | Spatial focus using mask guidance | Useful for region-based attention and targeted feature selection |
| 5 | MSR-capnet | [MSR-capnet.pdf](MSR-capnet.pdf) | Multi-scale / structural feature modeling | Relevant for improved object relationships and feature representation |
| 6 | PCSFTr | [PCSFTr(Positional-Channel Semantic Fusion Transformer).pdf](PCSFTr%28Positional-Channel%20Semantic%20Fusion%20Transformer%29.pdf) | Positional + channel + semantic fusion | Demonstrates advanced transformer fusion strategies |
| 7 | Feature refinement and rethinking attention | [Feature refinement and rethinking attention.pdf](Feature%20refinement%20and%20rethinking%20attention.pdf) | Attention refinement | Reinforces the idea of better visual feature processing |

> Note: The folder contains several related model families. Most of them share the same goal: improving image understanding, feature alignment, and caption generation quality.

---

## 1) TSFE (Two-Stage Feature Enhancement Network)

- File: [TSFE (Two-Stage Feature Enhancement Network).pdf](TSFE%20%28Two-Stage%20Feature%20Enhancement%20Network%29.pdf)
- Brief description: This architecture is built around a two-stage enhancement approach, where visual features are first improved and then used more effectively in the caption-generation process.
- Main idea: improve the quality of extracted image features before language generation so the model can produce more accurate and meaningful descriptions.
- Relevance: This is the core architecture of the project and the main model used for comparison and improvement.

---

## 2) CRSR (Cross-Modal Retrieval and Semantic Refinement)

- File: [CRSR (Cross-Modal Retrieval and Semantic Refinement).pdf](CRSR%20%28Cross-Modal%20Retrieval%20and%20Semantic%20Refinement%29.pdf)
- Brief description: This model focuses on linking visual content and textual semantics through cross-modal retrieval and refinement, aiming to better align image evidence with language descriptions.
- Main idea: retrieve the most relevant visual-textual information and refine semantic understanding before generating the caption.
- Relevance: Important for understanding how semantic alignment can improve sentence coherence and reduce weak or repetitive captions.

---

## 3) Feature refinement and rethinking attention

- File: [Feature refinement and rethinking attention.pdf](Feature%20refinement%20and%20rethinking%20attention.pdf)
- Brief description: This paper investigates how attention should be designed and used in vision-language systems, especially focusing on feature refinement and better information filtering.
- Main idea: attention is not only a weighting mechanism, but also a way to select the most useful visual information and suppress noisy or irrelevant patterns.
- Relevance: Very relevant to this project because improved attention usually leads to better scene understanding and more precise captions.

---

## 4) Mask-Guided Transformer

- File: [Mask-Guided Transformer.pdf](Mask-Guided%20Transformer.pdf)
- Brief description: This architecture uses mask-guided mechanisms to force the model to focus on informative image regions and reduce attention to irrelevant areas.
- Main idea: region selection and localization are improved by masking or controlling the visual field, which helps the model pay attention to objects and context.
- Relevance: Strongly related to object-focused captioning and better spatial understanding in complex scenes.

---

## 5) MSR-capnet

- File: [MSR-capnet.pdf](MSR-capnet.pdf)
- Brief description: This model combines multi-scale representation learning and a capsule-style view of features to preserve structure and relationships among visual elements.
- Main idea: capture richer relations between parts, objects, and scene context rather than relying only on flat feature maps.
- Relevance: Useful for understanding how structural information can improve semantic correctness and visual grounding in captions.

---

## 6) PCSFTr (Positional-Channel Semantic Fusion Transformer)

- File: [PCSFTr(Positional-Channel Semantic Fusion Transformer).pdf](PCSFTr%28Positional-Channel%20Semantic%20Fusion%20Transformer%29.pdf)
- Brief description: This model emphasizes fusion of positional, channel, and semantic information through transformer-based processing, improving the internal representation of image features.
- Main idea: combine spatial location, channel response, and semantic context to create a more informative representation for caption generation.
- Relevance: Important for understanding how transformer-based multi-source fusion can improve feature integration and generate richer descriptions.

---

## Summary of the Research Trend

Across all these models, a common pattern appears:

- better feature extraction leads to better image understanding;
- attention and fusion mechanisms improve object and context awareness;
- semantic refinement helps produce more natural and meaningful captions;
- reducing repetition and improving language quality is a major research goal.

This is exactly the motivation behind the improved TSFE approach: enhancing visual representation and language generation together to produce captions that are more coherent, more informative, and more semantically accurate.

---

## Final Note

This folder is intended as a research reference library for the TSFE project. It gathers the architecture ideas and related methods used to compare, justify, and improve the model design. For quick reading, each paper was grouped by its main contribution and its connection to the image-captioning task.

If needed, this file can be expanded later with:

- a full abstract summary for each paper;
- a comparison table of strengths and weaknesses;
- a list of which models inspired the final TSFE design.
