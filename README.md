# YWJLab-zh-whisper-brain-alignment

Code and results for the CogSci 2026 paper:

**Bridging Acoustics and Semantics: Native Language Experience and Hierarchical Temporal Integration in the Human Brain**

## Overview

This repository contains code, intermediate outputs, and analysis results for studying the alignment between Whisper-based speech representations and human MEG responses during naturalistic Chinese speech comprehension.

The project focuses on three main questions:

- whether larger Whisper models show stronger brain alignment (**scaling law**),
- whether a Chinese fine-tuned model better matches native Chinese listeners than a multilingual model (**native language effect**),
- whether acoustic- and speech-level representations rely on different temporal integration windows (**hierarchical temporal integration**).

This work has been accepted at **CogSci 2026**.

## Main Findings

Our analyses support three key conclusions:

1. **Scaling law**  
   Larger Whisper models generally achieve better encoding performance than smaller ones.

2. **Native language effect**  
   A Chinese fine-tuned Whisper model better predicts neural activity than the corresponding multilingual model, suggesting that native-language experience shapes brain-model alignment.

3. **Hierarchical temporal integration**  
   Low-level acoustic representations are associated with shorter effective integration windows, while higher-level speech representations require longer temporal context.

## Repository Structure

```bash
YWJLab-zh-whisper-brain-alignment/
├── images/
├── results/
│   ├── LightGBM_iterate-lag_sub/
│   │   ├── char-level/
│   │   └── word-level/
│   ├── LightGBM_iterate-context_sub/
│   │   ├── char-level/
│   │   └── word-level/
│   ├── LightGBM_iterate-model_sub/
│   │   ├── char-level/
│   │   └── word-level/
│   ├── mTRF_iterate-model_sub/
│   │   ├── char-level/
│   │   └── word-level/
│   ├── mTRF-PCA_iterate-model_sub/
│   │   ├── char-level/
│   │   └── word-level/
│   └── TF_iterate-model_sub/
│       ├── char-level/
│       └── word-level/
└── func/
    ├── whisper/
    └── ...
