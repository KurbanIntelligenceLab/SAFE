# SAFE: A Sparse Autoencoder-Based Framework for Robust Query Enrichment and Hallucination Mitigation in LLMs

**Authors**: Samir Abdaljalil, Filippo Pallucchini, Andrea Seveso, HASAN KURBAN, Fabio Mercorio, Erchin Serpedin

**Paper Link**: https://arxiv.org/abs/2503.03032
---

## 📖 Overview
SAFE (Sparse Autoencoder-based Framework for Robust Query Enrichment) is a novel framework that **detects and mitigates hallucinations in Large Language Models (LLMs)** by leveraging Sparse Autoencoders (SAEs).  

The framework operates in two main stages:
1. **Hallucination Detection** – plug-and-play detection using tools like SINdex, HaloCheck, or SelfCheckGPT.  
2. **Query Enrichment via SAEs** – extracting interpretable features to refine queries and reduce misleading activations.  

SAFE consistently improves query accuracy and mitigates hallucinations across multiple datasets and models.

---

## ✨ Key Contributions
- **SAFE Framework**: A training-free, plug-and-play method to mitigate hallucinations in LLMs.  
- **Evaluation**: Comprehensive experiments on multiple QA benchmarks (TruthfulQA, BioASQ, WikiDoc, HaluEval).  
- **Performance**: Achieves accuracy improvements up to **29.45%** compared to baseline models.  
- **Open Access**: Publicly available code and resources to foster reproducibility and further research.

---

## 📚 Citation

```bibtex
@inproceedings{
abdaljalil2025safe,
title={{SAFE}: A Sparse Autoencoder-Based Framework for Robust Query Enrichment and Hallucination Mitigation in {LLM}s},
author={Samir Abdaljalil and Filippo Pallucchini and Andrea Seveso and HASAN KURBAN and Fabio Mercorio and Erchin Serpedin},
booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
year={2025},
url={https://openreview.net/forum?id=731pFrZtGV}
}
```
 


