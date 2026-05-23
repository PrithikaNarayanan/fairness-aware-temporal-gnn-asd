# Fairness-Aware Temporal Graph Neural Networks for ASD Classification

A temporal graph neural network for classifying Autism Spectrum Disorder from multi-site resting-state fMRI connectivity, with an equalised-odds fairness objective to address acquisition-site bias.

Final project for ST457 (Graph Data Science), LSE Department of Statistics.

## Overview

Multi-site fMRI datasets like ABIDE I+II suffer from scanner and acquisition-site variability large enough that subjects cluster by scanner rather than by diagnosis. Most graph-based ASD classifiers compound this problem by collapsing each subject's scan into a single static connectivity matrix, ignoring the way functional networks reconfigure over time.

This project addresses both issues:

- **Temporal GNN architecture.** Each subject is represented as a sequence of sliding-window connectivity graphs (50-TR windows, 25-TR stride), processed by a GCN spatial encoder per window and aggregated by a GRU across windows.
- **Fairness-aware training.** A differentiable equalised-odds surrogate is added to the cross-entropy loss, equalising true and false positive rates across acquisition sites without penalising genuine prevalence differences.

## Dataset

ABIDE I + ABIDE II, accessed via the Preprocessed Connectomes Project (C-PAC pipeline, CC200 parcellation).

| | |
|---|---|
| Subjects | 1,230 (591 ASD, 639 TD) |
| Sites | 33 |
| ROIs | 200 (CC200 atlas) |
| Splits | 70 / 15 / 15, stratified by site and diagnosis |

346 subjects with incomplete brain coverage were excluded rather than imputed, to avoid biasing connectivity features.

## Pipeline

1. **Connectivity estimation.** Pearson correlation matrices per sliding window.
2. **Fisher z-transform.** Stabilises the sampling distribution near ±1.
3. **Site harmonisation.** Within-site z-score normalisation (ComBat-equivalent without covariate model).
4. **Top-15% edge thresholding.** Subject-wise percentile rule, producing tight density distribution.
5. **Node features.** Full connectivity row concatenated with a learnable ROI identity embedding.
6. **Sliding-window decomposition.** 50-TR windows, 25-TR stride, yielding 2–16 graphs per subject.

## Architecture

```
Per window:   G(w) ──► GCN (2 layers, d=64) ──► mean-pool ──► z(w) ∈ R^64
Across:       (z(0), ..., z(W-1)) ──► GRU (d=64) ──► h ──► MLP ──► p(ASD)
```

GCN was chosen over GIN and GAT for stability with continuous brain-connectivity weights on a small-sample multi-site cohort; the GAT baseline confirms higher seed variance.

## Results

Test set, mean ± std over 3 seeds (W_max=8), N=185.

| Configuration | AUC | EO gap | DP gap | Site-AUC var |
|---|---|---|---|---|
| LR baseline (19,900 features) | 0.700 | – | – | – |
| GCN static | 0.600 ± 0.044 | 0.022 | 0.0060 | 0.018 |
| GAT static | 0.626 ± 0.024 | 0.037 | 0.0116 | 0.022 |
| ResidualGCN | 0.614 ± 0.014 | 0.034 | 0.0106 | 0.025 |
| Temporal (no fairness) | 0.611 ± 0.021 | 0.016 | 0.0067 | 0.020 |
| **Temporal + EO fairness** | **0.620 ± 0.011** | **0.015** | **0.0061** | **0.017** |

Key findings:

- Logistic regression on the full connectivity vector outperforms all GNN configurations on raw AUC. Likely causes: ~85% of pairwise signal is discarded by edge thresholding, N=1,230 is below the regime where graph priors reliably beat strong linear baselines, and ASD-vs-TD is fundamentally a pairwise problem.
- The fairness objective produces modest but consistent improvements on EO and DP gaps, with side-benefits in sex and age fairness (incidental, through correlation with site).
- Longer temporal context helps: W_max=16 reaches 0.636 AUC, up from 0.616 at W_max=2.
- Calibration improves with the temporal architecture (ECE 0.189 → 0.140) but is not affected by the fairness objective itself.

Published ABIDE classification AUCs cap around 72–75% due to ASD phenotypic heterogeneity, fMRI noise, and multi-site variance; results above that range typically indicate data leakage.

## Tech stack

PyTorch · PyTorch Geometric · scikit-learn · NumPy · NetworkX · Matplotlib

## Limitations

- Top-15% thresholding discards information that the linear baseline is free to use.
- The EO penalty targets acquisition sites only; gains on sex- and age-fairness are incidental.
- All experiments remain within ABIDE I+II; cross-cohort generalisation of the fairness machinery is untested.

## References

Key methodological references are listed in the paper. Notable: Kipf & Welling (2017) for GCN, Cho et al. (2014) for GRU, Hardt et al. (2016) for equalised odds, Allen et al. (2014) for dynamic functional connectivity, Fortin et al. (2018) for site harmonisation.
