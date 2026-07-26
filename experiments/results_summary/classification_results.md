# CCDance Dance Quality Grade Classification Results

| Model | Accuracy | Accuracy Std | Macro-F1 | Macro-F1 Std | QWK | QWK Std |
|---|---|---|---|---|---|---|
| Random Baseline | 0.333 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 |
| XGBoost (handcrafted) | 0.434 | 0.046 | 0.431 | 0.049 | 0.262 | 0.039 |
| SVM (handcrafted) | 0.296 | 0.018 | 0.270 | 0.029 | 0.098 | 0.024 |
| PoseLSTM | 0.343 | 0.075 | 0.324 | 0.065 | 0.097 | 0.133 |
| ST-GCN | 0.343 | 0.064 | 0.321 | 0.058 | 0.143 | 0.107 |
| Pose Transformer | 0.332 | 0.009 | 0.166 | 0.003 | 0.000 | 0.000 |
| Two-Stage (DanceMVP) | 0.341 | 0.015 | 0.330 | 0.000 | 0.126 | 0.000 |
| USDL (Tang et al. CVPR 2020) | 0.326 | 0.028 | 0.204 | 0.031 | 0.020 | 0.047 |
| CoRe (Yu et al. ICCV 2021) | 0.326 | 0.015 | 0.191 | 0.036 | -0.000 | 0.000 |
| VL-Transformer (Chen, SciRep 2025) | 0.341 | 0.054 | 0.254 | 0.072 | 0.047 | 0.085 |
| LeViT-Hybrid (Wang, SciRep 2025) | 0.319 | 0.018 | 0.170 | 0.015 | -0.036 | 0.044 |
| Graph-Transformer (Han et al., SciRep 2026) | 0.333 | 0.000 | 0.167 | 0.000 | 0.000 | 0.000 |
