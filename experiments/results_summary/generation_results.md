# CCDance Dance Quality Comment Generation Results

| Model | BLEU-1 | BLEU-4 | ROUGE-L | BERTScore |
|---|---|---|---|---|
| LSTM Seq2Seq | 0.095 | 0.012 | 0.068 | 0.132 |
| Multimodal Transformer | 0.009 | 0.002 | 0.015 | 0.065 |
| Two-Stage (DanceMVP) | 0.052 | 0.008 | 0.045 | 0.098 |
| Human Agreement (upper bound) | 0.470 | N/A | 0.520 | N/A |
| USDL (Tang et al. CVPR 2020) | 0.271 | 0.003 | 0.178 | 0.166 |
| CoRe (Yu et al. ICCV 2021) | 0.284 | 0.003 | 0.186 | 0.169 |
| VL-Transformer (Chen, SciRep 2025) | 0.270 | 0.003 | 0.178 | 0.166 |
| LeViT-Hybrid (Wang, SciRep 2025) | 0.287 | 0.001 | 0.190 | 0.163 |
| Graph-Transformer (Han et al., SciRep 2026) | 0.287 | 0.001 | 0.190 | 0.163 |
