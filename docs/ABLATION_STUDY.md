# Zero-DCE Ablation Study

Impact of individual loss components on enhancement quality:

| Configuration | PSNR | SSIM | Notes |
|---------------|------|------|-------|
| Full model (all losses) | 17.8 | 0.82 | Best overall |
| Without $L_{spa}$ | 16.2 | 0.74 | Spatial artifacts visible |
| Without $L_{col}$ | 17.1 | 0.79 | Slight color cast |
| Without $L_{tv}$ | 16.9 | 0.77 | Curve discontinuities |
| Without $L_{exp}$ | 15.4 | 0.71 | Severe over/under exposure |
