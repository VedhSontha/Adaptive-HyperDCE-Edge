# Zero-DCE Non-Reference Loss Functions

Mathematical formulations for training Zero-DCE without reference ground truths:
- **Spatial Consistency Loss ($L_{spa}$)**: evaluates difference gradients between adjacent regions.
- **Exposure Control Loss ($L_{exp}$)**: measures local exposure deviance.
- **Color Constancy Loss ($L_{col}$)**: corrects pixel channel color casts.
- **Illumination Smoothness Loss ($L_{tv}$)**: preserves illumination monotonicity.
