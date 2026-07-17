# Illumination Smoothness Loss ($L_{tv}$) Calculations

Formulation to maintain parameter curve monotonicity:
- Computes total variation (TV) gradients horizontally and vertically.
- Restricts neighboring pixels from estimating high-variance parameter values.
