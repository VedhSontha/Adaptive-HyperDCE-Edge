# Exposure Control Loss ($L_{exp}$)

Formulation to maintain average exposure range:
- Equation: $L_{exp} = \frac{1}{M} \sum_{k=1}^M || Y_k - E ||^2$
- Defaults: exposure target $E = 0.6$
- Restricts local region highlights from over/under saturation.
