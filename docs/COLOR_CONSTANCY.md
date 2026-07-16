# Color Constancy Loss ($L_{col}$)

Formulation to balance color channels:
- Equation: $L_{col} = \sum_{\forall (p, q) \in C} (J^p - J^q)^2$ where $(p, q)$ are channel combinations (R, G, B).
- Eliminates color casts from enhanced results.
