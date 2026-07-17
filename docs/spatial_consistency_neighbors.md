# Spatial Consistency Loss ($L_{spa}$) Regions

Neighborhood pixel calculations:
- Loss evaluates a 4-neighborhood region for each pixel.
- Enforces structural continuity by keeping output difference gradients identical to raw inputs.
