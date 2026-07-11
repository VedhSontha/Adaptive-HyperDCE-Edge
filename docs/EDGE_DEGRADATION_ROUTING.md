# Edge Quality Routing Policy

Dynamically managing inference pipelines on the NVIDIA Jetson Nano:
- Quality gates evaluate degradation level of input camera frames.
- Low-light frames are routed to the DCE-Net model for active enhancement.
- High-quality frames bypass networks to conserve system CPU/GPU power.
