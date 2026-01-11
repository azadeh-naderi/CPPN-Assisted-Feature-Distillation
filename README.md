# CPPN-Assisted Feature Distillation (Images)

This project explores **using Compositional Pattern Producing Networks (CPPNs) as a helper module to distill richer features and knowledge from image datasets**. The core idea is to convert images into structured coordinate-based representations (and/or generated feature maps) using a CPPN, then use those representations to support **knowledge distillation** and improved representation learning for downstream models.

## Motivation

Standard distillation typically transfers knowledge from a teacher to a student via logits or intermediate features. Here, we add a CPPN-based pathway that can:

- Provide **smooth, structured, coordinate-conditioned representations**
- Act as an **auxiliary feature generator** to enrich the supervision signal
- Encourage the student to learn **more transferable, spatially meaningful features**
- Support experimentation with **image-to-coordinate encodings** and **feature-map distillation**

## High-Level Approach

1. **Input Image → CPPN Helper**
   - Convert an image to a coordinate grid representation (e.g., `(x, y)` plus optional channels)
   - Use a CPPN to produce a structured output (e.g., a reconstructed image, feature map, or embedding)

2. **Teacher / Student Distillation**
   - Distill knowledge using one or more signals:
     - Teacher logits (soft targets)
     - Teacher intermediate features
     - CPPN-produced auxiliary representations
     - Student feature alignment to CPPN/teacher outputs

3. **Training Objective (typical)**
   - A weighted combination of:
     - Classification loss (CE)
     - Distillation loss (KL on softened logits)
     - Feature matching (L2 / cosine / attention-based matching)
     - CPPN reconstruction or auxiliary loss (optional)

## What’s Inside

- CPPN helper module for generating structured representations from images
- Distillation training loops (teacher → student) with optional CPPN-assisted losses
- Dataset support for common image benchmarks (e.g., CIFAR-10/100, Fashion-MNIST, etc.)
- Logging utilities to track:
  - student / teacher accuracy
  - distillation loss curves
  - CPPN auxiliary loss (if used)
  - representation similarity metrics (optional)

## Repository Structure (suggested)

```text
cppn-feature-distillation/
├── README.md
├── requirements.txt
├── configs/
│   ├── cifar10.yaml
│   ├── cifar100.yaml
│   └── fashion_mnist.yaml
├── src/
│   ├── data/
│   │   ├── datasets.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── cppn.py
│   │   ├── teacher.py
│   │   └── student.py
│   ├── distill/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── metrics.py
│   ├── utils/
│   │   ├── seed.py
│   │   ├── logger.py
│   │   └── checkpoints.py
│   └── main.py
└── scripts/
    ├── train_distill.sh
    └── eval.sh
