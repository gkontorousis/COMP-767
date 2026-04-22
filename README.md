# COMP-767 Project Overview

This repository contains code and artifacts for sequence-rule learning experiments and evaluation.

## Project Layout

- `training.ipynb`  
  Main notebook used for experiments. It contains the core training loops, the evaluation loops, the obtaining of examples, baseline comparison setup etc. 
  
- `mdl_methods.py`  
  Shared MDL-oriented modeling utilities, including:
  - model/tokenizer loading and device/dtype setup,
  - prompt construction and chat rendering,
  - conditional/Bayesian scoring (`score_explanation`),
  - policy generation/sampling helpers for training and inference.

- `scratch/`  
  Early experimental scripts and sanity checks used during initial development. These are exploratory and not part of the primary pipeline.

- `project_paper/` and `poster/`  
  LaTeX sources for the project paper and poster, respectively.

- `guides/`  
  Practical guides used during development and experimentation workflows.
