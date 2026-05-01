# Foundation Models for Computational Pathology

Benchmarking and evaluation pipeline for histopathology foundation models, combining whole-slide image representations and pathology report embeddings for cross-modal retrieval tasks.

## Overview
This repository documents the evaluation layer of a master's thesis on multimodal retrieval in computational pathology.
The public version focuses on structure, reporting and reproducibility, while leaving out the clinical material and restricted model assets used in the original work.

## What the repository covers
- cross-modal retrieval between pathology images and report text;
- Recall@K benchmarking for image-to-report and report-to-image retrieval;
- AI4SKIN, SICAP and AI4SKIN vs ASSIST evaluation settings;
- majority-vote summaries and global dashboard generation;
- tokenizer and truncation analysis at the documentation level.

## Models referenced in the thesis workspace
- KEEP
- TITAN
- CONCH
- MUSK
- Patho-CLIP
- PRISM

## Datasets referenced in the thesis workspace
- AI4SKIN
- SICAP
- ASSIST

## Included here
- public reporting scripts;
- a cleaned notebook template for the evaluation dashboards;
- a notebook builder for the public evaluation output;
- aggregated, non-sensitive figures;
- example configuration and synthetic sample files;
- documentation for datasets, evaluation and privacy constraints.

## Data availability
This repository does not include clinical data, pathology reports, private checkpoints or restricted model weights.
It is intended to document the pipeline and preserve a reproducible public version of the analysis structure.

## Repository structure
- `docs/` for project notes, datasets, evaluation and privacy guidance
- `src/` for public Python utilities and reporting code
- `notebooks/` for cleaned notebook templates
- `figures/` for public figures
- `examples/` for synthetic examples
- `configs/` for configuration templates

## Running the public version
1. Provide your own properly licensed datasets and model checkpoints.
2. Adapt `configs/example_config.yaml`.
3. Run the reporting utilities on your local directory structure.

## Public analysis assets

- `src/evaluation/global_results_reporter.py`
  Scripted reporting layer for global metrics and figures.
- `src/visualization/build_public_dashboard_notebook.py`
  Helper used to generate the cleaned public dashboard notebook.
- `notebooks/evaluation_output_template.ipynb`
  Portfolio-facing notebook template for the final visual synthesis.

## Notes
- clinical source files, spreadsheets, embeddings and raw reports are not published;
- some thesis notebooks were reworked into simpler public templates;
- full reproduction still depends on private or licensed assets that are not distributed here.
