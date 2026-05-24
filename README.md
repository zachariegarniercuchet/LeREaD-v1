# LeREaD v1: A Dataset for Legal Reference Extraction and Disambiguation

## Overview

LeREaD v1 is a comprehensive dataset designed for legal reference extraction and disambiguation tasks. This repository contains annotated legal documents, evaluation scripts, and supporting materials for research and development in the legal NLP domain.

## Dataset Contents

### Annotated Documents (`data/annotated/`)
The main annotated dataset is organized into standard machine learning splits:
- **train/**: Training set documents with full annotations
- **test/**: Test set documents with full annotations  
- **dev/**: Development/validation set documents with full annotations

### Original Documents (`data/original/`)
Unannotated versions of the same documents used in the annotated split, organized by the same train/test/dev split structure. These serve as reference copies of the original legal documents.

### Supplementary Materials (`data/supplementary/`)

#### Inter-Annotator Agreement & LLM-Assisted Annotation Study
- **manual_verification/**: LLM-generated annotations that have been verified and corrected by human annotators. Used for evaluating LLM annotation quality and anchoring bias.
- **manual_extraction/**: Purely human-generated annotations focused on extraction tasks. Primarily used to support inter-annotator agreement (IAA) analysis.
- **llm_extraction/**: LLM-only versions of documents without human verification. Used as baseline for comparing LLM performance against human-verified annotations.

These supplementary materials support studies on:
- Inter-annotator agreement metrics
- LLM-assisted annotation productivity and quality improvements
- Quality assessment and comparison of human vs. LLM annotations
- Analysis of potential anchoring bias in LLM-assisted workflows

## Repository Structure

```
LeREaD/
├── data/
│   ├── annotated/          # Main annotated dataset (train/test/dev)
│   ├── original/           # Unannotated document copies (train/test/dev)
│   ├── supplementary/      # Study materials (verification, LLM extractions)
│   └── candidate_pool_metadata.csv
├── src/
│   ├── evaluation scripts and utilities
│   ├── processing utilities
│   └── prompts/            # Prompt templates and guidelines
├── dataset_statistics.ipynb
├── evaluation.ipynb
└── README.md
```

## Usage

The repository includes Jupyter notebooks for analysis and evaluation:
- `dataset_statistics.ipynb`: Statistical overview of the dataset
- `evaluation.ipynb`: Evaluation framework and metrics

## Acknowledgments

### Funding & Resources

This project was supported by Compute Canada, which provided computational resources essential for large-scale model benchmarking.

### AI-Assisted Development

This codebase has been developed with the assistance of AI tools including but not limited to:
- ChatGPT (OpenAI)
- Claude (Anthropic)
- GitHub Copilot (Microsoft)

In accordance with ACL guidelines, we acknowledge the use of code assistance tools during the development of this project. All generated code was reviewed for correctness and compliance with licensing requirements.

## Citation

[Citation information to be added upon publication]

## License

This work is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

Under this license, you are free to:
- Share and adapt the material for non-commercial purposes
- Provide appropriate attribution to the original authors

You are not permitted to:
- Use this material for commercial purposes

For more details, see [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/)

---

**Note**: This is an anonymized version of the LeREaD v1 repository prepared for the ACL review process.
