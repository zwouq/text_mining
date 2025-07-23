# Healthcare M&A Communication Analysis

Natural Language Processing analysis of healthcare merger and acquisition announcement communications using domain-specific embeddings, Concept Activation Vectors (CAVs), and explainable AI techniques.

## Project Overview

This study analyzes **193 healthcare M&A transactions** from 2021-2025, applying advanced NLP techniques to understand linguistic patterns in deal communications. The analysis reveals how communication strategies evolve across market cycles and identifies key linguistic predictors of deal characteristics in this sector.

## Repository Contents

### Main Analysis
- **`FINAL_text_mining.ipynb`** - Complete Jupyter notebook with all analysis code
  - Business feature extraction (17 M&A-specific features)
  - Custom Word2Vec embeddings trained on M&A corpus
  - Concept Activation Vectors (CAVs) for business concept extraction
  - Temporal analysis across market periods
  - SHAP explainability analysis

### Report
- **`Text_mining_Report_Kocheshkov.pdf`** - paper with methodology and findings

### Data (source: LSEG Workspace)
- **`transcripts/`** - Folder containing 193 M&A announcement transcripts (.txt files)
- **`FINAL_merged_transactions_and_financials.csv`** - Transaction metadata and financial information

## Key Findings

- **Strategic value communication** and **financial strength messaging** are the strongest predictors of communication patterns
- **Integration focus** shows highest individual impact (+0.12 SHAP value) in deal predictions
- Communication strategies evolve significantly across market cycles (2021 tech optimism → 2024 execution focus)
- Risk factor discussion dominates healthcare M&A communications due to regulatory complexity

## Methodology Highlights

1. **Domain-Specific Word2Vec**: Custom embeddings (200 dims, 8-word window) capture M&A semantic relationships
2. **Concept Activation Vectors**: Extract 6 business concepts with 95-100% classifier accuracy
3. **Temporal Analysis**: Track communication evolution across 5 market periods
4. **Explainable AI**: SHAP analysis provides interpretable feature importance

## Requirements

```python
pandas, numpy, matplotlib, seaborn
gensim, scikit-learn, transformers
shap, nltk, spacy
torch (for GPU acceleration)
```

## Usage

1. **Run the complete analysis**: Open `FINAL_text_mining.ipynb` and execute all cells
2. **Data paths**: Update file paths in the notebook if needed
3. **GPU optimization**: The code includes GPU acceleration setup for faster Word2Vec training (for Google Colab or Kaggle notebooks)

## Results Summary

- **Dataset**: 193 healthcare M&A transactions (2021-2025)
- **Geographic scope**: US (72.2%), Europe (14.4%), India (5.6%), Japan (3.7%)
- **Transaction types**: Acquisitions (48.7%), Mergers (10.9%), Asset acquisitions (8.3%)
- **Vocabulary**: 9,352 M&A-specific terms after filtering

## Academic Context

This work demonstrates how domain-specific NLP techniques can capture nuanced business communication patterns beyond traditional sentiment analysis, providing a replicable framework for business communication research.

---

**Author**: Maksim Kocheshkov  
**Institution**: University of Milan - Master's in Data Science  
**Course**: Natural Language Processing
