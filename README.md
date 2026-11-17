# Activation injection vs. prompt-based affective priming in Llama-3.1-8B digital twins

Code, generation logs and analysis scripts for the arXiv preprint (link to be added).
The study compares prompt-based affective priming (the J2 sub-study of the Twin-2K-500 Mega Study,
gratitude and lack-of-control domains) with MeanDiff activation injection in Llama-3.1-8B-Instruct.

## Contents
- `notebooks/` Colab notebooks: direction extraction, probe, injection, generation, LLM-judge parsing.
  The lack-of-control notebook file is the closest saved version; the final run used 5000 max new tokens and up to 3 attempts.
- `data/` One row per generation attempt (raw Llama output, judge output, parsed scores), gzip-compressed. No persona text.
- `paired_reanalysis.py` paired t-tests, d_z, Holm correction -> `results/paired_reanalysis_results.csv`
- `build_results.py` builds the LaTeX result tables and `results/within_path_results.csv`
- `make_figures.py` dose-response figures -> `figures/`
- `tfidf_baseline.py` bag-of-words baseline (needs the public `response.csv` from the Mega Study repo)

## Reproduce the statistics
```
pip install pandas numpy scipy scikit-learn matplotlib
python paired_reanalysis.py
python make_figures.py
```

## Scoring
Following the original study, Fatigue items 3, 7, 8 and Desire-for-Predictability items 1, 4, 5 are reverse-coded (8 - x) before averaging.

## Data notes
Persona summaries come from the public Twin-2K-500 dataset and are not redistributed here;
the notebooks fetch them from the public Hugging Face API. The Anthropic API key (LLM judge) is read from Colab secrets and is not included.

## References
Twin-2K-500 Mega Study: https://github.com/TianyiPeng/Twin-2K-500-Mega-Study
PsySET: https://github.com/aminbana/PsySET
