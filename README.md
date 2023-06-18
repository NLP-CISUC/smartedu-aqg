# SMARTEDU-AQG

Automatic Question Generation of Multiple Choice Questions

## Update (June 2023)

Code for generating Portuguese distractors of different types, from different resources, for two different datasets, and raking them with language models:
[https://github.com/NLP-CISUC/smartedu-aqg/blob/main/Generating_Ranking_Distractors_PT.ipynb]

Described in the paper:
Hugo Gonçalo Oliveira, Igor Caetano, Renato Matos, Hugo Amaro. _Generating and Ranking Distractors for Multiple Choice Questions in Portuguese_.
Proceedings of SLATE 2023.

## Old:

### Installing dependencies

To install dependencies run
```bash
python -m pip install -r requirements.txt
```

It is also needed to install the following packages separately
```bash
# Claucy
python -m pip install git+https://github.com/mmxgn/spacy-clausie.git

# SpaCy English language models 
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl
```

Note: This project was originally developed using Python 3.8.
