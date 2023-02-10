# SMARTEDU-AQG

Automatic Question Generation of Multiple Choice Questions

## Installing dependencies

To install dependencies run
```bash
python -m pip install -r requirements.txt
```

It is also needed to install the following packages separately
```bash
python -m pip install git+https://github.com/mmxgn/spacy-clausie.git

python -m spacy download en_core_web_sm

python -m spacy download en_core_web_lg

pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl
```

Note: This project was originally developed using Python 3.8.