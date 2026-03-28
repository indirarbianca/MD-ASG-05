# ASG 05 MD - Proper Pipeline for the Spaceship Titanic + Deploy

Predict which passengers were transported to an alternate dimension using Logistic Regression.

## Project Structure

```
MD-ASG-05/
├── pyproject.toml
├── requirements.txt
├── README.md
├── app.py
├── models/
│   ├── __init__.py
│   └── pipeline.pkl        
└── source/
    ├── __init__.py
    ├── config.py
    ├── ingest.py
    ├── preprocessing.py
    ├── training.py
    ├── evaluation.py
    └── pipeline.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

Place `train.csv` and `test.csv` inside a `data/` folder, then run:

```bash
python -m source.pipeline --mode train
```

This outputs a single `models/pipeline.pkl` containing the full sklearn Pipeline (preprocessor + model).

## Prediction

```bash
python -m source.pipeline --mode predict
```

## Streamlit App

```bash
streamlit run app.py
```
