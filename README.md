Phishing Detection Classifier

Overview
This project implements a baseline phishing detector for URLs or email text using a simple character n-gram TF-IDF model with Logistic Regression. It includes data loading, preprocessing, training, evaluation, and an EDA notebook.

Project Structure
- data/
- raw/ (place your dataset here)
- processed/
- notebooks/
- 01_eda.ipynb
- src/
- data.py
- train.py
- eval.py
- utils.py
- results/

Dataset Expectations
Provide a CSV file in data/raw (or any path you pass on the CLI). The dataset should contain:
- A text column (URL or email text)
- A label column (0/1 or strings like "phishing"/"benign")

If you don't pass column names, the scripts try to infer them from common names:
Text: url, email, text, body, content, message, subject
Label: label, is_phishing, phishing, target, class, y

Setup
1. Create and activate a virtual environment
2. Install dependencies:
   pip install -r requirements.txt

Train
python -m src.train --data-path data/raw/your_file.csv --model-out results/model.joblib

Evaluate
python -m src.eval --data-path data/raw/your_file.csv --model-path results/model.joblib

Outputs
- results/metrics.json
- results/confusion_matrix.png
