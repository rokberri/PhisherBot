
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tools.ml_tools import save_model
from tools.text_tools import preprocess_text_data


DATA_PATH = '../DATASETS/Phishing_Email.csv'
MODEL_DIR = 'saved_models'
ENCODER_PATH = 'encoder_LR.pkl'

os.makedirs(MODEL_DIR, exist_ok=True)

param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2'],
    'max_iter': [100, 500]
}

def train_and_save_models():
    df = pd.read_csv(DATA_PATH)
    X, y, tfidf_vectorizer = preprocess_text_data(df,'lr')
    

    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Сохранение LabelEncoder
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Обучение и сохранение моделей
    grid_search = GridSearchCV(
        LogisticRegression(),
        param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1
    )
    
    grid_search.fit(X, y_encoded)
    
    for params in grid_search.cv_results_['params']:
        model = LogisticRegression(**params)
        model.fit(X, y_encoded)
        params_str = "_".join([f"{k}_{v}" for k, v in params.items()])
        save_model(model, os.path.join(MODEL_DIR, f"lr_{params_str}.pkl"))

if __name__ == "__main__":
    train_and_save_models()