import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tools.text_tools import preprocess_text_data
from tools.ml_tools import save_model


DATA_PATH = '../DATASETS/Phishing_Email.csv'
MODEL_DIR = 'saved_models_DT'
ENCODER_PATH = 'encoder_DT.pkl'

os.makedirs(MODEL_DIR, exist_ok=True)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

def train_and_save_models():
    df = pd.read_csv(DATA_PATH)
    X, y, _ = preprocess_text_data(df,'dt')
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Сохранение LabelEncoder
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Обучение и сохранение моделей
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1
    )
    
    grid_search.fit(X, y_encoded)
    
    for params in grid_search.cv_results_['params']:
        model = DecisionTreeClassifier(**params, random_state=42)
        model.fit(X, y_encoded)
        params_str = "_".join([f"{k}_{str(v).replace(' ', '')}" for k, v in params.items()])
        save_model(model, os.path.join(MODEL_DIR, f"dt_{params_str}.pkl"))

if __name__ == "__main__":
    train_and_save_models()


    