import sys
from pathlib import Path
import pandas as pd
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tools.text_tools import TextPreprocessor
from tools.ml_tools import save_model


# Добавляем путь к корню проекта
sys.path.append(str(Path(__file__).parent.parent))

DATA_PATH = 'DATASETS/Phishing_Email.csv'
MODEL_DIR = 'saved_models_DT'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

def train_and_save_models():
    # Загрузка данных
    df = pd.read_csv(DATA_PATH)
    
    # Инициализация и использование TextPreprocessor
    preprocessor = TextPreprocessor()
    
    # Очистка текста
    texts = df['Email Text'].apply(lambda x: preprocessor.clean_text(x))
    
    # Векторизация текста
    vectorizer = TfidfVectorizer(
        max_features=10000,
        lowercase=True,
        stop_words='english',
        analyzer='word'
    )
    X = vectorizer.fit_transform(texts)
    
    # Кодирование меток
    y = df['Email Type']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Сохранение артефактов
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Обучение модели
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1
    )
    
    grid_search.fit(X, y_encoded)
    
    # Сохранение моделей
    for params in grid_search.cv_results_['params']:
        model = DecisionTreeClassifier(**params, random_state=42)
        model.fit(X, y_encoded)
        params_str = "_".join([f"{k}_{str(v).replace(' ', '')}" for k, v in params.items()])
        save_model(model, os.path.join(MODEL_DIR, f"dt_{params_str}.pkl"))

if __name__ == "__main__":
    train_and_save_models()