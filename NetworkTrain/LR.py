import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tools.text_tools import TextPreprocessor
from tools.ml_tools import save_model, LabelTransformer
import pickle


DATA_PATH = 'DATASETS/Phishing_Email.csv'
MODEL_DIR = 'saved_models_LR'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2'],
    'max_iter': [100, 500]
}

def train_and_save_models():
    # Загрузка данных
    df = pd.read_csv(DATA_PATH)
    
    # Инициализация TextPreprocessor
    preprocessor = TextPreprocessor()

    # Инициализация LabelEncoder
    encoder = LabelTransformer()
    
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
    
    # Преобразование меток
    y = encoder.fit_transform(df)
    
    # Сохранение артефактов
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    encoder.save(ENCODER_PATH)
    
    # Обучение модели
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
     # Получаем лучшую модель
    best_model = grid_search.best_estimator_

    # Получаем гиперпараметры лучшей модели
    best_params_str = "_".join([f"{k}_{str(v).replace(' ', '')}" for k, v in grid_search.best_params_.items()])
    print(best_params_str)

    # Сохранение лучшей модели 
    save_model(best_model, os.path.join(MODEL_DIR, 'lr.pkl'))

if __name__ == "__main__":
    train_and_save_models()