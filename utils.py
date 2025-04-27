import re
import pickle
import os
import numpy as np

def load_model(path):
    """Безопасная загрузка pickle-модели"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found!")
    
    if os.path.getsize(path) == 0:
        raise ValueError(f"Model file {path} is empty!")
    
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Failed to load model from {path}: {str(e)}")
# Загрузка предобученных компонентов

with open('encoders/tfidf_vectorizer_LR.pkl', 'rb') as f:
    vectorizer_LR = pickle.load(f)
with open('encoders/encoder_LR.pkl', 'rb') as f:
    encoder_LR = pickle.load(f)

def clean_text(text):
    """Базовая очистка текста (одинаковая для всех моделей)."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Удаляем всё, кроме букв и пробелов
    text = re.sub(r'\s+', ' ', text).strip()  # Удаляем лишние пробелы
    return text


def preprocess_text(text):
    """Универсальная очистка текста"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def decode_prediction(pred):
    """Декодирует числовое предсказание в текстовую метку"""
    # Преобразуем предсказание в 2D-массив
    pred_2d = np.array([pred]).reshape(1, -1)  # или np.array([pred]).reshape(1, -1)
    return encoder_LR.inverse_transform(pred_2d)[0][0]

def vectorize_text(text):
    """Векторизация с сохраненным векторайзером"""
    with open('encoders/tfidf_vectorizer_DT.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    cleaned = preprocess_text(text)
    return tfidf_vectorizer.transform([cleaned])