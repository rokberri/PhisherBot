# utils.py
import os
import pickle
import numpy as np
from pathlib import Path
from text_processor import TextPreprocessor

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

def preprocess_text(text):
    """
    Полная предобработка текста с использованием сохраненных артефактов
    """
    # Загрузка векторайзера
    vectorizer_path = "encoders/tfidf_vectorizer.pkl"
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Очистка текста
    preprocessor = TextPreprocessor()
    cleaned_text = preprocessor.clean_text(text)
    
    # Векторизация
    return vectorizer.transform([cleaned_text])

def decode_prediction(pred):
    """Декодирует числовое предсказание в текстовую метку"""
    encoder_path = "encoders/label_encoder.pkl"
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found at {encoder_path}")
    
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    
    pred_2d = np.array([pred]).reshape(1, -1)
    return encoder.inverse_transform(pred_2d)[0]