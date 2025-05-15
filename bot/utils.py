# utils.py
import os
import pickle
import numpy as np
from pathlib import Path
from text_processor import TextPreprocessor
import sqlite3


# ----------------------- DB UTILS ---------------------------------
def create_connection(db_file='data/bot_database.db'):
    """Создает подключение к указанной базе данных SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Exception as e:
        print(f"Ошибка при соединении с базой данных: {e}")
    return conn

def add_user(conn, telegram_id, username=None):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO users (telegram_id, username) VALUES (?, ?)",
        (telegram_id, username)
    )
    conn.commit()

def save_message(conn, sender_id, content, prediction):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (sender_id, content, prediction) VALUES (?, ?, ?)",
        (sender_id, content, prediction)
    )
    conn.commit()
    return cur.lastrowid

def report_message(conn, message_id, reporter_id, reason="User disagreed"):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO reported_messages (message_id, reported_by, report_reason) VALUES (?, ?, ?)",
        (message_id, reporter_id, reason)
    )
    conn.commit()

def make_admin(conn, telegram_id):
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET admin=TRUE WHERE telegram_id=?",
        (telegram_id,)
    )
    conn.commit()

def fetch_admins(conn):
    cur = conn.cursor()
    cur.execute("SELECT telegram_id FROM users WHERE admin=TRUE")
    rows = cur.fetchall()
    return [row[0] for row in rows]
# -----------------------------------------------------------------------------
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