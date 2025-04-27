import sys
from pathlib import Path
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tools.text_tools import TextPreprocessor
from tools.ml_tools import save_model
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

DATA_PATH = 'DATASETS/Phishing_Email.csv'
MODEL_DIR = 'saved_models_LSTM'
os.makedirs(MODEL_DIR, exist_ok=True)

# Параметры для LSTM
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 64
LSTM_UNITS = 64

def train_and_save_model():
    # Загрузка и предобработка данных
    df = pd.read_csv(DATA_PATH)
    preprocessor = TextPreprocessor()
    
    # Очистка текста
    texts = df['Email Text'].apply(lambda x: preprocessor.clean_text(x))
    labels = df['Email Type']
    
    # Токенизация текста
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Создание модели LSTM
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Обучение модели
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64
    )
    
    # Сохранение артефактов
    with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Сохранение модели
    save_model(model, os.path.join(MODEL_DIR, f"lstm.pkl"))
    
    print("Обучение завершено. Модель и артефакты сохранены в", MODEL_DIR)

if __name__ == "__main__":
    train_and_save_model()