import sys
from pathlib import Path
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import ParameterGrid
from tools.text_tools import TextPreprocessor
from tools.ml_tools import save_model

sys.path.append(str(Path(__file__).parent.parent))

DATA_PATH = 'DATASETS/Phishing_Email.csv'
MODEL_DIR = 'saved_models_LSTM'
os.makedirs(MODEL_DIR, exist_ok=True)

# Базовые параметры
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100

def create_model(lstm_units=64, dropout_rate=0.2, embedding_dim=64):
    """Функция для создания модели Keras"""
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, embedding_dim, input_length=MAX_SEQUENCE_LENGTH),
        LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_save_models():
    # Загрузка и предобработка данных
    df = pd.read_csv(DATA_PATH)
    preprocessor = TextPreprocessor()
    texts = df['Email Text'].apply(lambda x: preprocessor.clean_text(x))
    
    # Токенизация текста
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Кодирование меток
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Email Type'])
    
    # Сохранение артефактов
    with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    # Параметры для перебора
    param_grid = {
        'lstm_units': [32, 64],  # Уменьшено количество вариантов для скорости
        'dropout_rate': [0.2, 0.3],
        'embedding_dim': [32, 64],
        'epochs': [5],
        'batch_size': [32]
    }

    best_score = 0
    best_params = None
    best_model = None

    # Ручной перебор параметров
    for params in ParameterGrid(param_grid):
        print(f"\nTraining with params: {params}")
        
        model = create_model(
            lstm_units=params['lstm_units'],
            dropout_rate=params['dropout_rate'],
            embedding_dim=params['embedding_dim']
        )
        
        history = model.fit(
            X, y,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_split=0.2,
            verbose=1
        )
        
        # Оценка модели
        val_accuracy = max(history.history['val_accuracy'])
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Подготовка данных для сохранения
        model_data = {
            'model': model,
            'params': params,
            'val_accuracy': val_accuracy,
            'history': history.history
        }
        
        # Формирование имени файла
        params_str = "_".join([f"{k}_{str(v).replace(' ', '')}" for k, v in params.items()])
        model_path = os.path.join(MODEL_DIR, f"lstm_{params_str}.pkl")
        
        # Сохранение через save_model
        save_model(model_data, model_path)
        
        # Обновление лучшей модели
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_params = params
            best_model = clone_model(model)
            best_model.set_weights(model.get_weights())

    # Сохранение лучшей модели
    if best_model:
        best_model_data = {
            'model': best_model,
            'params': best_params,
            'val_accuracy': best_score,
            'is_best': True
        }
        save_model(best_model_data, os.path.join(MODEL_DIR, 'lstm_best.pkl'))
    
    print(f"\nBest params: {best_params}")
    print(f"Best validation accuracy: {best_score:.4f}")

if __name__ == "__main__":
    train_and_save_models()