from pathlib import Path
import pandas as pd
import os
import pickle
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import ParameterGrid
from tools.text_tools import TextPreprocessor
from tools.ml_tools import save_model, LabelTransformer


DATA_PATH = 'DATASETS/Phishing_Email.csv'
MODEL_DIR = 'saved_models_LSTM'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

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
    # Загрузка данных
    df = pd.read_csv(DATA_PATH)
    
     # Инициализация и использование TextPreprocessor
    preprocessor = TextPreprocessor()

    # Инициализация LabelEncoder
    encoder = LabelTransformer()

    # Очистка текста
    texts = df['Email Text'].apply(lambda x: preprocessor.clean_text(x))
    
    # Векторизация текста ???
    vectorizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    vectorizer.fit_on_texts(texts)
    sequences = vectorizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Преобразование меток
    y = encoder.fit_transform(df)
    
    # Сохранение артефактов
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    encoder.save(ENCODER_PATH)

    # Гиперпараметры для перебора
    param_grid = {
        'lstm_units': [32, 64],
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
        # print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Если эта модель лучше предыдущей, обновить best_model
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_params = params
            best_model = clone_model(model)
            best_model.set_weights(model.get_weights())  # Копируем веса лучшей модели

    # Сохранение лучшей модели
    if best_model is not None:
        best_model_data = {
            'model': best_model,
            'params': best_params,
            'val_accuracy': best_score,
            'is_best': True
        }
    save_model(best_model, os.path.join(MODEL_DIR, 'lstm.pkl'))
    
    print(f"\nBest params: {best_params}")
    print(f"Best validation accuracy: {best_score:.4f}")


if __name__ == "__main__":
    train_and_save_models()