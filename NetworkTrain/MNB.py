import pickle
import os
import pandas as pd
from tools.ml_tools import save_model, LabelTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from tools.text_tools import TextPreprocessor

DATA_PATH = 'DATASETS/Phishing_Email.csv'
MODEL_DIR = 'saved_models_MNB'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    save_model(model, os.path.join(MODEL_DIR, 'mnb.pkl'))

if __name__ == "__main__":
    train_and_save_models()