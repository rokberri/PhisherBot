import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Раскоментировать при первом прогоне
# nltk.download('stopwords')
# nltk.download('punkt')

def preprocess_text(text):
    """Очистка текста: удаление спецсимволов, стоп-слов, приведение к нижнему регистру."""
    if not isinstance(text, str):  # Проверяем, является ли текст строкой
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Удаляем пунктуацию
    words = word_tokenize(text)  # Разбиваем на слова
    words = [word for word in words if word not in stopwords.words('english')]  # Удаляем стоп-слова
    return ' '.join(words)

def extract_phishing_patterns(file_path, top_n=20):
    """Извлекает топ-N фишинговых паттернов из CSV-файла."""
    # Загрузка данных
    data = pd.read_csv(file_path)
    
    # Предобработка текста
    data['processed_text'] = data['Email Text'].apply(preprocess_text)
    
    # Разделение на фишинг и безопасные письма
    phishing_emails = data[data['Email Type'] == 'Phishing']['processed_text']
    # safe_emails = data[data['Email Type'] == 'Safe']['processed_text']
    
    # TF-IDF для фишинговых писем (учитываем словосочетания до 3 слов)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(phishing_emails)
    
    # Получаем термины и их веса
    terms = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1  # Суммируем веса по всем документам
    
    # Сортируем термины по убыванию важности
    term_scores = list(zip(terms, tfidf_scores))
    term_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Возвращаем топ-N терминов
    top_terms = [term for term, score in term_scores[:top_n]]
    return top_terms

# Пример использования
file_path = "../NetworkTrain/DATASETS/Phishing_Email.csv" 
phishing_patterns = extract_phishing_patterns(file_path)
print("Top phishing patterns:")
for pattern in phishing_patterns:
    print(f"- {pattern}")