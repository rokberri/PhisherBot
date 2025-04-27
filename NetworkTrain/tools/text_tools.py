import nltk
import emoji
import string
import re
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

#-----------------TEXT CLEAN UP-----------------#

# Function to remove HTML tags from text
def remove_html_tags(text):
     # Проверка, что текст не похож на путь к файлу или URL
    if re.match(r'^(\/|http|www)', text.strip()):
        return text  # Возвращаем исходный текст без обработки
    try:
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    except Exception:
        return text

# Define a function to remove URLs using regular expressions
def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

# Define the punctuation characters to remove
punctuation = string.punctuation

# Function to remove punctuation from text
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punctuation))


def remove_special_characters(text):
    # Define the pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'  # Matches any character that is not alphanumeric or whitespace
    
    # Replace special characters with an empty string
    clean_text = re.sub(pattern, '', text)
    
    return clean_text


# Function to remove numeric values from text
def remove_numeric(text):
    return re.sub(r'\d+', '', text)

# Define a function to remove non-alphanumeric characters
def remove_non_alphanumeric(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)



# Download NLTK stopwords corpus
nltk.download('stopwords')
nltk.download('punkt_tab')

# Get English stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Function to remove stop words from text
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function to remove emojis from text
def remove_emojis(text):
    return emoji.demojize(text)


def preprocess_text_data(df, text_column='Email Text', stem_column='Message_stemmed'):
    """
    Функция для предобработки текстовых данных:
    1. Очистка от знаков препинания, HTML тегов, URL, эмодзи, стоп-слов и других лишних символов.
    2. Токенизация и стемминг.
    3. Преобразование текста в числовые признаки с использованием CountVectorizer.

    :param df: pd.DataFrame - исходный DataFrame с текстовыми данными и метками.
    :param text_column: str - имя столбца с текстовыми данными, которые нужно обработать.
    :param stem_column: str - имя нового столбца, куда будет добавлен стеммированный текст.
    :return: tuple (X, y, cv) - возвращает преобразованные данные X (признаки), y (целевая переменная), cv (CountVectorizer).
    """
    
    # Преобразование текста в нижний регистр
    df[text_column] = df[text_column].str.lower()

    # Заполнение NaN значений пустыми строками
    df[text_column] = df[text_column].fillna('')

    # Удаление лишних пробелов
    df[text_column] = df[text_column].str.strip()
    
    # Применение функций очистки
    df[text_column] = df[text_column].apply(remove_html_tags)
    df[text_column] = df[text_column].apply(remove_urls)
    df[text_column] = df[text_column].apply(remove_punctuation)
    df[text_column] = df[text_column].apply(remove_special_characters)
    df[text_column] = df[text_column].apply(remove_numeric)
    df[text_column] = df[text_column].apply(remove_non_alphanumeric)
    df[text_column] = df[text_column].apply(remove_stopwords)
    df[text_column] = df[text_column].apply(remove_emojis)
    
    # Применение стемминга
    df = add_stemmed_column(df, column_name=text_column, new_column_name=stem_column)
    
    # Преобразование текста в числовые признаки с использованием CountVectorizer
    cv = CountVectorizer() # преобразует текст в числа с использованием BoW
    X = cv.fit_transform(df[stem_column])
    
    # Целевая переменная
    y = df['Email Type']
    ENCODER_PATH = 'label_encoder.pkl'
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
        
    y_encoded = label_encoder.fit_transform(y)
    
    # Возвращаем признаки X, целевую переменную y и обученный CountVectorizer
    return X, y_encoded, cv

# # Пример использования функции
# df = pd.read_csv('ML_flow/DATASETS/Phishing_Email.csv')

# # Обработать текстовые данные
# X, y, cv = preprocess_text_data(df)

# # Пример разделения на тренировочные и тестовые данные
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Теперь X_train и X_test готовы к обучению модели


#-----------------------------------------------#




# Initialize the Porter Stemmer
porter_stemmer = PorterStemmer()

# Apply stemming
def add_stemmed_column(df, column_name='Email Text', new_column_name='Message_stemmed'):    
    df[new_column_name] = df[column_name].apply(lambda x: ' '.join(
        [porter_stemmer.stem(word) for word in x.split()]))
    
    return df


def clean_up(message:string):
    clean_message = message.lower()
    clean_message = clean_message.strip()
    clean_message = remove_html_tags(clean_message)
    clean_message = remove_urls(clean_message)
    clean_message = remove_punctuation(clean_message)
    clean_message = remove_special_characters(clean_message)
    clean_message = remove_numeric(clean_message)
    clean_message = remove_non_alphanumeric(clean_message)
    clean_message = remove_stopwords(clean_message)
    clean_message = remove_emojis(clean_message)
    clean_message = remove_stopwords(clean_message)
    return clean_message






import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Загрузка ресурсов NLTK (выполнить один раз)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """Базовая очистка текста."""
    if not isinstance(text, str):
        return ""
    
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление спецсимволов и цифр
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Удаление пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_and_lemmatize(text):
    """Токенизация и лемматизация."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)

def remove_stopwords(text):
    """Удаление стоп-слов."""
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

def preprocess_text(text):
    """Комплексная предобработка текста."""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = tokenize_and_lemmatize(text)
    return text

def preprocess_for_lstm(texts, max_len=100):
    """Подготовка текста для LSTM."""
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded, tokenizer

def preprocess_for_tfidf(texts):
    """Подготовка текста для моделей на основе TF-IDF."""
    vectorizer = TfidfVectorizer(
    max_features=10000,  # Должно быть одинаковое значение!
    lowercase=True,
    stop_words='english',
    analyzer='word'
    )

    # Сохраняем векторайзер ОДИН раз для всех моделей
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def preprocess_text_data(df, model_type='lstm'):
    """Универсальный обработчик для датафрейма."""
    texts = df['Email Text'].apply(preprocess_text)
    labels = df['Email Type']
    
    if model_type == 'lstm':
        X, processor = preprocess_for_lstm(texts)
    elif model_type in ['dt', 'lr']:
        X, processor = preprocess_for_tfidf(texts)
    else:
        raise ValueError("Unknown model type")
    
    return X, labels, processor