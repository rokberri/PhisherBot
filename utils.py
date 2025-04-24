from bs4 import BeautifulSoup
import re
import string
import nltk
import emoji
import pandas as pd
from nltk.corpus import stopwords



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



nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to remove stop words from text
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function to remove emojis from text
def remove_emojis(text):
    return emoji.demojize(text)



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

