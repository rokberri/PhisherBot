import nltk
import string
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# common/text_preprocessor.py
import re
import string
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def remove_html_tags(self, text):
        if re.match(r'^(\/|http|www)', text.strip()):
            return text
        try:
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text()
        except Exception:
            return text
    
    def remove_urls(self, text):
        return re.sub(r'http\S+|www\S+', '', text)
    
    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_special_chars(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def remove_numeric(self, text):
        return re.sub(r'\d+', '', text)
    
    def remove_stopwords(self, text):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def stem_text(self, text):
        return ' '.join([self.stemmer.stem(word) for word in text.split()])
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        text = text.lower().strip()
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_punctuation(text)
        text = self.remove_special_chars(text)
        text = self.remove_numeric(text)
        text = self.remove_stopwords(text)
        text = self.stem_text(text)
        
        return text