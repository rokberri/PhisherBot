import pickle
import os
from sklearn.preprocessing import LabelEncoder


#-----------------ML TOOLS-----------------#
def save_model(model, filename):
    """
    Функция для сохранения обученной модели в файл.
    
    :param model: Обученная модель.
    :param filename: Имя файла для сохранения модели.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Модель сохранена в файл {filename}")

def save_history(history):
    # Сохранение истории
    with open("training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

def load_history(path="training_history.pkl"):
    # Загрузка истории
    with open(path, "rb") as f:
        return pickle.load(f)
    
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

#-------------------------------------------#


class LabelTransformer:
    def __init__(self, column_name="Email Type"):
        self.column_name = column_name
        self.label_encoder = LabelEncoder()

    def fit_transform(self, df):
        return self.label_encoder.fit_transform(df[self.column_name])

    def transform(self, labels):
        return self.label_encoder.transform(labels)

    def inverse_transform(self, labels):
        return self.label_encoder.inverse_transform(labels)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoder, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.label_encoder = pickle.load(f)