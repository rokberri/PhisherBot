import pickle
import os

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


