import yaml
import pickle
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config_path='configs/models_config.yaml'):
        self.models = {}
        self.load_models(config_path)
    
    def load_model_artifacts(self, model_path, model_type):
        """Загрузка артефактов модели с проверкой типа"""
        try:
            model_path = Path(model_path)
            
            if model_type == 'lstm':
                # Для LSTM моделей
                with open(model_path/'tokenizer.pkl', 'rb') as f:
                    tokenizer = pickle.load(f)
                
                model = load_model(model_path/'model.h5')
                return {
                    'type': 'lstm',
                    'model': model,
                    'preprocessor': tokenizer,
                    'predict_func': self._predict_lstm
                }
            else:
                # Для sklearn моделей
                with open(model_path/'vectorizer.pkl', 'rb') as f:
                    vectorizer = pickle.load(f)
                
                with open(model_path/'model.pkl', 'rb') as f:
                    model = pickle.load(f)
                    if isinstance(model, dict):
                        model = model['model']  # Извлекаем модель если сохранена в словаре
                
                return {
                    'type': 'sklearn',
                    'model': model,
                    'preprocessor': vectorizer,
                    'predict_func': self._predict_sklearn
                }
                
        except Exception as e:
            logger.error(f"Load failed: {str(e)}")
            return None

    def _predict_lstm(self, model, X):
        """Специальный метод предсказания для LSTM моделей"""
        # Для моделей с сигмоидой на выходе predict возвращает вероятность напрямую
        prediction = model.predict(X, verbose=0)

        # Если модель возвращает несколько значений (например, для мультикласса)
        if prediction.shape[1] > 1:
            return prediction[0][1]  # Вероятность класса 1 (фишинг)
        return float(prediction[0][0])  # Для бинарной классификации

    def _predict_sklearn(self, model, X):
        """Специфичный метод предсказания для sklearn"""
        return model.predict_proba(X)[0][1]  # Вероятность класса 1

    def load_models(self, config_path):
        """Загрузка всех моделей из конфига"""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            for model_cfg in config['models']['core'] + config['models']['plugins']:
                if not model_cfg['enabled']:
                    continue
                
                if model_data := self.load_model_artifacts(model_cfg['path'], model_cfg['type']):
                    model_data['weight'] = model_cfg['weight']
                    self.models[model_cfg['name']] = model_data
                    logger.info(f"Loaded {model_cfg['name']} ({model_cfg['type']})")
                else:
                    logger.warning(f"Failed to load {model_cfg['name']}")
                    
        except Exception as e:
            logger.error(f"Config error: {str(e)}")
            raise

    def preprocess(self, text, model_info):
        """Препроцессинг текста для конкретной модели"""
        try:
            if model_info['type'] == 'lstm':
                seq = model_info['preprocessor'].texts_to_sequences([text])
                return pad_sequences(seq, maxlen=100)
            return model_info['preprocessor'].transform([text])
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def predict_single(self, model_name, text):
        """Универсальный метод предсказания для всех моделей"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None
            
        model_info = self.models[model_name]
        
        try:
            # Препроцессинг текста
            if model_info['type'] == 'lstm':
                seq = model_info['preprocessor'].texts_to_sequences([text])
                X = pad_sequences(seq, maxlen=100)
            else:
                X = model_info['preprocessor'].transform([text])
            
            # Предсказание
            if model_info['type'] == 'lstm':
                return self._predict_lstm(model_info['model'], X)
            else:
                # Для sklearn моделей
                if hasattr(model_info['model'], 'predict_proba'):
                    return model_info['model'].predict_proba(X)[0][1]
                return float(model_info['model'].predict(X)[0])
                
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {str(e)}")
            return None

    def predict_all(self, text):
        """Предсказания всех моделей"""
        results = {}
        for name, data in self.models.items():
            prob = self.predict_single(name, text)
            if prob is not None:
                results[name] = {
                    'probability': prob,
                    'weight': data['weight']
                }
        return results

    def get_combined_prediction(self, text):
        """Комбинированное предсказание"""
        predictions = self.predict_all(text)
        if not predictions:
            return 0.5  # Нейтральное значение при ошибках
            
        total_weight = sum(m['weight'] for m in predictions.values())
        return sum(m['probability'] * m['weight'] for m in predictions.values()) / total_weight