import yaml
import importlib
from pathlib import Path
from typing import Dict, List
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class ModelLoader:
    @staticmethod
    def load_sklearn_model(config: dict):
        import pickle
        with open(Path(config['path']) / 'model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(Path(config['path']) / 'vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer

    @staticmethod
    def load_keras_model(config: dict):

        model = load_model(Path(config['path']) / 'model.h5')
        with open(Path(config['path']) / 'tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer

class ModelManager:
    def __init__(self, config_path: str = 'configs/models_config.yaml'):
        self.config = self._load_config(config_path)
        self.models = self._load_models()
        
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_models(self) -> Dict[str, dict]:
        models = {}
        
        # Загрузка core моделей
        for model_cfg in self.config['models']['core']:
            if model_cfg['enabled']:
                loader = getattr(ModelLoader, f"load_{model_cfg['type']}_model")
                model, preprocessor = loader(model_cfg)
                models[model_cfg['name']] = {
                    'model': model,
                    'preprocessor': preprocessor,
                    'weight': model_cfg['weight'],
                    'type': model_cfg['type']
                }
        
        # Загрузка плагинов
        for plugin_cfg in self.config['models']['plugins']:
            if plugin_cfg['enabled']:
                try:
                    loader = getattr(ModelLoader, f"load_{plugin_cfg['type']}_model")
                    model, preprocessor = loader(plugin_cfg)
                    models[plugin_cfg['name']] = {
                        'model': model,
                        'preprocessor': preprocessor,
                        'weight': plugin_cfg['weight'],
                        'type': plugin_cfg['type']
                    }
                except Exception as e:
                    print(f"Failed to load plugin {plugin_cfg['name']}: {str(e)}")
        
        return models
    
    def predict(self, text: str) -> dict:
        results = {}
        
        for name, model_info in self.models.items():
            try:
                if model_info['type'] == 'sklearn':
                    vectorized = model_info['preprocessor'].transform([text])
                    proba = model_info['model'].predict_proba(vectorized)[0][1]
                elif model_info['type'] == 'keras':
                    seq = model_info['preprocessor'].texts_to_sequences([text])
                    padded = pad_sequences(seq, maxlen=100)
                    proba = model_info['model'].predict(padded)[0][0]
                
                results[name] = {
                    'probability': float(proba),
                    'weight': model_info['weight']
                }
            except Exception as e:
                print(f"Prediction failed for {name}: {str(e)}")
        
        return results
    
    def get_combined_prediction(self, text: str) -> float:
        predictions = self.predict(text)
        total_weight = sum(m['weight'] for m in predictions.values())
        return sum(m['probability'] * m['weight'] for m in predictions.values()) / total_weight