import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tools.text_tools import TextPreprocessor
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

DATA_PATH = 'DATASETS/Phishing_Email.csv'

# --- 1. Загрузка моделей и вспомогательных объектов ---
# LSTM модель + токенизатор
# Загрузка модели LSTM
with open('saved_models_LSTM/lstm_best.pkl', 'rb') as f:
    lstm_data = pickle.load(f)
    lstm_model = lstm_data['model']  # Извлекаем модель Keras
with open("saved_models_LSTM/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Logistic Regression + TF-IDF векторизатор
with open("saved_models_LR/lr_C_1_max_iter_500_penalty_l2_solver_liblinear.pkl", "rb") as f:
    logistic_model = pickle.load(f)
with open("saved_models_LR/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Decision Tree
with open("saved_models_DT/dt_class_weight_None_criterion_gini_max_depth_10_min_samples_leaf_2_min_samples_split_2.pkl", "rb") as f:
    tree_model = pickle.load(f)

# LabelEncoder
with open("saved_models_DT/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# --- 2. Загрузка тестовых данных ---
test_data = pd.read_csv(DATA_PATH)
X_test = test_data["Email Text"]
y_test = label_encoder.transform(test_data["Email Type"])

# --- 3. Предобработка текста ---
preprocessor = TextPreprocessor()
X_test_cleaned = X_test.apply(preprocessor.clean_text)

# --- 4. Подготовка данных для каждой модели ---
# Для LSTM
X_test_seq = tokenizer.texts_to_sequences(X_test_cleaned)
X_test_pad = pad_sequences(X_test_seq, maxlen=200, padding="post")

# Для Logistic Regression и Decision Tree
X_test_tfidf = tfidf.transform(X_test_cleaned)

# --- 5. Предсказания ---
y_pred_lstm = (lstm_model.predict(X_test_pad) > 0.5).astype(int).flatten()
y_pred_logistic = logistic_model.predict(X_test_tfidf)
y_pred_tree = tree_model.predict(X_test_tfidf)

# --- 6. Вычисление метрик ---
models = {
    "LSTM": y_pred_lstm,
    "Logistic Regression": y_pred_logistic,
    "Decision Tree": y_pred_tree
}

metrics = {}
for name, y_pred in models.items():
    metrics[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

# --- 7. Визуализация ---
# Таблица метрик
metrics_df = pd.DataFrame(metrics).T
print(metrics_df)

# Bar-plot метрик
plt.figure(figsize=(10, 6))
metrics_df.plot(kind="bar", rot=0, colormap="viridis", alpha=0.8)
plt.title("Сравнение моделей")
plt.ylabel("Значение метрики")
plt.legend(loc="lower right")
plt.savefig("metrics_comparison.png")
plt.close()

# Confusion Matrix для LSTM
cm = confusion_matrix(y_test, y_pred_lstm)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix (LSTM)")
plt.savefig("confusion_matrix_lstm.png")
plt.close()