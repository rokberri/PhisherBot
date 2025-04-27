#-----------------ML METRICS-----------------#
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

def calculate_accuracy(y_true, y_pred):
    """
    Вычисляет точность (Accuracy).

    Параметры:
    - y_true: Истинные метки (список или массив).
    - y_pred: Предсказанные метки (список или массив).

    Возвращает:
    - Точность (Accuracy) как float.
    """
    return accuracy_score(y_true, y_pred)

def calculate_recall(y_true, y_pred):
    """
    Вычисляет полноту (Recall).

    Параметры:
    - y_true: Истинные метки (список или массив).
    - y_pred: Предсказанные метки (список или массив).

    Возвращает:
    - Полноту (Recall) как float.
    """
    return recall_score(y_true, y_pred)

def calculate_f1(y_true, y_pred):
    """
    Вычисляет F1-меру (F1-score).

    Параметры:
    - y_true: Истинные метки (список или массив).
    - y_pred: Предсказанные метки (список или массив).

    Возвращает:
    - F1-меру (F1-score) как float.
    """
    return f1_score(y_true, y_pred)

def calculate_roc_auc(y_true, y_pred_proba):
    """
    Вычисляет ROC-AUC.

    Параметры:
    - y_true: Истинные метки (список или массив).
    - y_pred_proba: Предсказанные вероятности для положительного класса (список или массив).

    Возвращает:
    - ROC-AUC как float.
    """
    return roc_auc_score(y_true, y_pred_proba)

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Оценивает модель с использованием метрик: точность, полнота, F1-мера и ROC-AUC.

    Параметры:
    - y_true: Истинные метки (список или массив).
    - y_pred: Предсказанные метки (список или массив).
    - y_pred_proba: Предсказанные вероятности для ROC-AUC (опционально, если модель возвращает вероятности).

    Возвращает:
    - Словарь с метриками: accuracy, recall, f1, roc_auc.
    """
    metrics = {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "recall": calculate_recall(y_true, y_pred),
        "f1": calculate_f1(y_true, y_pred),
        "roc_auc": calculate_roc_auc(y_true, y_pred_proba) if y_pred_proba is not None else None,
    }
    return metrics

def print_metrics(metrics):
    """
    Выводит метрики в читаемом формате.

    Параметры:
    - metrics: Словарь с метриками, возвращенный функцией evaluate_model.
    """
    print("Метрики модели:")
    print(f"Точность (Accuracy): {metrics['accuracy']:.4f}")
    print(f"Полнота (Recall): {metrics['recall']:.4f}")
    print(f"F1-мера (F1-score): {metrics['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    else:
        print("ROC-AUC: Недостаточно данных для расчета (требуются вероятности).")
#-----------------------------------------------#