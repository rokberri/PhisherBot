import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay, classification_report
)
import argparse
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tools.ml_tools import LabelTransformer
from tools.ml_tools import load_model




def evaluate_model(model, vectorizer, X_texts, y_true, title_prefix="Model", save_plots=False):
    # X_vec = vectorizer.transform(X_texts)
    
    # Проверяем тип векторизатора
    if isinstance(vectorizer, Tokenizer):
        # Для Keras Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        MAX_SEQUENCE_LENGTH = 100  # Настройте длину последовательности
        X_seq = vectorizer.texts_to_sequences(X_texts)
        X_vec = pad_sequences(X_seq, maxlen=MAX_SEQUENCE_LENGTH)
    elif hasattr(vectorizer, 'transform'):
        X_vec = vectorizer.transform(X_texts)
    else:
        raise ValueError("Unsupported vectorizer type!")

     # Получаем предсказания
    if hasattr(model, 'predict_proba'):
        y_pred_probs = model.predict_proba(X_vec)[:,1]
    else:
        y_pred_probs = model.predict(X_vec)
    
    y_proba = y_pred_probs
    y_pred = (y_pred_probs > 0.5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='binary'),
        "Recall": recall_score(y_true, y_pred, average='binary'),
        "F1-score": f1_score(y_true, y_pred, average='binary'),
    }

    if y_proba is not None:
        metrics["ROC AUC"] = roc_auc_score(y_true, y_proba)

    print(f"\n=== {title_prefix} Evaluation Report ===")
    print(classification_report(y_true, y_pred))
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # ROC кривая
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'ROC AUC = {metrics["ROC AUC"]:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title_prefix} - ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{title_prefix}_roc_curve.png')
        plt.show()

    # Бар-график метрик
    plt.figure(figsize=(6, 4))
    plt.bar(list(metrics.keys()), list(metrics.values()))
    plt.title(f'{title_prefix} - Evaluation Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{title_prefix}_metrics.png')
    plt.show()

    # Матрица ошибок
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.ax_.set_title(f'{title_prefix} - Confusion Matrix')
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{title_prefix}_confusion_matrix.png')
    plt.show()


def main(model_path, vectorizer_path, data_path, encoder_path=None, save_plots=False):
    model = load_model(model_path)
    vectorizer = load_model(vectorizer_path)
    df = pd.read_csv(data_path)

    text_column = "Email Text"
    label_column = "Email Type"

    X = df[text_column].astype(str).values

    if encoder_path:
        encoder = LabelTransformer()
        encoder.load(encoder_path)
        y = encoder.transform(df[label_column])

    
    model_name = os.path.basename(model_path).split('.')[0]
    evaluate_model(model, vectorizer, X, y, title_prefix=model_name, save_plots=save_plots)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model with test data.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model (.pkl)")
    parser.add_argument('--vectorizer', type=str, required=True, help="Path to the vectorizer (.pkl)")
    parser.add_argument('--data', type=str, required=True, help="Path to the CSV data file")
    parser.add_argument('--label_encoder', type=str, help="Path to the label encoder (.pkl)")
    parser.add_argument('--save_plots', action='store_true', help="Whether to save ROC and confusion matrix plots as PNG")

    args = parser.parse_args()
    main(
        model_path=args.model,
        vectorizer_path=args.vectorizer,
        data_path=args.data,
        encoder_path=args.label_encoder,
        save_plots=args.save_plots
    )
