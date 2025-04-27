from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, filters, CallbackContext, MessageHandler
from User import User 
from config import TOKEN
import pickle
from utils import load_model
from text_processor import TextPreprocessor
# Загрузка моделей
lr_model = load_model('models/lr_C_1.pkl')
dt_model = load_model('models/dt.pkl')
lstm_model = load_model('models/lstm.pkl')

secure_users = []

USER_CLASSES = {
    0 : 'ORD',
    1 : 'SPEC'
}

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hello {update.effective_user.first_name}')

async def role(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = User(update.effective_user.id, update.effective_user.first_name, USER_CLASSES[0])
    user.print_info()
    await update.message.reply_text(f'Your role is: {user.user_role}')


async def set_role(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = User(update.effective_user.id, update.effective_user.first_name, USER_CLASSES[1])
    user.user_type = USER_CLASSES[1]
    secure_users.append(user)
    user.print_info()
    return None

from config import SUSPICIOUS_PATTERNS

async def check_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Загрузка артефактов
    with open('encoders/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('encoders/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Инициализация препроцессора
    preprocessor = TextPreprocessor()
    
    # Очистка и предобработка текста
    raw_text = update.message.text
    cleaned_text = preprocessor.clean_text(raw_text)
    
    # Векторизация текста
    X = vectorizer.transform([cleaned_text])
    
    # Получаем вероятность вместо класса для LR
    lr_prob = lr_model.predict_proba(X)[0][1]  # Вероятность класса 1 (фишинг)

    # Получаем вероятность вместо класса для DT
    dt_prob = dt_model.predict_proba(X)[0][1]  # Вероятность класса 1 (фишинг)
    
    # Получаем вероятность вместо класса для DT
    lstm_prob = lstm_model.predict(X)[0][0]  # Вероятность класса 1 (фишинг)

    # Комбинированная оценка (можно настроить веса)
    combined_prob = (lr_prob * 0.45 + dt_prob * 0.15 + lstm_prob * 0.4)
    # combined_pred = 1 if combined_prob > 0.5 else 0
    risk_score = int(combined_prob * 100)
    risk_status = "🔴 Высокий риск" if risk_score > 70 else \
             "🟡 Средний риск" if risk_score > 30 else "🟢 Низкий риск"
    
    # Поиск подозрительных фраз
    detected_patterns = {}
    for pattern, reason in SUSPICIOUS_PATTERNS.items():
        if pattern in raw_text.lower():  # Поиск в исходном тексте (без очистки)
            detected_patterns[pattern] = reason
    
    # Формирование ответа с информацией от обеих моделей
    response = (
        f"🛡️ Комбинированный риск фишинга: {risk_score}% ({risk_status})\n"
        f"LR модель: {'⚠️ Phishing' if lr_model.predict(X)[0] == 1 else '✅ Ham'} ({int(lr_prob*100)}%)\n"
        f"DT модель: {'⚠️ Phishing' if dt_model.predict(X)[0] == 1 else '✅ Ham'} ({int(dt_prob*100)}%)\n"
        f"LSTM модель: {'⚠️ Phishing' if lstm_model.predict(X)[0] == 1 else '✅ Ham'} ({int(lstm_prob*100)}%)\n"
    )
    
    # Добавляем обнаруженные паттерны
    if detected_patterns:
        response += "\n🔍 Обнаружены подозрительные фразы:\n"
        for pattern, reason in detected_patterns.items():
            response += f"- {pattern}: {reason}\n"
    
    await update.message.reply_text(response)

  
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", hello))
app.add_handler(CommandHandler("get_info", role))
app.add_handler(CommandHandler("set_role", set_role))
app.add_handler(MessageHandler(None, check_email))


app.run_polling()