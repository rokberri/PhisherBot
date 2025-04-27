from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, filters, CallbackContext, MessageHandler
from User import User 
from config import TOKEN
import pickle
from utils import decode_prediction, preprocess_text, load_model, vectorize_text


# Загрузка моделей
lr_model = load_model('models/lr.pkl')
# dt_model = load_model('models/dt_class_weight_None_criterion_entropy_max_depth_None_min_samples_leaf_2_min_samples_split_10.pkl')

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
    text = update.message.text.lower()
    X = vectorize_text(text)
    
    # Получаем вероятность вместо класса
    lr_prob = lr_model.predict_proba(X)[0][1]  # Вероятность класса 1 (фишинг)
    lr_pred = 1 if lr_prob > 0.5 else 0  # Класс на основе порога 0.5
    
    # Поиск подозрительных фраз
    detected_patterns = {}
    for pattern, reason in SUSPICIOUS_PATTERNS.items():
        if pattern in text:
            detected_patterns[pattern] = reason
    
    # Оценка риска
    risk_score = int(lr_prob * 100)  # Используем вероятность, а не предсказанный класс
    risk_status = "🔴 Высокий риск" if risk_score > 70 else \
                 "🟡 Средний риск" if risk_score > 30 else "🟢 Низкий риск"
    
    # Формирование ответа
    response = (
        f"🛡️ Риск фишинга: {risk_score}% ({risk_status})\n"
        f"Модель: {'⚠️ Phishing' if lr_pred == 1 else '✅ Ham'}\n"
    )
    
    if detected_patterns:
        response += "\n🔍 Обнаружены подозрительные фразы:\n"
        for pattern, reason in detected_patterns.items():
            response += f"- '{pattern}': {reason}\n"
    
    await update.message.reply_text(response)
  
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", hello))
app.add_handler(CommandHandler("get_info", role))
app.add_handler(CommandHandler("set_role", set_role))
app.add_handler(MessageHandler(None, check_email))


app.run_polling()