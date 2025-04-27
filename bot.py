from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, filters, CallbackContext, MessageHandler
from User import User 
from config import TOKEN
import pickle
from utils import decode_prediction, preprocess_text, load_model, vectorize_text


# Загрузка моделей
lr_model = load_model('models/lr.pkl')
dt_model = load_model('models/dt_class_weight_None_criterion_entropy_max_depth_None_min_samples_leaf_2_min_samples_split_10.pkl')

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

async def check_email(update: Update, context: ContextTypes.DEFAULT_TYPE):

    text = update.message.text
        
    # Векторизация
    X = vectorize_text(text)

    # Предсказания
    # lr_pred = lr_model.predict(X)[0]
    dt_pred = dt_model.predict(X)[0]

    await update.message.reply_text(
        # f"LR: {'Phishing' if lr_pred == 1 else 'Ham'}\n"
        f"DT: {'Phishing' if dt_pred == 1 else 'Ham'}"
    )
  
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", hello))
app.add_handler(CommandHandler("get_info", role))
app.add_handler(CommandHandler("set_role", set_role))
app.add_handler(MessageHandler(None, check_email))


app.run_polling()