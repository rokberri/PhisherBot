from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, filters, CallbackContext, MessageHandler
from User import User 
from config import TOKEN
import pickle
# from utils import clean_up


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

# async def check_email(update: Update, context: CallbackContext) -> None:
#     chat_id = update.message.chat_id
#     message = update.message.text

     
#     with open('lr_model.pkl', 'rb') as file:
#         lr_model = pickle.load(file)
#     with open('cv.pkl', 'rb') as f:
#         cv = pickle.load(f)
#     predictions = lr_model.predict(cv.transform([clean_up(message)]))

    await update.message.reply_text(f'Result: {predictions[0]}')

app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", hello))
app.add_handler(CommandHandler("get_info", role))
app.add_handler(CommandHandler("set_role", set_role))
# app.add_handler(MessageHandler(None, check_email))


app.run_polling()