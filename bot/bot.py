from telegram import Update
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackQueryHandler, ApplicationBuilder, CommandHandler, ContextTypes, filters, CallbackContext, MessageHandler
from User import User 
from config import TOKEN, SUSPICIOUS_PATTERNS
from datetime import datetime
from utils import *
from model_manager import ModelManager

# Создаем соединение с базой данных
conn = create_connection()

# Менеджер моделей
model_manager = ModelManager()

USER_CLASSES = {
    0 : 'ORD',
    1 : 'SPEC'
}

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Привет, {update.effective_user.first_name}. Этот бот помогает проверять безопасность ваших электронных писем.')

async def role(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = User(update.effective_user.id, update.effective_user.first_name, USER_CLASSES[0])
    user.print_info()
    await update.message.reply_text(f'Ваш текущий статус: {user.user_role}')

async def check_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text.strip()
    user_id = update.effective_user.id
    username = update.effective_user.username
    timestamp = datetime.now()
    
    # Подключаемся к базе данных
    conn = create_connection()
    add_user(conn, user_id, username)

    # Прогоняем сообщение через модели
    predictions = model_manager.predict_all(message)
    combined = model_manager.get_combined_prediction(message)

    # Сохраняем сообщение в БД
    prediction_result = "Фишинг" if combined >= 0.85 else "Обычное письмо"
    message_id = save_message(conn, user_id, message, prediction_result)

    # Определяем уровень риска
    risk_score = min(100, max(0, int(combined * 100)))
    risk_level = "🔴 Высокий риск" if risk_score >= 85 else ("🟡 Средний риск" if risk_score >= 40 else "🟢 Низкий риск")

    # Формируем ответ пользователю
    response = f"📌 Результат проверки: {prediction_result}\n\n"
    response += f"🛡 Общий уровень риска: {risk_score}% ({risk_level})\n\n"
    response += "Детали анализа:\n"
    for name, pred in predictions.items():
        prob = pred["probability"] or 0
        weight = pred["weight"]
        response += f'- {name}: {prob * 100:.1f}% (Вес: {weight})\n'

    # Если вероятность высокая, отправляем уведомление администраторам
    if combined >= 0.85:
        admin_ids = fetch_admins(conn)
        notification = f"🆘 Предупреждение!\nСообщение было распознано как фишинг.\nПользователь: {username}, ID: {user_id}.\nСообщение: {message}\nОтправлено: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}."
        for admin_id in admin_ids:
            await context.bot.send_message(chat_id=admin_id, text=notification)
    
    # Формируем инлайн-клавиатуру с кнопкой "report"
    keyboard = [
        [InlineKeyboardButton("👎 Report", callback_data=f'report_{message_id}')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # await update.message.reply_markdown_v2(response)
    await update.message.reply_text(response, reply_markup=reply_markup)

    conn.close()

async def report_message_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_id = update.message.reply_to_message.message_id
    reporter_id = update.effective_user.id
    reason = "Пользователь выразил несогласие с результатом."

    conn = create_connection()
    report_message(conn, message_id, reporter_id, reason)
    await update.message.reply_text("Спасибо за ваш отзыв!")
    conn.close()

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Парсим callback_data
    action, message_id = query.data.split('_')

    if action == 'report':
        # Логика обработки жалобы
        # Например, сохраним жалобу в базе данных
        user_id = query.from_user.id
        conn = create_connection()
        report_message(conn, message_id, user_id, reason="Пользователь нажал кнопку 'report'")
        conn.close()

        # Уведомляем пользователя, что жалоба принята
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("Жалоба зарегистрирована. Спасибо за ваше участие!")

async def make_admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) != 1:
        await update.message.reply_text("Используйте команду /make_admin <ID пользователя>")
        return
    telegram_id = int(args[0])

    conn = create_connection()
    make_admin(conn, telegram_id)
    await update.message.reply_text(f"Пользователь с ID {telegram_id} назначен администратором.")
    conn.close()

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    # Регистрация обработчиков команд
    app.add_handler(CommandHandler("start", hello))
    app.add_handler(CommandHandler("role", role))
    app.add_handler(CommandHandler("report", report_message_command))
    app.add_handler(CommandHandler("make_admin", make_admin_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, check_email))

    print("Bot started.")
    app.run_polling()