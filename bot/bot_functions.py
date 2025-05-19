from model_manager import ModelManager
from datetime import datetime
from config import DEFAULT_ADMIN_ID, MEDIUM_THREAT_LEVEL, HIGH_THREAT_LEVEL
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import asyncio
from utils.db_utils import create_connection, add_user, save_message, fetch_admins, report_message, make_admin, get_reported_messages
import logging

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Менеджер моделей
model_manager = ModelManager()

USER_CLASSES = {
    0 : 'ORD',
    1 : 'SPEC'
}

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f'/start command received from user {update.effective_user.id}')
    await update.message.reply_text(f'Привет, {update.effective_user.first_name}. Этот бот помогает проверять безопасность ваших электронных писем.')

async def check_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text.strip()
    user_id = update.effective_user.id
    username = update.effective_user.username
    timestamp = datetime.now()
    
    # Подключаемся к базе данных
    conn = create_connection()
    add_user(conn, user_id, username)
    logger.info(f'Processing message from user {user_id}: "{message}"')

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
    if combined >= HIGH_THREAT_LEVEL:
        admin_ids = fetch_admins(conn)
        notification = f"🆘 Предупреждение!\nСообщение было распознано как фишинг.\nПользователь: {username}, ID: {user_id}.\nСообщение:\n {message}\nОтправлено: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}."
        for admin_id in admin_ids:
            await context.bot.send_message(chat_id=admin_id, text=notification)
        logger.warning(f'Suspicious message detected from user {user_id}')
    
    # Формируем инлайн-клавиатуру с кнопкой "report"
    keyboard = [
        [InlineKeyboardButton("👎 Report", callback_data=f'report:{message_id}')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(response, reply_markup=reply_markup)
    logger.info(f'Message processed successfully for user {user_id}')

    conn.close()

async def report_message_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Проверяем, есть ли сообщение, на которое ответили
    if update.message.reply_to_message is None:
        await update.message.reply_text("Используйте команду '/report' в ответ на конкретное сообщение.")
        return

    message_id = update.message.reply_to_message.message_id
    reporter_id = update.effective_user.id
    reason = "Пользователь выразил несогласие с результатом."

    conn = create_connection()
    report_message(conn, message_id, reporter_id, reason)
    await update.message.reply_text("Спасибо за ваш отзыв!")
    logger.info(f'Reported message {message_id} by user {reporter_id}')
    conn.close()

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    print(query.data)
    # Парсим callback_data
    action, message_id = query.data.split(':')

    if action == 'report':
        # Отмечаем сообщение как ожидающее рассмотрения
        user_id = query.from_user.id
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE messages SET reviewed=FALSE WHERE id=?", (message_id,))
        conn.commit()
        conn.close()

        # Уведомляем пользователя, что жалоба принята
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("Жалоба зарегистрирована. Спасибо за ваше участие!")
        logger.info(f'Button callback triggered for message {message_id}')

async def make_admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Проверяем права пользователя перед назначением нового администратора
    if not is_admin(update):
        await update.message.reply_text("У вас недостаточно прав для выполнения данной команды.")
        return
    
    args = context.args

    if len(args) != 1:
        await update.message.reply_text("Используйте команду /make_admin <ID пользователя>")
        return
    telegram_id = int(args[0])

    conn = create_connection()
    make_admin(conn, telegram_id)
    await update.message.reply_text(f"Пользователь с ID {telegram_id} назначен администратором.")
    logger.info(f'User {telegram_id} promoted to admin')
    conn.close()

def set_default_admin():
    # Устанавливаем дефолтного администратора при запуске
    conn = create_connection()

    def user_exists_and_is_admin(conn, telegram_id):
        """Проверяет, существует ли пользователь с указанным telegram_id и является ли он администратором."""
        cursor = conn.cursor()
        cursor.execute("SELECT admin FROM users WHERE telegram_id=?", (telegram_id,))
        result = cursor.fetchone()
        return bool(result and result[0])  # True, если пользователь существует и является администратором
    
    if not user_exists_and_is_admin(conn, DEFAULT_ADMIN_ID):
        add_user(conn,DEFAULT_ADMIN_ID)
        make_admin(conn, DEFAULT_ADMIN_ID)
        logger.info(f'Default admin {DEFAULT_ADMIN_ID} initialized')
    else: 
        logger.info(f'Default admin {DEFAULT_ADMIN_ID} already initialized')
    conn.close()

def is_admin(update: Update) -> bool:
    """Проверяет, является ли пользователь администратором."""
    user_id = update.effective_user.id
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT admin FROM users WHERE telegram_id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return bool(result and result[0])


async def show_reports_to_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает пять репортнутых сообщений администратору с возможностью выбрать статус."""
    # Проверяем, является ли пользователь администратором
    if not is_admin(update):
        await update.message.reply_text("У вас недостаточно прав для выполнения данной команды.")
        return

    conn = create_connection()
    reports = get_reported_messages(conn)
    if not reports:
        await update.message.reply_text("Нет сообщений, ожидающих рассмотрения.")
        return

    for report in reports:
        message_id, content, created_at, current_prediction = report
        buttons = [
            [InlineKeyboardButton("Фишинг", callback_data=f'set_fish:{message_id}'),
             InlineKeyboardButton("Безопасное письмо", callback_data=f'set_safe:{message_id}')],
        ]
        reply_markup = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(f"Сообщение:\n{content}\n\nТекущий статус: {current_prediction}\nВремя отправления: {created_at}",
                                       reply_markup=reply_markup)

    conn.close()


async def process_admin_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print('FUCK')
    query = update.callback_query
    choice, message_id = query.data.split(':')
    print(choice,' ', message_id)
    conn = create_connection()

    # Проверяем статус выбранного элемента
    if choice == 'set_fish':
        new_status = "Фишинг"
    elif choice == 'set_safe':
        new_status = "Безопасное письмо"
    else:
        await query.answer("Ошибка обработки.", show_alert=True)
        return

    # Обновляем статус основного сообщения и отмечаем его как рассмотренное
    cursor = conn.cursor()
    cursor.execute("UPDATE messages SET prediction=?, reviewed=TRUE WHERE id=?", (new_status, message_id))
    conn.commit()

    # Ответ пользователю
    await query.answer(f"Статус сообщения обновлён на '{new_status}'")
    await query.edit_message_reply_markup(reply_markup=None)

    conn.close()

async def unified_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Парсим callback_data
    action, message_id = query.data.split(':')

    if action == 'report':  # Первая кнопка для отправки жалобы
        # Берём информацию о пользователе и сообщение
        reporter_id = query.from_user.id
        reason = "Пользователь нажал кнопку 'Report'"

        conn = create_connection()
        report_message(conn, message_id)
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("Жалоба зарегистрирована. Спасибо за ваше участие!")
        conn.close()

    elif action.startswith('set'):  # Вторая кнопка для выбора статуса (fish/safe)
        # Асинхронно вызываем функцию process_admin_choice
        asyncio.create_task(process_admin_choice(update, context))

    else:
        await query.answer("Неизвестное действие.", show_alert=True)


async def undo_admin_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отменяет решение администратора и возвращает сообщение на этап ожидания."""
    query = update.callback_query
    _, message_id = query.data.split(':')

    conn = create_connection()
    undo_admin_decision(conn, message_id)
    conn.close()

    await query.answer("Решение отменено, сообщение возвращено на рассмотрение.")
    await query.edit_message_reply_markup(reply_markup=None)