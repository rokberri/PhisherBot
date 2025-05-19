from telegram.ext import CallbackQueryHandler, ApplicationBuilder, CommandHandler, filters, MessageHandler
from config import TOKEN, SUSPICIOUS_PATTERNS
from bot_functions import hello, report_message_command, make_admin_command, check_email, unified_button_callback, set_default_admin, show_reports_to_admin, process_admin_choice


if __name__ == "__main__":
    # Инициализируем бота 
    app = ApplicationBuilder().token(TOKEN).build()

    # Инициализируем дефолтного администратора
    set_default_admin()
    
    # Регистрация обработчиков команд
    app.add_handler(CommandHandler("start", hello))
    # app.add_handler(CommandHandler("report", report_message_command))
    # app.add_handler(CallbackQueryHandler(unified_button_callback))

    app.add_handler(CommandHandler("show_reports", show_reports_to_admin))
    # app.add_handler(CallbackQueryHandler(unified_button_callback))
    app.add_handler(CommandHandler("set_admin", make_admin_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, check_email))
    app.add_handler(CallbackQueryHandler(unified_button_callback))

    print("Bot started.")

    app.run_polling()