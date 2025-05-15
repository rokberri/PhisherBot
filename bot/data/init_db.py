import sqlite3

# Создаем подключение к базе данных
connection = sqlite3.connect('bot_database.db')
cursor = connection.cursor()

# Выполняем создание таблиц
with open('db_struct.sql', 'r') as file:
    schema_sql = file.read()
    cursor.executescript(schema_sql)

# Закрываем подключение
connection.close()