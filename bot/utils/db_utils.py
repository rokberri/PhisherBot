import sqlite3

def create_connection(db_file='data/bot_database.db'):
    """Создает подключение к указанной базе данных SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Exception as e:
        print(f"Ошибка при соединении с базой данных: {e}")
    return conn

def add_user(conn, telegram_id, username=None):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO users (telegram_id, username) VALUES (?, ?)",
        (telegram_id, username)
    )
    conn.commit()

def save_message(conn, sender_id, content, prediction):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (sender_id, content, prediction) VALUES (?, ?, ?)",
        (sender_id, content, prediction)
    )
    conn.commit()
    return cur.lastrowid

def report_message(conn, message_id):
    """Ставит сообщение на рассмотрение администратору."""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE messages SET reviewed=FALSE WHERE id=?",
        (message_id,)
    )
    conn.commit()

def make_admin(conn, telegram_id):
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET admin=TRUE WHERE telegram_id=?",
        (telegram_id,)
    )
    conn.commit()

def fetch_admins(conn):
    cur = conn.cursor()
    cur.execute("SELECT telegram_id FROM users WHERE admin=TRUE")
    rows = cur.fetchall()
    return [row[0] for row in rows]

def get_reported_messages(conn):
    """Получает список сообщений, которые ожидают рассмотрения администратором."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, content, created_at, prediction
        FROM messages
        WHERE reviewed=FALSE
        ORDER BY created_at ASC;
    """)
    return cursor.fetchall()

def undo_admin_decision(conn, message_id):
    """Отменяет решение администратора и восстанавливает сообщение в режиме ожидания рассмотрения."""
    cursor = conn.cursor()
    cursor.execute("UPDATE messages SET reviewed=FALSE WHERE id=?", (message_id,))
    conn.commit()
