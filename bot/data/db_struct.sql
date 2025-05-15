-- Таблица Users: хранится информация о пользователях
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER UNIQUE NOT NULL,      -- Уникальный идентификатор пользователя
    username TEXT,                            -- Имя пользователя в Telegram
    admin BOOLEAN DEFAULT FALSE               -- Признак админа (по умолчанию False)
);

-- Таблица Messages: хранятся все присланные сообщения
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id INTEGER REFERENCES users(id),   -- Внешний ключ на пользователя
    content TEXT NOT NULL,                     -- Само содержимое сообщения
    prediction TEXT NOT NULL,                  -- Решение нашей системы (например, "Фишинг"/"Обычный")
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Время поступления сообщения
);

-- Таблица Reported_Messages: сюда заносятся сообщения, с классификацией которых не согласен пользователь
CREATE TABLE IF NOT EXISTS reported_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER REFERENCES messages(id),  -- Внешний ключ на запись в таблице messages
    reported_by INTEGER REFERENCES users(id),    -- Кто сообщил о несоответствии
    report_reason TEXT,                          -- Причина возражения
    reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Когда жалоба была сделана
);