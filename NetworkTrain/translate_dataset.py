import os
import pandas as pd

DATA_PATH = 'DATASETS/test.csv'
TEST = 'sliced_parts/part_1.csv'

# import asyncio
# import pandas as pd
# from googletrans import Translator
# import os
# import asyncio
# import pandas as pd
# from googletrans import Translator

# CHUNK_SIZE = 4000  # Максимальная длина текста для перевода
# MAX_CONCURRENT_REQUESTS = 5  # Количество одновременных запросов

# async def translate_chunk(translator, text):
#     """Функция для перевода текста с учетом ограничения длины"""
#     parts = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
#     full_translation = ''
#     for part in parts:
#         translation_result = await translator.translate(part, src="auto", dest="ru")
#         full_translation += translation_result.text
#         print(full_translation)
#     return full_translation

# async def process_column(column_name, column_values):
#     """Асинхронная обработка столбца с применением ограничений на запросы"""
#     semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
#     translator = Translator()
#     tasks = []
#     indexes = []
#     for idx, value in column_values.items():
#         if isinstance(value, str):
#             async def wrap(idx=idx, value=value):  # Фиксация значений
#                 async with semaphore:
#                     return idx, await translate_chunk(translator, value)
#             tasks.append(asyncio.create_task(wrap()))
#             indexes.append(idx)
#     translated_results = await asyncio.gather(*tasks)
#     return dict(translated_results)

# async def main(file_path):
#     print(f"Обработка файла: {file_path}")  
#     # df = pd.read_csv(f"sliced_parts/{file_path}")
#     df = pd.read_csv(f"sliced_parts/{file_path}", encoding="utf-8", errors="replace")
#     columns_to_translate = ["Email Text"]
#     for column in columns_to_translate:
#         translated_dict = await process_column(column, df[column])
#         for idx, new_value in translated_dict.items():
#             df.at[idx, column] = new_value
#     df.to_csv(f"DATASETS/ru/translated_{file_path}.csv", index=False)

# if __name__ == "__main__":
#     # print(sorted(os.listdir("sliced_parts")))
#     for filename in os.listdir("sliced_parts"):
#         asyncio.run(main(filename))


# import os
# import pandas as pd
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
# from deep_translator import GoogleTranslator

# CHUNK_SIZE = 4000  # Максимум символов в части
# MAX_CONCURRENT_REQUESTS = 5

# # Обёртка для перевода в потоке
# def sync_translate_chunk(text):
#     translator = GoogleTranslator(source='auto', target='ru')
#     parts = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
#     translated = [translator.translate(part) for part in parts]
#     print(translated)
#     return ''.join(translated)

# async def async_translate_chunk(loop, text):
#     return await loop.run_in_executor(None, sync_translate_chunk, text)

# async def process_column(loop, column_values):
#     semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
#     tasks = {}

#     for idx, value in column_values.items():
#         if isinstance(value, str):
#             async def translate_one(idx=idx, value=value):
#                 async with semaphore:
#                     return idx, await async_translate_chunk(loop, value)
#             tasks[idx] = asyncio.create_task(translate_one())
#     results = await asyncio.gather(*tasks.values())
#     return dict(results)

# async def main(file_path):
#     print(f"🔄 Обработка файла: {file_path}")
#     df = pd.read_csv(f"sliced_parts/{file_path}", encoding="utf-8")

#     loop = asyncio.get_event_loop()
#     columns_to_translate = ["Email Text"]

#     for column in columns_to_translate:
#         translated = await process_column(loop, df[column])
#         for idx, text in translated.items():
#             df.at[idx, column] = text

#     os.makedirs("DATASETS/ru", exist_ok=True)
#     df.to_csv(f"DATASETS/ru/translated_{file_path}", index=False)
#     print(f"✅ Перевод завершён: translated_{file_path}")

# if __name__ == "__main__":
#     files = sorted(os.listdir("sliced_parts"))
#     for filename in files:
#         asyncio.run(main(filename))


import aiohttp
import asyncio
from tqdm.asyncio import tqdm
import pandas as pd

LIBRETRANSLATE_URL = 'http://localhost:5010/translate'
# LIBRETRANSLATE_URL = '"https://libretranslate.de"'
# async def translate_text(session, text, source='en', target='ru'):
#     payload = {
#         'q': text,
#         'source': source,
#         'target': target,
#         'format': 'text'
#     }
#     async with session.post(LIBRETRANSLATE_URL, json=payload) as resp:
#         if resp.status != 200:
#             return f"[Error: {resp.status}]"
#         data = await resp.json()
#         return data['translatedText']

import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio


# Ограничим параллелизм
MAX_CONCURRENT_REQUESTS = 1


async def translate_text(session, text, source='en', target='ru', retries=3):
    # url = "http://localhost:5010/translate"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        'q': text,
        'source': source,
        'target': target,
        'format': 'text'
    }

    for attempt in range(retries):
        try:
            async with session.post(LIBRETRANSLATE_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('translatedText', '[No translation]')
                else:
                    return f"[HTTP {resp.status}]"
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(1)  # Подождать перед повтором
                continue
            import traceback
            print(f"⚠️ Ошибка при запросе:\n{traceback.format_exc()}")
            return f"[Error: {str(e)}]"



async def translate_all(texts):
    async with aiohttp.ClientSession() as session:
        tasks = [translate_text(session, text) for text in texts]
        results = []
        for f in tqdm_asyncio.as_completed(tasks, desc="📤 Перевод", total=len(tasks)):
            result = await f
            results.append(result)
        return results

async def main():
    df = pd.read_csv(DATA_PATH)
    texts = df['text'].fillna('').astype(str).tolist()

    translations = await translate_all(texts)
    df['translated'] = translations
    df.to_csv('translated_data.csv', index=False)
    print("✅ Перевод завершён. Сохранено в translated_data.csv")

if __name__ == '__main__':
    asyncio.run(main())



