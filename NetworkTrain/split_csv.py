import os
import pandas as pd
DATA_PATH = 'DATASETS/Phishing_Email.csv'
# def clean_illegal_chars(df):
#     def clean_cell(cell):
#         if isinstance(cell, str):
#             return ''.join(c for c in cell if (
#                 c == '\t' or c == '\n' or c == '\r' or
#                 ('\x20' <= c <= '\ud7ff') or ('\ue000' <= c <= '\ufffd')
#             ))
#         return cell
#     return df.applymap(clean_cell)

def split_csv_by_lines(csv_path, max_lines=1000, output_dir="sliced_parts", base_name="part"):
    print(f"[1] Загрузка CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # print(f"[2] Очистка недопустимых символов...")
    # df = clean_illegal_chars(df)

    os.makedirs(output_dir, exist_ok=True)

    num_parts = (len(df) // max_lines) + (1 if len(df) % max_lines != 0 else 0)
    print(f"[3] Разбиение на {num_parts} частей по {max_lines} строк.")

    for part_number in range(num_parts):
        start = part_number * max_lines
        end = start + max_lines
        part_df = df.iloc[start:end]

        final_path = os.path.join(output_dir, f"{base_name}_{part_number + 1}.csv")
        part_df.to_csv(final_path, index=False)

        print(f"  → [{part_number + 1}] Сохранено {len(part_df)} строк → {final_path}")

    print(f"[✓] Готово. Всего файлов: {num_parts}")


if __name__ == "__main__":

    split_csv_by_lines(DATA_PATH, max_lines=1000)