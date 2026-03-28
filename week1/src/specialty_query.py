# -*- coding: utf-8 -*-
"""
Адаптированный запрос для специальности
Лабораторная работа №1
Автор: [ВАША ФАМИЛИЯ]
Специальность: [ВАША СПЕЦИАЛЬНОСТЬ]
Тема диплома: [ТЕМА ДИПЛОМНОЙ РАБОТЫ]
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from agent_core import YandexGPTClient

# Загрузка переменных окружения
load_dotenv()

# Определение корневой директории (week1)
BASE_DIR = Path(__file__).resolve().parent.parent


def get_specialty_prompt() -> str:
    """
    Возвращает промпт, адаптированный под специальность.

    ЗАДАНИЕ: Замените этот промпт на свой, релевантный вашей теме диплома.
    """

    prompt = """
Я студент технической специальности, работаю над дипломной темой:
"Информационная система для детекции графиков линии"

Прошу предоставить информацию по следущему вопросу:
1. Какие, технологии помогу удалить сетку с сложных технических графиков не повреждая нужную линию?
2. Если готовые и открытые(бесплатные) решения?

Требования к ответу:
• Ответ должен быть структурирован
• Используй технические термины
• Приведи конкретные примеры
• Объём: 300-500 слов
"""

    return prompt


def main():
    """Выполнение адаптированного запроса."""

    print("=" * 80)
    print("АДАПТИРОВАННЫЙ ЗАПРОС ПО СПЕЦИАЛЬНОСТИ")
    print("=" * 80)

    # Инициализация клиента
    iam_token = os.getenv("YANDEX_IAM_TOKEN")
    folder_id = os.getenv("YANDEX_FOLDER_ID")

    client = YandexGPTClient(iam_token, folder_id)

    # Получение персонализированного промпта
    prompt = get_specialty_prompt()

    print("\nЗАПРОС:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)

    # Выполнение запроса
    print("\nОТВЕТ МОДЕЛИ:")
    print("-" * 80)

    response = client.generate(prompt, temperature=0.5)
    print(response["text"])

    print("-" * 80)
    print(f"\nТокены: вход={response['tokens_input']}, выход={response['tokens_output']}")

    # Сохранение результата
    output_path = BASE_DIR / "docs" / "specialty_response.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ЗАПРОС:\n")
        f.write(prompt)
        f.write("\n\nОТВЕТ:\n")
        f.write(response["text"])

    print(f"\n✅ Результат сохранён в {output_path}")


if __name__ == "__main__":
    main()