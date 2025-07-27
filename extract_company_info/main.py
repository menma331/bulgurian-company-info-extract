import pandas as pd
from gliner import GLiNER

from utils import (
    clean_phone_number,
    clean_emails,
    filter_street_address,
    filter_postal_code,
    extract_postal_code_fallback,
    extract_text_from_html_page
)

# --- Инициализация модели и меток ---
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")

# Маппинг меток сущностей и их описаний для модели
label_map = {
    "phone": "phone number like +359 2 439 81 50 or starting with +359",
    "site": "website address like www.bank.bg or www.domain.com",
    "street_address": "street address that contains street name and number like '16 Srebarna Str.'",
    "city": "city name in Bulgaria (e.g. Sofia, Plovdiv)",
    "country": "country name like Bulgaria",
    "postal_code": "numeric postal code"
}

# Отдельная метка для email-адресов
email_label = {"email": "email address with @ symbol, e.g. info@company.bg"}

# Обратный маппинг для преобразования описаний в ключи
reverse_map = {v: k for k, v in label_map.items()}

# --- Извлечение текстов со страницы ---
texts = extract_text_from_html_page()


def extract_single_company_info(text: str) -> dict:
    """
    Извлекает и обрабатывает информацию об одной компании из текста.

    Использует модель GLiNER для извлечения сущностей, затем применяет
    дополнительные обработчики для очистки и нормализации данных.

    Args:
        text (str): Текст с информацией о компании в формате "Название, Страна, Адрес, ..."

    Returns:
        dict: Словарь с структурированной информацией о компании:
            - Сompany name: Название компании
            - Phone numbers: Очищенные телефонные номера
            - Emails: Очищенные email-адреса
            - Websites: Веб-сайты
            - Street address: Адрес улицы
            - City: Город
            - Country: Страна
            - Postal code: Почтовый индекс

    Example:
        > text = "ABC Capital, Bulgaria, Sofia 1000, 12 Main Str., +35921234567, info@abc.bg"
        > extract_single_company_info(text)
        {
            'Сompany name': 'ABC Capital',
            'Phone numbers': ['+35921234567'],
            'Emails': ['info@abc.bg'],
            'Websites': [],
            'Street address': '12 Main Str.',
            'City': 'Sofia',
            'Country': 'Bulgaria',
            'Postal code': '1000'
        }
    """
    # Извлекаем email отдельно
    email_ents = model.predict_entities(text, labels=list(email_label.values()))
    extracted_emails = [e['text'] for e in email_ents] if email_ents else []

    # Извлекаем остальные сущности
    ents = model.predict_entities(text, labels=list(label_map.values()))
    extracted = {key: [] for key in label_map}

    for e in ents:
        key = reverse_map.get(e['label'])
        if key:
            extracted[key].append(e['text'])

    # Очистка и пост-обработка сущностей
    cleaned_phones = clean_phone_number(extracted['phone'])
    cleaned_emails = clean_emails(extracted_emails)
    address_candidates = filter_street_address(extracted["street_address"])
    postal_candidates = filter_postal_code(extracted["postal_code"])
    postal_code = postal_candidates[0] if postal_candidates else extract_postal_code_fallback(text)

    return {
        "Сompany name": text.split(',')[0],
        "Phone numbers": cleaned_phones or "No phones detected",
        "Emails": cleaned_emails or "No emails detected",
        "Websites": extracted["site"] or "No website detected",
        "Street address": address_candidates[0] if address_candidates else "No street address detected",
        "City": extracted["city"][0] if extracted["city"] else "No city detected",
        "Country": extracted["country"][0] if extracted["country"] else "No country detected",
        "Postal code": postal_code or "No postal code detected"
    }


def extract_companies_info(texts: list[str]) -> list[dict]:
    """
    Обрабатывает список текстов компаний и возвращает структурированные данные.

    Args:
        texts (list[str]): Список текстов о компаниях

    Returns:
        list[dict]: Список словарей с информацией о компаниях

    Example:
        > texts = ["Company1, Bulgaria...", "Company2, Sofia..."]
        > extract_companies_info(texts)
        [
            {'Сompany name': 'Company1', ...},
            {'Сompany name': 'Company2', ...}
        ]
    """
    return [extract_single_company_info(text) for text in texts]


def main():
    """
    Основная функция выполнения скрипта.

    Выполняет:
    1. Извлечение информации о компаниях
    2. Сохранение в CSV-файл
    3. Вывод первых 5 записей для проверки
    """
    data = extract_companies_info(texts)
    df = pd.DataFrame(data)
    df.to_csv("fsc_companies_ner.csv", index=False)
    print('-' * 14, 'Результат готов', '-' * 14)
    print(df.head())


if __name__ == "__main__":
    main()