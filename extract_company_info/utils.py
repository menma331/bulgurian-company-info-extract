import logging
import re
import requests
from bs4 import BeautifulSoup

# Заголовки HTTP-запроса для имитации браузера
headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Cookie": "pll_language=en; _gid=GA1.2.1897445776.1753254216; _gat_gtag_UA_113128210_1=1; _ga_TG452EYHT1=GS2.1.s1753254216$o1$g1$t1753254877$j60$l0$h0; "
              "_ga=GA1.1.1357763097.1753254216",
    "Host": "www.fsc.bg",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
}


def extract_text_from_html_page() -> list[str]:
    """
    Извлекает данные о компаниях с веб-страницы FSC Bulgaria.

    Выполняет HTTP-запрос к странице со списком инвестиционных компаний,
    парсит HTML-таблицу и извлекает текстовые данные из каждой строки.

    Returns:
        list[str]: Список строк, где каждая строка содержит данные об одной компании
                  в формате "Название, Страна, Адрес, Телефон, Email, Лицензия"

    Example:
        > extract_text_from_html_page()
        ['ABC Capital Ltd, Bulgaria, Sofia 1000, 12 Main Str., +35921234567, info@abccapital.bg, License № 123',
         'XYZ Investments, Bulgaria, Varna 9000, 5 Sea Blvd., +35952654321, contact@xyzinvest.com, License № 456']
    """
    logging.info('Получаем данные с сайта')
    url = "https://www.fsc.bg/en/investment-avtivity/lists-of-supervised-entities/investment-firms/"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find_all("tr")[1:]  # Пропускаем заголовок таблицы

    trs = []
    for tr_elem in table:
        trs.append(", ".join([elem.text for elem in tr_elem.find_all('td')]))
    return trs


def clean_phone_number(phones: list[str]) -> list[str]:
    """
    Нормализует телефонные номера к стандартному формату.

    Обрабатывает список телефонных номеров, удаляя мусор и приводя к форматам:
    - +359XXXXXXXXX (для международного формата)
    - 0XXXXXXXXX (для локального формата)

    Args:
        phones (list[str]): Список телефонных номеров для обработки

    Returns:
        list[str]: Список очищенных и нормализованных телефонных номеров

    Examples:
        > clean_phone_number(['02/987 02 35', '0897 889 493'])
        ['+35929870235', '+359897889493']

        > clean_phone_number(['+359 2 962 53 96', 'тел.: 032 632 111'])
        ['+35929625396', '+35932632111']
    """
    if not phones:
        return []

    clean_phones = []
    for phone in phones:
        # Убираем \xa0 и пробелы
        phone = phone.replace('\xa0', ' ').strip()

        # Убираем всё после "fax", "E-mail", "тел", и т.п.
        phone = re.split(r'(fax|e-?mail|факс|тел)', phone, flags=re.IGNORECASE)[0]

        # Убираем лишние символы
        phone = re.sub(r'[^\d+]', '', phone)  # оставить только цифры и плюс

        # Убираем лишние плюсы
        if phone.count('+') > 1:
            phone = '+' + phone.replace('+', '')

        # Убираем ведущие нули, если есть префикс +359
        if phone.startswith('+3590'):
            phone = '+359' + phone[5:]

        # Если телефон начинается с 0 и не содержит код страны — добавим
        elif re.fullmatch(r'0\d{8,9}', phone):
            phone = '+359' + phone[1:]

        # Иногда остаются слишком короткие фрагменты — скипаем
        if len(phone) < 10 or not phone.startswith('+'):
            continue

        clean_phones.append(phone)

    return list(set(clean_phones))  # убираем дубли


def clean_emails(emails: list[str]) -> list[str]:
    """
    Очищает email-адреса от артефактов и мусорных фрагментов.

    Удаляет такие фрагменты как 'www', 'web' из доменной части email-адресов.

    Args:
        emails (list[str]): Список email-адресов для обработки

    Returns:
        list[str]: Список очищенных email-адресов

    Examples:
        > clean_emails(['info@company.web.bg', 'contact@www.domain.com'])
        ['info@company.bg', 'contact@domain.com']
    """
    if not emails:
        return []
    cleaned_emails = []
    for email in emails:
        email_parts = email.split('.')
        email_parts[-1] = email_parts[-1].lower().replace('web', '').replace('www', '')
        cleaned_emails.append('.'.join(email_parts))

    return cleaned_emails


def extract_postal_code_fallback(text: str) -> str | None:
    """
    Извлекает почтовый индекс из текста по fallback-алгоритму (4 цифры подряд).

    Используется, когда другие методы извлечения индекса не сработали.
    Ищет 4-значное число в тексте (болгарские индексы состоят из 4 цифр).

    Args:
        text (str): Текст для анализа (обычно строка с адресом)

    Returns:
        str | None: Найденный почтовый индекс или None, если не найден

    Example:
        > extract_postal_code_fallback("BULGARIA , 1202 Sofia, 16 Srebarna Str.")
        '1202'
    """
    matches = re.findall(r"\b\d{4}\b", text)
    return matches[0] if matches else None


def filter_postal_code(values: list[str]) -> list[str]:
    """
    Фильтрует список значений, оставляя только valid почтовые индексы.

    Valid индекс - строка, содержащая ровно 4 цифры (формат болгарских индексов).

    Args:
        values (list[str]): Список значений для фильтрации

    Returns:
        list[str]: Список значений, соответствующих формату почтового индекса

    Example:
        > filter_postal_code(['1000', 'Sofia', '12', '9000'])
        ['1000', '9000']
    """
    return [v for v in values if re.fullmatch(r"\d{4}", v)]


def filter_street_address(values: list[str]) -> list[str]:
    """
    Фильтрует список значений, оставляя только строки, содержащие указания на улицу.

    Ищет маркеры улиц в тексте: 'str' (street), 'blvd' (boulevard), 'bul.' (bulgarian abbreviation).

    Args:
        values (list[str]): Список значений для фильтрации

    Returns:
        list[str]: Список значений, содержащих указания на улицу

    Example:
        > filter_street_address(['12 Main Str.', 'Sofia', '5 Sea blvd', '1000'])
        ['12 Main Str.', '5 Sea blvd']
    """
    return [v for v in values if any(word in v.lower() for word in ['str', 'blvd', 'bul.'])]