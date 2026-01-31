import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import quote
import os

def clean_extracted_text(text):
    """Исправленная очистка текста БЕЗ разбивки слов на символы"""
    if not text:
        return ""

    # 1. Сначала удаляем раздел "Источники" и всё после него
    # Находим начало раздела "Источники" и обрезаем текст до этого места
    sources_patterns = [
        r'^Источники\s*$.*',
        r'^Примечания\s*$.*',
        r'^Ссылки\s*$.*',
        r'^Литература\s*$.*',
        r'^См\. также\s*$.*',
        r'^Внешние ссылки\s*$.*',
    ]

    lines = text.split('\n')
    cleaned_lines = []
    stop_processing = False

    for line in lines:
        line_stripped = line.strip()

        # Проверяем, не начался ли раздел "Источники" или подобный
        if not stop_processing:
            for pattern in sources_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    stop_processing = True
                    break

        if not stop_processing and line_stripped:
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # 2. Удаляем ссылки в формате [1], [2], [источник?] и т.д.
    text = re.sub(r'\[\d+\]', '', text)  # [1], [2], [3]
    text = re.sub(r'\[[^\]]*[?]\]', '', text)  # [источник?], [кто?]
    text = re.sub(r'\[[^\]]*\]', '', text)  # все остальные ссылки

    # 3. Удаляем копирайты и технические пометки (удаляем строки целиком)
    copyright_keywords = [
        'Материал из Википедии',
        'Страница последний раз',
        'Эта страница',
        'Toggle the table of contents',
        'Для улучшения этой статьи',
        'Вы можете помочь',
        'Основная статья',
        'Содержание',
        'Оглавление',
        'Navigation menu',
        'Править код',
        '[править]',
        'редактировать',
        'обсуждение',
        'вклад',
    ]

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line_lower = line.lower()
        should_remove = False

        # Проверяем, содержит ли строка ключевые слова для удаления
        for keyword in copyright_keywords:
            if keyword.lower() in line_lower:
                should_remove = True
                break

        # Также удаляем очень короткие строки (менее 4 символов), если это не числа
        if not should_remove and len(line.strip()) < 4:
            if not line.strip().isdigit():
                # Но не удаляем распространенные короткие слова
                common_short = ['и', 'в', 'на', 'с', 'по', 'у', 'о', 'к', 'а', 'но', 'за', 'от', 'до', 'из', 'без']
                if line.strip().lower() not in common_short:
                    should_remove = True

        if not should_remove and line.strip():
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # 4. ВАЖНО: УБИРАЕМ ВСЕ ДОПОЛНИТЕЛЬНЫЕ ПРОБЕЛЫ МЕЖДУ БУКВАМИ
    # Это основное исправление - убираем лишние пробелы внутри слов

    # Разбиваем на строки и обрабатываем каждую отдельно
    lines = text.split('\n')
    processed_lines = []

    for line in lines:
        # Оставляем пробелы только между словами, удаляем пробелы внутри слов
        # Простой подход: объединяем все пробелы в один, но не удаляем пробелы полностью
        line = re.sub(r'\s+', ' ', line).strip()
        processed_lines.append(line)

    text = '\n'.join(processed_lines)

    # 5. Исправляем специфические проблемы, но НЕ разбиваем слова
    # Вместо этого исправляем очевидные ошибки

    # Слипшиеся слова (только явные случаи)
    common_fixes = [
        # Предлоги слипшиеся с существительными
        (r'(\b[а-яё]{1,3})([А-ЯЁ][а-яё]+)', r'\1 \2'),  # изНочного → из Ночного
        (r'(\b[а-яё]+)([А-ЯЁ][а-яё]+)', r'\1 \2'),  # Десантируетв → Десантирует в
    ]

    for pattern, replacement in common_fixes:
        text = re.sub(pattern, replacement, text)

    # 6. Удаляем HTML-сущности
    text = re.sub(r'&[a-z]+;', ' ', text)

    # 7. Удаляем специальные символы, но сохраняем нормальные знаки препинания
    text = re.sub(r'[\u200b\u200c\u200d\uFEFF\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 8. Восстанавливаем нормальные пробелы после знаков препинания
    text = re.sub(r'([.,!?;:])([а-яёА-ЯЁ])', r'\1 \2', text)

    # 9. Удаляем лишние пустые строки
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # 10. Финальная чистка: удаляем строки только из спецсимволов
    final_lines = []
    for line in text.split('\n'):
        line_stripped = line.strip()
        # Удаляем строки, состоящие только из не-букв (кроме пунктуации в конце)
        if line_stripped and not re.match(r'^[^\wа-яА-ЯёЁ]*$', line_stripped):
            # Убираем "смайлики" типа ===, *** в конце
            line_stripped = re.sub(r'[=*~^_\-+]{3,}$', '', line_stripped)
            final_lines.append(line_stripped)

    text = '\n'.join(final_lines)
    return text.strip()

def get_character_text(character_name, base_url):
    """Получение чистого текста статьи о персонаже"""
    try:
        encoded_name = quote(character_name)
        url = f"{base_url}{encoded_name}"

        print(f"Получаю данные для: {character_name}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }

        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            print(f"  Ошибка HTTP {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')

        # Удаляем ненужные элементы
        for selector in ['script', 'style', 'noscript', 'iframe',
                         '.mw-editsection', '.reference', '.navbox',
                         '.infobox', '.sidebar', '.metadata', '.catlinks',
                         '#mw-navigation', '#mw-head', '#footer', '.toc',
                         'table', 'aside']:
            for element in soup.select(selector):
                element.decompose()

        # Ищем основной контент
        content_div = soup.find('div', {'id': 'mw-content-text'}) or \
                      soup.find('div', {'class': 'mw-parser-output'})

        if not content_div:
            return None

        # Удаляем ссылки, но сохраняем их текст БЕЗ добавления лишних пробелов
        for link in content_div.find_all('a'):
            link_text = link.get_text(strip=True)
            if link_text:
                # Просто заменяем ссылку её текстом
                link.replace_with(link_text)
            else:
                link.decompose()

        # Извлекаем текст более аккуратно
        text_parts = []

        # Обрабатываем основные элементы
        for element in content_div.find_all(['p', 'h2', 'h3', 'h4', 'li']):
            if element.get_text(strip=True):
                # Получаем текст элемента
                element_text = element.get_text(separator=' ', strip=True)

                if element.name.startswith('h'):
                    # Заголовки
                    level = int(element.name[1])
                    text_parts.append('#' * (level + 1) + ' ' + element_text)
                elif element.name == 'li':
                    # Элементы списка
                    text_parts.append('• ' + element_text)
                else:
                    # Обычные параграфы
                    text_parts.append(element_text)

        if text_parts:
            text = '\n\n'.join(text_parts)
        else:
            # Альтернативный метод: весь текст
            text = content_div.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Применяем очистку
        original_length = len(text) if text else 0
        text = clean_extracted_text(text)

        if not text:
            print(f"  Текст не извлечен")
            return None

        print(f"  Символов: {len(text)}")
        return text

    except Exception as e:
        print(f"  Ошибка: {type(e).__name__}: {str(e)[:100]}")
        return None

def main():
    """Основная функция"""
    BASE_URL = "https://7kingdoms.ru/wiki/"
    INPUT_FILE = "characters.txt"
    OUTPUT_DIR = "characters_clean"

    print("=" * 60)
    print("ПАРСЕР С ПРАВИЛЬНОЙ ОБРАБОТКОЙ ПРОБЕЛОВ")
    print("=" * 60)
    print("Удаляются: ссылки [1], раздел 'Источники', копирайты")
    print("=" * 60)

    # Читаем персонажей
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            characters = [line.strip() for line in f if line.strip()]

        if not characters:
            print(f"Файл {INPUT_FILE} пуст.")
            return

        print(f"Загружено {len(characters)} персонажей")

    except FileNotFoundError:
        print(f"Файл {INPUT_FILE} не найден.")
        return

    # Создаем директорию
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Обрабатываем персонажей
    successful = 0

    for i, character in enumerate(characters, 1):
        print(f"\n[{i}/{len(characters)}] Обработка: {character}")

        text = get_character_text(character, BASE_URL)

        if text:
            # Сохраняем файл
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', character)
            safe_name = re.sub(r'\s+', '_', safe_name.strip())
            filename = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"  ✓ Сохранено: {filename}")
            successful += 1
        else:
            print(f"  ✗ Не удалось получить текст")

        # Пауза между запросами
        if i < len(characters):
            time.sleep(2)

    print(f"\n" + "=" * 60)
    print(f"ГОТОВО! Успешно: {successful}/{len(characters)}")
    print(f"Файлы сохранены в папке: {OUTPUT_DIR}")
    print("=" * 60)

def repair_existing_files():
    """Исправление уже скачанных файлов с проблемой пробелов"""
    directory = input("Введите путь к папке с файлами для исправления: ").strip()

    if not os.path.exists(directory):
        print(f"Папка {directory} не существует.")
        return

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Исправляем разбитые слова
                # 1. Удаляем пробелы между каждой буквой в словах
                lines = content.split('\n')
                fixed_lines = []

                for line in lines:
                    # Если в строке есть смысловой текст, оставляем как есть
                    # (пробелы между словами уже должны быть правильными)
                    if line.strip():
                        # Просто убираем лишние пробелы
                        line = re.sub(r'\s+', ' ', line).strip()
                        fixed_lines.append(line)

                fixed_content = '\n'.join(fixed_lines)

                # 2. Удаляем раздел "Источники" если он есть
                fixed_content = re.sub(r'\nИсточники\s*\n.*', '', fixed_content, flags=re.DOTALL | re.IGNORECASE)

                # Сохраняем исправленный файл
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)

                print(f"Исправлен: {filename}")

            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")

if __name__ == "__main__":
    # Запуск основного парсера
    main()

    # Если нужно исправить уже скачанные файлы, раскомментируйте:
    #repair_existing_files()