#!/usr/bin/env python3
"""
Скрипт для проверки окружения
"""

import os
import sys

print("Проверка окружения RAG бота...")
print("="*50)

# Проверка папок
folders = [
    ("/app/models", "Папка с моделями GGUF"),
    ("/app/cache", "Папка для кэша"),
    ("/app/cache/sentence_transformers", "Кэш эмбеддингов")
]

for folder, description in folders:
    if os.path.exists(folder):
        print(f"✓ {description}: {folder}")
        # Показываем содержимое папки models
        if "models" in folder:
            files = os.listdir(folder)
            if files:
                print(f"  Файлы: {', '.join(files)}")
            else:
                print("  ⚠️  Папка пуста!")
    else:
        print(f"✗ {description}: НЕ СУЩЕСТВУЕТ")

# Проверка наличия .gguf файлов
model_dir = "/app/models"
if os.path.exists(model_dir):
    gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
    if gguf_files:
        print(f"\n✓ Найдены GGUF файлы: {len(gguf_files)}")
        for f in gguf_files[:3]:  # Показываем первые 3
            print(f"  • {f}")
    else:
        print(f"\n✗ В папке {model_dir} нет .gguf файлов!")
        print("  Поместите модель GGUF в эту папку")
        sys.exit(1)
else:
    print(f"\n✗ Папка {model_dir} не существует!")

print("\n" + "="*50)
print("Если все проверки пройдены, запускайте:")
print("  docker-compose run --rm rag-bot")