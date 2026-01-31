import os
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# Конфигурация
CHROMA_HOST = "chromadb"
CHROMA_PORT = 8000
MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "docs"
DATA_DIR = "/app/data"
CACHE_DIR = "/app/cache"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Создаем кэш директорию если нет
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def load_model_cached():
    """Загрузить модель с кэшированием на диск"""
    model_cache_path = os.path.join(CACHE_DIR, "sentence_transformers")

    # Создаем кэш директорию для sentence-transformers
    os.makedirs(model_cache_path, exist_ok=True)

    print(f"Загрузка модели {MODEL_NAME}...")
    start_time = time.time()

    try:
        # Пробуем загрузить модель с использованием кэша
        model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=model_cache_path,
            device='cpu'  # явно указываем CPU
        )

        load_time = time.time() - start_time
        print(f"Модель загружена за {load_time:.2f} секунд")

        # Сохраняем информацию о загрузке модели
        model_info_path = os.path.join(CACHE_DIR, "model_info.json")
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump({
                "model": MODEL_NAME,
                "loaded_at": datetime.now().isoformat(),
                "load_time_seconds": load_time,
                "cache_path": model_cache_path,
                "dimension": model.get_sentence_embedding_dimension()
            }, f, indent=2)

        return model

    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        # Пробуем загрузить без кэша как запасной вариант
        return SentenceTransformer(MODEL_NAME, device='cpu')

def load_and_chunk_files():
    """Загрузить файлы и разбить на чанки"""
    all_chunks = []

    print("Чтение файлов из", DATA_DIR)

    if not os.path.exists(DATA_DIR):
        print(f"Создаю директорию: {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]

    if not files:
        print(f"В {DATA_DIR} нет .txt файлов!")
        print(f"Поместите текстовые файлы в папку {DATA_DIR}")
        return all_chunks

    print(f"Найдено {len(files)} файлов")

    # Инициализируем сплиттер
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        keep_separator=True
    )

    total_text_length = 0

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            total_text_length += len(text)

            if not text:
                print(f"  {filename}: ПУСТОЙ файл, пропускаем")
                continue

            # Используем LangChain для разбиения
            chunks = text_splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "source": filename,
                    "chunk_id": f"{filename}_{i:04d}",
                    "chunk_num": i,
                    "char_count": len(chunk),
                    "word_count": len(chunk.split())
                })

            print(f"  {filename}: {len(chunks)} чанков, {len(text)} символов")

        except Exception as e:
            print(f"  Ошибка чтения {filename}: {e}")

    # Сохраняем метаданные чанков
    if all_chunks:
        metadata_path = os.path.join(CACHE_DIR, "chunks_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            metadata = [{
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "chunk_num": chunk["chunk_num"],
                "char_count": chunk["char_count"],
                "word_count": chunk["word_count"]
            } for chunk in all_chunks]
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\nОбщая статистика:")
        print(f"  Всего файлов: {len(files)}")
        print(f"  Всего чанков: {len(all_chunks)}")
        print(f"  Общий объем текста: {total_text_length:,} символов")
        print(f"  Средний размер чанка: {total_text_length // len(all_chunks):,} символов")

    return all_chunks

def create_index():
    """Создать векторный индекс"""
    print("=" * 60)
    print("🚀 Создание векторного индекса")
    print("=" * 60)

    total_start_time = time.time()
    stage_times = {}

    # Этап 1: Загрузка и чанкинг файлов
    stage_start = time.time()
    print("\n📁 1. Загрузка и обработка файлов...")
    chunks = load_and_chunk_files()
    stage_times["file_loading"] = time.time() - stage_start

    if not chunks:
        return

    # Этап 2: Загрузка модели
    stage_start = time.time()
    print("\n🤖 2. Загрузка модели эмбеддингов...")
    model = load_model_cached()
    if not model:
        return
    stage_times["model_loading"] = time.time() - stage_start

    # Этап 3: Подключение к ChromaDB
    stage_start = time.time()
    print("\n💾 3. Подключение к ChromaDB...")
    try:
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=chromadb.config.Settings(allow_reset=True)
        )
        print("   ✅ Подключено успешно!")
    except Exception as e:
        print(f"   ❌ Ошибка подключения: {e}")
        return
    stage_times["chroma_connection"] = time.time() - stage_start

    # Этап 4: Настройка коллекции
    stage_start = time.time()
    print("\n🗃️  4. Настройка коллекции...")
    try:
        # Удаляем старую коллекцию если существует
        existing_collections = client.list_collections()
        for col in existing_collections:
            if col.name == COLLECTION_NAME:
                client.delete_collection(COLLECTION_NAME)
                print(f"   🔄 Удалена старая коллекция '{COLLECTION_NAME}'")
                break
    except:
        pass

    try:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "База знаний из текстовых файлов",
                "model": MODEL_NAME,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "created_at": datetime.now().isoformat()
            }
        )
        print(f"   ✅ Создана коллекция '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"   ❌ Ошибка создания коллекции: {e}")
        return
    stage_times["collection_setup"] = time.time() - stage_start

    # Этап 5: Создание эмбеддингов
    stage_start = time.time()
    print(f"\n⚡ 5. Создание эмбеддингов для {len(chunks)} чанков...")
    print("   Это может занять некоторое время...")

    batch_size = 32
    processed = 0
    errors = 0

    for i in range(0, len(chunks), batch_size):
        batch_start = time.time()
        batch = chunks[i:i + batch_size]

        try:
            batch_texts = [chunk["text"] for chunk in batch]
            ids = [chunk["chunk_id"] for chunk in batch]

            # Создаем эмбеддинги с префиксом для E5
            texts_for_embedding = [f"passage: {text}" for text in batch_texts]

            embeddings = model.encode(
                texts_for_embedding,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=8  # Меньший размер батча для надежности
            )

            # Подготавливаем метаданные
            metadatas = [{
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "chunk_num": chunk["chunk_num"],
                "char_count": chunk["char_count"],
                "word_count": chunk["word_count"],
                "timestamp": datetime.now().isoformat()
            } for chunk in batch]

            # Добавляем в коллекцию
            collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=metadatas,
                ids=ids
            )

            processed += len(batch)
            batch_time = time.time() - batch_start

            # Прогресс
            progress = processed / len(chunks) * 100
            print(f"   📊 Обработано: {processed}/{len(chunks)} чанков ({progress:.1f}%) | "
                  f"Батч: {batch_time:.2f} сек")

        except Exception as e:
            errors += len(batch)
            print(f"   ❌ Ошибка в батче {i//batch_size}: {e}")

    stage_times["embedding_creation"] = time.time() - stage_start

    # Этап 6: Проверка и сохранение результатов
    stage_start = time.time()
    print("\n✅ 6. Завершение создания индекса...")

    try:
        count = collection.count()
        print(f"   📈 В коллекции: {count} документов")

        # Сохраняем детальную статистику
        stats = {
            "collection": COLLECTION_NAME,
            "total_documents": count,
            "total_chunks": len(chunks),
            "processed": processed,
            "errors": errors,
            "model": MODEL_NAME,
            "model_dimension": model.get_sentence_embedding_dimension(),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "created_at": datetime.now().isoformat(),
            "stage_times": stage_times,
            "total_time": time.time() - total_start_time,
            "files_processed": len([f for f in os.listdir(DATA_DIR) if f.endswith('.txt')])
        }

        # Сохраняем статистику
        stats_path = os.path.join(CACHE_DIR, "index_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

        print(f"   💾 Статистика сохранена: {stats_path}")

    except Exception as e:
        print(f"   ⚠️  Ошибка при сохранении статистики: {e}")

    stage_times["finalization"] = time.time() - stage_start
    total_time = time.time() - total_start_time

    # Финальный отчет
    print("\n" + "=" * 60)
    print("🎉 ИНДЕКС УСПЕШНО СОЗДАН!")
    print("=" * 60)

    print(f"\n📊 ОТЧЕТ О ВЫПОЛНЕНИИ:")
    print(f"   Общее время: {total_time:.2f} секунд ({total_time/60:.2f} минут)")
    print(f"\n   ⏱️  Время по этапам:")
    for stage_name, stage_time in stage_times.items():
        stage_percent = (stage_time / total_time) * 100
        readable_name = {
            "file_loading": "Загрузка файлов",
            "model_loading": "Загрузка модели",
            "chroma_connection": "Подключение к ChromaDB",
            "collection_setup": "Настройка коллекции",
            "embedding_creation": "Создание эмбеддингов",
            "finalization": "Завершение"
        }.get(stage_name, stage_name)

        print(f"     • {readable_name}: {stage_time:.2f} сек ({stage_percent:.1f}%)")

    print(f"\n   📈 Статистика:")
    print(f"     • Всего файлов: {stats.get('files_processed', 0)}")
    print(f"     • Всего чанков: {len(chunks)}")
    print(f"     • Успешно обработано: {processed}")
    print(f"     • Ошибок: {errors}")
    print(f"     • Модель: {MODEL_NAME}")
    print(f"     • Размерность: {model.get_sentence_embedding_dimension()}")
    print(f"\n   🔗 ChromaDB доступен по: http://localhost:8000")
    print(f"   📁 Коллекция: '{COLLECTION_NAME}'")
    print("=" * 60)

    # Сохраняем краткий отчет в отдельный файл
    summary_path = os.path.join(CACHE_DIR, "summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ОТЧЕТ О СОЗДАНИИ ВЕКТОРНОГО ИНДЕКСА\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Общее время: {total_time:.2f} секунд\n")
        f.write(f"Чанков создано: {len(chunks)}\n")
        f.write(f"Модель: {MODEL_NAME}\n")
        f.write(f"Коллекция: {COLLECTION_NAME}\n")

    print(f"\n📄 Краткий отчет сохранен: {summary_path}")

if __name__ == "__main__":
    try:
        create_index()
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()