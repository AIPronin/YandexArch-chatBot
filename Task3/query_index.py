import os
import sys
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb

# Конфигурация
CHROMA_HOST = "chromadb"
CHROMA_PORT = 8000
MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "docs"
CACHE_DIR = "/app/cache"

# Создаем кэш директорию если нет
os.makedirs(CACHE_DIR, exist_ok=True)

# Глобальная переменная для кэширования модели
_global_model = None
_model_load_time = None

def load_model_once():
    """Загрузить модель один раз и кэшировать в памяти"""
    global _global_model, _model_load_time

    if _global_model is not None:
        return _global_model, _model_load_time

    model_cache_path = os.path.join(CACHE_DIR, "sentence_transformers")
    os.makedirs(model_cache_path, exist_ok=True)

    print("🔄 Загрузка модели эмбеддингов...")
    start_time = time.time()

    try:
        # Используем кэш на диске
        _global_model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=model_cache_path,
            device='cpu'
        )

        load_time = time.time() - start_time
        _model_load_time = load_time

        print(f"✅ Модель загружена за {load_time:.2f} секунд")

        # Сохраняем время загрузки
        model_info = {
            "model": MODEL_NAME,
            "loaded_at": datetime.now().isoformat(),
            "load_time": load_time,
            "dimension": _global_model.get_sentence_embedding_dimension(),
            "cached_in_memory": True
        }

        with open(os.path.join(CACHE_DIR, "query_model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)

    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        # Пробуем загрузить без кэша
        _global_model = SentenceTransformer(MODEL_NAME, device='cpu')
        _model_load_time = time.time() - start_time

    return _global_model, _model_load_time

def safe_query_embedding(query):
    """Безопасное создание эмбеддинга для запроса"""
    model, _ = load_model_once()

    if not isinstance(query, str):
        query = str(query)

    query = query.strip()

    if not query:
        return [0.0] * 384  # размерность e5-small

    try:
        # Для E5 моделей нужен префикс "query: "
        query_for_embedding = f"query: {query}"

        embedding = model.encode(
            [query_for_embedding],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embedding.tolist()[0]

    except Exception as e:
        print(f"⚠️  Ошибка создания эмбеддинга: {e}")
        return [0.0] * 384

def search(query, n_results=5):
    """Поиск по индексу"""
    search_start = time.time()

    try:
        # Подключаемся к ChromaDB
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Ошибка подключения к ChromaDB: {e}")
        return None

    # Создаем эмбеддинг запроса
    query_embedding = safe_query_embedding(query)

    if not query_embedding:
        return None

    # Ищем похожие документы
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        search_time = time.time() - search_start
        results["search_time"] = search_time
        results["query"] = query

        return results
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return None

def print_results(results, query):
    """Вывод результатов"""
    if not results or not results.get('documents') or not results['documents'][0]:
        print("🤷 Ничего не найдено")
        return

    search_time = results.get('search_time', 0)

    print(f"\n🔍 Результаты поиска: '{query}'")
    print(f"⏱️  Время поиска: {search_time:.3f} сек")
    print("=" * 60)

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i] if results['metadatas'] else {}
        distance = results['distances'][0][i] if results['distances'] else 0

        relevance = max(0, 100 - (distance * 100))

        print(f"\n📄 Результат {i+1} [{relevance:.1f}%]")
        print(f"📂 Файл: {meta.get('source', 'Неизвестно')}")
        print(f"🔢 Чанк: {meta.get('chunk_num', '?')}")

        if len(doc) > 300:
            cut_pos = doc[:300].rfind('. ')
            if cut_pos > 200:
                text = doc[:cut_pos + 1] + "..."
            else:
                text = doc[:300] + "..."
        else:
            text = doc

        print(f"\n📝 Текст:\n{'-'*40}")
        print(text)
        print(f"{'-'*40}")

def check_index_status():
    """Проверить статус индекса"""
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collections = client.list_collections()

        if COLLECTION_NAME in [col.name for col in collections]:
            collection = client.get_collection(COLLECTION_NAME)
            count = collection.count()
            return True, count
        return False, 0
    except:
        return False, 0

def print_system_info():
    """Вывести информацию о системе"""
    print("\n" + "=" * 60)
    print("🔧 СИСТЕМНАЯ ИНФОРМАЦИЯ")
    print("=" * 60)

    # Информация о модели
    model, load_time = load_model_once()
    if model:
        print(f"🤖 Модель: {MODEL_NAME}")
        print(f"📏 Размерность: {model.get_sentence_embedding_dimension()}")
        print(f"⏱️  Время загрузки: {load_time:.2f} сек")
        print(f"💾 В памяти: {'Да' if _global_model else 'Нет'}")

    # Информация об индексе
    has_index, doc_count = check_index_status()
    print(f"\n🗃️  Индекс: {'✅ Найден' if has_index else '❌ Не найден'}")
    if has_index:
        print(f"📊 Документов: {doc_count}")

    # Информация о кэше
    cache_files = []
    if os.path.exists(CACHE_DIR):
        for root, dirs, files in os.walk(CACHE_DIR):
            for file in files:
                if file.endswith('.json') or file.endswith('.txt'):
                    cache_files.append(file)

    if cache_files:
        print(f"\n💾 Файлов в кэше: {len(cache_files)}")
        for file in sorted(cache_files)[:5]:  # показываем первые 5
            print(f"  📄 {file}")
        if len(cache_files) > 5:
            print(f"  ... и еще {len(cache_files) - 5}")

    print("=" * 60)

if __name__ == "__main__":
    # Загружаем модель при запуске
    print_system_info()

    if len(sys.argv) > 1:
        # Командный режим
        query = " ".join(sys.argv[1:])
        results = search(query)
        if results:
            print_results(results, query)
    else:
        # Интерактивный режим
        print("\n🎯 ИНТЕРАКТИВНЫЙ ПОИСК")
        print("Команды: help, stats, exit, clear")
        print("-" * 40)

        while True:
            try:
                query = input("\n🔎 Запрос: ").strip()

                if query.lower() in ['exit', 'quit']:
                    print("👋 Выход...")
                    break

                if query.lower() == 'help':
                    print("\n📘 СПРАВКА:")
                    print("  • Просто введите ваш вопрос")
                    print("  • Команды: help, stats, exit, clear")
                    continue

                if query.lower() == 'stats':
                    print_system_info()
                    continue

                if query.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue

                if not query:
                    continue

                start_time = time.time()
                results = search(query)
                total_time = time.time() - start_time

                if results:
                    print_results(results, query)
                else:
                    print("❌ Ошибка при поиске")

            except KeyboardInterrupt:
                print("\n👋 Выход...")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")