import argparse
import collections
import logging
import re
import threading
from typing import Dict, List

import matplotlib.pyplot as plt
import requests
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

# Налаштування логування
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s:%(message)s",
    filename="word_frequency.log",
    filemode="w",
)

# Отримання стоп-слів для англійської мови
STOP_WORDS = set(stopwords.words("english"))

def download_text(url: str) -> str:
    """
    Завантажує текст з заданої URL-адреси.

    :param url: URL-адреса для завантаження тексту.
    :return: Завантажений текст.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        response.encoding = response.apparent_encoding  # Визначення кодування
        return response.text
    except requests.RequestException as e:
        logging.error(f"Помилка при завантаженні тексту з URL {url}: {e}")
        raise SystemExit(f"Не вдалося завантажити текст з URL {url}. Деталі в логах.")


def map_worker(
    text_segment: str,
    partial_counts: Dict[str, int],
    lock: threading.Lock,
    stop_words: set = STOP_WORDS,
) -> None:
    """
    Обробляє сегмент тексту для підрахунку частоти слів.

    :param text_segment: Частина тексту для обробки.
    :param partial_counts: Словник для зберігання часткових результатів.
    :param lock: Блокування для синхронізації доступу до словника.
    """
    # Використання регулярного виразу для виділення слів
    words = re.findall(r"\b\w+\b", text_segment.lower())
    # Фільтрація стоп-слів
    words = [word for word in words if word not in stop_words]

    local_counter = collections.Counter(words)

    with lock:
        for word, count in local_counter.items():
            partial_counts[word] += count


def map_reduce(text: str, num_threads: int = 4) -> Dict[str, int]:
    """
    Виконує аналіз частоти слів за допомогою парадигми MapReduce з використанням багатопотоковості.

    :param text: Текст для аналізу.
    :param num_threads: Кількість потоків.
    :return: Словник з частотою використання слів.
    """
    # Розділення тексту на сегменти для кожного потоку
    text_length = len(text)
    segment_length = text_length // num_threads
    segments = [
        (
            text[i * segment_length : (i + 1) * segment_length]
            if i != num_threads - 1
            else text[i * segment_length :]
        )
        for i in range(num_threads)
    ]

    partial_counts = collections.defaultdict(int)
    lock = threading.Lock()
    threads = []

    for segment in segments:
        thread = threading.Thread(
            target=map_worker, args=(segment, partial_counts, lock)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return dict(partial_counts)


def visualize_top_words(word_counts: Dict[str, int], top_n: int = 10) -> None:
    """
    Візуалізує топ-слова за частотою використання у вигляді графіка.

    :param word_counts: Словник з частотою використання слів.
    :param top_n: Кількість топ-слів для відображення.
    """
    # Визначення топ-слів
    top_words = collections.Counter(word_counts).most_common(top_n)
    words, counts = zip(*top_words) if top_words else ([], [])

    plt.figure(figsize=(12, 8))
    bars = plt.bar(words, counts, color="skyblue")
    plt.xlabel("Слова", fontsize=14)
    plt.ylabel("Частота використання", fontsize=14)
    plt.title(f"Top {top_n} Слів за Частотою Використання", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Додавання значень над стовпцями
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.show()


def main():
    # Парсер аргументів командного рядка
    parser = argparse.ArgumentParser(
        description="Аналіз частоти слів за допомогою MapReduce."
    )
    parser.add_argument("url", type=str, help="URL-адреса для завантаження тексту.")
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Кількість топ-слів для візуалізації (за замовчуванням 10).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Кількість потоків для обробки (за замовчуванням 4).",
    )
    args = parser.parse_args()

    # Завантаження тексту
    text = download_text(args.url)

    # Аналіз частоти слів
    word_counts = map_reduce(text, num_threads=args.threads)

    # Візуалізація результатів
    visualize_top_words(word_counts, top_n=args.top)


if __name__ == "__main__":
    main()
