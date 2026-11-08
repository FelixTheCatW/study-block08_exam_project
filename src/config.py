"""Project configuration for Block 8 exam project.

Тема: консольный мини-помощник по учебным документам (RAG / поиск).

Проект повторяет обязательную часть ноутбука Block08_RAG_Text_TODO:
- документы разбиваются на чанки;
- TF-IDF находит ближайшие фрагменты;
- ответ собирается из точных предложений-цитат с указанием источника;
- сравниваются несколько настроек по метрикам Top-1 и Hit@K.

Изменены четыре параметра относительно исходного ноутбука:
CHUNK_SIZE, OVERLAP, SEARCH_METHOD, TOP_K. Добавлены пятый эксперимент E,
свой документ STUDENT и вопрос без подтверждённого ответа.
"""
from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
REPORTS_DIR = BASE_DIR / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


DOCUMENTS_PATH = DATA_DIR / "documents.json"
CHECK_QUESTIONS_PATH = DATA_DIR / "check_questions.json"


EXPERIMENTS_CSV = REPORTS_DIR / "experiments.csv"
RETRIEVAL_CHECK_CSV = REPORTS_DIR / "retrieval_check.csv"
SETTINGS_JSON = REPORTS_DIR / "settings.json"
DEMO_ANSWER_TXT = REPORTS_DIR / "demo_answer.txt"
CUSTOM_ANSWER_TXT = REPORTS_DIR / "custom_answer.txt"
NO_ANSWER_TXT = REPORTS_DIR / "no_answer.txt"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.md"
EXPERIMENTS_CHART = CHARTS_DIR / "experiments_metrics.png"
CHUNKS_CHART = CHARTS_DIR / "chunks_per_experiment.png"


RANDOM_STATE = 42

# Рабочие (изменённые) параметры обязательной части.
CHUNK_SIZE = 230
OVERLAP = 30
SEARCH_METHOD = "char_wb"
TOP_K = 3

# Пороги для «ответа с источниками».
MIN_SIMILARITY = 0.08
SENTENCE_MIN_SIMILARITY = 0.10
RELATIVE_SIMILARITY = 0.65
MAX_SENTENCES = 2

# Заведомо посторонний вопрос: ответа в базе нет.
UNKNOWN_QUESTION = "Сколько студенту нужно отжаться на брусьях?"
UNKNOWN_MIN_SIMILARITY = 0.20

# Свой документ и свой вопрос к нему.
CUSTOM_DOCUMENT_ID = "STUDENT"
CUSTOM_QUESTION = "Какой флаг включает интерактивный режим консольного помощника?"

DEMO_QUESTION = "Как делить данные при прогнозе будущих событий?"


# Пять экспериментов: A-D из ноутбука и новый E.
EXPERIMENTS = [
    {
        "name": "A: слова / 1 источник",
        "chunk_size": 380,
        "overlap": 45,
        "analyzer": "word",
        "top_k": 1,
    },
    {
        "name": "B: буквы / 2 источника",
        "chunk_size": 380,
        "overlap": 45,
        "analyzer": "char_wb",
        "top_k": 2,
    },
    {
        "name": "C: короткие чанки",
        "chunk_size": 200,
        "overlap": 30,
        "analyzer": "char_wb",
        "top_k": 3,
    },
    {
        "name": "D: длинные чанки",
        "chunk_size": 700,
        "overlap": 100,
        "analyzer": "char_wb",
        "top_k": 3,
    },
    {
        "name": "E: слова / без перекрытия / 5 источников",
        "chunk_size": 300,
        "overlap": 0,
        "analyzer": "word",
        "top_k": 5,
    },
]


# Факультатив (нужен интернет и установка пакетов).
NEURAL_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
