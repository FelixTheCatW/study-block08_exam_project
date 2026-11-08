"""Project configuration for Block 8 exam project.

Тема: консольный мини-помощник по учебным документам (RAG / поиск).
Единственный источник данных — база знаний `D:\\study\\handbook`.

- документы разбиваются на чанки;
- TF-IDF находит ближайшие фрагменты;
- ответ собирается из точных предложений-цитат с указанием источника;
- сравниваются несколько настроек по метрикам Top-1 и Hit@K.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.env_loader import load_env_file


BASE_DIR = Path(__file__).resolve().parent.parent
load_env_file(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


RANDOM_STATE = 42

# Рабочие параметры для корпуса Handbook.
CHUNK_SIZE = 600
OVERLAP = 100
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


# --- База знаний Handbook ----------------------------------------------
# Путь можно переопределить переменной окружения HANDBOOK_DIR.
HANDBOOK_SOURCE_DIR = Path(os.environ.get("HANDBOOK_DIR", r"D:\study\handbook"))
HANDBOOK_CORPUS_PATH = DATA_DIR / "handbook_documents.json"
HANDBOOK_STATS_PATH = REPORTS_DIR / "handbook_corpus_stats.json"
HANDBOOK_CHECK_QUESTIONS_PATH = DATA_DIR / "handbook_check_questions.json"

HANDBOOK_REPORTS_DIR = REPORTS_DIR / "handbook"
HANDBOOK_CHARTS_DIR = HANDBOOK_REPORTS_DIR / "charts"
HANDBOOK_EXPERIMENTS_CSV = HANDBOOK_REPORTS_DIR / "experiments.csv"
HANDBOOK_RETRIEVAL_CHECK_CSV = HANDBOOK_REPORTS_DIR / "retrieval_check.csv"
HANDBOOK_SETTINGS_JSON = HANDBOOK_REPORTS_DIR / "settings.json"
HANDBOOK_DEMO_ANSWER_TXT = HANDBOOK_REPORTS_DIR / "demo_answer.txt"
HANDBOOK_CUSTOM_ANSWER_TXT = HANDBOOK_REPORTS_DIR / "custom_answer.txt"
HANDBOOK_NO_ANSWER_TXT = HANDBOOK_REPORTS_DIR / "no_answer.txt"
HANDBOOK_FINAL_REPORT_PATH = HANDBOOK_REPORTS_DIR / "final_report.md"
HANDBOOK_EXPERIMENTS_CHART = HANDBOOK_CHARTS_DIR / "experiments_metrics.png"
HANDBOOK_CHUNKS_CHART = HANDBOOK_CHARTS_DIR / "chunks_per_experiment.png"
HANDBOOK_CORPUS_COMPOSITION_CHART = (
    HANDBOOK_CHARTS_DIR / "corpus_composition.png"
)
HANDBOOK_DOCUMENT_SIZES_CHART = HANDBOOK_CHARTS_DIR / "document_sizes.png"
HANDBOOK_TOPK_GAIN_CHART = HANDBOOK_CHARTS_DIR / "topk_gain.png"

HANDBOOK_DEMO_QUESTION = "Что такое переобучение и как его распознать?"
HANDBOOK_CUSTOM_QUESTION = (
    "Почему для KMeans обязательно масштабирование признаков?"
)
HANDBOOK_UNKNOWN_QUESTION = UNKNOWN_QUESTION


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


# DeepSeek API (OpenAI-совместимый endpoint).
# Ключ читается из переменной окружения или локального .env и не коммитится.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_URL = os.environ.get(
    "DEEPSEEK_API_URL",
    "https://api.deepseek.com/chat/completions",
)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TIMEOUT = int(os.environ.get("DEEPSEEK_TIMEOUT", "90"))
