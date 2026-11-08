"""Block 8 exam project: консольный помощник по базе знаний Handbook.

Единственный источник данных — `D:\\study\\handbook`.

Запуск:
    python main.py --build-corpus
    python main.py

Режимы:
    python main.py --ask
    python main.py --question "Что такое переобучение?"
    python main.py --neural
    python main.py --llm

Порядок полного сценария:
1. Загрузка корпуса Handbook и контрольных вопросов.
2. Построение TF-IDF индекса.
3. Демонстрационный вопрос, контрольный вопрос и случай без ответа.
4. Проверка поиска: Top-1 и Hit@K.
5. Пять экспериментов с разными чанками и способом поиска.
6. Сохранение CSV/JSON/TXT, графиков и итогового отчёта.
"""
from __future__ import annotations

import argparse
import sys

from src.config import (
    CHUNK_SIZE,
    HANDBOOK_CHARTS_DIR,
    HANDBOOK_CHECK_QUESTIONS_PATH,
    HANDBOOK_CHUNKS_CHART,
    HANDBOOK_CORPUS_PATH,
    HANDBOOK_CUSTOM_ANSWER_TXT,
    HANDBOOK_CUSTOM_QUESTION,
    HANDBOOK_DEMO_ANSWER_TXT,
    HANDBOOK_DEMO_QUESTION,
    HANDBOOK_EXPERIMENTS_CHART,
    HANDBOOK_EXPERIMENTS_CSV,
    HANDBOOK_FINAL_REPORT_PATH,
    HANDBOOK_NO_ANSWER_TXT,
    HANDBOOK_REPORTS_DIR,
    HANDBOOK_RETRIEVAL_CHECK_CSV,
    HANDBOOK_SETTINGS_JSON,
    HANDBOOK_SOURCE_DIR,
    HANDBOOK_STATS_PATH,
    HANDBOOK_UNKNOWN_QUESTION,
    OVERLAP,
    SEARCH_METHOD,
    TOP_K,
    UNKNOWN_MIN_SIMILARITY,
)
from src.corpus_builder import HandbookCorpusBuilder
from src.data_loader import DataLoader
from src.pipeline import (
    CorpusConfig,
    run_corpus_pipeline,
    run_interactive,
    run_single_question,
)


def handbook_config() -> CorpusConfig:
    """Настройки единственного корпуса — базы знаний Handbook."""
    return CorpusConfig(
        name="handbook",
        documents_path=HANDBOOK_CORPUS_PATH,
        check_questions_path=HANDBOOK_CHECK_QUESTIONS_PATH,
        reports_dir=HANDBOOK_REPORTS_DIR,
        charts_dir=HANDBOOK_CHARTS_DIR,
        experiments_csv=HANDBOOK_EXPERIMENTS_CSV,
        retrieval_check_csv=HANDBOOK_RETRIEVAL_CHECK_CSV,
        settings_json=HANDBOOK_SETTINGS_JSON,
        demo_answer_txt=HANDBOOK_DEMO_ANSWER_TXT,
        custom_answer_txt=HANDBOOK_CUSTOM_ANSWER_TXT,
        no_answer_txt=HANDBOOK_NO_ANSWER_TXT,
        final_report_path=HANDBOOK_FINAL_REPORT_PATH,
        experiments_chart=HANDBOOK_EXPERIMENTS_CHART,
        chunks_chart=HANDBOOK_CHUNKS_CHART,
        demo_question=HANDBOOK_DEMO_QUESTION,
        custom_question=HANDBOOK_CUSTOM_QUESTION,
        unknown_question=HANDBOOK_UNKNOWN_QUESTION,
        unknown_min_similarity=UNKNOWN_MIN_SIMILARITY,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        search_method=SEARCH_METHOD,
        top_k=TOP_K,
        report_context={
            "title": (
                "Блок 8. Мини-помощник по базе знаний Handbook "
                "(RAG / TF-IDF)"
            ),
            "subtitle": (
                "Поиск по учебным конспектам и Word-документам из "
                "`D:\\study\\handbook`. Корпус собран локально: точные "
                "дубли удалены, служебные и потенциально чувствительные "
                "файлы исключены."
            ),
            "source_note": (
                "Корпус собран из тематических md-конспектов, "
                "`konspekty/` и `istochniki/`. В него не включены PDF, "
                "изображения и сырые данные."
            ),
            "custom_section_title": "5. Контрольный вопрос по Handbook",
            "custom_section_note": (
                "Проверяем, что система находит нужный раздел базы знаний "
                "по вопросу о масштабировании признаков для KMeans."
            ),
            "charts_note": (
                "Графики: `reports/handbook/charts/experiments_metrics.png` "
                "(Top-1 и Hit@K) и "
                "`reports/handbook/charts/chunks_per_experiment.png` "
                "(число чанков)."
            ),
            "run_commands": (
                "python main.py --build-corpus\n"
                "python main.py\n"
                "python main.py --ask\n"
                "python main.py --question \"Что такое переобучение?\"\n"
                "python -m pytest -q"
            ),
            "artifacts_note": (
                "- `reports/handbook/experiments.csv` — таблица пяти опытов;\n"
                "- `reports/handbook/retrieval_check.csv` — проверка поиска;\n"
                "- `reports/handbook/settings.json` — параметры запуска;\n"
                "- `reports/handbook/demo_answer.txt` — демонстрационный ответ;\n"
                "- `reports/handbook/custom_answer.txt` — контрольный ответ;\n"
                "- `reports/handbook/no_answer.txt` — пример без ответа;\n"
                "- `reports/handbook/charts/*.png` — графики;\n"
                "- `reports/handbook/final_report.md` — этот отчёт;\n"
                "- `reports/handbook_corpus_stats.json` — статистика сборки корпуса."
            ),
        },
        limitations=[
            "Корпус собран автоматически из md/docx; PDF, изображения и сырые данные не включены.",
            "В Handbook есть пересекающиеся конспекты и первоисточники, поэтому один ответ может находиться в нескольких документах.",
            "TF-IDF не понимает синонимы без общих слов.",
            "Ответ — точные цитаты, а не генерация текста языковой моделью.",
            "Параметры чанков и порог сходства нужно настраивать на своих вопросах.",
            "Перед публикацией корпуса и отчётов нужно проверить права и персональные данные.",
        ],
    )


def build_handbook_corpus() -> None:
    """Собрать корпус Handbook и вывести статистику."""
    print("Сборка корпуса Handbook...")
    builder = HandbookCorpusBuilder(
        source_dir=HANDBOOK_SOURCE_DIR,
        output_path=HANDBOOK_CORPUS_PATH,
        stats_path=HANDBOOK_STATS_PATH,
    )
    documents, stats = builder.build()
    print(f"Документов в корпусе: {len(documents)}")
    print(f"Слов: {stats['total_words']}")
    print(f"Пропущено чувствительных/служебных: {stats['skipped_sensitive']}")
    print(f"Пропущено дублей: {stats['skipped_duplicates_count']}")
    print(f"Корпус: {HANDBOOK_CORPUS_PATH}")
    print(f"Статистика: {HANDBOOK_STATS_PATH}")


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Мини-помощник по базе знаний Handbook (TF-IDF / RAG)."
    )
    parser.add_argument(
        "--build-corpus",
        action="store_true",
        help="собрать корпус Handbook из D:\\study\\handbook и выйти",
    )
    parser.add_argument(
        "--ask",
        action="store_true",
        help="интерактивный режим вопросов в терминале",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="ответить на один вопрос и завершиться",
    )
    parser.add_argument(
        "--neural",
        action="store_true",
        help="факультативно сравнить с нейросетевым поиском",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="факультативно сгенерировать ответ локальной LLM",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа консольного приложения."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()

    if args.build_corpus:
        build_handbook_corpus()
        return

    config = handbook_config()
    if not config.documents_path.exists():
        build_handbook_corpus()

    documents = DataLoader(config.documents_path).load_documents()

    if args.question:
        run_single_question(documents, config, args.question)
        return
    if args.ask:
        run_interactive(documents, config)
        return

    run_corpus_pipeline(
        config,
        run_neural=args.neural,
        run_llm=args.llm,
    )


if __name__ == "__main__":
    main()
