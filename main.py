"""Block 8 exam project: консольный мини-помощник по учебным документам.

Запуск полного сценария:
    python main.py

Режимы:
    python main.py --ask
    python main.py --question "Что показывает precision?"
    python main.py --neural
    python main.py --llm

Порядок полного сценария:
1. Загрузка базы знаний и контрольных вопросов.
2. Построение TF-IDF индекса с изменёнными параметрами.
3. Демонстрационный вопрос, свой документ и случай без ответа.
4. Проверка поиска: Top-1 и Hit@K.
5. Пять экспериментов с разными чанками и способом поиска.
6. Сохранение CSV/JSON/TXT, графиков и итогового отчёта.
"""
from __future__ import annotations

import argparse
import sys

from src.assistant import RagAssistant
from src.config import (
    CHARTS_DIR,
    CHECK_QUESTIONS_PATH,
    CHUNK_SIZE,
    CHUNKS_CHART,
    CUSTOM_ANSWER_TXT,
    CUSTOM_QUESTION,
    DEMO_ANSWER_TXT,
    DEMO_QUESTION,
    DOCUMENTS_PATH,
    EXPERIMENTS_CHART,
    EXPERIMENTS_CSV,
    FINAL_REPORT_PATH,
    NO_ANSWER_TXT,
    OVERLAP,
    REPORTS_DIR,
    RETRIEVAL_CHECK_CSV,
    SEARCH_METHOD,
    SETTINGS_JSON,
    TOP_K,
    UNKNOWN_MIN_SIMILARITY,
    UNKNOWN_QUESTION,
)
from src.data_loader import DataLoader
from src.evaluator import SearchEvaluator
from src.experiments import ExperimentRunner
from src.report_builder import ReportBuilder
from src.visualizer import Visualizer


def build_assistant(documents: list[dict[str, str]]) -> RagAssistant:
    """Собрать помощника с рабочими параметрами из config."""
    return RagAssistant(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        search_method=SEARCH_METHOD,
        top_k=TOP_K,
    )


def print_result(assistant: RagAssistant, result: dict[str, object]) -> None:
    """Вывести результат поиска и ответ в консоль."""
    print(assistant.format_result(result))
    print()


def run_interactive(documents: list[dict[str, str]]) -> None:
    """Интерактивный режим: вопросы вводятся в терминале."""
    assistant = build_assistant(documents)
    print(
        f"Помощник готов. Документов: {len(documents)}, "
        f"чанков: {len(assistant.records)}. Введите 'exit' для выхода.\n"
    )
    while True:
        question = input("Ваш вопрос: ").strip()
        if question.lower() in {"exit", "quit", "выход"}:
            break
        if not question:
            continue
        print_result(assistant, assistant.ask(question))


def run_single_question(documents: list[dict[str, str]], question: str) -> None:
    """Ответить на один вопрос и завершиться."""
    assistant = build_assistant(documents)
    print_result(assistant, assistant.ask(question))


def run_neural_comparison(
    assistant: RagAssistant,
    check_questions: list[tuple[str, str]],
) -> None:
    """Факультативно сравнить нейросетевой поиск с TF-IDF."""
    from src.neural_retriever import NeuralRetriever

    if not NeuralRetriever.is_available():
        print(
            "Нейросетевой поиск пропущен: не установлен sentence-transformers. "
            "Установите пакет и запустите снова."
        )
        return

    try:
        neural = NeuralRetriever().build(assistant.records)
        neural_found = neural.search(DEMO_QUESTION, top_k=TOP_K)
        _, neural_summary = SearchEvaluator.check(
            neural, check_questions, top_k=TOP_K
        )
    except (ImportError, OSError) as exc:
        print("Нейросетевой поиск пропущен:", str(exc)[:250])
        return

    print("Нейросетевой поиск (эмбеддинги):")
    for number, item in enumerate(neural_found, 1):
        print(
            f"  {number}. {item['doc_id']} — {item['title']} "
            f"(сходство {float(item['score']):.3f})"
        )
    print(
        "Нейросеть: Top-1 = "
        f"{neural_summary['top1_accuracy']:.3f}, Hit@K = "
        f"{neural_summary['hit_at_k']:.3f}\n"
    )


def run_llm_demo(assistant: RagAssistant, question: str) -> None:
    """Факультативно сгенерировать ответ локальной LLM."""
    from src.llm_generator import LocalLLMGenerator

    if not LocalLLMGenerator.is_available():
        print(
            "Локальная LLM пропущена: не установлены transformers/torch. "
            "Обязательная часть работает без неё."
        )
        return

    found = assistant.search(question)
    try:
        generator = LocalLLMGenerator().load()
        answer = generator.generate(question, found)
    except (ImportError, OSError) as exc:
        print("Локальная LLM пропущена:", str(exc)[:250])
        return

    print("ОТВЕТ ЯЗЫКОВОЙ МОДЕЛИ (проверьте по источникам):\n")
    print(answer)


def run_full_pipeline(args: argparse.Namespace) -> None:
    """Полный сценарий: поиск, опыты, отчёты, графики."""
    for directory in (REPORTS_DIR, CHARTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Данные.
    print("1. Загрузка базы знаний и контрольных вопросов...")
    documents = DataLoader(DOCUMENTS_PATH).load_documents()
    check_questions = DataLoader.load_check_questions(CHECK_QUESTIONS_PATH)
    print(f"Документов: {len(documents)}, вопросов: {len(check_questions)}")

    # 2. Рабочий индекс.
    print("2. Построение TF-IDF индекса...")
    assistant = build_assistant(documents)
    print(
        f"CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={OVERLAP}, "
        f"SEARCH_METHOD={SEARCH_METHOD}, TOP_K={TOP_K}, "
        f"чанков={len(assistant.records)}"
    )

    # 3. Демонстрация: основной, свой и посторонний вопросы.
    print("3. Демонстрационные вопросы...")
    demo_result = assistant.ask(DEMO_QUESTION)
    print_result(assistant, demo_result)

    custom_result = assistant.ask(CUSTOM_QUESTION)
    print("Свой вопрос к своему документу:")
    print_result(assistant, custom_result)

    unknown_result = assistant.ask(
        UNKNOWN_QUESTION, min_similarity=UNKNOWN_MIN_SIMILARITY
    )
    print("Вопрос без подтверждённого ответа:")
    print_result(assistant, unknown_result)

    # 4. Проверка поиска.
    print("4. Проверка поиска по контрольным вопросам...")
    detail_df, summary = SearchEvaluator.check(
        assistant.retriever, check_questions, top_k=TOP_K
    )
    print(
        f"Top-1: {summary['top1_accuracy']:.3f} | "
        f"Hit@K: {summary['hit_at_k']:.3f}"
    )

    # 5. Эксперименты A-E.
    print("5. Пять экспериментов...")
    results_df, _ = ExperimentRunner(documents, check_questions).run()
    print(results_df.to_string(index=False))

    # 6. Сохранение артефактов.
    print("6. Сохранение CSV/JSON/TXT...")
    DataLoader.save_dataframe(results_df, EXPERIMENTS_CSV)
    DataLoader.save_dataframe(detail_df, RETRIEVAL_CHECK_CSV)
    DataLoader.save_text(
        "Вопрос: " + DEMO_QUESTION + "\n" + str(demo_result["answer"]),
        DEMO_ANSWER_TXT,
    )
    DataLoader.save_text(
        "Вопрос: " + CUSTOM_QUESTION + "\n" + str(custom_result["answer"]),
        CUSTOM_ANSWER_TXT,
    )
    DataLoader.save_text(
        "Вопрос: " + UNKNOWN_QUESTION + "\n" + str(unknown_result["answer"]),
        NO_ANSWER_TXT,
    )

    settings = {
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "search_method": SEARCH_METHOD,
        "top_k": TOP_K,
        "question": DEMO_QUESTION,
        "top1_accuracy": summary["top1_accuracy"],
        "hit_at_k": summary["hit_at_k"],
        "unknown_min_similarity": UNKNOWN_MIN_SIMILARITY,
        "custom_question": CUSTOM_QUESTION,
        "custom_source": custom_result["found"][0]["doc_id"]
        if custom_result["found"]
        else None,
        "documents": len(documents),
        "chunks": len(assistant.records),
        "note": (
            "TF-IDF: поиск и подтверждённые выдержки; "
            "LLM выключена по умолчанию"
        ),
    }
    DataLoader.save_json(settings, SETTINGS_JSON)

    # 7. Графики.
    print("7. Сохранение графиков...")
    visualizer = Visualizer(CHARTS_DIR)
    visualizer.save_experiment_metrics(
        results_df,
        filename=EXPERIMENTS_CHART.name,
        title="Сравнение настроек поиска (Top-1 и Hit@K)",
    )
    visualizer.save_chunks_per_experiment(
        results_df,
        filename=CHUNKS_CHART.name,
        title="Количество чанков при разных настройках",
    )

    # 8. Итоговый отчёт.
    print("8. Формирование итогового отчёта...")
    limitations = [
        "База знаний небольшая и учебная: 17 документов.",
        "TF-IDF не понимает синонимы без общих слов; это не нейросетевой поиск.",
        "Ответ собирается из цитат, а не генерируется языковой моделью.",
        "Порог сходства нужно настраивать на своих вопросах.",
        "Семантический поиск и локальная LLM — факультатив и требуют ресурсов.",
        "Найденная цитата не заменяет проверку прав доступа и содержания.",
    ]
    report_builder = ReportBuilder(FINAL_REPORT_PATH)
    report_text = report_builder.build_report(
        documents=documents,
        settings=settings,
        results_df=results_df,
        baseline_summary=summary,
        custom_result=custom_result,
        unknown_result=unknown_result,
        limitations=limitations,
    )
    report_builder.save(report_text)

    # 9. Факультативные сравнения.
    if args.neural:
        print("9. Факультатив: нейросетевой поиск...")
        run_neural_comparison(assistant, check_questions)
    if args.llm:
        print("10. Факультатив: локальная языковая модель...")
        run_llm_demo(assistant, DEMO_QUESTION)

    print("Готово.")
    print("Отчёт:", FINAL_REPORT_PATH)
    print("CSV:", EXPERIMENTS_CSV)
    print("Проверка поиска:", RETRIEVAL_CHECK_CSV)


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Мини-помощник по учебным документам (TF-IDF / RAG)."
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
    documents = DataLoader(DOCUMENTS_PATH).load_documents()

    if args.question:
        run_single_question(documents, args.question)
        return
    if args.ask:
        run_interactive(documents)
        return

    run_full_pipeline(args)


if __name__ == "__main__":
    main()
