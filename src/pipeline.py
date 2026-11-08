"""Reusable RAG pipeline for the Handbook corpus (блок 8).

Сценарий запускается на единственном корпусе — базе знаний
`D:\\study\\handbook`. Логика поиска, экспериментов и отчётов вынесена
из `main.py`, чтобы её можно было использовать в ноутбуке.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.assistant import RagAssistant
from src.data_loader import DataLoader
from src.evaluator import SearchEvaluator
from src.experiments import ExperimentRunner
from src.report_builder import ReportBuilder
from src.visualizer import Visualizer


@dataclass(frozen=True)
class CorpusConfig:
    """Настройки одного корпуса и его артефактов."""

    name: str
    documents_path: Path
    check_questions_path: Path
    reports_dir: Path
    charts_dir: Path
    experiments_csv: Path
    retrieval_check_csv: Path
    settings_json: Path
    demo_answer_txt: Path
    custom_answer_txt: Path
    no_answer_txt: Path
    final_report_path: Path
    experiments_chart: Path
    chunks_chart: Path
    demo_question: str
    custom_question: str
    unknown_question: str
    unknown_min_similarity: float
    chunk_size: int
    overlap: int
    search_method: str
    top_k: int
    report_context: dict[str, str]
    limitations: list[str]


def build_assistant(
    documents: list[dict[str, str]],
    config: CorpusConfig,
) -> RagAssistant:
    """Собрать помощника с параметрами конкретного корпуса."""
    return RagAssistant(
        documents,
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        search_method=config.search_method,
        top_k=config.top_k,
    )


def print_result(assistant: RagAssistant, result: dict[str, object]) -> None:
    """Вывести результат поиска и ответ в консоль."""
    print(assistant.format_result(result))
    print()


def run_interactive(
    documents: list[dict[str, str]],
    config: CorpusConfig,
    deepseek: bool = False,
) -> None:
    """Интерактивный режим: вопросы вводятся в терминале."""
    assistant = build_assistant(documents, config)
    print(
        f"Помощник готов. Корпус: {config.name}. "
        f"Документов: {len(documents)}, чанков: {len(assistant.records)}. "
        "Введите 'exit' для выхода.\n"
    )
    while True:
        question = input("Ваш вопрос: ").strip()
        if question.lower() in {"exit", "quit", "выход"}:
            break
        if not question:
            continue
        print_result(assistant, assistant.ask(question))
        if deepseek:
            answer = generate_with_deepseek(assistant, question)
            if answer:
                print("ОТВЕТ DEEPSEEK (проверьте по источникам):\n")
                print(answer)
                print()


def run_single_question(
    documents: list[dict[str, str]],
    config: CorpusConfig,
    question: str,
    deepseek: bool = False,
) -> None:
    """Ответить на один вопрос и завершиться."""
    assistant = build_assistant(documents, config)
    print_result(assistant, assistant.ask(question))
    if deepseek:
        answer = generate_with_deepseek(assistant, question)
        if answer:
            print("ОТВЕТ DEEPSEEK (проверьте по источникам):\n")
            print(answer)


def generate_with_deepseek(
    assistant: RagAssistant,
    question: str,
) -> str | None:
    """Сгенерировать ответ через DeepSeek по найденным источникам."""
    from src.deepseek_generator import DeepSeekGenerator

    if not DeepSeekGenerator.is_available():
        print(
            "DeepSeek пропущен: не задан DEEPSEEK_API_KEY. "
            "Добавьте ключ в локальный .env."
        )
        return None

    found = assistant.search(question)
    if not found:
        print("DeepSeek: нет найденных источников.")
        return None

    try:
        return DeepSeekGenerator().generate(question, found)
    except Exception as exc:  # noqa: BLE001 - CLI должен не падать
        print("DeepSeek API недоступен:", str(exc)[:300])
        return None


def run_neural_comparison(
    assistant: RagAssistant,
    check_questions: list[tuple[str, str | list[str]]],
    config: CorpusConfig,
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
        neural_found = neural.search(config.demo_question, top_k=config.top_k)
        _, neural_summary = SearchEvaluator.check(
            neural, check_questions, top_k=config.top_k
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


def run_deepseek_demo(assistant: RagAssistant, question: str) -> None:
    """Сгенерировать ответ DeepSeek по найденным источникам."""
    answer = generate_with_deepseek(assistant, question)
    if answer:
        print("ОТВЕТ DEEPSEEK (проверьте по источникам):\n")
        print(answer)


def run_corpus_pipeline(
    config: CorpusConfig,
    run_neural: bool = False,
    run_llm: bool = False,
    run_deepseek: bool = False,
) -> None:
    """Полный сценарий: поиск, опыты, отчёты, графики."""
    for directory in (config.reports_dir, config.charts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Данные.
    print(f"1. Загрузка корпуса «{config.name}»...")
    documents = DataLoader(config.documents_path).load_documents()
    check_questions = DataLoader.load_check_questions(
        config.check_questions_path
    )
    print(f"Документов: {len(documents)}, вопросов: {len(check_questions)}")

    # 2. Рабочий индекс.
    print("2. Построение TF-IDF индекса...")
    assistant = build_assistant(documents, config)
    print(
        f"CHUNK_SIZE={config.chunk_size}, OVERLAP={config.overlap}, "
        f"SEARCH_METHOD={config.search_method}, TOP_K={config.top_k}, "
        f"чанков={len(assistant.records)}"
    )

    # 3. Демонстрация.
    print("3. Демонстрационные вопросы...")
    demo_result = assistant.ask(config.demo_question)
    print_result(assistant, demo_result)

    custom_result = assistant.ask(config.custom_question)
    print("Контрольный вопрос по корпусу:")
    print_result(assistant, custom_result)

    unknown_result = assistant.ask(
        config.unknown_question,
        min_similarity=config.unknown_min_similarity,
    )
    print("Вопрос без подтверждённого ответа:")
    print_result(assistant, unknown_result)

    # 4. Проверка поиска.
    print("4. Проверка поиска по контрольным вопросам...")
    detail_df, summary = SearchEvaluator.check(
        assistant.retriever, check_questions, top_k=config.top_k
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
    DataLoader.save_dataframe(results_df, config.experiments_csv)
    DataLoader.save_dataframe(detail_df, config.retrieval_check_csv)
    DataLoader.save_text(
        "Вопрос: " + config.demo_question + "\n" + str(demo_result["answer"]),
        config.demo_answer_txt,
    )
    DataLoader.save_text(
        "Вопрос: "
        + config.custom_question
        + "\n"
        + str(custom_result["answer"]),
        config.custom_answer_txt,
    )
    DataLoader.save_text(
        "Вопрос: "
        + config.unknown_question
        + "\n"
        + str(unknown_result["answer"]),
        config.no_answer_txt,
    )

    settings = {
        "corpus": config.name,
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "search_method": config.search_method,
        "top_k": config.top_k,
        "question": config.demo_question,
        "top1_accuracy": summary["top1_accuracy"],
        "hit_at_k": summary["hit_at_k"],
        "unknown_min_similarity": config.unknown_min_similarity,
        "custom_question": config.custom_question,
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
    DataLoader.save_json(settings, config.settings_json)

    # 7. Графики.
    print("7. Сохранение графиков...")
    visualizer = Visualizer(config.charts_dir)
    visualizer.save_experiment_metrics(
        results_df,
        filename=config.experiments_chart.name,
        title="Сравнение настроек поиска (Top-1 и Hit@K)",
    )
    visualizer.save_chunks_per_experiment(
        results_df,
        filename=config.chunks_chart.name,
        title="Количество чанков при разных настройках",
    )

    # 8. Итоговый отчёт.
    print("8. Формирование итогового отчёта...")
    report_builder = ReportBuilder(config.final_report_path)
    report_text = report_builder.build_report(
        documents=documents,
        settings=settings,
        results_df=results_df,
        baseline_summary=summary,
        custom_result=custom_result,
        unknown_result=unknown_result,
        limitations=config.limitations,
        context=config.report_context,
    )
    report_builder.save(report_text)

    # 9. Факультативные сравнения.
    if run_neural:
        print("9. Факультатив: нейросетевой поиск...")
        run_neural_comparison(assistant, check_questions, config)
    if run_llm:
        print("10. Факультатив: локальная языковая модель...")
        run_llm_demo(assistant, config.demo_question)
    if run_deepseek:
        print("11. DeepSeek API: генерация ответа...")
        run_deepseek_demo(assistant, config.demo_question)

    print("Готово. Корпус:", config.name)
    print("Отчёт:", config.final_report_path)
    print("CSV:", config.experiments_csv)
    print("Проверка поиска:", config.retrieval_check_csv)
