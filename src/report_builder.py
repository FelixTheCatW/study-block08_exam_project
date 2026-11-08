"""Report builder module (блок 8).

Формирует итоговый markdown-отчёт: цель, база знаний, изменённые
параметры, пять экспериментов, свой документ, случай без ответа,
сравнение TF-IDF / нейропоиска / RAG с LLM и ограничения.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class ReportBuilder:
    """Класс формирования итогового markdown-отчёта."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

    def build_report(
        self,
        documents: list[dict[str, str]],
        settings: dict[str, object],
        results_df: pd.DataFrame,
        baseline_summary: dict[str, float],
        custom_result: dict[str, object],
        unknown_result: dict[str, object],
        limitations: list[str],
        context: dict[str, str] | None = None,
    ) -> str:
        """Сформировать текст отчёта."""
        context = context or {}
        title = context.get(
            "title",
            "Блок 8. Мини-помощник по базе знаний Handbook (RAG / TF-IDF)",
        )
        subtitle = context.get(
            "subtitle",
            "Поиск по учебным конспектам и Word-документам из "
            "`D:\\study\\handbook`. Корпус собран локально: точные дубли "
            "удалены, служебные и потенциально чувствительные файлы "
            "исключены.",
        )
        source_note = context.get(
            "source_note",
            "Корпус собран из тематических md-конспектов, `konspekty/` "
            "и `istochniki/`. В него не включены PDF, изображения и "
            "сырые данные.",
        )
        custom_section_title = context.get(
            "custom_section_title", "5. Контрольный вопрос по Handbook"
        )
        custom_section_note = context.get(
            "custom_section_note",
            "Проверяем, что система находит нужный раздел базы знаний "
            "по вопросу о масштабировании признаков для KMeans.",
        )

        source_text = self._markdown_table(
            pd.DataFrame(
                [
                    {
                        "источник": doc["id"],
                        "название": doc["title"],
                        "символов": len(doc["text"]),
                    }
                    for doc in documents
                ]
            )
        )
        experiments_text = self._markdown_table(results_df)
        notes_text = "\n".join(
            f"{number}. {note}"
            for number, note in enumerate(self._experiment_notes(results_df), 1)
        )
        limitations_text = "\n".join(f"- {item}" for item in limitations)
        charts_note = context.get(
            "charts_note",
            "Графики: `reports/handbook/charts/experiments_metrics.png` "
            "(Top-1 и Hit@K) и "
            "`reports/handbook/charts/chunks_per_experiment.png` "
            "(число чанков).",
        )
        run_commands = context.get(
            "run_commands",
            "python -m venv .venv\n"
            ".venv\\Scripts\\activate\n"
            "pip install -r requirements.txt\n\n"
            "python main.py --build-corpus\n"
            "python main.py\n"
            "python -m pytest -q\n\n"
            "python main.py --ask\n"
            "python main.py --question \"Что такое переобучение?\"\n"
            "python main.py --neural\n"
            "python main.py --llm",
        )
        artifacts_note = context.get(
            "artifacts_note",
            "- `reports/handbook/experiments.csv` — таблица пяти опытов;\n"
            "- `reports/handbook/retrieval_check.csv` — проверка поиска;\n"
            "- `reports/handbook/settings.json` — параметры запуска;\n"
            "- `reports/handbook/demo_answer.txt` — демонстрационный ответ;\n"
            "- `reports/handbook/custom_answer.txt` — контрольный ответ;\n"
            "- `reports/handbook/no_answer.txt` — пример без ответа;\n"
            "- `reports/handbook/charts/*.png` — графики;\n"
            "- `reports/handbook/final_report.md` — этот отчёт;\n"
            "- `reports/handbook_corpus_stats.json` — статистика корпуса.",
        )

        custom_found = custom_result["found"]
        custom_top = custom_found[0] if custom_found else None
        custom_source = (
            f"{custom_top['doc_id']} — {custom_top['title']} "
            f"(сходство {float(custom_top['score']):.3f})"
            if custom_top
            else "ничего не найдено"
        )

        unknown_found = unknown_result["found"]
        unknown_top = unknown_found[0] if unknown_found else None
        unknown_source = (
            f"{unknown_top['doc_id']} — {unknown_top['title']} "
            f"(сходство {float(unknown_top['score']):.3f})"
            if unknown_top
            else "ничего не найдено"
        )

        report_text = f"""# {title}

{subtitle}

## 1. Цель проекта

Задать вопрос своими словами, найти подтверждающий документ и получить
ответ с указанием источника. Обязательная часть использует **TF-IDF** —
статистический поиск без обучения глубокой нейросети. Это поисковая
часть RAG, а не генерация текста языковой моделью.

## 2. База знаний

Всего документов: {len(documents)}. {source_note}

{source_text}

## 3. Изменённые параметры

Относительно исходного ноутбука изменены четыре параметра:

| Параметр | Значение |
|---|---|
| `CHUNK_SIZE` | {settings["chunk_size"]} |
| `OVERLAP` | {settings["overlap"]} |
| `SEARCH_METHOD` | `{settings["search_method"]}` |
| `TOP_K` | {settings["top_k"]} |

Базовый вариант на {len(documents)} документах дал **Top-1 =
{baseline_summary["top1_accuracy"]:.3f}** и **Hit@K =
{baseline_summary["hit_at_k"]:.3f}** на контрольных вопросах.

## 4. Пять экспериментов

{experiments_text}

### Что видно по таблице и графикам

{notes_text}

{charts_note}

## {custom_section_title}

{custom_section_note}

**Вопрос:** {custom_result["question"]}

**Первый найденный источник:** {custom_source}

**Ответ системы:**

```text
{custom_result["answer"]}
```

## 6. Случай, когда подтверждённого ответа нет

**Посторонний вопрос:** {unknown_result["question"]}

**Лучший найденный источник:** {unknown_source}

**Ответ системы:**

```text
{unknown_result["answer"]}
```

Порог сходства для этого примера поднят до
`{settings["unknown_min_similarity"]}`. Поиск всё равно возвращает
какой-то фрагмент, но система не выдаёт его за подтверждённый ответ.
Это важное правило ответственного ИИ: лучше сказать «нет данных»,
чем выдумать факт.

## 7. TF-IDF, нейросетевой поиск и полноценный RAG с LLM

| Подход | Что делает | Глубокое обучение? | Ограничение |
|---|---|---|---|
| **TF-IDF** | Сравнивает слова / группы символов и их статистическую важность | Нет | Не понимает синонимы и смысл без общих слов |
| **Семантический поиск** | Сравнивает эмбеддинги предобученной нейросети | Да, используется готовая модель | Может ошибаться и требует ресурсов |
| **RAG с LLM** | Поиск + передача фрагментов языковой модели, которая пишет новый ответ | Да | LLM может галлюцинировать; ответ надо проверять по источникам |

Наш помощник реализует **Retrieval** и выдаёт точные выдержки. Он не
называет это генерацией, пока не подключена языковая модель. В
факультативном режиме `--neural` можно сравнить нейросетевой поиск с
TF-IDF, а в `--llm` — попробовать локальную LLM и проверить, не
придумала ли она факты.

## 8. Ограничения

{limitations_text}

## 9. Как запустить

```bash
{run_commands}
```

## 10. Артефакты

{artifacts_note}
"""
        return report_text

    @staticmethod
    def _experiment_notes(results_df: pd.DataFrame) -> list[str]:
        """Сформировать объяснение результатов по строкам таблицы."""
        notes: list[str] = []

        best_top1 = results_df.loc[results_df["top1"].idxmax()]
        best_hit = results_df.loc[results_df["hit_at_k"].idxmax()]
        notes.append(
            f"Лучший Top-1 у варианта «{best_top1['опыт']}»: "
            f"{best_top1['top1']:.3f}. Лучший Hit@K у «{best_hit['опыт']}»: "
            f"{best_hit['hit_at_k']:.3f}."
        )

        a = ReportBuilder._row_by_code(results_df, "A")
        b = ReportBuilder._row_by_code(results_df, "B")
        if a is not None and b is not None:
            notes.append(
                "A и B отличаются способом поиска при одинаковых чанках: "
                f"word даёт Top-1 {a['top1']:.3f} / Hit@K {a['hit_at_k']:.3f}, "
                f"char_wb — {b['top1']:.3f} / {b['hit_at_k']:.3f}. "
                "Поиск по группам букв устойчивее к окончаниям, а Top-K "
                "добавляет второй шанс найти верный документ."
            )

        c = ReportBuilder._row_by_code(results_df, "C")
        d = ReportBuilder._row_by_code(results_df, "D")
        if c is not None and d is not None:
            notes.append(
                "C и D показывают влияние размера чанка: короткие чанки "
                f"({int(c['чанк'])} символов) дают Top-1 {c['top1']:.3f}, "
                f"длинные ({int(d['чанк'])} символов) — {d['top1']:.3f}. "
                "Короткие чанки точнее нацелены на предложение-ответ, но "
                "могут терять контекст; длинные сохраняют контекст, но "
                "размывают сходство."
            )

        e = ReportBuilder._row_by_code(results_df, "E")
        if e is not None:
            notes.append(
                "Новый эксперимент E меняет сразу способ поиска, overlap и "
                f"Top-K: word, overlap={int(e['overlap'])}, "
                f"top_k={int(e['top_k'])}. Он даёт Top-1 {e['top1']:.3f} и "
                f"Hit@K {e['hit_at_k']:.3f}. Большое Top-K повышает шанс "
                "найти нужный источник, но добавляет шум в выдачу; "
                "overlap=0 уменьшает число чанков и может разрывать мысль "
                "на границе."
            )
        return notes

    @staticmethod
    def _row_by_code(results_df: pd.DataFrame, code: str):
        """Найти строку эксперимента по его коду (A, B, C, D, E)."""
        mask = results_df["опыт"].astype(str).str.startswith(f"{code}:")
        if not mask.any():
            return None
        return results_df.loc[mask.idxmax()]

    @staticmethod
    def _markdown_table(dataframe: pd.DataFrame) -> str:
        """Собрать markdown-таблицу без зависимости от tabulate."""
        headers = list(dataframe.columns)
        lines = [
            "| " + " | ".join(str(header) for header in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for _, row in dataframe.iterrows():
            values = [str(row[header]) for header in headers]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def save(self, report_text: str) -> None:
        """Сохранить markdown-отчёт."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(report_text, encoding="utf-8")
