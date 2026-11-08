"""Tests for the Handbook corpus builder."""
import docx

from src.corpus_builder import HandbookCorpusBuilder


def _write_docx(path, text):
    document = docx.Document()
    document.add_paragraph(text)
    document.save(str(path))


def test_builder_collects_md_and_docx(tmp_path):
    source = tmp_path / "handbook"
    source.mkdir()
    (source / "README.md").write_text(
        "# База знаний\n\n" + "Учебный текст. " * 20, encoding="utf-8"
    )
    (source / "konspekty").mkdir()
    (source / "konspekty" / "ml.md").write_text(
        "# Конспект ML\n\n" + "Переобучение и метрики. " * 20,
        encoding="utf-8",
    )
    (source / "istochniki").mkdir()
    _write_docx(source / "istochniki" / "guide.docx", "Текст руководства " * 20)

    output = tmp_path / "corpus.json"
    stats_path = tmp_path / "stats.json"
    documents, stats = HandbookCorpusBuilder(
        source, output, stats_path
    ).build()

    assert output.is_file()
    assert stats_path.is_file()
    assert len(documents) == 3
    assert stats["included_documents"] == 3
    assert stats["by_kind"] == {"md": 2, "docx": 1}
    assert {doc["id"] for doc in documents} == {
        "README",
        "konspekty/ml",
        "istochniki/guide",
    }
    for document in documents:
        assert {"id", "title", "text", "source_path", "words"} <= set(document)


def test_builder_skips_duplicates_and_sensitive_files(tmp_path):
    source = tmp_path / "handbook"
    source.mkdir()
    text = "# Заметка\n\n" + "Одинаковый текст. " * 20
    (source / "a.md").write_text(text, encoding="utf-8")
    (source / "b.md").write_text(text, encoding="utf-8")
    _write_docx(source / "Экспорт_чата.docx", "Секретный чат " * 20)

    documents, stats = HandbookCorpusBuilder(
        source, tmp_path / "corpus.json"
    ).build()

    assert len(documents) == 1
    assert stats["skipped_duplicates_count"] == 1
    assert stats["skipped_sensitive"] == 1
