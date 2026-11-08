"""Integration checks for the real Handbook corpus."""
from src.config import CHUNK_SIZE, OVERLAP, TOP_K


def test_handbook_corpus_metadata(handbook_documents):
    ids = [doc["id"] for doc in handbook_documents]

    assert len(handbook_documents) >= 70
    assert len(ids) == len(set(ids))
    assert {"README", "08-ml", "konspekty/ml"} <= set(ids)
    assert {doc["category"] for doc in handbook_documents} == {
        "handbook",
        "konspekt",
        "source",
    }
    for doc in handbook_documents:
        assert {"id", "title", "text", "source_path", "kind", "words"} <= set(doc)
        assert doc["text"].strip()
        assert doc["words"] > 0


def test_handbook_questions_reference_existing_documents(
    handbook_documents, handbook_check_questions
):
    doc_ids = {doc["id"] for doc in handbook_documents}

    for question, expected in handbook_check_questions:
        expected_set = {expected} if isinstance(expected, str) else set(expected)
        assert expected_set, question
        assert expected_set <= doc_ids, (question, expected_set - doc_ids)


def test_handbook_parameters_are_sane():
    assert CHUNK_SIZE >= 100
    assert 0 <= OVERLAP < CHUNK_SIZE
    assert TOP_K >= 1
