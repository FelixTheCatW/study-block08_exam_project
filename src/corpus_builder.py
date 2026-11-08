"""Handbook corpus builder (блок 8).

Собирает текстовые учебные материалы из `D:\\study\\handbook` в единый
список документов для RAG:
- markdown-конспекты и тематические файлы;
- Word-документы из папки `istochniki`;
- отбрасывает явно служебные/чувствительные файлы;
- удаляет точные дубли по содержимому;
- сохраняет метаданные: путь, блок, тип, язык, число слов.

Тяжёлые данные, изображения, PDF и архивы в корпус не попадают.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    import docx
except ImportError:  # pragma: no cover - зависит от окружения
    docx = None


SKIP_DIRS = {
    ".git", ".venv", ".venv~", "venv", "node_modules", "__pycache__",
    ".idea", ".pytest_cache", ".pytest_tmp", ".ipynb_checkpoints",
    "site-packages", ".kilo", "worktrees",
}
TEXT_EXTENSIONS = {".md", ".docx"}
SENSITIVE_NAME_PATTERNS = (
    "экспорт_чата",
    "доступы",
    "pwd",
    "password",
    "passwords",
    "secret",
    "token",
    "api_key",
    "api-ключ",
    "моя мисия",
)
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def normalize_text(text: str) -> str:
    """Привести текст к предсказуемому виду без потери содержания."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    """Посчитать слова/токены простым регулярным выражением."""
    return len(WORD_RE.findall(text))


def detect_language(text: str) -> str:
    """Очень простой определитель языка: ru / en / mix."""
    cyrillic = len(CYRILLIC_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cyrillic > latin * 2:
        return "ru"
    if latin > cyrillic * 2:
        return "en"
    return "mix"


def content_hash(text: str) -> str:
    """Хэш нормализованного текста для удаления точных дублей."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class HandbookCorpusBuilder:
    """Класс сборки корпуса из папки Handbook."""

    def __init__(
        self,
        source_dir: Path,
        output_path: Path,
        stats_path: Path | None = None,
        include_sensitive: bool = False,
    ):
        self.source_dir = Path(source_dir)
        self.output_path = Path(output_path)
        self.stats_path = Path(stats_path) if stats_path else None
        self.include_sensitive = include_sensitive

    def build(self) -> tuple[list[dict[str, object]], dict[str, object]]:
        """Собрать корпус и сохранить JSON + статистику."""
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"Папка Handbook не найдена: {self.source_dir}. "
                "Укажите путь через переменную окружения HANDBOOK_DIR."
            )
        if docx is None:
            raise ImportError(
                "Для чтения .docx установите пакет python-docx: "
                "pip install python-docx"
            )

        files = sorted(
            path for path in self.source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS
        )

        documents: list[dict[str, object]] = []
        seen_hashes: dict[str, str] = {}
        stats: dict[str, object] = {
            "source_dir": str(self.source_dir),
            "output_path": str(self.output_path),
            "scanned_files": len(files),
            "included_documents": 0,
            "skipped_sensitive": [],
            "skipped_duplicates": [],
            "skipped_empty": [],
            "total_words": 0,
            "total_chars": 0,
            "by_category": {},
            "by_kind": {},
            "by_block": {},
            "by_language": {},
        }

        for path in files:
            if self._is_skipped(path):
                continue
            if not self.include_sensitive and self._is_sensitive(path):
                stats["skipped_sensitive"].append(self._relative(path))
                continue

            text = normalize_text(self._extract_text(path))
            if len(text) < 40:
                stats["skipped_empty"].append(self._relative(path))
                continue

            digest = content_hash(text)
            if digest in seen_hashes:
                stats["skipped_duplicates"].append(
                    {
                        "path": self._relative(path),
                        "duplicate_of": seen_hashes[digest],
                    }
                )
                continue
            seen_hashes[digest] = self._relative(path)

            relative_path = self._relative(path)
            doc_id = self._make_doc_id(path)
            title = self._extract_title(path, text)
            category = self._category(path)
            block = self._block(path)
            kind = path.suffix.lower().lstrip(".")
            words = count_words(text)
            language = detect_language(text)

            documents.append(
                {
                    "id": doc_id,
                    "title": title,
                    "text": text,
                    "source_path": relative_path,
                    "category": category,
                    "block": block,
                    "kind": kind,
                    "language": language,
                    "chars": len(text),
                    "words": words,
                    "hash": digest,
                }
            )

            stats["total_words"] += words
            stats["total_chars"] += len(text)
            self._increment(stats["by_category"], category)
            self._increment(stats["by_kind"], kind)
            self._increment(stats["by_block"], block)
            self._increment(stats["by_language"], language)

        # На всякий случай проверим уникальность идентификаторов.
        ids = [str(doc["id"]) for doc in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Получились неуникальные id документов")

        stats["included_documents"] = len(documents)
        stats["skipped_sensitive"] = len(stats["skipped_sensitive"])
        stats["skipped_duplicates_count"] = len(stats["skipped_duplicates"])
        stats["skipped_empty_count"] = len(stats["skipped_empty"])

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.stats_path:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            self.stats_path.write_text(
                json.dumps(stats, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return documents, stats

    def _is_skipped(self, path: Path) -> bool:
        relative = path.relative_to(self.source_dir)
        return any(part in SKIP_DIRS for part in relative.parts)

    def _is_sensitive(self, path: Path) -> bool:
        lowered = path.name.lower()
        return any(pattern in lowered for pattern in SENSITIVE_NAME_PATTERNS)

    def _extract_text(self, path: Path) -> str:
        if path.suffix.lower() == ".docx":
            return self._extract_docx(path)
        return self._read_text_file(path)

    @staticmethod
    def _read_text_file(path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "cp1251"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _extract_docx(path: Path) -> str:
        document = docx.Document(str(path))
        parts: list[str] = []
        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                parts.append(paragraph.text)
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return "\n".join(parts)

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.source_dir).as_posix()

    def _make_doc_id(self, path: Path) -> str:
        """Стабильный и читаемый id: относительный путь без расширения."""
        return path.relative_to(self.source_dir).with_suffix("").as_posix()

    def _extract_title(self, path: Path, text: str) -> str:
        """Взять понятный заголовок: heading для md, имя файла для docx."""
        relative = path.relative_to(self.source_dir)
        if relative.parts[0] == "konspekty":
            return f"Конспект: {path.stem}"
        if path.suffix.lower() == ".docx":
            stem = path.stem.strip()
            if stem.lower() in {"заметки", "заметка"}:
                return f"{stem} ({path.parent.name})"
            return stem[:160]
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                if title:
                    return title[:160]
        return path.stem

    def _category(self, path: Path) -> str:
        relative = path.relative_to(self.source_dir)
        if relative.parts[0] == "konspekty":
            return "konspekt"
        if relative.parts[0] == "istochniki":
            return "source"
        return "handbook"

    def _block(self, path: Path) -> str:
        relative = path.relative_to(self.source_dir)
        parts = relative.parts
        if parts[0] == "istochniki" and len(parts) > 2:
            return parts[1]
        if parts[0] == "konspekty":
            return path.stem
        return path.stem if parts[0] != "README.md" else "handbook"

    @staticmethod
    def _increment(counter: Counter, key: str) -> None:
        counter[key] = counter.get(key, 0) + 1
