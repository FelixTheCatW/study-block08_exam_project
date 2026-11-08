"""Tiny .env loader without external dependencies.

Читает файл `.env` в формате KEY=VALUE и выставляет значения в
os.environ, не перезаписывая уже заданные переменные окружения.
Секреты не логируются.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Загрузить переменные из .env, если файл существует."""
    path = Path(path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
