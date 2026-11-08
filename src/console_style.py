"""ANSI-подсветка консольного вывода (блок 8).

Цвета включаются только в терминале. При перенаправлении вывода,
в тестах или при `NO_COLOR` остаётся обычный текст. Принудительно
включить цвета можно переменной `FORCE_COLOR=1`.
"""
from __future__ import annotations

import os
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}


def color_enabled() -> bool:
    """Проверить, нужно ли использовать ANSI-цвета."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def style(
    text: object,
    color: str | None = None,
    bold: bool = False,
    dim: bool = False,
) -> str:
    """Обернуть текст в ANSI-последовательности, если цвета включены."""
    text = str(text)
    if not color_enabled():
        return text
    prefix = ""
    if bold:
        prefix += BOLD
    if dim:
        prefix += DIM
    if color:
        prefix += COLORS.get(color, "")
    return f"{prefix}{text}{RESET}" if prefix else text


def question(text: object) -> str:
    """Цвет вопроса."""
    return style(text, "cyan", bold=True)


def label(text: object) -> str:
    """Цвет подписи-заголовка."""
    return style(text, "magenta", bold=True)


def source(text: object) -> str:
    """Цвет источника."""
    return style(text, "yellow")


def answer(text: object) -> str:
    """Цвет ответа."""
    return style(text, "green")


def warning(text: object) -> str:
    """Цвет предупреждения или отказа."""
    return style(text, "red", bold=True)


def muted(text: object) -> str:
    """Приглушённый текст."""
    return style(text, "white", dim=True)
