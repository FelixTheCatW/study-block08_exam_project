"""DeepSeek answer generation (RAG, блок 8).

Берёт найденные TF-IDF фрагменты и просит модель DeepSeek составить
ответ только по этим источникам. Ключ читается из DEEPSEEK_API_KEY
(или локального .env) и никогда не печатается.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from src.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TIMEOUT,
)

SYSTEM_PROMPT = (
    "Ты помощник по учебным документам. Отвечай только по данным в "
    "контексте. Не придумывай факты. Если ответа нет, напиши: "
    "«В источниках нет ответа». Всегда указывай коды источников "
    "в квадратных скобках, например [konspekty/ml]."
)


class DeepSeekGenerator:
    """Класс генерации ответа через DeepSeek API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEEPSEEK_MODEL,
        api_url: str = DEEPSEEK_API_URL,
        timeout: int = DEEPSEEK_TIMEOUT,
    ):
        self.api_key = (api_key or DEEPSEEK_API_KEY).strip()
        self.model = model
        self.api_url = api_url
        self.timeout = timeout

    @staticmethod
    def is_available() -> bool:
        """Проверить, задан ли ключ DeepSeek."""
        return bool(DEEPSEEK_API_KEY)

    def generate(
        self,
        question: str,
        found: list[dict[str, object]],
        max_tokens: int = 500,
    ) -> str:
        """Сгенерировать ответ по вопросу и найденным фрагментам."""
        if not self.api_key:
            raise RuntimeError(
                "Не задан DEEPSEEK_API_KEY. Положите ключ в .env или "
                "переменную окружения."
            )
        if not found:
            raise RuntimeError("Нет найденных фрагментов для генерации.")

        context = "\n\n".join(
            f"[{item['doc_id']}] {item['title']}\n{item['text']}"
            for item in found
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Вопрос: {question}\n\nКонтекст:\n{context}",
                },
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = self._post(payload)
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Неожиданный ответ DeepSeek API: {str(data)[:500]}"
            ) from exc

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        """Отправить запрос с запасным вариантом /v1/chat/completions."""
        urls = [self.api_url]
        if "/v1/" not in self.api_url:
            urls.append(
                self.api_url.replace(
                    "api.deepseek.com/",
                    "api.deepseek.com/v1/",
                )
            )

        last_error: Exception | None = None
        for url in urls:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(
                    f"DeepSeek API HTTP {exc.code}: {body}"
                )
                if exc.code == 404 and url != urls[-1]:
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(
                    f"DeepSeek API недоступен: {exc.reason}"
                )
                raise last_error from exc

        raise last_error or RuntimeError("DeepSeek API: неизвестная ошибка")
