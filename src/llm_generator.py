"""Optional local LLM answer generation (блок 8, факультатив).

Подключает небольшую открытую инструкционную модель через `transformers`
и генерирует ответ по найденным фрагментам. Требует много памяти и
интернета для загрузки весов. Обязательная часть проекта работает без
этого модуля. Ответ LLM всегда нужно проверять по источникам.
"""
from __future__ import annotations

from src.config import LLM_MODEL_ID

SYSTEM_PROMPT = (
    "Отвечай только по данным в контексте. Если ответа нет, напиши: "
    "«В источниках нет ответа». Всегда указывай номера источников."
)


class LocalLLMGenerator:
    """Класс генерации ответа локальной языковой моделью."""

    def __init__(self, model_id: str = LLM_MODEL_ID):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    @staticmethod
    def is_available() -> bool:
        """Проверить, установлены ли transformers и torch."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self) -> "LocalLLMGenerator":
        """Ленивая загрузка токенизатора и модели."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        return self

    def generate(
        self,
        question: str,
        found: list[dict[str, object]],
        max_new_tokens: int = 170,
    ) -> str:
        """Сгенерировать ответ по вопросу и найденным фрагментам."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Сначала вызовите load()")

        context = "\n\n".join(
            f"[{item['doc_id']}] {item['text']}" for item in found
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Вопрос: {question}\nКонтекст:\n{context}",
            },
        ]
        chat = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([chat], return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
