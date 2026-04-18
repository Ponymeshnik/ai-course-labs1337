# -*- coding: utf-8 -*-
"""
API клиент для взаимодействия с внешними сервисами
Лабораторная работа №4
Дисциплина: Искусственный интеллект
"""

import os
import logging
from typing import Dict, Any, Optional

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class YandexGPTClient:
    """Клиент для работы с YandexGPT API."""

    def __init__(
        self,
        iam_token: Optional[str] = None,
        folder_id: Optional[str] = None
    ):
        self.iam_token = iam_token or os.getenv("YANDEX_IAM_TOKEN")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.base_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

        if not self.iam_token or not self.folder_id:
            logger.warning("YANDEX_IAM_TOKEN или YANDEX_FOLDER_ID не настроены")

    def classify_text(self, text: str, categories: list = None) -> Dict[str, Any]:
        """Классификация текста с помощью YandexGPT."""
        if categories is None:
            categories = ["техническая_поддержка", "продажа", "вопрос", "жалоба"]

        system_prompt = (
            f"Классифицируй заявку как одну из категорий: {', '.join(categories)}. "
            "Ответь только названием категории."
        )

        return self._call_gpt(system_prompt, text)

    def analyze_security_incident(self, log_entry: str) -> Dict[str, Any]:
        """Анализ инцидента безопасности."""
        system_prompt = (
            "Классифицируй инцидент безопасности по уровню критичности: "
            "critical, high, medium, low. Также определи тип атаки. "
            'Ответь в формате JSON: {"level": "...", "type": "...", "recommendation": "..."}'
        )

        return self._call_gpt(system_prompt, log_entry)

    def _call_gpt(self, system_prompt: str, user_text: str) -> Dict[str, Any]:
        """Вызов API YandexGPT."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id
        }

        body = {
            "modelUri": f"gpt://{self.folder_id}/yandexGPT/latest",
            "completionOptions": {
                "temperature": 0.3,
                "maxTokens": 100
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_text}
            ]
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=body,
                timeout=30
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка вызова YandexGPT: {e}")
            return {"success": False, "error": str(e)}
