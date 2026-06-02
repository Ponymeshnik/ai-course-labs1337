# -*- coding: utf-8 -*-
"""
Обработчик webhook для тестирования workflow
Лабораторная работа №4
Дисциплина: Искусственный интеллект
Автор: Мишакин Илья Геннадьевич
Группа: ФИТ-221
Дата: 18 марта 2026
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

import requests
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()


class WorkflowClient:
    """
    Клиент для взаимодействия с n8n workflow.

    Атрибуты:
        base_url: URL n8n сервера
        webhook_path: Путь webhook
        secret: Секретный ключ для аутентификации
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5678",
        webhook_path: str = "application",
        secret: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.webhook_path = webhook_path
        self.secret = secret or os.getenv("WEBHOOK_SECRET")
        self.webhook_url = f"{self.base_url}/webhook/{webhook_path}"

        logger.info(f"WorkflowClient инициализирован: {self.webhook_url}")

    def send_application(
        self,
        message: str,
        contact: str,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Отправка заявки в workflow.

        Args:
            message: Текст заявки
            contact: Контактная информация
            priority: Приоритет (low, normal, high)

        Returns:
            Dict: Ответ от workflow
        """
        payload = {
            "message": message,
            "contact": contact,
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        }

        headers = {
            "Content-Type": "application/json"
        }

        if self.secret:
            headers["X-Webhook-Secret"] = self.secret

        logger.info(f"Отправка заявки: {message[:50]}...")

        try:
            response = requests.post(
                self.webhook_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()

            logger.info("Workflow выполнил обработку")

            return {
                "success": True,
                "response": result,
                "status_code": response.status_code
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка отправки: {e}")
            return {
                "success": False,
                "error": str(e),
                "status_code": None
            }

    def send_security_incident(
        self,
        log_entry: str,
        source_ip: str,
        event_type: str
    ) -> Dict[str, Any]:
        """
        Отправка инцидента безопасности (для ИБ-специальности).

        Args:
            log_entry: Лог события
            source_ip: IP-адрес источника
            event_type: Тип события

        Returns:
            Dict: Ответ от workflow
        """
        payload = {
            "log_entry": log_entry,
            "source_ip": source_ip,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat()
        }

        headers = {
            "Content-Type": "application/json"
        }

        if self.secret:
            headers["X-Webhook-Secret"] = self.secret

        webhook_url = f"{self.base_url}/webhook/security-incident"

        logger.info(f"Отправка инцидента ИБ: {event_type}")

        try:
            response = requests.post(
                webhook_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()

            logger.info("Инцидент обработан")

            return {
                "success": True,
                "response": result,
                "status_code": response.status_code
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка отправки: {e}")
            return {
                "success": False,
                "error": str(e),
                "status_code": None
            }

    def check_workflow_status(self) -> Dict[str, Any]:
        """Проверка доступности n8n."""
        try:
            response = requests.get(
                f"{self.base_url}/healthz",
                timeout=5
            )
            return {
                "available": response.status_code == 200,
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }


# Точка входа для тестирования
if __name__ == "__main__":
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №4")
    print("Тестирование workflow automation")
    print("=" * 80)

    client = WorkflowClient()

    # Проверка доступности
    print("\nПроверка доступности n8n...")
    status = client.check_workflow_status()

    if status.get("available"):
        print("✅ n8n доступен")
    else:
        print(f"❌ n8n недоступен: {status.get('error')}")
        exit(1)

    # Тестовая заявка
    print("\n" + "=" * 80)
    print("ТЕСТОВАЯ ЗАЯВКА")
    print("=" * 80)

    result = client.send_application(
        message="Не работает вход в систему, ошибка 403",
        contact="user@example.com",
        priority="high"
    )

    if result["success"]:
        print("✅ Заявка отправлена успешно")
        print(f"Статус код: {result['status_code']}")
    else:
        print(f"❌ Ошибка: {result['error']}")
    print("=" * 80)