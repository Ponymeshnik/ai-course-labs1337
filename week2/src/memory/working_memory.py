# -*- coding: utf-8 -*-
"""
Краткосрочная (рабочая) память агента
Лабораторная работа №2
"""

from typing import List, Dict, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class WorkingMemory:
    """
    Краткосрочная память для хранения текущей сессии.

    Хранит историю сообщений в буфере с ограничением размера.
    """

    def __init__(self, max_messages: int = 50):
        """
        Инициализация рабочей памяти.

        Args:
            max_messages: Максимальное количество сообщений
        """
        self.max_messages = max_messages
        self._messages: deque = deque(maxlen=max_messages)
        logger.info(f"Рабочая память инициализирована (max: {max_messages})")

    def add_message(self, role: str, content: str) -> None:
        """
        Добавление сообщения в память.

        Args:
            role: Роль отправителя (user/assistant/system)
            content: Содержание сообщения
        """
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Получение всех сообщений.

        Returns:
            List[Dict]: Список сообщений
        """
        return list(self._messages)

    def get_last(self, n: int = 5) -> List[Dict[str, str]]:
        """
        Получение последних n сообщений.

        Args:
            n: Количество сообщений

        Returns:
            List[Dict]: Последние сообщения
        """
        return list(self._messages)[-n:]

    def clear(self) -> None:
        """Очистка памяти."""
        self._messages.clear()
        logger.debug("Рабочая память очищена")

    def __len__(self) -> int:
        return len(self._messages)
