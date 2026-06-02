# -*- coding: utf-8 -*-
"""
Ядро AI-агента с поддержкой инструментов и памяти
Лабораторная работа №2
Дисциплина: Искусственный интеллект
Автор: [ФИО]
Группа: [НОМЕР ГРУППЫ]
Дата: 2026
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool

# Локальные импорты
from tools.search_tool import SearchTool
from tools.calc_tool import CalculateTool
from tools.custom_tool import CustomTool  # Адаптируемый инструмент
from memory.working_memory import WorkingMemory
from memory.semantic_memory import SemanticMemory
from guardrails.input_validator import InputGuardrails

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """
    Конфигурация AI-агента.

    Атрибуты:
        name: Имя агента
        version: Версия агента
        max_iterations: Максимальное количество итераций ReAct
        temperature: Параметр креативности LLM
        memory_enabled: Флаг включения памяти
        guardrails_enabled: Флаг включения защиты
        verbose: Режим подробного логирования
    """
    name: str = "TechnicalAssistant"
    version: str = "2.0"
    max_iterations: int = 10
    temperature: float = 0.7
    memory_enabled: bool = True
    guardrails_enabled: bool = True
    verbose: bool = True


@dataclass
class AgentResponse:
    """
    Структурированный ответ агента.

    Атрибуты:
        success: Флаг успешного выполнения
        answer: Текстовый ответ
        steps: Список выполненных шагов
        duration_ms: Время выполнения в миллисекундах
        tokens_used: Оценка использованных токенов
        error: Сообщение об ошибке (если есть)
    """
    success: bool
    answer: str
    steps: List[Dict]
    duration_ms: int
    tokens_used: int
    error: Optional[str] = None


class AIAgent:
    """
    Основной класс AI-агента.

    Реализует архитектуру с поддержкой:
    - Множественных инструментов
    - Краткосрочной и долгосрочной памяти
    - Механизмов безопасности (guardrails)
    - Паттерна ReAct для принятия решений

    Пример использования:
        >>> agent = AIAgent()
        >>> response = agent.run("Найди информацию о Python 3.12")
        >>> print(response.answer)
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Инициализация AI-агента.

        Args:
            config: Конфигурация агента (опционально, используется default)

        Raises:
            ValueError: Если не найдены необходимые переменные окружения
        """
        self.config = config or AgentConfig()
        logger.info(f"Инициализация агента: {self.config.name} v{self.config.version}")

        # Инициализация LLM
        self.llm = self._init_llm()
        logger.info("LLM инициализирован")

        # Инициализация инструментов
        self.tools = self._init_tools()
        logger.info(f"Доступно инструментов: {len(self.tools)}")

        # Инициализация памяти
        self.working_memory = None
        self.semantic_memory = None
        if self.config.memory_enabled:
            self.working_memory = WorkingMemory()
            self.semantic_memory = SemanticMemory()
            logger.info("Память активирована")

        # Инициализация guardrails
        self.guardrails = None
        if self.config.guardrails_enabled:
            self.guardrails = InputGuardrails()
            logger.info("Guardrails активированы")

        # Инициализация LangChain-агента (системный промпт)
        self.agent = self._init_agent()
        logger.info("Агент готов к работе")

        # Статистика
        self.request_count = 0
        self.total_tokens = 0
        self._last_result = {}

    def _init_llm(self):
        """
        Инициализация LLM-клиента (YandexGPT).

        Returns:
            YandexGPTLangChain: Настроенный клиент LLM

        Raises:
            ValueError: Если не найдены учётные данные
        """
        from langchain_community.llms import YandexGPT

        iam_token = os.getenv("YANDEX_IAM_TOKEN")
        folder_id = os.getenv("YANDEX_FOLDER_ID")

        if not iam_token or not folder_id:
            raise ValueError(
                "Не найдены YANDEX_IAM_TOKEN или YANDEX_FOLDER_ID. "
                "Проверьте файл .env"
            )

        return YandexGPT(
            iam_token=iam_token,
            folder_id=folder_id,
            temperature=self.config.temperature,
            max_tokens=1000
        )

    def _init_tools(self) -> List[BaseTool]:
        """
        Инициализация инструментов агента.

        Returns:
            List[BaseTool]: Список доступных инструментов
        """
        tools = [
            SearchTool().to_langchain_tool(),
            CalculateTool().to_langchain_tool(),
            CustomTool().to_langchain_tool(),  # Адаптируемый инструмент
        ]
        return tools

    def _init_agent(self):
        """
        Инициализация системного промпта для ReAct-агента.

        Returns:
            str: Системный промпт
        """
        # Формируем описание инструментов
        tools_desc = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
        tools_names = ", ".join([tool.name for tool in self.tools])

        return (
            "Ты — технический ассистент. У тебя есть доступ к следующим инструментам:\n"
            f"{tools_desc}\n\n"
            f"Доступные инструменты: {tools_names}\n\n"
            "Для выполнения задач используй формат ReAct:\n"
            "Thought: <рассуждение>\n"
            "Action: <имя инструмента>\n"
            "Action Input: <входные данные инструмента>\n"
            "Observation: <результат инструмента>\n"
            "... (можно повторить N раз) ...\n"
            "Thought: <рассуждение>\n"
            "Final Answer: <финальный ответ пользователю>\n\n"
            "Правила:\n"
            "1. Всегда начинай с Thought\n"
            "2. Action должен быть ровно один за раз\n"
            "3. Используй только доступные инструменты\n"
            "4. Когда готов ответить — напиши Final Answer\n"
            "5. Отвечай подробно и по делу"
        ).format(tools_names=tools_names)

    def run(self, query: str, session_id: Optional[str] = None) -> AgentResponse:
        """
        Выполнение запроса к агенту.

        Args:
            query: Текстовый запрос пользователя
            session_id: Идентификатор сессии (опционально)

        Returns:
            AgentResponse: Структурированный ответ агента
        """
        start_time = time.time()
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._last_result = {}

        # Проверка безопасности входа
        if self.guardrails:
            security_check = self.guardrails.validate_input(query)
            if not security_check.is_safe:
                logger.warning(f"Запрос отклонён guardrails: {security_check.reason}")
                return AgentResponse(
                    success=False,
                    answer="Запрос отклонён системой безопасности.",
                    steps=[],
                    duration_ms=0,
                    tokens_used=0,
                    error=security_check.reason
                )

        try:
            # Выполнение запроса через ReAct-цикл
            logger.info(f"Обработка запроса: {query[:100]}...")
            answer, steps = self._react_loop(query)

            # Сохранение в память
            if self.working_memory:
                self.working_memory.add_message("user", query)
                self.working_memory.add_message("assistant", answer)

            if self.semantic_memory and session_id:
                self.semantic_memory.add_document(
                    content=f"Query: {query}\nAnswer: {answer}",
                    metadata={"session_id": session_id, "type": "interaction"}
                )

            # Формирование ответа
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)

            self.request_count += 1
            tokens_used = self._estimate_tokens(query, answer)
            self.total_tokens += tokens_used

            self._last_result = {"intermediate_steps": steps}

            response = AgentResponse(
                success=True,
                answer=answer,
                steps=steps,
                duration_ms=duration_ms,
                tokens_used=tokens_used
            )

            logger.info(f"Запрос выполнен за {duration_ms}мс")
            return response

        except Exception as e:
            logger.error(f"Ошибка выполнения: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                answer="Произошла ошибка при обработке запроса.",
                steps=[],
                duration_ms=0,
                tokens_used=0,
                error=str(e)
            )

    def _react_loop(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Ручной ReAct-цикл для моделей без bind_tools.

        Args:
            query: Запрос пользователя

        Returns:
            Tuple[str, List[Dict]]: (финальный_ответ, шаги)
        """
        import re

        system_prompt = self.agent  # self.agent теперь содержит системный промпт
        messages = [SystemMessage(content=system_prompt)]
        messages.append(HumanMessage(content=query))

        steps = []
        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1

            # Генерация ответа
            full_prompt = "\n".join([
                msg.content if hasattr(msg, 'content') else str(msg)
                for msg in messages
            ])

            if self.config.verbose:
                logger.info(f"--- Итерация {iteration} ---")

            response = self.llm.invoke(full_prompt)
            response_text = response if isinstance(response, str) else str(response)

            if self.config.verbose:
                logger.debug(f"LLM ответ:\n{response_text[:500]}")

            # Поиск Action
            action_match = re.search(
                r"^Action:\s*([^\r\n]+)$", response_text, re.IGNORECASE | re.MULTILINE
            )
            action_input_match = re.search(
                r"^Action Input:\s*(.+)$", response_text, re.IGNORECASE | re.MULTILINE
            )
            final_answer_match = re.search(
                r"^Final Answer:\s*(.+)$",
                response_text,
                re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )

            # Если найден Final Answer — завершаем
            if final_answer_match and not action_match:
                return final_answer_match.group(1).strip(), steps

            # Если найден Action — выполняем инструмент
            if action_match and action_input_match:
                action_name = action_match.group(1).strip()
                action_input = action_input_match.group(1).strip()

                # Поиск инструмента по имени
                tool_map = {tool.name.lower(): tool for tool in self.tools}
                tool = tool_map.get(action_name.lower())

                if tool:
                    try:
                        # Выполнение инструмента
                        observation = tool.invoke(action_input)
                        obs_text = observation if isinstance(observation, str) else str(observation)

                        steps.append({
                            "step": iteration,
                            "tool": action_name,
                            "input": action_input,
                            "output": obs_text[:500]
                        })

                        # Добавляем в историю
                        messages.append(AIMessage(content=response_text))
                        messages.append(HumanMessage(content=f"Observation: {obs_text}"))

                        if self.config.verbose:
                            logger.info(f"Инструмент '{action_name}' -> {obs_text[:200]}")

                    except Exception as e:
                        error_msg = f"Ошибка инструмента '{action_name}': {e}"
                        logger.error(error_msg)
                        messages.append(AIMessage(content=response_text))
                        messages.append(HumanMessage(content=f"Observation: {error_msg}"))
                else:
                    # Инструмент не найден
                    available = ", ".join([t.name for t in self.tools])
                    error_msg = f"Инструмент '{action_name}' не найден. Доступные: {available}"
                    messages.append(AIMessage(content=response_text))
                    messages.append(HumanMessage(content=f"Observation: {error_msg}"))
            else:
                # Нет Action или Final Answer — завершаем
                return response_text.strip(), steps

        # Превышено количество итераций
        return "Превышено максимальное количество итераций.", steps

    def _estimate_tokens(self, input_text: str, output_text: str) -> int:
        """
        Оценка количества использованных токенов.

        Args:
            input_text: Входной текст
            output_text: Выходной текст

        Returns:
            int: Приблизительное количество токенов
        """
        # Приблизительно: 1 токен ≈ 4 символа для русского
        return (len(input_text) + len(output_text)) // 4

    def get_stats(self) -> Dict:
        """
        Получение статистики агента.

        Returns:
            Dict: Статистика использования
        """
        return {
            "name": self.config.name,
            "version": self.config.version,
            "tools_count": len(self.tools),
            "memory_enabled": self.config.memory_enabled,
            "guardrails_enabled": self.config.guardrails_enabled,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens
        }

    def save_session(self, session_id: str) -> bool:
        """
        Сохранение текущей сессии в долговременную память.

        Args:
            session_id: Идентификатор сессии

        Returns:
            bool: Успешность сохранения
        """
        if self.working_memory and self.semantic_memory:
            content = str(self.working_memory.get_messages())
            self.semantic_memory.add_document(
                content=content,
                metadata={"session_id": session_id, "type": "full_session"}
            )
            return True
        return False


# Точка входа для тестирования
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №2")
    print("Тестирование AI-агента с инструментами и памятью")
    print("=" * 80)
    agent = AIAgent()

    test_query = "Найди информацию о последнем обновлении YandexGPT и рассчитай, сколько месяцев прошло с момента выхода"

    print(f"\nЗапрос: {test_query}\n")

    response = agent.run(test_query)

    print("\n" + "=" * 80)
    print("ОТВЕТ АГЕНТА")
    print("=" * 80)
    print(response.answer)
    print("=" * 80)
    print(f"Время выполнения: {response.duration_ms}мс")
    print(f"Использовано токенов: {response.tokens_used}")
    print(f"Статус: {'Успех' if response.success else 'Ошибка'}")
    if response.error:
        print(f"Ошибка: {response.error}")
    print("=" * 80)
