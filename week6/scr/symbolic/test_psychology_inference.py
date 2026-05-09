# -*- coding: utf-8 -*-
"""
Тест инференса психологических правил с извлечением фактов через LLM.
"""
import json
import re
from typing import Any, Dict

from .FIT_base import get_psychology_rules
from .rule_engine import RuleEngine
from ..neural.llm_client import LLMClient


def extract_json(text: str) -> Dict[str, Any]:
    """Извлечение JSON-объекта из ответа LLM."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("LLM не вернула JSON-объект")

    return json.loads(match.group(0))


def extract_psychology_facts(complaint: str) -> Dict[str, Any]:
    """Извлечение психологических признаков из жалобы пользователя."""
    llm = LLMClient()
    prompt = f"""
Проанализируй жалобу пользователя и оцени признаки по шкале 0-100.

Верни строго JSON без Markdown и пояснений:
{{
  "stress_level": число от 0 до 100,
  "sleep_hours": число часов сна в сутки,
  "burnout_score": число от 0 до 100,
  "anxiety_score": число от 0 до 100,
  "mood_score": число от 0 до 100
}}

Правила оценки:
- stress_level: общий уровень стресса
- sleep_hours: примерное количество сна
- burnout_score: признаки эмоционального выгорания
- anxiety_score: уровень тревожности
- mood_score: настроение, где 0 — очень плохое, 100 — хорошее

Жалоба:
{complaint}
"""

    response = llm.generate(
        prompt=prompt,
        system_prompt=(
            "Ты — ассистент для первичного психологического скрининга. "
            "Не ставь диагнозы. Возвращай только JSON."
        ),
        temperature=0.1,
        max_tokens=300,
    )

    if not response.get("success"):
        raise RuntimeError(response.get("text", "Ошибка LLM"))

    facts = extract_json(response["text"])

    return {
        "stress_level": int(facts.get("stress_level", 0)),
        "sleep_hours": float(facts.get("sleep_hours", 8)),
        "burnout_score": int(facts.get("burnout_score", 0)),
        "anxiety_score": int(facts.get("anxiety_score", 0)),
        "mood_score": int(facts.get("mood_score", 50)),
    }


def run_inference(complaint: str) -> None:
    """Запуск LLM-извлечения фактов и символьного инференса."""
    engine = RuleEngine()
    engine.add_rules(get_psychology_rules())

    facts = extract_psychology_facts(complaint)
    result = engine.infer(facts)

    print("Жалоба:")
    print(complaint)

    print("\nИзвлечённые LLM факты:")
    for key, value in facts.items():
        print(f" {key}: {value}")

    print("\nЗаключения:")
    if result.conclusions:
        for conclusion in result.conclusions:
            print(f" • {conclusion}")
    else:
        print(" • Правила не сработали")

    print(f"\nОбъяснение:\n{result.explanation}")


if __name__ == "__main__":
    user_complaint = (
        "Последние недели я почти не сплю часа 4 максимум, постоянно тревожусь по поводу простых поступков,"
        "чувствую сильную усталость из-за того чтоя просто тревожусь , раздражение и будто полностью выгорел."
    )
    run_inference(user_complaint)
