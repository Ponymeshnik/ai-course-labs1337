# -*- coding: utf-8 -*-
"""
Правила для психологического AI-ассистента
Лабораторная работа №6
"""
from .rule_engine import Rule, RulePriority


def get_psychology_rules() -> list:
    """Получение правил психологического скрининга."""
    return [
        Rule(
            rule_id="PSY_001",
            name="Критический уровень стресса",
            condition=lambda f: f.get("stress_level", 0) > 80,
            conclusion=(
                "КРИТИЧЕСКИЙ СТРЕСС: Рекомендуется обратиться "
                "к психологу в ближайшие 24 часа"
            ),
            priority=RulePriority.CRITICAL,
            description="Уровень стресса превышает 80 баллов",
            domain="psychology",
        ),
        Rule(
            rule_id="PSY_002",
            name="Хроническое недосыпание",
            condition=lambda f: f.get("sleep_hours", 8) < 4,
            conclusion=(
                "НЕДОСЫПАНИЕ: Рекомендуется улучшить гигиену сна "
                "и обратиться к сомнологу"
            ),
            priority=RulePriority.HIGH,
            description="Сон менее 4 часов в сутки",
            domain="psychology",
        ),
        Rule(
            rule_id="PSY_003",
            name="Выгорание",
            condition=lambda f: f.get("burnout_score", 0) > 70,
            conclusion=(
                "ВЫГОРАНИЕ: Необходимо снизить нагрузку, "
                "ввести регулярные перерывы"
            ),
            priority=RulePriority.HIGH,
            description="Индекс выгорания выше 70 баллов",
            domain="psychology",
        ),
        Rule(
            rule_id="PSY_004",
            name="Повышенная тревожность",
            condition=lambda f: f.get("anxiety_score", 0) > 60,
            conclusion=(
                "ТРЕВОЖНОСТЬ: Рекомендуются техники релаксации "
                "(дыхательные упражнения, медитация)"
            ),
            priority=RulePriority.MEDIUM,
            description="Уровень тревожности выше 60 баллов",
            domain="psychology",
        ),
        Rule(
            rule_id="PSY_005",
            name="Депрессивное состояние",
            condition=lambda f: f.get("mood_score", 50) < 30,
            conclusion=(
                "ДЕПРЕССИВНЫЕ СИМПТОМЫ: Рекомендуется срочная "
                "консультация психотерапевта"
            ),
            priority=RulePriority.CRITICAL,
            description="Оценка настроения ниже 30 баллов",
            domain="psychology",
        ),
    ]


# Использование
if __name__ == "__main__":
    rules = get_psychology_rules()
    print(f"Загружено {len(rules)} правил психологического скрининга")

    for rule in rules:
        print(
            f" • {rule.rule_id}: {rule.name} "
            f"(приоритет: {rule.priority.name})"
        )