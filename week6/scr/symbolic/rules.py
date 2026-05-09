# Файл: src/symbolic/rules.py
# -*- coding: utf-8 -*-
"""
Пример символьной системы правил
"""


class Rule:
    """Правило в формате IF-THEN."""

    def __init__(self, condition: str, conclusion: str, priority: int = 1):
        self.condition = condition  # Логическое условие
        self.conclusion = conclusion  # Вывод при истинности
        self.priority = priority  # Приоритет правила

    def evaluate(self, facts: dict) -> bool:
        """Оценка истинности условия."""
        # Простая реализация для примера
        return eval(self.condition, {}, facts)


class RuleEngine:
    """Движок логического вывода на правилах."""

    def __init__(self):
        self.rules = []

    def add_rule(self, rule: Rule):
        """Добавление правила."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def infer(self, facts: dict) -> list:
        """Логический вывод на основе фактов."""
        conclusions = []
        for rule in self.rules:
            if rule.evaluate(facts):
                conclusions.append(rule.conclusion)
        return conclusions


# Пример использования
engine = RuleEngine()
engine.add_rule(Rule("temperature > 38", "диагноз = 'лихорадка'", priority=2))
engine.add_rule(Rule("pressure < 90", "диагноз = 'гипотония'", priority=2))
engine.add_rule(Rule("temperature > 40", "диагноз = 'критическое состояние'", priority=1))
facts = {"temperature": 39.5, "pressure": 95}
conclusions = engine.infer(facts)
print(conclusions)  # ["диагноз = 'лихорадка'"]
