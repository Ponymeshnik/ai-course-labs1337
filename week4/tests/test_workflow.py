# -*- coding: utf-8 -*-
"""
Тесты для workflow
Лабораторная работа №4
Дисциплина: Искусственный интеллект
"""

import pytest
from src.webhook_handler import WorkflowClient
from src.api_client import YandexGPTClient


class TestWorkflowClient:
    """Тесты для WorkflowClient."""

    def test_client_initialization(self):
        """Проверка инициализации клиента."""
        client = WorkflowClient()
        assert client.base_url == "http://localhost:5678"
        assert client.webhook_path == "application"

    def test_client_custom_url(self):
        """Проверка инициализации с кастомным URL."""
        client = WorkflowClient(base_url="http://test:5678", webhook_path="test")
        assert client.base_url == "http://test:5678"
        assert client.webhook_path == "test"


class TestYandexGPTClient:
    """Тесты для YandexGPTClient."""

    def test_client_initialization_no_credentials(self):
        """Проверка инициализации без креденшалов."""
        client = YandexGPTClient(iam_token="", folder_id="")
        assert client.iam_token == ""
        assert client.folder_id == ""

    def test_client_initialization_with_credentials(self):
        """Проверка инициализации с креденшалами."""
        client = YandexGPTClient(iam_token="test_token", folder_id="test_folder")
        assert client.iam_token == "test_token"
        assert client.folder_id == "test_folder"
