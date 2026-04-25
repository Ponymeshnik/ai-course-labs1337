# -*- coding: utf-8 -*-
from typing import Any, Dict
import os
from langchain_core.prompts import ChatPromptTemplate

from rag.rag_pipeline import RAGPipeline
from rag.vector_store import VectorStoreManager


class OOPRAGPipeline(RAGPipeline):
    def __init__(
        self,
        vectorstore: VectorStoreManager,
        llm=None,
        top_k: int = 5
    ):
        super().__init__(vectorstore, llm, top_k)
        self.rag_prompt = ChatPromptTemplate.from_template("""
Ты — ассистент по объектно-ориентированному программированию в python.
Используй только предоставленный контекст.
Если информации недостаточно, прямо скажи об этом.

Если вопрос касается ООП:
- дай краткое определение
- объясни идею простыми словами
- если возможно, укажи пример из контекста
- если уместно, свяжи ответ с принципами ООП

Контекст:
{context}

Вопрос: {question}

Ответ:
""")

    def query(
        self,
        question: str,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        result = super().query(question, include_sources)

        if result.get("success"):
            result["domain"] = "object_oriented_programming"
            result["oop_topics"] = self._detect_oop_topics(question)

        return result

    def _detect_oop_topics(self, question: str) -> Dict[str, bool]:
        lowered_question = question.lower()

        return {
            "classes_objects": any(keyword in lowered_question for keyword in ["класс", "объект", "class", "object"]),
            "inheritance": any(keyword in lowered_question for keyword in ["наследование", "inheritance"]),
            "polymorphism": any(keyword in lowered_question for keyword in ["полиморф", "polymorphism"]),
            "encapsulation": any(keyword in lowered_question for keyword in ["инкапсуляц", "encapsulation"]),
            "abstraction": any(keyword in lowered_question for keyword in ["абстракц", "abstraction"]),
            "interfaces": any(keyword in lowered_question for keyword in ["интерфейс", "interface", "abstract class", "абстрактный класс"]),
            "composition": any(keyword in lowered_question for keyword in ["композици", "агрегаци", "composition", "aggregation"])
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain_community.llms import YandexGPT
    from rag.document_loader import DocumentLoader
    from rag.chunking import ChunkingStrategy

    load_dotenv()

    loader = DocumentLoader(source_directory="./docs")
    documents = loader.load_document("./docs/oop.pdf")

    if not documents:
        print("No documents found")
    else:
        chunks = ChunkingStrategy.split_documents(
            documents,
            strategy="recursive",
            chunk_size=512,
            chunk_overlap=50
        )
        vectorstore = VectorStoreManager(
            persist_directory="./data/chroma_db",
            collection_name="oop_documents"
        )

        stats = vectorstore.add_documents(chunks)
        print(f"Добавил документ проверяй {stats['documents_added']}")

        iam_token = os.getenv("YANDEX_IAM_TOKEN")
        folder_id = os.getenv("YANDEX_FOLDER_ID")
        llm = None

        if iam_token and folder_id:
            llm = YandexGPT(
                temperature=0.3,
                max_tokens=500,
                folder_id=folder_id,
                iam_token=iam_token
            )
            print("yandexgpt подключен")
        else:
            print("yandexgpt не подключен(логично)")

        rag = OOPRAGPipeline(vectorstore=vectorstore, llm=llm, top_k=5)
        test_questions = ["Что такое наследование?", "Что такое абстракция?"]
        for i, question in enumerate(test_questions, 1):
            print("\n" + "=" * 60)
            print(f"Вопрос {i}: {question}")
            print("|/" * 10)

            result = rag.query(question)
            print(f"Ответ: {result['answer']}")
            print("|/" * 10)
            print()