"""
测试配置 - pytest fixtures
"""
import os
import sys

import pytest

# 将 backend 目录加入 sys.path，使测试中可使用 `from services.xxx import ...`
# 的包内导入（与源码结构保持一致）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# settings 单例必须在 sys.path 就绪后再导入
from core.config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """为所有测试 mock 关键配置，防止误操作"""
    # 注意: monkeypatch.setattr 不支持 "模块.属性.子属性" 字符串路径，
    # 应直接对 settings 实例属性打补丁（pydantic BaseSettings 实例可写）。
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(settings, "VECTOR_DB_PATH", "./test_chroma_db")
    monkeypatch.setattr(settings, "DB_PATH", "./test_data/chat.db")
    monkeypatch.setattr(settings, "API_AUTH_ENABLED", False)


@pytest.fixture
def sample_documents():
    """测试用文档列表"""
    from langchain_core.documents import Document
    return [
        Document(
            page_content="Python 是一门高级编程语言，以简洁易读著称，广泛应用于 AI 和数据科学领域。",
            metadata={"source": "python_intro.pdf", "page": 1},
        ),
        Document(
            page_content="RAG (Retrieval-Augmented Generation) 结合了检索和生成两种技术。",
            metadata={"source": "rag_paper.pdf", "page": 2},
        ),
        Document(
            page_content="机器学习是人工智能的一个分支，通过数据训练模型来做出预测。",
            metadata={"source": "ml_basics.pdf", "page": 1},
        ),
        Document(
            page_content="LangChain 是一个用于构建 LLM 应用的框架，支持链式调用和 Agent。",
            metadata={"source": "langchain_doc.pdf", "page": 3},
        ),
        Document(
            page_content="FastAPI 是 Python 的现代 Web 框架，支持异步处理和自动 API 文档生成。",
            metadata={"source": "fastapi_doc.pdf", "page": 1},
        ),
    ]
