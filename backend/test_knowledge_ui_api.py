"""知识库管理回归：内存数据库 + 替代索引服务，不接触真实文件、向量库或模型。

运行 unittest；加 --serve 在 8767 提供真实页面和接口供浏览器验收。
测试账号 kb_admin / Test123456、kb_user / Test123456 仅存在于测试服务。
"""

import asyncio
from contextlib import asynccontextmanager, ExitStack
import sys
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.api.knowledge_base import router
from app.core.database import get_db
from app.core.security import hash_password
from app.models import Chunk, Document, User
from app.rag.ingest import IngestResult
from test_account import create_test_app as account_app


class TestIndexService:
    def __init__(self):
        self.calls = 0

    async def reindex(self, db, doc):
        self.calls += 1
        await asyncio.sleep(.25)  # 给浏览器检查处理中状态的时间。
        if doc.id == 22:
            return IngestResult(success=False, error="原始文件不存在，无法重新索引")
        await db.execute(delete(Chunk).where(Chunk.document_id == doc.id))
        for i in range(2):
            db.add(Chunk(document_id=doc.id, chunk_index=i, parent_index=0,
                         content=f"重建后的片段 {i + 1}", parent_content="重建后的完整上下文",
                         char_count=9, vector_id=f"test-new-{doc.id}-{i}"))
        doc.status = "indexed"
        doc.chunk_count = 2
        doc.parent_chunk_count = 1
        doc.char_count = 18
        doc.error_msg = None
        await db.commit()
        return IngestResult(success=True, doc_id=doc.id, chunk_count=2)


def create_test_app():
    app = account_app()
    app.include_router(router)
    original_lifespan = app.router.lifespan_context
    service = TestIndexService()
    app.state.test_index = service

    @asynccontextmanager
    async def lifespan(app):
        async with original_lifespan(app):
            async for db in app.dependency_overrides[get_db]():
                for role in ("admin", "user"):
                    db.add(User(username=f"kb_{role}", password_hash=hash_password("Test123456"),
                                nickname="知识库验收", role=role, is_active=True))
                for i in range(1, 23):
                    db.add(Document(id=i, file_name=("服饰资料 <img src=x onerror=alert(1)>.txt" if i == 21 else f"资料{i}.txt"),
                                    stored_name=f"test-{i}.txt", file_type="txt", file_size=100,
                                    file_path=f"nonexistent-test-file-{i}",
                                    status="failed" if i == 22 else "parsing" if i == 20 else "indexed",
                                    error_msg="测试文件丢失" if i == 22 else None,
                                    chunk_count=23 if i == 21 else 0,
                                    parent_chunk_count=1 if i == 21 else 0,
                                    char_count=230 if i == 21 else 0))
                await db.flush()
                for i in range(23):
                    db.add(Chunk(document_id=21, chunk_index=i, parent_index=0,
                                 content=f"第 {i + 1} 段：面料保养说明\n<img src=x onerror=alert(1)>" + "长文本" * 20,
                                 parent_content="父片段上下文\n<script>alert(1)</script>",
                                 page=3, sheet_name="面料 <b>资料</b>", char_count=100,
                                 vector_id=f"test-old-{i}"))
                await db.commit()
            with ExitStack() as stack:
                stack.enter_context(patch("app.rag.ingest.get_ingest_service", return_value=service))
                stack.enter_context(patch("app.rag.vectorstore.get_vector_store", return_value=Mock(count=lambda: 23)))
                stack.enter_context(patch("app.rag.bm25.get_bm25_index", return_value=Mock(get_stats=lambda: {})))
                stack.enter_context(patch("app.rag.semantic_cache.get_semantic_cache", return_value=Mock(get_stats=lambda: {})))
                yield

    app.router.lifespan_context = lifespan

    @app.get("/api/health")
    async def health():
        return {"llm_model": "隔离验收（不调用模型）"}

    return app


class KnowledgeUiApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_test_app()
        self.client = self.enterContext(TestClient(self.app))
        self.admin = self.login("admin")
        self.user = self.login("user")

    def login(self, role):
        response = self.client.post("/api/auth/login", json={"username": f"kb_{role}", "password": "Test123456"})
        self.assertEqual(response.status_code, 200)
        return {"Authorization": "Bearer " + response.json()["access_token"]}

    def test_document_pagination(self):
        first = self.client.get("/api/kb/documents?page=1&page_size=20", headers=self.admin).json()
        last = self.client.get("/api/kb/documents?page=2&page_size=20", headers=self.admin).json()
        self.assertEqual((first["total"], first["total_pages"], len(first["items"])), (22, 2, 20))
        self.assertEqual([d["id"] for d in last["items"]], [2, 1])

    def test_chunk_pagination_and_metadata(self):
        first = self.client.get("/api/kb/documents/21/chunks?page_size=20", headers=self.admin).json()
        last = self.client.get("/api/kb/documents/21/chunks?page=2&page_size=20", headers=self.admin).json()
        self.assertEqual((first["total"], first["total_pages"]), (23, 2))
        self.assertEqual([c["chunk_index"] for c in last["items"]], [20, 21, 22])
        self.assertIn("<img", first["items"][0]["content"])
        self.assertIn("<script>", first["items"][0]["parent_content"])
        self.assertEqual(first["items"][0]["page"], 3)

    def test_empty_failed_missing_and_invalid_page(self):
        empty = self.client.get("/api/kb/documents/1/chunks", headers=self.admin).json()
        failed = self.client.get("/api/kb/documents/22/chunks", headers=self.admin).json()
        self.assertEqual(empty["items"], [])
        self.assertEqual(failed["document"]["error_msg"], "测试文件丢失")
        self.assertEqual(self.client.get("/api/kb/documents/999/chunks", headers=self.admin).status_code, 404)
        self.assertEqual(self.client.get("/api/kb/documents/21/chunks?page=0", headers=self.admin).status_code, 422)

    def test_admin_only(self):
        for method, path in [("get", "/api/kb/documents"), ("get", "/api/kb/documents/21/chunks"),
                             ("post", "/api/kb/documents/21/reindex")]:
            with self.subTest(path=path):
                self.assertEqual(getattr(self.client, method)(path).status_code, 401)
                self.assertEqual(getattr(self.client, method)(path, headers=self.user).status_code, 403)
        self.assertEqual(self.app.state.test_index.calls, 0)

    def test_reindex_refreshes_preview(self):
        result = self.client.post("/api/kb/documents/21/reindex", headers=self.admin)
        self.assertEqual(result.status_code, 200)
        preview = self.client.get("/api/kb/documents/21/chunks", headers=self.admin).json()
        self.assertEqual(preview["total"], 2)
        self.assertEqual(preview["document"]["chunk_count"], 2)
        self.assertEqual(preview["items"][0]["content"], "重建后的片段 1")

    def test_reindex_failure(self):
        result = self.client.post("/api/kb/documents/22/reindex", headers=self.admin)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.json()["code"], 40045)
        self.assertIn("原始文件不存在", result.json()["message"])
        self.assertEqual(self.client.post("/api/kb/documents/999/reindex", headers=self.admin).status_code, 404)


if __name__ == "__main__":
    if "--serve" in sys.argv:
        import uvicorn
        uvicorn.run(create_test_app(), host="127.0.0.1", port=8767)
    else:
        unittest.main()
