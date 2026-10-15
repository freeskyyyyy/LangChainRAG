"""问答持久化与缓存控制回归；内存数据库和替代生成器，不调用模型。"""

import asyncio
from contextlib import asynccontextmanager, ExitStack
import json
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine

from app.api.chat import router
from app.core.database import upgrade_schema
from app.core.security import hash_password
from app.models import ChatSession, Message, User
from test_account import create_test_app as account_app


class TestChain:
    def __init__(self):
        self.calls = []

    async def astream(self, question, history, use_cache, top_k):
        self.calls.append({"question": question, "history": history, "use_cache": use_cache})
        if question in {"引用校验", "资料引用校验"}:
            from tests.test_rag_regression import make_chain, chunk
            chunks = [chunk(), chunk()] if question == "资料引用校验" else []
            answer = "保养说明[1][99]，代码 `[1]`。" if chunks else "按通用知识回答：说明[1][99]，代码 `[1]`。"
            chain, _ = make_chain(chunks, answer)
            async for event in chain.astream(question, use_cache=False):
                yield event
            return
        await asyncio.sleep(.02)
        source = "system" if "模型" in question else "knowledge_base" if "资料" in question else "general"
        citations = [{"index": 1, "file_name": "测试.txt", "content": "测试资料", "score": .5}] if source == "knowledge_base" else []
        yield {"event": "token", "data": {"text": ("缓存回答" if use_cache else "新回答") + ("[1]" if citations else "")}}
        yield {"event": "citations", "data": citations}
        yield {"event": "done", "data": {"knowledge_source": source, "cache_hit": use_cache,
                                          "no_answer": source == "general", "latency_ms": 20}}


def create_test_app():
    app = account_app()
    app.include_router(router)
    original = app.router.lifespan_context
    chain = TestChain()
    app.state.test_chain = chain

    @asynccontextmanager
    async def lifespan(app):
        async with original(app):
            async with app.state.test_sessions() as db:
                db.add(User(username="chat_test", nickname="问答验收", role="user", is_active=True,
                            password_hash=hash_password("Test123456")))
                await db.commit()
            with ExitStack() as stack:
                stack.enter_context(patch("app.api.chat.AsyncSessionLocal", app.state.test_sessions))
                stack.enter_context(patch("app.rag.chain.get_rag_chain", return_value=chain))
                stack.enter_context(patch("app.api.chat.check_rate_limit"))
                yield

    app.router.lifespan_context = lifespan
    return app


def events(response):
    result = []
    kind = None
    for line in response.text.splitlines():
        if line.startswith("event:"):
            kind = line[6:].strip()
        elif line.startswith("data:"):
            result.append({"event": kind, "data": json.loads(line[5:])})
    return result


class ChatReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.app = create_test_app()
        self.client = self.enterContext(TestClient(self.app))
        login = self.client.post("/api/auth/login", json={"username": "chat_test", "password": "Test123456"})
        self.headers = {"Authorization": "Bearer " + login.json()["access_token"]}

    def ask(self, question, **kwargs):
        response = self.client.post("/api/chat/stream", headers=self.headers, json={"question": question, **kwargs})
        self.assertEqual(response.status_code, 200)
        result = events(response)
        self.assertNotIn("error", [event["event"] for event in result])
        return result

    def history(self, session_id):
        response = self.client.get(f"/api/sessions/{session_id}/messages", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_answer_sources_survive_reload(self):
        for question, source in [("通用问题", "general"), ("什么模型", "system"), ("资料问题", "knowledge_base")]:
            with self.subTest(source=source):
                result = self.ask(question)
                saved = next(event["data"] for event in result if event["event"] == "saved")
                messages = self.history(saved["session_id"])
                self.assertEqual(messages[-1]["knowledge_source"], source)
                self.assertIsNone(messages[0]["knowledge_source"])

    def test_regenerate_bypasses_cache_and_replaces_answer(self):
        result = self.ask("通用问题")
        session_id = next(e["data"]["session_id"] for e in result if e["event"] == "saved")
        result = self.ask("通用问题", session_id=session_id, regenerate=True, use_cache=True)
        done = next(e["data"] for e in result if e["event"] == "done")
        self.assertFalse(done["cache_hit"])
        self.assertFalse(self.app.state.test_chain.calls[-1]["use_cache"])
        self.assertEqual(self.app.state.test_chain.calls[-1]["history"], [])
        history = self.history(session_id)
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertEqual(history[-1]["content"], "新回答")
        self.assertEqual(history[-1]["knowledge_source"], "general")

    def seed_history(self, count):
        async def seed():
            async with self.app.state.test_sessions() as db:
                session = ChatSession(user_id=1, title="长对话")
                db.add(session)
                await db.flush()
                for index in range(count):
                    db.add(Message(session_id=session.id, role="user" if index % 2 == 0 else "assistant",
                                   content=f"历史{index}"))
                await db.commit()
                return session.id
        return self.client.portal.call(seed)

    def test_recent_messages_are_returned_in_chronological_order(self):
        session_id = self.seed_history(205)
        history = self.history(session_id)
        self.assertEqual(len(history), 200)
        self.assertEqual((history[0]["content"], history[-1]["content"]), ("历史5", "历史204"))
        self.assertTrue(all(m["knowledge_source"] is None for m in history))

    def test_generation_receives_latest_ten_messages(self):
        session_id = self.seed_history(14)
        self.ask("继续这个话题", session_id=session_id)
        history = self.app.state.test_chain.calls[-1]["history"]
        self.assertEqual([m["content"] for m in history], [f"历史{i}" for i in range(4, 14)])

    def test_validated_tokens_match_persisted_history(self):
        for question, source, count in [("引用校验", "general", 0), ("资料引用校验", "knowledge_base", 1)]:
            with self.subTest(source=source):
                result = self.ask(question)
                saved = next(e["data"] for e in result if e["event"] == "saved")
                content = "".join(e["data"]["text"] for e in result if e["event"] == "token")
                history = self.history(saved["session_id"])[-1]
                self.assertEqual(history["content"], content)
                self.assertNotIn("[99]", content)
                self.assertIn("`[1]`", content)
                self.assertEqual(history["knowledge_source"], source)
                self.assertEqual(len(history["citations"]), count)


class SchemaUpgradeTests(unittest.IsolatedAsyncioTestCase):
    async def test_upgrade_old_schema_preserves_rows_and_is_idempotent(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with engine.begin() as conn:
                await conn.exec_driver_sql("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
                await conn.exec_driver_sql("INSERT INTO messages (id, content) VALUES (1, 'original message')")
                await conn.run_sync(upgrade_schema)
                await conn.run_sync(upgrade_schema)
                rows = (await conn.exec_driver_sql("SELECT id, content, knowledge_source FROM messages")).all()
                self.assertEqual(rows, [(1, "original message", None)])
                await conn.exec_driver_sql("INSERT INTO messages (id, content, knowledge_source) VALUES (2, 'new message', 'general')")
                self.assertEqual((await conn.exec_driver_sql("SELECT knowledge_source FROM messages WHERE id=2")).scalar(), "general")
        finally:
            await engine.dispose()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        import uvicorn
        uvicorn.run(create_test_app(), host="127.0.0.1", port=8768)
    else:
        unittest.main()
