r"""个人设置接口回归；使用独立内存数据库，不修改实际账号或知识库。

运行：..\.venv\Scripts\python.exe -X utf8 -m unittest test_account -v
浏览器验收：同一命令去掉 -m unittest，改为 test_account.py --serve。
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys
import unittest

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.auth import router as auth_router
from app.api.sessions import router as sessions_router
from app.core.database import Base, get_db
from app.core.exceptions import register_exception_handlers


def create_test_app():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def lifespan(app):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        try:
            yield
        finally:
            await engine.dispose()

    async def test_db():
        async with sessions() as db:
            yield db

    app = FastAPI(lifespan=lifespan)
    app.dependency_overrides[get_db] = test_db
    register_exception_handlers(app)
    app.include_router(auth_router)
    app.include_router(sessions_router)
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/app")
    async def page():
        return FileResponse(static / "index.html")

    return app


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.client = self.enterContext(TestClient(create_test_app()))
        self.username = "account_" + self._testMethodName
        response = self.client.post("/api/auth/register", json={
            "username": self.username, "password": "Original123", "nickname": "原昵称",
        })
        self.assertEqual(response.status_code, 201)
        login = self.client.post("/api/auth/login", json={
            "username": self.username, "password": "Original123",
        })
        self.assertEqual(login.status_code, 200)
        self.headers = {"Authorization": "Bearer " + login.json()["access_token"]}

    def change_password(self, old="Original123", new="Changed456"):
        return self.client.post("/api/auth/change-password", headers=self.headers, json={
            "old_password": old, "new_password": new,
        })

    def test_nickname_persists(self):
        response = self.client.patch("/api/auth/me", headers=self.headers,
                                     json={"nickname": "  新昵称🌿  "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["nickname"], "新昵称🌿")
        user = self.client.get("/api/auth/me", headers=self.headers).json()
        self.assertEqual(user["nickname"], "新昵称🌿")
        self.assertEqual(user["username"], self.username)
        self.assertEqual(user["role"], "user")
        self.assertNotIn("password_hash", user)

    def test_wrong_old_password(self):
        response = self.change_password(old="Incorrect123")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["code"], 40120)
        self.assertEqual(self.client.get("/api/auth/me", headers=self.headers).status_code, 200)
        response = self.client.post("/api/auth/login", json={
            "username": self.username, "password": "Original123",
        })
        self.assertEqual(response.status_code, 200)

    def test_new_password_login(self):
        self.assertEqual(self.change_password().status_code, 200)
        for password, expected in [("Original123", 401), ("Changed456", 200)]:
            with self.subTest(password_version=expected):
                response = self.client.post("/api/auth/login", json={
                    "username": self.username, "password": password,
                })
                self.assertEqual(response.status_code, expected)

    def test_password_validation(self):
        for password, expected in [("short", 422), ("密" * 25, 422),
                                   (" Spaces123", 422), ("Original123", 400)]:
            with self.subTest(expected=expected, length=len(password)):
                self.assertEqual(self.change_password(new=password).status_code, expected)

    def test_invalid_token(self):
        response = self.client.post("/api/auth/change-password",
                                    headers={"Authorization": "Bearer invalid"},
                                    json={"old_password": "Original123", "new_password": "Changed456"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["code"], 40102)
        self.assertEqual(self.client.patch("/api/auth/me", json={"nickname": "其他人"}).status_code, 401)

    def test_profile_validation(self):
        response = self.client.patch("/api/auth/me", headers=self.headers, json={"nickname": "字" * 51})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.get("/api/auth/me", headers=self.headers).json()["nickname"], "原昵称")


if __name__ == "__main__":
    if "--serve" in sys.argv:
        import uvicorn
        uvicorn.run(create_test_app(), host="127.0.0.1", port=8766)
    else:
        unittest.main()
