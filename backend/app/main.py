r"""
FastAPI 应用主入口 —— 把所有零件组装起来

【这个文件干什么的】
如果把这个项目比作一家餐厅：
  · app/api/      是各个档口的厨师（负责具体菜品）
  · app/services/ 是配菜和调料（业务规则）
  · app/rag/      是核心的秘制配方（RAG 引擎）
  · app/core/     是水电煤气（基础设施）
  · main.py       就是**店长**——把所有人和设备组织起来，开门营业。

【怎么启动】
    cd backend
    ..\.venv\Scripts\python.exe run.py
或者直接：
    ..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload

启动后打开 http://127.0.0.1:8000/docs 就能看到自动生成的接口文档页。
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.cache import get_cache
from app.core.config import settings
from app.core.database import close_db, init_db
from app.core.exceptions import register_exception_handlers
from app.core.logging import logger, setup_logging
from app.core.rate_limit import get_limiter
from app.schemas.common import StatsResponse

APP_VERSION = "1.0.0"


# ==================================================================
# 生命周期管理 —— 程序"开门"和"打烊"时要做的事
# ==================================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    【什么是 lifespan】
    FastAPI 提供的一个"钩子"：
      · yield 之前的代码 → 程序启动时执行一次
      · yield 之后的代码 → 程序关闭时执行一次

    什么时候用？凡是"只需要做一次、且必须做完才能开始服务"的事情：
      · 初始化日志系统
      · 建数据库表
      · 预热缓存
      · 建立连接池

    为什么不能写在模块顶层？因为那里是同步的，而数据库操作是异步的。
    lifespan 支持 async，能正常 await。

    【如果一个启动步骤失败了会怎样】
    这里会直接抛异常，程序拒绝启动。
    这是**故意的**——比如数据库连不上，与其带病运行让用户看到一堆报错，
    不如立刻启动失败并给出明确原因。这叫"快速失败"（Fail Fast）。
    """
    # ---------------- 启动 ----------------
    print(f"\n{'=' * 68}")
    print(f"  {settings.APP_NAME} 正在启动…")
    print(f"{'=' * 68}\n")

    setup_logging()
    logger.info(f"应用启动 | 版本={APP_VERSION} | 环境={settings.APP_ENV}")

    # 确保数据目录都存在
    settings.ensure_dirs()

    # 建表（已存在的表不会被改动）
    await init_db()
    logger.info("数据库已就绪")

    # 初始化缓存（提前建好，避免第一个请求时才知道有问题）
    get_cache()
    get_limiter()

    print(f"\n{'=' * 68}")
    print(f"  ✅ 启动完成")
    print(f"  接口文档: http://127.0.0.1:8000/docs")
    print(f"  健康检查: http://127.0.0.1:8000/api/health")
    print(f"{'=' * 68}\n")

    yield  # ← 程序在此期间对外提供服务

    # ---------------- 关闭 ----------------
    logger.info("应用正在关闭…")
    await close_db()
    logger.info("应用已关闭")


# ==================================================================
# 创建应用
# ==================================================================
def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""

    app = FastAPI(
        title=settings.APP_NAME,
        version=APP_VERSION,
        description=(
            "基于 LangChain 的企业级 RAG 知识库问答系统。\n\n"
            "**功能**：多用户多会话 · 知识库管理 · 流式问答 · 引用溯源\n\n"
            "**测试步骤**：先调用 `/api/auth/login` 用 admin / 123456 登录，"
            "拿到令牌后点右上角 Authorize 填入，即可调试其他接口。"
        ),
        docs_url="/docs",           # 交互式文档页（可以直接在这里测接口）
        redoc_url="/redoc",         # 另一种风格的文档页
        openapi_url="/openapi.json",
        lifespan=lifespan,
        # 开发环境才开放文档页。生产环境关掉可以少暴露一些信息给攻击者。
        debug=settings.DEBUG,
    )

    # ---------------- 1. CORS 跨域配置 ----------------
    # 【什么是跨域】前端跑在 5173 端口（Vite），后端跑在 8000 端口（FastAPI）。
    # 浏览器有个安全规则：不同"源"（协议+域名+端口）之间默认不能互相请求。
    # 所以后端必须明确告诉浏览器"我允许 5173 来访问我"，否则前端一个请求都发不出去。
    #
    # allow_origins 里列的是"允许哪些地址来访问"。
    # ⚠️ 生产环境绝不能用 ["*"]（允许任何网站），那等于把接口开放给全世界。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",    # Vite 开发服务器默认端口
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,                      # 允许携带 Cookie/认证信息
        allow_methods=["*"],                         # 允许所有 HTTP 方法
        allow_headers=["*"],                         # 允许所有请求头
        expose_headers=["X-Process-Time"],           # 允许前端读取这个自定义响应头
    )

    # ---------------- 2. 请求日志中间件 ----------------
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """
        记录每个请求的耗时。

        【为什么要记耗时】
        没有这个，你只知道"系统有时候慢"，但不知道慢在哪。
        有了它，翻日志就能看到"哪个接口花了 3 秒"。

        同时把耗时写进响应头 X-Process-Time，前端在浏览器开发者工具的
        网络面板里就能直接看到每个请求花了多久。
        """
        start = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{elapsed_ms:.1f}ms"

        # 只记录慢请求和出错的请求，避免日志被正常请求刷屏
        if elapsed_ms > 1000 or response.status_code >= 400:
            logger.info(
                f"{request.method} {request.url.path} "
                f"→ {response.status_code} ({elapsed_ms:.0f}ms)"
            )

        return response

    # ---------------- 3. 注册统一异常处理 ----------------
    # 必须在创建 app 之后、启动之前完成，否则异常不会被接管
    register_exception_handlers(app)

    # ---------------- 4. 挂载路由 ----------------
    # 延迟导入：避免"模块顶层就 import，导致循环依赖"的问题
    from app.api import auth, chat, knowledge_base, sessions

    app.include_router(auth.router)            # 认证：注册/登录/改密码
    app.include_router(sessions.router)        # 会话：增删改查
    app.include_router(chat.router)            # 问答：流式问答 + 反馈
    app.include_router(knowledge_base.router)  # 知识库：文档管理（仅管理员）

    # ---------------- 5. 前端页面 ----------------
    # 静态资源目录。里面放的都是**本来就该公开**的前端文件
    # （html / css / js），没有任何密钥或用户数据。
    # 【安全上要注意的一件事】
    # 这个目录以后**绝对不能**放 .env、数据库文件、用户上传的原始文件。
    # 挂载目录等于把里面所有文件都挂到公网上，
    # 谁猜到文件名谁就能下载。以后要加文件时请记住这一条。
    STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ---------------- 6. 系统接口 ----------------
    @app.get("/", tags=["系统"], summary="首页")
    async def root() -> dict:
        """访问根路径时给个提示，避免看到 404 让人困惑"""
        return {
            "app": settings.APP_NAME,
            "version": APP_VERSION,
            "web": "/app",
            "demo": "/demo",
            "docs": "/docs",
            "message": "服务运行中。浏览器访问 /app 使用问答界面，访问 /docs 查看接口文档",
        }

    @app.get("/demo", tags=["系统"], summary="测试页面（浏览器直接使用）", include_in_schema=False)
    async def demo_page() -> FileResponse:
        """
        提供一个"开箱即用"的网页测试界面。

        【为什么要有这个】
        正式的 Vue 前端还没做，但后端功能已经完整了。
        如果只能通过 /docs 那种开发者工具来测，很不直观——
          · 看不到打字机效果
          · 看不到引用卡片长什么样
          · 没法上传文件

        所以先给一个单文件 HTML，让它"现在就能用"。
        等 Vue 前端做完后，这个页面可以留着当调试工具。

        【为什么不用 StaticFiles 挂载整个目录】
        挂载目录会把目录下所有文件都暴露出去（包括我们不想公开的东西）。
        这里只明确暴露**一个文件**，最小化暴露面。
        """
        page = STATIC_DIR / "demo.html"

        if not page.exists():
            raise HTTPException(
                status_code=404,
                detail=f"测试页面不存在：{page}",
            )

        return FileResponse(page, media_type="text/html")

    @app.get("/app", tags=["系统"], summary="问答主界面（浏览器直接使用）", include_in_schema=False)
    async def web_app() -> FileResponse:
        """
        正式的问答界面（三个文件组成：index.html + app.css + app.js）。

        【和 /demo 的区别】
          /demo  —— 最初那个单文件测试页，所有代码挤在一个文件里
          /app   —— 正式界面，结构、样式、逻辑分成三个文件

        两个都保留：
          · 日常使用走 /app
          · 想排查"是不是前端代码写错了"时，用 /demo 对照，
            因为 /demo 是完全独立的，不受新代码影响
        """
        page = STATIC_DIR / "index.html"

        if not page.exists():
            raise HTTPException(status_code=404, detail=f"页面不存在：{page}")

        return FileResponse(page, media_type="text/html")

    @app.get(
        "/api/health",
        response_model=StatsResponse,
        tags=["系统"],
        summary="健康检查",
        description="返回服务状态和各组件信息，用于监控和排查问题。",
    )
    async def health() -> StatsResponse:
        """
        健康检查。

        【这个接口有什么用】
          · 部署后确认服务是否活着
          · 排查问题时快速看清"缓存是不是空的""用的哪个模型"
          · 前端可以在页面右下角显示一个"服务正常"的小绿点

        ⚠️ 注意这里**不返回**任何密钥信息，只返回组件名称。
        """
        cache_stats = get_cache().stats()
        limiter_stats = get_limiter().stats()

        db_kind = "SQLite" if "sqlite" in settings.DATABASE_URL else "其他"

        return StatsResponse(
            status="ok",
            app_name=settings.APP_NAME,
            version=APP_VERSION,
            database=db_kind,
            cache=cache_stats,
            rate_limit=limiter_stats,
            llm_model=settings.LLM_MODEL,
            embedding_model=settings.EMBEDDING_MODEL,
            rerank_model=settings.RERANK_MODEL,
        )

    return app


# ------------------------------------------------------------------
# 模块级变量：uvicorn 通过 "app.main:app" 找到它
# ------------------------------------------------------------------
app = create_app()
