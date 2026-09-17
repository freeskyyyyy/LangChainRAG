r"""
一键启动脚本

【怎么用】
    cd backend
    ..\.venv\Scripts\python.exe run.py

然后浏览器打开 http://127.0.0.1:8000/docs

【它帮你做了什么】
  · 检查 .env 配置和密钥是否就绪（省得启动到一半才报错）
  · 检查数据库是否初始化过
  · 启动 Uvicorn 服务器
  · 打印清晰的使用提示

【关于 --reload（自动重载）】
开发模式下，代码一保存服务器就自动重启，不用手动关了再开，非常方便。
但它会额外开一个进程监控文件变化，略微消耗性能，所以生产环境要关掉。
"""

import sys
from pathlib import Path

# ==================================================================
# 强制标准输出使用 UTF-8 编码
# ==================================================================
# 【为什么必须加这一段——这是实测踩出来的坑】
#
# 脚本里用了一些符号来美化输出，比如 📖 ❌ ✅。
#
# 直接在这个命令行里跑时，Python 会用 Windows 的 Unicode 控制台 API，
# 这些符号能正常显示，一切正常。
#
# 但**通过 .bat 文件启动时**（也就是你双击的时候），标准输出被重定向了，
# Python 就改用系统默认编码 —— 中文 Windows 上是 **GBK**。
# 而 GBK 字符集里**根本没有 📖 这些符号**，于是：
#
#     UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f4d6'
#
# 程序当场崩溃，服务起不来。
#
# 【解决办法】
# 主动把输出流重新配置成 UTF-8，这样不管输出到哪里（控制台、管道、文件）
# 都能正确编码。errors="replace" 是双保险：万一真有编码不了的字符，
# 用问号代替而不是让整个程序崩溃。
#
# 这段代码看起来不起眼，但它是"双击就能用"和"双击就崩溃"的分界线。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# 把当前目录（backend/）加入模块搜索路径，
# 这样 `from app.xxx import` 才能正常工作
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import uvicorn  # noqa: E402

from app.core.config import settings  # noqa: E402

LINE = "=" * 68


def preflight_check() -> bool:
    """
    启动前检查。有任何一项没准备好就给出明确提示，而不是让程序跑起来再报错。
    """
    problems: list[str] = []

    # ---------- 检查 1：配置文件 ----------
    env_file = BACKEND_DIR / ".env"
    if not env_file.exists():
        problems.append(
            "缺少配置文件 backend/.env\n"
            "     解决方法：把 backend/.env.example 复制一份改名为 .env，然后填入 API Key"
        )

    # ---------- 检查 2：API Key ----------
    key = settings.DASHSCOPE_API_KEY
    if not key or key.startswith("sk-请填入") or len(key) < 20:
        problems.append(
            "配置里的 DASHSCOPE_API_KEY 无效或未填写\n"
            "     解决方法：去 https://bailian.console.aliyun.com 创建 API Key，填入 backend/.env"
        )

    # ---------- 检查 3：数据库是否初始化 ----------
    db_file = BACKEND_DIR / "data" / "app.db"
    if not db_file.exists():
        problems.append(
            "数据库还没初始化\n"
            "     解决方法：先运行一次  ..\\.venv\\Scripts\\python.exe init_db.py"
        )

    if problems:
        print(f"\n{LINE}")
        print("  ❌ 启动前检查未通过")
        print(LINE)
        for i, p in enumerate(problems, 1):
            print(f"\n  {i}. {p}")
        print(f"\n{LINE}\n")
        return False

    return True


def main() -> None:
    if not preflight_check():
        sys.exit(1)

    host = "127.0.0.1"
    port = 8000

    print(f"\n{LINE}")
    print(f"  {settings.APP_NAME}".center(66))
    print(LINE)
    print(f"\n  正在启动服务…\n")
    print(f"  📖 接口文档页   http://{host}:{port}/docs     ← 在浏览器打开这个测接口")
    print(f"  📄 另一种文档   http://{host}:{port}/redoc")
    print(f"  ❤️  健康检查     http://{host}:{port}/api/health")
    print(f"  🔑 管理员账号   {settings.ADMIN_USERNAME} / {settings.ADMIN_PASSWORD}")
    print(f"\n  模型配置：对话={settings.LLM_MODEL} | "
          f"向量={settings.EMBEDDING_MODEL} | 重排={settings.RERANK_MODEL}")
    print(f"\n  ⏹  按 Ctrl+C 停止服务")
    print(f"\n{LINE}\n")

    # ---------------- 启动服务器 ----------------
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        # ⚠️ 这里原本写的是 reload=settings.DEBUG，等于"永远开着热重载"。
        # 已改为独立的 RELOAD 开关，默认关 —— 原因见 config.py 里的长注释
        #（热重载会留下杀不干净的僵尸进程，用旧代码悄悄应答，极难排查）。
        reload=settings.RELOAD,
        reload_dirs=[str(BACKEND_DIR / "app")],
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False,           # 关掉 Uvicorn 自带的访问日志（我们有自己的，更清晰）
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n  服务已停止。\n")
    except Exception as exc:
        print(f"\n❌ 启动失败: {type(exc).__name__}: {exc}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
