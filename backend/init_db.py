r"""
数据库初始化脚本

【干什么的】
1. 创建所有数据表（如果不存在）
2. 创建初始管理员账号（admin / 123456）
3. 打印一张"体检报告"，告诉你表和账号的情况

【什么时候运行】
  · 第一次启动项目之前，跑一次
  · 以后每次改了数据表结构，也跑一次（已存在的表不会被破坏）

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe init_db.py

【为什么不用 Alembic 这种专业迁移工具】
Alembic 适合团队协作和线上环境（能精确控制每次结构变更）。
个人项目用 create_all + 重建数据库就够了，省掉一大堆配置文件。
将来真上线了再引入也不迟。
"""

import asyncio
import sys

# ------------------------------------------------------------------
# 强制标准输出使用 UTF-8。
# 原因见 run.py 里的详细说明：Windows 中文环境下用 GBK 编码时，
# ✅ 这类符号会让程序直接崩溃（实测踩到过）。
# ------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, ".")

from sqlalchemy import inspect, select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.database import AsyncSessionLocal, Base, engine  # noqa: E402
from app.core.logging import logger  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.models import User, UserRole  # noqa: E402

LINE = "=" * 68


async def create_tables() -> list[str]:
    """创建所有表，返回实际存在的表名列表"""
    # 这行导入至关重要：它触发 app/models/__init__.py，
    # 从而让 SQLAlchemy "见过"所有模型，知道要建哪些表。
    import app.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # 顺便查一下数据库里现在到底有哪些表
        tables = await conn.run_sync(lambda c: inspect(c).get_table_names())

    return sorted(tables)


async def create_admin() -> tuple[bool, str]:
    """
    创建初始管理员账号。

    返回值：(是否新建了, 说明文字)
    如果 admin 已存在，就**不覆盖**——否则你改了密码，下次跑这个脚本又被重置回 123456，
    那样就太危险了。
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(User).where(User.username == settings.ADMIN_USERNAME)
        )
        existing = result.scalar_one_or_none()

        if existing:
            return False, f"管理员账号已存在（用户名：{existing.username}），未做任何改动"

        admin = User(
            username=settings.ADMIN_USERNAME,
            password_hash=hash_password(settings.ADMIN_PASSWORD),
            nickname="系统管理员",
            role=UserRole.ADMIN,
            is_active=True,
        )
        db.add(admin)
        await db.commit()
        return True, f"管理员账号创建成功（用户名：{settings.ADMIN_USERNAME}）"


async def show_summary(tables: list[str]) -> None:
    """打印数据库体检报告"""
    print(f"\n{LINE}")
    print("  数据库体检报告")
    print(LINE)

    print(f"\n  数据库类型: {'SQLite（单文件，零安装）' if 'sqlite' in settings.DATABASE_URL else '其他'}")
    print(f"  数据库位置: {settings.DATABASE_URL}")

    print(f"\n  共创建 {len(tables)} 张表:")
    table_desc = {
        "users": "用户账号、密码哈希、角色",
        "sessions": "会话（一个用户可有多个会话）",
        "messages": "聊天消息（含引用的知识片段、耗时等）",
        "documents": "知识库文档（上传的原始文件）",
        "chunks": "知识片段（文档切分后的小段）",
        "feedbacks": "问答反馈（点赞/点踩）",
        "qa_logs": "问答日志（性能与质量统计数据）",
    }
    for i, t in enumerate(tables, 1):
        desc = table_desc.get(t, "（附加表）")
        print(f"    {i}. {t:<12} —— {desc}")

    # 统计各表数据量
    print("\n  各表数据量:")
    AsyncSessionLocal  # noqa: B018
    from sqlalchemy import func, text

    async with engine.begin() as conn:
        for t in tables:
            n = await conn.scalar(text(f"SELECT COUNT(*) FROM {t}"))
            print(f"    {t:<12} {n} 条")


async def main() -> None:
    print(f"\n{LINE}")
    print("  正在初始化数据库……".center(60))
    print(LINE)

    # ---------- 步骤 1：建表 ----------
    print("\n  [1/3] 创建数据表……")
    tables = await create_tables()
    print(f"        ✅ 完成，数据库中共有 {len(tables)} 张表")

    # ---------- 步骤 2：建管理员 ----------
    print("\n  [2/3] 创建初始管理员……")
    created, msg = await create_admin()
    print(f"        {'✅' if created else 'ℹ️ '} {msg}")
    if created:
        print(f"        用户名: {settings.ADMIN_USERNAME}")
        print(f"        密码  : {settings.ADMIN_PASSWORD}")
        print("        ⚠️ 请尽快登录后修改密码！")

    # ---------- 步骤 3：体检报告 ----------
    print("\n  [3/3] 生成体检报告……")
    await show_summary(tables)

    await engine.dispose()

    print(f"\n{LINE}")
    print("  ✅ 数据库初始化完成，可以启动后端服务了。")
    print(LINE + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        logger.exception("数据库初始化失败")
        print(f"\n❌ 初始化失败: {type(exc).__name__}: {exc}\n")
        sys.exit(1)
