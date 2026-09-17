"""
数据库连接模块 —— 负责建立和维护与数据库的连接

【这个文件干什么的】
1. 创建「数据库引擎」——可以理解成数据库的连接池/管家
2. 定义所有数据表都要继承的基类 Base
3. 提供一个 get_db() 函数，每个接口请求都会拿到一个独立的数据库会话，
   请求结束自动关闭，不会泄漏连接

【为什么用 async（异步）】
FastAPI 是异步框架。如果数据库操作用同步写法，一个请求在等数据库返回时，
整个服务器都得干等着，其他人就得排队。异步写法让等待期间服务器能去处理别人的请求。
这就是「高并发」的关键。

【关于 SQLite 和 MySQL 的切换】
下面 _normalize_db_url() 会把你配置里的相对路径转成绝对路径。
因为 SQLite 的路径是相对于「启动程序时所在的目录」，如果你从别的地方
启动程序，就会找不到数据库文件。转成绝对路径就一劳永逸了。

将来要换 MySQL，只需改 .env 里的一行：
    DATABASE_URL=mysql+aiomysql://用户名:密码@localhost:3306/rag_kb?charset=utf8mb4
（记得先 pip install aiomysql）
"""

from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import BACKEND_DIR, settings


def _normalize_db_url(url: str) -> str:
    """
    把配置里的数据库地址规范化。

    主要处理 SQLite 的相对路径问题：
        sqlite+aiosqlite:///./data/app.db
    会被转换成绝对路径，例如：
        sqlite+aiosqlite:///D:/Coding Program/LangChainRAG项目/backend/data/app.db
    这样无论从哪个目录启动程序，都能正确找到数据库文件。
    """
    prefix = "sqlite+aiosqlite:///"
    if not url.startswith(prefix):
        return url  # 不是 SQLite（比如 MySQL），原样返回

    raw_path = url[len(prefix):]
    db_path = Path(raw_path)

    if not db_path.is_absolute():
        db_path = (BACKEND_DIR / raw_path.lstrip("./")).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)

    # Windows 路径要转成正斜杠，SQLAlchemy 才认
    return f"{prefix}{db_path.as_posix()}"


DATABASE_URL = _normalize_db_url(settings.DATABASE_URL)


# ------------------------------------------------------------------
# 数据库引擎
#   echo=DEBUG 时会把执行的 SQL 打印到控制台，方便开发时看清程序到底查了什么。
#   上线时要关掉（太吵且影响性能）。
# ------------------------------------------------------------------
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,  # 取连接前先 ping 一下，防止用到已失效的连接
)


# ------------------------------------------------------------------
# 打开 SQLite 的外键约束 —— 不加这一段，所有 ON DELETE CASCADE 都是摆设
# ------------------------------------------------------------------
# 【踩过的坑：外键约束默认是关的】
# 我们的表定义里写了不少 `ondelete="CASCADE"`，比如
#     删掉一条消息 → 挂在它上面的"点赞点踩"应该跟着消失
# 看起来是数据库在保证这件事，代码里也这么注释了。
#
# 但 SQLite 有个很坑的设计：**外键约束默认关闭**，
# 必须对**每一条连接**单独执行 PRAGMA foreign_keys=ON 才会生效。
# 不开的话，那些 CASCADE 就只是一句注释，删父记录时子记录原封不动地留着。
#
# 【它造成的真实后果（实测出来的）】
# 删掉一条消息后，那条消息上的"踩"留了下来变成孤儿记录。
# 而 SQLite 分配新编号的规则是"当前最大号 + 1"——
# 于是**刚被删掉的号会被下一条新消息捡走**，
# 那条没主的"踩"就顺势贴到了新消息上。
#
# 表现出来就是：用户重新生成了一次回答，新回答上却自己冒出一个 👎。
# 用户会以为"这系统怎么自己给我点了个踩"。
#
# 这类问题的可怕之处在于它**完全不报错**：
# 接口返回 200，数据写得进去，只是悄悄写错了地方。
#
# 【为什么用事件钩子而不是在代码里手动删】
# 手动删要记得住"删消息时要删评价、删会话时要删消息、删用户时要删会话……"，
# 漏一处就又是孤儿。挂在连接的建立事件上，是**一次性把所有表的关系都管住**，
# 而且以后新增表也自动受保护，不用再记一遍。
@event.listens_for(engine.sync_engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    """每建立一条数据库连接，就打开外键约束"""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


# ------------------------------------------------------------------
# 会话工厂：每次需要操作数据库时，用它造一个「会话」
#   expire_on_commit=False 很关键：提交后对象不会过期，
#   否则异步场景下读取提交后的对象属性会触发额外查询甚至报错。
# ------------------------------------------------------------------
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """
    所有数据表模型的基类。
    在 app/models/ 里定义的每一张表，都要继承它，SQLAlchemy 才知道
    "这是一张表"，才能自动建表。
    """


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI 依赖注入用的「取数据库会话」函数。

    用法（在接口函数参数里写 db: AsyncSession = Depends(get_db)）：
        请求进来 → 造一个会话给你用 → 请求结束 → 自动关闭

    这种写法叫「依赖注入」，好处是你不用手动管开关连接，
    就算接口中途抛异常，Session 也会被正确关闭，不会泄漏。
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()  # 出错就回滚，保证数据一致
            raise


async def init_db() -> None:
    """
    建表函数：根据 app/models/ 里的定义，在数据库中创建所有还不存在的表。
    程序启动时调用一次。已存在的表不会被改动。
    """
    # 这一行必须写：导入 models 才能让 SQLAlchemy 感知到所有表定义
    from app import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """关闭数据库连接池。程序退出时调用，释放资源。"""
    await engine.dispose()
