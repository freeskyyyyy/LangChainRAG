"""
会话服务层 —— 会话的增删改查

================================================================
【本文件最重要的东西：数据隔离】
================================================================
需求 3 要求"每个用户有每个用户的独立会话"。这句话的技术含义是：

    用户 A 绝对不能读到、改到、删掉 用户 B 的任何会话。

怎么保证？看下面 get_owned_session() 这个函数——**每一个**涉及
具体会话 ID 的操作，都必须先经过它。它做一件事：

    同时用 [会话ID] 和 [当前用户ID] 两个条件去查。
    查不到 → 说明这个会话要么不存在，要么不属于你 → 一律当作"不存在"处理。

【为什么用"两个条件一起查"，而不是"先查出来再比较 user_id"】
两种写法都行，但一起查更不容易出错：
  · 分开写时，容易漏掉那个 if 判断（尤其是后来加新接口时）
  · 一起写时，忘了校验的情况在结构上就不可能出现——你不调用它就没数据

【为什么"不存在"和"不是你的"要返回同样的错误】
如果"不是你的"返回 403、"不存在"返回 404，攻击者就能靠这个差异
**探测出系统里有哪些会话 ID**（先随便试，403 说明存在，404 说明不存在）。
统一返回 404，就堵住了这条信息泄露路径。
"""

from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.core.logging import logger
from app.models import ChatSession, Message

# 会话标题长度上限（数据库字段是 255，这里留点余量）
MAX_TITLE_LENGTH = 100


class SessionService:
    """会话服务"""

    # ==============================================================
    # 核心安全函数 —— 所有涉及具体会话的操作都要先过这一关
    # ==============================================================
    @staticmethod
    async def get_owned_session(
        db: AsyncSession, session_id: int, user_id: int
    ) -> ChatSession:
        """
        获取"属于这个用户的"会话。不属于或不存在，一律抛 404。

        这是本文件的安全基石。新增任何会话相关接口时，
        **第一步都应该是调用它**。
        """
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,  # ← 关键：把归属条件写进查询里
            )
        )
        session = result.scalar_one_or_none()

        if session is None:
            logger.warning(
                f"会话访问被拒 | session_id={session_id} 请求者user_id={user_id} "
                f"（会话不存在，或不属于该用户）"
            )
            raise NotFoundError("会话不存在", code=40420)

        return session

    # ==============================================================
    # 列表
    # ==============================================================
    @staticmethod
    async def list_sessions(db: AsyncSession, user_id: int) -> list[dict]:
        """
        列出某个用户的全部会话。

        排序规则：置顶的在前，然后按最后更新时间倒序（最近聊过的排在前面）。
        这正是用户在聊天软件里习惯的排列方式。

        【为什么要"一次性"把消息条数也算出来】
        如果先查会话列表，再对每个会话单独查一次消息条数，
        10 个会话就是 11 次数据库查询（这叫 N+1 查询问题，是常见性能陷阱）。
        这里用一条 SQL 的 GROUP BY 一次性算完，只有 1 次查询。

        会话数量通常不会太多（几十个），所以这里不分页，一次全返回。
        """
        # 子查询：统计每个会话的消息条数
        count_subq = (
            select(
                Message.session_id,
                func.count(Message.id).label("message_count"),
            )
            .group_by(Message.session_id)
            .subquery()
        )

        stmt = (
            select(
                ChatSession,
                func.coalesce(count_subq.c.message_count, 0).label("message_count"),
            )
            .outerjoin(count_subq, ChatSession.id == count_subq.c.session_id)
            .where(ChatSession.user_id == user_id)
            .order_by(
                ChatSession.is_pinned.desc(),   # 置顶的排前面
                ChatSession.updated_at.desc(),  # 最近更新的排前面
            )
        )

        result = await db.execute(stmt)

        sessions: list[dict] = []
        for row in result.all():
            session_obj, message_count = row[0], row[1]
            sessions.append(
                {
                    "id": session_obj.id,
                    "title": session_obj.title,
                    "is_pinned": session_obj.is_pinned,
                    "created_at": session_obj.created_at,
                    "updated_at": session_obj.updated_at,
                    "message_count": message_count,
                }
            )

        return sessions

    # ==============================================================
    # 新建
    # ==============================================================
    @staticmethod
    async def create_session(
        db: AsyncSession, user_id: int, title: Optional[str] = None
    ) -> ChatSession:
        """新建一个会话"""
        session = ChatSession(
            user_id=user_id,
            title=(title or "新对话")[:MAX_TITLE_LENGTH],
            is_pinned=False,
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)

        logger.info(f"新建会话 | id={session.id} user_id={user_id}")
        return session

    # ==============================================================
    # 重命名 / 置顶
    # ==============================================================
    @staticmethod
    async def update_session(
        db: AsyncSession,
        session_id: int,
        user_id: int,
        title: Optional[str] = None,
        is_pinned: Optional[bool] = None,
    ) -> ChatSession:
        """修改会话标题或置顶状态"""
        session = await SessionService.get_owned_session(db, session_id, user_id)

        if title is not None:
            session.title = title.strip()[:MAX_TITLE_LENGTH] or "新对话"
        if is_pinned is not None:
            session.is_pinned = is_pinned

        await db.commit()
        await db.refresh(session)
        return session

    # ==============================================================
    # 删除
    # ==============================================================
    @staticmethod
    async def delete_session(db: AsyncSession, session_id: int, user_id: int) -> None:
        """
        删除会话。

        会话下面的消息会因为外键的 ondelete="CASCADE" 自动一起删掉，
        不需要我们手动处理（这是数据库层面的保证，比在代码里删更可靠）。
        """
        session = await SessionService.get_owned_session(db, session_id, user_id)

        await db.delete(session)
        await db.commit()

        logger.info(f"删除会话 | id={session_id} user_id={user_id}")

    # ==============================================================
    # 历史消息
    # ==============================================================
    @staticmethod
    async def get_messages(
        db: AsyncSession, session_id: int, user_id: int, limit: int = 200
    ) -> list[Message]:
        """
        获取某个会话的历史消息。

        【为什么要 limit】
        如果一个会话聊了几千轮，一次性全查出来会占用大量内存，
        前端渲染也会卡。限制最多返回最近 200 条，这是个务实的折中。

        排序用 id 升序（= 时间升序），这样对话顺序是从上到下、
        最新的在最下面，符合聊天软件的习惯。
        """
        await SessionService.get_owned_session(db, session_id, user_id)  # 先做归属校验

        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.id.asc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    # ==============================================================
    # 会话标题自动生成
    # ==============================================================
    @staticmethod
    def make_title_from_question(question: str, max_len: int = 20) -> str:
        """
        根据用户的第一句话生成会话标题。

        【为什么要自动生成】
        用户新建会话时懒得想标题，全都是"新对话"，
        过几天回来根本分不清哪个是哪个。自动取第一句话的前 20 个字，
        一眼就能认出来。

        这个小细节对体验的提升很明显，实现成本却几乎为零。
        """
        title = question.strip().replace("\n", " ").replace("\r", " ")

        # 把连续空格压成一个，避免标题里出现一排空格
        while "  " in title:
            title = title.replace("  ", " ")

        if len(title) <= max_len:
            return title or "新对话"

        return title[:max_len] + "…"

    # ==============================================================
    # 统计
    # ==============================================================
    @staticmethod
    async def count_sessions(db: AsyncSession, user_id: Optional[int] = None) -> int:
        """统计会话总数（不传 user_id 就是全站统计，管理端用）"""
        stmt = select(func.count()).select_from(ChatSession)
        if user_id is not None:
            stmt = stmt.where(ChatSession.user_id == user_id)
        return await db.scalar(stmt) or 0
