"""
会话表 与 消息表

【这两张表的关系】
一个用户有多个会话（比如"手机咨询""售后问题""配件咨询"）
一个会话有多条消息（用户问的 + 助手答的，交替出现）

    User ──< ChatSession ──< Message

【关键设计：数据隔离】
ChatSession 上的 user_id 字段是"谁能看到这个会话"的唯一依据。
所有涉及会话的接口，都必须先校验 session.user_id == 当前登录用户 id，
否则用户 A 只要猜个数字就能看到用户 B 的聊天记录。
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    func,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User


class MessageRole:
    """消息角色常量"""

    USER = "user"            # 用户发的
    ASSISTANT = "assistant"  # AI 回答的


class ChatSession(Base, TimestampMixin):
    """
    会话表 —— 一次"主题对话"的容器

    比如用户点"新建对话"问手机的事，就产生一个会话；
    再点"新建对话"问售后的事，就是另一个会话。
    这样不同主题的对话不会混在一起，历史记录也清晰。
    """

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属用户（数据隔离的关键字段）",
    )

    title: Mapped[str] = mapped_column(
        String(255),
        default="新对话",
        nullable=False,
        comment="会话标题（可用首条问题自动生成）",
    )

    is_pinned: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否置顶（重要的会话钉在最上面）",
    )

    # ---------------- 关联关系 ----------------
    user: Mapped["User"] = relationship("User", back_populates="sessions")

    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.id",  # 按时间正序排列，聊天记录从上到下
        lazy="selectin",
    )

    # ---------------- 索引 ----------------
    # 按「用户 + 是否置顶 + 最后更新时间」建复合索引。
    # 因为这正好是前端拉会话列表时的查询条件：
    #   「查我的会话，置顶的在前，最近聊过的在前」
    # 建了索引，即使有上万个会话，查询也是毫秒级。
    __table_args__ = (
        Index("ix_sessions_user_pinned_updated", "user_id", "is_pinned", "updated_at"),
    )

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id} user_id={self.user_id} title={self.title!r}>"


class Message(Base):
    """
    消息表 —— 一条具体的消息（用户问的，或 AI 答的）

    注意这张表没有继承 TimestampMixin，因为我们只需要 created_at，
    不需要 updated_at（消息发出去就不会改了）。
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属会话",
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="角色：user 用户 / assistant AI回答",
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="消息内容（Markdown 格式）",
    )

    # ------------------------------------------------------------------
    # citations：这条回答引用了知识库里的哪些片段
    #
    # 存成 JSON 是因为它的结构是"一个列表，每项含文件名/页码/分数/原文"，
    # 用 JSON 直接塞进去最省事，不用为了它再建一张关联表。
    #
    # 结构示例：
    # [
    #   {"index": 1, "file_name": "iPhone16说明书.pdf", "page": 3,
    #    "score": 0.87, "content": "电池容量为 3561mAh..."},
    #   ...
    # ]
    # ------------------------------------------------------------------
    citations: Mapped[Optional[List[dict[str, Any]]]] = mapped_column(
        JSON,
        nullable=True,
        comment="引用的知识库片段（含文档名/页码/相似度/原文）",
    )

    # ------------------------------------------------------------------
    # 下面几个字段是"过程数据"，用来支撑性能优化和问题排查：
    #   rewritten_query —— 多轮改写后的完整问题（调试时能看出改写对不对）
    #   tokens          —— 消耗的令牌数（算成本用）
    #   latency_ms      —— 响应耗时（性能优化的量化证据）
    #   缓存命中情况     —— 判断缓存到底有没有起作用
    # ------------------------------------------------------------------
    rewritten_query: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="多轮改写后的查询（便于调试）",
    )

    prompt_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="提示词消耗的令牌数",
    )

    completion_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="生成内容消耗的令牌数",
    )

    latency_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="本次回答耗时（毫秒）",
    )

    cache_hit: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否命中了语义缓存",
    )

    retrieval_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="检索命中的知识片段数",
    )

    top_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="检索结果的最高相似度分",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),  # 必须给默认值，否则插入数据时会报错
        nullable=False,
        comment="发送时间",
    )

    # ---------------- 关联关系 ----------------
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

    __table_args__ = (
        # 查某个会话的历史消息时，按「会话ID + 发送时间」查最常用
        Index("ix_messages_session_created", "session_id", "created_at"),
    )

    def __repr__(self) -> str:
        preview = (self.content[:20] + "...") if len(self.content) > 20 else self.content
        return f"<Message id={self.id} role={self.role} content={preview!r}>"
