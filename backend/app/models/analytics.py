"""
问答反馈表 与 问答日志表

【这两张表是干什么的】
它们不参与核心问答流程，而是为"持续优化"提供数据。

  Feedback（反馈表）—— 用户对某条回答点赞/点踩。这是**唯一来自人类的
                        质量评价**，比任何自动指标都真实。积累起来就能看出
                        "哪类问题系统答得不好"，进而针对性改进。

  QALog（问答日志表）—— 每次问答的过程数据（检索到几条、最高分多少、
                        有没有命中缓存、耗时多久）。用来做统计分析：
                        "缓存在一周内省下了多少次 API 调用"、
                        "平均响应时间是多少"，这些都是优化的量化依据。
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class FeedbackRating:
    """反馈类型常量"""

    LIKE = 1     # 点赞
    DISLIKE = -1  # 点踩


class Feedback(Base):
    """问答反馈表 —— 用户对某条 AI 回答的评价"""

    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="被评价的消息",
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="评价人",
    )

    rating: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="1 = 点赞，-1 = 点踩",
    )

    comment: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="可选评论（例如：答非所问）",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # 同一个用户对同一条消息只能评价一次（防止刷赞）。
        # 这用「唯一索引」来保证——数据库层面强制约束，比在代码里判断更可靠，
        # 因为并发情况下代码判断可能来不及（两个请求同时通过检查再同时插入）。
        Index("ix_feedback_msg_user_unique", "message_id", "user_id", unique=True),
    )

    def __repr__(self) -> str:
        kind = "赞" if self.rating > 0 else "踩"
        return f"<Feedback id={self.id} message={self.message_id} {kind}>"


class QALog(Base):
    """
    问答日志表 —— 每次问答的过程数据

    这张表会随时间增长（每问一次加一条），所以要注意：
      · 建好索引，保证统计查询够快
      · 将来数据量大了可以做定期归档（比如只保留最近 90 天）
    """

    __tablename__ = "qa_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="提问用户",
    )

    session_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
        comment="所属会话",
    )

    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="用户原始问题",
    )

    rewritten_question: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="多轮改写后的问题（用于评估改写效果）",
    )

    # ---------------- 检索质量指标 ----------------
    retrieval_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="检索命中的片段数",
    )

    top_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="最高相似度分（低于阈值会触发『知识库未找到』）",
    )

    # ---------------- 性能指标 ----------------
    cache_hit: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否命中语义缓存（命中则完全不调用 API）",
    )

    latency_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="总耗时（毫秒）",
    )

    retrieval_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="检索阶段耗时（毫秒），用于定位性能瓶颈在哪一步",
    )

    generation_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="生成阶段耗时（毫秒）",
    )

    # ---------------- 内容安全 ----------------
    # 检索最高分低于阈值时置 True。用来统计"有多少问题是知识库答不了的"，
    # 这个比例是判断知识库覆盖度的重要指标——如果太高，说明该补资料了。
    no_answer: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否属于『知识库中未找到』的情况",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        # 统计面板最常见的查询："最近 N 天有多少次问答"、"今天问了多少次"
        Index("ix_qalog_created_user", "created_at", "user_id"),
    )

    def __repr__(self) -> str:
        q = (self.question[:20] + "...") if len(self.question) > 20 else self.question
        return f"<QALog id={self.id} {q!r} {self.latency_ms}ms>"
