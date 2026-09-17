"""
数据模型统一出口

【这个文件为什么必须存在，而且必须把所有模型都导入进来】

SQLAlchemy 有个特点：它只有在"见过"某个模型类之后，才知道有这张表存在。
建表时它能建出的表 = 它见过的那些模型。

所以如果某个模型文件没被 import 过，那张表就**不会被创建**，
而且不会报任何错——你会一脸懵地发现"表怎么没了"。

`app/core/database.py` 里的 `init_db()` 会执行 `from app import models`，
那一步就会触发本文件，进而把下面所有模型都加载进来。
这样就能保证"只要这里列了的表，都会被创建"。

【加这张清单的好处】
把"系统一共有哪些表"这件事集中在一处展示，一目了然。
"""

from app.models.analytics import Feedback, FeedbackRating, QALog
from app.models.base import TimestampMixin
from app.models.chat import ChatSession, Message, MessageRole
from app.models.knowledge import Chunk, Document, DocumentStatus
from app.models.user import User, UserRole

# 明确声明本模块对外导出哪些名字。
# 好处：① 编辑器能给出准确提示 ② 防止 `from app.models import *` 时
# 把不该导出的东西也带出去
__all__ = [
    # ---- 用户 ----
    "User",
    "UserRole",
    # ---- 会话与消息 ----
    "ChatSession",
    "Message",
    "MessageRole",
    # ---- 知识库 ----
    "Document",
    "DocumentStatus",
    "Chunk",
    # ---- 统计与反馈 ----
    "Feedback",
    "FeedbackRating",
    "QALog",
    # ---- 基类 ----
    "TimestampMixin",
]
