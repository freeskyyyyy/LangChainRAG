"""
用户表

【存什么】
账号、密码哈希、角色（管理员/普通用户）、启用状态。

【安全要点】
password_hash 字段存的是 bcrypt 哈希后的乱码，不是明文密码。
即使数据库文件被人拿走，也无法反推出原始密码。
"""

from datetime import datetime
from typing import TYPE_CHECKING, List

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin

if TYPE_CHECKING:
    # 只在做类型检查时导入，避免运行时的循环导入问题
    from app.models.chat import ChatSession


class UserRole:
    """
    角色常量。

    为什么用类常量而不是直接写字符串 "admin"？
    因为字符串写错了不会被发现（"admn" 也能跑，只是逻辑失效）。
    用常量的话，写错名字编辑器会直接报错，这就是"防手滑"。
    """

    ADMIN = "admin"       # 管理员：能管理知识库、看统计
    USER = "user"         # 普通用户：只能问答


class User(Base, TimestampMixin):
    """用户表"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,        # 用户名不能重复
        index=True,         # 加索引，登录时按用户名查会快很多
        nullable=False,
        comment="用户名（登录用）",
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="bcrypt 哈希后的密码（绝无明文）",
    )

    nickname: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="昵称（展示用，可以随意改）",
    )

    role: Mapped[str] = mapped_column(
        String(20),
        default=UserRole.USER,
        nullable=False,
        index=True,
        comment="角色：admin 管理员 / user 普通用户",
    )

    avatar: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="头像地址",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="是否启用（管理员可禁用某个账号）",
    )

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="最后登录时间",
    )

    # ---------------- 关联关系 ----------------
    # 一个用户拥有多个会话。
    # cascade="all, delete-orphan" 的意思是：删掉用户时，他的会话也一并删掉，
    # 不会留下一堆找不到主人的"孤儿数据"。
    sessions: Mapped[List["ChatSession"]] = relationship(
        "ChatSession",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # ---------------- 便捷属性 ----------------
    @property
    def is_admin(self) -> bool:
        """判断是不是管理员。代码里用它比每次都写 == 'admin' 更安全。"""
        return self.role == UserRole.ADMIN

    def __repr__(self) -> str:
        """调试时打印对象用，不会泄露密码哈希"""
        return f"<User id={self.id} username={self.username!r} role={self.role}>"
