"""
数据表公共基类与混入（Mixin）

【什么是 Mixin】
一个"零件"，可以被多个表"装上"，装上后就自动获得里面的字段。

比如每个表都需要「创建时间」和「更新时间」这两个字段。
如果每张表都手写一遍，既啰嗦又容易写得不一致。
把它做成 TimestampMixin，哪张表需要就继承一下，字段自动就有了。

【SQLAlchemy 2.0 的新写法说明】
    Mapped[int]         ← 声明这个字段的 Python 类型是整数
    mapped_column(...)  ← 声明数据库层面的细节（类型、约束、默认值等）

这种写法比老版本（Column(Integer)）的好处是：类型检查工具能真正
理解字段类型，写错代码时编辑器会提前警告。
"""

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column


class TimestampMixin:
    """给表加上「创建时间」和「更新时间」两个字段"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),  # 数据库层面自动填当前时间，不依赖程序
        nullable=False,
        comment="创建时间",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),  # 每次更新记录时自动刷新
        nullable=False,
        comment="最后更新时间",
    )
