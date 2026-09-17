"""
通用数据结构 —— 分页、统一响应包装等各处复用的结构

【Pydantic 模型是干什么的】
它是"数据守门员"。前端发来的数据必须先通过它的检查，才能进入业务逻辑。

举个例子，如果定义了：
    class UserCreate(BaseModel):
        username: str = Field(min_length=3, max_length=50)

那么前端传来 username="a"（太短）或 username=123（不是文字）时，
Pydantic 会立刻拦下来并给出清晰的错误，**你的业务代码根本不会被执行**。

【这样设计的好处】
  · 脏数据进不来，业务逻辑不用到处写 if 判断
  · 错误提示自动生成，不用手写校验代码
  · 类型明确，编辑器能提前发现写错的字段名
  · 配合 FastAPI 还能自动生成 API 文档
"""

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMBase(BaseModel):
    """
    所有"从数据库对象转换而来"的响应模型的基类。

    【from_attributes = True 是什么意思】
    我们的用户信息存在 SQLAlchemy 的 User 对象里，
    但接口要返回的是 JSON，需要把对象转成 Pydantic 模型。

    设了这个选项后，就能直接写：
        UserOut.model_validate(user_obj)
    它会自动按字段名从对象属性里取值，不用手写一堆赋值语句。

    如果不开这个选项，就得写：
        UserOut(id=user.id, username=user.username, ...)   # 又长又容易漏
    """

    model_config = ConfigDict(from_attributes=True)


class PageParams(BaseModel):
    """分页查询参数（用在 GET 请求的查询字符串里）"""

    page: int = Field(default=1, ge=1, description="第几页，从 1 开始")
    page_size: int = Field(default=20, ge=1, le=100, description="每页多少条，最大 100")

    @property
    def offset(self) -> int:
        """
        计算 SQL 的 OFFSET（跳过多少条）。

        为什么最大只允许 100 条？
        防止有人传 page_size=999999 把整个数据库拉出来，
        那会占用大量内存和带宽，甚至拖垮服务。这叫"资源耗尽攻击"。
        """
        return (self.page - 1) * self.page_size


class PageResult(BaseModel, Generic[T]):
    """
    分页结果的标准结构。

    前端拿到后就知道：总共多少条、总共有几页、当前这页的数据是什么。
    """

    items: list[T] = Field(default_factory=list, description="当前页的数据")
    total: int = Field(default=0, description="总条数")
    page: int = Field(default=1, description="当前页码")
    page_size: int = Field(default=20, description="每页条数")

    @property
    def total_pages(self) -> int:
        """总页数（向上取整，所以 + page_size - 1）"""
        if self.page_size <= 0:
            return 0
        return (self.total + self.page_size - 1) // self.page_size


class SuccessResponse(BaseModel):
    """
    简单的成功响应。

    有些操作（比如"删除会话"）不需要返回数据，只需要告诉前端"成功了"。
    这时候用它比返回一个空字典清晰。
    """

    success: bool = True
    message: str = "操作成功"
    data: Optional[Any] = None


class StatsResponse(BaseModel):
    """系统统计信息（健康检查接口用）"""

    status: str = "ok"
    app_name: str = ""
    version: str = "1.0.0"
    database: str = ""
    cache: dict[str, Any] = Field(default_factory=dict)
    rate_limit: dict[str, Any] = Field(default_factory=dict)
    llm_model: str = ""
    embedding_model: str = ""
    rerank_model: str = ""
