"""
会话与问答相关的数据结构

【这个文件和前端的关系最密切】
下面这些结构决定了"前端能拿到什么、能发什么"。
前端开发者（也就是我写 Vue 那部分时）就照着这些字段来写界面。
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.common import ORMBase


# ==================================================================
# 引用片段 —— 整个系统最有价值的数据结构之一
# ==================================================================
class Citation(BaseModel):
    """
    一条知识库引用。

    【它就是"可溯源"的具体体现】
    模型回答完之后，我们会告诉前端："这次回答参考了这 3 个片段"，
    每个片段带上文件名、页码、相似度、原文。

    前端拿到后在回答下面渲染成可点击的卡片，
    用户点开就能看到"这句话是从哪份资料第几页来的"。

    【为什么要有 index 字段】
    因为回答正文里会有 [1] [2] 这样的标注，
    index 就是用来跟正文对应上的：
        "该手机电池容量为 3561mAh [1]"
        用户点 [1] → 高亮下面第 1 张引用卡片
    """

    index: int = Field(..., description="引用编号，对应正文里的 [1] [2]")
    file_name: str = Field(..., description="来源文件名")
    page: Optional[int] = Field(default=None, description="所在页码（PDF/Word 有，其他格式可能为 null）")
    sheet_name: Optional[str] = Field(default=None, description="所在工作表名（Excel 有）")
    score: float = Field(..., description="相似度分数 0~1，越高越相关")
    content: str = Field(..., description="片段原文，供用户点开查看")
    document_id: Optional[int] = Field(default=None, description="文档 ID，便于前端跳转")


# ==================================================================
# 会话
# ==================================================================
class SessionCreate(BaseModel):
    """新建会话请求"""

    title: Optional[str] = Field(default=None, max_length=255, description="会话标题，不填则用默认名")


class SessionUpdate(BaseModel):
    """修改会话请求（重命名 / 置顶）"""

    title: Optional[str] = Field(default=None, max_length=255, description="新标题")
    is_pinned: Optional[bool] = Field(default=None, description="是否置顶")


class SessionOut(ORMBase):
    """返回给前端的会话信息"""

    id: int
    title: str
    is_pinned: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    message_count: int = Field(default=0, description="该会话的消息条数（列表页展示用）")


# ==================================================================
# 消息
# ==================================================================
class MessageOut(ORMBase):
    """
    返回给前端的消息。

    【注意这里也没有 session_id】
    前端拿到消息时已经知道是哪个会话了（因为是通过某个会话的接口取的），
    再返回一遍是冗余。少一个字段就少一分信息泄露的风险。
    """

    id: int
    role: Literal["user", "assistant"] = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容（Markdown）")
    citations: Optional[list[dict[str, Any]]] = Field(
        default=None, description="AI 回答引用的知识库片段"
    )
    created_at: Optional[datetime] = None
    latency_ms: Optional[int] = Field(default=None, description="本次回答耗时（毫秒）")
    cache_hit: bool = Field(default=False, description="是否命中了缓存")
    knowledge_source: Optional[Literal["knowledge_base", "general", "system"]] = Field(
        default=None, description="回答来源；旧消息或未完成生成时可能未知"
    )
    feedback: Optional[int] = Field(
        default=None, description="当前用户对这条消息的评价：1 赞 / -1 踩 / null 未评价"
    )


# ==================================================================
# 提问
# ==================================================================
class ChatRequest(BaseModel):
    """
    提问请求

    【注意这个接口不返回普通 JSON，而是返回 SSE 流】
    因为回答是"一个字一个字蹦出来"的，不是一次性返回。
    详见 app/api/chat.py。
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户问题",
    )

    session_id: Optional[int] = Field(
        default=None,
        description="会话 ID。不传则自动新建一个会话（适合『进来就问』的场景）",
    )

    use_cache: bool = Field(
        default=True,
        description="是否允许使用缓存。调试时设为 false 可以强制走完整流程",
    )

    regenerate: bool = Field(
        default=False,
        description=(
            "是否为『重新生成』。为 true 时：不新建用户消息，"
            "而是删掉该会话最后一条 AI 回答，用同一个问题重答一次。"
            "需要同时传 session_id"
        ),
    )

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="本次检索召回多少条。不传用系统默认值",
    )

    @field_validator("question")
    @classmethod
    def _clean_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("问题不能为空")
        return v

    @model_validator(mode="after")
    def _check_regenerate(self):
        """
        「重新生成」必须指定会话。

        【为什么这条校验值得写】
        不写的话也不会崩：session_id 为空时，下面的流程会
        老老实实**新建一个空会话**，然后"重新生成"出一条全新的回答。
        接口返回 200，前端显示正常，看起来一切正常 ——

        但用户点的是"重新生成"，预期是"把刚才那条重答一遍"，
        结果却在侧栏凭空多出一个新会话。这种"不报错的错"最难查。

        宁可在门口就拦住，给一句说得清的话。
        """
        if self.regenerate and self.session_id is None:
            raise ValueError("重新生成时必须指定 session_id（没有会话就无从重新生成）")
        return self


class FeedbackRequest(BaseModel):
    """问答反馈请求（点赞/点踩）"""

    message_id: int = Field(..., description="要评价的消息 ID")
    rating: int = Field(..., description="1 = 点赞，-1 = 点踩")
    comment: Optional[str] = Field(default=None, max_length=500, description="可选评论")

    @field_validator("rating")
    @classmethod
    def _check_rating(cls, v: int) -> int:
        if v not in (1, -1):
            raise ValueError("rating 只能是 1（点赞）或 -1（点踩）")
        return v


# ==================================================================
# SSE 流式事件（仅作文档说明用，实际发送的是 dict）
# ==================================================================
class StreamEvent:
    """
    SSE 事件类型说明。

    这不是 Pydantic 模型，只是一份"协议文档"放在代码里，方便查阅。
    实际的流式响应是逐个发送下面这些事件：

        event: rewrite     → 问题被改写后的样子（多轮对话时）
        event: retrieval   → 检索完成，带命中数量和最高分
        event: token       → 一个字/词，前端追加到气泡里
        event: citations   → 引用片段列表
        event: done        → 回答结束，带消息 ID 和耗时
        event: error       → 出错了，带错误信息
    """

    REWRITE = "rewrite"
    RETRIEVAL = "retrieval"
    TOKEN = "token"
    CITATIONS = "citations"
    DONE = "done"
    ERROR = "error"


__all__ = [
    "Citation",
    "SessionCreate",
    "SessionUpdate",
    "SessionOut",
    "MessageOut",
    "ChatRequest",
    "FeedbackRequest",
    "StreamEvent",
]
