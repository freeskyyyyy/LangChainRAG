"""
会话接口 —— 会话的增删改查 + 历史消息

【安全提示】
本文件里**每一个**涉及具体会话 ID 的接口，第一行都会调用
SessionService.get_owned_session() 做归属校验。
这不是可选项——漏掉任何一个，就会产生"用户 A 能看用户 B 聊天记录"的严重漏洞。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import CurrentUser
from app.models import Feedback, Message
from app.schemas.chat import MessageOut, SessionCreate, SessionOut, SessionUpdate
from app.schemas.common import SuccessResponse
from app.services.session_service import SessionService

router = APIRouter(prefix="/api/sessions", tags=["会话"])


@router.get(
    "",
    response_model=list[SessionOut],
    summary="我的会话列表",
    description="返回当前登录用户的全部会话，置顶的在前，最近聊过的在前。",
)
async def list_sessions(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[SessionOut]:
    """
    获取会话列表。

    注意：这里只查 `user_id == 当前用户` 的会话，别人的会话根本不会被查出来。
    """
    rows = await SessionService.list_sessions(db, user.id)
    return [SessionOut(**row) for row in rows]


@router.post(
    "",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="新建会话",
)
async def create_session(
    data: SessionCreate,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SessionOut:
    """新建一个会话。标题不填的话默认叫"新对话"。"""
    session = await SessionService.create_session(db, user.id, data.title)
    return SessionOut(
        id=session.id,
        title=session.title,
        is_pinned=session.is_pinned,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=0,
    )


@router.patch(
    "/{session_id}",
    response_model=SessionOut,
    summary="重命名或置顶会话",
)
async def update_session(
    session_id: int,
    data: SessionUpdate,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SessionOut:
    """
    修改会话信息。

    可以做两件事：
      · 改标题（title）
      · 置顶/取消置顶（is_pinned）
    只传要改的那个字段就行。
    """
    session = await SessionService.update_session(
        db, session_id, user.id, title=data.title, is_pinned=data.is_pinned
    )
    return SessionOut(
        id=session.id,
        title=session.title,
        is_pinned=session.is_pinned,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.delete(
    "/{session_id}",
    response_model=SuccessResponse,
    summary="删除会话",
    description="会话下的所有消息会一并删除，不可恢复。",
)
async def delete_session(
    session_id: int,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SuccessResponse:
    """删除会话（连同它的全部消息）"""
    await SessionService.delete_session(db, session_id, user.id)
    return SuccessResponse(message="会话已删除")


@router.get(
    "/{session_id}/messages",
    response_model=list[MessageOut],
    summary="获取会话的历史消息",
    description="用于用户重新登录后恢复历史对话，或切换会话时加载内容。",
)
async def get_messages(
    session_id: int,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[MessageOut]:
    """
    获取历史消息。

    【顺带把"我的评价"一起查出来】
    每条 AI 回答下面有赞/踩按钮。用户重新打开历史对话时，
    如果之前点过赞，按钮应该是高亮的（而不是显示成没点过的样子）。

    我们把所有消息的评价一次查出来（一条 SQL），
    再用字典做映射，避免对每条消息都查一次数据库（N+1 问题）。
    """
    messages = await SessionService.get_messages(db, session_id, user.id)

    # 一次性查出当前用户对这些消息的所有评价
    from sqlalchemy import select

    if messages:
        message_ids = [m.id for m in messages]
        fb_result = await db.execute(
            select(Feedback).where(
                Feedback.user_id == user.id,
                Feedback.message_id.in_(message_ids),
            )
        )
        # 做成 {消息ID: 评价值} 的字典，方便 O(1) 查找
        my_feedback: dict[int, int] = {
            fb.message_id: fb.rating for fb in fb_result.scalars().all()
        }
    else:
        my_feedback = {}

    result: list[MessageOut] = []
    for msg in messages:
        item = MessageOut.model_validate(msg)
        item.feedback = my_feedback.get(msg.id)
        result.append(item)

    return result
