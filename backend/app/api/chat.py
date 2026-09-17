"""
问答接口 —— 流式问答 + 反馈

================================================================
【为什么这个接口的写法和其他接口完全不同】
================================================================
其他接口都是"收到请求 → 处理 → 返回一个 JSON"。
但这个接口返回的是 **SSE 流**（Server-Sent Events）：

    HTTP/1.1 200 OK
    Content-Type: text/event-stream

    event: status
    data: {"stage":"rewriting"}

    event: token
    data: {"text":"根据"}

    event: token
    data: {"text":"资料"}

    ...

也就是说，**一个 HTTP 请求，服务器可以持续不断往回推数据**，
直到全部说完为止。这正是打字机效果的实现方式。

【为什么要这么设计】
大模型生成一段 300 字的回答要 3-5 秒。
如果不流式，用户就得盯着白屏等 5 秒；
流式的话 0.3 秒后就开始有字出来了，用户感觉"系统很跟手"。
总耗时其实一样，但**等待焦虑完全不同**。

【流式带来的麻烦：什么时候存数据库】
非流式很简单：生成完 → 存库 → 返回。
流式就不行：数据是一点点推出去的，得等全部推完才知道完整答案是什么。

这里的做法是：在生成器内部一边推送一边收集，等流结束后
再把完整的回答和引用一起写入数据库。
"""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal, get_db
from app.core.deps import CurrentUser
from app.core.exceptions import BadRequestError, NotFoundError
from app.core.logging import logger
from app.core.rate_limit import check_rate_limit
from app.models import ChatSession, Feedback, Message, MessageRole, QALog
from app.schemas.chat import ChatRequest, FeedbackRequest
from app.schemas.common import SuccessResponse
from app.services.session_service import SessionService

router = APIRouter(prefix="/api/chat", tags=["问答"])


# ==================================================================
# 流式问答
# ==================================================================
@router.post(
    "/stream",
    summary="流式问答（SSE）",
    description=(
        "以 SSE 流的形式返回回答，实现打字机效果。\n\n"
        "事件类型依次为：\n"
        "- `status` 当前进行到哪一步\n"
        "- `rewrite` 问题改写详情（多轮对话时）\n"
        "- `retrieval` 检索结果概况\n"
        "- `token` 回答的一个片段\n"
        "- `citations` 引用的知识片段列表\n"
        "- `done` 结束\n"
        "- `error` 出错\n\n"
        "**注意**：前端需要用 `fetch` + `ReadableStream` 或 EventSource 来接收，"
        "普通的 axios 请求拿不到流式数据。"
    ),
    response_class=StreamingResponse,
)
async def chat_stream(
    data: ChatRequest,
    user: CurrentUser,
):
    """
    流式问答。

    ⚠️ 注意这个函数签名里**没有** `db: AsyncSession = Depends(get_db)`。

    为什么：FastAPI 的依赖注入会在**请求结束时**自动关闭数据库会话。
    而我们要在流式生成器里持续用数据库（一边生成一边存消息），
    那时请求的依赖生命周期可能已经结束了，会话被关掉就会报错。

    所以这里在生成器**内部**自己开一个独立的数据库会话，
    用完自己关，生命周期完全由自己掌控。
    """
    # 限流：问答接口是最贵的操作（要调大模型），必须限流
    check_rate_limit(f"chat:{user.id}")

    # 校验会话归属（如果指定了会话）
    if data.session_id is not None:
        async with AsyncSessionLocal() as db:
            await SessionService.get_owned_session(db, data.session_id, user.id)

    return StreamingResponse(
        _generate_stream(data, user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",      # 告诉浏览器不要缓存这个流
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",        # 如果将来用 Nginx 反代，这一行禁止它缓冲
        },
    )


async def _generate_stream(data: ChatRequest, user):
    """
    流式生成器 —— 真正的核心逻辑。

    【生成器是什么】
    一个有 yield 的函数。每 yield 一次，就往外"吐"一块数据，
    然后暂停在那里等下次被调用。StreamingResponse 会不断地
    从这个生成器取数据、通过网络发给前端。

    这样就能实现"边算边发"，而不是"全算完再发"。
    """
    import json

    session_id = data.session_id
    user_message_id: Optional[int] = None
    assistant_content = ""
    final_citations: list = []
    final_stats: dict = {}

    # 每个 SSE 事件的格式：event: xxx\ndata: {...}\n\n
    def sse(event: str, payload) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    try:
        # ---------------- 准备阶段 ----------------
        async with AsyncSessionLocal() as db:
            # 没有指定会话就新建一个（用户直接进来就问的场景）
            if session_id is None:
                session = await SessionService.create_session(
                    db, user.id, title=SessionService.make_title_from_question(data.question)
                )
                session_id = session.id
                # 新会话的 ID 要第一时间告诉前端，否则它不知道往哪个会话里塞消息
                yield sse("session", {"session_id": session_id, "title": session.title})
            else:
                # 如果会话还是"新对话"，用第一个问题给它命名
                result = await db.execute(
                    select(ChatSession).where(ChatSession.id == session_id)
                )
                session = result.scalar_one_or_none()
                if session and session.title == "新对话":
                    session.title = SessionService.make_title_from_question(data.question)
                    yield sse("session", {"session_id": session_id, "title": session.title})
                    await db.commit()

            # ---------------- 重新生成：先把上一次的回答删掉 ----------------
            # 【为什么是"删掉"而不是"再追加一条"】
            # 如果只是再答一遍，历史里就会变成：
            #     问 → 答① → 问 → 答② → 问 → 答③
            # 用户点两次「重新生成」，就多出两对一模一样的问答，
            # 会话越用越长越乱。而用户的真实意图是"这次不算，重来"，
            # 所以要**替换**掉上一条回答（ChatGPT、通义千问都是这个行为）。
            if data.regenerate:
                last_ai = await db.execute(
                    select(Message)
                    .where(
                        Message.session_id == session_id,
                        Message.role == MessageRole.ASSISTANT,
                    )
                    .order_by(Message.id.desc())
                    .limit(1)
                )
                doomed = last_ai.scalar_one_or_none()
                # 找不到说明上一次压根没答出来（比如中途报错），
                # 那就没什么可删的，直接往下走重新答一遍即可
                if doomed is not None:
                    await db.delete(doomed)
                    await db.commit()

            # 取出历史消息（多轮对话需要）
            history_rows = await SessionService.get_messages(db, session_id, user.id, limit=10)
            history = [
                {"role": m.role, "content": m.content}
                for m in history_rows
                if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
            ]

            if data.regenerate:
                # 【这里很容易漏，漏了会让模型以为用户问了两遍】
                # 要重答的那条用户消息**本身就在 history 的末尾**
                #（因为我们上面只删了 AI 回答，没删用户消息），
                # 而 data.question 传的也是同一句话。
                # 两者一起喂进去，模型会看到：
                #     [..., "真丝连衣裙怎么洗", "真丝连衣裙怎么洗"]
                # 于是它可能理解成"用户在追问/强调"，答得反而更啰嗦。
                # 所以把末尾连续的用户消息摘掉，让 history 停在"上一轮 AI 回答"处，
                # 和正常提问时的历史结构保持一致。
                while history and history[-1]["role"] == MessageRole.USER:
                    history.pop()

            # 先把用户的问题存进数据库
            # （重新生成时**不存** —— 同一个问题没必要存两遍）
            if not data.regenerate:
                user_msg = Message(
                    session_id=session_id,
                    role=MessageRole.USER,
                    content=data.question,
                )
                db.add(user_msg)
                await db.commit()
                await db.refresh(user_msg)
                user_message_id = user_msg.id

        # ---------------- 调用 RAG 主链 ----------------
        from app.rag.chain import get_rag_chain

        chain = get_rag_chain()

        async for event in chain.astream(
            data.question,
            history=history,
            use_cache=data.use_cache,
            top_k=data.top_k,
        ):
            etype = event["event"]
            payload = event["data"]

            # 顺带收集数据，供流结束后写库
            if etype == "token":
                assistant_content += payload.get("text", "")
            elif etype == "citations":
                final_citations = payload
            elif etype == "done":
                final_stats = payload
            elif etype == "error":
                # 出错也要存用户消息（已经存过了），然后结束
                yield sse("error", payload)
                break

            yield sse(etype, payload)

        # ---------------- 流结束，写入数据库 ----------------
        async with AsyncSessionLocal() as db:
            if assistant_content:
                assistant_msg = Message(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=assistant_content,
                    citations=final_citations,
                    rewritten_query=final_stats.get("search_query"),
                    latency_ms=final_stats.get("latency_ms"),
                    cache_hit=bool(final_stats.get("cache_hit")),
                    retrieval_count=len(final_citations),
                    top_score=final_stats.get("top_score"),
                )
                db.add(assistant_msg)

            # 更新会话的"最后活动时间"，让它在列表里排到前面
            result = await db.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            session = result.scalar_one_or_none()
            if session:
                from datetime import datetime, timezone

                session.updated_at = datetime.now(timezone.utc)

            # 记录问答日志（统计面板的数据来源）
            db.add(
                QALog(
                    user_id=user.id,
                    session_id=session_id,
                    question=data.question,
                    rewritten_question=final_stats.get("search_query"),
                    retrieval_count=len(final_citations),
                    top_score=final_stats.get("top_score"),
                    cache_hit=bool(final_stats.get("cache_hit")),
                    latency_ms=final_stats.get("latency_ms"),
                    no_answer=bool(final_stats.get("no_answer")),
                )
            )

            await db.commit()

            # 把消息 ID 补发给前端
            if assistant_content:
                await db.refresh(assistant_msg)
                yield sse(
                    "saved",
                    {
                        "session_id": session_id,
                        "user_message_id": user_message_id,
                        "assistant_message_id": assistant_msg.id,
                    },
                )

    except Exception as e:
        logger.exception(f"流式问答异常 | user={user.username} | {data.question[:50]!r}")
        yield sse(
            "error",
            {
                "message": "生成回答时出错了，请稍后重试",
                "detail": f"{type(e).__name__}: {str(e)[:200]}",
            },
        )


# ==================================================================
# 问答反馈
# ==================================================================
@router.post(
    "/feedback",
    response_model=SuccessResponse,
    summary="给回答点赞或点踩",
    description="同一条消息同一个用户只能评价一次，重复评价会覆盖之前的。",
)
async def submit_feedback(
    data: FeedbackRequest,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SuccessResponse:
    """
    提交问答反馈。

    【为什么需要这个功能】
    这是**唯一来自真实用户的质量评价**。自动指标（比如检索分数）
    只能说明"检索到了相关片段"，但回答得好不好、有没有帮到人，
    只有用户知道。

    积累起来就能看出："哪类问题系统答得不好"，进而针对性优化。

    【为什么要校验消息归属】
    必须确认这条消息属于该用户的某个会话，否则
    用户可以给别人的回答点赞（数据污染），
    甚至能通过"消息ID存不存在"来探测系统里有多少条消息。
    """
    # 校验：这条消息必须属于当前用户的某个会话
    result = await db.execute(
        select(Message, ChatSession)
        .join(ChatSession, Message.session_id == ChatSession.id)
        .where(Message.id == data.message_id, ChatSession.user_id == user.id)
    )
    row = result.first()

    if row is None:
        raise NotFoundError("消息不存在", code=40430)

    message, _session = row

    if message.role != MessageRole.ASSISTANT:
        raise BadRequestError("只能对 AI 的回答进行评价", code=40030)

    # 同一条消息同一用户只能评价一次 → 有则更新，无则新建
    existing_result = await db.execute(
        select(Feedback).where(
            Feedback.message_id == data.message_id,
            Feedback.user_id == user.id,
        )
    )
    existing = existing_result.scalar_one_or_none()

    if existing:
        existing.rating = data.rating
        existing.comment = data.comment
        action = "更新"
    else:
        db.add(
            Feedback(
                message_id=data.message_id,
                user_id=user.id,
                rating=data.rating,
                comment=data.comment,
            )
        )
        action = "记录"

    await db.commit()

    kind = "点赞" if data.rating > 0 else "点踩"
    logger.info(f"反馈{action} | user={user.username} | message={data.message_id} | {kind}")

    return SuccessResponse(message=f"{kind}成功，感谢反馈")
