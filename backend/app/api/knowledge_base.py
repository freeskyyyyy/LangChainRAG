"""
知识库管理接口 —— 仅管理员可用

================================================================
【这个文件里最值得注意的一行代码】
================================================================
本文件所有接口的参数里都有这么一段：

    admin: AdminUser

`AdminUser` 是 app/core/deps.py 里定义的类型别名，等价于
`User = Depends(get_current_admin)`。它做两件事：
  1. 验证登录（没登录 → 401）
  2. 验证是管理员（不是 → 403）

【为什么强调这个】
需求要求"只有管理员能打开知识库管理页面"。
如果只在前端隐藏菜单，懂技术的人用 Postman 直接调接口就绕过去了——
前端隐藏**只是体验优化，不是安全措施**。

真正的拦截就在这里：只要接口参数里写了 AdminUser，
不管请求从哪来（浏览器、脚本、别的网站），都会被拦住。
而且这是**声明式**的——看一眼函数签名就知道这个接口要什么权限，
不会出现"某个接口忘了加校验"的疏漏。
"""

from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.deps import AdminUser
from app.core.exceptions import BadRequestError, NotFoundError
from app.core.logging import logger
from app.core.rate_limit import check_rate_limit
from app.models import Chunk, Document, DocumentStatus
from app.rag.loader import get_parser_info, is_supported
from app.schemas.common import PageResult, SuccessResponse

router = APIRouter(prefix="/api/kb", tags=["知识库管理"])


# ==================================================================
# 文档列表
# ==================================================================
@router.get(
    "/documents",
    summary="文档列表（仅管理员）",
    description="支持分页、按文件名搜索、按状态筛选。",
)
async def list_documents(
    admin: AdminUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1, description="第几页"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
    keyword: Optional[str] = Query(None, description="按文件名模糊搜索"),
    doc_status: Optional[str] = Query(
        None,
        alias="status",
        description="按状态筛选：pending / parsing / indexed / failed",
    ),
) -> dict:
    """获取知识库文档列表"""
    stmt = select(Document)
    count_stmt = select(func.count()).select_from(Document)

    if keyword:
        pattern = f"%{keyword}%"
        stmt = stmt.where(Document.file_name.ilike(pattern))
        count_stmt = count_stmt.where(Document.file_name.ilike(pattern))

    if doc_status:
        stmt = stmt.where(Document.status == doc_status)
        count_stmt = count_stmt.where(Document.status == doc_status)

    total = await db.scalar(count_stmt) or 0

    stmt = (
        stmt.order_by(Document.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    docs = result.scalars().all()

    items = [
        {
            "id": d.id,
            "file_name": d.file_name,
            "file_type": d.file_type,
            "file_size": d.file_size,
            "file_size_display": _human_size(d.file_size),
            "status": d.status,
            "chunk_count": d.chunk_count,
            "parent_chunk_count": d.parent_chunk_count,
            "char_count": d.char_count,
            "uploaded_by": d.uploaded_by,
            "description": d.description,
            "error_msg": d.error_msg,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if page_size else 0,
    }


# ==================================================================
# 上传文档
# ==================================================================
@router.post(
    "/documents/upload",
    status_code=status.HTTP_201_CREATED,
    summary="上传文档并自动入库（仅管理员）",
    description=(
        "支持 PDF / Word / Excel / CSV / Markdown / TXT / HTML。\n\n"
        "上传后会自动完成：解析 → 清洗 → 切分 → 向量化 → 入库。\n"
        "这个过程可能耗时几秒到几十秒（取决于文件大小），"
        "所以接口是同步等待的——前端应显示进度提示。"
    ),
)
async def upload_document(
    admin: AdminUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(..., description="要上传的文档"),
    description: Optional[str] = Form(None, description="备注说明（可选）"),
) -> dict:
    """
    上传文档。

    【三道安全检查】
      1. **扩展名白名单**：只允许配置里列出的格式。
         为什么不用黑名单（禁止 .exe 之类）？因为黑名单永远列不全
         （.exe、.bat、.cmd、.scr、.pif……），而白名单是"只允许这些"，
         天然更安全。
      2. **文件大小限制**：防止有人传个 2GB 的文件把服务器撑爆。
      3. **文件名重命名**：见 ingest.py 里的说明（防路径穿越）。

    【还有一个容易忽略的点】
    即使扩展名是 .pdf，文件内容也可能是别的（把 exe 改名成 pdf）。
    真正的防御是"解析时用对应库去读"——如果内容不是合法 PDF，
    pypdf 会直接报错。所以解析失败本身也是一道防线。
    """
    check_rate_limit(f"upload:{admin.id}")

    if not file.filename:
        raise BadRequestError("文件名不能为空", code=40040)

    # ---- 修复中文文件名乱码（原因见 _fix_chinese_filename 的说明）----
    original_name = _fix_chinese_filename(file.filename)

    # ---- 检查 1：扩展名 ----
    if not is_supported(original_name):
        info = get_parser_info()
        raise BadRequestError(
            f"不支持的文件格式。当前支持：{', '.join(info['supported_extensions'])}",
            code=40041,
            detail={"file_name": original_name, "supported": info["supported_extensions"]},
        )

    # ---- 读取文件内容 ----
    content = await file.read()

    # ---- 检查 2：文件大小 ----
    if len(content) == 0:
        raise BadRequestError("文件是空的", code=40042)

    if len(content) > settings.max_upload_bytes:
        raise BadRequestError(
            f"文件太大（{_human_size(len(content))}），"
            f"上限是 {settings.MAX_UPLOAD_SIZE_MB}MB",
            code=40043,
        )

    logger.info(
        f"收到上传 | 文件={original_name} | "
        f"{_human_size(len(content))} | 上传者={admin.username}"
    )

    # ---- 执行入库 ----
    from app.rag.ingest import get_ingest_service

    result = await get_ingest_service().ingest(
        db=db,
        file_content=content,
        original_name=original_name,
        uploaded_by=admin.id,
        description=description,
    )

    if not result.success:
        raise BadRequestError(
            f"文档处理失败：{result.error}",
            code=40044,
            detail={"file_name": original_name},
        )

    return result.to_dict()


# ==================================================================
# 删除文档
# ==================================================================
@router.delete(
    "/documents/{doc_id}",
    response_model=SuccessResponse,
    summary="删除文档（仅管理员）",
    description="会同时删除向量、片段记录和磁盘文件，不可恢复。",
)
async def delete_document(
    doc_id: int,
    admin: AdminUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SuccessResponse:
    """删除文档及其全部关联数据"""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()

    if doc is None:
        raise NotFoundError("文档不存在", code=40440)

    from app.rag.ingest import get_ingest_service

    info = await get_ingest_service().delete_document(db, doc)

    logger.info(f"管理员删除文档 | {admin.username} 删除了 {info['file_name']}")

    return SuccessResponse(
        message=f"文档「{info['file_name']}」已删除",
        data=info,
    )


# ==================================================================
# 查看片段（解析预览）
# ==================================================================
@router.get(
    "/documents/{doc_id}/chunks",
    summary="查看文档切分出的片段（仅管理员）",
    description="用于检查切分质量——看看切出来的片段是不是语义完整、长度是否合适。",
)
async def get_document_chunks(
    doc_id: int,
    admin: AdminUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict:
    """
    查看某个文档切分出的片段。

    【为什么这个功能有用】
    切分质量直接决定检索质量，但它是个"黑盒"——
    上传完只能看到"切出了 47 个片段"，看不到切得对不对。

    有了这个接口，管理员能直接翻看每个片段，判断：
      · 有没有被生硬地切断句子
      · 有没有把不相关的内容混在一起
      · 长度是否合适（太短信息不全，太长检索不准）

    这是调参的依据，也是排查"为什么检索不准"的第一步。
    """
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise NotFoundError("文档不存在", code=40440)

    count_stmt = select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id)
    total = await db.scalar(count_stmt) or 0

    stmt = (
        select(Chunk)
        .where(Chunk.document_id == doc_id)
        .order_by(Chunk.chunk_index)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    chunks_result = await db.execute(stmt)
    chunks = chunks_result.scalars().all()

    return {
        "document": {
            "id": doc.id,
            "file_name": doc.file_name,
            "status": doc.status,
            "chunk_count": doc.chunk_count,
            "parent_chunk_count": doc.parent_chunk_count,
            "char_count": doc.char_count,
        },
        "items": [
            {
                "id": c.id,
                "chunk_index": c.chunk_index,
                "parent_index": c.parent_index,
                "content": c.content,
                "parent_content": c.parent_content,
                "page": c.page,
                "sheet_name": c.sheet_name,
                "char_count": c.char_count,
            }
            for c in chunks
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if page_size else 0,
    }


# ==================================================================
# 重新索引
# ==================================================================
@router.post(
    "/documents/{doc_id}/reindex",
    summary="重新索引文档（仅管理员）",
    description="按当前的切分参数重新处理该文档。调整过切分参数后用它生效。",
)
async def reindex_document(
    doc_id: int,
    admin: AdminUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """重新索引一份文档"""
    check_rate_limit(f"reindex:{admin.id}")

    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if doc is None:
        raise NotFoundError("文档不存在", code=40440)

    from app.rag.ingest import get_ingest_service

    reindex_result = await get_ingest_service().reindex(db, doc)

    if not reindex_result.success:
        raise BadRequestError(
            f"重新索引失败：{reindex_result.error}", code=40045
        )

    return reindex_result.to_dict()


# ==================================================================
# 知识库统计
# ==================================================================
@router.get(
    "/stats",
    summary="知识库统计（仅管理员）",
)
async def get_stats(
    admin: AdminUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """知识库整体情况统计"""
    # 文档数量（按状态分组）
    status_result = await db.execute(
        select(Document.status, func.count(Document.id)).group_by(Document.status)
    )
    by_status = {s: c for s, c in status_result.all()}

    total_docs = sum(by_status.values())

    # 总字数、总片段数
    agg_result = await db.execute(
        select(
            func.coalesce(func.sum(Document.char_count), 0),
            func.coalesce(func.sum(Document.chunk_count), 0),
        )
    )
    total_chars, total_chunks = agg_result.one()

    # 向量库实际数量（用于和数据库记录比对，发现不一致时说明有脏数据）
    vector_count = 0
    try:
        from app.rag.vectorstore import get_vector_store

        vector_count = get_vector_store().count()
    except Exception as e:
        logger.warning(f"读取向量库统计失败: {e}")

    # BM25 索引状态
    bm25_stats: dict = {}
    try:
        from app.rag.bm25 import get_bm25_index

        bm25_stats = get_bm25_index().get_stats()
    except Exception as e:
        logger.warning(f"读取 BM25 统计失败: {e}")

    # 缓存状态
    cache_stats: dict = {}
    try:
        from app.rag.semantic_cache import get_semantic_cache

        cache_stats = get_semantic_cache().get_stats()
    except Exception as e:
        logger.warning(f"读取缓存统计失败: {e}")

    return {
        "documents": {
            "total": total_docs,
            "by_status": by_status,
            "indexed": by_status.get(DocumentStatus.INDEXED, 0),
            "failed": by_status.get(DocumentStatus.FAILED, 0),
        },
        "content": {
            "total_chars": total_chars or 0,
            "total_chunks": total_chunks or 0,
            "vector_chunks": vector_count,
            # 这两个数字不一致说明有脏数据（比如删文档时向量没删干净）
            "consistent": vector_count == (total_chunks or 0),
        },
        "bm25_index": bm25_stats,
        "semantic_cache": cache_stats,
        "supported_formats": get_parser_info(),
    }


# ==================================================================
# 检索测试（调参用）
# ==================================================================
@router.post(
    "/search",
    summary="检索测试：只看检索结果，不生成回答（仅管理员）",
    description=(
        "用于调试和调参。返回检索到的片段及其各阶段分数，"
        "但不调用大模型生成回答——所以又快又省额度。\n\n"
        "排查「为什么这个问题答得不好」时，第一步就是用它看检索到了什么。"
    ),
)
async def search_test(
    admin: AdminUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    query: str = Form(..., description="测试查询"),
    top_k: int = Form(5, ge=1, le=20, description="返回条数"),
    compare: bool = Form(False, description="是否对比三种检索配置的效果"),
) -> dict:
    """
    检索测试。

    【这个接口的价值】
    调试 RAG 系统时，最常遇到的问题是"回答不对"。
    但"回答不对"可能是两个完全不同的原因：
      A. 检索没找到正确资料 → 问题在检索环节
      B. 检索对了但模型没用 → 问题在提示词环节

    这个接口把检索结果单独暴露出来，就能立刻分辨是 A 还是 B。
    没有它的话，你只能盯着一个错误答案瞎猜。
    """
    check_rate_limit(f"search:{admin.id}")

    from app.rag.retriever import get_retriever

    retriever = get_retriever()

    results, debug = retriever.retrieve(query, top_k=top_k, return_debug=True)

    payload: dict = {
        "query": query,
        "results": [
            {
                "rank": i + 1,
                "file_name": r.get("file_name"),
                "page": r.get("page"),
                "content": r.get("content"),
                "parent_content": r.get("parent_content"),
                "vector_score": r.get("score") if r.get("source") != "bm25" else None,
                "bm25_score": r.get("score") if r.get("source") == "bm25" else None,
                "rrf_score": r.get("rrf_score"),
                "rerank_score": r.get("rerank_score"),
                "hit_both": r.get("hit_both", False),
                "sources": r.get("sources", []),
            }
            for i, r in enumerate(results)
        ],
        "timings": debug,
    }

    # 可选的三种配置对比
    if compare:
        payload["comparison"] = {}
        for name, rerank, bm25 in (
            ("vector_only", False, False),
            ("hybrid", False, True),
            ("hybrid_rerank", True, True),
        ):
            original = (retriever.enable_rerank, retriever.enable_bm25)
            retriever.enable_rerank, retriever.enable_bm25 = rerank, bm25
            r = retriever.retrieve(query, top_k=top_k)
            retriever.enable_rerank, retriever.enable_bm25 = original

            payload["comparison"][name] = [
                {
                    "rank": i + 1,
                    "file_name": x.get("file_name"),
                    "score": x.get("rerank_score") or x.get("score"),
                    "content": (x.get("content") or "")[:100],
                }
                for i, x in enumerate(r)
            ]

    return payload


# ==================================================================
# 辅助函数
# ==================================================================
def _human_size(size: int) -> str:
    """把字节数转成人看得懂的大小（1536 → "1.5 KB"）"""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.2f} GB"


def _fix_chinese_filename(raw: str) -> str:
    """
    修复中文文件名乱码（防御性处理）。

    ================================================================
    【先说结论：浏览器上传是正常的，不需要修】
    ================================================================
    实测验证：用 UTF-8 字节传中文文件名（这正是现代浏览器的行为），
    FastAPI 能正确处理，数据库里存的就是「售后服务手册.md」。

    ================================================================
    【那什么时候会出问题】
    ================================================================
    某些命令行工具在有中文的 Windows 上，会把文件名按**系统编码（GBK）**
    转换后再发送，而不是标准要求的 UTF-8。

    实测：用 curl 上传「售后服务手册.md」，到达服务端时变成了：
        ÊÛºó·þÎñÊÖ²á.md

    反推这些字符的字节码是 CA DB BA F3 B7 FE ...，
    正是「售后服务手册」的 **GBK 编码**——
    也就是说这串字节被当成 latin-1 逐字节"翻译"成字符了。

    ================================================================
    【修复方法：反向翻译】
    ================================================================
    先把字符按 latin-1 编回原始字节，再按正确的编码去解：

        错误字符串 .encode('latin-1')  →  原始字节
        原始字节   .decode('utf-8')    →  正确的文字

    优先按 UTF-8 解（标准行为）；解不通再试 GBK（兼容老工具）。

    ================================================================
    【安全性：对正常的文件名绝不会改坏】
    ================================================================
      · 纯英文文件名：两种编码下结果一样，转换后原样不变
      · 本来就是正确的中文：encode('latin-1') 会直接抛异常
        （因为中文字符不在 latin-1 范围内），于是走兜底返回原值
      · 转换失败：一律返回原值，绝不硬转

    也就是说，这个函数**只在确实检测到乱码时才动手**，其余情况是透明的。
    """
    # 只处理"看起来像乱码"的情况：纯 ASCII 或已经正常的中文都不用管
    try:
        raw_bytes = raw.encode("latin-1")
    except UnicodeEncodeError:
        # 有字符超出 latin-1 范围 → 说明这是正常的 Unicode 字符串，不用修
        return raw

    # 依次尝试常见编码
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            candidate = raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

        # 简单校验：解出来的结果应该"更像正常文件名"
        # 比如不含 latin-1 的那些怪字符（Ê Û º 这类）
        if candidate and _looks_like_valid_name(candidate):
            logger.debug(f"修复文件名编码 | {raw!r} → {candidate!r}（按 {encoding} 解码）")
            return candidate

    return raw


def _looks_like_valid_name(text: str) -> bool:
    """
    粗略判断一个字符串"像不像正常的文件名"。

    判断依据：**不应该包含大量拉丁扩展字符**。

    正常情况下，文件名里的字符要么是 ASCII（英文数字），
    要么是各种文字（中文、日文等）。
    而"被错误解码"的乱码会塞满 Ê Û º ó þ 这类拉丁扩展字符——
    真实文件名几乎不会长这样。
    """
    if not text:
        return False

    # 统计拉丁扩展字符（U+0080 ~ U+024F）的占比
    latin_extended = sum(1 for ch in text if 0x80 <= ord(ch) <= 0x024F)

    # 超过 20% 就认为是乱码
    return (latin_extended / len(text)) < 0.2
