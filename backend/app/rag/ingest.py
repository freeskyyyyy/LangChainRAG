"""
文档入库流水线 —— 把上传的文件变成可检索的知识

================================================================
【完整流程】
================================================================
上传文件
   ↓
① 解析（loader.py）      各种格式 → 统一的"文本 + 元数据"
   ↓
② 清洗（cleaner.py）     去页眉页脚、修 PDF 换行、删隐藏字符
   ↓
③ 切分（splitter.py）    切成父子两层：子块检索、父块喂模型
   ↓
④ 向量化（embeddings.py）每段变成语义坐标
   ↓
⑤ 入库（vectorstore.py） 存进 Chroma
   ↓
⑥ 存元数据（SQLite）     供管理页面展示、支持删除和重建
   ↓
⑦ 让缓存失效             BM25 索引和语义缓存都要重建，否则搜不到新内容

================================================================
【为什么要把第 ⑦ 步单独强调】
================================================================
这是个非常容易漏掉、又非常影响体验的细节：

  · BM25 索引是内存里的，新建/删除文档后必须让它失效重建，
    否则新上传的资料**永远搜不到**（因为索引里还是旧数据）
  · 语义缓存里可能存着"这个问题知识库里没有"的旧结论，
    上传新资料后，同样的问题应该能答上来了，
    但缓存会拦住它，导致**新资料形同虚设**

这两个坑都很隐蔽——功能看起来是好的，只是"新数据不生效"，
而且重启程序后又好了，特别难排查。
"""

from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.models import Chunk, Document, DocumentStatus
from app.rag.cleaner import TextCleaner
from app.rag.loader import DocumentParser
from app.rag.splitter import ParentChildChunk, ParentChildSplitter


@dataclass
class IngestResult:
    """入库结果"""

    success: bool
    doc_id: Optional[int] = None
    file_name: str = ""
    chunk_count: int = 0
    parent_count: int = 0
    char_count: int = 0
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "chunk_count": self.chunk_count,
            "parent_count": self.parent_count,
            "char_count": self.char_count,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "warnings": self.warnings,
        }


class IngestService:
    """文档入库服务"""

    def __init__(self):
        self.cleaner = TextCleaner()
        self.splitter = ParentChildSplitter()

    # ==============================================================
    # 保存上传的文件
    # ==============================================================
    @staticmethod
    def save_upload(content: bytes, original_name: str) -> Path:
        """
        把上传的文件存到磁盘。

        【为什么要重命名文件】
        三个原因：
        1. **防止重名覆盖**：两个用户都传"商品资料.docx"，后传的会覆盖前一个
        2. **防止路径穿越攻击**：恶意文件名可能是 "../../windows/system32/xx"，
           直接用它拼路径会把文件写到系统目录去。用自动生成的 UUID 做文件名
           就彻底杜绝了这个风险。
        3. **防止特殊字符问题**：中文文件名、空格、emoji 在某些文件系统上会出问题

        原始文件名会存在数据库里（Document.file_name），
        展示给用户看的是它，磁盘上用的是 UUID。
        """
        settings.ensure_dirs()

        ext = Path(original_name).suffix.lower()
        # UUID 是几乎不可能重复的随机字符串
        stored_name = f"{uuid.uuid4().hex}{ext}"
        target = settings.upload_path / stored_name

        target.write_bytes(content)
        logger.info(f"文件已保存 | 原名={original_name} | 存储名={stored_name} | {len(content)} 字节")

        return target

    # ==============================================================
    # 主流程
    # ==============================================================
    async def ingest(
        self,
        db: AsyncSession,
        file_content: bytes,
        original_name: str,
        uploaded_by: Optional[int] = None,
        description: Optional[str] = None,
    ) -> IngestResult:
        """
        完整的入库流程。

        【一个重要的设计决定：解析是"同步阻塞"的】
        解析 PDF、切分文本都是 CPU 密集操作，会占满一个核。
        在异步框架里，如果直接 await 执行，会卡住整个服务器。

        所以这里用 asyncio.to_thread 把重活丢到线程池。
        这样在解析一份大文档的几十秒里，其他用户依然能正常提问。
        """
        t_start = time.perf_counter()
        result = IngestResult(success=False, file_name=original_name)

        # ---------------- 步骤 1：保存文件 ----------------
        try:
            file_path = await asyncio.to_thread(
                self.save_upload, file_content, original_name
            )
        except Exception as e:
            result.error = f"文件保存失败：{e}"
            logger.error(result.error)
            return result

        # ---------------- 步骤 2：创建数据库记录 ----------------
        doc = Document(
            file_name=original_name,
            stored_name=file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            file_size=len(file_content),
            file_path=str(file_path),
            status=DocumentStatus.PARSING,
            uploaded_by=uploaded_by,
            description=description,
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        result.doc_id = doc.id

        logger.info(f"开始处理文档 | id={doc.id} | 文件={original_name}")

        # ---------------- 步骤 3：解析 → 清洗 → 切分 ----------------
        try:
            chunks = await asyncio.to_thread(
                self._process_file, file_path, doc.id, original_name
            )
        except Exception as e:
            # 处理失败要把状态写回数据库，让管理员在管理页面看到失败原因，
            # 而不是"传了半天没反应也不知道为什么"
            doc.status = DocumentStatus.FAILED
            doc.error_msg = f"{type(e).__name__}: {str(e)[:500]}"
            await db.commit()

            result.error = str(e)
            result.elapsed_seconds = time.perf_counter() - t_start
            logger.error(f"文档处理失败 | id={doc.id} | {type(e).__name__}: {e}")
            return result

        if not chunks:
            doc.status = DocumentStatus.FAILED
            doc.error_msg = "文件里没有提取到有效内容（可能是空文件或扫描件）"
            await db.commit()

            result.error = doc.error_msg
            result.elapsed_seconds = time.perf_counter() - t_start
            return result

        # ---------------- 步骤 4：存入向量库 ----------------
        try:
            from app.rag.vectorstore import get_vector_store

            vs = get_vector_store()
            added = await asyncio.to_thread(vs.add_chunks, chunks)
        except Exception as e:
            doc.status = DocumentStatus.FAILED
            doc.error_msg = f"向量化或存储失败：{str(e)[:400]}"
            await db.commit()

            result.error = doc.error_msg
            result.elapsed_seconds = time.perf_counter() - t_start
            logger.error(f"向量库写入失败 | id={doc.id} | {e}")
            return result

        # ---------------- 步骤 5：写入片段元数据（SQLite）----------------
        try:
            await self._save_chunk_metadata(db, doc, chunks)
        except Exception as e:
            # 元数据写入失败不影响检索（向量已经进去了），
            # 但要记警告——因为管理页面的"片段预览"会看不到内容
            result.warnings.append(f"片段元数据保存失败（不影响检索）：{e}")
            logger.warning(f"片段元数据保存失败 | doc_id={doc.id} | {e}")

        # ---------------- 步骤 6：更新文档状态 ----------------
        parent_count = len({c.parent_index for c in chunks})
        total_chars = sum(len(c.child_content) for c in chunks)

        doc.status = DocumentStatus.INDEXED
        doc.chunk_count = added
        doc.parent_chunk_count = parent_count
        doc.char_count = total_chars
        doc.error_msg = None
        await db.commit()

        # ---------------- 步骤 7：让缓存失效 ----------------
        # ⚠️ 这一步绝对不能漏（原因见文件开头的说明）
        await self._invalidate_caches()

        result.success = True
        result.chunk_count = added
        result.parent_count = parent_count
        result.char_count = total_chars
        result.elapsed_seconds = time.perf_counter() - t_start

        logger.info(
            f"文档入库完成 | id={doc.id} | {original_name} | "
            f"{parent_count} 父块 / {added} 子块 | "
            f"耗时 {result.elapsed_seconds:.1f} 秒"
        )
        return result

    # ==============================================================
    # 内部：处理文件（在子线程里执行）
    # ==============================================================
    def _process_file(
        self,
        file_path: Path,
        doc_id: int,
        original_name: str = "",
    ) -> list[ParentChildChunk]:
        """解析 → 清洗 → 切分（这三个都是 CPU 密集操作）"""
        # ---- 解析 ----
        docs = DocumentParser.parse(file_path)
        logger.debug(f"解析完成 | doc_id={doc_id} | {len(docs)} 个文档块")

        # ------------------------------------------------------------------
        # ⚠️ 关键修正：把"来源文件名"改回原始名字
        #
        # 为什么会错：我们在存文件时用 UUID 重命名了（防止重名覆盖和路径穿越攻击），
        # 所以解析器从磁盘路径里读到的是 "3d8948bd8eda432296a65e67d76d86aa.md"
        # 这种乱码名字。
        #
        # 如果不改回来，引用卡片就会显示这串 UUID 给用户看——
        # 用户根本不知道那是什么文件。这是诊断时实测发现的问题。
        # ------------------------------------------------------------------
        if original_name:
            for doc in docs:
                doc.metadata["source"] = original_name
                doc.metadata["stored_name"] = file_path.name

        # ---- 清洗 ----
        docs = self.cleaner.clean_documents(docs)
        logger.debug(f"清洗完成 | doc_id={doc_id} | 剩余 {len(docs)} 个文档块")

        if not docs:
            raise ValueError("清洗后没有剩下任何有效内容")

        # ---- 切分 ----
        chunks = self.splitter.split(docs, doc_id=doc_id)
        logger.debug(f"切分完成 | doc_id={doc_id} | {len(chunks)} 个子块")

        return chunks

    # ==============================================================
    # 内部：保存片段元数据
    # ==============================================================
    async def _save_chunk_metadata(
        self,
        db: AsyncSession,
        doc: Document,
        chunks: list[ParentChildChunk],
    ) -> None:
        """
        把片段信息写进 SQLite。

        为什么向量库里已经存了还要再存一份？
        见 models/knowledge.py 里 Chunk 类的说明——主要是为了：
        · 管理页面能快速列出片段（不用去向量库捞）
        · 删除时能知道要删哪些向量 ID
        · 将来重新索引时有原始文本可用
        """
        records = [
            Chunk(
                document_id=doc.id,
                chunk_index=c.child_index,
                content=c.child_content,
                parent_content=c.parent_content,
                parent_index=c.parent_index,
                page=c.metadata.get("page"),
                sheet_name=c.metadata.get("sheet_name"),
                vector_id=c.vector_id,
                char_count=len(c.child_content),
            )
            for c in chunks
        ]

        db.add_all(records)
        await db.commit()

    # ==============================================================
    # 内部：让缓存失效
    # ==============================================================
    @staticmethod
    async def _invalidate_caches() -> None:
        """让 BM25 索引和语义缓存失效（放在线程里执行，避免阻塞）"""
        def _do() -> None:
            try:
                from app.rag.bm25 import invalidate as invalidate_bm25

                invalidate_bm25()
            except Exception as e:
                logger.warning(f"BM25 索引失效失败: {e}")

            try:
                from app.rag.semantic_cache import invalidate_semantic_cache

                invalidate_semantic_cache()
            except Exception as e:
                logger.warning(f"语义缓存失效失败: {e}")

        await asyncio.to_thread(_do)

    # ==============================================================
    # 删除文档
    # ==============================================================
    async def delete_document(self, db: AsyncSession, doc: Document) -> dict[str, Any]:
        """
        删除文档及其全部关联数据。

        要删四样东西（漏一样就会留下垃圾）：
          1. 向量库里的向量
          2. SQLite 里的片段记录
          3. SQLite 里的文档记录
          4. 磁盘上的原始文件
        """
        doc_id = doc.id
        file_name = doc.file_name
        deleted_vectors = 0

        # ---- 1. 删向量 ----
        try:
            from app.rag.vectorstore import get_vector_store

            vs = get_vector_store()
            deleted_vectors = await asyncio.to_thread(vs.delete_by_document, doc_id)
        except Exception as e:
            logger.error(f"删除向量失败 | doc_id={doc_id} | {e}")

        # ---- 2、3. 删数据库记录（片段会因级联删除自动一起删掉）----
        stored_path = Path(doc.file_path) if doc.file_path else None
        await db.delete(doc)
        await db.commit()

        # ---- 4. 删磁盘文件 ----
        try:
            if stored_path and stored_path.exists():
                stored_path.unlink()
        except Exception as e:
            logger.warning(f"删除磁盘文件失败 | {stored_path} | {e}")

        # ---- 5. 让缓存失效 ----
        await self._invalidate_caches()

        logger.info(f"文档已删除 | id={doc_id} | {file_name} | 移除 {deleted_vectors} 个向量")

        return {
            "doc_id": doc_id,
            "file_name": file_name,
            "deleted_vectors": deleted_vectors,
        }

    # ==============================================================
    # 重新索引
    # ==============================================================
    async def reindex(self, db: AsyncSession, doc: Document) -> IngestResult:
        """
        重新索引一份文档。

        【什么时候用】
        调整了切分参数（chunk_size、重叠等）之后，想让已有文档
        按新参数重新切分入库。

        做法：先删掉旧的向量和片段，再重新跑一遍处理流程。
        原始文件还在磁盘上，所以不用重新上传。
        """
        file_path = Path(doc.file_path)
        if not file_path.exists():
            return IngestResult(
                success=False,
                doc_id=doc.id,
                file_name=doc.file_name,
                error="原始文件不存在，无法重新索引（可能已被手动删除）",
            )

        logger.info(f"开始重新索引 | doc_id={doc.id} | {doc.file_name}")

        # ---- 清掉旧的向量和片段 ----
        try:
            from app.rag.vectorstore import get_vector_store

            await asyncio.to_thread(
                get_vector_store().delete_by_document, doc.id
            )
        except Exception as e:
            logger.warning(f"清理旧向量失败（继续执行）: {e}")

        from sqlalchemy import delete as sql_delete

        await db.execute(sql_delete(Chunk).where(Chunk.document_id == doc.id))
        await db.commit()

        # ---- 重新处理 ----
        doc.status = DocumentStatus.PARSING
        await db.commit()

        t_start = time.perf_counter()
        result = IngestResult(success=False, doc_id=doc.id, file_name=doc.file_name)

        try:
            chunks = await asyncio.to_thread(
                self._process_file, file_path, doc.id, doc.file_name
            )
            if not chunks:
                raise ValueError("重新处理后没有产生任何片段")

            from app.rag.vectorstore import get_vector_store

            added = await asyncio.to_thread(get_vector_store().add_chunks, chunks)
            await self._save_chunk_metadata(db, doc, chunks)

            parent_count = len({c.parent_index for c in chunks})
            doc.status = DocumentStatus.INDEXED
            doc.chunk_count = added
            doc.parent_chunk_count = parent_count
            doc.char_count = sum(len(c.child_content) for c in chunks)
            doc.error_msg = None
            await db.commit()

            await self._invalidate_caches()

            result.success = True
            result.chunk_count = added
            result.parent_count = parent_count
            result.char_count = doc.char_count
            result.elapsed_seconds = time.perf_counter() - t_start

            logger.info(f"重新索引完成 | doc_id={doc.id} | {added} 个子块")

        except Exception as e:
            doc.status = DocumentStatus.FAILED
            doc.error_msg = f"重新索引失败：{str(e)[:400]}"
            await db.commit()

            result.error = str(e)
            result.elapsed_seconds = time.perf_counter() - t_start
            logger.error(f"重新索引失败 | doc_id={doc.id} | {e}")

        return result


# ==================================================================
# 单例
# ==================================================================
_service: Optional[IngestService] = None


def get_ingest_service() -> IngestService:
    """获取全局入库服务（单例）"""
    global _service
    if _service is None:
        _service = IngestService()
    return _service
