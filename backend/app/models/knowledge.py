"""
知识库文档表 与 知识片段表

【这两张表的关系】
上传一份 PDF → 一条 Document 记录
这份 PDF 被切成 50 个小段 → 50 条 Chunk 记录

    Document ──< Chunk

【为什么片段要单独存一张表？既然原文都在向量库里了】
三个理由：
1. **管理页面要能预览**：管理员上传后想看看"切出来的片段长什么样、合不合理"，
   直接从 SQLite 查比去向量库捞方便得多。
2. **删除时要清理干净**：删文档时，要把它在向量库里的所有向量一并删掉。
   向量库里的向量是按 ID 存的，得先知道这个文档产生了哪些 ID。
3. **重建索引要用**：将来如果调整了切分参数想重新切分，需要原始文本。

【关于父子文档策略的两个字段】
content        = 子块（200~400 字）→ 拿去向量化，用于"精准检索"
parent_content = 父块（1000~1500 字）→ 检索命中子块后，返回它所在的父块给模型，
                 让模型看到更完整的上下文。
比喻：用目录精确找到是哪一章（子块），然后把整章拿给模型看（父块）。
"""

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.base import TimestampMixin

if TYPE_CHECKING:
    pass


class DocumentStatus:
    """文档处理状态常量"""

    PENDING = "pending"     # 已上传，排队等待解析
    PARSING = "parsing"     # 正在解析、切分、向量化
    INDEXED = "indexed"     # 已入库，可以参与检索了
    FAILED = "failed"       # 处理失败（格式损坏、加密、内容为空等）


class Document(Base, TimestampMixin):
    """知识库文档表 —— 一份被上传的原始文件"""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ---------------- 文件信息 ----------------
    file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="原始文件名（展示给管理员看的）",
    )

    stored_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="实际存储的文件名（重命名过，防止重名覆盖和路径攻击）",
    )

    file_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="文件类型：pdf / docx / xlsx / md / txt / html",
    )

    file_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="文件大小（字节）",
    )

    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="文件在磁盘上的完整路径",
    )

    # ---------------- 处理状态 ----------------
    status: Mapped[str] = mapped_column(
        String(20),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
        comment="处理状态：pending / parsing / indexed / failed",
    )

    error_msg: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="失败原因（解析失败时记录，供管理员排查）",
    )

    # ---------------- 处理结果统计 ----------------
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="切分出的子块数量",
    )

    parent_chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="切分出的父块数量",
    )

    char_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="文档总字数（解析清洗后）",
    )

    # ---------------- 元信息 ----------------
    uploaded_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="上传者（用户被删时置空，但文档保留）",
    )

    description: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="备注说明（管理员可选填，例如：2026秋季新品资料）",
    )

    # ---------------- 关联关系 ----------------
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",  # 删文档时，片段一并删除
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} file={self.file_name!r} status={self.status}>"


class Chunk(Base):
    """
    知识片段表 —— 文档被切开后的一个个小段

    这是 RAG 检索的最小单位：用户提问 → 检索到最相关的几个片段 → 交给模型组织回答。
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属文档",
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="片段在原文档中的序号（从 0 开始）",
    )

    # ---------------- 两层文本（父子文档策略）----------------
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="子块文本（用于向量检索，较短较精准）",
    )

    parent_content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="父块文本（检索命中后交给模型，上下文更完整）",
    )

    parent_index: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="所属父块的序号",
    )

    # ---------------- 定位信息（用于回答里标注出处）----------------
    page: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="所在页码（PDF/Word 有效）",
    )

    sheet_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="所在工作表名（Excel 有效）",
    )

    # ---------------- 与向量库的对应关系 ----------------
    vector_id: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
        index=True,
        comment="对应向量库中的向量 ID（删除文档时靠它找到要删的向量）",
    )

    # ---------------- 其他 ----------------
    char_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="片段字数",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
        comment="创建时间",
    )

    # ---------------- 关联关系 ----------------
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        # 按「所属文档 + 片段序号」查询最常用（管理页预览片段、重建索引时）
        Index("ix_chunks_doc_index", "document_id", "chunk_index"),
    )

    def __repr__(self) -> str:
        preview = (self.content[:20] + "...") if len(self.content) > 20 else self.content
        return f"<Chunk id={self.id} doc={self.document_id} idx={self.chunk_index} {preview!r}>"
