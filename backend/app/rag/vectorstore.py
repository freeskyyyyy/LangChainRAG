"""
向量库封装 —— 语义坐标的"仓库"

================================================================
【Chroma 在这里扮演什么角色】
================================================================
切分出来的子块，每个都会：
  · 被向量化成一串 1024 个数字（语义坐标）
  · 连同原文和元数据一起存进 Chroma

用户提问时，问题也被向量化，然后 Chroma 帮我们**快速找出坐标最接近的
那几段文字**——这就是"按意思搜索"。

【为什么不用普通数据库做这件事】
假设有 10 万个片段，每个是 1024 维向量。
用户提问 → 要和全部 10 万个向量算一次距离 → 10 万次计算。
每次问答都这么干，太慢了。

Chroma 内部用了 **HNSW**（一种"近似最近邻"索引算法）：
它预先建好一张"导航图"，找最近邻时不用遍历全部，
而是像"顺着路标走"一样快速逼近目标，几十毫秒就能在 10 万个向量里找到最像的。

代价是"近似"——可能漏掉极少数的真正最近邻。但在实际业务里，
找到的第 3 名和第 1 名差别不大（后面还有重排序兜底），
所以这个取舍完全值得。

================================================================
【两个关键设计】
================================================================
1. **距离度量用余弦（cosine）而不是欧氏距离**
   余弦只看"方向"不看"长度"，能避免"长句子的向量天然比短句子长"
   导致的偏差。对文本相似度来说，余弦是更合适的选择。

2. **Chroma 直接复用我们的 EmbeddingService**
   EmbeddingService 恰好有 Chroma 需要的两个方法
   （embed_documents / embed_query），所以可以直接传进去当"向量化函数"。
   好处是：**向量化的缓存、批量、重试逻辑全部自动生效**，
   不需要在 Chroma 这边再写一遍。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import logger
from app.rag.embeddings import EmbeddingService, get_embedding_service
from app.rag.splitter import ParentChildChunk


class VectorStoreService:
    """
    向量库服务

    用法：
        vs = VectorStoreService()
        vs.add_chunks(chunks)                       # 存入
        results = vs.search("电池多大", top_k=20)   # 检索
        vs.delete_by_document(doc_id=1)             # 按文档删除
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | None = None,
        embedder: EmbeddingService | None = None,
    ):
        from langchain_chroma import Chroma

        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.persist_dir = str(persist_dir or settings.chroma_path)

        # 复用全局向量化服务（缓存、批量、重试都自动生效）
        self.embedder = embedder or get_embedding_service()

        logger.info(f"正在连接向量库 | 集合={self.collection_name} | 目录={self.persist_dir}")

        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            persist_directory=self.persist_dir,
            # 余弦距离：值越小表示越相似（0 = 完全相同方向）
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ==============================================================
    # 写入
    # ==============================================================
    def add_chunks(self, chunks: Sequence[ParentChildChunk]) -> int:
        """
        把切分好的子块存进向量库。

        【存的是什么】
          · 向量：由 child_content（子块）算出来 —— 小，检索才准
          · 原文：child_content（给人看）
          · 元数据：
              - parent_content：父块原文 —— 检索命中后交给模型的是这个
              - file_name/page/sheet_name：用于引用溯源
              - doc_id：用于按文档批量删除

        【为什么要显式传 embeddings 而不让 Chroma 自己算】
        因为我们要用自己那套带缓存和批量的向量化逻辑。
        如果让 Chroma 自己算，它内部是逐条调用的，
        既没有缓存也没有批量，慢好几倍。
        """
        if not chunks:
            return 0

        texts = [c.child_content for c in chunks]
        ids = [c.vector_id for c in chunks]
        metadatas = [self._build_metadata(c) for c in chunks]

        # 批量算向量（带缓存：内容没变的话这里几乎不花时间）
        logger.info(f"正在向量化 {len(texts)} 个子块…")
        embeddings = self.embedder.embed_documents(texts)

        # upsert = update + insert
        # 用 upsert 而不是 add：重复索引同一份文档时是**覆盖**而不是新增，
        # 避免向量库里堆积重复数据
        self._store._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"已存入向量库 | {len(ids)} 个片段 | 集合={self.collection_name}")
        return len(ids)

    @staticmethod
    def _build_metadata(chunk: ParentChildChunk) -> dict[str, Any]:
        """
        构造写入向量库的元数据。

        ⚠️ Chroma 的元数据只支持 str / int / float / bool，
        不支持 list 和 None。所以这里要把 None 转成空字符串，
        把其他类型统一转成能接受的形式。
        """
        m = chunk.metadata

        meta: dict[str, Any] = {
            # ---- 唯一标识 ----
            # 存一份在元数据里，这样检索结果返回时能拿到它，
            # 用于和 BM25 的结果做去重（两路可能命中同一片段）
            "vid": chunk.vector_id,
            # ---- 引用溯源必需 ----
            "file_name": str(m.get("source") or m.get("file_name") or "未知文档"),
            "page": int(m["page"]) if isinstance(m.get("page"), int) else -1,
            # ---- 父子文档策略必需 ----
            "parent_content": chunk.parent_content,
            # ---- 删除和重建必需 ----
            "doc_id": str(m.get("doc_id", "")),
            "parent_index": int(chunk.parent_index),
            "child_index": int(chunk.child_index),
        }

        # 可选字段
        if m.get("sheet_name"):
            meta["sheet_name"] = str(m["sheet_name"])
        if m.get("section"):
            meta["section"] = str(m["section"])
        if m.get("title"):
            meta["title"] = str(m["title"])

        return meta

    # ==============================================================
    # 检索
    # ==============================================================
    def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        语义检索：找出和 query 意思最接近的若干片段。

        参数：
            query           —— 查询文本（用户问题或改写后的问题）
            top_k           —— 返回多少条
            score_threshold —— 相似度低于这个值的不返回
            where           —— 元数据过滤条件，比如 {"doc_id": "3"} 只搜某份文档

        返回：列表，每项包含
            {
                "content":      子块原文,
                "parent_content": 父块原文（交给模型的是这个）,
                "score":        相似度 0~1（越大越相关）,
                "file_name":    来源文件名,
                "page":         页码,
                ...
            }

        【相似度分数怎么算的】
        Chroma 用余弦距离返回（0 = 完全一致，2 = 完全相反）。
        我们换算成更直观的"相似度"：similarity = 1 - distance
        这样 1 = 完全相关，0 = 毫无关系，符合直觉。
        """
        k = top_k or settings.RETRIEVAL_TOP_K

        logger.debug(f"向量检索 | query={query[:50]!r} | top_k={k}")

        try:
            results = self._store.similarity_search_with_score(
                query=query,
                k=k,
                filter=where,
            )
        except Exception as e:
            logger.error(f"向量检索失败: {type(e).__name__}: {e}")
            raise

        items: list[dict[str, Any]] = []

        for doc, distance in results:
            # 余弦距离 → 相似度
            score = max(0.0, min(1.0, 1.0 - float(distance)))

            if score_threshold is not None and score < score_threshold:
                continue

            meta = doc.metadata or {}
            items.append(
                {
                    "content": doc.page_content,
                    "parent_content": meta.get("parent_content") or doc.page_content,
                    "score": round(score, 4),
                    "file_name": meta.get("file_name", "未知文档"),
                    "page": meta.get("page", -1) if meta.get("page", -1) > 0 else None,
                    "sheet_name": meta.get("sheet_name"),
                    "section": meta.get("section"),
                    "doc_id": meta.get("doc_id"),
                    "parent_index": meta.get("parent_index"),
                    "child_index": meta.get("child_index"),
                    "source": "vector",   # 标记来源，方便调试混合检索
                }
            )

        logger.debug(f"向量检索完成 | 命中 {len(items)} 条 | 最高分 {items[0]['score'] if items else 0}")
        return items

    # ==============================================================
    # 删除
    # ==============================================================
    def delete_by_document(self, doc_id: int | str) -> int:
        """
        删除某个文档的全部向量。

        【为什么用 where 过滤删除而不是按 ID 列表删】
        因为一个文档可能产生几百个片段，要把它们的 ID 全查出来
        再传给删除接口，既慢又占内存。
        用元数据过滤（where={"doc_id": "3"}）让 Chroma 自己去筛，
        一次调用就搞定。
        """
        before = self.count()
        try:
            self._store._collection.delete(where={"doc_id": str(doc_id)})
        except Exception as e:
            logger.error(f"删除向量失败 | doc_id={doc_id} | {type(e).__name__}: {e}")
            raise

        after = self.count()
        removed = before - after
        logger.info(f"已删除文档向量 | doc_id={doc_id} | 移除 {removed} 个片段")
        return removed

    def delete_all(self) -> None:
        """
        清空整个向量库。

        ⚠️ 危险操作：会删掉所有知识库内容。
        管理接口里调用前必须二次确认。

        【踩过的坑：where={} 是不合法的】
        我最初写的是 `_collection.delete(where={})`，想当然地以为
        "传个空条件 = 删全部"（SQL 里 DELETE FROM t 确实是这样）。
        但 ChromaDB 不吃这一套，它会直接抛：
            ValueError: Expected where to have exactly one operator, got {}
        它要求 where 里**必须正好有一个操作符**，{} 是零个，不合法。

        这个坑一直没被发现，是因为这个方法当时**根本没被调用过**
        （"清空知识库"的接口还没接）。直到写切片对比脚本要清沙盒，
        它才第一次真正被执行，然后当场就炸了。

        教训：**没被调用过的代码等于没被验证过的代码**，
        写的时候再自信也不算数。

        现在改用 reset_collection()：删掉集合再原样重建。
        它会连 collection_metadata 一起重建，所以
        "余弦距离（cosine）"这个设定不会丢
        —— 这个很重要，丢了的话相似度的算法就变了，
        整个检索的结果会整体失真，而且不报错，很难查。
        """
        try:
            self._store.reset_collection()
            logger.warning("向量库已清空！")
        except Exception as e:
            logger.error(f"清空向量库失败: {e}")
            raise

    # ==============================================================
    # 统计与查询
    # ==============================================================
    def count(self) -> int:
        """向量库里现在有多少个片段"""
        try:
            return self._store._collection.count()
        except Exception:
            return 0

    def get_document_chunks(self, doc_id: int | str, limit: int = 500) -> list[dict[str, Any]]:
        """
        取出某个文档的所有片段（管理页面预览用）。

        注意这里**不做向量检索**，只是按元数据过滤把原文捞出来。
        """
        try:
            raw = self._store._collection.get(
                where={"doc_id": str(doc_id)},
                limit=limit,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.error(f"读取文档片段失败 | doc_id={doc_id} | {e}")
            return []

        items: list[dict[str, Any]] = []
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []

        for i, _vid in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            items.append(
                {
                    "content": documents[i] if i < len(documents) else "",
                    "parent_content": (meta or {}).get("parent_content", ""),
                    "file_name": (meta or {}).get("file_name", ""),
                    "page": (meta or {}).get("page"),
                    "child_index": (meta or {}).get("child_index"),
                }
            )

        items.sort(key=lambda x: x.get("child_index") or 0)
        return items

    def get_stats(self) -> dict[str, Any]:
        """向量库统计信息"""
        return {
            "collection": self.collection_name,
            "persist_dir": self.persist_dir,
            "total_chunks": self.count(),
            "embedding_model": self.embedder.model,
            "embedding_stats": self.embedder.get_stats(),
        }

    def health_check(self) -> tuple[bool, str]:
        """
        健康检查：确认向量库可用。

        做一次"写入-读取-删除"的小循环，确保不是只连上了但实际不能用。
        """
        try:
            test_id = "__health_check__"
            self._store._collection.upsert(
                ids=[test_id],
                embeddings=[[0.0] * 1024],
                documents=["健康检查"],
                metadatas=[{"doc_id": "__test__"}],
            )
            self._store._collection.delete(ids=[test_id])
            return True, f"正常，当前 {self.count()} 个片段"
        except Exception as e:
            return False, f"异常：{type(e).__name__}: {e}"


# ==================================================================
# 单例
# ==================================================================
_store: VectorStoreService | None = None


def get_vector_store() -> VectorStoreService:
    """获取全局向量库服务（单例）"""
    global _store
    if _store is None:
        _store = VectorStoreService()
    return _store
