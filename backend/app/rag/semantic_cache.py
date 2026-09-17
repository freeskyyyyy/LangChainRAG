"""
语义缓存 —— 意思相近的问题直接返回缓存答案

================================================================
【和普通缓存有什么区别】
================================================================
普通缓存（key-value）要求**问题一字不差**才能命中：

    缓存里存了："你们支持退货吗"
    用户问："支持退货吗"        → ❌ 没命中（少了"你们"两个字）
    用户问："能退货不"          → ❌ 没命中（完全不同的写法）
    用户问："你们支持退货吗"     → ✅ 命中

但真实的用户不会用一模一样的措辞提问。所以普通缓存的命中率很低。

语义缓存是**按"意思"判断**：

    缓存里存了："你们支持退货吗"（向量坐标 A）
    用户问："能退货吗"          → 算出的坐标离 A 很近 → ✅ 命中！
    用户问："退货怎么操作"       → 坐标离 A 稍远，但仍在阈值内 → ✅ 命中

关键指标：**余弦相似度 > 0.95 才算命中**。
这个阈值定得很高是故意的——宁可少命中（多算一次），
也不能错误命中（把"退货政策"的答案给了问"换货政策"的人）。

================================================================
【能省多少】
================================================================
省钱：命中时**完全不调用任何 API**：
  · 省一次向量化（小幅）
  · 省一次检索 + 重排序（中幅）
  · 省一次大模型生成（**这是大头**，通常是总成本的 80% 以上）

提速：从 2~5 秒降到 50 毫秒以内。

在实际使用中，客服类问答的重复率通常很高
（大家都问那几个问题：能退货吗、多久到货、怎么付款），
所以语义缓存的命中率往往相当可观。

================================================================
【本实现的取舍】
================================================================
用**线性扫描**比对所有缓存项。为什么不建向量索引？
  · 缓存条目通常只有几百条（有上限控制），线性扫描也就几毫秒
  · 建索引的复杂度远高于收益，属于"过早优化"

但有个必须做的处理：**给缓存设上限**。
否则用户一直问新问题，缓存会无限膨胀，最后撑爆内存。
上限到了就淘汰最久没用过的（LRU 策略）。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional

from app.core.config import settings
from app.core.logging import logger
from app.rag.embeddings import get_embedding_service


class SemanticCache:
    """
    语义缓存

    用法：
        cache = SemanticCache()
        hit = cache.lookup("能退货吗")          # 返回缓存的答案，或 None
        if hit is None:
            ...  # 走完整流程
            cache.store("能退货吗", answer, citations)
    """

    def __init__(
        self,
        threshold: float | None = None,
        max_size: int = 300,
        ttl: int | None = None,
    ):
        """
        参数：
            threshold —— 相似度阈值，超过才算命中。默认 0.95。
                         调低会增加命中率但可能返回不准确的答案；
                         调高会更保守。0.95 是较安全的取值。
            max_size  —— 最多缓存多少条。超过就淘汰最旧的。
            ttl       —— 缓存有效期（秒）。默认 1 小时。
                         为什么要过期？因为知识库可能更新了，
                         旧答案可能已经不对了。
        """
        self.threshold = threshold if threshold is not None else settings.SEMANTIC_CACHE_THRESHOLD
        self.max_size = max_size
        self.ttl = ttl or settings.SEMANTIC_CACHE_TTL

        # OrderedDict 而非普通 dict：它**记录插入顺序**，
        # 这样淘汰时能快速找到"最旧的"那一条（把它移到末尾即可）。
        # 这是实现 LRU 的标准做法。
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

        self._hits = 0
        self._misses = 0

        logger.info(
            f"语义缓存已初始化 | 阈值={self.threshold} | "
            f"容量={self.max_size} | 有效期={self.ttl}秒"
        )

    # ==============================================================
    # 查询
    # ==============================================================
    def lookup(self, question: str) -> Optional[dict[str, Any]]:
        """
        查找是否有语义相近的缓存。

        返回：命中的缓存数据（含 answer、citations、原始问题），或 None
        """
        with self._lock:
            if not self._entries:
                self._misses += 1
                return None

            try:
                embedder = get_embedding_service()
                query_vec = embedder.embed_query(question)
            except Exception as e:
                # 向量化失败时不能让问答挂掉，直接当作"没命中"
                logger.warning(f"语义缓存查询失败（视为未命中）: {e}")
                self._misses += 1
                return None

            now = time.time()
            best_score = -1.0
            best_key: Optional[str] = None

            for key, entry in self._entries.items():
                # 过期的直接跳过（顺便标记待清理）
                if now - entry["created_at"] > self.ttl:
                    continue

                score = embedder.cosine_similarity(query_vec, entry["vector"])
                if score > best_score:
                    best_score = score
                    best_key = key

            if best_key is not None and best_score >= self.threshold:
                entry = self._entries[best_key]
                # 命中后把它移到末尾，表示"最近刚用过"，
                # 这样它在 LRU 淘汰时优先级最低
                self._entries.move_to_end(best_key)
                self._hits += 1

                logger.info(
                    f"语义缓存命中 | 相似度={best_score:.4f} | "
                    f"原问题「{entry['question'][:30]}」← 新问题「{question[:30]}」"
                )

                return {
                    **entry["data"],
                    "cache_similarity": round(best_score, 4),
                    "cache_matched_question": entry["question"],
                }

            self._misses += 1
            logger.debug(f"语义缓存未命中 | 最高相似度={best_score:.4f} < 阈值{self.threshold}")
            return None

    # ==============================================================
    # 存入
    # ==============================================================
    def store(
        self,
        question: str,
        answer: str,
        citations: Optional[list[dict[str, Any]]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        存入一条缓存。

        【什么情况下不该存】
          · 答案是"没找到" —— 这种答案是临时的（可能知识库补了资料就能答了），
            缓存它会让后续同样的问题一直答不上来。**这是个重要的细节。**
          · 答案为空
        """
        if not question or not answer:
            return

        # 不缓存"没找到"的答案（原因见上面的注释）
        from app.rag.prompt import NO_CONTEXT_ANSWER

        if NO_CONTEXT_ANSWER in answer or "知识库中没有找到" in answer:
            logger.debug(f"跳过缓存（答案是『未找到』类型）| {question[:30]!r}")
            return

        try:
            embedder = get_embedding_service()
            vector = embedder.embed_query(question)
        except Exception as e:
            logger.warning(f"语义缓存写入失败（跳过）: {e}")
            return

        with self._lock:
            # 同一条问题重复存入时，先删旧的（保证更新）
            key = f"{hash(question)}"
            self._entries.pop(key, None)

            self._entries[key] = {
                "question": question,
                "vector": vector,
                "data": {
                    "answer": answer,
                    "citations": citations or [],
                    **(extra or {}),
                },
                "created_at": time.time(),
            }

            # 超出容量 → 淘汰最旧的一条
            # popitem(last=False) 移除的是"最早插入"的那条
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)

            self._entries.move_to_end(key)  # 新存入的算"最近使用"

    # ==============================================================
    # 维护
    # ==============================================================
    def clear(self) -> None:
        """清空缓存（知识库更新后应该调用，避免返回过时答案）"""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            logger.info(f"语义缓存已清空（原有 {count} 条）")

    def cleanup_expired(self) -> int:
        """
        清理过期条目。

        为什么要主动清理？因为 lookup 只在"遍历时跳过"过期项，
        并不真的删除它们。如果一直没人查询，过期的数据会一直占内存。
        定期调用一次可以真正释放空间。
        """
        now = time.time()
        with self._lock:
            expired = [
                k for k, e in self._entries.items()
                if now - e["created_at"] > self.ttl
            ]
            for k in expired:
                self._entries.pop(k, None)

        if expired:
            logger.debug(f"语义缓存清理了 {len(expired)} 条过期记录")
        return len(expired)

    def get_stats(self) -> dict[str, Any]:
        """统计信息（能直观看出缓存省了多少次 API 调用）"""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._entries),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{(self._hits / total * 100) if total else 0:.1f}%",
                "threshold": self.threshold,
                "ttl_seconds": self.ttl,
            }


# ==================================================================
# 单例
# ==================================================================
_cache: Optional[SemanticCache] = None
_cache_lock = threading.Lock()


def get_semantic_cache() -> SemanticCache:
    """获取全局语义缓存（单例）"""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = SemanticCache()
    return _cache


def invalidate_semantic_cache() -> None:
    """
    让语义缓存失效（知识库更新后调用）。

    为什么必须调用：如果管理员上传了新资料、修正了错误信息，
    但缓存里还是旧答案，用户就会一直拿到过时的信息。
    """
    get_semantic_cache().clear()
