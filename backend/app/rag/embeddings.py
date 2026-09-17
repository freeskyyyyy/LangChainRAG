"""
向量化模块 —— 把文字变成"语义坐标"

================================================================
【先理解"向量"到底是什么】
================================================================
计算机没法直接比较两句话"意思像不像"。所以要把文字变成一串数字：

    "iPhone 16 的电池容量"  →  [0.023, -0.871, 0.442, ..., 0.109]
                                  └────── 1024 个数字 ──────┘

这串数字就是这句话在"语义空间"里的坐标，好比给每段文字发了一个 GPS 定位。

关键特性：**意思相近的文字，坐标也相近**。

    "电池耐用吗"   → 坐标 A
    "续航怎么样"   → 坐标 B     ← A 和 B 非常接近（虽然用词完全不同）
    "屏幕多大"     → 坐标 C     ← C 离 A、B 很远

所以检索时不用匹配关键词，而是比"坐标距离"——这就是**语义检索**。
"电池耐用"能搜出"续航长"，这是关键词搜索做不到的。

================================================================
【本模块要解决的三个工程问题】
================================================================
1. **慢**：每次调用云端 API 约 200ms。一份 100 页的文档切成 500 个片段，
   就是 100 秒。→ 解决：**批量提交**（一次发 10 个，而不是一个一个发）

2. **重复浪费**：同一段文字被反复向量化（重新索引时、多文档有重复内容时）
   既费钱又费时间。→ 解决：**内容哈希缓存**（同样的文字直接返回上次的结果）

3. **限流**：阿里云对接口有 QPS 限制，发太快会被拒绝。
   → 解决：**限速 + 失败重试（指数退避）**

这三个都是"工程细节"，但正是它们把一个能跑的 Demo 和一个能用的系统区分开。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Sequence

from app.core.cache import get_cache
from app.core.config import settings
from app.core.logging import logger

# ------------------------------------------------------------------
# 批量提交的参数
#
# 【为什么是 10】
# 阿里云百炼对向量化接口有单次条数限制（通常 10 条）。
# 超了会直接报错。保守起见设 10。
#
# 【为什么要有重试】
# 网络抖动、服务端偶发 500、瞬间触发限流……这些都是一次性的，
# 重试一下就好。但如果重试太猛（比如立刻重试 100 次），
# 反而会加重对方负担、更容易被限流。
# 所以用"指数退避"：等 1 秒、2 秒、4 秒……每次都等更久。
# ------------------------------------------------------------------
EMBED_BATCH_SIZE = 10
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0
REQUEST_INTERVAL = 0.1   # 每次批量请求之间至少间隔 0.1 秒，避免触发 QPS 限制


class EmbeddingService:
    """
    向量化服务

    用法：
        svc = EmbeddingService()
        vector = svc.embed_query("iPhone 16 电池多大")     # 单条
        vectors = svc.embed_documents(["文本1", "文本2"])   # 批量（推荐）
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from langchain_community.embeddings import DashScopeEmbeddings

        self.model = model or settings.EMBEDDING_MODEL
        self.api_key = api_key or settings.DASHSCOPE_API_KEY

        if not self.api_key:
            raise ValueError("未配置 DASHSCOPE_API_KEY，无法调用向量化接口")

        self._client = DashScopeEmbeddings(
            model=self.model,
            dashscope_api_key=self.api_key,
        )

        # 缓存：{内容哈希: 向量}
        # 复用全局缓存实例（将来切 Redis 时这里自动跟着切）
        self._cache = get_cache()
        self._cache_ttl = 7 * 24 * 3600   # 向量不会过期（同一模型下结果永远一样），给 7 天

        # 统计（用于观察缓存效果）
        self.stats = {"api_calls": 0, "cache_hits": 0, "texts_embedded": 0, "retries": 0}

        logger.info(f"向量化服务已初始化 | 模型={self.model} | 批量大小={EMBED_BATCH_SIZE}")

    # ==============================================================
    # 缓存键
    # ==============================================================
    def _cache_key(self, text: str) -> str:
        """
        生成缓存键。

        【为什么用文本的哈希而不是文本本身】
        文本可能很长（一个片段 500 字），用它当键既占内存又慢。
        哈希值固定 64 字符，查找快，且不同文本几乎不可能撞哈希
        （SHA-256 的碰撞概率低到可以忽略）。

        【为什么键里要带模型名】
        不同模型产生的向量维度、数值范围都不同，绝不能混用。
        带上模型名可以保证"换模型后不会错误地复用旧向量"。
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"emb:{self.model}:{digest}"

    # ==============================================================
    # 单条向量化（走缓存）
    # ==============================================================
    def embed_query(self, text: str) -> list[float]:
        """
        把一段文字变成向量（优先读缓存）。

        这个方法主要用于**用户提问**——每次问答都要把问题向量化，
        而热门问题会被反复问，所以缓存命中率会很高。
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("要向量化的文本不能为空")

        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached

        vector = self._embed_with_retry([text])[0]
        self._cache.set(key, vector, ttl=self._cache_ttl)
        return vector

    # ==============================================================
    # 批量向量化（入库主力）
    # ==============================================================
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """
        批量向量化（入库时的主力方法）。

        【流程】
          1. 先查缓存，把已经算过的挑出来
          2. 只把没算过的发给 API（分批提交）
          3. 结果写回缓存
          4. 按原始顺序拼装返回

        【为什么要"先查缓存再发请求"】
        重新索引一份文档时，绝大部分片段内容没变，全部命中缓存，
        API 调用量可能降低 90% 以上——既省钱又快，这就是缓存的价值。
        """
        texts = [t.strip() for t in texts]
        if not texts:
            return []

        # ---- 第一步：查缓存 ----
        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            if not text:
                # 空文本给个全零向量占位（正常流程不该出现，防御性处理）
                results[i] = [0.0] * 1024
                continue

            cached = self._cache.get(self._cache_key(text))
            if cached is not None:
                results[i] = cached
                self.stats["cache_hits"] += 1
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        hit_count = len(texts) - len(miss_texts)
        if hit_count:
            logger.debug(f"向量化缓存命中 {hit_count}/{len(texts)} 条")

        # ---- 第二步：只对未命中的部分调用 API ----
        if miss_texts:
            logger.info(
                f"需要向量化 {len(miss_texts)} 条文本"
                f"（缓存命中 {hit_count} 条，共 {len(texts)} 条）"
            )

            new_vectors: list[list[float]] = []
            total_batches = (len(miss_texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

            for batch_idx in range(total_batches):
                start = batch_idx * EMBED_BATCH_SIZE
                batch = miss_texts[start : start + EMBED_BATCH_SIZE]

                vectors = self._embed_with_retry(batch)
                new_vectors.extend(vectors)

                # 批次之间稍微歇一下，避免触发 QPS 限制
                if batch_idx < total_batches - 1:
                    time.sleep(REQUEST_INTERVAL)

                if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                    logger.info(f"向量化进度 {batch_idx + 1}/{total_batches} 批")

            # ---- 第三步：写回缓存并填充结果 ----
            for idx, text, vec in zip(miss_indices, miss_texts, new_vectors):
                results[idx] = vec
                self._cache.set(self._cache_key(text), vec, ttl=self._cache_ttl)

        self.stats["texts_embedded"] += len(texts)

        # 理论上不会有 None 了，兜个底防止类型检查报错
        return [r if r is not None else [0.0] * 1024 for r in results]

    # ==============================================================
    # 带重试的 API 调用
    # ==============================================================
    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """
        调用向量化接口，失败自动重试。

        【指数退避（Exponential Backoff）】
        第 1 次失败 → 等 1 秒重试
        第 2 次失败 → 等 2 秒重试
        第 3 次失败 → 等 4 秒重试
        还失败 → 放弃并抛异常

        为什么等待时间要翻倍？
        因为失败往往是因为"对方忙"。如果所有客户端都立刻重试，
        对方会更忙，形成"重试风暴"。
        等待时间翻倍能让系统有机会喘口气，也让重试真正有意义。
        这是分布式系统的标准做法。
        """
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                self.stats["api_calls"] += 1
                return self._client.embed_documents(texts)

            except Exception as e:
                last_error = e
                self.stats["retries"] += 1

                if attempt < MAX_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"向量化失败（第 {attempt + 1} 次），{delay:.0f} 秒后重试 | "
                        f"{type(e).__name__}: {str(e)[:100]}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"向量化失败（已重试 {MAX_RETRIES} 次，放弃）| "
                        f"{type(e).__name__}: {str(e)[:200]}"
                    )

        raise RuntimeError(
            f"向量化接口调用失败（已重试 {MAX_RETRIES} 次）：{last_error}"
        ) from last_error

    # ==============================================================
    # 异步版本（供 FastAPI 异步环境使用）
    # ==============================================================
    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """
        异步批量向量化。

        【为什么要包一层 to_thread】
        上面的 embed_documents 是同步的，里面有 time.sleep 和网络等待。
        如果在异步环境里直接调用，会**阻塞整个事件循环**——
        也就是说，一个人在上传文档时，其他所有人的请求都得卡着等。

        用 asyncio.to_thread 把它丢到线程池里执行，
        事件循环就能腾出手去处理别人的请求。
        """
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """异步单条向量化"""
        return await asyncio.to_thread(self.embed_query, text)

    # ==============================================================
    # 工具方法
    # ==============================================================
    def get_stats(self) -> dict[str, Any]:
        """返回统计信息（用于观察缓存效果和成本）"""
        total = self.stats["api_calls"]
        return {
            **self.stats,
            "cache_hit_ratio": (
                f"{self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['api_calls']):.1%}"
            ),
            "model": self.model,
        }

    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        """
        计算两个向量的余弦相似度（-1 到 1，越接近 1 越相似）。

        【什么时候用】
        语义缓存要用它：把新问题和缓存里存的问题向量比一下，
        如果相似度 > 0.95，就认为"是同一个问题"，直接返回缓存的答案。

        【什么是余弦相似度】
        不看向量的长度，只看它们"指向的方向"是否一致。
        好比两个人手指指向天空的同一个位置——距离可能不同，
        但"方向"一致，就说明他们在说同一件事。
        用方向而不是距离，能避免"长句子 vs 短句子"因为长度差异被误判为不相似。
        """
        import math

        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


# ==================================================================
# 单例
# ==================================================================
_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """获取全局向量化服务（单例，避免重复创建客户端）"""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
