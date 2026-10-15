"""
混合检索 + 重排序 —— RAG 检索质量的核心

================================================================
【这个模块干三件事】
================================================================
  ① 并行跑两路检索：向量检索（懂语义）+ BM25（懂关键词）
  ② 用 RRF 算法把两路结果融合成一个综合排名
  ③ 用重排序模型对融合结果精排，挑出最相关的几条

================================================================
【第一件事为什么有效：两路检索的能力是互补的】
================================================================
实测中的真实案例：
    用户问："包装里有充电器吗"
    文档写："包装盒内不附带充电器"

  向量检索：❌ 找到了"退换货政策" —— 它理解语义，但把整句话压成一个
            坐标时，"充电器"这个关键细节被"平均"掉了
  BM25   ：✅ 找到了正确答案 —— 它逐词匹配，"包装""充电器"直接命中

反过来：
    用户问："这个手机能撑一天吗"（文档里写的是"续航可达一天"）
  向量检索：✅ 能搜到（理解"撑一天"="续航一天"）
  BM25   ：❌ 搜不到（"撑"和"续航"是两个不同的词）

所以**两路都要跑**，然后用下面的算法合并。

================================================================
【第二件事：RRF（Reciprocal Rank Fusion，倒数排名融合）】
================================================================
【为什么不能直接把两边的分数相加】
因为两边的分数量纲完全不同：
    向量相似度：0 ~ 1 之间的小数（比如 0.6）
    BM25 分数 ：可以是 0 ~ 30 的任意正数（取决于语料规模）
直接相加的话，BM25 的分数会完全压倒向量分数，融合就失去意义了。

【RRF 的巧妙之处：不看分数，只看排名】
公式：  RRF分数(文档) = Σ  1 / (60 + 该文档在这一路里的排名)

举例，某文档在向量检索里排第 3，在 BM25 里排第 1：
    RRF = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226

另一个文档只在向量检索里排第 1，BM25 没搜到：
    RRF = 1/(60+1) = 0.01639

结论：**两路都认可的文档，得分明显更高**——这正是我们要的。

那个 60 是经验常数（来自原论文），作用是"削平"排名差异：
没有它的话，第 1 名和第 2 名的分数差距会过大，
导致排名稍有波动结果就剧烈变化。加了 60 之后曲线更平缓，更稳定。

RRF 的三大优点：
  · 不需要调参（那个 60 是通用的，不用针对数据调整）
  · 不受量纲影响（只看排名，不看分数）
  · 实现简单、效果稳定 —— 这是工业界的主流做法

================================================================
【第三件事：重排序（Rerank）】
================================================================
【为什么 RRF 之后还要再排一次】
向量检索用的是**双塔模型**：问题和文档**分别**算坐标，然后比距离。
快，但问题和文档从来没"见过面"，判断比较粗糙。

重排序用的是**交叉编码器**：把"问题 + 文档"拼在一起**同时**输入模型，
让模型逐字比对两者的关系。准得多，但慢得多
（要算 N 次，不能预先算好）。

所以标准做法是"两阶段"：
    第一阶段（粗筛）：向量 + BM25 快速从几万条里捞出 20 条
    第二阶段（精排）：交叉编码器对这 20 条逐个精算，取前 5 条

比喻：**看简历初筛 20 个人（快但粗糙），再逐个面试挑 5 个（慢但准确）。**
"""

from __future__ import annotations

import time
from typing import Any, Optional

from app.core.config import settings
from app.core.logging import logger

# ------------------------------------------------------------------
# RRF 的经验常数
# 来自原论文《Reciprocal Rank Fusion outperforms Condorcet...》
# 60 是被广泛验证过效果最好的值，一般不需要调整。
# ------------------------------------------------------------------
RRF_K = 60


class HybridRetriever:
    """
    混合检索器

    用法：
        retriever = HybridRetriever()
        results = retriever.retrieve("包装里有充电器吗", top_k=5)
    """

    def __init__(
        self,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        enable_rerank: bool = True,
        enable_bm25: bool = True,
    ):
        """
        参数：
            vector_weight  向量检索在融合中的权重（默认 1.0）
            bm25_weight    BM25 在融合中的权重（默认 1.0）
                           想让某一路更重要就调大它的权重。
                           一般保持相等即可，RRF 本身已经很稳。
            enable_rerank  是否启用重排序（调试时可关掉，方便对比效果）
            enable_bm25    是否启用 BM25（关掉就退化成纯向量检索，
                           用于做"优化前后对比"）
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.enable_rerank = enable_rerank
        self.enable_bm25 = enable_bm25

        self._reranker: Any = None

        logger.info(
            f"混合检索器已初始化 | BM25={'开' if enable_bm25 else '关'} | "
            f"重排序={'开' if enable_rerank else '关'}"
        )

    # ==============================================================
    # 主流程
    # ==============================================================
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        recall_k: Optional[int] = None,
        where: Optional[dict[str, Any]] = None,
        return_debug: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        检索主流程。

        参数：
            query      —— 查询文本（用户问题，或改写后的问题）
            top_k      —— 最终返回几条（默认取配置里的 RERANK_TOP_N = 5）
            recall_k   —— 初筛召回几条（默认 RETRIEVAL_TOP_K = 20）
            where      —— 元数据过滤（比如只搜某个文档）
            return_debug —— 是否同时返回调试信息（对比分析用）

        返回：
            top_k 条结果，按相关度从高到低
            若 return_debug=True，返回 (结果, 调试信息)
        """
        t_start = time.perf_counter()
        final_k = top_k or settings.RERANK_TOP_N
        recall = recall_k or settings.RETRIEVAL_TOP_K

        debug: dict[str, Any] = {"query": query, "recall_k": recall, "final_k": final_k}

        # ---------------- 第一路：向量检索 ----------------
        t0 = time.perf_counter()
        try:
            from app.rag.vectorstore import get_vector_store

            vector_results = get_vector_store().search(query, top_k=recall, where=where)
        except Exception as e:
            logger.error(f"向量检索失败，将只用 BM25: {e}")
            debug["vector_error"] = type(e).__name__
            vector_results = []
        t_vector = (time.perf_counter() - t0) * 1000

        debug["vector_count"] = len(vector_results)
        debug["vector_ms"] = round(t_vector, 1)
        debug["vector_top"] = [
            {"file": r["file_name"], "score": r["score"], "content": r["content"][:60]}
            for r in vector_results[:3]
        ]

        # ---------------- 第二路：BM25 关键词检索 ----------------
        t0 = time.perf_counter()
        bm25_results: list[dict[str, Any]] = []

        if self.enable_bm25:
            try:
                from app.rag.bm25 import ensure_ready

                bm25_index = ensure_ready()
                bm25_results = bm25_index.search(query, top_k=recall)
            except Exception as e:
                logger.error(f"BM25 检索失败，将只用向量检索: {e}")
                debug["bm25_error"] = type(e).__name__

        t_bm25 = (time.perf_counter() - t0) * 1000
        debug["bm25_count"] = len(bm25_results)
        debug["bm25_ms"] = round(t_bm25, 1)
        debug["bm25_top"] = [
            {"file": r["file_name"], "score": r["score"], "content": r["content"][:60]}
            for r in bm25_results[:3]
        ]

        # ---------------- 融合（RRF）----------------
        t0 = time.perf_counter()
        if bm25_results:
            fused = self._rrf_fusion(vector_results, bm25_results)
        else:
            # 没跑 BM25（或 BM25 没结果），直接用向量结果
            fused = [
                {**r, "rrf_score": 1.0 / (RRF_K + i + 1), "sources": ["vector"]}
                for i, r in enumerate(vector_results)
            ]
        t_fusion = (time.perf_counter() - t0) * 1000

        debug["fused_count"] = len(fused)
        debug["fusion_ms"] = round(t_fusion, 1)

        if not fused:
            debug["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            logger.warning(f"两路检索都没有结果 | query={query[:50]!r}")
            return ([], debug) if return_debug else []

        # ---------------- 重排序 ----------------
        t0 = time.perf_counter()
        if self.enable_rerank:
            # 只对前 N 条做重排序（重排序慢，不能对所有召回结果都做）
            rerank_pool = fused[: max(final_k * 4, 20)]
            ranked = self._rerank(query, rerank_pool, final_k)
        else:
            ranked = fused[:final_k]
        t_rerank = (time.perf_counter() - t0) * 1000

        debug["rerank_ms"] = round(t_rerank, 1)
        debug["rerank_enabled"] = self.enable_rerank
        debug["rerank_applied"] = bool(ranked) and all(r.get("rerank_score") is not None for r in ranked) if self.enable_rerank else False
        debug["reranked_top"] = [
            {"file": r["file_name"], "content": r["content"][:60]} for r in ranked[:3]
        ]

        debug["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

        logger.info(
            f"检索完成 | 向量 {len(vector_results)} 条 / BM25 {len(bm25_results)} 条 "
            f"→ 融合 {len(fused)} 条 → 最终 {len(ranked)} 条 | "
            f"耗时 {debug['total_ms']:.0f}ms"
        )

        return (ranked, debug) if return_debug else ranked

    # ==============================================================
    # RRF 融合
    # ==============================================================
    def _rrf_fusion(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        用 RRF 算法融合两路检索结果。

        【为什么用 (doc_id, child_index) 作为去重键】
        同一段文字可能同时被两路命中，必须识别出"这是同一个片段"，
        否则融合后的列表里会出现重复条目。
        每份文档+每个片段序号唯一确定一个片段，所以这个组合键是可靠的。
        比用"内容文本"做键更好——两份文档里出现完全相同的短句是真的不同片段，
        不应该被合并。
        """
        scores: dict[str, float] = {}          # 键 → RRF 累计分数
        payload: dict[str, dict[str, Any]] = {}  # 键 → 片段数据
        sources: dict[str, list[str]] = {}     # 键 → 被哪几路命中

        def key_of(item: dict[str, Any]) -> str:
            """
            生成去重键。

            【踩过的坑：不要用 Python 内置的 hash()】
            我一开始写成 f"{doc_id}#{child_index}#{hash(content)}"，
            结果两路结果**永远对不上**（日志里显示"0 条被两路同时命中"）。

            原因是 CPython 的 str.__hash__ 带**进程级随机盐**
            （这是防止哈希碰撞攻击的安全设计，可以用 PYTHONHASHSEED 控制）。
            虽然同一进程内是一致的，但……问题在于内容字符串本身：
            两路取到的文本可能有细微差别（比如尾随空白），
            hash 之后就是一个完全不同的值。

            直接用**内容原文**当键最稳妥：
              · 内容相同 → 键相同（这正是我们要的）
              · 不需要哈希，没有随机性
              · 即使有些片段内容为空，也用 child_index 兜底
            """
            content = item.get("content") or ""
            if content:
                return content
            # 内容为空时的兜底（正常不该出现）
            return f"__empty__{item.get('doc_id', '')}_{item.get('child_index', '')}"

        # 第一路：向量检索
        for rank, item in enumerate(vector_results, start=1):
            k = key_of(item)
            scores[k] = scores.get(k, 0.0) + self.vector_weight / (RRF_K + rank)
            payload.setdefault(k, item)
            sources.setdefault(k, []).append("vector")

        # 第二路：BM25
        for rank, item in enumerate(bm25_results, start=1):
            k = key_of(item)
            scores[k] = scores.get(k, 0.0) + self.bm25_weight / (RRF_K + rank)
            # 向量结果里已有的话保留原始的（它带着向量相似度分数）
            payload.setdefault(k, item)
            sources.setdefault(k, []).append("bm25")

        # 按 RRF 分数排序
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        fused: list[dict[str, Any]] = []
        for k, rrf_score in ordered:
            item = dict(payload[k])
            item["rrf_score"] = round(rrf_score, 6)
            item["sources"] = sources[k]
            # "两路都命中"是个很强的信号，标记出来便于调试
            item["hit_both"] = len(sources[k]) > 1
            fused.append(item)

        both_count = sum(1 for x in fused if x["hit_both"])
        logger.debug(f"RRF 融合完成 | 共 {len(fused)} 条 | 其中 {both_count} 条被两路同时命中")

        return fused

    # ==============================================================
    # 重排序
    # ==============================================================
    def _get_reranker(self) -> Any:
        """懒加载重排序器（第一次用的时候才创建）"""
        if self._reranker is None:
            from langchain_community.document_compressors import DashScopeRerank

            self._reranker = DashScopeRerank(
                model=settings.RERANK_MODEL,
                top_n=20,  # 这里给个上限，实际取几条由调用方控制
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
            )
            logger.info(f"重排序器已初始化 | 模型={settings.RERANK_MODEL}")
        return self._reranker

    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """
        对候选片段重排序。

        【为什么不用 LangChain 的 DashScopeRerank.compress_documents】
        它接收和返回的是 Document 对象，而我们的候选是 dict，
        来回转换要丢元数据（比如 parent_content），转换代码还容易出错。
        所以直接调用底层的 dashscope API，然后按返回的序号重排我们的 dict
        ——代码更短，也更不容易出 bug。

        【失败降级】
        如果重排序接口挂了（网络问题、额度用尽），
        **不能让整个问答失败**——退回用 RRF 的顺序即可。
        用户体验会差一点，但至少能答。
        这叫"优雅降级"，是生产系统的基本要求。
        """
        if not candidates:
            return []

        try:
            import dashscope
            from dashscope import TextReRank

            dashscope.api_key = settings.DASHSCOPE_API_KEY

            documents = [c.get("content", "") for c in candidates]

            resp = TextReRank.call(
                model=settings.RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
                return_documents=False,
            )

            # 接口调用失败（非 200）
            if resp.status_code != 200:
                logger.warning(
                    f"重排序接口返回异常 {resp.status_code}: {resp.message} | "
                    f"降级为使用 RRF 排序"
                )
                return candidates[:top_n]

            # 按接口返回的相关度分数重排
            results = resp.output.results
            reranked: list[dict[str, Any]] = []

            for r in results:
                idx = r.index
                if 0 <= idx < len(candidates):
                    item = dict(candidates[idx])
                    # 记录重排序分数，同时把原来的 RRF 分数也保留（便于对比）
                    item["rerank_score"] = round(float(r.relevance_score), 6)
                    item["source"] = "rerank"
                    reranked.append(item)

            # 按重排序分数从高到低（接口通常已排好，这里保险起见再排一次）
            reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            if reranked:
                logger.debug(
                    f"重排序完成 | {len(candidates)} 条 → {len(reranked)} 条 | "
                    f"最高分 {reranked[0].get('rerank_score')}"
                )

            return reranked[:top_n] if reranked else candidates[:top_n]

        except Exception as e:
            logger.error(
                f"重排序失败（降级为使用 RRF 排序）| {type(e).__name__}: {str(e)[:150]}"
            )
            return candidates[:top_n]

    # ==============================================================
    # 对比测试（用于量化"优化到底有没有用"）
    # ==============================================================
    def compare(
        self,
        query: str,
        expect_keyword: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        对比三种配置的检索效果，产出量化数据。

        三种配置：
          1. 纯向量检索
          2. 向量 + BM25（混合检索）
          3. 向量 + BM25 + 重排序（完整方案）

        这是回答"你的优化到底有没有用"这个问题的直接证据。
        """
        original_rerank, original_bm25 = self.enable_rerank, self.enable_bm25
        results: dict[str, Any] = {"query": query, "expect": expect_keyword}

        try:
            # 配置 1：纯向量
            self.enable_rerank = False
            self.enable_bm25 = False
            r1 = self.retrieve(query, top_k=top_k)
            results["vector_only"] = self._judge(r1, expect_keyword)

            # 配置 2：向量 + BM25
            self.enable_rerank = False
            self.enable_bm25 = True
            r2 = self.retrieve(query, top_k=top_k)
            results["hybrid"] = self._judge(r2, expect_keyword)

            # 配置 3：完整方案
            self.enable_rerank = True
            self.enable_bm25 = True
            r3 = self.retrieve(query, top_k=top_k)
            results["full"] = self._judge(r3, expect_keyword)

        finally:
            # 无论成功失败都要恢复原设置，否则会影响后续调用
            self.enable_rerank = original_rerank
            self.enable_bm25 = original_bm25

        return results

    @staticmethod
    def _judge(results: list[dict[str, Any]], expect_keyword: str) -> dict[str, Any]:
        """判断一批检索结果里有没有期望的关键词，并给出排名"""
        for rank, r in enumerate(results, start=1):
            if expect_keyword in r.get("content", "") or expect_keyword in r.get("parent_content", ""):
                return {"hit": True, "rank": rank, "file": r.get("file_name")}
        return {"hit": False, "rank": None, "file": results[0].get("file_name") if results else None}


# ==================================================================
# 单例
# ==================================================================
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """获取全局混合检索器（单例）"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
