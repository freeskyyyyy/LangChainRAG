"""
RAG 主链 —— 把所有零件串成一条完整的问答流水线

================================================================
【完整流程】
================================================================
用户提问
   ↓
① 查语义缓存 ──命中──→ 直接返回（不调用任何 API，50ms 内）
   ↓ 未命中
② 多轮问题改写（把"它多少钱"补全成"iPhone 16 多少钱"）
   ↓
③ 混合检索（向量 + BM25）→ RRF 融合
   ↓
④ 重排序（精排取 Top5）
   ↓
⑤ 相关性阈值检查 ──分数太低──→ 使用通用知识提示词生成回答
   ↓ 通过
⑥ 组装提示词（资料编号 + 历史对话 + 问题）
   ↓
⑦ 流式生成（SSE 打字机）
   ↓
⑧ 附上引用溯源 + 写入语义缓存

================================================================
【为什么用"流式"而不是"生成完再返回"】
================================================================
不流式：用户点发送 → 白屏转圈 4 秒 → 整段答案"啪"一下全出来
流式  ：用户点发送 → 0.3 秒后开始逐字蹦出

总耗时其实一样，但**体感差别巨大**——等待时能看到东西在动，
用户就不会觉得"卡死了"。这是体验优化的经典手法。

实现上，流式会带来一个麻烦：**生成完之后才能确定引用了哪些资料**。
所以引用是作为最后一个事件单独发送的（见 SSE 的 citations 事件）。
"""

from __future__ import annotations

import re
import time
from typing import Any, AsyncGenerator, Optional

from app.core.config import settings
from app.core.logging import logger
from app.rag.prompt import (
    build_context,
    build_general_prompt,
    build_rag_prompt,
    format_history,
)
from app.rag.retriever import HybridRetriever, get_retriever
from app.rag.rewriter import QueryRewriter, get_rewriter
from app.rag.semantic_cache import get_semantic_cache
from app.rag.citations import CitationFilter, validate_answer


class RAGChain:
    """
    RAG 问答主链

    用法：
        chain = RAGChain()
        # 非流式（测试用）
        result = chain.ask("电池多大")
        print(result["answer"], result["citations"])
        # 流式（接口用）
        async for event in chain.astream("电池多大"):
            print(event)
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        rewriter: Optional[QueryRewriter] = None,
        enable_cache: bool = True,
    ):
        self.retriever = retriever or get_retriever()
        self.rewriter = rewriter or get_rewriter()
        self.enable_cache = enable_cache

        self._llm: Any = None
        self._chain: Any = None

        logger.info(
            f"RAG 主链已初始化 | 模型={settings.LLM_MODEL} | "
            f"语义缓存={'开' if enable_cache else '关'}"
        )

    # ==============================================================
    # 懒加载模型
    # ==============================================================
    def _get_llm(self, streaming: bool = False):
        """
        获取大模型实例。

        【为什么 streaming 要分开创建】
        LangChain 里，同一个模型实例的 streaming 参数是固定的。
        流式和非流式需要两个不同实例。

        这里用字典做了简单的缓存，避免重复创建。
        """
        from langchain_community.chat_models import ChatTongyi

        cache_key = "streaming" if streaming else "normal"
        if not hasattr(self, "_llm_cache"):
            self._llm_cache: dict[str, Any] = {}

        if cache_key not in self._llm_cache:
            self._llm_cache[cache_key] = ChatTongyi(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
                streaming=streaming,
            )
            logger.debug(f"已创建模型实例 | streaming={streaming}")

        return self._llm_cache[cache_key]

    def _get_prompt_chain(self, streaming: bool = False, knowledge_source: str = "knowledge_base"):
        """构建 LCEL 链：提示词 → 模型"""
        from langchain_core.output_parsers import StrOutputParser

        llm = self._get_llm(streaming=streaming)
        prompt = build_general_prompt() if knowledge_source == "general" else build_rag_prompt()

        if streaming:
            # 流式链不用 StrOutputParser（它会缓冲全部内容才输出，
            # 那样就失去流式意义了）。直接返回模型的流。
            return prompt | llm
        return prompt | llm | StrOutputParser()

    # ==============================================================
    # 准备阶段（两条路径共用）
    # ==============================================================
    def _prepare(
        self,
        question: str,
        history: Optional[list[dict[str, str]]] = None,
        use_cache: bool = True,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        执行"生成回答之前"的所有步骤。

        抽出来的好处：流式和非流式两条路径共用同一套准备逻辑，
        不会出现"改了一处忘了另一处"的不一致问题。

        返回一个上下文对象，包含：
            cached        —— 如果命中缓存，这里就是答案；否则为 None
            search_query  —— 实际用于检索的问题（可能是改写后的）
            chunks        —— 检索到的片段
            prompt_input  —— 组装好的提示词变量
            timings       —— 各步骤耗时
        """
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        history = history or []

        # 模型身份属于系统配置，不需要检索或让模型猜测自己的身份。
        direct_answer = _model_identity_answer(question)
        if direct_answer is not None:
            return {
                "cached": None, "direct_answer": direct_answer,
                "knowledge_source": "system", "no_answer": False,
                "search_query": question, "chunks": [], "prompt_input": None,
                "rewrite": None, "top_score": 0.0,
                "cache_allowed": False, "timings": timings,
            }

        # 多轮对话的答案依赖历史，不使用仅以问题文本匹配的共享缓存。
        cache_allowed = self.enable_cache and use_cache and not history

        # ---------------- ① 语义缓存 ----------------
        if cache_allowed:
            t0 = time.perf_counter()
            cached = get_semantic_cache().lookup(question)
            timings["cache_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            if cached:
                timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
                return {
                    "cached": cached,
                    "search_query": question,
                    "chunks": [],
                    "prompt_input": None,
                    "timings": timings,
                    "rewrite": None,
                    "top_score": cached.get("cache_similarity", 1.0),
                }
        else:
            timings["cache_ms"] = 0.0

        # ---------------- ② 多轮问题改写 ----------------
        t0 = time.perf_counter()
        rewrite_result = self.rewriter.rewrite(question, history)
        search_query = rewrite_result.rewritten
        timings["rewrite_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # ---------------- ③④ 混合检索 + 重排序 ----------------
        t0 = time.perf_counter()
        try:
            chunks = self.retriever.retrieve(search_query, top_k=top_k)
        except Exception as e:
            logger.warning(f"资料检索不可用，切换为通用知识回答 | {type(e).__name__}")
            chunks = []
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # ---------------- ⑤ 相关性阈值检查 ----------------
        # 【这一步是 RAG 可靠性的关键】
        # 低相关片段不能硬塞给模型；清空后让模型切换到通用知识回答，
        # 并在提示词中要求它明确标注依据，避免把通用知识伪装成库内资料。
        #
        # ⚠️ 注意这里用 _get_threshold() 而不是直接读配置：
        #    重排序分数和向量相似度的量纲完全不同，必须各用各的阈值。
        #    （详见 config.py 里这两个阈值的注释说明）
        top_score = self._normalize_score(chunks[0]) if chunks else 0.0
        threshold = self._get_threshold(chunks[0]) if chunks else settings.RERANK_SCORE_THRESHOLD

        knowledge_source = "knowledge_base"
        if not chunks or top_score < threshold:
            logger.info(
                f"知识库没有足够相关的资料，改用模型通用知识回答 | "
                f"最高分={top_score:.4f} < 阈值{threshold} | "
                f"query={search_query[:50]!r}"
            )
            chunks = []
            knowledge_source = "general"

        # ---------------- ⑥ 组装提示词 ----------------
        t0 = time.perf_counter()
        context = (
            build_context(chunks, max_chunks=settings.MAX_CONTEXT_CHUNKS)
            if chunks
            else "知识库没有检索到与问题直接相关的资料。请基于通用知识回答，并明确标注依据。"
        )
        prompt_input = {
            "context": context, "question": search_query,
            "history": format_history(history),
        }
        timings["prompt_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "cached": None,
            "cache_allowed": cache_allowed,
            "no_answer": knowledge_source == "general",
            "knowledge_source": knowledge_source,
            "search_query": search_query,
            "chunks": chunks,
            "prompt_input": prompt_input,
            "timings": timings,
            "rewrite": rewrite_result,
            "top_score": top_score,
        }

    @staticmethod
    def _normalize_score(chunk: dict[str, Any]) -> float:
        """
        把不同来源的分数统一到 0~1 区间，用于和阈值比较。

        【为什么需要归一化】
        检索结果可能来自三个不同的打分体系：
          · 向量检索：余弦相似度，本来就在 0~1
          · BM25   ：原始分数，可能是 0~30 的任意正数
          · 重排序 ：相关度，通常在 0~1

        如果直接拿 BM25 的 12.5 去和阈值 0.3 比，永远都会通过，
        阈值检查就形同虚设。所以这里统一处理。
        """
        if "rerank_score" in chunk:
            return float(chunk["rerank_score"])
        if chunk.get("source") == "bm25":
            # BM25 分数没有天然上界，用一个经验公式压缩到 0~1
            raw = float(chunk.get("score", 0))
            return min(1.0, raw / (raw + 5.0)) if raw > 0 else 0.0
        return float(chunk.get("score", 0.0))

    @staticmethod
    def _get_threshold(chunk: dict[str, Any]) -> float:
        """
        根据分数的来源，选用对应的阈值。

        【为什么必须分开】
        实测数据（详见 config.py 的注释）表明两种分数的量纲完全不同：
            重排序分数：正确答案落到 0.167~0.320，无关内容 0.006~0.160
            向量相似度：正确答案落到 0.582~2.407，无关内容 0.265~1.360

        用同一个阈值去卡两种分数，必然有一种会被误判：
          · 用 0.3 卡重排序 → 正确答案 0.167 被当成不相关（实测踩过）
          · 用 0.15 卡向量分 → 无关内容 0.265 也被放行，兜底机制失效

        所以这里按来源分发。
        """
        if "rerank_score" in chunk:
            return settings.RERANK_SCORE_THRESHOLD
        if chunk.get("source") == "bm25":
            # BM25 经过上面的压缩公式后，量纲接近向量相似度
            return settings.SCORE_THRESHOLD
        return settings.SCORE_THRESHOLD

    # ==============================================================
    # 组装引用信息
    # ==============================================================
    @staticmethod
    def build_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        把检索到的片段转成"引用卡片"数据。

        前端拿到后会在回答下方渲染成可展开的卡片，
        显示"出自哪份文件、第几页、相关度多少、原文是什么"。

        【为什么要带上 content（子块原文）】
        用户点开引用卡片时，期望看到的是"模型参考的那段话"。
        但模型实际看到的是父块（更长）。如果展示父块，
        用户可能觉得"这么多内容，到底哪句被用了"。
        所以展示**子块**（那个精准命中的小段），更契合用户的预期。
        """
        citations: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks, start=1):
            context = (chunk.get("parent_content") or chunk.get("content") or "").strip()
            content = (chunk.get("content") or "").strip() or context
            if not context:
                continue

            score = chunk.get("rerank_score")
            if score is None:
                score = chunk.get("score") or 0.0

            citations.append(
                {
                    "index": i,
                    "file_name": chunk.get("file_name", "未知文档"),
                    "page": chunk.get("page"),
                    "sheet_name": chunk.get("sheet_name"),
                    "score": round(float(score), 4),
                    "content": content,
                    "document_id": _to_int_or_none(chunk.get("doc_id")),
                }
            )

        return citations

    @classmethod
    def _validate_answer(
        cls, answer: str, chunks: list[dict[str, Any]], knowledge_source: str
    ) -> tuple[str, list[dict[str, Any]], str]:
        """只接受已提供给模型且正文实际使用的资料编号。"""
        citations = cls.build_citations(chunks[:settings.MAX_CONTEXT_CHUNKS])
        return validate_answer(answer, citations, knowledge_source)

    # ==============================================================
    # 非流式问答（测试和调试用）
    # ==============================================================
    def ask(
        self,
        question: str,
        history: Optional[list[dict[str, str]]] = None,
        use_cache: bool = True,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """一次性返回完整答案（不流式）"""
        t_start = time.perf_counter()
        ctx = self._prepare(question, history, use_cache, top_k)
        timings = ctx["timings"]

        if ctx.get("direct_answer") is not None:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            return {
                "answer": ctx["direct_answer"], "citations": [], "cache_hit": False,
                "search_query": question, "rewritten": None, "chunks": [],
                "timings": timings, "top_score": 0.0,
                "knowledge_source": "system", "no_answer": False,
            }

        # ---- 命中缓存 ----
        if ctx.get("cached"):
            cached = ctx["cached"]
            answer, citations, source = validate_answer(
                cached["answer"], cached.get("citations", []), cached.get("knowledge_source", "knowledge_base")
            )
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            return {
                "answer": answer,
                "citations": citations,
                "cache_hit": True,
                "cache_similarity": cached.get("cache_similarity"),
                "matched_question": cached.get("cache_matched_question"),
                "search_query": question,
                "rewritten": None,
                "chunks": [],
                "timings": timings,
                "top_score": cached.get("cache_similarity", 1.0),
                "knowledge_source": source,
                "no_answer": source == "general",
            }

        # ---- 调模型生成 ----
        t0 = time.perf_counter()
        chain = self._get_prompt_chain(streaming=False, knowledge_source=ctx["knowledge_source"])
        answer = chain.invoke(ctx["prompt_input"])
        timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        answer, citations, knowledge_source = self._validate_answer(answer, ctx["chunks"], ctx["knowledge_source"])
        ctx["knowledge_source"] = knowledge_source
        ctx["no_answer"] = knowledge_source == "general"
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

        # ---- 写入语义缓存 ----
        if ctx["cache_allowed"]:
            try:
                get_semantic_cache().store(
                    question, answer, citations,
                    extra={"knowledge_source": ctx["knowledge_source"]},
                )
            except Exception as e:
                logger.warning(f"写入语义缓存失败（不影响回答）: {e}")

        return {
            "answer": answer,
            "citations": citations,
            "cache_hit": False,
            "search_query": ctx["search_query"],
            "rewritten": _rewrite_info(ctx.get("rewrite")),
            "chunks": ctx["chunks"],
            "timings": timings,
            "top_score": ctx["top_score"],
            "knowledge_source": ctx["knowledge_source"],
            "no_answer": ctx["no_answer"],
        }

    # ==============================================================
    # 流式问答（接口用）
    # ==============================================================
    async def astream(
        self,
        question: str,
        history: Optional[list[dict[str, str]]] = None,
        use_cache: bool = True,
        top_k: Optional[int] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        流式问答，逐个事件 yield 出来。

        事件类型：
            {"event": "status",    "data": {...}}  当前进行到哪一步
            {"event": "rewrite",   "data": {...}}  问题改写的详情
            {"event": "retrieval", "data": {...}}  检索结果概况
            {"event": "token",     "data": {"text": "..."}}  一个片段的内容
            {"event": "citations", "data": [...]}  引用列表
            {"event": "done",      "data": {...}}  结束
            {"event": "error",     "data": {...}}  出错
        """
        import asyncio

        t_start = time.perf_counter()

        try:
            # ---------------- 准备阶段（同步逻辑，放到线程池避免阻塞）----------------
            yield {"event": "status", "data": {"stage": "rewriting", "message": "正在理解你的问题…"}}

            ctx = await asyncio.to_thread(
                self._prepare, question, history, use_cache, top_k
            )

            if ctx.get("direct_answer") is not None:
                yield {"event": "status", "data": {"stage": "system_info", "message": "正在读取模型配置"}}
                for piece in _chunk_text(ctx["direct_answer"], 12):
                    yield {"event": "token", "data": {"text": piece}}
                yield {"event": "citations", "data": []}
                yield {
                    "event": "done",
                    "data": {
                        "cache_hit": False, "knowledge_source": "system", "no_answer": False,
                        "search_query": question, "top_score": 0.0,
                        "latency_ms": round((time.perf_counter() - t_start) * 1000),
                        "timings": {"total_ms": round((time.perf_counter() - t_start) * 1000, 1)},
                    },
                }
                return

            # ---- 命中缓存：直接返回 ----
            if ctx.get("cached"):
                cached = ctx["cached"]
                answer, citations, source = validate_answer(
                    cached["answer"], cached.get("citations", []), cached.get("knowledge_source", "knowledge_base")
                )
                yield {
                    "event": "status",
                    "data": {
                        "stage": "cache_hit",
                        "message": f"命中缓存（相似度 {cached.get('cache_similarity', 0):.2f}）",
                    },
                }
                # 虽然是缓存，也按"逐字"发送，保持体验一致
                for piece in _chunk_text(answer, 12):
                    yield {"event": "token", "data": {"text": piece}}
                    await asyncio.sleep(0.008)

                yield {"event": "citations", "data": citations}
                yield {
                    "event": "done",
                    "data": {
                        "cache_hit": True,
                        "cache_similarity": cached.get("cache_similarity"),
                        "knowledge_source": source,
                        "no_answer": source == "general",
                        "matched_question": cached.get("cache_matched_question"),
                        "latency_ms": round((time.perf_counter() - t_start) * 1000),
                        "timings": ctx["timings"],
                    },
                }
                return

            # ---- 改写信息 ----
            if ctx.get("rewrite") and ctx["rewrite"].changed:
                yield {
                    "event": "rewrite",
                    "data": {
                        "original": ctx["rewrite"].original,
                        "rewritten": ctx["rewrite"].rewritten,
                        "reason": ctx["rewrite"].reason,
                    },
                }

            # ---- 报告回答依据 ----
            chunks = ctx["chunks"]
            if ctx["knowledge_source"] == "general":
                yield {
                    "event": "status",
                    "data": {
                        "stage": "general_knowledge",
                        "message": "资料库没有找到直接相关内容，正在按通用知识回答",
                        "top_score": ctx["top_score"],
                    },
                }
            else:
                # ---- 检索成功：报告检索情况 ----
                yield {
                    "event": "retrieval",
                    "data": {
                        "count": len(chunks),
                        "top_score": ctx["top_score"],
                        "sources": [
                            {"file": c.get("file_name"), "page": c.get("page")}
                            for c in chunks[:5]
                        ],
                    },
                }
                yield {
                    "event": "status",
                    "data": {"stage": "generating", "message": "正在组织回答…"},
                }

            # ---------------- 流式生成 ----------------
            t_gen = time.perf_counter()
            full_answer: list[str] = []
            candidates = self.build_citations(chunks[:settings.MAX_CONTEXT_CHUNKS]) if ctx["knowledge_source"] == "knowledge_base" else []
            citation_filter = CitationFilter(c["index"] for c in candidates)

            chain = self._get_prompt_chain(streaming=True, knowledge_source=ctx["knowledge_source"])

            # astream 逐个产出模型输出的片段
            async for piece in chain.astream(ctx["prompt_input"]):
                # ------------------------------------------------------------------
                # ⚠️ 这里有个容易踩的坑：
                #
                # 模型在最后会发一个"内容为空、只带统计信息"的结束块
                # （里面有 token 用量、结束原因等）。
                #
                # 我一开始写成 `getattr(piece, "content", None) or str(piece)`，
                # 意思是"没有 content 属性就转成字符串"。
                # 但空字符串是**假值**，所以那个结束块也走了 str(piece) 分支，
                # 结果把整个对象的调试表示拼进了答案里：
                #
                #   "...[1]。content='' additional_kwargs={} response_metadata={...}"
                #                    ↑ 这堆垃圾就这样暴露给用户了
                #
                # 正确的判断是"有没有 content 属性"，而不是"content 是否为空"。
                # ------------------------------------------------------------------
                if hasattr(piece, "content"):
                    text = piece.content or ""
                else:
                    text = str(piece)

                if not text:
                    continue

                text = citation_filter.feed(text)
                if text:
                    full_answer.append(text)
                    yield {"event": "token", "data": {"text": text}}

            tail = citation_filter.feed("", final=True)
            if tail:
                full_answer.append(tail)
                yield {"event": "token", "data": {"text": tail}}

            answer = "".join(full_answer)
            timings = dict(ctx["timings"])
            timings["generation_ms"] = round((time.perf_counter() - t_gen) * 1000, 1)
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

            # ---------------- 引用溯源 ----------------
            citations = [c for c in candidates if c["index"] in citation_filter.used]
            knowledge_source = "knowledge_base" if citations else "general"
            ctx["knowledge_source"] = knowledge_source
            ctx["no_answer"] = knowledge_source == "general"
            yield {"event": "citations", "data": citations}

            # ---------------- 写入语义缓存 ----------------
            if ctx["cache_allowed"] and answer:
                try:
                    await asyncio.to_thread(
                        get_semantic_cache().store, question, answer, citations,
                        extra={"knowledge_source": ctx["knowledge_source"]}
                    )
                except Exception as e:
                    logger.warning(f"写入语义缓存失败: {e}")

            yield {
                "event": "done",
                "data": {
                    "cache_hit": False,
                    "latency_ms": round((time.perf_counter() - t_start) * 1000),
                    "search_query": ctx["search_query"],
                    "top_score": ctx["top_score"],
                    "knowledge_source": ctx["knowledge_source"],
                    "no_answer": ctx["no_answer"],
                    "timings": timings,
                },
            }

        except Exception as e:
            logger.exception(f"流式问答出错 | question={question[:50]!r}")
            yield {
                "event": "error",
                "data": {
                    "message": _generation_error_message(e),
                    "detail": f"{type(e).__name__}: {str(e)[:200]}",
                },
            }


# ==================================================================
# 辅助函数
# ==================================================================
def _model_identity_answer(question: str) -> Optional[str]:
    """只识别询问当前助手模型身份的完整句式。"""
    normalized = re.sub(r"[\s?？。.!！]", "", question)
    pattern = (
        r"(?:请问|请告诉我)?(?:你|您|这个系统|这个助手)(?:现在)?"
        r"(?:(?:用的|使用的|采用的|调用的|的|底层)?模型(?:名称|名字)?"
        r"(?:是什么|是哪个|是哪种|叫什么|叫什么名字)"
        r"|(?:是|用的|使用的|使用|采用|调用|基于)(?:什么|哪个|哪种)"
        r"(?:大语言|语言|大|AI)?模型)(?:呢|呀|啊|吗)?"
    )
    if re.fullmatch(pattern, normalized, flags=re.IGNORECASE):
        return f"当前配置的对话模型是 **{settings.LLM_MODEL}**，由阿里云百炼（DashScope）提供。"
    return None


def _generation_error_message(error: Exception) -> str:
    """网络连接失败时明确说明原因，避免误报为知识库没有答案。"""
    from requests.exceptions import ConnectionError, Timeout

    if isinstance(error, (ConnectionError, Timeout)):
        return "无法连接模型服务，请检查后端的联网权限、网络或代理设置后重试。"
    return "生成回答时出错了，请稍后重试"


def _chunk_text(text: str, size: int = 12) -> list[str]:
    """
    把文本切成小块，用于模拟流式输出。

    【什么情况下用】
    缓存命中和兜底回答都是"已经完整的文本"，本来可以一次发完。
    但为了和前端的打字机效果保持一致（否则有时逐字出、有时整段出，很跳），
    这里也切成小块发送，每块之间稍作停顿。
    """
    return [text[i : i + size] for i in range(0, len(text), size)]


def _to_int_or_none(value: Any) -> Optional[int]:
    """安全地转整数（用于 document_id）"""
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (ValueError, TypeError):
        return None


def _rewrite_info(result: Any) -> Optional[dict[str, Any]]:
    """把改写结果转成字典（没有改写就返回 None）"""
    if result is None or not getattr(result, "changed", False):
        return None
    return result.to_dict()


# ==================================================================
# 单例
# ==================================================================
_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """获取全局 RAG 主链（单例）"""
    global _chain
    if _chain is None:
        _chain = RAGChain()
    return _chain
