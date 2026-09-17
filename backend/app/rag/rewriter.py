"""
多轮问题改写 —— 把"它多少钱"变成"iPhone 16 多少钱"

================================================================
【要解决什么问题】
================================================================
多轮对话里，用户会用代词和省略：

    用户：iPhone 16 Pro 的电池怎么样？
    客服：这款手机内置 3561mAh 电池……
    用户：**它**多少钱？            ← "它"是谁？

如果直接拿"它多少钱"去检索知识库，什么都搜不到——
因为资料里没有叫"它"的商品。

解决办法：**先让模型结合上文，把问题补全成完整的问题，再去检索。**

    输入：历史对话 + "它多少钱"
    输出："iPhone 16 Pro 多少钱"
    然后拿这个完整问题去检索 → 能搜到了

================================================================
【关键的性能优化：不是每次都要改写】
================================================================
改写要额外调用一次大模型，意味着：
  · 多花约 1 秒
  · 多花一次 token 的钱

但**大部分问题其实不需要改写**。比如"你们支持退货吗"——
它本身就是完整独立的，硬要改写只是浪费时间。

所以本模块会**先用规则判断"这题需不需要改写"**，
只在真正需要时才调用模型。这个优化能省掉大约 60%~70% 的改写调用。

【判断规则】
需要改写的信号：
  ① 含代词：它、这个、那个、该、此、上述……
  ② 很短（少于 8 个字）：短问题大概率省略了主语
  ③ 以"那么""那""还有呢"这类承接词开头
  ④ 含"呢"结尾的追问："保修呢？"

不需要改写：
  ① 没有历史对话（第一轮问什么都是完整的）
  ② 问题足够长且不含代词
"""

from __future__ import annotations

import re
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.core.logging import logger
from app.rag.prompt import build_rewrite_prompt, format_history

# ------------------------------------------------------------------
# 需要改写的信号词
# ------------------------------------------------------------------
PRONOUNS = {
    "它", "他", "她", "这个", "那个", "这些", "那些", "该", "此",
    "上述", "上面", "前面", "其", "此款", "这款", "那款", "这种", "那种",
}

# 口语里常见的省略开头
LEADING_WORDS = {
    "那", "那么", "还有", "另外", "再", "顺便", "对了", "嗯", "所以",
}

# 追问结尾
TRAILING_PATTERNS = [
    re.compile(r"呢[?？]?$"),
    re.compile(r"怎么样[?？]?$"),
    re.compile(r"如何[?？]?$"),
]

# 短问题的字数阈值
SHORT_QUESTION_THRESHOLD = 8


class QueryRewriter:
    """
    问题改写器

    用法：
        rewriter = QueryRewriter()
        result = rewriter.rewrite("它多少钱", history=[...])
        print(result.rewritten)   # "iPhone 16 Pro 多少钱"
        print(result.rewritten == ... )  # 检查是否真的改了
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._chain: Any = None

    def _get_chain(self):
        """懒加载改写链（第一次用时才创建，省启动时间）"""
        if self._chain is None:
            from langchain_community.chat_models import ChatTongyi

            llm = ChatTongyi(
                model=settings.LLM_MODEL,
                temperature=0.0,   # 改写要稳定，不要创造性 → 温度设为 0
                dashscope_api_key=settings.DASHSCOPE_API_KEY,
            )
            # LCEL 写法：提示词 → 模型 → 取纯文本
            # 竖线 | 表示"把左边的东西的输出，喂给右边"
            self._chain = build_rewrite_prompt() | llm | StrOutputParser()
            logger.info("问题改写器已初始化")
        return self._chain

    # ==============================================================
    # 规则判断：需不需要改写
    # ==============================================================
    @staticmethod
    def needs_rewrite(question: str, has_history: bool) -> tuple[bool, str]:
        """
        判断这个问题需不需要改写。

        返回：(是否需要, 原因说明)

        【为什么这个判断值得单独写一个函数】
        因为它直接影响成本和速度，而且逻辑需要不断调整。
        独立出来之后，① 容易测试 ② 容易优化 ③ 决策过程可解释
        （日志里能看到"为什么改了"或"为什么没改"）。
        """
        if not has_history:
            return False, "第一轮提问，没有上文可参考"

        q = (question or "").strip()
        if not q:
            return False, "问题为空"

        # ---- 信号 1：含代词 ----
        for pronoun in PRONOUNS:
            if pronoun in q:
                return True, f"含代词「{pronoun}」"

        # ---- 信号 2：问题太短 ----
        # 去掉标点后算长度
        pure = re.sub(r"[^\w一-鿿]", "", q)
        if len(pure) <= SHORT_QUESTION_THRESHOLD:
            return True, f"问题过短（{len(pure)} 字），可能省略了主语"

        # ---- 信号 3：承接词开头 ----
        for word in LEADING_WORDS:
            if q.startswith(word):
                return True, f"以承接词「{word}」开头"

        # ---- 信号 4：追问句式 ----
        for pattern in TRAILING_PATTERNS:
            if pattern.search(q):
                return True, "是追问句式"

        return False, "问题已完整独立，无需改写"

    # ==============================================================
    # 主流程
    # ==============================================================
    def rewrite(
        self,
        question: str,
        history: Optional[list[dict[str, str]]] = None,
    ) -> "RewriteResult":
        """
        改写问题。

        参数：
            question —— 用户的最新问题
            history  —— 历史消息列表，每项 {"role": "user"/"assistant", "content": "..."}

        返回：RewriteResult（含改写前后的问题、是否改写、原因、耗时）
        """
        import time

        history = history or []
        t0 = time.perf_counter()

        # ---- 先判断需不需要改写 ----
        need, reason = self.needs_rewrite(question, bool(history))

        if not self.enabled or not need:
            logger.debug(f"跳过改写 | 原因：{reason}")
            return RewriteResult(
                original=question,
                rewritten=question,
                changed=False,
                reason=reason,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # ---- 调用模型改写 ----
        try:
            chain = self._get_chain()
            history_text = format_history(history)

            raw = chain.invoke(
                {"history": history_text, "question": question}
            )

            rewritten = self._clean_output(raw, question)

            changed = rewritten != question
            logger.info(
                f"问题改写 | 「{question}」 → 「{rewritten}」 | 原因：{reason}"
            )

            return RewriteResult(
                original=question,
                rewritten=rewritten,
                changed=changed,
                reason=reason,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        except Exception as e:
            # 改写失败不能让整个问答挂掉——直接用原问题继续。
            # 最坏的结果是这一轮追问答不上来，但主流程不受影响。
            logger.warning(
                f"问题改写失败，使用原问题继续 | {type(e).__name__}: {str(e)[:150]}"
            )
            return RewriteResult(
                original=question,
                rewritten=question,
                changed=False,
                reason=f"改写失败（{type(e).__name__}），降级使用原问题",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    # ==============================================================
    # 输出清洗
    # ==============================================================
    @staticmethod
    def _clean_output(raw: str, original: str) -> str:
        """
        清洗模型输出。

        【为什么要清洗】
        即使提示词说了"只输出改写后的问题"，模型还是经常会：
          · 加上"改写后的问题是："这样的前缀
          · 用引号把结果包起来
          · 前后带换行和空格
          · 输出多行（把解释也写上了）

        这些都会污染检索——因为我们直接拿这个字符串去搜。
        所以必须做一轮清理，并且加一道"安全检查"：
        如果清洗后的结果明显不合理（太长、太短、为空），
        就放弃改写，回退到原问题。
        """
        if not raw:
            return original

        text = raw.strip()

        # 去掉常见的前缀
        for prefix in (
            "改写后的问题：", "改写后的问题:", "改写后：", "改写后:",
            "改写结果：", "改写结果:", "问题：", "问题:",
            "Rewritten question:", "Rewritten:",
        ):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # 去掉包裹的引号
        for quote_pair in (('"', '"'), ("'", "'"), ("「", "」"), ("《", "》"), ("“", "”")):
            if text.startswith(quote_pair[0]) and text.endswith(quote_pair[1]) and len(text) > 2:
                text = text[1:-1].strip()

        # 只取第一行（模型有时会输出多行）
        if "\n" in text:
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            text = lines[0] if lines else original

        # ---- 安全检查 ----
        if not text:
            return original

        # 结果过长 → 模型可能在胡说，放弃
        if len(text) > max(len(original) * 4, 100):
            logger.warning(f"改写结果过长，放弃改写 | {text[:60]!r}")
            return original

        # 结果过短 → 可能只输出了一个词，放弃
        if len(text) < 2:
            return original

        return text


class RewriteResult:
    """改写结果（用一个类而不是字典，方便点属性，也更清晰）"""

    __slots__ = ("original", "rewritten", "changed", "reason", "elapsed_ms")

    def __init__(
        self,
        original: str,
        rewritten: str,
        changed: bool,
        reason: str,
        elapsed_ms: float = 0.0,
    ):
        self.original = original
        self.rewritten = rewritten
        self.changed = changed
        self.reason = reason
        self.elapsed_ms = round(elapsed_ms, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "changed": self.changed,
            "reason": self.reason,
            "elapsed_ms": self.elapsed_ms,
        }


# ==================================================================
# 单例
# ==================================================================
_rewriter: Optional[QueryRewriter] = None


def get_rewriter() -> QueryRewriter:
    """获取全局改写器（单例）"""
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter
