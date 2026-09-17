"""
BM25 关键词检索 —— 补上向量检索的短板

================================================================
【为什么"向量检索"已经能搜了，还要再来一套"关键词检索"】
================================================================
向量检索擅长"按意思找"，但它有个明显短板：**精确词匹配不敏感**。

实测中的真实案例：
    用户问："包装里有充电器吗"
    文档写："包装盒内不附带充电器"
    向量检索结果：❌ 找到了"退换货政策"（完全无关）
    正确答案：   ✅ 应该在"电池续航"那段

为什么向量检索失手了？
因为"包装里有"和"包装盒内"在语义空间里确实相近，
但整体句子的语义重心被"充电器"和"退货配件"这些词拉偏了。
向量是把整句话压缩成一串数字，细节会被"平均"掉。

而 BM25 是**精确词匹配**：它只看"问题里的词，在文档里出现了几次"。
"包装"和"充电器"这两个词，恰好同时出现在正确答案里，
BM25 一眼就能抓住。

================================================================
【BM25 是什么】
================================================================
它是搜索引擎用了几十年的经典排序算法（Google 早期也用了类似思想）。
核心思想就三点：

  1. **词频（TF）**：某个词在文档里出现越多，这篇文档越可能相关。
     但会做"饱和处理"——出现 10 次不比出现 5 次强一倍，
     因为关键词出现太多次往往是堆砌，不是真的更相关。

  2. **逆文档频率（IDF）**：越罕见的词越重要。
     "的"、"是"这种词到处都是，不能说明任何问题；
     而"充电器"只在少数文档里出现，一旦命中就很有价值。

  3. **文档长度归一化**：长文档天然更容易命中词，所以要打个折扣。
     否则"越长的文档越容易被搜到"，会压制短而精准的文档。

================================================================
【中文为什么必须用 jieba 分词】
================================================================
BM25 的原理是"统计词的出现次数"，前提是得先知道**什么是"一个词"**。

英文靠空格就能切：  "battery is durable" → ["battery", "is", "durable"]

中文没有空格："电池耐用吗" → 如果按字切 → ["电","池","耐","用","吗"]
   这样"电池"和"池耐"都变成了"词"，统计毫无意义。

jieba 能正确切成 ["电池", "耐用", "吗"]——这才是真正的词。

================================================================
【两路检索怎么合并 —— 这是下一个模块（retriever.py）的事】
简单说：不看两边的分数（量纲不同没法比），**只看排名**，
用 RRF 算法把两路的名次融成一个综合名次。详见 retriever.py。
"""

from __future__ import annotations

import re
import threading
from typing import Any, Optional

from app.core.logging import logger

# ------------------------------------------------------------------
# 中文停用词表
#
# 【什么是停用词】
# 那些"到处都是、不携带信息"的词。它们在几乎每篇文档里都出现，
# IDF 极低，参与计算只会增加噪音、拖慢速度。
#
# 比如"的、了、吗、是、在"——去掉它们，"iPhone 的电池容量是多少"
# 变成 "iPhone 电池容量"，检索意图反而更清晰。
#
# 注意：**不要过度删除**。像"不"、"没"这种否定词，虽然在停用词表里
# 很常见，但删掉会改变语义（"可以用" vs "不可以用"是完全相反的！）。
# 所以本表刻意保留了否定词。
# ------------------------------------------------------------------
STOPWORDS: set[str] = {
    # 助词、语气词
    "的", "了", "着", "过", "呢", "吗", "吧", "啊", "呀", "哦", "嗯", "哈",
    "地", "得", "之", "其", "所", "以", "及", "与", "和", "跟", "同",
    # 代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那",
    "这个", "那个", "这些", "那些", "什么", "哪个", "哪些",
    # 常见动词/介词
    "是", "有", "在", "为", "被", "把", "给", "对", "从", "到", "向",
    "请问", "请", "帮", "帮忙", "想要", "需要", "可以", "能不能", "能否",
    # 数量词
    "一个", "一些", "一下", "多少", "几个",
    # 标点和空白（分词后可能残留）
    "，", "。", "！", "？", "；", "：", "、", "（", "）", "《", "》",
    " ", "\n", "\t", "", "-", "—", "…", "·",
}

# 需要保留的否定词（它们不在停用词表里，这里单独声明以示强调）
NEGATION_WORDS = {"不", "没", "非", "无", "别"}

# ------------------------------------------------------------------
# 领域自定义词典
#
# 【为什么必须补充】
# jieba 的通用词典对电商和数码领域的专有名词识别很差。实测踩到的例子：
#
#     "可以用花呗付款吗"  →  jieba 切成 ['用花', '呗', '付款']
#                              ↑「花呗」被劈成两半，BM25 完全失效
#
# 补进词典后：           →  ['花呗', '付款']   ← 正确
#
# 同样需要补充的还有：品牌名（iPhone、MateBook）、
# 电商特有说法（无理由退货、以旧换新）、售后术语（保修期、三包）等。
#
# 【怎么维护这个表】
# 每次发现"某个词被切错导致检索不准"，就把它加进来。
# 也可以把它移到外部配置文件，让管理员在后台维护。
# ------------------------------------------------------------------
DOMAIN_WORDS = [
    # 品牌与产品线
    "iPhone", "iPad", "MacBook", "MateBook", "AirPods", "Apple", "华为", "小米",
    "通义千问", "百炼",
    # 电商与支付
    "花呗", "白条", "云闪付", "对公转账", "免息", "分期", "分期付款",
    "以旧换新", "优惠券", "满减", "包邮", "运费", "次日达", "货到付款",
    "无理由退货", "七天无理由", "七天无理由退货", "退换货", "退款",
    # 售后与保修
    "保修期", "三包", "保修卡", "发票", "电子发票", "纸质发票",
    "进水", "进液", "人为损坏", "私自拆机", "原装配件", "授权维修",
    # 数码参数
    "电池容量", "续航", "快充", "无线充电", "充电器", "充电头", "数据线",
    "刷新率", "分辨率", "像素", "处理器", "内存", "固态硬盘", "长焦",
    "超广角", "主摄", "前置摄像头", "防水等级", "防尘防水", "机身重量",
    "钛金属", "铝合金", "触控屏", "全面屏", "屏幕尺寸",
]

# 词典是否已加载（只加载一次）
_domain_dict_loaded = False


def _ensure_jieba_ready() -> None:
    """
    加载 jieba 及其领域词典。

    【为什么要单独抽一个函数】
    因为 jieba 的词典加载有两个"一次性成本"，必须确保只做一次：
      1. 加载主词典（约 1.2 秒）
      2. 添加自定义词（很快，但重复添加会越积越多，占内存）
    """
    global _domain_dict_loaded

    import jieba

    if not _domain_dict_loaded:
        for word in DOMAIN_WORDS:
            # add_word 让 jieba 把这些词当成一个整体，不再拆开
            jieba.add_word(word)
        _domain_dict_loaded = True
        logger.debug(f"jieba 领域词典已加载（{len(DOMAIN_WORDS)} 个词）")


def tokenize(text: str) -> list[str]:
    """
    把中文文本切成词列表（供 BM25 使用）。

    【处理流程】
      1. 转小写（英文大小写统一，"iPhone" 和 "iphone" 应该算同一个词）
      2. jieba 分词（含领域词典）
      3. 去掉停用词
      4. 去掉纯标点和空白
      5. 保留否定词（"不"、"没"等，它们改变语义）

    【jieba 的懒加载】
    jieba 第一次使用时需要加载词典（约 1.2 秒），之后就是毫秒级。
    这里已经做了处理：只在第一次调用时加载。
    """
    if not text:
        return []

    _ensure_jieba_ready()

    import jieba

    # 转小写，让英文匹配不受大小写影响
    lowered = text.lower()

    # 精确模式分词（jieba.cut 默认就是精确模式，会尽量切出合理的词）
    words = jieba.lcut(lowered)

    result: list[str] = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        # 纯标点/符号，扔掉
        if re.fullmatch(r"[\W_]+", w, flags=re.UNICODE):
            continue
        # 停用词扔掉（但否定词例外——它们携带关键语义）
        if w in STOPWORDS and w not in NEGATION_WORDS:
            continue
        result.append(w)

    return result


class BM25Index:
    """
    BM25 关键词索引

    【为什么放在内存里而不是数据库】
    BM25 计算需要遍历全部文档的词频统计，放在内存里是微秒级；
    如果每次查询都去数据库捞一遍再算，就完全失去了意义。
    个人项目的知识库规模（几千到几万个片段）放在内存里，
    占用也就几十兆，完全可接受。

    【什么时候重建】
      · 第一次使用（懒加载）
      · 上传/删除文档后（标记失效，下次查询时自动重建）
    重建的成本：1 万个片段大约 1-2 秒。如果觉得频繁重建太慢，
    后面可以改成"增量添加"，但个人项目没必要。
    """

    def __init__(self):
        self._bm25: Any = None
        self._chunks: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._dirty = True          # 是否需要重建
        self._build_count = 0       # 重建次数（监控用）

    # ==============================================================
    # 构建索引
    # ==============================================================
    def build(self, chunks: list[dict[str, Any]]) -> None:
        """
        用给定的片段列表重建索引。

        参数 chunks：每项需包含
            {"content": 子块文本, "parent_content": 父块文本,
             "file_name": ..., "page": ..., "doc_id": ...}
        """
        from rank_bm25 import BM25Okapi

        with self._lock:
            if not chunks:
                self._bm25 = None
                self._chunks = []
                self._dirty = False
                logger.debug("BM25 索引已清空（没有片段）")
                return

            # 分词：把每个片段切成词列表
            tokenized = [tokenize(c.get("content", "")) for c in chunks]

            # 过滤掉分词后为空的片段（比如纯符号内容），
            # 否则 BM25Okapi 会因空文档报错或产生奇怪结果
            valid = [(t, c) for t, c in zip(tokenized, chunks) if t]

            if not valid:
                self._bm25 = None
                self._chunks = []
                self._dirty = False
                return

            tokens, self._chunks = zip(*valid)
            tokens = list(tokens)

            self._bm25 = BM25Okapi(tokens)
            self._dirty = False
            self._build_count += 1

            avg_len = sum(len(t) for t in tokens) / len(tokens)
            logger.info(
                f"BM25 索引构建完成 | {len(tokens)} 个片段 | "
                f"平均每片段 {avg_len:.1f} 个词"
            )

    def mark_dirty(self) -> None:
        """标记索引已失效，下次查询时自动重建"""
        with self._lock:
            self._dirty = True

    # ==============================================================
    # 检索
    # ==============================================================
    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """
        BM25 关键词检索。

        返回：列表，每项包含 content / score / file_name / page 等，
              score 是 BM25 原始分数（**只在本次查询内部可比**，
              不能和向量检索的相似度直接比较——量纲完全不同）。
        """
        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []

            query_tokens = tokenize(query)
            if not query_tokens:
                logger.debug(f"BM25: 查询分词后为空 | query={query!r}")
                return []

            scores = self._bm25.get_scores(query_tokens)

            # 取分数最高的 top_k 个
            # 用 argsort 而不是 sorted（后者要构造整个元组列表，更慢）
            indexed = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]

            results: list[dict[str, Any]] = []
            for idx, score in indexed:
                # BM25 分数为 0 表示"一个查询词都没匹配到"，这类结果没意义，过滤掉
                if score <= 0:
                    continue

                chunk = self._chunks[idx]
                results.append(
                    {
                        "content": chunk.get("content", ""),
                        "parent_content": chunk.get("parent_content") or chunk.get("content", ""),
                        "score": round(float(score), 4),
                        "file_name": chunk.get("file_name", "未知文档"),
                        "page": chunk.get("page"),
                        "sheet_name": chunk.get("sheet_name"),
                        "doc_id": chunk.get("doc_id"),
                        "matched_words": self._explain_match(query_tokens, chunk.get("content", "")),
                        "source": "bm25",   # 标记来源，便于调试混合检索
                    }
                )

            logger.debug(
                f"BM25 检索 | query={query[:40]!r} | "
                f"分词={query_tokens} | 命中 {len(results)} 条"
            )
            return results

    @staticmethod
    def _explain_match(query_tokens: list[str], content: str) -> list[str]:
        """
        解释"为什么这条被检索到了"——列出实际命中的查询词。

        【这个功能有什么用】
        调试检索问题时，光看分数不知道原因。
        列出命中的词，一眼就能看出"哦，是因为'充电器'这个词命中了"。
        这是可解释性设计，排查问题时非常省时间。
        """
        content_lower = content.lower()
        return [w for w in query_tokens if w in content_lower]

    # ==============================================================
    # 状态
    # ==============================================================
    @property
    def is_ready(self) -> bool:
        """索引是否已构建且未失效"""
        with self._lock:
            return self._bm25 is not None and not self._dirty

    @property
    def size(self) -> int:
        """索引里的片段数量"""
        with self._lock:
            return len(self._chunks)

    def get_stats(self) -> dict[str, Any]:
        """索引统计信息"""
        with self._lock:
            return {
                "chunks_indexed": len(self._chunks),
                "is_ready": self._bm25 is not None and not self._dirty,
                "needs_rebuild": self._dirty,
                "build_count": self._build_count,
                "stopwords_count": len(STOPWORDS),
            }


# ==================================================================
# 单例 + 懒加载
# ==================================================================
_index: Optional[BM25Index] = None
_index_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    """
    获取全局 BM25 索引（单例）。

    注意这里**不自动构建**——构建需要读取全部片段，
    而这件事依赖向量库。所以由调用方（retriever）负责
    在需要时调用 ensure_ready()。
    """
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = BM25Index()
    return _index


def ensure_ready(force: bool = False) -> BM25Index:
    """
    确保 BM25 索引可用（不可用就从向量库拉数据重建）。

    【为什么要懒加载而不是启动时就建】
    如果程序启动时就构建，启动会多花几秒；而且如果用户只是
    想登录看看历史记录（不问答），这几秒就是白花的。
    懒加载让"第一次真正用到时才付出这个成本"。
    """
    index = get_bm25_index()

    if index.is_ready and not force:
        return index

    # 从向量库读取全部片段来重建索引
    try:
        from app.rag.vectorstore import get_vector_store

        vs = get_vector_store()
        raw = vs._store._collection.get(include=["documents", "metadatas"])

        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []

        chunks: list[dict[str, Any]] = []
        for i, vid in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            meta = meta or {}
            chunks.append(
                {
                    # 唯一 ID：两路检索结果合并时靠它去重
                    # （同一段文字可能同时被向量检索和 BM25 检索命中）
                    "id": vid,
                    "content": documents[i] if i < len(documents) else "",
                    "parent_content": meta.get("parent_content", ""),
                    "file_name": meta.get("file_name", ""),
                    "page": meta.get("page") if meta.get("page", -1) > 0 else None,
                    "sheet_name": meta.get("sheet_name"),
                    "doc_id": meta.get("doc_id"),
                    "child_index": meta.get("child_index"),
                }
            )

        index.build(chunks)
    except Exception as e:
        logger.error(f"BM25 索引构建失败: {type(e).__name__}: {e}")

    return index


def invalidate() -> None:
    """
    让 BM25 索引失效（上传或删除文档后调用）。

    下次检索时会自动重建，从而包含最新数据。
    """
    get_bm25_index().mark_dirty()
    logger.debug("BM25 索引已标记失效，将在下次检索时重建")
