"""
父子文档切分器 —— 兼顾"检索精准"和"上下文完整"

================================================================
【先说清楚要解决的矛盾】
================================================================
RAG 要把长文档切成小块才能检索。但块的大小是个两难：

  块切得**小**（200 字）
    ✅ 检索准：这一小块主题单一，向量坐标清晰，容易被准确命中
    ❌ 但喂给模型时信息不全：模型只看到半句话，答不完整

  块切得**大**（1500 字）
    ✅ 信息全：模型能看到完整上下文
    ❌ 检索不准：一大块里混了好几个主题，向量坐标被"平均"了，
                搜什么都能命中一点，反而分不出哪个最相关

================================================================
【解法：父子文档（Small-to-Big）】
================================================================
把文档同时切成两层：

  ┌─────────────────────────────────────────┐
  │  父块（约 1400 字）                       │  ← 喂给模型看这个
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
  │  │ 子块 1   │ │ 子块 2   │ │ 子块 3   │ │  ← 检索用这些
  │  │ (350字)  │ │ (350字)  │ │ (350字)  │ │
  │  └──────────┘ └──────────┘ └──────────┘ │
  └─────────────────────────────────────────┘

流程：
  1. 用**子块**去向量检索 → 命中精准（因为子块主题单一）
  2. 查出这个子块属于哪个**父块**
  3. 把**父块**交给模型 → 上下文完整

比喻：**用目录精确找到是哪一章（子块），然后把整章内容拿给模型看（父块）。**

================================================================
【关键设计：为什么要先合并再切】
================================================================
❌ 错误做法：对每个文档块单独切父块
   Markdown 解析器按标题把文档切成了 6 个 200 字的小块，
   每块都小于父块目标（1400 字）→ 每个都自成一个父块 → **父子退化成 1:1，策略完全失效**。

✅ 正确做法：先把所有文档块**合并成一大篇**，再切父块
   同时维护一张"字符位置 → 第几页"的映射表，
   这样切出来的父块能跨页、能包含多个子块，策略才真正生效，
   而且每个块仍然知道自己的页码（引用溯源需要）。

================================================================
【为什么用"递归字符切分"而不是"按字数硬切"】
================================================================
硬切（每 400 字切一刀）的后果：
    原文："...iPhone 16 的电池容量为 3561mAh，支持 20W 快充。..."
    硬切后：片段A="...iPhone 16 的电池容量为 3561mAh，支持 20"
            片段B="W 快充。..."
"20W 快充"被劈成两半，两个片段都无法回答"支持多少瓦快充"。

递归字符切分的做法是：**按优先级尝试分隔符**
    先试 "\n\n"（段落）→ 太大就试 "\n"（换行）
    → 还大就试 "。"（句号）→ 再大就试 "，"（逗号）→ 最后才硬切
这样绝大部分切分都落在自然的语义边界上，不会劈断句子。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logging import logger


# ==================================================================
# 结构性小标题的识别规则
# ==================================================================
# 【这个正则是在找什么】
# 找那些"标志着新章节开始"的行首标记。命中了就说明这里该断一刀，
# 不能让这一节的内容跟下一节混进同一个检索片段里。
#
# 【认识一下这几种写法】
#   一、春季服装          ← 中文数字 + 顿号（最常见）
#   第三章 售后服务        ← "第X章/节/条"
#   1. 纯棉材质           ← 阿拉伯数字 + 点（最常见）
#   1.2.3 参数说明        ← 多级编号
#   (1) 注意事项          ← 括号编号（中英文括号都认）
#   ## 二级标题           ← Markdown 标题
#
# 【为什么要写成 ^... 开头】
# 只认"出现在行首"的编号。因为"水温≤30℃，1. 先用冷水泡" 这种
# 句子中间的"1." 不是章节标记，断在那儿会劈断句子。
#
# 【为什么数字后面必须跟空白】
# 这一条能挡掉一大类误判。比如 "3.5 元的商品" 或 "165-175cm"，
# 它们虽然以数字和点开头，但点后面紧跟的是数字不是空格，
# 所以不会被当成章节标题。
STRUCT_HEADING_RE = re.compile(
    r"^(?:"
    r"[一二三四五六七八九十]+[、．.]"                      # 一、  二、
    r"|第[一二三四五六七八九十百零〇0-9]+[章节節部条篇]"    # 第一章  第2节
    r"|(?:\d+\.(?:\d+\.)*)\s"                             # 1.   1.2   1.2.3
    r"|[(（]\s*[0-9]+\s*[)）]"                            # (1)  （1）
    r"|#{1,6}\s"                                          # Markdown 标题
    r")"
)


# ==================================================================
# 中文友好的分隔符优先级
# ==================================================================
# 【为什么必须自定义】
# LangChain 默认的分隔符是 ["\n\n", "\n", " ", ""] —— 这是**为英文设计的**，
# 因为英文靠空格断词。但中文句子内部没有空格，所以 " " 这一档对中文几乎无效，
# 会导致大量内容只能在最后一级"硬切"，反而劈断句子。
SEPARATORS_ZH = [
    "\n\n",     # 段落之间（最强的语义边界）
    "\n",       # 换行
    "。",       # 句号（中文句子结束）
    "！",       # 感叹号
    "？",       # 问号
    "；",       # 分号
    "，",       # 逗号（中文）
    ". ",       # 英文句号
    "! ",
    "? ",
    "; ",
    ", ",
    " ",        # 空格（英文单词边界）
    "",         # 最后兜底：硬切
]


@dataclass
class ParentChildChunk:
    """一个"子块 + 它的父块"的组合（切分环节的输出单位）"""

    child_content: str          # 子块文本（用于向量检索）
    parent_content: str         # 父块文本（用于喂给模型）
    child_index: int            # 子块序号（从 0 开始）
    parent_index: int           # 所属父块序号（从 0 开始）
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def vector_id(self) -> str:
        """
        生成一个稳定、唯一的向量 ID。

        【为什么不用随机 UUID】
        如果用随机 ID，每次重新索引同一个片段都会得到不同的 ID，
        结果就是"重复索引"时向量库里会堆积大量重复向量（旧的还删不掉）。

        用"内容 + 位置"算哈希：
          · 内容没变 → ID 不变 → 重复索引是覆盖而不是新增
          · 内容变了 → ID 变化 → 能正确区分新旧内容
        """
        raw = (
            f"{self.metadata.get('doc_id', '')}:"
            f"{self.parent_index}:{self.child_index}:"
            f"{self.child_content[:100]}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class ParentChildSplitter:
    """父子文档切分器"""

    def __init__(
        self,
        child_size: int | None = None,
        child_overlap: int | None = None,
        parent_size: int | None = None,
        structure_aware: bool | None = None,
        min_section_chars: int | None = None,
    ):
        """
        参数（默认值来自配置文件 .env）：

            child_size     子块大小（字数）。越小检索越准，但语义越不完整。
            child_overlap  相邻子块的重叠字数。
                           为什么要重叠？因为切分点难免会切断语义，
                           让相邻块有 80 字交集，能保证关键信息至少
                           完整出现在其中一个块里。
            parent_size    父块大小。要明显大于 child_size，
                           否则父子策略会退化成 1:1（就白做了）。

            structure_aware    是否按文档自身的编号结构切分。
                               开 = 有编号的文档切得准（推荐）
                               关 = 一律按字数切（用于做对比实验）

            min_section_chars  小节合并阈值。小于这个字数的小节
                               会被并进下一节，避免出现"只有一行标题、
                               没有任何内容"的碎片。
        """
        self.child_size = child_size or settings.CHUNK_SIZE
        self.child_overlap = child_overlap or settings.CHUNK_OVERLAP
        self.parent_size = parent_size or settings.PARENT_CHUNK_SIZE

        self.structure_aware = (
            settings.STRUCTURE_AWARE_SPLIT if structure_aware is None else structure_aware
        )
        self.min_section_chars = min_section_chars or settings.MIN_SECTION_CHARS

        # 安全校验：父块必须足够大，否则父子策略没意义
        if self.parent_size < self.child_size * 1.5:
            self.parent_size = int(self.child_size * 3)

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_size,
            chunk_overlap=self.child_overlap,
            separators=SEPARATORS_ZH,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True,
            add_start_index=True,   # 让每个块带上"在原文中的起始位置"
        )

        # 父块**不设重叠**：父块是给模型看的上下文，
        # 重叠会导致同一内容在提示词里出现两遍，浪费 token 还可能让模型困惑。
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_size,
            chunk_overlap=0,
            separators=SEPARATORS_ZH,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True,
            add_start_index=True,
        )

    # ==============================================================
    # 主流程
    # ==============================================================
    def split(self, docs: list[Document], doc_id: int | str = "") -> list[ParentChildChunk]:
        """
        把解析好的文档切成父子块。

        参数：
            docs    —— loader 解析出来的 Document 列表
            doc_id  —— 文档在数据库里的 ID，写进元数据用于后续删除和溯源
        """
        if not docs:
            return []

        # ---- 第一步：贪心打包成若干个"父块候选组" ----
        #
        # 【为什么不能简单地把所有文档块合并成一大篇】
        # 我一开始就是这么写的，结果踩了坑：
        # 一份商品资料有"产品参数""售后政策""支付说明"等 6 个不相关章节，
        # 全合并之后，如果总长度不到父块大小(1400字)，就会变成**1 个**块——
        # 于是"手机电池"和"退货政策"被塞进了同一个片段，
        # 检索"能退货吗"时，这个混杂的片段因为同时包含两类内容，
        # 向量坐标被"平均"了，和任何问题都不够像，检索质量直接崩掉。
        #
        # 正确做法是**贪心打包**：按顺序往箱子里装文档块，
        # 装到接近 parent_size 就先封箱、开新箱。
        # 这样既保证每个箱子足够大（父子比例正常），
        # 又不会把不相关的内容硬凑在一起。
        groups = self._pack_into_groups(docs)

        if not groups:
            return []

        # ---- 第二步：每个组内部切父块，父块内部再切子块 ----
        all_chunks: list[ParentChildChunk] = []
        child_index = 0
        parent_index = 0

        for group_text, page_map in groups:
            for p_start, p_text in self._split_into_parents(group_text):
                if not p_text:
                    continue

                # 父块在组内文本里的起始位置 → 查出它属于第几页
                p_page_meta = self._lookup_page(p_start, page_map)

                # 切子块。注意这里不再是直接调 LangChain 的切分器，
                # 而是先按文档结构切、切不动了再交给切分器兜底。
                # 详细逻辑见 _make_children()。
                for c_offset_in_parent, c_text in self._make_children(p_text):
                    # 子块在组内文本里的绝对位置 = 父块起点 + 子块在父块内的偏移
                    c_page_meta = self._lookup_page(
                        p_start + c_offset_in_parent, page_map
                    )

                    meta: dict[str, Any] = {
                        "doc_id": doc_id,
                        "parent_index": parent_index,
                        "child_index": child_index,
                        "parent_char_count": len(p_text),
                        "child_char_count": len(c_text),
                    }
                    # 页码等信息以子块自己的位置为准（同一个父块可能跨页）
                    meta.update(c_page_meta)
                    # 保留原始文档级元数据里其他有用字段（比如 section、title）
                    for k, v in p_page_meta.items():
                        meta.setdefault(k, v)

                    all_chunks.append(
                        ParentChildChunk(
                            child_content=c_text,
                            parent_content=p_text,
                            child_index=child_index,
                            parent_index=parent_index,
                            metadata=meta,
                        )
                    )
                    child_index += 1

                parent_index += 1

        parent_count = len({c.parent_index for c in all_chunks})
        logger.info(
            f"切分完成 | 输入 {len(docs)} 个文档块 → 打包成 {len(groups)} 组 → "
            f"{parent_count} 个父块 / {len(all_chunks)} 个子块 "
            f"（平均每父块 {len(all_chunks) / max(1, parent_count):.1f} 个子块）"
        )

        return all_chunks

    # ==============================================================
    # 父块切分：同样优先在结构边界上断开
    # ==============================================================
    def _split_into_parents(self, group_text: str) -> list[tuple[int, str]]:
        """
        把一个组切成若干父块，返回 [(在组内的起始位置, 父块文本), ...]

        【注意元组的顺序是"位置在前、文本在后"】
        和 _split_by_structure、_make_children 保持一致。
        （我第一版把文档注释写反了，结果 split() 里按注释去解包，
          拿到的是"整数当文本、文本当整数"，直接报
          TypeError: object of type 'int' has no len()。
          这种低级错误靠注释和代码对不上是查不出来的，
          所以统一约定：**所有返回位置的元组，位置一律放第一个**。）

        【为什么父块也要看结构】
        父块本来是纯按字数切的（凑够 1400 字就断）。
        问题是这一刀落在哪全看运气 —— 实测中就有一刀正好劈在
        「灯芯绒材质」这一节的中间，结果：
            上一片结尾：3. 灯芯绒材质 …洗涤：手洗或机洗…
            下一片开头：养护：阴凉阴干…4. 麂皮绒材质…
        灯芯绒的「怎么洗」和「怎么晾」被拆到了两片里。

        虽然靠 top_k 取 5 片基本都能兜回来，但引用卡片会显示半截内容，
        用户看着别扭。既然子块已经能按结构切了，父块用同一套逻辑
        几乎没有额外成本，顺手做掉。

        【算法和小节合并是同一个思路】
        一个一个小节往箱子里装：
          装得下 → 继续装
          装不下 → 封箱，开新箱
          单个小节本身就超重 → 先封箱，再把这个大节单独递归切
        """
        result: list[tuple[int, str]] = []

        sections = (
            self._split_by_structure(group_text)
            if self.structure_aware
            else [(0, group_text)]
        )

        buf = ""
        buf_start = 0

        for sec_offset, sec_text in sections:
            # 这一节自己就超过父块大小 → 先把手上的封箱，再单独处理它
            if len(sec_text) > self.parent_size:
                if buf:
                    result.append((buf_start, buf))
                    buf = ""
                for d in self.parent_splitter.create_documents([sec_text]):
                    t = d.page_content.strip()
                    if t:
                        result.append(
                            (sec_offset + d.metadata.get("start_index", 0), t)
                        )
                continue

            if not buf:
                buf_start = sec_offset
                buf = sec_text
            elif len(buf) + len(sec_text) + 2 <= self.parent_size:
                buf = buf + "\n\n" + sec_text      # +2 是两个换行符
            else:
                result.append((buf_start, buf))
                buf_start = sec_offset
                buf = sec_text

        if buf:
            result.append((buf_start, buf))

        return result

    # ==============================================================
    # 子块切分：先按结构切，切不动再按字数兜底
    # ==============================================================
    def _make_children(self, text: str) -> list[tuple[int, str]]:
        """
        把一个父块切成若干子块。

        返回 [(子块在父块内的起始位置, 子块文本), ...]

        【策略：两层】
        第一层 —— 按文档自己的编号结构切（主力）
                  每个小节独立成片，不跟别的小节混在一起
        第二层 —— 按字数递归切（兜底）
                  只用在两种情况下：
                    · 某个小节本身还是太长（超过 child_size）
                    · 整篇文档压根没有编号结构（散文、新闻稿那种）

        【为什么不干脆只用第二层】
        因为按字数切是"盲切"—— 它不知道哪里是语义边界，只知道凑够字数。
        用它切那份洗涤养护资料，就会出现"一片里三种面料"的问题。
        编号结构是**作者自己标出来的语义边界**，比任何算法猜的都要准，
        所以能用就一定要用。
        """
        result: list[tuple[int, str]] = []

        sections = (
            self._split_by_structure(text)
            if self.structure_aware
            else [(0, text)]
        )

        for sec_offset, sec_text in sections:
            if len(sec_text) <= self.child_size:
                # 这一节本身就够小 → 原样作为一整个子块，不再往下切。
                # 这是整个改造的关键：一个 150 字的「雪纺材质」就是一片，
                # 检索时不会跟「冰丝」「麂皮绒」互相干扰。
                result.append((sec_offset, sec_text))
                continue

            # 这一节还是太大 → 交给递归切分器按句子边界继续切
            for d in self.child_splitter.create_documents([sec_text]):
                t = d.page_content.strip()
                if t:
                    result.append((sec_offset + d.metadata.get("start_index", 0), t))

        return result

    def _split_by_structure(self, text: str) -> list[tuple[int, str]]:
        """
        按文档自身的编号小标题切分。

        返回 [(小节在原文中的起始位置, 小节文本), ...]

        【分三步做】
        1. 标出每一行在原文中的字符位置（后面查页码要用）
        2. 找出所有"章节起点"的行，在这些行前面断一刀
        3. 把太短的小节并进下一节

        【第 3 步为什么必须要有】
        那份洗涤养护资料里有这么一行：
            一、春季服装（纯棉、薄牛仔、针织棉、轻薄化纤）
        它只是一行标题，自己只有 25 个字，下面才跟着具体内容。
        如果让它单独成片，检索时它匹配不上任何具体问题
        （"雪纺怎么洗"跟这行字的相似度很低），等于白占一个位置。
        所以让它跟紧跟着的第一节合并。
        """
        # ---- 第 1 步：算出每一行的起始位置 ----
        lines = text.split("\n")
        line_starts: list[int] = []
        pos = 0
        for line in lines:
            line_starts.append(pos)
            pos += len(line) + 1        # +1 是换行符本身占的位置

        # ---- 第 2 步：找出所有章节起点 ----
        boundaries = [0]
        for i, line in enumerate(lines):
            if i == 0:
                continue
            stripped = line.strip()
            if stripped and STRUCT_HEADING_RE.match(stripped):
                boundaries.append(i)
        boundaries.append(len(lines))    # 收尾哨兵

        raw_sections: list[tuple[int, str]] = []
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]

            # 跳过小节开头的空行，让起始位置对准正文的第一个字
            # （不然页码查找会偏到上一节去）
            while a < b and not lines[a].strip():
                a += 1

            sec_text = "\n".join(lines[a:b]).strip()
            if sec_text:
                raw_sections.append((line_starts[a], sec_text))

        if not raw_sections:
            return [(0, text.strip())]

        # ---- 第 3 步：合并过短的小节 ----
        merged: list[tuple[int, str]] = []
        pending_text = ""
        pending_offset = 0

        for offset, sec in raw_sections:
            if not pending_text:
                pending_offset = offset
                pending_text = sec
            else:
                pending_text = pending_text + "\n\n" + sec

            # 攒够了就封存，不够就继续跟下一节合并
            if len(pending_text) >= self.min_section_chars:
                merged.append((pending_offset, pending_text))
                pending_text = ""

        # 循环结束后如果还攥着一小截没封存，说明它后面没有更多内容了，
        # 那就并进上一节（不能丢掉，否则内容会凭空消失）
        if pending_text:
            if merged:
                prev_offset, prev_text = merged[-1]
                merged[-1] = (prev_offset, prev_text + "\n\n" + pending_text)
            else:
                merged.append((pending_offset, pending_text))

        return merged

    # ==============================================================
    # 合并 + 页码映射
    # ==============================================================
    def _pack_into_groups(
        self,
        docs: list[Document],
    ) -> list[tuple[str, list[tuple[int, int, dict[str, Any]]]]]:
        """
        贪心打包：把文档块按顺序装进若干个"箱子"，每个箱子装到接近 parent_size。

        返回：[ (箱子文本, 页码映射表), ... ]

        页码映射表是 [(起始位置, 结束位置, 元数据), ...]，
        用于后面反查"某个字符位置属于第几页"。

        【算法：贪心装箱（First Fit）】
        --------------------------------
        从头开始，一个接一个往当前箱子里放文档块：
          · 如果放进去还没超过 parent_size → 继续放
          · 如果放进去会超过 → 先把当前箱子封好，开一个新箱子放它

        为什么叫"贪心"？因为每次只考虑"眼前这个能不能塞下"，
        不做全局最优规划。它不保证箱子装得最满，但实现简单、效果好，
        是打包问题的经典解法。

        【一个重要的边界情况】
        如果**单个文档块本身就比 parent_size 大**（比如一份 5000 字的
        纯文本文件），那就让它独占一个箱子——后面 parent_splitter
        会负责把它切小。不能因为它太大就跳过，否则内容会丢失。

        【为什么不直接把所有文档合并成一大篇】
        见 split() 方法里那段注释：会把不相关的章节混在一起，
        导致检索质量崩塌。这是我在测试中真实踩到的坑。
        """
        groups: list[tuple[str, list[tuple[int, int, dict[str, Any]]]]] = []

        current_parts: list[str] = []
        current_map: list[tuple[int, int, dict[str, Any]]] = []
        current_len = 0
        separator = "\n\n"
        sep_len = len(separator)

        def flush() -> None:
            """把当前箱子封好，加入结果列表"""
            nonlocal current_parts, current_map, current_len
            if current_parts:
                groups.append((separator.join(current_parts), current_map))
            current_parts = []
            current_map = []
            current_len = 0

        for doc in docs:
            text = doc.page_content.strip()
            if not text:
                continue

            text_len = len(text)
            # 加上分隔符后的实际占用长度
            projected = current_len + (sep_len if current_parts else 0) + text_len

            # 箱子已非空，且再放就超了 → 先封箱
            if current_parts and projected > self.parent_size:
                flush()

            # 计算这个文档块在新位置上的起始偏移
            start = current_len + (sep_len if current_parts else 0)
            current_parts.append(text)
            current_map.append((start, start + text_len, dict(doc.metadata)))
            current_len = start + text_len

        flush()  # 最后一箱

        return groups

    @staticmethod
    def _lookup_page(
        position: int,
        page_map: list[tuple[int, int, dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        给定一个字符位置，查出它落在哪一页，返回那一页的元数据。

        如果位置超出了所有区间（理论上不会发生，但防御性处理），
        返回最后一页的元数据。
        """
        if not page_map:
            return {}

        # 二分查找：找到第一个"结束位置 > position"的区间
        lo, hi = 0, len(page_map) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if page_map[mid][1] <= position:
                lo = mid + 1
            else:
                hi = mid

        start, _end, meta = page_map[lo]

        result: dict[str, Any] = {"page": meta.get("page")}
        if meta.get("sheet_name"):
            result["sheet_name"] = meta["sheet_name"]
        if meta.get("section"):
            result["section"] = meta["section"]
        if meta.get("source"):
            result["source"] = meta["source"]
        if meta.get("title"):
            result["title"] = meta["title"]

        return result

    # ==============================================================
    # 质量检查
    # ==============================================================
    def analyze_chunks(self, chunks: list[ParentChildChunk]) -> dict:
        """
        分析切分质量。

        【关注哪个指标】
          1. 平均每父块含几个子块 —— 如果是 1.0，说明父子策略没生效
          2. 过小碎片比例 —— 如果偏高，说明分隔符或参数需要调整
        """
        if not chunks:
            return {"total": 0}

        child_lengths = [len(c.child_content) for c in chunks]
        parent_lengths = [len(c.parent_content) for c in chunks]
        parent_count = len({c.parent_index for c in chunks})

        tiny = sum(1 for n in child_lengths if n < self.child_size * 0.3)

        return {
            "total_chunks": len(chunks),
            "total_parents": parent_count,
            "child_avg_chars": round(sum(child_lengths) / len(child_lengths), 1),
            "child_min_chars": min(child_lengths),
            "child_max_chars": max(child_lengths),
            "parent_avg_chars": round(sum(parent_lengths) / len(parent_lengths), 1),
            "tiny_chunks": tiny,
            "tiny_ratio": f"{tiny / len(chunks):.1%}",
            "avg_children_per_parent": round(len(chunks) / max(1, parent_count), 2),
        }


# ==================================================================
# 便捷函数
# ==================================================================
def split_documents(docs: list[Document], doc_id: int | str = "") -> list[ParentChildChunk]:
    """切分文档的快捷方式"""
    return ParentChildSplitter().split(docs, doc_id=doc_id)
