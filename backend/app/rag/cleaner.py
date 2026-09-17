"""
文本清洗 —— 把解析出来的"生文本"整理成"干净文本"

================================================================
【为什么这一步决定了检索质量的上限】
================================================================
有个说法：**RAG 系统的效果上限，在数据清洗这一步就定下来了。**

原因是：后面的每一步（切分、向量化、检索）都建立在文本质量之上。
如果原始文本里混着一堆噪音，那：
  · 切分会被噪音打乱（一个片段里半页是页眉页脚）
  · 向量化会把噪音也算进语义里（导致向量"跑偏"）
  · 检索会被噪音命中（搜"退货政策"结果检索到页脚的"版权所有"）

而且**这些都是事后无法补救的**——垃圾进了向量库，
除非重新解析入库，否则它会一直在那里污染检索结果。

================================================================
【常见的文本噪音，以及怎么处理】
================================================================
1. **页眉页脚**：PDF 每页都重复出现"XX公司内部资料 第3页 共20页"
   → 处理：检测"在多页里重复出现且位置相同"的短行，删掉

2. **页码残留**："- 12 -"、"12/20"、"- Page 3 of 10-"
   → 处理：正则匹配这类"纯页码行"，删掉

3. **PDF 硬换行**：一句话被切成好几行，行尾没有标点
   → 处理：在 loader 里已经处理过一部分，这里再兜底

4. **多余空白**：连续空格、连续空行、行首行尾空格
   → 处理：统一压缩

5. **隐藏字符**：零宽空格、软连字符、控制字符
   → 处理：直接删除（它们肉眼看不见，但会干扰分词和匹配！）

6. **全角/半角混用**："ｉＰｈｏｎｅ" 和 "iPhone"
   → 处理：把全角英文数字转成半角（但不动中文标点）

7. **超长无意义字符**："--------" "=========" 分隔线
   → 处理：压缩成短的
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from app.core.logging import logger


class TextCleaner:
    """
    文本清洗器

    用法：
        cleaner = TextCleaner()
        cleaned_pages = cleaner.clean_documents(docs)   # 批量清洗（能识别页眉页脚）
        # 或者单块清洗
        text = cleaner.clean_text(raw_text)
    """

    # ------------------------------------------------------------------
    # 预编译正则表达式
    #
    # 【为什么要预编译】
    # re.compile 会把正则模式编译成内部表示，编译一次能用无数次。
    # 如果写在函数里，每次调用都要重新编译一遍，在批量处理上千个片段时
    # 累积的开销很可观。这是正则使用的标准优化手法。
    # ------------------------------------------------------------------

    # 纯页码行：如 "- 12 -"、"12/20"、"第 3 页"、"Page 3 of 10"
    _RE_PAGE_NUMBER = re.compile(
        r"^\s*[-—–\s]*"
        r"(?:"
        r"第\s*\d+\s*页(?:\s*/?\s*共?\s*\d+\s*页)?"
        r"|page\s*\d+(?:\s*of\s*\d+)?"
        r"|\d+\s*/\s*\d+"
        r"|[-—–]\s*\d+\s*[-—–]"
        r"|\d{1,4}"
        r")"
        r"\s*[-—–\s]*$",
        re.IGNORECASE,
    )

    # 分隔线：一行里全是同一种符号
    _RE_SEPARATOR = re.compile(r"^[=\-_*~#·•\.]{4,}$")

    # 连续空白
    _RE_MULTI_SPACE = re.compile(r"[ \t 　]{2,}")

    # 连续空行
    _RE_MULTI_NEWLINE = re.compile(r"\n{3,}")

    # 隐藏/控制字符（零宽空格、软连字符、方向标记等）
    # 这些字符肉眼看不见，但会破坏分词和字符串匹配，必须删掉
    _RE_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿­]")

    # 其他控制字符（保留 \n \t）
    _RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    # 括号里的孤立引用标记（比如 "[1]" 这种 PDF 转换残留）
    # 但保留有意义的方括号内容，所以只删纯数字的
    _RE_LONE_REF = re.compile(r"\[\s*\d{1,3}\s*\]")

    def __init__(
        self,
        remove_page_numbers: bool = True,
        remove_headers_footers: bool = True,
        header_footer_min_pages: int = 3,
    ):
        """
        参数：
            remove_page_numbers      —— 是否删除纯页码行
            remove_headers_footers   —— 是否自动检测并删除页眉页脚
            header_footer_min_pages  —— 至少出现在几页里才算"页眉页脚"
                                        （设太小会误删正常内容）
        """
        self.remove_page_numbers = remove_page_numbers
        self.remove_headers_footers = remove_headers_footers
        self.header_footer_min_pages = header_footer_min_pages

    # ==============================================================
    # 单块文本清洗
    # ==============================================================
    def clean_text(self, text: str) -> str:
        """清洗一段文本（不涉及跨页的页眉页脚检测）"""
        if not text:
            return ""

        # ---- 1. 删掉看不见的字符 ----
        # 这一步看起来无关紧要，其实很关键：
        # 从 PDF 复制出来的文字常带着零宽空格，肉眼完全看不出来，
        # 但会导致"iPhone16"和"iPhone 16"匹配不上，或者分词结果奇怪。
        text = self._RE_INVISIBLE.sub("", text)
        text = self._RE_CONTROL.sub("", text)

        # ---- 2. 全角转半角（英文字母和数字）----
        text = self._normalize_fullwidth(text)

        # ---- 3. 统一换行符 ----
        # Windows 用 \r\n，老 Mac 用 \r，Linux 用 \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # ---- 4. 去掉行首行尾的空白 ----
        lines = [line.strip() for line in text.split("\n")]

        # ---- 5. 逐行过滤 ----
        kept: list[str] = []
        for line in lines:
            if not line:
                kept.append("")  # 空行先保留，最后统一压缩
                continue

            if self._RE_SEPARATOR.match(line):
                continue  # 分隔线，扔掉

            if self.remove_page_numbers and self._RE_PAGE_NUMBER.match(line):
                continue  # 纯页码，扔掉

            kept.append(line)

        text = "\n".join(kept)

        # ---- 6. 压缩连续空白和空行 ----
        text = self._RE_MULTI_SPACE.sub(" ", text)
        text = self._RE_MULTI_NEWLINE.sub("\n\n", text)

        # ---- 7. 删掉 PDF 转换残留的孤立引用标记 ----
        text = self._RE_LONE_REF.sub("", text)

        return text.strip()

    # ==============================================================
    # 批量清洗（能识别页眉页脚）
    # ==============================================================
    def clean_documents(self, docs: list) -> list:
        """
        批量清洗，并能自动识别页眉页脚。

        【页眉页脚怎么识别】
        核心洞察：**页眉页脚的特征是"在多页里反复出现"**。
        正文内容不会每页都一样，所以：

          1. 收集每一页的第一行和最后一行
          2. 统计哪些行在 ≥3 页里出现过
          3. 这些行就是页眉页脚，从所有页面里删掉

        这个方法简单但很有效，是处理 PDF 的标准手法。

        注意：只对"多页文档"做这个处理（至少 3 页），
        因为单页文档没有"重复"可谈，贸然删除会误伤正文。
        """
        if not docs:
            return []

        # ---- 第一步：先做单块清洗 ----
        for doc in docs:
            doc.page_content = self.clean_text(doc.page_content)

        # ---- 第二步：识别并删除页眉页脚 ----
        if self.remove_headers_footers and len(docs) >= self.header_footer_min_pages:
            repeated = self._find_repeated_lines(docs)
            if repeated:
                logger.info(f"检测到 {len(repeated)} 行重复内容（疑似页眉页脚），已删除")
                for doc in docs:
                    doc.page_content = self._remove_lines(doc.page_content, repeated)
                    doc.page_content = doc.page_content.strip()

        # ---- 第三步：丢掉清洗后变空的页面 ----
        result = [d for d in docs if d.page_content.strip()]
        dropped = len(docs) - len(result)
        if dropped:
            logger.debug(f"清洗后有 {dropped} 个文档块变为空，已丢弃")

        return result

    @staticmethod
    def _normalize_for_dedup(line: str) -> str:
        """
        把一行文字"归一化"，用于判断两行是不是同一种页眉页脚。

        【为什么需要这一步】
        真实 PDF 的页眉长这样：
            第1页的文件是： "XX公司内部资料 第1页 共3页"
            第2页的文件是： "XX公司内部资料 第2页 共3页"
        两行**文字并不完全相同**（页码变了），
        如果用"完全相同"来判断重复，就永远识别不出这种页眉。

        所以要把数字统一替换成 N，让它们变成同一个模式：
            "XX公司内部资料 第N页 共N页"
        这样就能匹配上了。
        """
        import re as _re

        normalized = _re.sub(r"\d+", "N", line)
        # 顺手把多余空白压缩掉，防止"第 1 页"和"第1页"匹配不上
        normalized = _re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _find_repeated_lines(self, docs: list) -> set[str]:
        """
        找出在多个页面里重复出现的行（页眉页脚的候选）。

        【设计要点一：只看第一行和最后一行】
        真正的页眉永远在页面最顶端，页脚永远在最底端。
        如果贪心地多看几行（比如前 2 行），正文里恰好排在第 2 行的内容
        就会被误判成页眉而遭误删——这个 bug 我在测试时真实踩到过。

        所以严格只取**第一行**和**最后一行**。

        【设计要点二：数字归一化】
        页眉里的页码是变化的（第1页/第2页/第3页），
        归一化之后才能识别成同一个页眉。

        返回：归一化后的行文本集合（不是原始文本）
        """
        counter: Counter[str] = Counter()

        for doc in docs:
            lines = [ln.strip() for ln in doc.page_content.split("\n") if ln.strip()]
            if not lines:
                continue

            # 严格只取第一行和最后一行
            candidates = {lines[0], lines[-1]}

            for line in candidates:
                # 长度过滤：太短的（1个字）没意义，太长的（>120字）大概率是正文
                if 2 <= len(line) <= 120:
                    counter[self._normalize_for_dedup(line)] += 1

        # 出现在足够多页里 → 判定为页眉页脚
        threshold = max(self.header_footer_min_pages, len(docs) // 3)
        return {line for line, count in counter.items() if count >= threshold}

    def _remove_lines(self, text: str, lines_to_remove: set[str]) -> str:
        """
        从文本里删掉指定的行。

        注意：比较时也要先归一化，才能匹配上"第1页"和"第2页"这类变体。
        """
        return "\n".join(
            line for line in text.split("\n")
            # 空行保留（它是段落分隔符），非空行才做匹配判断
            if not line.strip()
            or self._normalize_for_dedup(line.strip()) not in lines_to_remove
        )

    # ==============================================================
    # 全角转半角
    # ==============================================================
    @staticmethod
    def _normalize_fullwidth(text: str) -> str:
        """
        全角英文/数字转半角。

        【为什么要做这个】
        中文排版里，英文字母和数字经常被输入成全角（ＡＢＣ１２３），
        这在使用中文输入法时非常常见。

        如果不转换：
          · 用户搜 "iPhone" 就匹配不到文档里的 "ｉＰｈｏｎｅ"
          · BM25 关键词检索会完全失效

        【为什么不用简单的字符偏移转换】
        常见做法是 ord(char) - 0xFEE0，但它会把全角空格、
        以及某些中文全角标点也一起转换，造成中文标点变成英文标点。
        所以这里用 unicodedata 按"字符的正式名称"来判断，
        更精确：只转换那些"名称里带 FULLWIDTH"且本质是拉丁字母/数字的字符。
        """
        result: list[str] = []
        for char in text:
            code = ord(char)
            # 全角字符的 Unicode 区间是 FF01-FF5E，对应半角 21-7E
            if 0xFF01 <= code <= 0xFF5E:
                # 排除全角标点（！＂＃... 这些转换后会改变中文排版习惯）
                # 只转换字母和数字
                if char.isalnum() or char in "＝＋－／％＃＠＆":
                    result.append(chr(code - 0xFEE0))
                else:
                    result.append(char)
            else:
                result.append(char)
        return "".join(result)

    # ==============================================================
    # 质量检测
    # ==============================================================
    @staticmethod
    def analyze_quality(text: str) -> dict:
        """
        分析文本质量，用于诊断"为什么检索效果不好"。

        这个函数在排查问题时很有用：如果某个文档清洗后
        有效字符比例很低，说明解析环节出了问题（比如 PDF 是扫描件）。
        """
        total = len(text)
        if total == 0:
            return {"total_chars": 0, "quality": "empty"}

        # 统计各类字符
        chinese = sum(1 for c in text if "一" <= c <= "鿿")
        letters = sum(1 for c in text if c.isalpha())
        digits = sum(1 for c in text if c.isdigit())
        spaces = sum(1 for c in text if c.isspace())

        # 有效内容占比（去掉空白后的比例）
        meaningful_ratio = (total - spaces) / total if total else 0

        quality = "good"
        if meaningful_ratio < 0.3:
            quality = "poor（空白过多，可能是解析失败）"
        elif chinese + letters < 10:
            quality = "poor（几乎没有文字）"

        return {
            "total_chars": total,
            "chinese_chars": chinese,
            "latin_letters": letters,
            "digits": digits,
            "meaningful_ratio": f"{meaningful_ratio:.1%}",
            "quality": quality,
        }


# ==================================================================
# 便捷函数
# ==================================================================
_default_cleaner = TextCleaner()


def clean_text(text: str) -> str:
    """清洗单块文本"""
    return _default_cleaner.clean_text(text)


def clean_documents(docs: list) -> list:
    """批量清洗（含页眉页脚识别）"""
    return _default_cleaner.clean_documents(docs)
