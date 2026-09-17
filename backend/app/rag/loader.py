"""
多格式文档解析器 —— 把各种格式的文件读成"文本 + 元数据"

================================================================
【为什么这一步要单独写一个模块】
================================================================
不同格式的文件，读法完全不同：
  · PDF   ：要一页一页读，还要处理换行截断问题
  · Word  ：段落和表格是两种不同的结构
  · Excel ：是二维表格，得转成文字才看得懂
  · HTML  ：全是标签，得扒掉
  · TXT   ：直接读，但要猜编码（GBK 还是 UTF-8）

但后面的所有步骤（清洗、切分、向量化）都希望拿到**同一种东西**——
LangChain 的 Document 对象：

    Document(
        page_content = "iPhone 16 的电池容量为 3561mAh...",   ← 文本
        metadata     = {"page": 3, "source": "说明书.pdf"}    ← 元数据
    )

所以这一步的核心任务就是：**把五花八门的格式，统一成这一个形状**。
统一之后，后面的流水线就不用关心原始文件是什么格式了。

【为什么元数据很重要】
元数据里的 page（页码）、sheet_name（工作表名）这些，
最终会出现在**引用卡片**上，告诉用户"这句话出自第 3 页"。
没有它，回答就无法溯源——那就退化成普通聊天机器人了。
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.core.logging import logger

# ==================================================================
# 支持的格式
# ==================================================================
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".csv", ".md", ".txt", ".html", ".htm",
}


class UnsupportedFormatError(Exception):
    """不支持的文件格式"""


class DocumentParser:
    """
    文档解析器

    用法：
        docs = await asyncio.to_thread(DocumentParser.parse, file_path)
    注意解析是 CPU 密集型的同步操作，在异步环境里要用 to_thread 包一层，
    否则会阻塞整个事件循环（导致别人的请求卡住）。
    """

    @staticmethod
    def parse(file_path: str | Path) -> list[Document]:
        """
        解析文件，返回 Document 列表。

        入口函数：按扩展名分发到具体的解析器。
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"不支持的文件格式「{ext}」，当前支持：{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # 分发表：扩展名 → 处理函数
        handlers = {
            ".pdf": DocumentParser._parse_pdf,
            ".docx": DocumentParser._parse_docx,
            ".doc": DocumentParser._parse_docx,
            ".xlsx": DocumentParser._parse_excel,
            ".xls": DocumentParser._parse_excel,
            ".csv": DocumentParser._parse_csv,
            ".md": DocumentParser._parse_text,
            ".txt": DocumentParser._parse_text,
            ".html": DocumentParser._parse_html,
            ".htm": DocumentParser._parse_html,
        }

        handler = handlers[ext]
        docs = handler(path)

        # 给每个文档补上统一的元数据
        for d in docs:
            d.metadata.setdefault("source", path.name)
            d.metadata["file_type"] = ext.lstrip(".")

        logger.info(f"解析完成 | 文件={path.name} | 格式={ext} | 产出 {len(docs)} 个文档块")
        return docs

    # ==============================================================
    # PDF
    # ==============================================================
    @staticmethod
    def _parse_pdf(path: Path) -> list[Document]:
        """
        解析 PDF，一页产出一个 Document。

        【PDF 解析的两个老大难】
        1. **换行截断**：PDF 是按"文字在页面上的坐标"存储的，
           不是按段落。所以一句话经常被硬生生切成好几行：
               "iPhone 16 的电池
                容量为 3561mAh"
           直接拼起来会变成"电池  容量"，中间多个空格。
           解决办法：如果一个行尾不是句号/问号/分号等结束符，
           就和下一行直接拼接（不加空格）。

        2. **扫描件**：如果 PDF 是扫描出来的图片，里面根本没有文字层，
           提取出来会是空白。这种情况需要 OCR 技术，本项目不支持，
           但我们会**检测并明确报错**，而不是悄悄产出一个空文档。
        """
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        docs: list[Document] = []
        empty_pages = 0

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"PDF 第 {page_num} 页提取失败: {e}")
                continue

            text = DocumentParser._fix_pdf_linebreaks(text)

            if not text.strip():
                empty_pages += 1
                continue

            docs.append(
                Document(
                    page_content=text.strip(),
                    metadata={"page": page_num, "total_pages": len(reader.pages)},
                )
            )

        if not docs:
            if empty_pages:
                raise ValueError(
                    f"这份 PDF 的 {empty_pages} 页全是空白。"
                    "如果它是扫描件（图片版 PDF），本项目无法直接读取文字——"
                    "需要先用 OCR 工具转成文字版 PDF。"
                )
            raise ValueError("这份 PDF 里没有提取到任何文字内容")

        return docs

    @staticmethod
    def _fix_pdf_linebreaks(text: str) -> str:
        """
        修复 PDF 提取出来的换行截断问题。

        规则：如果一行结尾不是"句子结束符"，说明这句话还没说完，
             就把换行去掉、直接和下一行拼接。
        """
        if not text:
            return text

        # 这些符号表示"一句话说完了"，后面该换行
        enders = "。！？；：.!?;:）)】」』\"'"
        lines = text.split("\n")
        merged: list[str] = []

        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                merged.append("")  # 空行保留（段落分隔）
                continue

            if merged and merged[-1] and merged[-1][-1] not in enders:
                # 上一行没说完 → 直接拼接，不加空格（中文不需要）
                merged[-1] += stripped.lstrip()
            else:
                merged.append(stripped)

        # 清理多余空行
        result = "\n".join(merged)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result

    # ==============================================================
    # Word (.docx)
    # ==============================================================
    @staticmethod
    def _parse_docx(path: Path) -> list[Document]:
        """
        解析 Word 文档。

        【为什么段落和表格要分开处理】
        Word 文档里，段落（段落1、段落2...）和表格是两套独立的结构。
        如果不单独处理表格，表格内容会被完全丢掉——
        而商品资料里的"规格参数表"恰恰是最有价值的部分！

        所以这里：段落按顺序读，表格转成"键 | 值"的文字格式插进去。
        """
        from docx import Document as DocxDocument

        doc = DocxDocument(str(path))
        parts: list[str] = []

        # ---- 读段落 ----
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # 保留标题层级信息（用 Markdown 的 # 表示），
            # 这样切分和检索时能看出内容的层次结构
            style = (para.style.name or "").lower()
            if "heading 1" in style or "标题 1" in style:
                parts.append(f"# {text}")
            elif "heading 2" in style or "标题 2" in style:
                parts.append(f"## {text}")
            elif "heading 3" in style or "标题 3" in style:
                parts.append(f"### {text}")
            else:
                parts.append(text)

        # ---- 读表格 ----
        for table_idx, table in enumerate(doc.tables, start=1):
            rows: list[str] = []
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                # 跳过全空行
                if any(cells):
                    rows.append(" | ".join(cells))

            if rows:
                parts.append(f"\n【表格 {table_idx}】")
                parts.extend(rows)

        if not parts:
            raise ValueError("这份 Word 文档里没有提取到任何文字内容")

        # Word 没有可靠的分页信息（分页是渲染时才确定的），
        # 所以整体作为一个文档块，页码记为 None
        return [Document(page_content="\n".join(parts), metadata={"page": None})]

    # ==============================================================
    # Excel (.xlsx)
    # ==============================================================
    @staticmethod
    def _parse_excel(path: Path) -> list[Document]:
        """
        解析 Excel，**每个工作表产出一个 Document**。

        【为什么要转成文字而不是原样保留表格】
        表格本身没法做向量检索（向量化的是文字）。
        所以要把每一行转成类似这样的句子：

            商品名称: iPhone 16 Pro | 屏幕尺寸: 6.3 英寸 | 电池: 3561mAh

        这样向量化之后，"iPhone 16 的屏幕多大"才能检索到这一行。

        【为什么要保留表头】
        数据行本身是一堆值，脱离表头就没有意义。
        "3561mAh" 单独看不知道是什么，"电池容量: 3561mAh" 才完整。
        所以每一行都要带上列名。
        """
        import openpyxl

        wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
        docs: list[Document] = []

        for sheet in wb.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue

            # 第一行当表头
            header_row = rows[0]
            headers = [
                str(h).strip() if h is not None else f"列{i + 1}"
                for i, h in enumerate(header_row)
            ]

            lines: list[str] = [f"【工作表：{sheet.title}】"]
            line_count = 0

            for row in rows[1:]:
                # 跳过全空行
                if not any(cell is not None and str(cell).strip() for cell in row):
                    continue

                pairs = []
                for i, cell in enumerate(row):
                    if cell is None or not str(cell).strip():
                        continue
                    col_name = headers[i] if i < len(headers) else f"列{i + 1}"
                    value = str(cell).strip().replace("\n", " ")
                    pairs.append(f"{col_name}: {value}")

                if pairs:
                    lines.append(" | ".join(pairs))
                    line_count += 1

            if line_count == 0:
                continue  # 这个工作表只有表头没有数据，跳过

            docs.append(
                Document(
                    page_content="\n".join(lines),
                    metadata={"page": None, "sheet_name": sheet.title},
                )
            )

        wb.close()

        if not docs:
            raise ValueError("这份 Excel 里没有提取到任何数据行")

        return docs

    # ==============================================================
    # CSV
    # ==============================================================
    @staticmethod
    def _parse_csv(path: Path) -> list[Document]:
        """
        解析 CSV。

        【编码问题】
        CSV 文件在中文 Windows 上经常是 GBK 编码，而在 Mac/Linux 上是 UTF-8。
        用错编码会读出乱码。所以这里按顺序试几种常见编码，
        哪个能成功解码就用哪个（这叫"编码嗅探"）。
        """
        raw = path.read_bytes()
        text: str | None = None
        used_encoding = ""

        for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"):
            try:
                text = raw.decode(encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ValueError("无法识别这个 CSV 文件的编码格式")

        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if any(cell.strip() for cell in row)]

        if not rows:
            raise ValueError("这个 CSV 文件是空的")

        headers = [h.strip() or f"列{i + 1}" for i, h in enumerate(rows[0])]
        lines: list[str] = []

        for row in rows[1:]:
            pairs = [
                f"{headers[i] if i < len(headers) else f'列{i + 1}'}: {cell.strip()}"
                for i, cell in enumerate(row)
                if cell.strip()
            ]
            if pairs:
                lines.append(" | ".join(pairs))

        if not lines:
            raise ValueError("这个 CSV 只有表头，没有数据行")

        logger.debug(f"CSV 编码识别为 {used_encoding}")
        return [
            Document(
                page_content="\n".join(lines),
                metadata={"page": None, "encoding": used_encoding},
            )
        ]

    # ==============================================================
    # Markdown / 纯文本
    # ==============================================================
    @staticmethod
    def _parse_text(path: Path) -> list[Document]:
        """
        解析 Markdown 或纯文本。

        Markdown 有个额外的好处：它的标题层级（# ## ###）能反映文档结构。
        我们利用这一点，**按一级/二级标题把文档切成多个块**，
        这样每一块的主题都比较单一，检索时更精准。
        """
        raw = path.read_bytes()
        text: str | None = None
        for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None or not text.strip():
            raise ValueError("这个文件是空的，或者编码无法识别")

        # Markdown 按标题切分；纯文本就整块返回
        if path.suffix.lower() == ".md":
            sections = DocumentParser._split_markdown_by_heading(text)
            if len(sections) > 1:
                return sections
            return [Document(page_content=text, metadata={"page": None})]

        return [Document(page_content=text, metadata={"page": None})]

    @staticmethod
    def _split_markdown_by_heading(text: str) -> list[Document]:
        """按 Markdown 的一级/二级标题把文档切成若干块"""
        lines = text.split("\n")
        sections: list[Document] = []
        current: list[str] = []
        current_title = ""

        for line in lines:
            # 匹配 # 或 ## 开头的标题行
            if re.match(r"^#{1,2}\s+\S", line):
                if current:
                    content = "\n".join(current).strip()
                    if content:
                        sections.append(
                            Document(
                                page_content=content,
                                metadata={"page": None, "section": current_title},
                            )
                        )
                current = [line]
                current_title = line.lstrip("# ").strip()
            else:
                current.append(line)

        # 最后一块
        if current:
            content = "\n".join(current).strip()
            if content:
                sections.append(
                    Document(
                        page_content=content,
                        metadata={"page": None, "section": current_title},
                    )
                )

        return sections

    # ==============================================================
    # HTML
    # ==============================================================
    @staticmethod
    def _parse_html(path: Path) -> list[Document]:
        """
        解析 HTML（比如保存下来的商品详情页）。

        【关键：只留正文，扔掉噪音】
        网页里 90% 的内容是没用的：导航栏、广告、页脚、脚本、样式。
        如果不去掉，它们会被向量化并进入检索，污染结果
        （比如搜"退货政策"，可能检索到导航栏里的"退货"两个字）。

        所以这里要主动删掉 script、style、nav、footer、header 这些标签，
        并优先提取 <title> 和正文区域。
        """
        from bs4 import BeautifulSoup

        raw = path.read_bytes()
        text: str | None = None
        for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ValueError("HTML 文件编码无法识别")

        soup = BeautifulSoup(text, "html.parser")

        # 先删掉噪音标签
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
            tag.decompose()

        # 标题单独提出来
        title = soup.title.get_text(strip=True) if soup.title else ""

        # 正文：优先找语义标签，找不到就取整个 body
        body = soup.find("article") or soup.find("main") or soup.body or soup
        content = body.get_text(separator="\n", strip=True)

        # 清理多余空行
        content = re.sub(r"\n{3,}", "\n\n", content)

        if not content.strip():
            raise ValueError("这个网页里没有提取到正文内容")

        prefix = f"【网页标题】{title}\n\n" if title else ""
        return [Document(page_content=prefix + content, metadata={"page": None, "title": title})]


# ==================================================================
# 便捷函数
# ==================================================================
def parse_file(file_path: str | Path) -> list[Document]:
    """解析单个文件的快捷方式"""
    return DocumentParser.parse(file_path)


def is_supported(file_name: str) -> bool:
    """判断文件格式是否支持"""
    return Path(file_name).suffix.lower() in SUPPORTED_EXTENSIONS


def get_parser_info() -> dict[str, Any]:
    """返回支持的格式信息（管理页面展示用）"""
    return {
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "count": len(SUPPORTED_EXTENSIONS),
        "format_map": {
            "pdf": "PDF 文档（按页解析）",
            "docx": "Word 文档（段落 + 表格）",
            "xlsx": "Excel 表格（按工作表解析）",
            "csv": "CSV 表格",
            "md": "Markdown（按标题分块）",
            "txt": "纯文本",
            "html": "网页（自动去广告和导航）",
        },
    }
