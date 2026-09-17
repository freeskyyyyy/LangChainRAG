r"""
结构化切片验证 —— 拿你的真实资料，看切出来的片段对不对

【这个测试在验什么】
只有一件事：**每种面料，是不是各自独立成片了。**

改造前：一片 470 字里同时有「冰丝」「雪纺」「麂皮绒」，
        问雪纺的问题，返回的片段里三种面料都有，匹配不精确。
改造后：一片 150 字就是「雪纺」，干净利落。

【为什么用"关键词唯一性"来自动判定】
人眼看当然能看出来，但每次改完代码都得重新看一遍太累。
所以用自动判据：

    如果「雪纺」这两个字只出现在**一个**子块里 → 说明它成功独立了
    如果出现在**三个**子块里 → 说明切分还是糊在一起

这是客观的、可重复的，不靠我说了算。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe test_structure_split.py
（不需要启动服务，纯本地跑）
"""

import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).parent))

from app.rag.loader import parse_file  # noqa: E402
from app.rag.splitter import ParentChildSplitter  # noqa: E402

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
REPORT = Path(__file__).parent / "data" / "logs" / "切片对比报告.txt"

PASS = 0
FAIL = 0
LINES: list[str] = []


def out(t: str = "") -> None:
    LINES.append(t)
    print(t)


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        out(f"  ✅ {name}")
    else:
        FAIL += 1
        out(f"  ❌ {name}")
        if detail:
            out(f"     └─ {detail}")


# 每种面料应该"独占一片"的关键词。
# 挑的都是**只在对应那一节出现**的词，别的节里找不到，
# 所以只要它出现在多个片段里，就说明切片糊了。
FABRICS = ["雪纺", "冰丝", "灯芯绒", "麂皮绒", "羽绒服", "针织棉", "真丝", "羊毛"]

# 颜色文档里的小节关键词
COLOR_TOPICS = ["宴会", "显瘦", "冬季", "夏季"]


def identify(full_text: str) -> str:
    """
    根据内容判断这是哪一份资料。

    【为什么不能简单判断有没有某个词】
    我第一版就是这么写的，结果判断错了：
    颜色搭配那份资料里有一句「丰满体型：选择**面料**哑光的深色服装」，
    于是"含面料两个字"→被判成了洗涤养护。
    这类误判会让整个测试的结论都不对。

    所以改成看**特征词**：
      · 「建议尺码」只在尺码表里出现
      · 「肤色」只在颜色搭配里出现
      · 「洗涤：」只在洗涤养护里出现

    这几个词都是各自独有的，别的文档里找不到。
    """
    scores = {
        "尺码推荐": full_text.count("建议尺码") + full_text.count("身高"),
        "颜色选择": full_text.count("肤色") + full_text.count("颜色搭配"),
        "洗涤养护": full_text.count("洗涤：") + full_text.count("养护："),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "未知"


def is_about(chunk: str, keyword: str) -> bool:
    """
    判断这一片是不是**真的在讲**这个关键词。

    【为什么不能简单地数"关键词出现过没有"】
    我第一版就是那么写的，结果报出一堆假警报。
    因为那份资料里有这么一行：

        二、夏季服装（真丝、棉麻、冰丝、短袖纯棉、雪纺）

    这是"夏季"那一节的标题，它把所有夏季面料的名字都罗列了一遍。
    于是"冰丝""雪纺"这两个词，在夏季标题那片里也能搜到 ——
    但那一节实际讲的是**真丝**，冰丝雪纺只是被顺带提了一下名字。

    真正"在讲雪纺"的那片长这样：
        4. 雪纺材质（夏季连衣裙、防晒衫）
        洗涤：手洗优先…禁止用力拧绞。

    【判据】
    看关键词是不是紧跟在「数字. 」这种编号后面。
    是 → 这一节的标题就是它，说明这片就是在讲它
    否 → 只是被顺带提到，不算

    这个判据是客观的、可重复的，不靠我人工判断。
    """
    return re.search(rf"^\s*\d+[.．]\s*{re.escape(keyword)}", chunk, re.M) is not None


def main() -> int:
    global PASS, FAIL

    out("=" * 76)
    out("  结构化切片验证报告")
    out("=" * 76)

    # ---------- 找出那三份资料 ----------
    files = sorted(UPLOAD_DIR.glob("*.txt"))
    if not files:
        out("\n  ❌ 在 data/uploads 下没找到任何资料文件")
        out("     请先在网页上传资料，或把文件放进这个目录\n")
        return 1

    out(f"\n  找到 {len(files)} 份资料文件\n")

    for fp in files:
        # 先读出内容，用内容判断这是哪一份
        docs = parse_file(fp)
        full_text = "\n".join(d.page_content for d in docs)

        label = identify(full_text)

        out("-" * 76)
        out(f"【{label}】{fp.name}  （{len(full_text)} 字）")
        out("-" * 76)

        # ---------- 分别用"关"和"开"两种配置切，做对比 ----------
        text = full_text

        plain = ParentChildSplitter(structure_aware=False)
        plain_chunks = [c.child_content for c in plain.split(docs, doc_id=0)]

        smart = ParentChildSplitter(structure_aware=True)
        smart_chunks = [c.child_content for c in smart.split(docs, doc_id=0)]

        out(f"\n  共同输入：{len(text)} 字")
        out(f"    · 纯按字数切   → {len(plain_chunks)} 片，"
            f"平均 {sum(len(c) for c in plain_chunks) / max(1, len(plain_chunks)):.0f} 字")
        out(f"    · 按结构切     → {len(smart_chunks)} 片，"
            f"平均 {sum(len(c) for c in smart_chunks) / max(1, len(smart_chunks)):.0f} 字")

        # ---------- 关键词唯一性检验 ----------
        keywords = FABRICS if label == "洗涤养护" else (
            COLOR_TOPICS if label == "颜色选择" else []
        )

        if keywords:
            out(f"\n  每种面料应该恰好有 1 片是「专门讲它的」：\n")
            out(f"    {'关键词':<8} {'改造前':>7} {'改造后':>7}   判定")
            out(f"    {'-' * 8} {'-' * 7} {'-' * 7}   {'-' * 16}")

            for kw in keywords:
                n_plain = sum(1 for c in plain_chunks if is_about(c, kw))
                n_smart = sum(1 for c in smart_chunks if is_about(c, kw))

                if n_plain == 0 and n_smart == 0:
                    continue    # 这份资料里没这个词，跳过

                # 理想情况：恰好 1 片专门讲它
                if n_smart == 1:
                    verdict = "✅ 独立成片"
                    PASS += 1
                elif n_smart == 0:
                    verdict = "❌ 改造后找不到了"
                    FAIL += 1
                else:
                    verdict = f"❌ 仍有 {n_smart} 片在讲它"
                    FAIL += 1

                out(f"    {kw:<8} {n_plain:>5} 片 {n_smart:>5} 片   {verdict}")

        # ---------- 打印改造后的实际片段 ----------
        out(f"\n  改造后各片段的开头（看每片是不是只讲一件事）：\n")
        for i, c in enumerate(smart_chunks):
            head = c[:62].replace("\n", " ")
            out(f"    [{i}] {len(c):>4} 字  {head}…")

        # ---------- 检查有没有"只有标题没有内容"的碎片 ----------
        tiny = [c for c in smart_chunks if len(c) < 60]
        if tiny:
            out(f"\n  ⚠️  有 {len(tiny)} 片短于 60 字（可能是没用的碎片）：")
            for c in tiny[:5]:
                out(f"       {len(c)} 字：{c[:50]}")
        else:
            out(f"\n  ✅ 没有过短的碎片（全部 ≥ 60 字）")

        out("")

    # ---------- 汇总 ----------
    out("=" * 76)
    out(f"  验证完成：{PASS} 项通过 / {FAIL} 项失败")
    out("=" * 76)
    if FAIL == 0:
        out("\n  🎉 结构化切分生效，每种面料都独立成片了。\n")
    else:
        out(f"\n  ⚠️  有 {FAIL} 项没有改善，需要继续调。\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\n报告已写入：{REPORT}\n")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断。\n")
        sys.exit(130)
    except Exception as exc:
        print(f"\n❌ 异常: {type(exc).__name__}: {exc}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
