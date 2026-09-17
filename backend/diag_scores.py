r"""
分数分布诊断 —— 用实测数据来决定"相关性阈值"该设多少

【为什么要做这个诊断】
我一开始把阈值定成 0.3，是凭"向量相似度的直觉"拍脑袋定的。
结果发现：重排序模型的分数量纲完全不一样——
换了说法的问题（"这个手机电池能用多久"）只有 0.20 分，
而它其实是**完全正确的匹配**。阈值 0.3 把正确答案拦掉了。

正确做法不是继续猜，而是**实测**：
  · 拿一批"知识库里确实有答案"的问题，记下它们的分数
  · 再拿一批"知识库里根本没有"的问题，记下它们的分数
  · 两组分数之间必然有一条分界线，阈值就设在那条线上

这个脚本就是干这件事的。它只做检索，不调用大模型生成，
所以几乎不消耗额度（只花向量化的钱）。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe diag_scores.py
"""

import asyncio
import statistics
import sys

sys.path.insert(0, ".")

from app.core.database import AsyncSessionLocal, init_db  # noqa: E402
from app.models import Document  # noqa: E402
from app.rag.ingest import get_ingest_service  # noqa: E402
from app.rag.retriever import HybridRetriever  # noqa: E402
from test_corpus import CORPUS  # noqa: E402

LINE = "=" * 78

# ------------------------------------------------------------------
# 测试问题分两类
# ------------------------------------------------------------------

# 第一类：知识库里**确实有**答案的（应该得高分）
ANSWERABLE = [
    ("这个手机电池能用多久", "电池续航"),
    ("电池容量是多少", "电池续航"),
    ("包装里有充电器吗", "电池续航"),
    ("笔记本有多重", "笔记本产品"),
    ("屏幕刷新率是多少", "笔记本产品"),
    ("支持退货吗", "售后政策"),
    ("保修期多久", "售后政策"),
    ("可以用花呗付款吗", "支付说明"),
    ("发票怎么开", "支付说明"),
    ("发货一般几天到", "物流说明"),
    ("运费怎么算", "物流说明"),
    ("这手机能防水吗", "商品参数"),
]

# 第二类：知识库里**完全没有**的（应该得低分）
UNANSWERABLE = [
    "你们公司总部在哪栋楼办公",
    "有没有卖冰箱",
    "老板的电话是多少",
    "支持货到付款吗",
    "可以开发票吗怎么抵扣增值税",
    "今天天气怎么样",
]


async def main() -> int:
    await init_db()

    print(f"\n{LINE}")
    print("  相关性分数分布诊断".center(66))
    print(f"{LINE}")

    ingest = get_ingest_service()
    doc_ids: list[int] = []

    # ---------------- 入库 ----------------
    print(f"\n  正在入库测试语料…")
    async with AsyncSessionLocal() as db:
        for section_name, text in CORPUS:
            result = await ingest.ingest(
                db, text.strip().encode("utf-8"), f"{section_name}.md"
            )
            if result.success:
                doc_ids.append(result.doc_id)

    print(f"  ✅ {len(doc_ids)} 份文档已入库\n")

    retriever = HybridRetriever()

    # ==============================================================
    # 测第一类：应该有答案的
    # ==============================================================
    print(f"{'-' * 78}")
    print("  第一类：知识库里【有】答案的问题（期望得高分）")
    print(f"{'-' * 78}\n")
    print(f"  {'问题':<22} {'向量分':>8} {'重排分':>8}  实际检索到的文件")
    print("  " + "-" * 72)

    answerable_scores: list[float] = []
    vector_scores: list[float] = []
    misses = 0

    for question, expect_section in ANSWERABLE:
        # 关掉重排序先拿向量分
        retriever.enable_rerank = False
        v_results = retriever.retrieve(question, top_k=5)
        v_score = max((r.get("score", 0) for r in v_results), default=0)

        # 开重排序拿重排分
        retriever.enable_rerank = True
        r_results = retriever.retrieve(question, top_k=5)
        r_score = r_results[0].get("rerank_score", 0) if r_results else 0
        top_file = r_results[0].get("file_name", "?") if r_results else "?"

        vector_scores.append(v_score)
        answerable_scores.append(r_score)

        # 检查是不是检索到了正确的文件
        correct = expect_section in top_file
        if not correct:
            misses += 1
        mark = "✅" if correct else "❌"

        print(f"  {mark} {question:<20} {v_score:>8.4f} {r_score:>8.4f}  {top_file}")

    # ==============================================================
    # 测第二类：知识库里没有答案的
    # ==============================================================
    print(f"\n{'-' * 78}")
    print("  第二类：知识库里【没有】答案的问题（期望得低分）")
    print(f"{'-' * 78}\n")
    print(f"  {'问题':<22} {'向量分':>8} {'重排分':>8}  实际检索到的文件")
    print("  " + "-" * 72)

    unanswerable_rerank: list[float] = []
    unanswerable_vector: list[float] = []

    for question in UNANSWERABLE:
        retriever.enable_rerank = False
        v_results = retriever.retrieve(question, top_k=5)
        v_score = max((r.get("score", 0) for r in v_results), default=0)

        retriever.enable_rerank = True
        r_results = retriever.retrieve(question, top_k=5)
        r_score = r_results[0].get("rerank_score", 0) if r_results else 0
        top_file = r_results[0].get("file_name", "?") if r_results else "?"

        unanswerable_vector.append(v_score)
        unanswerable_rerank.append(r_score)

        print(f"    {question:<20} {v_score:>8.4f} {r_score:>8.4f}  {top_file}")

    # ==============================================================
    # 分析：找分数分界线
    # ==============================================================
    print(f"\n{LINE}")
    print("  分数分布分析")
    print(f"{LINE}\n")

    def describe(name: str, values: list[float]) -> dict:
        lo, hi = min(values), max(values)
        return {
            "name": name,
            "min": lo,
            "max": hi,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
        }

    a_rerank = describe("有答案-重排分", answerable_scores)
    u_rerank = describe("无答案-重排分", unanswerable_rerank)
    a_vector = describe("有答案-向量分", vector_scores)
    u_vector = describe("无答案-向量分", unanswerable_vector)

    for group in ((a_vector, u_vector), (a_rerank, u_rerank)):
        print(f"  【{group[0]['name'].split('-')[1]}】")
        for d in group:
            print(
                f"     {d['name']:<16} 最低 {d['min']:.4f} | "
                f"最高 {d['max']:.4f} | 平均 {d['mean']:.4f} | 中位 {d['median']:.4f}"
            )

        # 判定有没有清晰的分界线
        gap = group[0]["min"] - group[1]["max"]
        print(f"     → 两组之间的间隙：{gap:.4f}")
        if gap > 0:
            suggested = (group[1]["max"] + group[0]["min"]) / 2
            print(f"     ✅ 存在清晰分界线，建议阈值设为 {suggested:.4f}"
                  f"（在两组之间取中点，两边都留有余量）")
        else:
            print(f"     ⚠️  两组分数有重叠，无法用单一阈值完美区分")
            print(f"        重叠区间：[{group[0]['min']:.4f}, {group[1]['max']:.4f}]")
            # 用一个折中值：宁可漏掉一些有答案的，也不能让没答案的通过
            suggested = group[1]["max"] * 0.9 if group[1]["max"] > 0 else 0.01
            print(f"        建议偏保守取 {suggested:.4f}（倾向于『宁可说不知道』）")
        print()

    # ==============================================================
    # 结论
    # ==============================================================
    print(f"{LINE}")
    print("  诊断结论")
    print(f"{LINE}\n")

    print(f"  检索准确率：{len(ANSWERABLE) - misses}/{len(ANSWERABLE)} "
          f"（{((len(ANSWERABLE) - misses) / len(ANSWERABLE)):.0%}）")

    print(f"\n  ⚠️  当前配置的阈值 {0.3} 明显偏高。")
    print(f"     有答案的问题最低分只有 {a_rerank['min']:.4f}，")
    print(f"     会被阈值拦掉，导致『知识库里有答案、却回答没找到』。")
    print(f"\n     建议改为：")
    print(f"       向量相似度阈值：{a_vector['min'] * 0.6:.2f} 左右")
    print(f"       重排序分数阈值：{a_rerank['min'] * 0.6:.2f} 左右")

    # ==============================================================
    # 清理
    # ==============================================================
    print(f"\n{'-' * 78}")
    async with AsyncSessionLocal() as db:
        from sqlalchemy import select

        for doc_id in doc_ids:
            q = await db.execute(select(Document).where(Document.id == doc_id))
            doc = q.scalar_one_or_none()
            if doc:
                await ingest.delete_document(db, doc)
    print(f"  ✅ 测试数据已清理")
    print(f"{LINE}\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except Exception as exc:
        print(f"\n❌ 诊断异常: {type(exc).__name__}: {exc}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
