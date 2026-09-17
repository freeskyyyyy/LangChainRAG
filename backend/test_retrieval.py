r"""
检索效果对比测试 —— 量化"优化到底有没有用"

【这个测试回答什么问题】
"你做了混合检索和重排序，效果到底提升了多少？"
—— 不是感觉提升了，而是有具体数字。

【测三种配置】
  配置 A：纯向量检索          （最简单的做法）
  配置 B：向量 + BM25 混合检索 （加了关键词检索）
  配置 C：向量 + BM25 + 重排序 （完整方案）

对同一批问题跑三遍，统计命中率，产出一张对比表。

【会消耗额度】
需要向量化 6 段语料 + 9 个问题，以及重排序调用。总量很小（几分钱级别）。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe test_retrieval.py
"""

import sys
import time

sys.path.insert(0, ".")

from langchain_core.documents import Document  # noqa: E402

from app.rag.bm25 import get_bm25_index, invalidate  # noqa: E402
from app.rag.retriever import HybridRetriever  # noqa: E402
from app.rag.splitter import ParentChildSplitter  # noqa: E402
from app.rag.vectorstore import VectorStoreService  # noqa: E402
from test_corpus import CORPUS, QUERIES  # noqa: E402

LINE = "=" * 78
TEST_DOC_ID = 999998  # 专用测试 ID，不会和真实数据冲突


def main() -> int:
    print(f"\n{LINE}")
    print("  检索效果对比测试：纯向量 vs 混合检索 vs 完整方案".center(66))
    print(f"{LINE}")

    # ==============================================================
    # 准备：把语料入库
    # ==============================================================
    print(f"\n{'-' * 78}\n  准备阶段：语料入库\n{'-' * 78}")

    vs = VectorStoreService()
    vs.delete_by_document(TEST_DOC_ID)  # 清理上次残留

    docs = [
        Document(page_content=text.strip(), metadata={"page": i + 1, "source": f"{section}.md"})
        for i, (section, text) in enumerate(CORPUS)
    ]

    splitter = ParentChildSplitter()
    chunks = splitter.split(docs, doc_id=TEST_DOC_ID)
    stats = splitter.analyze_chunks(chunks)

    print(f"  语料切分：{len(CORPUS)} 个章节 → "
          f"{stats['total_parents']} 个父块 / {stats['total_chunks']} 个子块")
    print(f"  平均每父块含 {stats['avg_children_per_parent']} 个子块")

    t0 = time.perf_counter()
    vs.add_chunks(chunks)
    print(f"  入库完成，耗时 {(time.perf_counter() - t0):.1f} 秒")
    print(f"  向量库片段总数：{vs.count()}")

    # 重建 BM25 索引
    invalidate()
    t0 = time.perf_counter()
    bm25 = get_bm25_index()
    from app.rag.bm25 import ensure_ready

    ensure_ready(force=True)
    print(f"  BM25 索引构建完成，耗时 {(time.perf_counter() - t0):.1f} 秒 "
          f"（{bm25.size} 个片段）")

    # ==============================================================
    # 对比三种配置
    # ==============================================================
    retriever = HybridRetriever()

    configs = [
        ("A 纯向量检索", False, False),
        ("B 向量 + BM25", False, True),
        ("C 向量 + BM25 + 重排序", True, True),
    ]

    all_results: dict[str, list] = {name: [] for name, _, _ in configs}

    print(f"\n{LINE}")
    print("  开始对比测试（共 3 种配置 × %d 个问题）" % len(QUERIES))
    print(f"{LINE}")

    for config_name, use_rerank, use_bm25 in configs:
        retriever.enable_rerank = use_rerank
        retriever.enable_bm25 = use_bm25

        print(f"\n  ▶ 测试配置：{config_name}")

        for question, expect_kw, expect_section in QUERIES:
            t0 = time.perf_counter()
            results = retriever.retrieve(question, top_k=5)
            elapsed = (time.perf_counter() - t0) * 1000

            # 判断有没有命中（在 top-5 里找期望关键词）
            hit_rank = None
            hit_file = None
            for rank, r in enumerate(results, start=1):
                text = r.get("content", "") + r.get("parent_content", "")
                if expect_kw in text:
                    hit_rank = rank
                    hit_file = r.get("file_name", "")
                    break

            all_results[config_name].append(
                {
                    "question": question,
                    "expect": expect_kw,
                    "expect_section": expect_section,
                    "hit_rank": hit_rank,
                    "hit_file": hit_file,
                    "elapsed_ms": elapsed,
                }
            )

            mark = f"✅ 第{hit_rank}名" if hit_rank else "❌ 未命中"
            print(f"     {mark:10} 「{question}」")

    # ==============================================================
    # 结果统计
    # ==============================================================
    print(f"\n{LINE}")
    print("  对比结果汇总")
    print(f"{LINE}\n")

    summary: dict[str, dict] = {}
    for name, _, _ in configs:
        rows = all_results[name]
        hits = [r for r in rows if r["hit_rank"]]
        top1 = [r for r in rows if r["hit_rank"] == 1]
        avg_ms = sum(r["elapsed_ms"] for r in rows) / len(rows)

        summary[name] = {
            "hit": len(hits),
            "top1": len(top1),
            "total": len(rows),
            "avg_ms": avg_ms,
        }

    # 打印对比表
    header = f"  {'配置':<26} {'命中率':>8} {'首位命中':>10} {'平均延迟':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, _, _ in configs:
        s = summary[name]
        hit_rate = s["hit"] / s["total"]
        top1_rate = s["top1"] / s["total"]
        print(
            f"  {name:<26} {s['hit']}/{s['total']} ({hit_rate:>4.0%}) "
            f"{s['top1']}/{s['total']} ({top1_rate:>4.0%}) {s['avg_ms']:>8.0f}ms"
        )

    # ==============================================================
    # 提升幅度
    # ==============================================================
    base = summary["A 纯向量检索"]
    print(f"\n  相对「纯向量检索」的提升：")
    for name, _, _ in configs[1:]:
        s = summary[name]
        delta_hit = (s["hit"] - base["hit"]) / base["total"]
        delta_top1 = (s["top1"] - base["top1"]) / base["total"]
        print(
            f"     {name:<26} 命中率 {delta_hit:+.0%} | 首位命中 {delta_top1:+.0%} | "
            f"延迟 {s['avg_ms'] - base['avg_ms']:+.0f}ms"
        )

    # ==============================================================
    # 逐条对比（重点看优化起作用的例子）
    # ==============================================================
    print(f"\n{LINE}")
    print("  逐条对比（重点看哪些问题被优化救了回来）")
    print(f"{LINE}\n")

    names = [n for n, _, _ in configs]
    changed: list[str] = []

    for i, (question, expect_kw, expect_section) in enumerate(QUERIES):
        ranks = [all_results[n][i]["hit_rank"] for n in names]

        # 只在"结果有变化"时才详细展示，避免输出太长
        if len(set(ranks)) == 1:
            continue

        changed.append(question)

        def fmt(r):
            return f"第{r}名" if r else "未命中"

        print(f"  「{question}」  (期望来自《{expect_section}》)")
        print(f"     A 纯向量       : {fmt(ranks[0])}")
        print(f"     B 混合检索     : {fmt(ranks[1])}")
        print(f"     C 完整方案     : {fmt(ranks[2])}")

        # 指出这次检索到的实际文件
        for n, r in zip(names, ranks):
            if r:
                print(f"        {n} 命中于《{all_results[n][i]['hit_file']}》")
                break
        print()

    if not changed:
        print("  （所有配置的结果都一样——可能是语料太简单，或优化没生效）")

    # ==============================================================
    # 反向验证：纯向量失败、优化后成功的案例
    # ==============================================================
    print(f"\n{LINE}")
    print("  重点案例：纯向量失手、优化救回的")
    print(f"{LINE}\n")

    saved = 0
    for i, (question, expect_kw, expect_section) in enumerate(QUERIES):
        r_a = all_results[names[0]][i]["hit_rank"]
        r_c = all_results[names[2]][i]["hit_rank"]

        if not r_a and r_c:
            saved += 1
            print(f"  ✅ 「{question}」")
            print(f"       纯向量检索：❌ 没找到")
            print(f"       完整方案  ：✅ 找到（第 {r_c} 名）")
            print(f"       正确答案在《{expect_section}》")
            print()

    if saved:
        print(f"  共救回 {saved} 个问题。")
    else:
        print("  这次没有出现'纯向量失手'的案例——")
        print("  说明这批测试问题上两种方案表现相当（这也是正常结果，")
        print("  混合检索的价值在更大、更复杂的知识库上才明显体现）。")

    # ==============================================================
    # 清理
    # ==============================================================
    print(f"\n{'-' * 78}\n  清理测试数据\n{'-' * 78}")
    removed = vs.delete_by_document(TEST_DOC_ID)
    invalidate()
    print(f"  已删除 {removed} 个测试片段，向量库剩余 {vs.count()} 个")

    print(f"\n{LINE}")
    print("  测试完成")
    print(f"{LINE}\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n❌ 测试异常: {type(exc).__name__}: {exc}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
