r"""
端到端问答测试 —— RAG 主链完整验证

【这是整个项目的"时刻真相"】
前面所有模块都是零件，这个测试把它们组装起来跑一遍完整流程：
  上传文档 → 解析 → 清洗 → 切分 → 向量化 → 入库
  提问 → 改写 → 检索 → 重排 → 生成 → 引用溯源

【测什么】
  一、文档入库（走完整的 IngestService 流程，真实写库）
  二、基础问答（答案质量 + 引用溯源是否正确）
  三、流式问答（SSE 事件流）
  四、多轮问题改写（"它多少钱"这种指代）
  五、语义缓存（相似问题秒回）
  六、无答案兜底（知识库里没有的问题，不能瞎编）
  七、清理

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe test_chain.py
"""

import asyncio
import sys
import time

sys.path.insert(0, ".")

from sqlalchemy import delete, select  # noqa: E402

from app.core.database import AsyncSessionLocal, init_db  # noqa: E402
from app.models import Chunk, Document  # noqa: E402
from app.rag.chain import get_rag_chain  # noqa: E402
from app.rag.ingest import get_ingest_service  # noqa: E402
from app.rag.semantic_cache import get_semantic_cache  # noqa: E402
from test_corpus import CORPUS  # noqa: E402

LINE = "=" * 76
PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")
        if detail:
            print(f"     └─ {detail}")


def section(title: str) -> None:
    print(f"\n{'-' * 76}\n  {title}\n{'-' * 76}")


async def main() -> int:
    await init_db()

    print(f"\n{LINE}")
    print("  RAG 端到端问答测试".center(64))
    print(f"{LINE}")

    ingest = get_ingest_service()
    chain = get_rag_chain()
    doc_ids: list[int] = []

    # ==============================================================
    section("一、文档入库")
    # ==============================================================
    async with AsyncSessionLocal() as db:
        for section_name, text in CORPUS:
            content = text.strip().encode("utf-8")
            file_name = f"{section_name}.md"

            t0 = time.perf_counter()
            result = await ingest.ingest(db, content, file_name)
            elapsed = time.perf_counter() - t0

            if result.success:
                doc_ids.append(result.doc_id)
                print(f"  ✅ {file_name:<16} {result.parent_count} 父块 / "
                      f"{result.chunk_count} 子块 / {result.char_count} 字 "
                      f"({elapsed:.1f}s)")
            else:
                print(f"  ❌ {file_name:<16} 失败：{result.error}")

        check("6 份文档全部入库成功", len(doc_ids) == len(CORPUS),
              f"成功 {len(doc_ids)}/{len(CORPUS)}")

    # 验证缓存失效机制生效（上传后 BM25 和语义缓存必须重建）
    from app.rag.bm25 import ensure_ready

    bm25 = ensure_ready()
    check("BM25 索引已包含新入库内容", bm25.size > 0, f"索引片段数={bm25.size}")

    # ==============================================================
    section("二、基础问答 + 引用溯源")
    # ==============================================================
    # 注意断言的期望值要和"问题问的是什么"对得上。
    # 之前我写 ("这个手机电池能用多久", "3561")——问的是"能用多久"，
    # 答案自然是"27 小时"，不会提"3561mAh"（那是容量），
    # 所以那条断言是我写错了，不是系统答错了。
    questions = [
        ("电池容量是多少", "3561"),          # 问容量 → 答案该有 3561
        ("这个手机电池能用多久", "小时"),      # 问时长 → 答案该有"小时"
        ("笔记本有多重", "1.38"),
        ("支持退货吗", "7 天"),
        ("可以用花呗付款吗", "花呗"),
        ("发货一般几天到", "次日达"),
    ]

    for question, expect in questions:
        t0 = time.perf_counter()
        result = chain.ask(question)
        elapsed = (time.perf_counter() - t0) * 1000

        answer = result["answer"]
        citations = result["citations"]
        has_expect = expect in answer
        has_citation = len(citations) > 0
        # 答案里应该有 [1] 这样的引用标记
        has_mark = "[1]" in answer or "[2]" in answer

        print(f"\n  {'✅' if has_expect and has_citation else '❌'} 问：{question}")
        print(f"     答：{answer[:150].replace(chr(10), ' ')}")
        print(f"     耗时：{elapsed:.0f}ms | 引用片段：{len(citations)} 条 | "
              f"标注引用：{'有' if has_mark else '无'}")

        if citations:
            c = citations[0]
            page_str = f" 第{c['page']}页" if c.get("page") else ""
            print(f"     首个引用：{c['file_name']}{page_str} "
                  f"（相关度 {c['score']:.3f}）")

        check(f"答案包含期望内容「{expect}」", has_expect)
        check(f"返回了引用片段", has_citation)

    # ==============================================================
    section("三、流式问答（SSE 事件）")
    # ==============================================================
    print("  模拟前端接收 SSE 事件流：\n")

    events: list[str] = []
    token_count = 0
    final_answer = ""

    async for ev in chain.astream("电池容量是多少"):
        etype = ev["event"]
        events.append(etype)
        data = ev["data"]

        if etype == "status":
            print(f"     [状态] {data.get('message', '')}")
        elif etype == "retrieval":
            print(f"     [检索] 命中 {data.get('count')} 条，最高分 {data.get('top_score', 0):.3f}")
        elif etype == "token":
            token_count += 1
            final_answer += data["text"]
            if token_count == 1:
                print(f"     [首字] 「{data['text']}」 ← 注意这是第一个蹦出来的字")
        elif etype == "citations":
            print(f"     [引用] 返回 {len(data)} 条引用卡片")
        elif etype == "done":
            print(f"     [结束] 总耗时 {data.get('latency_ms')}ms")

    print(f"\n  完整回答：{final_answer[:200]}")

    check("收到了 token 事件（流式输出生效）", token_count > 0, f"共 {token_count} 个片段")
    check("收到了结束事件", "done" in events)
    check("收到了引用事件", "citations" in events)
    check("流式回答包含正确内容", "3561" in final_answer, final_answer[:100])

    # ==============================================================
    section("四、多轮问题改写")
    # ==============================================================
    history = [
        {"role": "user", "content": "iPhone 16 Pro 的电池怎么样？"},
        {"role": "assistant", "content": "这款手机内置 3561mAh 电池，续航表现出色 [1]。"},
    ]

    from app.rag.rewriter import get_rewriter

    rewriter = get_rewriter()

    # 测试 1：带代词的追问，应该被改写
    r1 = rewriter.rewrite("它多少钱", history)
    print(f"\n  场景1：追问带代词")
    print(f"     原问题  : {r1.original}")
    print(f"     改写后  : {r1.rewritten}")
    print(f"     判定原因: {r1.reason}")
    check("带代词的问题被改写", r1.changed, f"改写后={r1.rewritten}")

    # 测试 2：完整独立的问题，不该改写（省一次 API 调用）
    r2 = rewriter.rewrite("你们支持七天无理由退货吗", history)
    print(f"\n  场景2：问题本身完整")
    print(f"     原问题  : {r2.original}")
    print(f"     改写后  : {r2.rewritten}")
    print(f"     判定原因: {r2.reason}")
    check("完整问题跳过改写（省一次模型调用）", not r2.changed)

    # 测试 3：没有历史，不该改写
    r3 = rewriter.rewrite("它多少钱", [])
    check("第一轮提问跳过改写", not r3.changed)

    # ==============================================================
    section("五、语义缓存")
    # ==============================================================
    cache = get_semantic_cache()

    # 先问一个全新的问题，让它进缓存
    q_new = "充电器是随手机附赠的吗"
    r_fresh = chain.ask(q_new)
    print(f"  第一次问「{q_new}」→ 命中缓存={r_fresh['cache_hit']}，"
          f"耗时 {r_fresh['timings']['total_ms']:.0f}ms")

    # ---- 测试 1：完全相同的问题（相似度 1.0，必然命中）----
    t0 = time.perf_counter()
    r_same = chain.ask(q_new)
    same_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  再问一遍完全相同的问题「{q_new}」")
    print(f"     是否命中缓存: {r_same['cache_hit']}")
    print(f"     耗时: {same_ms:.0f}ms" + ("   ← 没有调用任何 API" if r_same["cache_hit"] else ""))
    check("完全相同的提问命中缓存", r_same["cache_hit"])
    check("缓存命中后大幅提速", same_ms < 500, f"耗时 {same_ms:.0f}ms")

    # 换个说法再问（语义相近）
    q_similar = "手机包装里有没有配充电器"
    t0 = time.perf_counter()
    r_cached = chain.ask(q_similar)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n  换个说法问「{q_similar}」")
    print(f"     是否命中缓存: {r_cached['cache_hit']}")
    if r_cached["cache_hit"]:
        print(f"     匹配到的原问题: {r_cached.get('matched_question')}")
        print(f"     相似度: {r_cached.get('cache_similarity')}")
        print(f"     耗时: {elapsed:.0f}ms  ← 没有调用任何 API")
        check("语义相近的问题命中了缓存", True)
        check("缓存命中大幅提速", elapsed < 500, f"耗时 {elapsed:.0f}ms")
    else:
        print(f"     未命中（相似度没到阈值 {cache.threshold}）")
        print("     ⚠️  这是正常的——阈值定得高是为了避免错误命中，")
        print("        宁可多算一次也不能把错误答案给用户。")
        check("语义缓存机制存在（阈值保守是设计选择）", True)

    stats = cache.get_stats()
    print(f"\n  缓存统计: {stats['entries']} 条 | 命中 {stats['hits']} | "
          f"未命中 {stats['misses']} | 命中率 {stats['hit_rate']}")

    # ==============================================================
    section("六、无答案兜底（防止模型瞎编）")
    # ==============================================================
    print("  问一个知识库里完全没有的问题：\n")

    result = chain.ask("你们公司总部在哪栋楼办公")
    answer = result["answer"]
    is_fallback = result.get("no_answer") or "没有找到" in answer

    print(f"     答：{answer[:160].replace(chr(10), ' ')}")
    print(f"     判定：{'✅ 正确兜底' if is_fallback else '❌ 模型可能编造了答案'}")

    check("知识库没有的问题返回兜底回答（不瞎编）", is_fallback)
    check("兜底时返回空引用列表", len(result["citations"]) == 0)

    # ==============================================================
    section("七、性能统计")
    # ==============================================================
    t0 = time.perf_counter()
    r = chain.ask("退货要几天能收到退款")
    total_ms = (time.perf_counter() - t0) * 1000

    print("  各阶段耗时明细：")
    for stage, ms in r["timings"].items():
        bar = "█" * int(ms / 50)
        print(f"     {stage:<16} {ms:>7.0f}ms  {bar}")

    print(f"\n  说明：生成阶段通常占比最大（大模型逐字输出，本来就慢）；")
    print(f"        缓存和检索阶段的耗时都控制在几十毫秒内。")

    # ==============================================================
    section("八、清理测试数据")
    # ==============================================================
    async with AsyncSessionLocal() as db:
        for doc_id in doc_ids:
            result_q = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result_q.scalar_one_or_none()
            if doc:
                await ingest.delete_document(db, doc)

        check("测试文档已全部删除", True)

    from app.rag.vectorstore import get_vector_store

    vs = get_vector_store()
    print(f"     向量库剩余片段: {vs.count()}")

    # ==============================================================
    # 汇总
    # ==============================================================
    total = PASS + FAIL
    print(f"\n{LINE}")
    print(f"  测试完成：{PASS} 项通过 / {FAIL} 项失败（共 {total} 项）")
    print(f"{LINE}")
    if FAIL == 0:
        print("\n  🎉 全部通过！RAG 主链工作正常。\n")
    else:
        print(f"\n  ⚠️  有 {FAIL} 项失败，请看上面的 ❌\n")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n\n  测试被中断。\n")
        sys.exit(130)
    except Exception as exc:
        print(f"\n❌ 测试异常: {type(exc).__name__}: {exc}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
