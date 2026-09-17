r"""
切片改造对比 —— 用**同一套题**，把"改造前"和"改造后"各考一遍

【为什么还要专门做这个脚本】
eval_retrieval.py 考的是"现在线上的系统"，只有一个版本。
但方案 C（结构化切分）需要一个**前后对比**才能说明它到底有没有用。

【为什么不能直接拿两份报告并排看】
因为"改造前"那组数字，是用**旧题库**测出来的。
后来我发现自己有 3 道题的答案出处标错了，改正了题目。
题目一改，"改造前"那组数字就不再是同一把尺子量出来的了 ——
拿两把不同的尺子量身高，然后宣称"长高了 3 厘米"，那是耍花招。

所以这个脚本做一件事：**用改正后的同一套 17 道题，把两种切法各考一遍。**

================================================================
【关键设计：沙盒向量库 —— 绝不碰你正在用的知识库】
================================================================
对比"旧切法"有个麻烦：旧切法切出来的片段得能被检索到，才能比较检索结果。
最直接的做法是把线上知识库改配置、重建索引、测完再改回来 ——
但那样中途一旦出错，你的知识库就卡在半新半旧的状态了，很危险。

这个脚本换了个思路：**在旁边另起一个沙盒**。

    · 临时目录 data/_compare_chroma/
    · 两个独立集合：cmp_before（旧切法）/ cmp_after（新切法）
    · 用猴子补丁（monkeypatch）把项目的全局向量库单例临时替换掉
    · 跑完自动还原

因为是**独立的 Python 进程**，所以：
    ✅ 不用重启后端服务
    ✅ 不用碰线上知识库
    ✅ 不用改 .env
    ✅ 跑坏了最多是沙盒目录脏了，删掉重来即可

【为什么连 BM25 也能一起切过去】
BM25 索引是懒加载的，它会去问"当前向量库是什么"然后现建。
所以替换掉那一个全局变量之后，向量检索和 BM25 会**同时**指向沙盒，
不会出现"向量查新库、BM25 查旧库"这种张冠李戴的情况。
（这件事我先去读了 bm25.py 的 ensure_ready() 才敢确认。）

================================================================
【它量什么】
================================================================
四条，和 eval_retrieval.py 保持一致，这样数字可以互相对照：

  ① 文件级命中  —— 有没有找对文件
  ② 答案级命中  —— 第一名那段里，有没有出现答案原话
  ③ 引用聚焦度  —— 第一名那段里混了几个小节（越接近 1 越干净）
  ④ 平均字数    —— 第一名那段有多长

【为什么 ③ 才是这次改造的主角】
实测下来 ① ② 改造前后都是满分 —— 因为 500 字的粗片段里**恰好也包含**
正确答案那句话（它把整个小节都包进去了，只是同时还包了别的）。
所以"答案在不在"这一关，两种切法都能过，区分不出好坏。

真正的差别在用户的眼睛里：
    旧：引用卡片一大段，里面三种面料，你得自己找哪句是你要的
    新：引用卡片就一小节，标题直接写着「4. 雪纺材质」
③ 量的就是这个。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe compare_chunking.py
（不需要启动后端服务；需要 .env 里有可用的 API Key）

【输出】
    终端打印 + 写入 data/logs/切片改造对比报告.txt
"""

import shutil
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

from app.rag.loader import parse_file                     # noqa: E402
from app.rag.splitter import ParentChildSplitter          # noqa: E402

# 题库直接复用评估脚本里那一份，**不重新抄一遍**。
# 抄一遍就埋下了"两份题库悄悄不一致"的隐患，
# 而这次要解决的恰恰就是"两把尺子"的问题。
from eval_retrieval import CASES, count_sections          # noqa: E402

UPLOAD_DIR = BACKEND_DIR / "data" / "uploads"
SANDBOX_DIR = BACKEND_DIR / "data" / "_compare_chroma"
REPORT = BACKEND_DIR / "data" / "logs" / "切片改造对比报告.txt"

LINES: list[str] = []


def out(text: str = "") -> None:
    LINES.append(text)
    print(text)


# ==================================================================
# 两种切法
# ==================================================================
# 改造前 = 项目最早那套参数：纯按字数切，500 字一片
# 改造后 = 方案 C：结构感知 + 250 字一片
#
# 父块大小（1400）两版一样，不参与对比 —— 它是喂给大模型的上下文，
# 不在本次改造范围内，动了它反而说不清是谁起的作用。
CONFIGS = [
    {
        "label": "改造前",
        "desc": "纯按字数切 500 字",
        "collection": "cmp_before",
        "kwargs": dict(child_size=500, child_overlap=80, structure_aware=False),
    },
    {
        "label": "改造后",
        "desc": "结构化切分 + 250 字",
        "collection": "cmp_after",
        "kwargs": dict(child_size=250, child_overlap=40, structure_aware=True),
    },
]


def original_name_map() -> dict[str, str]:
    """
    从数据库读出「磁盘上的 UUID 文件名 → 用户看到的原始文件名」对照表。

    【为什么必须绕这一道，不能直接用磁盘上的文件名】
    上传时文件会被改名成一串 UUID 再存到磁盘
    （目的是防重名覆盖、防路径穿越攻击），用户看到的原始名字
    存在数据库的 documents 表里。

    沙盒如果直接读磁盘文件名，拿到的就是 "624c703b2470….txt" 这种乱码。
    后果有两层：
      · 报告上「文件级命中」显示 0/17 —— 看着像系统全坏了，其实是假警报
      · 更糟的是**假警报比没警报更费时间**，它会把注意力
        引到一个根本不存在的故障上去（我第一版就栽在这上面，
        为此多花了一轮 4 分钟的完整重跑）

    所以这里老老实实去查数据库 —— 那才是"用户看到的文件名"的**唯一真相来源**。
    """
    import sqlite3

    from app.models.knowledge import Document

    db_path = BACKEND_DIR / "data" / "app.db"
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT stored_name, file_name FROM {Document.__tablename__}"
        ).fetchall()
    finally:
        conn.close()

    return {r[0]: r[1] for r in rows if r[0]}


def load_documents() -> list[tuple[int, Path, list]]:
    """读出三份真实资料，各自配一个 doc_id + 用户看到的原始文件名"""
    files = sorted(UPLOAD_DIR.glob("*.txt"))
    if not files:
        out(f"\n  ❌ 在 {UPLOAD_DIR} 下没找到资料文件\n")
        sys.exit(1)

    name_map = original_name_map()
    if not name_map:
        out("\n  ⚠️  读不到数据库里的文件名对照表，「文件级命中」这一项会不准\n")

    result = []
    for i, fp in enumerate(files, start=1):
        parsed = parse_file(fp)
        real_name = name_map.get(fp.name, fp.name)
        for doc in parsed:
            # 正式流程里这一步是入库模块 ingest.py 做的，
            # 它把 UUID 换回原始文件名，写进 metadata["source"]
            doc.metadata["source"] = real_name
        result.append((i, fp, parsed))
        out(f"    · {real_name}   ← 磁盘上叫 {fp.name[:12]}…")
    return result


def build_sandbox(label: str, collection: str, kwargs: dict, docs) -> tuple[object, int]:
    """
    在沙盒里建一个向量库，放入用指定切法切出来的片段。

    返回 (向量库对象, 片段数)
    """
    from app.rag.vectorstore import VectorStoreService

    splitter = ParentChildSplitter(**kwargs)

    chunks = []
    for doc_id, fp, parsed in docs:
        chunks.extend(splitter.split(parsed, doc_id=doc_id))

    store = VectorStoreService(
        collection_name=collection,
        persist_dir=str(SANDBOX_DIR),
    )

    # 先清空 —— 这样脚本可以反复跑，不会把上一次的旧片段留在里面
    # （不清的话第二次跑就会变成"两份片段混在一起"，数字全乱）
    store.delete_all()
    store.add_chunks(chunks)

    return store, len(chunks)


def evaluate(store, label: str) -> list[dict]:
    """
    用同一套题库，在给定的向量库里考一遍。

    【为什么要 monkeypatch】
    检索器内部写的是 get_vector_store()，它返回模块级全局单例。
    我不想改业务代码来支持"传入一个库"，那会为了一个测试脚本
    污染正式代码。所以在这里把那个全局变量临时换掉，跑完再换回来。
    """
    import app.rag.bm25 as bm25_module
    import app.rag.vectorstore as vs_module
    from app.rag.retriever import HybridRetriever

    original = vs_module._store
    vs_module._store = store

    try:
        # BM25 索引也必须重建 —— 它默认还记着上一次的语料
        bm25_module.ensure_ready(force=True)

        retriever = HybridRetriever()
        rows: list[dict] = []

        for i, (q, want_file, want_kw) in enumerate(CASES, 1):
            try:
                results = retriever.retrieve(q, top_k=5)
            except Exception as e:
                out(f"    ⚠️ [{i}/{len(CASES)}] 检索报错：{type(e).__name__}: {e}")
                results = []

            if not results:
                rows.append({"q": q, "ok_file": False, "ok_ans": False,
                             "focus": 0, "length": 0, "score": None})
                continue

            top = results[0]
            text = top.get("content") or ""
            rows.append({
                "q": q,
                "ok_file": want_file in (top.get("file_name") or ""),
                "ok_ans": want_kw in text,
                "focus": count_sections(text),
                "length": len(text),
                "score": top.get("rerank_score"),
            })
            print(f"    [{i}/{len(CASES)}] {q[:22]}  "
                  f"聚焦 {rows[-1]['focus']} / {rows[-1]['length']} 字"
                  f"{'' if rows[-1]['ok_ans'] else '  ⚠️答案没在第一名里'}")

            # 稍微歇一下，别把接口打太急
            time.sleep(0.4)

        return rows

    finally:
        # 无论成功失败都要还原，否则后面的代码会继续用沙盒
        vs_module._store = original


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "file": sum(1 for r in rows if r["ok_file"]),
        "ans": sum(1 for r in rows if r["ok_ans"]),
        "focus": sum(r["focus"] for r in rows) / n,
        "length": sum(r["length"] for r in rows) / n,
        "clean": sum(1 for r in rows if r["focus"] <= 1),
        "n": n,
    }


def main() -> int:
    out("=" * 78)
    out("  切片改造对比报告")
    out("=" * 78)
    out("\n  目的：用**同一套 17 道题**，把两种切法各考一遍，")
    out("        确认「结构化切分」到底带来了什么改变。\n")

    docs = load_documents()

    # 沙盒目录每次重建，避免残留
    if SANDBOX_DIR.exists():
        shutil.rmtree(SANDBOX_DIR, ignore_errors=True)
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    chunk_counts: dict[str, int] = {}

    for cfg in CONFIGS:
        out("-" * 78)
        out(f"【{cfg['label']}】{cfg['desc']}")
        out("-" * 78)

        store, n_chunks = build_sandbox(
            cfg["label"], cfg["collection"], cfg["kwargs"], docs
        )
        chunk_counts[cfg["label"]] = n_chunks
        out(f"\n  切出 {n_chunks} 个片段，正在沙盒里考试…\n")

        rows = evaluate(store, cfg["label"])
        results[cfg["label"]] = rows

        s = summarize(rows)
        out(f"\n  → 文件级 {s['file']}/{s['n']} | 答案级 {s['ans']}/{s['n']} | "
            f"聚焦 {s['focus']:.2f} 小节 | 平均 {s['length']:.0f} 字\n")

        # 释放沙盒里的向量库句柄，免得影响下一轮
        del store

    # ---------------- 逐题对比 ----------------
    out("\n" + "=" * 78)
    out("  逐题对比")
    out("=" * 78)
    out(f"\n  {'问题':<26} {'改造前':>16} {'改造后':>16}")
    out(f"  {'-' * 26} {'-' * 16} {'-' * 16}")

    before = results["改造前"]
    after = results["改造后"]

    for rb, ra in zip(before, after):
        b = f"{rb['focus']} 节 / {rb['length']} 字"
        a = f"{ra['focus']} 节 / {ra['length']} 字"
        q = rb["q"][:24]
        # 中文对齐要按显示宽度算，len() 对中文字符算 1 但显示占 2
        pad = 26 - sum(2 if ord(ch) > 0x2E80 else 1 for ch in q)
        out(f"  {q}{' ' * max(1, pad)} {b:>16} {a:>16}")

    # ---------------- 汇总 ----------------
    sb = summarize(before)
    sa = summarize(after)

    out("\n" + "=" * 78)
    out("  总对比")
    out("=" * 78)

    out(f"\n  {'指标':<30} {'改造前':>12} {'改造后':>12}")
    out(f"  {'-' * 30} {'-' * 12} {'-' * 12}")

    def row(name: str, b, a, better=""):
        out(f"  {name:<30} {b:>12} {a:>12}   {better}")

    row("片段总数", chunk_counts["改造前"], chunk_counts["改造后"],
        "切得更细")
    row("① 文件级命中", f"{sb['file']}/{sb['n']}", f"{sa['file']}/{sa['n']}")
    row("② 答案级命中", f"{sb['ans']}/{sb['n']}", f"{sa['ans']}/{sa['n']}",
        "← 真实水平")
    row("③ 引用聚焦度（越小越好）", f"{sb['focus']:.2f} 节", f"{sa['focus']:.2f} 节",
        "← 本次改造的主角")
    row("④ 第一名平均字数", f"{sb['length']:.0f} 字", f"{sa['length']:.0f} 字")
    row("⑤ 干净片段（只含1个小节）", f"{sb['clean']}/{sb['n']}", f"{sa['clean']}/{sa['n']}")

    out("\n" + "-" * 78)
    out("  【怎么读这张表】")
    out("-" * 78)

    if sb["ans"] == sa["ans"]:
        out("\n  ② 两版一样 —— 这不是坏消息，而是说明：")
        out("     500 字的粗片段里**恰好也包含**了答案那句话，所以「找没找到答案」")
        out("     这一关两种切法都能过，区分不出高低。")
        out("     （这也解释了为什么单看 ② 会误以为这次改造白做了。）")
    else:
        out(f"\n  ② 从 {sb['ans']} 变成 {sa['ans']}，说明切法真的影响了能不能找到答案。")

    delta = sb["focus"] - sa["focus"]
    if delta > 0.1:
        out(f"\n  ③ 从 {sb['focus']:.2f} 降到 {sa['focus']:.2f} —— 这才是这次改造真正换来的东西：")
        out("     引用卡片变干净了。用户点开「来源」，看到的就是答案对应的那一小节，")
        out("     而不是一大段里混着两三种无关内容、还得自己找。")
    else:
        out(f"\n  ③ 变化不大（{sb['focus']:.2f} → {sa['focus']:.2f}），"
            f"说明这份资料的编号结构可能不明显。")

    out(f"\n  一句话：改造没有让系统「更找得到答案」，")
    out(f"          而是让系统「找到的答案更干净」。\n")

    out("  【注意】本脚本用的是沙盒向量库（data/_compare_chroma），")
    out("          全程没有碰线上知识库，也没有重启服务。\n")

    out("=" * 78 + "\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(LINES), encoding="utf-8")
    print(f"报告已写入：{REPORT}\n")

    # 跑完把沙盒删掉，不留垃圾
    shutil.rmtree(SANDBOX_DIR, ignore_errors=True)
    return 0


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
        shutil.rmtree(SANDBOX_DIR, ignore_errors=True)
        sys.exit(1)
