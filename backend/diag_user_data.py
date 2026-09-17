r"""
真实数据体检脚本 —— 用你自己上传的 3 份资料做端到端验证

【为什么要跑这个】
前面所有测试用的都是我编的假数据（手机、笔记本）。
真实数据的结构、用词、篇幅都不一样，必须用真数据重新验一遍，
才能知道：切片切得对不对、检索能不能命中、阈值还合不合适。

【输出到哪里】
结果同时写到 data/logs/数据体检报告.txt，
因为终端显示中文会乱码，写文件再用编辑器看最清楚。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe diag_user_data.py
（需要后端服务已启动）
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# 强制 UTF-8 输出（原因见 run.py 的说明）
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

BASE = "http://127.0.0.1:8000"
REPORT = Path(__file__).parent / "data" / "logs" / "数据体检报告.txt"

# 报告内容先攒在内存里，最后一次性写文件
LINES: list[str] = []


def out(text: str = "") -> None:
    LINES.append(text)
    print(text)


def get(path: str, token: str) -> dict:
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def search(query: str, token: str, top_k: int = 5) -> dict:
    """调检索测试接口（只检索，不调大模型，所以很快）"""
    boundary = "----DiagBoundary7788"
    body = b""
    for k, v in (("query", query), ("top_k", str(top_k)), ("compare", "true")):
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        body += f"{v}\r\n".encode()
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        BASE + "/api/kb/search",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode("utf-8", errors="ignore")[:300]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    # ---------------- 登录 ----------------
    req = urllib.request.Request(
        BASE + "/api/auth/login",
        data=json.dumps({"username": "admin", "password": "123456"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            token = json.loads(r.read())["access_token"]
    except Exception as e:
        print(f"登录失败，服务是不是没启动？{e}")
        return 1

    out("=" * 74)
    out("  真实数据体检报告")
    out("=" * 74)

    # ---------------- 一、文档清单 ----------------
    out("\n【一】已入库的文档\n")
    docs = get("/api/kb/documents", token)
    items = docs if isinstance(docs, list) else (
        docs.get("items") or docs.get("documents") or docs.get("data") or []
    )
    for i, d in enumerate(items, 1):
        name = d.get("file_name") or d.get("original_name") or d.get("name")
        out(f"  {i}. {name}")
        out(f"     状态={d.get('status')} | "
            f"父块={d.get('parent_count')} 子块={d.get('chunk_count')} | "
            f"{d.get('char_count')} 字 | ID={d.get('id')}")

    stats = get("/api/kb/stats", token)
    out(f"\n  合计：{stats['documents']['total']} 份文档 / "
        f"{stats['content']['vector_chunks']} 个检索片段 / "
        f"{stats['content']['total_chars']} 字")

    # ---------------- 二、切片质量 ----------------
    out("\n" + "=" * 74)
    out("【二】切片质量抽查（片段预览接口）")
    out("=" * 74)
    out("\n  看什么：每片是不是一个完整的语义单元。")
    out("  切得好的话，一片讲的是一件事；切得差会出现『半句话』或『两件不相关的事挤一起』。\n")

    for d in items[:6]:
        did = d.get("id")
        name = d.get("file_name") or d.get("original_name") or d.get("name")
        if did is None:
            continue
        try:
            chunks = get(f"/api/kb/documents/{did}/chunks?limit=3", token)
        except Exception as e:
            out(f"  [{name}] 预览失败: {e}")
            continue
        clist = chunks if isinstance(chunks, list) else (
            chunks.get("items") or chunks.get("chunks") or []
        )
        out(f"\n  ── {name} （共 {len(clist)} 片显示前 3 片）──")
        for c in clist[:3]:
            text = (c.get("content") or "").replace("\n", " ")
            out(f"    · [{c.get('chunk_index', c.get('index'))}] {len(text)} 字：{text[:150]}")

    # ---------------- 三、真实问题检索测试 ----------------
    out("\n" + "=" * 74)
    out("【三】真实问题检索测试")
    out("=" * 74)
    out("\n  说明：『期望』是我根据你资料内容预先判断的正确答案出处。")
    out("  看『命中』那一列——第一位就是期望来源，说明检索准确。\n")

    cases = [
        ("我身高168 体重120斤，穿什么尺码", "尺码对照"),
        ("190cm 210斤以上穿什么码", "尺码对照"),
        ("黄皮适合穿什么颜色", "颜色搭配"),
        ("面试的时候穿什么颜色合适", "颜色搭配"),
        ("冬天适合穿什么颜色的衣服", "颜色搭配"),
        ("真丝连衣裙怎么洗", "洗涤养护"),
        ("羽绒服可以机洗吗", "洗涤养护"),
        ("雪纺洗的时候能拧干吗", "洗涤养护"),
        ("羊毛大衣能用洗衣机洗吗", "洗涤养护"),
        ("我腿粗想显瘦穿什么颜色", "颜色搭配"),
        ("公司食堂几点开饭", "（资料里没有，应该拒答）"),
        ("这款手机电池多大", "（资料里没有，应该拒答）"),
    ]

    hit_top1 = 0
    hit_any = 0
    counted = 0

    for q, expect in cases:
        res = search(q, token, top_k=5)
        if res.get("error"):
            out(f"\n  问题：{q}\n    ❌ 检索失败: {res['error']}")
            continue

        results = res.get("results", [])
        out(f"\n  问题：{q}")
        out(f"    期望来源：{expect}")

        if not results:
            out("    （无任何召回）")
            continue

        for r in results[:3]:
            flag = "★" if r["rank"] == 1 else " "
            out(f"    {flag} [{r['rank']}] {r['file_name']:<16} "
                f"向量={r.get('vector_score') or '-':<6} "
                f"重排={r.get('rerank_score') or '-':<6} "
                f"两路命中={r.get('hit_both')}")
            out(f"          {(r.get('content') or '')[:90].replace(chr(10), ' ')}")

        # 判断命中
        if "资料里没有" not in expect:
            counted += 1
            top1_name = results[0]["file_name"]
            if expect in top1_name:
                hit_top1 += 1
            if any(expect in r["file_name"] for r in results):
                hit_any += 1

    # ---------------- 四、分数分布（判断阈值是否需要重调） ----------------
    out("\n" + "=" * 74)
    out("【四】分数分布（判断阈值还合不合适）")
    out("=" * 74)
    out("\n  有答案的问题 vs 没答案的问题，分数应该拉得开。")
    out("  如果重叠了，说明要重新调 .env 里的阈值。\n")

    answerable = [q for q, e in cases if "资料里没有" not in e]
    unanswerable = [q for q, e in cases if "资料里没有" in e]

    def collect(qs: list[str]) -> tuple[list[float], list[float]]:
        rr, vv = [], []
        for q in qs:
            res = search(q, token, top_k=1)
            for r in res.get("results", [])[:1]:
                if r.get("rerank_score") is not None:
                    rr.append(r["rerank_score"])
                if r.get("vector_score") is not None:
                    vv.append(r["vector_score"])
        return rr, vv

    a_rr, a_vv = collect(answerable)
    u_rr, u_vv = collect(unanswerable)

    def rng(xs: list[float]) -> str:
        return f"{min(xs):.3f} ~ {max(xs):.3f}" if xs else "（无数据）"

    out(f"  重排序分数   有答案 {rng(a_rr)}   无答案 {rng(u_rr)}")
    out(f"  向量相似度   有答案 {rng(a_vv)}   无答案 {rng(u_vv)}")

    if a_rr and u_rr:
        gap = min(a_rr) - max(u_rr)
        out(f"\n  重排序分数最小值（有答案）{min(a_rr):.3f} "
            f"− 最大值（无答案）{max(u_rr):.3f} = {gap:+.3f}")
        if gap > 0:
            out(f"  ✅ 分得开，中间点 {((min(a_rr) + max(u_rr)) / 2):.3f} 可以当阈值")
        else:
            out("  ⚠️  重叠了，需要重新调阈值")

    # ---------------- 五、汇总 ----------------
    out("\n" + "=" * 74)
    out("  汇总")
    out("=" * 74)
    if counted:
        out(f"\n  检索准确率：首位命中 {hit_top1}/{counted} "
            f"（{hit_top1 / counted * 100:.0f}%） | "
            f"前五命中 {hit_any}/{counted}（{hit_any / counted * 100:.0f}%）")
    out(f"  知识库规模：{stats['documents']['total']} 份 / "
        f"{stats['content']['vector_chunks']} 片段 / "
        f"{stats['content']['total_chars']} 字")
    out("\n" + "=" * 74 + "\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\n报告已写入：{REPORT}")
    print("（终端里中文可能显示乱码，用记事本或 VSCode 打开这个文件看正常）\n")
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
        sys.exit(1)
