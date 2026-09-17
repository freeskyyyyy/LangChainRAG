r"""
批量重建索引 —— 改了切片参数之后，用它让新参数生效

【什么时候需要跑这个】
知识库里的文档是**按当时的切分参数**切好后存起来的。
如果你后来改了 .env 里的切片长度、或者改了切分逻辑，
已经入库的文档**不会自动重切** —— 它们还是老样子。

这个脚本就是把已经上传过的文档，用**当前**的参数重新切一遍。

【它做的事】
对每一份已入库的文档：
    1. 读出磁盘上保存的原始文件
    2. 重新解析、重新切分
    3. 删掉旧的向量，写入新的向量
    4. 更新数据库里的片段记录

原始文件本身不会被改动，所以随时可以重切，不会丢数据。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe reindex_all.py
（需要后端服务已启动）
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

BASE = "http://127.0.0.1:8000"
REPORT = Path(__file__).parent / "data" / "logs" / "重建索引报告.txt"
LINES: list[str] = []


def out(t: str = "") -> None:
    LINES.append(t)
    print(t)


def call(method: str, path: str, token: str | None = None) -> tuple[int, object]:
    headers = {"Authorization": "Bearer " + token} if token else {}
    r = urllib.request.Request(BASE + path, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
            return e.code, payload.get("message") or payload
        except json.JSONDecodeError:
            return e.code, raw[:200]
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def main() -> int:
    out("=" * 70)
    out("  批量重建索引")
    out("=" * 70)

    # ---- 登录 ----
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
        out(f"\n  ❌ 登录失败：{e}")
        out("     请先启动后端服务（run.py）\n")
        return 1

    # ---- 重建前 ----
    status, before = call("GET", "/api/kb/stats", token)
    if status != 200:
        out(f"\n  ❌ 取统计失败：{before}\n")
        return 1

    n_docs_before = before["documents"]["total"]
    n_chunks_before = before["content"]["vector_chunks"]

    out(f"\n  重建前：{n_docs_before} 份文档 / {n_chunks_before} 个片段")
    out(f"  （平均每份 {n_chunks_before / max(1, n_docs_before):.1f} 个片段）\n")

    # ---- 逐份重建 ----
    status, docs = call("GET", "/api/kb/documents?page=1&page_size=100", token)
    items = docs.get("items", []) if isinstance(docs, dict) else []

    if not items:
        out("  ⚠️  知识库里没有文档，没什么可重建的。\n")
        return 0

    out("-" * 70)
    ok = 0
    failed = 0

    for i, d in enumerate(items, 1):
        name = d.get("file_name", f"文档{d.get('id')}")
        out(f"\n  [{i}/{len(items)}] {name}")
        out(f"      重建前：{d.get('chunk_count')} 个片段 / {d.get('char_count')} 字")

        t0 = time.perf_counter()
        status, result = call("POST", f"/api/kb/documents/{d['id']}/reindex", token)
        elapsed = time.perf_counter() - t0

        if status == 200 and isinstance(result, dict) and result.get("success"):
            ok += 1
            new_count = result.get("chunk_count", "?")
            old_count = d.get("chunk_count", 0)
            delta = (new_count - old_count) if isinstance(new_count, int) else 0
            arrow = f"（{delta:+d}）" if delta else "（不变）"
            out(f"      重建后：{new_count} 个片段 {arrow}  耗时 {elapsed:.1f} 秒")
        else:
            failed += 1
            out(f"      ❌ 失败：{result}")

        # 限流是每分钟 20 次，重建比较重，慢一点更稳
        time.sleep(1.0)

    # ---- 重建后 ----
    status, after = call("GET", "/api/kb/stats", token)
    n_chunks_after = after["content"]["vector_chunks"] if status == 200 else "?"

    out("\n" + "=" * 70)
    out("  重建完成")
    out("=" * 70)
    out(f"\n  成功 {ok} 份 / 失败 {failed} 份")
    out(f"  片段总数：{n_chunks_before} → {n_chunks_after}")

    if isinstance(n_chunks_after, int) and n_chunks_before:
        ratio = n_chunks_after / n_chunks_before
        out(f"  变化：{ratio:.2f} 倍")
        if ratio > 1.5:
            out("\n  ✅ 片段变多了，说明切得更细了 —— 这正是结构化切分想要的效果：")
            out("     每片只讲一件事，检索时不容易被相邻内容干扰。")
        elif ratio < 0.8:
            out("\n  ⚠️  片段变少了，请确认这是预期的。")
        else:
            out("\n  ℹ️  片段数量变化不大，可能这份资料本来就没有明显的编号结构。")

    out("\n  ⚠️ 请重新跑一次检索评估，确认效果真的变好了：")
    out("     ..\\.venv\\Scripts\\python.exe eval_retrieval.py\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\n报告已写入：{REPORT}\n")
    return 0 if failed == 0 else 1


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
