r"""
HTTP 层端到端测试 —— 模拟浏览器真实调用

【为什么还要测这个，前面不是已经测过 RAG 主链了吗】
因为"Python 里直接调函数能跑通"和"通过 HTTP 接口能跑通"是两回事。
中间隔着一层 HTTP，有很多东西可能出问题：
  · 请求参数怎么传（JSON 格式对不对）
  · 登录令牌有没有正确带上
  · SSE 流的事件格式对不对、浏览器能不能解析
  · 流式过程中数据库会话的生命周期

这些只有真正发 HTTP 请求才能验证。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe test_http_chat.py
（需要后端服务已经启动）
"""

import json
import sys
import urllib.error
import urllib.request

# ------------------------------------------------------------------
# 强制标准输出用 UTF-8
# ------------------------------------------------------------------
# 【为什么必须加这一段】
# 脚本里用了 ✅ ❌ 这类符号来标记通过/失败。
# 直接在命令行里跑时，Python 走 Windows 的控制台接口，显示正常；
# 但一旦输出被**管道接走**（比如 `python test_http_chat.py | tail`），
# Python 就改用系统默认编码 —— 中文 Windows 上是 GBK，
# 而 GBK 里没有 ✅ 这个字符，于是：
#
#     UnicodeEncodeError: 'gbk' codec can't encode character '✅'
#
# 表现是：测试**其实跑完了**，但结果一行都没打出来，只看到一堆乱码报错。
# 这个脚本是唯一漏加这段的（其他测试脚本都有），所以一直被这个问题绊。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

BASE = "http://127.0.0.1:8000"
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
    print(f"\n{'-' * 72}\n  {title}\n{'-' * 72}")


def login(username: str, password: str) -> str | None:
    req = urllib.request.Request(
        f"{BASE}/api/auth/login",
        data=json.dumps({"username": username, "password": password}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())["access_token"]
    except Exception as e:
        print(f"     登录失败: {e}")
        return None


def stream_chat(token: str, question: str, session_id: int | None = None) -> dict:
    """
    调用流式问答接口，解析 SSE 事件。

    【为什么要自己解析 SSE】
    浏览器的 EventSource 只能发 GET 请求，而问答需要发 POST（问题可能很长）。
    所以前端实际会用 fetch + ReadableStream 自己解析，
    这里就是模拟那个过程——如果这里能解析成功，
    说明格式是对的，前端也能解析。

    SSE 格式：
        event: 事件类型
        data: JSON 数据
        (空行表示一个事件结束)
    """
    body = {"question": question}
    if session_id:
        body["session_id"] = session_id

    req = urllib.request.Request(
        f"{BASE}/api/chat/stream",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Accept": "text/event-stream",
        },
    )

    result = {
        "events": [],
        "answer": "",
        "citations": [],
        "error": None,
        "done": None,
        "session_id": session_id,
    }

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            buffer = ""
            while True:
                raw = resp.read(1024)
                if not raw:
                    break

                buffer += raw.decode("utf-8", errors="ignore")
                blocks = buffer.split("\n\n")
                buffer = blocks.pop()

                for block in blocks:
                    if not block.strip():
                        continue

                    etype, data_str = "message", ""
                    for line in block.split("\n"):
                        if line.startswith("event:"):
                            etype = line[6:].strip()
                        elif line.startswith("data:"):
                            data_str += line[5:].strip()

                    if not data_str:
                        continue

                    try:
                        payload = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    result["events"].append(etype)

                    if etype == "token":
                        result["answer"] += payload.get("text", "")
                    elif etype == "citations":
                        result["citations"] = payload
                    elif etype == "session":
                        result["session_id"] = payload.get("session_id")
                    elif etype == "done":
                        result["done"] = payload
                    elif etype == "error":
                        result["error"] = payload

    except Exception as e:
        result["error"] = {"message": f"{type(e).__name__}: {e}"}

    return result


def upload_file(token: str, file_name: str, content: bytes) -> dict:
    """用 multipart 表单上传文件"""
    boundary = "----RAGTestBoundary1234567890"
    body = b""

    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'.encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += content
    body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{BASE}/api/kb/documents/upload",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        try:
            return {"error": json.loads(raw).get("message", raw)}
        except json.JSONDecodeError:
            return {"error": raw[:200]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


SAMPLE_DOC = """# 产品售后服务手册

## 一、退货政策

本店支持 7 天无理由退货。消费者自签收之日起 7 日内，
在不影响二次销售的前提下可申请退货。

退货商品需保持包装完整、配件齐全、赠品一并退回。
以下商品不适用无理由退货：已激活的数码产品、定制类商品、贴身衣物。

退款将在商品验收通过后 1-3 个工作日内原路退回。

## 二、换货政策

自签收之日起 15 日内，商品出现质量问题可申请换货。
换货商品由本店承担来回运费。

## 三、价格保护

商品签收后 30 日内，如遇本店官方降价，可申请差价补偿。
差价将以优惠券形式返还，有效期 90 天。

注意：参与限时秒杀、清仓特卖的商品不参与价保。

## 四、发票说明

本店支持开具电子发票和纸质发票。
电子发票在订单完成后自动发送至预留邮箱。
如需开具增值税专用发票，请联系客服并提供公司开票信息。
"""


def main() -> int:
    print(f"\n{'=' * 72}")
    print("  HTTP 端到端测试（模拟浏览器调用）".center(62))
    print(f"{'=' * 72}")

    # ---------------- 准备：管理员登录 ----------------
    section("〇、管理员登录")
    admin_token = login("admin", "123456")
    check("管理员登录成功", bool(admin_token))
    if not admin_token:
        print("\n  ⚠️  请先启动后端服务：run.py\n")
        return 1

    # ---------------- 上传文档 ----------------
    section("一、上传知识库文档")
    result = upload_file(admin_token, "售后服务手册.md", SAMPLE_DOC.encode("utf-8"))

    # ⚠️ 注意这里判断的是 result["error"] 的**值**，不是键是否存在。
    #    接口成功时也会返回 "error": null 这个键，
    #    所以写 `if "error" in result` 会永远成立，把成功当失败。
    #    我一开始就是这么写的，结果测试一直报"上传失败"却给不出原因——
    #    因为 result["error"] 是 None，打印出来是空的。
    if result.get("error"):
        check("文档上传成功", False, result["error"])
        doc_id = None
    else:
        check("文档上传成功", result.get("success", False), str(result)[:200])
        doc_id = result.get("doc_id")
        print(f"     文档 ID: {doc_id} | "
              f"{result.get('parent_count')} 父块 / {result.get('chunk_count')} 子块 | "
              f"{result.get('char_count')} 字")
        # 顺带验证中文文件名没有乱码
        check(
            "中文文件名未乱码",
            result.get("file_name") == "售后服务手册.md",
            f"实际文件名: {result.get('file_name')!r}",
        )

    # ---------------- 知识库统计 ----------------
    section("二、知识库统计接口")
    req = urllib.request.Request(
        f"{BASE}/api/kb/stats",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            stats = json.loads(r.read())
        check("统计接口可用", True)
        print(f"     文档数: {stats['documents']['total']} | "
              f"片段数: {stats['content']['vector_chunks']} | "
              f"总字数: {stats['content']['total_chars']}")
    except Exception as e:
        check("统计接口可用", False, str(e))

    # ---------------- 流式问答 ----------------
    section("三、流式问答（SSE）")
    print("  发送问题：「买错了可以退货吗」\n")

    t_result = stream_chat(admin_token, "买错了可以退货吗")

    if t_result["error"]:
        check("流式问答成功", False, str(t_result["error"]))
    else:
        check("收到了 token 事件", "token" in t_result["events"])
        check("收到了引用事件", "citations" in t_result["events"])
        check("收到了结束事件", "done" in t_result["events"])

        print(f"\n     事件序列: {' → '.join(dict.fromkeys(t_result['events']))}")
        print(f"\n     回答内容:\n     {t_result['answer'][:220]}")

        has_answer = len(t_result["answer"]) > 10
        check("生成了有效回答", has_answer)

        cites = t_result["citations"]
        check("返回了引用片段", len(cites) > 0, f"引用数={len(cites)}")

        if cites:
            print(f"\n     引用片段:")
            for c in cites[:3]:
                page = f" 第{c['page']}页" if c.get("page") else ""
                print(f"       [{c['index']}] {c['file_name']}{page} "
                      f"(相关度 {c['score']:.3f})")
                print(f"           {c['content'][:80]}...")

        # 检查引用标注是否出现在正文里
        import re

        marks = re.findall(r"\[(\d+)\]", t_result["answer"])
        check("回答正文里有引用标注", len(marks) > 0, f"找到标注 {marks}")

        if t_result["done"]:
            d = t_result["done"]
            print(f"\n     耗时: {d.get('latency_ms')}ms | 缓存命中: {d.get('cache_hit')}")
            if d.get("timings"):
                print("     各阶段耗时:")
                for k, v in d["timings"].items():
                    print(f"       {k:<16} {v:.0f}ms")

    # ---------------- 多轮对话 ----------------
    section("四、多轮对话（追问）")
    sid = t_result.get("session_id")
    print(f"  会话 ID: {sid}")
    print("  追问：「那换货呢」← 这个问题省略了主语，需要靠上文补全\n")

    if sid:
        r2 = stream_chat(admin_token, "那换货呢", session_id=sid)

        if r2["error"]:
            check("追问成功", False, str(r2["error"]))
        else:
            check("追问成功", len(r2["answer"]) > 5)
            print(f"     回答: {r2['answer'][:180]}")

            # 检查是否触发了问题改写
            if "rewrite" in r2["events"]:
                print("     ✅ 触发了问题改写（说明多轮理解生效）")
            else:
                print("     ℹ️  未触发改写（问题可能已足够完整）")

    # ---------------- 权限测试 ----------------
    section("五、权限隔离（普通用户不能管知识库）")
    import time as _t

    uname = f"httptest{int(_t.time()) % 100000}"
    req = urllib.request.Request(
        f"{BASE}/api/auth/register",
        data=json.dumps({"username": uname, "password": "test123456"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            user_data = json.loads(r.read())
        check("普通用户注册成功", True)
    except Exception as e:
        check("普通用户注册成功", False, str(e))
        user_data = None

    if user_data:
        user_token = login(uname, "test123456")
        check("普通用户登录成功", bool(user_token))

        if user_token:
            # 试着访问管理员接口
            req = urllib.request.Request(
                f"{BASE}/api/kb/documents",
                headers={"Authorization": f"Bearer {user_token}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    check("普通用户访问知识库管理被拒绝", False, f"竟然成功了，状态码 {r.status}")
            except urllib.error.HTTPError as e:
                check("普通用户访问知识库管理被拒绝（403）", e.code == 403,
                      f"状态码={e.code}（期望 403）")

    # ---------------- 检索测试接口 ----------------
    section("六、检索测试接口（调参用，不调大模型）")
    import time as _time

    boundary = "----RAGBoundary9999"
    body = b""
    for k, v in (("query", "价格保护"), ("top_k", "3"), ("compare", "true")):
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        body += f"{v}\r\n".encode()
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{BASE}/api/kb/search",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {admin_token}",
        },
    )

    t0 = _time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            search_result = json.loads(r.read())
        elapsed = (_time.perf_counter() - t0) * 1000

        check("检索测试接口可用", True)
        print(f"     耗时 {elapsed:.0f}ms（没调大模型，所以很快）")

        print(f"\n     检索到 {len(search_result['results'])} 条:")
        for r in search_result["results"]:
            print(f"       [{r['rank']}] {r['file_name']} | "
                  f"向量 {r['vector_score'] or '-'} | "
                  f"重排 {r['rerank_score'] or '-'} | "
                  f"命中两路: {r['hit_both']}")

        if "comparison" in search_result:
            print(f"\n     三种配置对比（同一问题）:")
            for cfg, items in search_result["comparison"].items():
                top = items[0] if items else {}
                label = {
                    "vector_only": "纯向量    ",
                    "hybrid": "向量+BM25 ",
                    "hybrid_rerank": "完整方案  ",
                }.get(cfg, cfg)
                print(f"       {label} 第1名得分 {top.get('score', 0):.4f} | "
                      f"{(top.get('content') or '')[:40]}...")

    except Exception as e:
        check("检索测试接口可用", False, str(e))

    # ---------------- 清理 ----------------
    section("七、清理测试数据")
    if doc_id:
        req = urllib.request.Request(
            f"{BASE}/api/kb/documents/{doc_id}",
            method="DELETE",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.loads(r.read())
            check("测试文档已删除", True)
        except Exception as e:
            check("测试文档已删除", False, str(e))

    # 删除测试账号的会话（账号本身留着，不影响）
    print(f"     测试账号「{uname}」保留在数据库中（不影响使用）")

    # ---------------- 汇总 ----------------
    total = PASS + FAIL
    print(f"\n{'=' * 72}")
    print(f"  测试完成：{PASS} 项通过 / {FAIL} 项失败（共 {total} 项）")
    print(f"{'=' * 72}")
    if FAIL == 0:
        print("\n  🎉 全部通过！HTTP 接口层工作正常，浏览器可以正常使用了。\n")
    else:
        print(f"\n  ⚠️  有 {FAIL} 项失败，请看上面的 ❌\n")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n  测试被中断。\n")
        sys.exit(130)
    except Exception as exc:
        print(f"\n❌ 测试异常: {type(exc).__name__}: {exc}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
