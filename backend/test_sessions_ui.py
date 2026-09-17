r"""
多会话功能端到端测试 —— 模拟新前端"会话列表"那一整条链路

【为什么单独测这个】
新前端最核心的新功能就是多会话。它涉及一串前后端配合的动作：
    点"新建对话" → 提问 → 后端自动建会话 → 侧栏出现新会话
    → 切到旧会话 → 历史消息能加载回来

这一串里任何一环断了，用户看到的就是"侧栏是空的"或者
"点历史会话进去什么都没有"。这类问题在页面上很难排查，
但用脚本按顺序打一遍接口，哪里断了立刻就能看出来。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe test_sessions_ui.py
（需要后端服务已启动）
"""

import json
import sys
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
REPORT = Path(__file__).parent / "data" / "logs" / "多会话测试报告.txt"
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


def req(method: str, path: str, token: str | None = None,
        body: dict | None = None) -> tuple[int, object]:
    headers = {}
    data = None
    if token:
        headers["Authorization"] = "Bearer " + token
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")

    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=180) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw[:200]


def stream_chat(token: str, question: str, session_id: int | None) -> dict:
    """发一条消息，把 SSE 流解析成结构化结果"""
    body = {"question": question}
    if session_id is not None:
        body["session_id"] = session_id

    r = urllib.request.Request(
        BASE + "/api/chat/stream",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
        },
    )

    result = {"session_id": None, "title": None, "answer": "", "citations": [], "error": None}

    with urllib.request.urlopen(r, timeout=180) as resp:
        buffer = ""
        while True:
            raw = resp.read(2048)
            if not raw:
                break
            buffer += raw.decode("utf-8", errors="replace")
            blocks = buffer.split("\n\n")
            buffer = blocks.pop()

            for block in blocks:
                if not block.strip():
                    continue
                etype, dstr = "message", ""
                for line in block.split("\n"):
                    if line.startswith("event:"):
                        etype = line[6:].strip()
                    elif line.startswith("data:"):
                        dstr += line[5:].strip()
                if not dstr:
                    continue
                try:
                    payload = json.loads(dstr)
                except json.JSONDecodeError:
                    continue

                if etype == "session":
                    result["session_id"] = payload.get("session_id")
                    result["title"] = payload.get("title")
                elif etype == "token":
                    result["answer"] += payload.get("text", "")
                elif etype == "citations":
                    result["citations"] = payload or []
                elif etype == "error":
                    result["error"] = payload

    return result


def main() -> int:
    out("=" * 74)
    out("  多会话功能端到端测试")
    out("=" * 74)

    # ---------- 登录 ----------
    out("\n【1】登录")
    status, data = req("POST", "/api/auth/login",
                       body={"username": "admin", "password": "123456"})
    check("管理员登录成功", status == 200, f"状态码 {status}")
    if status != 200 or not isinstance(data, dict):
        return 1
    token = data["access_token"]

    # ---------- 记录初始状态 ----------
    status, before = req("GET", "/api/sessions", token)
    check("会话列表接口可用", status == 200, f"状态码 {status}")
    n_before = len(before) if isinstance(before, list) else 0
    out(f"     当前已有 {n_before} 个会话")

    # ---------- 第一次提问应自动建会话 ----------
    out("\n【2】模拟点『新建对话』后提问（session_id 传 null）")
    out("     这一步验证：后端会不会自动建会话，并把标题推回来\n")

    r1 = stream_chat(token, "我身高168体重120斤，穿什么尺码", None)
    check("后端返回了新建的会话 ID", r1["session_id"] is not None,
          f"实际收到：{r1['session_id']}")
    check("会话标题根据问题自动生成", bool(r1["title"]),
          f"实际标题：{r1['title']!r}")
    check("生成了回答", len(r1["answer"]) > 10,
          f"回答长度 {len(r1['answer'])}")
    out(f"     会话 ID: {r1['session_id']}")
    out(f"     自动标题: {r1['title']}")
    out(f"     回答前 90 字: {r1['answer'][:90]}")

    sid1 = r1["session_id"]

    # ---------- 侧栏应出现新会话 ----------
    out("\n【3】侧栏列表应出现这个新会话")
    status, after = req("GET", "/api/sessions", token)
    n_after = len(after) if isinstance(after, list) else 0
    check("会话总数 +1", n_after == n_before + 1,
          f"提问前 {n_before} → 提问后 {n_after}")

    found = next((s for s in (after or []) if s["id"] == sid1), None)
    check("新会话出现在列表里", found is not None)
    if found:
        out(f"     列表中的标题：{found['title']}")
        out(f"     消息条数：{found['message_count']}（一问一答应为 2）")
        check("消息条数正确", found["message_count"] == 2,
              f"实际 {found['message_count']}")

    # ---------- 追问：应进同一个会话 ----------
    out("\n【4】在同一会话里追问（会话应累积消息，不新建）")
    r2 = stream_chat(token, "那体重100斤身高160呢", sid1)

    # 【这里为什么允许 None】
    # 后端只在两种情况下才推 session 事件：
    #   1. 用户没指定会话，后端新建了一个
    #   2. 旧会话的标题还是默认的"新对话"，这次要补上真标题
    # 追问时这两个条件都不满足，所以**正确行为就是不推事件**。
    # 前端此时本来就已经知道自己在哪个会话里（currentSessionId），
    # 不需要后端再告诉它一遍。
    # 所以判断标准是：要么没推事件，要么推的必须是同一个会话 ID。
    check("追问没有跑到别的会话去",
          r2["session_id"] in (None, sid1),
          f"期望 None 或 {sid1}，实际 {r2['session_id']}")
    check("追问生成了回答", len(r2["answer"]) > 5)

    status, again = req("GET", "/api/sessions", token)
    n_again = len(again) if isinstance(again, list) else 0
    check("会话总数没有增加", n_again == n_after,
          f"{n_after} → {n_again}")

    # ---------- 第二个会话 ----------
    out("\n【5】再开一个新对话（验证能同时存在多个会话）")
    r3 = stream_chat(token, "真丝连衣裙怎么洗", None)
    sid2 = r3["session_id"]
    check("建立了第二个独立会话", sid2 is not None and sid2 != sid1,
          f"第一个 {sid1}，第二个 {sid2}")

    status, multi = req("GET", "/api/sessions", token)
    check("现在列表里有两个会话", len(multi) == n_before + 2,
          f"实际 {len(multi)}")

    # ---------- 切回旧会话，历史消息应完整 ----------
    out("\n【6】切回第一个会话（模拟点侧栏里的历史会话）")
    status, msgs = req("GET", f"/api/sessions/{sid1}/messages", token)
    check("历史消息接口可用", status == 200, f"状态码 {status}")

    if isinstance(msgs, list):
        roles = [m["role"] for m in msgs]
        out(f"     取回 {len(msgs)} 条消息，角色序列：{roles}")
        check("历史消息条数正确（两问两答 = 4）", len(msgs) == 4,
              f"实际 {len(msgs)}")
        check("角色是 user/assistant 交替",
              roles == ["user", "assistant", "user", "assistant"],
              f"实际 {roles}")

        ai_msgs = [m for m in msgs if m["role"] == "assistant"]
        check("AI 回答的引用片段被保存下来了",
              any(m.get("citations") for m in ai_msgs),
              "引用为空，历史记录里引用卡片会消失")
        check("AI 回答的耗时被保存下来了",
              any(m.get("latency_ms") for m in ai_msgs))

        if ai_msgs and ai_msgs[0].get("citations"):
            c = ai_msgs[0]["citations"][0]
            out(f"     第一条回答的第1个引用：{c.get('file_name')} "
                f"(相关度 {c.get('score')})")
    else:
        check("历史消息是列表", False, str(msgs)[:120])

    # ---------- 重命名 / 置顶 ----------
    out("\n【7】重命名与置顶")
    status, renamed = req("PATCH", f"/api/sessions/{sid1}", token,
                          {"title": "尺码咨询（测试改名）"})
    check("重命名成功", status == 200, f"状态码 {status}")
    if isinstance(renamed, dict):
        check("标题已更新", renamed.get("title") == "尺码咨询（测试改名）",
              f"实际 {renamed.get('title')!r}")

    status, pinned = req("PATCH", f"/api/sessions/{sid1}", token, {"is_pinned": True})
    check("置顶成功", status == 200 and isinstance(pinned, dict) and pinned.get("is_pinned"))

    status, ordered = req("GET", "/api/sessions", token)
    if isinstance(ordered, list) and ordered:
        check("置顶的会话排在最前面", ordered[0]["id"] == sid1,
              f"第一位是 {ordered[0]['id']}，期望 {sid1}")
        out(f"     列表首位：{ordered[0]['title']}")

    # ---------- 归属校验 ----------
    out("\n【8】安全：别人的会话能不能被看到")
    import time as _t

    uname = f"multitest{int(_t.time()) % 100000}"
    status, _ = req("POST", "/api/auth/register",
                    body={"username": uname, "password": "test123456"})
    status, u = req("POST", "/api/auth/login",
                    body={"username": uname, "password": "test123456"})
    if isinstance(u, dict):
        utoken = u["access_token"]
        status, mine = req("GET", "/api/sessions", utoken)
        check("新用户看不到别人的会话",
              isinstance(mine, list) and len(mine) == 0,
              f"竟然看到了 {len(mine) if isinstance(mine, list) else '?'} 个")

        status, _ = req("GET", f"/api/sessions/{sid1}/messages", utoken)
        check("新用户读不了别人的会话内容（应 403 或 404）",
              status in (403, 404), f"状态码 {status}")
    else:
        check("测试用户登录", False, "注册或登录失败")

    # ---------- 清理 ----------
    out("\n【9】清理测试产生的会话")
    for sid in (sid1, sid2):
        if sid:
            status, _ = req("DELETE", f"/api/sessions/{sid}", token)
            out(f"     删除会话 {sid} → 状态码 {status}")

    status, final = req("GET", "/api/sessions", token)
    check("测试会话已清理干净",
          isinstance(final, list) and len(final) == n_before,
          f"期望 {n_before}，实际 {len(final) if isinstance(final, list) else '?'}")

    # ---------- 汇总 ----------
    total = PASS + FAIL
    out("\n" + "=" * 74)
    out(f"  测试完成：{PASS} 项通过 / {FAIL} 项失败（共 {total} 项）")
    out("=" * 74)
    if FAIL == 0:
        out("\n  🎉 多会话这条链路完全打通，页面上可以正常用了。\n")
    else:
        out(f"\n  ⚠️  有 {FAIL} 项失败，请看上面的 ❌\n")

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
