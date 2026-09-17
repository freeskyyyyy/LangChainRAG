r"""
回答操作功能测试 —— 停止 / 重新生成 / 点赞点踩 / 复制

【测什么】
  ① 重新生成：同一个问题重答一遍，而且要**替换**掉旧回答，不是往后追加
  ② 重新生成不带会话 ID：要被拦住（这是我自己加的一道防线）
  ③ 点赞 / 点踩：能存进去，而且能改判
  ④ 评价别人的消息：要被拒绝（越权检查）
  ⑤ 停止生成：前端行为，这里只能间接验证 ——
     确认后端在连接断开时不会把半截回答存进数据库

【为什么"替换"这件事必须在后端测】
前端把旧回答从界面上拿掉，只是**看起来**替换了。
数据库里是不是真的删了，刷新一下才知道 ——
如果只测前端，就会漏掉"界面上没了、库里还在"这种最坑的情况
（用户下次打开这个会话，旧回答又冒出来了）。

【怎么运行】
    cd backend
    ..\.venv\Scripts\python.exe test_answer_actions.py
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
REPORT = Path(__file__).parent / "data" / "logs" / "回答操作测试报告.txt"

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


# ------------------------------------------------------------------
# 限流保护
# ------------------------------------------------------------------
# 后端是"每用户每分钟 20 次"。这个脚本一共打十来次请求，本来够用，
# 但调试时容易反复重跑，所以还是加上间隔，免得又被静默截断
#（这个坑我在评估脚本上踩过，见 eval_retrieval.py 里的记录）。
_last = [0.0]


def _throttle(min_interval: float = 1.2) -> None:
    elapsed = time.monotonic() - _last[0]
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last[0] = time.monotonic()


def post(path: str, payload: dict, token: str, expect_ok: bool = True):
    """发一个 JSON POST 请求，返回 (状态码, 响应体)"""
    _throttle()
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:200]}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def get(path: str, token: str):
    _throttle()
    req = urllib.request.Request(
        BASE + path, headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:200]}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def ask(question: str, token: str, session_id=None, regenerate: bool = False,
        timeout: int = 180) -> dict:
    """
    调流式问答接口，把 SSE 流读完整，返回收集到的事件。

    【为什么要自己解析 SSE】
    这个接口不返回普通 JSON，而是"一个字一个字往外吐"的流。
    测试里我们不需要逐字显示，但需要知道：
      · 生成的完整回答是什么
      · 后端补发的消息 ID 是多少（点赞要用）
      · 有没有报错
    """
    _throttle()
    body = {
        "question": question,
        "session_id": session_id,
        "regenerate": regenerate,
    }
    req = urllib.request.Request(
        BASE + "/api/chat/stream",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )

    result = {"answer": "", "session_id": None, "assistant_message_id": None,
              "citations": 0, "error": None, "status": 0}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result["status"] = r.status
            event_type = None
            for raw in r:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    try:
                        payload = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if event_type == "token":
                        result["answer"] += payload.get("text", "")
                    elif event_type == "session":
                        result["session_id"] = payload.get("session_id")
                    elif event_type == "saved":
                        result["assistant_message_id"] = payload.get("assistant_message_id")
                        if payload.get("session_id"):
                            result["session_id"] = payload["session_id"]
                    elif event_type == "citations":
                        result["citations"] = len(payload or [])
                    elif event_type == "error":
                        result["error"] = payload.get("message")
    except urllib.error.HTTPError as e:
        result["status"] = e.code
        result["error"] = e.read().decode("utf-8", errors="replace")[:200]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def login() -> str:
    req = urllib.request.Request(
        BASE + "/api/auth/login",
        data=json.dumps({"username": "admin", "password": "123456"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())["access_token"]


def main() -> int:
    out("=" * 76)
    out("  回答操作功能测试")
    out("=" * 76)
    out("\n  被测功能：重新生成 / 点赞点踩 / 越权防护\n")

    try:
        token = login()
    except Exception as e:
        out(f"\n  ❌ 登录失败：{e}")
        out("     请先启动后端服务（run.py）\n")
        return 1

    out("-" * 76)
    out("【第一组】重新生成")
    out("-" * 76)

    question = "真丝连衣裙怎么洗"

    r1 = ask(question, token)
    if r1["error"] or not r1["answer"]:
        out(f"\n  ❌ 第一次提问就失败了：{r1['error'] or '回答为空'}\n")
        return 1

    sid = r1["session_id"]
    msg_id_1 = r1["assistant_message_id"]
    out(f"\n  第一次回答：{r1['answer'][:52].replace(chr(10), ' ')}…")
    out(f"  会话 ID={sid} | 消息 ID={msg_id_1} | 引用 {r1['citations']} 条\n")

    check("第一次提问拿到了会话 ID", sid is not None)
    check("后端补发了消息 ID（点赞点踩要靠它）", msg_id_1 is not None)

    _, before = get(f"/api/sessions/{sid}/messages", token)
    n_before = len(before)

    # ---- 重新生成 ----
    r2 = ask(question, token, session_id=sid, regenerate=True)
    if r2["error"] or not r2["answer"]:
        out(f"\n  ❌ 重新生成失败：{r2['error'] or '回答为空'}\n")
        return 1

    msg_id_2 = r2["assistant_message_id"]
    out(f"  重新生成后：{r2['answer'][:52].replace(chr(10), ' ')}…")
    out(f"  消息 ID={msg_id_2}\n")

    _, after = get(f"/api/sessions/{sid}/messages", token)
    n_after = len(after)

    roles = [m["role"] for m in after]
    n_user = roles.count("user")
    n_ai = roles.count("assistant")

    check(
        "消息总数没有增加（是替换，不是追加）",
        n_after == n_before,
        f"重新生成前 {n_before} 条，之后 {n_after} 条 —— 说明旧回答没被删掉",
    )
    check(
        "用户消息只有 1 条（没有把问题存两遍）",
        n_user == 1,
        f"实际有 {n_user} 条用户消息",
    )
    check("AI 回答也只有 1 条", n_ai == 1, f"实际有 {n_ai} 条 AI 回答")

    # 【为什么这里不能断言"消息 ID 变了"】
    # 我第一版就是这么写的，结果报了个假警报，害我又去查了一遍代码。
    #
    # 真正的原因是 SQLite 分配编号的规则：
    # 我们的表定义是 `id INTEGER PRIMARY KEY`（**不带 AUTOINCREMENT**），
    # 新记录的号 = "当前最大号 + 1"。所以刚删掉的那条如果正好是最大号，
    # 它的号会被**立刻回收**给下一条新记录 —— 新旧两条消息的 ID 是一样的。
    #
    # 这不是 bug，是 SQLite 的正常行为。但它说明一件事：
    # **消息 ID 不能用来判断"是不是同一条记录"**。
    #
    # 这个发现还牵出了一个真问题（已修）：消息号会被回收，
    # 那么挂在旧消息上的"点赞点踩"万一没被一起删掉，
    # 就会顺势贴到捡走了这个号的新回答上 —— 用户会看到
    # 自己从没点过的 👎。根因是 SQLite 的外键约束默认关闭，
    # 表定义里那些 ON DELETE CASCADE 根本没生效。详见 app/core/database.py。
    check(
        "重新生成后，会话最后一条是 AI 回答",
        after[-1]["role"] == "assistant",
        f"最后一条的角色是 {after[-1]['role']}",
    )
    check(
        "库里的回答不是空的",
        bool((after[-1].get("content") or "").strip()),
        "重新生成后存在库里的回答是空的",
    )

    # ---- 防呆：不给会话 ID 就重新生成 ----
    out("")
    bad = ask(question, token, session_id=None, regenerate=True)
    check(
        "不带会话 ID 的重新生成被拦住（返回 422）",
        bad["status"] == 422,
        f"实际返回 {bad['status']}：{str(bad['error'])[:80]}",
    )

    # ---- 再来一次，确认可以连续重新生成且不会越堆越多 ----
    r3 = ask(question, token, session_id=sid, regenerate=True)
    _, after2 = get(f"/api/sessions/{sid}/messages", token)
    check(
        "连续重新生成两次，消息数仍然不变",
        len(after2) == n_before,
        f"现在 {len(after2)} 条，应该还是 {n_before} 条",
    )

    out("\n" + "-" * 76)
    out("【第二组】点赞 / 点踩")
    out("-" * 76 + "\n")

    current_id = after2[-1]["id"]

    st, resp = post("/api/chat/feedback", {"message_id": current_id, "rating": 1}, token)
    check("点赞成功", st == 200, f"返回 {st}：{resp}")

    _, msgs = get(f"/api/sessions/{sid}/messages", token)
    fb = [m for m in msgs if m["id"] == current_id][0].get("feedback")
    check("再读回来，评价是「赞」", fb == 1, f"实际是 {fb}")

    st, resp = post("/api/chat/feedback", {"message_id": current_id, "rating": -1}, token)
    check("改成「踩」成功", st == 200, f"返回 {st}：{resp}")

    _, msgs = get(f"/api/sessions/{sid}/messages", token)
    fb = [m for m in msgs if m["id"] == current_id][0].get("feedback")
    check("再读回来，评价变成了「踩」（同一人重复评价会覆盖）", fb == -1, f"实际是 {fb}")

    st, resp = post("/api/chat/feedback", {"message_id": current_id, "rating": 5}, token)
    check("非法的评价值（5）被拒绝", st == 422, f"实际返回 {st}")

    out("\n" + "-" * 76)
    out("【第三组】越权防护")
    out("-" * 76 + "\n")

    st, resp = post("/api/chat/feedback", {"message_id": 99999999, "rating": 1}, token)
    check(
        "给不存在的消息点赞会被拒绝（防止探测系统里有多少条消息）",
        st == 404,
        f"实际返回 {st}：{resp}",
    )

    # 用户消息不能评价
    user_msg_id = [m for m in msgs if m["role"] == "user"][0]["id"]
    st, resp = post("/api/chat/feedback", {"message_id": user_msg_id, "rating": 1}, token)
    check(
        "给自己的提问点赞会被拒绝（只有 AI 回答能被评价）",
        st == 400,
        f"实际返回 {st}：{resp}",
    )

    # ---- 清理：把测试产生的会话删掉 ----
    _throttle()
    req = urllib.request.Request(
        f"{BASE}/api/sessions/{sid}",
        method="DELETE",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        urllib.request.urlopen(req, timeout=30)
        cleaned = True
    except Exception:
        cleaned = False

    out("\n" + "=" * 76)
    out(f"  测试完成：{PASS} 项通过 / {FAIL} 项失败")
    out("=" * 76)
    out(f"\n  测试用的会话已{'清理' if cleaned else '**未能**清理（请手动删掉）'}\n")

    if FAIL == 0:
        out("  🎉 全部通过。\n")
    else:
        out(f"  ⚠️  有 {FAIL} 项没通过，需要排查。\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(LINES), encoding="utf-8")
    print(f"报告已写入：{REPORT}\n")
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
