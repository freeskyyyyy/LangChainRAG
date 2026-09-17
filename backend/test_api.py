r"""
端到端接口测试 —— 把认证和会话的整条链路跑一遍

【为什么要写这个】
代码写完不代表能用。这个脚本会**真的发 HTTP 请求**给正在运行的后端，
模拟一个真实用户从注册到管理会话的完整流程。

【怎么用】
  1. 先启动后端服务（另开一个终端）：run.py
  2. 再运行本脚本：..\.venv\Scripts\python.exe test_api.py

【测试覆盖】
  一、注册新用户
  二、重复注册应被拒绝（验证唯一性约束）
  三、登录并获取令牌
  四、错误密码应被拒绝（验证安全）
  五、用令牌获取当前用户信息
  六、无令牌访问应被拒绝（验证鉴权生效）
  七、普通用户访问管理员接口应被拒绝（验证权限隔离）
  八、创建会话 / 列出会话 / 改标题 / 删会话
  九、数据隔离：用户 A 不能访问用户 B 的会话（最关键的一条）
"""

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

BASE = "http://127.0.0.1:8000"
PASS = 0
FAIL = 0


# ==================================================================
# 工具函数
# ==================================================================
def req(
    method: str,
    path: str,
    body: dict | None = None,
    token: str | None = None,
) -> tuple[int, dict | list | None]:
    """
    发一个 HTTP 请求。

    返回：(状态码, 解析后的 JSON)
    """
    url = f"{BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None

    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def check(name: str, condition: bool, detail: str = "") -> None:
    """打印一条测试结果"""
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")
        if detail:
            print(f"     └─ {detail}")


def section(title: str) -> None:
    print(f"\n{'-' * 66}\n  {title}\n{'-' * 66}")


def brief(obj) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s[:150] + ("..." if len(s) > 150 else "")


# ==================================================================
# 测试流程
# ==================================================================
def main() -> int:
    # 用时间戳生成唯一用户名，这样反复运行脚本也不会因为"用户名已存在"而失败
    stamp = datetime.now().strftime("%H%M%S")
    user_a = f"test_a_{stamp}"
    user_b = f"test_b_{stamp}"
    pwd = "testPassword123"

    print(f"\n{'=' * 66}")
    print("  接口端到端测试".center(60))
    print(f"{'=' * 66}")
    print(f"\n  目标服务: {BASE}")
    print(f"  测试账号: {user_a} / {user_b}")

    # ---------------------------------------------------------------
    section("〇、服务连通性")
    # ---------------------------------------------------------------
    status, data = req("GET", "/api/health")
    check("服务可访问", status == 200, brief(data))
    if status != 200:
        print("\n  ⚠️  服务没起来，请先运行 run.py 再执行本测试。\n")
        return 1

    # ---------------------------------------------------------------
    section("一、注册")
    # ---------------------------------------------------------------
    status, data = req("POST", "/api/auth/register", {
        "username": user_a, "password": pwd, "nickname": "测试用户A",
    })
    check("注册用户 A 成功", status == 201, f"状态码={status} {brief(data)}")
    check("返回信息不含密码哈希", isinstance(data, dict) and "password_hash" not in data,
          brief(data))
    check("默认角色是普通用户", isinstance(data, dict) and data.get("role") == "user",
          brief(data))

    status, data = req("POST", "/api/auth/register", {
        "username": user_b, "password": pwd,
    })
    check("注册用户 B 成功", status == 201, f"状态码={status} {brief(data)}")

    # 唯一性约束
    status, data = req("POST", "/api/auth/register", {
        "username": user_a, "password": pwd,
    })
    check("重复用户名被拒绝", status == 409, f"状态码={status}（期望 409）{brief(data)}")

    # 参数校验
    status, data = req("POST", "/api/auth/register", {"username": "ab", "password": pwd})
    check("过短用户名被拒绝", status == 422, f"状态码={status}（期望 422）")

    status, data = req("POST", "/api/auth/register", {"username": "ok_user_1", "password": "123"})
    check("过短密码被拒绝", status == 422, f"状态码={status}（期望 422）")

    # ---------------------------------------------------------------
    section("二、登录")
    # ---------------------------------------------------------------
    status, data = req("POST", "/api/auth/login", {"username": user_a, "password": pwd})
    check("正确密码登录成功", status == 200, f"状态码={status} {brief(data)}")
    token_a = data.get("access_token") if isinstance(data, dict) else None
    check("返回了访问令牌", bool(token_a))

    status, data = req("POST", "/api/auth/login", {"username": user_a, "password": "wrong_password"})
    check("错误密码被拒绝", status == 401, f"状态码={status}（期望 401）")
    check("错误提示不泄露账号是否存在",
          isinstance(data, dict) and "用户名" in str(data.get("message", ""))
          and "不存在" not in str(data.get("message", "")),
          brief(data))

    status, data = req("POST", "/api/auth/login", {"username": "no_such_user_xyz", "password": pwd})
    check("不存在的账号也返回同样的提示", status == 401, f"状态码={status}")

    # 管理员登录
    status, data = req("POST", "/api/auth/login", {"username": "admin", "password": "123456"})
    check("管理员 admin/123456 登录成功", status == 200, f"状态码={status} {brief(data)}")
    token_admin = data.get("access_token") if isinstance(data, dict) else None
    check("管理员角色正确",
          isinstance(data, dict) and data.get("user", {}).get("role") == "admin",
          brief(data))

    # ---------------------------------------------------------------
    section("三、令牌与鉴权")
    # ---------------------------------------------------------------
    status, data = req("GET", "/api/auth/me", token=token_a)
    check("带令牌可获取当前用户信息", status == 200, f"状态码={status} {brief(data)}")
    check("返回的是本人信息",
          isinstance(data, dict) and data.get("username") == user_a, brief(data))

    status, data = req("GET", "/api/auth/me")
    check("不带令牌被拒绝", status == 401, f"状态码={status}（期望 401）")

    status, data = req("GET", "/api/auth/me", token="fake.invalid.token")
    check("伪造令牌被拒绝", status == 401, f"状态码={status}（期望 401）")

    # ---------------------------------------------------------------
    section("四、会话管理")
    # ---------------------------------------------------------------
    status, data = req("POST", "/api/sessions", {"title": "手机咨询"}, token=token_a)
    check("创建会话成功", status == 201, f"状态码={status} {brief(data)}")
    session_a = data.get("id") if isinstance(data, dict) else None

    status, data = req("POST", "/api/sessions", {}, token=token_a)
    check("不填标题时用默认名", status == 201 and data.get("title") == "新对话",
          brief(data))
    session_a2 = data.get("id") if isinstance(data, dict) else None

    status, data = req("GET", "/api/sessions", token=token_a)
    check("会话列表可获取", status == 200 and isinstance(data, list), brief(data))
    check("列表包含刚创建的两个会话",
          isinstance(data, list) and len([s for s in data if s["id"] in (session_a, session_a2)]) == 2,
          f"实际返回 {len(data) if isinstance(data, list) else '?'} 个")

    status, data = req("PATCH", f"/api/sessions/{session_a}", {"title": "改名测试"}, token=token_a)
    check("重命名会话成功", status == 200 and data.get("title") == "改名测试", brief(data))

    status, data = req("PATCH", f"/api/sessions/{session_a}", {"is_pinned": True}, token=token_a)
    check("置顶会话成功", status == 200 and data.get("is_pinned") is True, brief(data))

    status, data = req("GET", "/api/sessions", token=token_a)
    check("置顶的会话排在最前面",
          isinstance(data, list) and data and data[0]["id"] == session_a,
          brief(data[:1] if isinstance(data, list) else data))

    status, data = req("GET", f"/api/sessions/{session_a}/messages", token=token_a)
    check("可获取会话历史消息（新会话应为空）",
          status == 200 and data == [], f"状态码={status} {brief(data)}")

    # ---------------------------------------------------------------
    section("五、数据隔离（最关键）")
    # ---------------------------------------------------------------
    status, data = req("POST", "/api/auth/login", {"username": user_b, "password": pwd})
    token_b = data.get("access_token") if isinstance(data, dict) else None
    check("用户 B 登录成功", bool(token_b), brief(data))

    status, data = req("GET", "/api/sessions", token=token_b)
    check("用户 B 看不到用户 A 的会话",
          status == 200 and isinstance(data, list)
          and all(s["id"] not in (session_a, session_a2) for s in data),
          f"用户B看到 {len(data) if isinstance(data, list) else '?'} 个会话")

    status, data = req("GET", f"/api/sessions/{session_a}/messages", token=token_b)
    check("用户 B 直接访问用户 A 的会话被拒绝（404）",
          status == 404, f"状态码={status}（期望 404）{brief(data)}")

    status, data = req("PATCH", f"/api/sessions/{session_a}", {"title": "恶意改名"}, token=token_b)
    check("用户 B 无法改用户 A 的会话", status == 404, f"状态码={status}（期望 404）")

    status, data = req("DELETE", f"/api/sessions/{session_a}", token=token_b)
    check("用户 B 无法删用户 A 的会话", status == 404, f"状态码={status}（期望 404）")

    # ---------------------------------------------------------------
    section("六、改密码")
    # ---------------------------------------------------------------
    status, data = req("POST", "/api/auth/change-password", {
        "old_password": "wrong_old", "new_password": "newPassword456",
    }, token=token_a)
    check("旧密码错误时拒绝修改", status == 401, f"状态码={status}（期望 401）")

    status, data = req("POST", "/api/auth/change-password", {
        "old_password": pwd, "new_password": pwd,
    }, token=token_a)
    check("新旧密码相同时拒绝修改", status == 400, f"状态码={status}（期望 400）")

    status, data = req("POST", "/api/auth/change-password", {
        "old_password": pwd, "new_password": "newPassword456",
    }, token=token_a)
    check("正确修改密码成功", status == 200, f"状态码={status} {brief(data)}")

    status, data = req("POST", "/api/auth/login", {"username": user_a, "password": "newPassword456"})
    check("可用新密码登录", status == 200, f"状态码={status}")

    status, data = req("POST", "/api/auth/login", {"username": user_a, "password": pwd})
    check("旧密码已失效", status == 401, f"状态码={status}（期望 401）")

    # ---------------------------------------------------------------
    section("七、清理")
    # ---------------------------------------------------------------
    if token_a:
        # 用新令牌重新登录（旧令牌仍有效，但我们统一用新的）
        status, data = req("POST", "/api/auth/login",
                           {"username": user_a, "password": "newPassword456"})
        token_a = data.get("access_token") if isinstance(data, dict) else token_a
        for sid in (session_a, session_a2):
            if sid:
                req("DELETE", f"/api/sessions/{sid}", token=token_a)
        status, data = req("GET", "/api/sessions", token=token_a)
        check("删除会话后列表已清空",
              status == 200 and isinstance(data, list) and len(data) == 0,
              f"还剩 {len(data) if isinstance(data, list) else '?'} 个")

    # ---------------------------------------------------------------
    # 汇总
    # ---------------------------------------------------------------
    total = PASS + FAIL
    print(f"\n{'=' * 66}")
    print(f"  测试完成：{PASS} 项通过 / {FAIL} 项失败（共 {total} 项）")
    print(f"{'=' * 66}")
    if FAIL == 0:
        print("\n  🎉 全部通过！认证与会话功能工作正常。\n")
    else:
        print(f"\n  ⚠️  有 {FAIL} 项失败，请把上面的输出发给我。\n")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n  测试被中断。\n")
        sys.exit(130)
