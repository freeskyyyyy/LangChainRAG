"""真实 Chromium 页面验收；先运行 test_account.py --serve。

使用独立浏览器配置和隔离服务，关闭后不保留测试账号。
截图写入 data/logs/account-desktop.png 和 account-mobile.png。
"""

import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

import websocket


def main(scenario=None, url="http://127.0.0.1:8766/app"):
    browser = next((p for p in [
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
    ] if p.exists()), None)
    if browser is None:
        raise RuntimeError("需要本地 Chrome 或 Edge")

    with tempfile.TemporaryDirectory(prefix="rag-account-ui-") as profile:
        process = subprocess.Popen([
            str(browser), "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", "--remote-debugging-port=0",
            f"--user-data-dir={profile}", "about:blank",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ws = None
        try:
            port_file = Path(profile) / "DevToolsActivePort"
            deadline = time.monotonic() + 20
            while not port_file.exists():
                if time.monotonic() > deadline or process.poll() is not None:
                    raise RuntimeError("无头浏览器启动失败")
                time.sleep(.1)
            time.sleep(.5)
            port = port_file.read_text().splitlines()[0]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list") as response:
                target = next(t for t in json.load(response) if t["type"] == "page")
            ws = websocket.create_connection(target["webSocketDebuggerUrl"], suppress_origin=True, timeout=30)
            sequence = 0

            def call(method, **params):
                nonlocal sequence
                sequence += 1
                ws.send(json.dumps({"id": sequence, "method": method, "params": params}))
                deadline = time.monotonic() + 30
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError(method)
                    result = json.loads(ws.recv())
                    if result.get("id") == sequence:
                        if "error" in result:
                            raise RuntimeError(result["error"])
                        return result.get("result", {})

            def evaluate(source):
                result = call("Runtime.evaluate", expression=source, awaitPromise=True, returnByValue=True)
                if "exceptionDetails" in result:
                    raise AssertionError(result["exceptionDetails"])
                return result.get("result", {}).get("value")

            helpers = """
                const el = id => document.getElementById(id);
                const wait = async predicate => {
                    const deadline = Date.now() + 10000;
                    while (!predicate()) {
                        if (Date.now() > deadline) throw new Error('等待页面状态超时');
                        await new Promise(r => setTimeout(r, 30));
                    }
                };
                const check = (condition, message) => { if (!condition) throw new Error(message); };
                const password = (old, next, confirm = next) => {
                    el('current-password').value = old;
                    el('new-password').value = next;
                    el('confirm-password').value = confirm;
                    el('password-form').requestSubmit();
                };
            """

            def run(source):
                return evaluate("(async () => {" + helpers + source + "})()")

            def ready():
                for _ in range(100):
                    if evaluate("document.readyState === 'complete' && !!document.getElementById('account-btn')"):
                        return
                    time.sleep(.1)
                raise AssertionError("页面未加载")

            call("Page.enable")
            call("Emulation.setDeviceMetricsOverride", width=1280, height=900, deviceScaleFactor=1, mobile=False)
            call("Page.navigate", url=url)
            ready()
            if scenario is not None:
                scenario(call, evaluate, run, ready)
                return
            username = "ui_" + str(time.time_ns())
            print(run("""
                el('to-register').click();
                el('reg-username').value = USERNAME;
                el('reg-password').value = 'Original123';
                el('reg-password2').value = 'Original123';
                el('register-btn').click();
                await wait(() => el('app-view').style.display === 'flex');
                el('account-btn').click();
                check(el('account-dialog').open, '设置弹窗未打开');
                el('profile-nickname').value = '   ';
                el('profile-form').requestSubmit();
                check(el('profile-error').textContent.includes('空白'), '空白昵称未拦截');
                el('profile-nickname').value = '<img src=x onerror=alert(1)>昵称';
                el('profile-form').requestSubmit();
                await wait(() => el('profile-status').textContent.includes('已保存'));
                check(!el('whoami').querySelector('img'), '昵称被解释为 HTML');
                check(el('whoami').textContent.includes('<img'), '昵称保存未生效');
                password('Original123', 'Changed456', 'Mismatch789');
                check(el('password-error').textContent.includes('不一致'), '两次密码不一致未拦截');
                password('Original123', '密'.repeat(25));
                check(el('password-error').textContent.includes('72'), 'UTF-8 字节限制未生效');
                password('Original123', 'Original123');
                check(el('password-error').textContent.includes('相同'), '相同密码未拦截');
                password('Incorrect123', 'Changed456');
                await wait(() => el('password-error').textContent.includes('当前密码不正确'));
                check(el('account-dialog').open && localStorage.getItem('rag_token'), '旧密码错误导致退出');
                check(el('current-password').value === '' && el('new-password').value === '', '请求后密码未清理');
                el('profile-nickname').value = '验收昵称';
                el('profile-form').requestSubmit();
                await wait(() => el('whoami').textContent === '验收昵称');
                return 'PASS: 注册、昵称、XSS、密码校验、错误旧密码保留登录';
            """.replace("USERNAME", json.dumps(username))))

            call("Page.reload")
            time.sleep(.2)
            ready()
            print(run("""
                await wait(() => el('whoami').textContent === '验收昵称');
                el('account-btn').click();
                check(el('profile-nickname').value === '验收昵称', '刷新后昵称未恢复');
                return 'PASS: 刷新后昵称持久化';
            """))
            output = Path(__file__).parent / "data/logs"
            output.mkdir(parents=True, exist_ok=True)
            for name, width, height in [("desktop", 1280, 900), ("mobile", 375, 740)]:
                call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
                time.sleep(.15)
                print(run("""
                    const rect = el('account-dialog').getBoundingClientRect();
                    check(rect.left >= 0 && rect.right <= innerWidth, '弹窗超出屏幕');
                    check(document.documentElement.scrollWidth <= innerWidth, '页面横向溢出');
                    return 'PASS: 宽度 ' + innerWidth + ' 的布局';
                """))
                screenshot = call("Page.captureScreenshot", format="png")
                (output / f"account-{name}.png").write_bytes(base64.b64decode(screenshot["data"]))

            print(run("""
                password('Original123', 'Changed456');
                await wait(() => el('login-view').style.display === 'flex');
                check(!localStorage.getItem('rag_token'), '修改密码后未清除令牌');
                check(!el('account-dialog').open, '修改密码后弹窗未关闭');
                check(el('new-password').value === '', '修改密码后未清空输入');
                el('password').value = 'Changed456';
                el('login-btn').click();
                await wait(() => el('app-view').style.display === 'flex');
                check(el('whoami').textContent === '验收昵称', '新密码登录未恢复用户');
                el('account-btn').click();
                el('current-password').value = 'Changed456';
                el('account-close').click();
                await wait(() => el('current-password').value === '');
                return 'PASS: 改密码退出、新密码登录、关闭弹窗清理';
            """))
            # 模拟令牌过期，确认真正的认证 401 仍会回到登录页。
            print(run("""
                el('account-btn').click();
                state.token = 'expired-test-token';
                password('Changed456', 'Another789');
                await wait(() => el('login-view').style.display === 'flex');
                check(!el('account-dialog').open && !localStorage.getItem('rag_token'), '过期令牌未退出');
                return 'PASS: 无效令牌退出';
            """))
        finally:
            if ws:
                try:
                    ws.send(json.dumps({"id": 999999, "method": "Browser.close"}))
                except Exception:
                    pass
                ws.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
