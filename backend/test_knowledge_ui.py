"""先启动 test_knowledge_ui_api.py --serve，再运行本脚本。

使用真实页面与隔离 API；索引操作由测试服务替代，不调用模型。
"""

import base64
from pathlib import Path
import time

from test_account_ui import main as browser_main


def scenario(call, evaluate, run, ready):
    print(run("""
        el('username').value = 'kb_admin';
        el('password').value = 'Test123456';
        el('login-btn').click();
        await wait(() => document.querySelector('.doc-open[data-id="21"]'));
        check(el('doc-page').textContent === '1 / 2', '文档分页错误');
        el('doc-next').click();
        await wait(() => el('doc-page').textContent === '2 / 2');
        check(document.querySelectorAll('.doc-open').length === 2, '文档尾页数量错误');
        el('doc-prev').click();
        await wait(() => document.querySelector('.doc-open[data-id="21"]'));
        document.querySelector('.doc-open[data-id="21"]').click();
        await wait(() => document.querySelectorAll('.chunk-card').length === 20);
        check(el('kb-dialog').open, '预览未打开');
        check(el('chunk-page').textContent.includes('共 23 段'), '片段总数错误');
        check(!el('kb-filename').querySelector('img'), '文件名 XSS');
        check(!el('chunk-list').querySelector('img,script,b'), '片段或元数据 XSS');
        check(el('chunk-list').textContent.includes('页码 3'), '页码缺失');
        document.querySelector('.chunk-card summary').click();
        check(document.querySelector('.chunk-card details').open, '父片段未展开');
        check(document.querySelector('.chunk-card details').textContent.includes('<script>'), '父片段被解释为 HTML');
        const fetchBeforeFailure = window.fetch;
        window.fetch = async (path, options) => {
            if (String(path).includes('/chunks?page=2')) throw new Error('尾页加载失败');
            return fetchBeforeFailure(path, options);
        };
        el('chunk-next').click();
        await wait(() => !el('kb-retry').hidden);
        check(el('kb-error').textContent.includes('尾页加载失败'), '尾页错误未显示');
        window.fetch = fetchBeforeFailure;
        el('kb-retry').click();
        await wait(() => el('chunk-page').textContent.includes('第 2'));
        check(document.querySelectorAll('.chunk-card').length === 3, '片段尾页数量错误');
        el('chunk-prev').click();
        await wait(() => document.querySelectorAll('.chunk-card').length === 20);
        return 'PASS: 文档分页、片段分页及尾页重试、父片段、定位信息、HTML 转义';
    """))

    output = Path(__file__).parent / "data/logs"
    output.mkdir(parents=True, exist_ok=True)
    for name, width, height in [("desktop", 1280, 900), ("mobile", 375, 740)]:
        call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
        time.sleep(.15)
        print(run("""
            el('kb-dialog').scrollTop = 0;
            const rect = el('kb-dialog').getBoundingClientRect();
            check(rect.left >= 0 && rect.right <= innerWidth, '弹窗超出屏幕');
            check(el('kb-dialog').scrollWidth <= el('kb-dialog').clientWidth, '弹窗内容横向溢出');
            check(document.documentElement.scrollWidth <= innerWidth, '页面横向溢出');
            return 'PASS: 宽度 ' + innerWidth + ' 的片段预览布局';
        """))
        screenshot = call("Page.captureScreenshot", format="png")
        (output / f"knowledge-{name}.png").write_bytes(base64.b64decode(screenshot["data"]))

    print(run("""
        window.originalFetch = window.fetch;
        window.reindexPosts = 0;
        window.fetch = async (path, options) => {
            if (String(path).endsWith('/reindex')) window.reindexPosts++;
            return originalFetch(path, options);
        };
        el('kb-reindex').click();
        check(!el('kb-reindex-confirm').hidden, '没有二次确认');
        el('kb-reindex-cancel').click();
        check(window.reindexPosts === 0, '取消仍发起重建');
        el('kb-reindex').click();
        el('kb-reindex-ok').click();
        el('kb-reindex-ok').click();
        check(el('kb-close').disabled && el('kb-reindex').disabled, '重建未锁定操作');
        const cancel = new Event('cancel', {cancelable: true});
        el('kb-dialog').dispatchEvent(cancel);
        check(cancel.defaultPrevented, '重建中 Escape 未阻止');
        await wait(() => el('kb-notice').textContent.includes('重建完成') && !el('kb-close').disabled);
        check(window.reindexPosts === 1, '重复发起重建');
        check(document.querySelectorAll('.chunk-card').length === 2, '重建后预览未刷新');
        check(el('chunk-list').textContent.includes('重建后的片段'), '仍展示旧片段');
        await wait(() => document.querySelector('.doc-open[data-id="21"] .doc-count').textContent === '2段');
        el('kb-close').click();
        await wait(() => !el('kb-dialog').open && !kbView.doc);
        document.querySelector('.doc-open[data-id="22"]').click();
        await wait(() => !kbView.loading);
        check(el('chunk-list').textContent.includes('暂无片段'), '空状态缺失');
        check(el('kb-doc-error').textContent.includes('测试文件丢失'), '失败原因缺失');
        el('kb-reindex').click();
        el('kb-reindex-ok').click();
        await wait(() => el('kb-error').textContent.includes('原始文件不存在') && !el('kb-close').disabled);
        check(!el('kb-reindex').disabled, '失败后无法重试');
        el('kb-close').click();
        await wait(() => !kbView.doc);
        document.querySelector('.doc-open[data-id="20"]').click();
        await wait(() => !kbView.loading);
        check(el('kb-reindex').disabled, '处理中仍可再次重建');
        el('kb-close').click();
        await wait(() => !kbView.doc);
        return 'PASS: 重建确认、取消、防重复、成功刷新、失败恢复、处理中限制';
    """))

    print(run("""
        window.fetch = async (path, options) => {
            if (String(path).includes('/21/chunks')) throw new Error('模拟断网');
            return originalFetch(path, options);
        };
        document.querySelector('.doc-open[data-id="21"]').click();
        await wait(() => !el('kb-retry').hidden);
        check(el('kb-error').textContent.includes('模拟断网'), '加载失败未提示');
        window.fetch = originalFetch;
        el('kb-retry').click();
        await wait(() => document.querySelectorAll('.chunk-card').length === 2);
        el('kb-close').click();
        await wait(() => !kbView.doc);
        // 延迟旧请求：关闭 A、打开 B 后，A 的结果不得覆盖 B。
        window.releaseChunks = null;
        window.fetch = async (path, options) => {
            if (String(path).includes('/21/chunks')) {
                await new Promise(resolve => { window.releaseChunks = resolve; });
            }
            return originalFetch(path, options);
        };
        document.querySelector('.doc-open[data-id="21"]').click();
        await wait(() => window.releaseChunks !== null);
        el('kb-close').click();
        await wait(() => !kbView.doc);
        document.querySelector('.doc-open[data-id="22"]').click();
        await wait(() => !kbView.loading);
        window.releaseChunks();
        await new Promise(resolve => setTimeout(resolve, 300));
        check(el('kb-filename').textContent === '资料22.txt', '旧请求覆盖了新文档');
        window.fetch = originalFetch;
        state.token = 'expired-test-token';
        await loadChunks();
        check(!el('kb-dialog').open && !localStorage.getItem('rag_token'), '失效登录未关闭预览');
        el('username').value = 'kb_user';
        el('password').value = 'Test123456';
        el('login-btn').click();
        await wait(() => el('app-view').style.display === 'flex');
        check(el('kb-section').style.display === 'none', '普通用户看到了知识库管理');
        return 'PASS: 断网重试、旧请求隔离、登录过期、普通用户隐藏入口';
    """))


if __name__ == "__main__":
    browser_main(scenario, "http://127.0.0.1:8767/app")
