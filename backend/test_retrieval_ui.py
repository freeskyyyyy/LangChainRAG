"""先启动 test_knowledge_ui_api.py --serve；检索器替代，不调用模型。"""

import base64
from pathlib import Path
import time

from test_account_ui import main as browser_main


def scenario(call, evaluate, run, ready):
    print(run("""
        el('username').value = 'kb_admin';
        el('password').value = 'Test123456';
        el('login-btn').click();
        await wait(() => document.querySelector('.doc-open'));
        el('retrieval-open').click();
        check(el('retrieval-dialog').open && document.activeElement === el('retrieval-query'), '弹窗焦点错误');
        el('retrieval-query').value = '   ';
        await runRetrieval();
        check(el('retrieval-error').textContent.includes('请输入'), '空白问题未校验');
        el('retrieval-query').value = '保养';
        el('retrieval-top-k').value = '21';
        await runRetrieval();
        check(el('retrieval-error').textContent.includes('1～20'), '条数未校验');
        el('retrieval-top-k').value = '2';
        el('retrieval-compare').checked = true;
        window.actualFetch = window.fetch;
        window.searchPosts = 0;
        window.fetch = async (path, options) => {
            if (String(path).endsWith('/kb/search')) { window.searchPosts++; await new Promise(r => setTimeout(r, 200)); }
            return actualFetch(path, options);
        };
        el('retrieval-form').requestSubmit();
        el('retrieval-form').requestSubmit();
        check(el('retrieval-submit').disabled, '检索未锁定');
        await wait(() => el('retrieval-status').textContent.includes('检索完成'));
        check(searchPosts === 1, '重复发起检索');
        check(document.querySelectorAll('#retrieval-results > .chunk-card').length === 2, '返回条数错误');
        check(el('retrieval-results').textContent.includes('重排序分数 0.0000'), '零分显示错误');
        check(el('retrieval-results').textContent.includes('总耗时 6.5 ms'), '耗时缺失');
        check(el('retrieval-results').textContent.includes('两路均命中'), '召回来源缺失');
        check(el('retrieval-results').textContent.includes('工作表 <b>一</b>'), '工作表缺失');
        check(!el('retrieval-results').querySelector('script,img,b:not(li b)'), '结果被解释为 HTML');
        check(document.querySelectorAll('.retrieval-comparison h3').length === 4, '三种方案缺失');
        document.querySelector('#retrieval-results details summary').click();
        check(document.querySelector('#retrieval-results details').open, '父片段未展开');
        return 'PASS: 输入校验、防重复、来源/耗时/分数、零分、父片段、三方案对比、HTML 转义';
    """))
    output = Path(__file__).parent / 'data/logs'
    output.mkdir(parents=True, exist_ok=True)
    for name, width, height in [('desktop', 1280, 900), ('mobile', 375, 740)]:
        call('Emulation.setDeviceMetricsOverride', width=width, height=height, deviceScaleFactor=1, mobile=False)
        time.sleep(.2)
        print(run("""
            el('retrieval-dialog').scrollTop = 0;
            const rect = el('retrieval-dialog').getBoundingClientRect();
            check(rect.left >= 0 && rect.right <= innerWidth, '弹窗超出屏幕');
            check(el('retrieval-dialog').scrollWidth <= el('retrieval-dialog').clientWidth, '内容溢出');
            return 'PASS: 检索布局 ' + innerWidth;
        """))
        shot = call('Page.captureScreenshot', format='png')
        (output / f'retrieval-{name}.png').write_bytes(base64.b64decode(shot['data']))
    print(run("""
        window.fetch = actualFetch;
        el('retrieval-query').value = '没有资料';
        el('retrieval-compare').checked = false;
        el('retrieval-form').requestSubmit();
        await wait(() => !retrievalView.busy);
        check(el('retrieval-results').textContent.includes('没有命中片段'), '空状态缺失');
        check(!document.querySelector('.retrieval-comparison'), '旧对比结果残留');
        window.fetch = async (path, options) => {
            if (String(path).endsWith('/kb/search')) throw new Error('模拟断网');
            return actualFetch(path, options);
        };
        el('retrieval-query').value = '保养';
        el('retrieval-form').requestSubmit();
        await wait(() => !el('retrieval-retry').hidden);
        check(el('retrieval-error').textContent.includes('模拟断网'), '错误信息缺失');
        el('retrieval-query').value = '没有资料';
        window.fetch = actualFetch;
        el('retrieval-retry').click();
        await wait(() => !retrievalView.busy);
        check(document.querySelectorAll('#retrieval-results > .chunk-card').length === 2, '重试未使用上次参数');
        window.releaseSearch = null;
        window.searchSignal = null;
        window.fetch = async (path, options) => {
            if (String(path).endsWith('/kb/search')) {
                window.searchSignal = options.signal;
                await new Promise(resolve => window.releaseSearch = resolve);
                // 故意忽略中断，验证晚返回的结果也不能写回。
                return actualFetch(path, {...options, signal: undefined});
            }
            return actualFetch(path, options);
        };
        el('retrieval-query').value = '旧问题';
        el('retrieval-form').requestSubmit();
        await wait(() => window.releaseSearch);
        el('retrieval-close').click();
        await wait(() => !retrievalView.busy);
        check(searchSignal.aborted, '关闭未中断浏览器请求');
        el('retrieval-open').click();
        window.fetch = actualFetch;
        el('retrieval-query').value = '没有资料';
        el('retrieval-form').requestSubmit();
        await wait(() => !retrievalView.busy);
        window.releaseSearch();
        await new Promise(resolve => setTimeout(resolve, 300));
        check(el('retrieval-results').textContent.includes('没有命中片段'), '旧请求覆盖了新结果');
        state.token = 'expired-test-token';
        await runRetrieval();
        check(!el('retrieval-dialog').open && !localStorage.getItem('rag_token'), '过期未退出');
        check(!el('retrieval-query').value && !retrievalView.last, '退出未清理测试问题');
        el('username').value = 'kb_user';
        el('password').value = 'Test123456';
        el('login-btn').click();
        await wait(() => el('app-view').style.display === 'flex');
        openRetrieval();
        check(!el('retrieval-dialog').open && el('kb-section').style.display === 'none', '普通用户打开检索');
        return 'PASS: 空结果、失败重试参数、关闭中断、旧请求隔离、失效登录清理、权限入口';
    """))


if __name__ == '__main__':
    browser_main(scenario, 'http://127.0.0.1:8767/app')
