"""文档筛选与批量上传浏览器验收；先启动 test_knowledge_ui_api.py --serve。"""

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
        check(el('doc-result-summary').textContent === '共 22 份文档', '总数缺失');
        el('doc-next').click();
        await wait(() => el('doc-page').textContent === '2 / 2');
        el('doc-keyword').value = '服饰';
        el('doc-filter').requestSubmit();
        await wait(() => el('doc-result-summary').textContent === '共 1 份文档（已筛选）');
        check(el('doc-page').textContent === '1 / 1', '搜索未回到第一页');
        check(document.querySelectorAll('.doc-open').length === 1, '搜索不准确');
        check(!el('doc-list').querySelector('img'), '文件名被解释为 HTML');
        check(el('doc-list').textContent.includes('100 B'), '文件大小缺失');
        document.querySelector('.doc-open').click();
        await wait(() => !kbView.loading);
        check(el('kb-file-info').textContent.includes('100 B') && el('kb-file-info').textContent.includes('上传于'), '详情信息缺失');
        el('kb-close').click();
        await wait(() => !kbView.doc);
        el('doc-keyword').value = '';
        el('doc-status').value = 'failed';
        el('doc-status').dispatchEvent(new Event('change'));
        await wait(() => document.querySelector('.doc-open[data-id="22"]') && document.querySelectorAll('.doc-open').length === 1);
        el('doc-keyword').value = '不存在';
        el('doc-filter').requestSubmit();
        await wait(() => el('doc-list').textContent.includes('没有符合条件'));
        el('doc-filter-clear').click();
        await wait(() => el('doc-result-summary').textContent === '共 22 份文档');
        window.actualFetch = window.fetch;
        window.releaseDocs = null;
        window.fetch = async (path, options) => {
            if (String(path).includes('keyword=slow')) await new Promise(resolve => window.releaseDocs = resolve);
            return actualFetch(path, options);
        };
        el('doc-keyword').value = 'slow';
        el('doc-filter').requestSubmit();
        await wait(() => window.releaseDocs);
        el('doc-keyword').value = '服饰';
        el('doc-filter').requestSubmit();
        await wait(() => el('doc-result-summary').textContent === '共 1 份文档（已筛选）');
        window.releaseDocs();
        await new Promise(resolve => setTimeout(resolve, 300));
        check(document.querySelector('.doc-open[data-id="21"]'), '旧搜索覆盖新结果');
        window.fetch = actualFetch;
        el('doc-filter-clear').click();
        await wait(() => el('doc-result-summary').textContent === '共 22 份文档');
        return 'PASS: 搜索、状态筛选、重置、空结果、文件信息、旧搜索隔离';
    """))
    print(run("""
        window.uploadPosts = [];
        window.fetch = async (path, options) => {
            if (String(path).endsWith('/documents/upload')) window.uploadPosts.push(options.body.get('file').name);
            return actualFetch(path, options);
        };
        const files = new DataTransfer();
        files.items.add(new File(['一'], '批量一.txt', {type: 'text/plain'}));
        files.items.add(new File(['二'], '批量二.txt', {type: 'text/plain'}));
        files.items.add(new File(['失败'], '失败重试.txt', {type: 'text/plain'}));
        files.items.add(new File(['x'], '非法.exe'));
        files.items.add(new File([], '空文件.txt'));
        el('file-input').files = files.files;
        el('file-input').dispatchEvent(new Event('change'));
        check(el('file-input').value === '', '文件选择框未清空');
        check(el('upload-area').disabled && el('upload-clear').disabled, '上传中未锁定');
        uploadFiles([new File(['x'], '重复.txt')]);
        await wait(() => !uploads.busy);
        check(uploadPosts.join('|') === '批量一.txt|批量二.txt|失败重试.txt', '批量丢文件或提交非法文件');
        check(el('upload-summary').textContent === '处理完成：2 个成功，3 个失败', '汇总不正确');
        check(el('upload-results').textContent.includes('模拟解析失败'), '失败原因未保留');
        check(el('upload-results').textContent.includes('文件是空的'), '空文件提示缺失');
        await wait(() => el('doc-result-summary').textContent === '共 24 份文档');
        window.confirm = () => true;
        const retry = document.querySelector('[data-retry]');
        retry.click();
        retry.click();
        await wait(() => !uploads.busy && uploads.items[2].status === 'success');
        check(uploadPosts.filter(name => name === '失败重试.txt').length === 2, '重试重复请求');
        check(uploadPosts.filter(name => name === '批量一.txt').length === 1, '重传了成功文件');
        await wait(() => el('doc-result-summary').textContent === '共 25 份文档');
        check(el('upload-summary').textContent === '处理完成：3 个成功，2 个失败', '重试未更新状态');
        return 'PASS: 批量文件快照、逐文件结果、格式/空文件校验、防重复、失败单项重试、刷新列表';
    """))
    output = Path(__file__).parent / 'data/logs'
    output.mkdir(parents=True, exist_ok=True)
    for name, width, height in [('desktop', 1280, 900), ('mobile', 375, 740)]:
        call('Emulation.setDeviceMetricsOverride', width=width, height=height, deviceScaleFactor=1, mobile=False)
        time.sleep(.15)
        print(run("""
            if (innerWidth < 600) el('sidebar').classList.add('open');
            await wait(() => el('sidebar').getBoundingClientRect().left >= -1);
            el('kb-body').scrollTop = 0;
            check(el('kb-body').scrollWidth <= el('kb-body').clientWidth, '知识库内容溢出');
            check(document.documentElement.scrollWidth <= innerWidth, '页面横向溢出');
            return 'PASS: 管理界面布局 ' + innerWidth;
        """))
        screenshot = call('Page.captureScreenshot', format='png')
        (output / f'knowledge-management-{name}.png').write_bytes(base64.b64decode(screenshot['data']))
        print(run("""
            el('upload-feedback').hidden = true;
            el('doc-filter').scrollIntoView({block: 'nearest'});
            return 'PASS: 筛选控件可滚动访问';
        """))
        screenshot = call('Page.captureScreenshot', format='png')
        (output / f'knowledge-filters-{name}.png').write_bytes(base64.b64decode(screenshot['data']))
        run("el('upload-feedback').hidden = false;")
    print(run("""
        el('upload-clear').click();
        check(el('upload-feedback').hidden, '结果未清除');
        window.releaseUpload = null;
        window.fetch = async (path, options) => {
            if (String(path).endsWith('/documents/upload')) await new Promise(resolve => window.releaseUpload = resolve);
            return actualFetch(path, options);
        };
        uploadFiles([new File(['x'], '中断一.txt'), new File(['y'], '中断二.txt')]);
        await wait(() => window.releaseUpload);
        el('logout-btn').click();
        check(el('upload-feedback').hidden && !uploads.busy, '退出未清理上传状态');
        window.releaseUpload();
        await new Promise(resolve => setTimeout(resolve, 500));
        check(el('upload-feedback').hidden && uploads.items.length === 0, '退出后旧上传写回界面');
        check(el('doc-keyword').value === '' && el('doc-status').value === '', '退出未清理筛选');
        window.fetch = actualFetch;
        return 'PASS: 清除结果、上传期间退出清理、停止后续文件';
    """))


if __name__ == '__main__':
    browser_main(scenario, 'http://127.0.0.1:8767/app')
