"""先启动 test_chat_reliability.py --serve；真实页面、替代生成器、内存数据。"""

from test_account_ui import main as browser_main


def scenario(call, evaluate, run, ready):
    print(run(r"""
        window.actualFetch = window.fetch;
        window.heldInitialSessions = false;
        window.releaseInitialSessions = null;
        window.fetch = async (path, options) => {
            if (String(path) === '/api/sessions' && !window.heldInitialSessions) {
                window.heldInitialSessions = true;
                await new Promise(resolve => window.releaseInitialSessions = resolve);
            }
            return actualFetch(path, options);
        };
        el('username').value = 'chat_test';
        el('password').value = 'Test123456';
        el('login-btn').click();
        await wait(() => el('app-view').style.display === 'flex' && el('welcome'));
        el('question').value = '通用问题';
        el('send-btn').click();
        await wait(() => !state.sending && document.querySelector('.msg.ai .answer-meta'));
        const generalSession = state.currentSessionId;
        check(generalSession !== null, '快速发送未保存会话');
        window.releaseInitialSessions();
        await new Promise(resolve => setTimeout(resolve, 300));
        check(state.currentSessionId === generalSession, '登录初始化覆盖了新会话');
        window.fetch = actualFetch;
        check(el('messages').textContent.includes('通用知识回答'), '实时来源标签缺失');
        newChat();
        await switchSession(generalSession);
        check(el('messages').textContent.includes('通用知识回答'), '重载后通用来源丢失');
        check(el('messages').textContent.includes('缓存回答'), '历史回答内容错误');
        document.querySelector('[data-act="regen"]').click();
        await wait(() => !state.sending && document.querySelector('.msg.ai .answer').textContent.includes('新回答'));
        check(document.querySelectorAll('.msg.ai').length === 1, '重答追加了重复回答');
        check(!document.querySelector('.meta-tag.cache'), '重答仍命中缓存');
        newChat();
        await switchSession(generalSession);
        check(el('messages').textContent.includes('新回答') && el('messages').textContent.includes('通用知识回答'), '重答未持久化');
        newChat();
        el('question').value = '什么模型';
        el('send-btn').click();
        await wait(() => !state.sending && document.querySelector('.msg.ai .answer-meta'));
        const systemSession = state.currentSessionId;
        newChat();
        await switchSession(systemSession);
        check(el('messages').textContent.includes('系统配置'), '重载后系统来源丢失');
        newChat();
        el('question').value = '资料问题';
        el('send-btn').click();
        await wait(() => !state.sending && document.querySelector('.msg.ai .answer-meta'));
        const kbSession = state.currentSessionId;
        newChat();
        await switchSession(kbSession);
        check(el('messages').textContent.includes('测试.txt'), '重载后知识库引用丢失');
        check(!el('messages').textContent.includes('通用知识回答'), '知识库回答来源错误');
        const preview = document.createElement('div');
        preview.innerHTML = md('正文[1][2]，`[1]`，```\n[1]\n```，[1](https://example.com)，[1][link]', [{index: 1}, {index: 2}]);
        check(preview.querySelectorAll('.cite-ref').length === 2, '代码或链接被当成引用 / 相邻引用丢失');
        check(!preview.querySelector('code .cite-ref'), '代码引用仍可点击');
        preview.innerHTML = md('无资料[1]');
        check(!preview.querySelector('.cite-ref'), '无资料编号被渲染为可点击引用');
        for (const question of ['引用校验', '资料引用校验']) {
            newChat();
            el('question').value = question;
            el('send-btn').click();
            await wait(() => !state.sending && document.querySelector('.msg.ai .answer-meta'));
            const citationSession = state.currentSessionId;
            check(!document.querySelector('.answer').textContent.includes('[99]'), '流式正文保留虚构引用');
            check(!document.querySelector('code .cite-ref'), '代码内容被渲染为引用');
            const expected = question === '资料引用校验' ? 1 : 0;
            check(document.querySelectorAll('.answer .cite-ref').length === expected, '正文有效引用数量错误');
            newChat();
            await switchSession(citationSession);
            check(document.querySelectorAll('.answer .cite-ref').length === expected, '历史引用显示不一致');
            check(!document.querySelector('.answer').textContent.includes('[99]'), '历史仍保留虚构引用');
        }
        return 'PASS: 来源/引用历史恢复、重新生成、代码及链接保护、流式虚构引用过滤与持久化一致';
    """))


if __name__ == '__main__':
    browser_main(scenario, 'http://127.0.0.1:8768/app')
