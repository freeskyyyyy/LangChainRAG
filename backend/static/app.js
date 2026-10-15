/* ==================================================================
   知识库问答助手 · 前端逻辑
   ==================================================================

   【这个文件负责什么】
   所有"点了之后会发生什么"的逻辑：登录、会话切换、发消息、上传文件。

   【和另外两个文件的分工】
     index.html  —— 页面骨架（有哪些元素）
     app.css     —— 外观样式（长什么样）
     app.js      —— 本文件，行为逻辑（点了会怎样）

   【一个关键设计：多会话是怎么工作的】
   后端把"会话"和"消息"分开存：
     · 会话表：一个会话就是一条记录，有标题、是否置顶
     · 消息表：每条消息属于某个会话

   前端这边只需要记住一个变量 —— currentSessionId（当前在哪个会话里）。
   发消息时把它带上，后端就知道该往哪个会话里存。
   如果它是 null，说明用户点了"新建对话"还没说话，
   后端会自动建一个新会话，并通过 SSE 把新会话的 ID 推回来。

   【为什么不在点"新建对话"时就立刻建会话】
   因为用户很可能点完"新建"又不说话，直接切回旧会话。
   那样数据库里就会堆一堆没有内容、标题都叫"新对话"的空会话，
   侧栏列表会变得又长又脏。等真正开口说话了再建，才干净。
   ================================================================== */

'use strict';

// ==================================================================
//  一、全局状态
// ==================================================================
const state = {
  token: localStorage.getItem('rag_token') || '',
  user: null,
  sessions: [],          // 会话列表
  currentSessionId: null, // 当前会话；null = 还没开始的新对话
  sending: false,        // 正在生成回答，防止重复发送
  abort: null,           // 用来中断流式请求
  welcomeShown: false,
};

// ==================================================================
//  二、工具函数
// ==================================================================

const $ = id => document.getElementById(id);

/**
 * 转义 HTML 特殊字符。
 *
 * 【为什么每条用户输入都要过这一道】
 * 如果直接把用户输入拼进 innerHTML，用户把昵称改成
 *   <img src=x onerror=alert(1)>
 * 这段代码就会在别人浏览器里执行 —— 这叫 XSS 攻击。
 * 转义之后，尖括号会变成 &lt; &gt;，浏览器把它当普通文字显示，不执行。
 *
 * 凡是"用户能控制的内容"要塞进 innerHTML 的地方，都必须先过这个函数。
 */
function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

let toastTimer = null;
function toast(msg, ms = 2400) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), ms);
}

/** 统一的请求封装：自动带令牌、自动处理 401、自动抛中文错误 */
async function api(path, options = {}) {
  const headers = options.headers || {};
  if (state.token) headers['Authorization'] = 'Bearer ' + state.token;
  if (!(options.body instanceof FormData) && options.body) {
    headers['Content-Type'] = 'application/json';
  }

  const res = await fetch(path, { ...options, headers });

  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }

  // 修改密码时输错旧密码也是 401，但不代表登录令牌失效。
  const wrongCurrentPassword = path === '/api/auth/change-password' && data?.code === 40120;
  if (res.status === 401 && !wrongCurrentPassword) {
    // 令牌过期或失效 → 退回登录页
    doLogout(true);
    throw new Error('登录已过期，请重新登录');
  }

  if (!res.ok) {
    // 后端的错误信息字段叫 message（见 app/core/exceptions.py）
    throw new Error((data && (data.message || data.detail)) || `请求失败（${res.status}）`);
  }
  return data;
}

/**
 * 极简 Markdown 渲染。
 *
 * 【为什么不用现成的库】
 * 引一个 markdown 库要多加载几百 KB，而知识库问答的回答
 * 其实只会用到粗体、列表、代码块、引用标注这几种语法。
 * 自己写四十行就够了，还省掉一次网络请求。
 *
 * 【顺序很重要】
 * 第一步必须先把 & < > 转义掉，再处理 Markdown 语法。
 * 反过来的话，先处理语法就会把用户输入的 <script> 当成正常标签留下来。
 */
function md(text, citations = []) {
  let html = esc(text);
  const protectedParts = [];
  let prefix = '\u0000MARKDOWN';
  while (html.includes(prefix)) prefix += '_';
  const protect = value => {
    const token = prefix + protectedParts.length + '\u0000';
    protectedParts.push({ token, value });
    return token;
  };

  // 代码和数字链接标签先隔离，避免被后面的引用 / 列表替换处理。
  html = html.replace(/(`{3,}|~{3,})([\s\S]*?)\1/g, (_, fence, content) => protect('<pre><code>' + content + '</code></pre>'));
  html = html.replace(/(`+)([^\n]*?)\1/g, (_, ticks, content) => protect('<code>' + content + '</code>'));
  html = html.replace(/\[[^\]\n]*\]\([^\n]*?\)|\[[^\]\n]*\]\[(?![0-9]+\])[^\]\n]*\]|\[[0-9]+\]:[^\n]*/g, match => protect(match));
  // 粗体
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // 引用标注 [1] → 可点击的小标签
  // data-cite 里存的是引用编号，点它就能跳到下面第 N 张引用卡片。
  // 用 data- 属性而不是把编号写进 onclick，是因为回答里可能有用户
  // 自己输入的 [1]，走属性 + 事件委托更干净，也不会拼出可执行的代码。
  const valid = new Set((citations || []).map(c => Number(c.index)));
  html = html.replace(/(?<!\\)\[([0-9]+)\]/g, (match, index) => valid.has(Number(index))
    ? `<span class="cite-ref" data-cite="${Number(index)}" title="点击查看第 ${Number(index)} 条出处">${match}</span>` : match);
  // 无序列表
  html = html.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');
  html = html.replace(/<\/ul>\s*<ul>/g, '');

  // 按空行切段落
  html = html.split(/\n{2,}/).map(p => {
    p = p.trim();
    if (!p) return '';
    // 已经是块级元素的，不要再包 <p>
    if (p.startsWith('<ul>') || p.startsWith('<pre') || protectedParts.some(part => part.value.startsWith('<pre>') && p.startsWith(part.token))) return p;
    return '<p>' + p.replace(/\n/g, '<br>') + '</p>';
  }).join('');

  for (const part of protectedParts) html = html.replaceAll(part.token, part.value);
  return html;
}

// 滚动到底部
function scrollBottom() {
  const m = $('messages');
  m.scrollTop = m.scrollHeight;
}

// ==================================================================
//  三、登录 / 注册 / 登出
// ==================================================================

function showView(name) {
  $('login-view').style.display    = name === 'login' ? 'flex' : 'none';
  $('register-view').style.display = name === 'register' ? 'flex' : 'none';
  $('app-view').style.display      = name === 'app' ? 'flex' : 'none';
}

async function doLogin() {
  const username = $('username').value.trim();
  const password = $('password').value;
  const errBox = $('login-err');

  const fail = msg => {
    errBox.textContent = msg;
    errBox.style.display = 'block';
  };

  if (!username || !password) return fail('请输入用户名和密码');
  errBox.style.display = 'none';

  const btn = $('login-btn');
  btn.disabled = true;
  btn.textContent = '登录中…';

  try {
    const data = await api('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    state.token = data.access_token;
    state.user = data.user;
    localStorage.setItem('rag_token', data.access_token);
    await enterApp();
  } catch (e) {
    fail(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '登 录';
  }
}

async function doRegister() {
  const username = $('reg-username').value.trim();
  const pwd = $('reg-password').value;
  const pwd2 = $('reg-password2').value;
  const errBox = $('register-err');

  const fail = msg => {
    errBox.textContent = msg;
    errBox.style.display = 'block';
  };

  if (!username || !pwd) return fail('用户名和密码都要填');
  if (pwd.length < 6) return fail('密码至少 6 位');
  if (pwd !== pwd2) return fail('两次输入的密码不一样');
  errBox.style.display = 'none';

  const btn = $('register-btn');
  btn.disabled = true;
  btn.textContent = '注册中…';

  try {
    await api('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, password: pwd }),
    });
    toast('注册成功，正在自动登录…');
    // 注册完直接帮他登录，省得再输一遍
    $('username').value = username;
    $('password').value = pwd;
    showView('login');
    await doLogin();
  } catch (e) {
    fail(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '注 册';
  }
}

function doLogout(silent = false) {
  if (state.abort) state.abort.abort();
  resetKnowledgeView();
  if ($('account-dialog').open) $('account-dialog').close();
  clearAccountForms();
  // 主动退出时告诉后端一声（后端可记录登出日志）；
  // 被动退出（令牌过期）就跳过，因为那个请求本身也会 401
  if (!silent && state.token) {
    api('/api/auth/logout', { method: 'POST' }).catch(() => {});
  }

  state.token = '';
  state.user = null;
  state.sessions = [];
  state.currentSessionId = null;
  localStorage.removeItem('rag_token');

  $('password').value = '';
  $('reg-password').value = '';
  $('reg-password2').value = '';
  showView('login');
  $('username').focus();
}

// 个人设置：复用已有认证接口，不切换或清空当前会话。
let accountBusy = false;

function clearAccountForms() {
  $('profile-form').reset();
  $('password-form').reset();
  $('profile-error').textContent = '';
  $('password-error').textContent = '';
  $('profile-status').textContent = '';
}

function openAccountDialog() {
  if (!state.user || accountBusy) return;
  clearAccountForms();
  $('account-summary').textContent = `账号：${state.user.username} · ${state.user.role === 'admin' ? '管理员' : '普通用户'}`;
  $('profile-nickname').value = state.user.nickname || state.user.username;
  $('account-dialog').showModal();
  $('profile-nickname').focus();
}

function setAccountBusy(busy, button, label) {
  accountBusy = busy;
  $('account-dialog').setAttribute('aria-busy', String(busy));
  $('account-dialog').querySelectorAll('input, button').forEach(el => { el.disabled = busy; });
  button.textContent = label;
}

async function saveProfile(e) {
  e.preventDefault();
  if (accountBusy || !state.user) return;
  const nickname = $('profile-nickname').value.trim();
  const error = $('profile-error');
  error.textContent = '';
  $('profile-status').textContent = '';
  if (!nickname || [...nickname].length > 50) {
    error.textContent = '昵称请输入 1～50 个字符，不能全是空白';
    return;
  }
  const token = state.token;
  const button = $('profile-save');
  setAccountBusy(true, button, '保存中…');
  try {
    const user = await api('/api/auth/me', {
      method: 'PATCH', body: JSON.stringify({ nickname }),
    });
    if (state.token !== token) return;
    state.user = user;
    $('whoami').textContent = user.nickname || user.username;
    $('profile-nickname').value = user.nickname || user.username;
    $('profile-status').textContent = '昵称已保存';
  } catch (err) {
    if (state.token === token) error.textContent = err.message;
  } finally {
    setAccountBusy(false, button, '保存昵称');
  }
}

async function savePassword(e) {
  e.preventDefault();
  if (accountBusy || !state.user) return;
  const error = $('password-error');
  error.textContent = '';
  const oldPassword = $('current-password').value;
  const newPassword = $('new-password').value;
  if (state.sending) {
    error.textContent = '请等待回答完成，或停止生成后再修改密码';
    return;
  }
  if (!oldPassword) {
    error.textContent = '请输入当前密码';
    return;
  }
  if ([...newPassword].length < 6 || [...newPassword].length > 64) {
    error.textContent = '新密码长度须为 6～64 个字符';
    return;
  }
  if (new TextEncoder().encode(newPassword).length > 72 || newPassword.trim() !== newPassword) {
    error.textContent = '新密码不能超过 72 字节，首尾不能有空白';
    return;
  }
  if (newPassword === oldPassword) {
    error.textContent = '新密码不能与当前密码相同';
    return;
  }
  if (newPassword !== $('confirm-password').value) {
    error.textContent = '两次输入的新密码不一致';
    return;
  }
  const token = state.token;
  const username = state.user.username;
  const button = $('password-save');
  setAccountBusy(true, button, '修改中…');
  try {
    await api('/api/auth/change-password', {
      method: 'POST', body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
    });
    if (state.token !== token) return;
    doLogout(true);
    $('username').value = username;
    $('password').focus();
    toast('密码修改成功，请使用新密码登录', 4000);
  } catch (err) {
    if (state.token === token) error.textContent = err.message;
  } finally {
    // 成功、失败都不保留密码；昵称和会话不受失败请求影响。
    $('password-form').reset();
    setAccountBusy(false, button, '修改密码');
  }
}

/** 进入主界面 */
async function enterApp() {
  showView('app');

  const isAdmin = state.user.role === 'admin';
  $('whoami').textContent = state.user.nickname || state.user.username;

  const badge = $('role-badge');
  badge.textContent = isAdmin ? '管理员' : '普通用户';
  badge.className = 'badge' + (isAdmin ? ' admin' : '');

  // 知识库管理只有管理员能看。普通用户即使看到也没用 ——
  // 后端的 /api/kb/* 接口会直接返回 403。
  $('kb-section').style.display = isAdmin ? 'flex' : 'none';

  newChat(false);          // 默认停在一个空白的新对话上
  await loadSessions();

  if (isAdmin) {
    loadStats();
    loadDocs();
  }
}

// ==================================================================
//  四、会话管理
// ==================================================================

async function loadSessions() {
  try {
    state.sessions = await api('/api/sessions');
    renderSessionList();
  } catch (e) {
    $('session-list').innerHTML =
      `<div class="s-empty">会话列表加载失败<br>${esc(e.message)}</div>`;
  }
}

function renderSessionList() {
  const box = $('session-list');

  if (!state.sessions.length) {
    box.innerHTML = '<div class="s-empty">还没有对话<br>点右上角 ＋ 开始</div>';
    return;
  }

  box.innerHTML = state.sessions.map(s => {
    const active = s.id === state.currentSessionId ? ' active' : '';
    const pin = s.is_pinned ? '<span class="s-pin">📌</span>' : '';
    const count = s.message_count ? `（${s.message_count} 条）` : '';

    return `<div class="session-item${active}" data-id="${s.id}">
      <span class="s-icon">💬</span>
      <span class="s-title" title="${esc(s.title)}${count}">${esc(s.title)}</span>
      ${pin}
      <span class="s-actions">
        <button class="s-act" data-act="rename" title="重命名">✏️</button>
        <button class="s-act" data-act="pin"    title="${s.is_pinned ? '取消置顶' : '置顶'}">📌</button>
        <button class="s-act danger" data-act="del" title="删除">🗑</button>
      </span>
    </div>`;
  }).join('');
}

/** 新建对话：只清空当前状态，不立刻在后端建会话（原因见文件头注释） */
function newChat(focus = true) {
  state.currentSessionId = null;
  renderSessionList();
  renderWelcome();
  if (focus) $('question').focus();
}

async function switchSession(id) {
  if (state.sending) {
    toast('正在生成回答，请等它说完再切换');
    return;
  }
  if (id === state.currentSessionId) return;

  state.currentSessionId = id;
  renderSessionList();

  const box = $('messages');
  box.innerHTML = '<div class="s-empty" style="padding-top:60px">正在加载历史记录…</div>';

  try {
    const msgs = await api(`/api/sessions/${id}/messages`);
    box.innerHTML = '';

    if (!msgs.length) {
      renderWelcome();
      return;
    }

    // 边渲染边记住"上一条用户消息说了什么" ——
    // 每条 AI 回答都要靠它才能「重新生成」（见 renderHistoryMessage）
    let lastUserText = '';
    msgs.forEach(m => {
      if (m.role === 'user') lastUserText = m.content;
      renderHistoryMessage(m, lastUserText);
    });
    scrollBottom();
  } catch (e) {
    box.innerHTML = `<div class="s-empty" style="padding-top:60px">加载失败：${esc(e.message)}</div>`;
  }
}

function openRenameModal(session) {
  const mask = $('rename-mask');
  const input = $('rename-input');

  input.value = session.title;
  mask.classList.add('show');
  input.focus();
  input.select();

  // 先解绑上一次的处理器，避免重复绑定导致"点一次触发多次"
  const okBtn = $('rename-ok');
  const newOk = okBtn.cloneNode(true);
  okBtn.parentNode.replaceChild(newOk, okBtn);

  const close = () => mask.classList.remove('show');

  const submit = async () => {
    const title = input.value.trim();
    if (!title) return toast('名字不能为空');
    close();
    try {
      await api(`/api/sessions/${session.id}`, {
        method: 'PATCH',
        body: JSON.stringify({ title }),
      });
      await loadSessions();
      toast('已重命名');
    } catch (e) {
      toast('重命名失败：' + e.message);
    }
  };

  newOk.onclick = submit;
  input.onkeydown = e => {
    if (e.key === 'Enter') submit();
    if (e.key === 'Escape') close();
  };
}

async function togglePin(session) {
  try {
    await api(`/api/sessions/${session.id}`, {
      method: 'PATCH',
      body: JSON.stringify({ is_pinned: !session.is_pinned }),
    });
    await loadSessions();
    toast(session.is_pinned ? '已取消置顶' : '已置顶');
  } catch (e) {
    toast('操作失败：' + e.message);
  }
}

async function deleteSession(session) {
  if (!confirm(`确定删除「${session.title}」吗？\n这段对话的全部消息会一并删除，无法恢复。`)) {
    return;
  }
  try {
    await api(`/api/sessions/${session.id}`, { method: 'DELETE' });
    // 删掉的正好是当前会话 → 退回空白新对话
    if (session.id === state.currentSessionId) {
      state.currentSessionId = null;
      renderWelcome();
    }
    await loadSessions();
    toast('已删除');
  } catch (e) {
    toast('删除失败：' + e.message);
  }
}

// ==================================================================
//  五、消息渲染
// ==================================================================

function renderWelcome() {
  $('messages').innerHTML = `
    <div class="welcome" id="welcome">
      <h2>🌿 你好，我是知识库问答助手</h2>
      <p>
        我会优先参考你上传的资料回答，并标注资料来源。<br>
        资料不足时，也可以基于模型的通用知识回答，并说明回答依据。
      </p>
      <div class="sugg-grid">
        <button class="sugg">总结已上传资料的主要内容</button>
        <button class="sugg">资料中有哪些注意事项</button>
        <button class="sugg">根据资料整理一份操作清单</button>
        <button class="sugg">什么是检索增强生成（RAG）</button>
      </div>
    </div>`;

  document.querySelectorAll('.sugg').forEach(el => {
    el.onclick = () => {
      $('question').value = el.textContent;
      send();
    };
  });
}

/**
 * 造一条 AI 消息的骨架（还没填内容）。
 *
 * 【为什么要把骨架抽成函数】
 * 因为"实时生成的那条"和"从历史里读回来的那条"必须是**一模一样**的结构，
 * 否则用户会发现：刚答完的消息有操作按钮，刷新一下按钮就没了。
 * 第一版就是各写各的，结果历史消息没有操作按钮。
 * 抽成同一个函数之后，两边想不一样都难。
 */
function createAiBubble() {
  const wrap = document.createElement('div');
  wrap.className = 'msg ai';
  wrap.innerHTML = `<div class="bubble">
      <div class="status-line" style="display:none">
        <span class="dots"><i></i><i></i><i></i></span>
        <span class="status-text">正在检索资料…</span>
      </div>
      <div class="answer" style="display:none"></div>
      <div class="citations" style="display:none"></div>
      <div class="answer-meta" style="display:none"></div>
    </div>`;
  return wrap;
}

/**
 * 填充回答底部那条信息栏：左边是"耗时/缓存"标签，右边是一排操作按钮。
 *
 * 【为什么信息栏放在 .answer 外面】
 * 因为回答是逐字刷新的 —— 每来一个字就把 .answer 的内容整个重写一遍。
 * 如果按钮放在 .answer 里面，每刷新一次按钮就会被冲掉重建，
 * 用户正要点"复制"的时候按钮突然消失、位置还变了，根本点不中。
 */
function renderMetaBar(wrap, { tags = [], messageId = null, feedback = null, question = '' } = {}) {
  const bar = wrap.querySelector('.answer-meta');
  bar.style.display = 'flex';

  const actions = `
    <div class="meta-actions">
      <button class="act-btn" data-act="copy" title="把回答复制到剪贴板">📋 复制</button>
      <button class="act-btn" data-act="regen" title="用同一个问题重新回答一遍">🔄 重新生成</button>
      ${messageId ? `
        <button class="act-btn${feedback === 1 ? ' on' : ''}"  data-act="up"   title="有帮助">👍</button>
        <button class="act-btn${feedback === -1 ? ' on' : ''}" data-act="down" title="没帮助">👎</button>
      ` : ''}
    </div>`;

  bar.innerHTML = tags.join('') + actions;

  // 「重新生成」需要知道原来是哪个问题。直接挂在 DOM 上，
  // 点的时候就近取 —— 不用再去翻会话历史、算它是第几条消息。
  wrap.dataset.question = question;
  wrap.dataset.messageId = messageId || '';
}

/**
 * 渲染一条历史消息（从数据库取回来的）
 *
 * prevQuestion 是这条 AI 回答所回答的那个问题，
 * 点「重新生成」时要用它（见上面 renderMetaBar 的说明）。
 */
function renderHistoryMessage(m, prevQuestion = '') {
  const box = $('messages');

  if (m.role === 'user') {
    box.insertAdjacentHTML('beforeend',
      `<div class="msg user"><div class="bubble">${esc(m.content)}</div></div>`);
    return;
  }

  const wrap = createAiBubble();
  box.appendChild(wrap);

  const answerEl = wrap.querySelector('.answer');
  answerEl.style.display = 'block';
  answerEl.innerHTML = md(m.content, m.citations);

  if (m.citations && m.citations.length) {
    renderCitations(wrap.querySelector('.citations'), m.citations);
  }

  // 历史消息底部同样显示当时的耗时和缓存情况，外加操作按钮
  const tags = [];
  if (m.knowledge_source === 'general') tags.push('<span class="meta-tag">💡 通用知识回答</span>');
  if (m.knowledge_source === 'system') tags.push('<span class="meta-tag">⚙️ 系统配置</span>');
  if (m.cache_hit) tags.push('<span class="meta-tag cache">⚡ 缓存</span>');
  if (m.latency_ms) tags.push(`<span class="meta-tag">耗时 ${(m.latency_ms / 1000).toFixed(2)} 秒</span>`);

  renderMetaBar(wrap, {
    tags,
    messageId: m.id,          // 历史消息的 ID 后端直接返回了，可以评价
    feedback: m.feedback,     // 之前评过的话，按钮要显示成"已选中"
    question: prevQuestion,
  });
}

function renderCitations(el, items) {
  el.style.display = 'block';
  el.innerHTML =
    `<div class="citations-title">📎 参考了 ${items.length} 个知识库片段（点击展开原文）</div>` +
    items.map(c => {
      const loc = c.page ? ` 第 ${c.page} 页`
                : (c.sheet_name ? ` 工作表「${c.sheet_name}」` : '');
      const pct = Math.round((c.score || 0) * 100);
      // data-idx 是给"点正文里的 [1] 跳过来"用的 —— 见 jumpToCitation()
      return `<div class="cite" data-idx="${esc(c.index)}">
        <div class="cite-head">
          <span class="cite-idx">${esc(c.index)}</span>
          <span class="cite-name">${esc(c.file_name)}${esc(loc)}</span>
          <span class="cite-score">相关度 ${pct}%</span>
          <span class="cite-toggle">▾</span>
        </div>
        <div class="cite-body">${esc(c.content)}</div>
      </div>`;
    }).join('');

  // 点标题栏展开/收起
  el.querySelectorAll('.cite-head').forEach(head => {
    head.onclick = () => head.parentElement.classList.toggle('open');
  });
}

// ==================================================================
//  六、提问（流式）
// ==================================================================

async function send() {
  const input = $('question');
  const question = input.value.trim();
  if (!question || state.sending) return;

  input.value = '';
  input.style.height = 'auto';

  const msgs = $('messages');
  const welcome = $('welcome');
  if (welcome) welcome.remove();

  // ---- 用户消息 ----
  msgs.insertAdjacentHTML('beforeend',
    `<div class="msg user"><div class="bubble">${esc(question)}</div></div>`);

  // ---- AI 消息容器 ----
  const aiWrap = createAiBubble();
  msgs.appendChild(aiWrap);
  scrollBottom();

  await streamAnswer(aiWrap, question, { regenerate: false });
}

/**
 * 真正干活的函数：把问题发给后端，把流式回答画进 wrap 这个容器里。
 *
 * 【为什么要和 send() 拆开】
 * 因为「重新生成」要做的是同一件事，只差一个参数。
 * 拆开之后两条路共用同一套流式解析逻辑 ——
 * 否则"重新生成"那条得再抄一遍 SSE 解析，抄错了极难发现
 *（第一版就是这么干的，抄漏了一行，重新生成时引用卡片死活不显示）。
 */
async function streamAnswer(wrap, question, { regenerate = false } = {}) {
  state.sending = true;
  setComposerBusy(true);

  const statusEl   = wrap.querySelector('.status-line');
  const statusText = wrap.querySelector('.status-text');
  const answerEl   = wrap.querySelector('.answer');
  const citeEl     = wrap.querySelector('.citations');

  statusEl.style.display = 'flex';
  statusText.textContent = regenerate ? '正在重新回答…' : '正在检索资料…';
  answerEl.style.display = 'none';
  answerEl.innerHTML = '';

  let answerText = '';
  let citations = [];
  let donePayload = null;
  let messageId = null;   // 后端存库后补发回来的消息 ID，点赞点踩要用
  let stopped = false;    // 用户按了「停止」
  let failed  = false;    // 出错了（错误信息要留在屏幕上，不能收尾时被抹掉）

  const setStatus = txt => { statusText.textContent = txt; };

  // 【「停止生成」是怎么实现的】
  // fetch 支持一个 signal 参数 —— 可以把它理解成一张"取消令牌"。
  // 执行 controller.abort() 之后，正在等待的 reader.read() 会立刻抛出
  // 一个叫 AbortError 的异常，循环随即结束，网络连接断开。
  //
  // 【关键：这是真的停，不是只把界面停住】
  // 连接一断，后端那个生成器也会被取消，后面的模型调用就不会发生了。
  // 所以用户按停止是**真的省下了钱**，而不是"眼不见为净"。
  state.abort = new AbortController();

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      signal: state.abort.signal,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + state.token,
      },
      body: JSON.stringify({
        question,
        session_id: state.currentSessionId,
        regenerate,
      }),
    });

    if (!res.ok) {
      const t = await res.text();
      throw new Error(`服务返回 ${res.status}：${t.slice(0, 150)}`);
    }

    // ---- 手动解析 SSE 流 ----
    // 【为什么不能用浏览器的 EventSource】
    // 因为 EventSource 只支持 GET 请求，而提问必须用 POST
    //（问题可能很长，塞进网址里会超长）。
    // 所以只能自己 fetch 拿数据流，再按 SSE 的格式解析。
    //
    // SSE 的格式很简单：一个事件由若干行组成，行之间用 \n 分隔，
    // 事件之间用空行分隔：
    //     event: token
    //     data: {"text":"你"}
    //     （空行）
    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const blocks = buffer.split('\n\n');
      // 最后一段可能只收到半截，留在缓冲区等下一轮，不能现在就解析
      buffer = blocks.pop();

      for (const block of blocks) {
        if (!block.trim()) continue;

        let eventType = 'message';
        let dataStr = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) eventType = line.slice(6).trim();
          else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
        }
        if (!dataStr) continue;

        let payload;
        try { payload = JSON.parse(dataStr); } catch { continue; }

        // ---------------- 分派各类事件 ----------------
        switch (eventType) {
          case 'session':
            // 后端建好了新会话（或给旧会话补了标题）
            state.currentSessionId = payload.session_id;
            break;

          case 'status':
            setStatus(payload.message || '处理中…');
            break;

          case 'rewrite':
            setStatus(`理解你的问题为：${payload.rewritten}`);
            break;

          case 'retrieval':
            setStatus(`找到 ${payload.count} 条相关资料（最高相关度 ${(payload.top_score || 0).toFixed(2)}）`);
            break;

          case 'token':
            // 第一个字到达时，把"等待"换成正文区
            if (!answerText) {
              statusEl.style.display = 'none';
              answerEl.style.display = 'block';
            }
            answerText += payload.text;
            answerEl.innerHTML = md(answerText, citations) + '<span class="cursor"></span>';
            scrollBottom();
            break;

          case 'citations':
            citations = payload || [];
            if (citations.length) renderCitations(citeEl, citations);
            break;

          // 生成结束。这里**只记数据，不动界面** ——
          // 界面统一留到下面的 finally 里收尾。
          //
          // 【为什么非要这么绕】
          // 因为"正常答完""出错了""被用户按了停止"这三种收场，
          // 都需要做同一件事：摆正状态行、补上操作按钮、刷新会话列表。
          // 如果分别在三个地方各写一遍，迟早会漏掉一个
          //（第一版就漏了：停止之后那条回答没有操作按钮，用户想复制都点不到）。
          case 'done':
            donePayload = payload;
            break;

          // 后端把消息存进数据库之后，会把消息 ID 补发回来。
          // 有了它才能给这条回答点赞点踩 —— 评价是挂在消息 ID 上的。
          case 'saved':
            messageId = payload.assistant_message_id;
            if (payload.session_id) state.currentSessionId = payload.session_id;
            break;

          case 'error':
            failed = true;
            statusEl.style.display = 'flex';
            statusEl.innerHTML = `<span style="color:var(--danger)">❌ ${esc(payload.message || '出错了')}</span>`;
            break;
        }
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      // 用户自己按的「停止」，不是故障，不用弹红字吓人
      stopped = true;
    } else {
      failed = true;
      statusEl.style.display = 'flex';
      statusEl.innerHTML = `<span style="color:var(--danger)">❌ ${esc(e.message)}</span>`;
    }
  } finally {
    state.sending = false;
    state.abort = null;
    setComposerBusy(false);

    // ---------------- 收尾（三种收场共用这一段） ----------------
    if (!failed) statusEl.style.display = 'none';   // 出错时红字要留着
    answerEl.style.display = 'block';

    if (!answerText) {
      answerEl.innerHTML = `<p style="color:var(--text-3)">${
        stopped ? '（已停止，这次的回答没有生成出来）' : '（没有生成任何内容）'
      }</p>`;
    } else {
      answerEl.innerHTML = md(answerText, citations);
    }

    const tags = [];
    if (stopped) {
      // 半截回答也要保留 —— 用户按停止通常正是因为"该说的已经说完了"，
      // 把已经生成的部分丢掉反而更让人恼火
      tags.push('<span class="meta-tag">⏹ 已停止</span>');
    }
    if (donePayload) {
      if (donePayload.knowledge_source === 'general') {
        tags.push('<span class="meta-tag">💡 通用知识回答</span>');
      }
      if (donePayload.knowledge_source === 'system') {
        tags.push('<span class="meta-tag">⚙️ 系统配置</span>');
      }
      if (donePayload.cache_hit) {
        const sim = (donePayload.cache_similarity || 1).toFixed(2);
        tags.push(`<span class="meta-tag cache">⚡ 命中缓存（相似度 ${sim}），未消耗额度</span>`);
      }
      if (donePayload.latency_ms) {
        tags.push(`<span class="meta-tag">耗时 ${(donePayload.latency_ms / 1000).toFixed(2)} 秒</span>`);
      }
    }

    renderMetaBar(wrap, { tags, messageId, question });
    scrollBottom();

    // 【为什么每次问答完都要刷新会话列表】
    // 因为第一次提问时后端会新建一个会话，并且用你的问题自动起标题。
    // 不刷新的话，侧栏里就看不到这个新会话。
    // 同时，已有会话的"最后更新时间"变了，排序也应该跟着变。
    if (state.currentSessionId) loadSessions();
  }
}

/** 生成过程中：藏起发送按钮，亮出停止按钮（两个按钮占同一个位置） */
function setComposerBusy(busy) {
  $('send-btn').style.display = busy ? 'none' : 'flex';
  $('stop-btn').style.display = busy ? 'flex' : 'none';
}

// ==================================================================
//  六之二、回答上的四个操作：复制 / 重新生成 / 赞 / 踩
// ==================================================================

/**
 * 重新生成：用同一个问题，把这条回答重答一遍。
 *
 * 【为什么是"替换"而不是"再追加一条"】
 * 如果只是往后接一条，用户点两次重新生成，历史里就会变成：
 *     问 → 答① → 问 → 答② → 问 → 答③
 * 但他心里想的是"刚才那个不算，重来"。
 * 所以这里先把旧回答从界面上拿掉，后端也会把库里那条删掉（见 api/chat.py）。
 * ChatGPT、通义千问都是这个行为，用户的预期就是这样的。
 */
async function regenerate(wrap) {
  if (state.sending) return toast('正在生成回答，请先等它说完');

  const question = wrap.dataset.question || '';
  if (!question) return toast('找不到对应的原问题，没法重新生成');
  if (!state.currentSessionId) return toast('这段对话还没存好，稍等一下再试');

  wrap.remove();                      // 旧回答整个拿掉
  const fresh = createAiBubble();
  $('messages').appendChild(fresh);
  scrollBottom();

  await streamAnswer(fresh, question, { regenerate: true });
}

/** 复制回答（只复制正文，不含"耗时 2.4 秒"这类信息条） */
async function copyAnswer(wrap, btn) {
  const text = (wrap.querySelector('.answer')?.innerText || '').trim();
  if (!text) return toast('这条回答是空的，没什么可复制');

  let ok = false;
  try {
    // 正规接口。要求"安全上下文"（https 或本机地址），
    // 我们跑在 127.0.0.1 上，是满足的。
    await navigator.clipboard.writeText(text);
    ok = true;
  } catch {
    // 万一环境不满足（比如以后部署到 http 的服务器上），退回老办法
    ok = legacyCopy(text);
  }

  if (!ok) return toast('复制失败，请手动选中文字复制');

  const old = btn.innerHTML;
  btn.innerHTML = '✅ 已复制';
  btn.classList.add('on');
  setTimeout(() => {
    btn.innerHTML = old;
    btn.classList.remove('on');
  }, 1500);
}

/** 老式复制：造一个看不见的输入框，选中它，执行复制命令 */
function legacyCopy(text) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.top = '-1000px';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch { ok = false; }
  document.body.removeChild(ta);
  return ok;
}

/**
 * 点赞 / 点踩。
 *
 * 【为什么不能"再点一次取消"】
 * 后端的 rating 只接受 1 和 -1，不接受 0 —— 也就是**没法表达"取消评价"**。
 * 所以这里如实处理：再点同一个就当重复了，提示一句；点另一个则改判。
 * （要做成可取消，得改后端加一个 rating=0 的语义，暂时不值当。）
 */
async function submitFeedback(wrap, rating, btn) {
  const messageId = Number(wrap.dataset.messageId || 0);
  if (!messageId) return toast('这条回答还没存好，稍等一下再评价');

  if (btn.classList.contains('on')) {
    return toast(rating === 1 ? '已经点过赞了' : '已经点过踩了');
  }

  try {
    await api('/api/chat/feedback', {
      method: 'POST',
      body: JSON.stringify({ message_id: messageId, rating }),
    });
    wrap.querySelector('[data-act="up"]').classList.toggle('on', rating === 1);
    wrap.querySelector('[data-act="down"]').classList.toggle('on', rating === -1);
    toast(rating === 1 ? '谢谢反馈 👍' : '收到，我会努力答得更好');
  } catch (e) {
    toast('评价失败：' + e.message);
  }
}

/**
 * 点正文里的 [1] → 跳到下面第 1 张引用卡片。
 *
 * 【为什么要顺手把它展开】
 * 引用卡片默认是折叠的（只显示两行摘要）。跳过去却不展开的话，
 * 用户还得再点一下才知道里面写了什么 —— 那这一跳就只完成了一半。
 */
function jumpToCitation(refEl) {
  const wrap = refEl.closest('.msg.ai');
  if (!wrap) return;

  const idx = refEl.dataset.cite;
  const card = wrap.querySelector(`.cite[data-idx="${idx}"]`);
  if (!card) {
    // 模型有时候会引用一个不存在的编号（比如只给了 3 个片段却写 [5]）
    return toast(`没有第 ${idx} 条引用，可能回答里的编号标错了`);
  }

  card.classList.add('open');
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });

  // 闪一下告诉用户"就是这一张"。1.4 秒后摘掉，恢复正常外观。
  card.classList.remove('flash');
  void card.offsetWidth;          // 强制浏览器重排，否则连点两次不会重新播放
  card.classList.add('flash');
  setTimeout(() => card.classList.remove('flash'), 1400);
}

// ==================================================================
//  七、知识库管理（仅管理员）
// ==================================================================

async function loadStats() {
  if (!state.user || state.user.role !== 'admin') return;
  const token = state.token;
  const request = ++statsRequest;
  try {
    const [s, health] = await Promise.all([
      api('/api/kb/stats'),
      api('/api/health'),
    ]);
    if (token !== state.token || request !== statsRequest) return;
    $('kb-stat').innerHTML =
      `<b>文档</b> ${s.documents.total} 份 &nbsp;·&nbsp; ` +
      `<b>片段</b> ${s.content.vector_chunks} 个<br>` +
      `<b>总字数</b> ${s.content.total_chars.toLocaleString()}<br>` +
      `<b>模型</b> ${esc(health.llm_model)}<br>` +
      `<b>缓存命中率</b> ${esc(s.semantic_cache.hit_rate || '0%')}`;
  } catch (e) {
    if (token !== state.token || request !== statsRequest) return;
    $('kb-stat').textContent = '统计加载失败：' + e.message;
  }
}

const kbView = { doc: null, page: 1, pages: 0, loading: false, rebuilding: false, request: 0 };
let docsPage = 1;
let docsPages = 0;
let docsRequest = 0;
let statsRequest = 0;
let docFilters = { keyword: '', status: '' };
const uploads = { items: [], busy: false, generation: 0 };
const retrievalView = { busy: false, request: 0, abort: null, last: null };
const documentStatus = { indexed: '已索引', parsing: '处理中', pending: '待处理', failed: '处理失败' };

function fileSize(bytes) {
  if (!Number.isFinite(bytes)) return '大小未知';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileDate(value) {
  if (!value) return '时间未知';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '时间未知' : date.toLocaleString('zh-CN');
}

function applyDocFilters() {
  docFilters = { keyword: $('doc-keyword').value.trim(), status: $('doc-status').value };
  return loadDocs(1);
}

function resetKnowledgeView() {
  resetRetrieval();
  $('retrieval-form').reset();
  if ($('retrieval-dialog').open) $('retrieval-dialog').close();
  kbView.request++;
  kbView.doc = null;
  kbView.loading = false;
  kbView.rebuilding = false;
  docsRequest++;
  statsRequest++;
  docsPage = 1;
  docsPages = 0;
  docFilters = { keyword: '', status: '' };
  $('doc-filter').reset();
  $('doc-result-summary').textContent = '';
  $('doc-list').setAttribute('aria-busy', 'false');
  uploads.generation++;
  uploads.items = [];
  uploads.busy = false;
  renderUploads();
  if ($('kb-dialog').open) $('kb-dialog').close();
  $('chunk-list').replaceChildren();
  $('doc-list').replaceChildren();
  $('doc-pagination').hidden = true;
}

async function loadDocs(page = docsPage) {
  if (!state.user || state.user.role !== 'admin') return;
  const request = ++docsRequest;
  const token = state.token;
  $('doc-list').setAttribute('aria-busy', 'true');
  $('doc-result-summary').textContent = '正在加载文档…';
  $('doc-prev').disabled = true;
  $('doc-next').disabled = true;
  try {
    const params = new URLSearchParams({ page, page_size: 20 });
    if (docFilters.keyword) params.set('keyword', docFilters.keyword);
    if (docFilters.status) params.set('status', docFilters.status);
    const data = await api('/api/kb/documents?' + params);
    if (request !== docsRequest || token !== state.token) return;
    // 删除最后一页的最后一项后，回到有效页。
    if (page > Math.max(1, data.total_pages)) return loadDocs(Math.max(1, data.total_pages));
    docsPage = data.page;
    docsPages = data.total_pages;
    $('doc-result-summary').textContent = `共 ${data.total} 份文档${docFilters.keyword || docFilters.status ? '（已筛选）' : ''}`;
    $('doc-pagination').hidden = docsPages <= 1;
    $('doc-page').textContent = `${docsPage} / ${Math.max(1, docsPages)}`;
    $('doc-prev').disabled = docsPage <= 1;
    $('doc-next').disabled = docsPage >= docsPages;
    const list = $('doc-list');

    if (!data.items.length) {
      list.innerHTML = `<div class="s-empty" style="padding:14px">${docFilters.keyword || docFilters.status ? '没有符合条件的文档，请调整搜索或状态筛选' : '还没有上传任何资料'}</div>`;
      return;
    }

    list.innerHTML = data.items.map(d => {
      const icon = { indexed: '📄', parsing: '⏳', pending: '⏳', failed: '⚠️' }[d.status] || '📄';
      const color = d.status === 'failed' ? 'color:var(--danger)' : '';
      return `<div class="doc-item" style="${color}">
        <button class="doc-open" type="button" data-id="${d.id}" title="${esc(d.file_name)} · ${esc(documentStatus[d.status] || d.status)} · 点击预览片段">
          <span aria-hidden="true">${icon}</span>
          <span class="name">${esc(d.file_name)}</span>
          <span class="doc-count">${d.chunk_count}段</span>
          <span class="doc-meta">${esc(documentStatus[d.status] || d.status)} · ${esc(fileSize(d.file_size))} · ${esc((d.file_type || '').toUpperCase())}</span>
        </button>
        <button class="del" type="button" data-id="${d.id}" data-name="${esc(d.file_name)}" aria-label="删除 ${esc(d.file_name)}">×</button>
      </div>`;
    }).join('');

    list.querySelectorAll('.doc-open').forEach(el => {
      el.onclick = () => openDocument(data.items.find(d => d.id === Number(el.dataset.id)));
    });
    list.querySelectorAll('.del').forEach(el => {
      el.onclick = () => removeDoc(Number(el.dataset.id), el.dataset.name);
    });
  } catch (e) {
    if (request !== docsRequest || token !== state.token) return;
    $('doc-result-summary').textContent = '文档加载失败';
    $('doc-pagination').hidden = true;
    $('doc-list').innerHTML =
      `<div class="s-empty" style="color:var(--danger)">加载失败：${esc(e.message)}</div><button class="btn-ghost" id="docs-retry" type="button">重试</button>`;
    $('docs-retry').onclick = () => loadDocs(page);
  } finally {
    if (request === docsRequest && token === state.token) $('doc-list').setAttribute('aria-busy', 'false');
  }
}

function updateChunkControls() {
  const busy = kbView.loading || kbView.rebuilding;
  $('chunk-prev').disabled = busy || kbView.page <= 1;
  $('chunk-next').disabled = busy || kbView.page >= kbView.pages;
  $('kb-reindex').disabled = busy || !kbView.doc || ['parsing', 'pending'].includes(kbView.doc.status);
  $('kb-reindex-ok').disabled = busy;
  $('kb-reindex-cancel').disabled = kbView.rebuilding;
  $('kb-close').disabled = kbView.rebuilding;
  $('kb-retry').disabled = busy;
  $('chunk-list').setAttribute('aria-busy', String(busy));
}

function renderDocumentSummary() {
  const d = kbView.doc;
  $('kb-filename').textContent = d.file_name;
  $('kb-doc-summary').textContent = `${documentStatus[d.status] || d.status} · ${d.chunk_count} 个子片段 · ${d.parent_chunk_count} 个父片段 · ${d.char_count} 字`;
  $('kb-doc-error').textContent = d.error_msg || '';
  $('kb-file-info').textContent = `${(d.file_type || '').toUpperCase()} · ${fileSize(d.file_size)} · 上传于 ${fileDate(d.created_at)}` +
    (d.description ? `\n备注：${d.description}` : '');
}

function openDocument(doc) {
  if (!doc || state.user?.role !== 'admin' || kbView.rebuilding) return;
  kbView.doc = doc;
  kbView.page = 1;
  kbView.pages = 0;
  $('kb-reindex-confirm').hidden = true;
  $('kb-notice').textContent = '';
  renderDocumentSummary();
  if (!$('kb-dialog').open) $('kb-dialog').showModal();
  loadChunks(1);
}

async function loadChunks(page = kbView.page) {
  if (!kbView.doc) return;
  const id = kbView.doc.id;
  const token = state.token;
  const request = ++kbView.request;
  const current = () => request === kbView.request && token === state.token && kbView.doc?.id === id;
  kbView.page = page;
  kbView.loading = true;
  $('kb-error').textContent = '';
  $('kb-retry').hidden = true;
  $('chunk-list').textContent = '正在加载片段…';
  $('chunk-page').textContent = '';
  updateChunkControls();
  try {
    const data = await api(`/api/kb/documents/${id}/chunks?page=${page}&page_size=20`);
    if (!current()) return;
    if (page > Math.max(1, data.total_pages)) return loadChunks(Math.max(1, data.total_pages));
    kbView.doc = { ...kbView.doc, ...data.document };
    kbView.page = data.page;
    kbView.pages = data.total_pages;
    renderDocumentSummary();
    $('chunk-page').textContent = `第 ${data.page} / ${Math.max(1, data.total_pages)} 页 · 共 ${data.total} 段`;
    $('chunk-list').innerHTML = data.items.length ? data.items.map(c => {
      const location = [c.page != null ? `页码 ${c.page}` : '', c.sheet_name ? `工作表 ${c.sheet_name}` : ''].filter(Boolean).join(' · ');
      return `<article class="chunk-card">
        <div class="chunk-meta">片段 ${esc(c.chunk_index + 1)} · ${esc(c.char_count)} 字${location ? ' · ' + esc(location) : ''}</div>
        <div class="chunk-content">${esc(c.content)}</div>
        ${c.parent_content ? `<details><summary>展开父片段${c.parent_index != null ? ' ' + esc(c.parent_index + 1) : ''}</summary><div class="chunk-content">${esc(c.parent_content)}</div></details>` : ''}
      </article>`;
    }).join('') : '<p class="s-empty">暂无片段。可检查文档状态，或重建索引后重试。</p>';
  } catch (e) {
    if (!current()) return;
    $('chunk-list').replaceChildren();
    $('kb-error').textContent = '片段加载失败：' + e.message;
    $('kb-retry').hidden = false;
  } finally {
    if (current()) {
      kbView.loading = false;
      updateChunkControls();
    }
  }
}

async function reindexDocument() {
  if (!kbView.doc || kbView.loading || kbView.rebuilding || $('kb-reindex-confirm').hidden) return;
  const id = kbView.doc.id;
  const token = state.token;
  kbView.rebuilding = true;
  $('kb-error').textContent = '';
  $('kb-notice').textContent = '正在重建索引，请稍候…';
  updateChunkControls();
  try {
    const result = await api(`/api/kb/documents/${id}/reindex`, { method: 'POST' });
    if (token !== state.token || kbView.doc?.id !== id) return;
    $('kb-notice').textContent = `重建完成，共 ${result.chunk_count} 个片段。`;
    await loadChunks(1);
  } catch (e) {
    if (token !== state.token || kbView.doc?.id !== id) return;
    $('kb-notice').textContent = '';
    // 失败也重新读取，展示服务端的实际状态和仍可用的片段。
    await loadChunks(1);
    if (token !== state.token || kbView.doc?.id !== id) return;
    $('kb-error').textContent = '重建失败：' + e.message;
  } finally {
    if (token === state.token && kbView.doc?.id === id) {
      kbView.rebuilding = false;
      $('kb-reindex-confirm').hidden = true;
      updateChunkControls();
      await Promise.all([loadDocs(), loadStats()]);
    }
  }
}

async function removeDoc(id, name) {
  if (!confirm(`确定删除「${name}」吗？\n该文档的所有知识片段会一并删除，无法恢复。`)) return;
  try {
    await api('/api/kb/documents/' + id, { method: 'DELETE' });
    toast('已删除');
    await loadDocs();
    await loadStats();
  } catch (e) {
    toast('删除失败：' + e.message);
  }
}

function renderUploads() {
  $('upload-feedback').hidden = uploads.items.length === 0;
  $('upload-area').disabled = uploads.busy;
  $('file-input').disabled = uploads.busy;
  $('upload-clear').disabled = uploads.busy;
  const success = uploads.items.filter(i => i.status === 'success').length;
  const failed = uploads.items.filter(i => i.status === 'failed' || i.status === 'invalid').length;
  const done = success + failed;
  $('upload-summary').textContent = uploads.busy
    ? `正在处理 ${Math.min(done + 1, uploads.items.length)} / ${uploads.items.length} 个文件，请等待解析和入库完成…`
    : `处理完成：${success} 个成功，${failed} 个失败`;
  const labels = { pending: '等待处理', processing: '上传并解析中…', success: '入库成功', failed: '上传或处理失败', invalid: '文件不符合要求' };
  $('upload-results').innerHTML = uploads.items.map((item, index) =>
    `<div class="upload-result ${item.status === 'invalid' ? 'failed' : item.status}">
      <span>${esc(item.file.name)} · ${esc(fileSize(item.file.size))}</span>
      <small>${esc(labels[item.status])}${item.message ? '：' + esc(item.message) : ''}</small>
      ${item.status === 'failed' ? `<button class="btn-ghost" type="button" data-retry="${index}" ${uploads.busy ? 'disabled' : ''}>重新上传</button>` : ''}
    </div>`).join('');
  $('upload-results').querySelectorAll('[data-retry]').forEach(button => {
    button.onclick = () => {
      if (uploads.busy) return;
      const item = uploads.items[Number(button.dataset.retry)];
      if (!confirm(`重新上传「${item.file.name}」？\n如果上次断网，服务端可能已入库，请先查看文档列表，避免重复上传。`)) return;
      processUploads([item]);
    };
  });
}

async function processUploads(items) {
  if (uploads.busy || state.user?.role !== 'admin') return;
  const generation = uploads.generation;
  const token = state.token;
  const current = () => generation === uploads.generation && token === state.token;
  uploads.busy = true;
  for (const item of items) {
    if (!current()) return;
    if (item.status === 'invalid') continue;
    item.status = 'processing';
    item.message = '';
    renderUploads();
    const form = new FormData();
    form.append('file', item.file);
    try {
      const r = await api('/api/kb/documents/upload', { method: 'POST', body: form });
      if (!current()) return;
      item.status = 'success';
      item.message = `${r.chunk_count} 个片段`;
    } catch (e) {
      if (!current()) return;
      item.status = 'failed';
      item.message = e.message;
    }
    renderUploads();
  }
  if (!current()) return;
  uploads.busy = false;
  renderUploads();
  await Promise.all([loadDocs(), loadStats()]);
}

async function uploadFiles(files) {
  // FileList 会随 input.value 清空而变化；先复制，保留批量选择的全部文件。
  const selected = Array.from(files || []);
  if (!selected.length || state.user?.role !== 'admin') return;
  if (uploads.busy) { toast('请等待当前上传完成'); return; }
  uploads.items = selected.map(file => {
    const supported = /\.(pdf|docx?|xlsx?|csv|md|txt|html?)$/i.test(file.name);
    const message = !supported ? '不支持此文件格式' : !file.size ? '文件是空的' : '';
    return { file, status: message ? 'invalid' : 'pending', message };
  });
  renderUploads();
  await processUploads(uploads.items);
}

function resetRetrieval() {
  retrievalView.request++;
  retrievalView.abort?.abort();
  retrievalView.abort = null;
  retrievalView.busy = false;
  retrievalView.last = null;
  $('retrieval-submit').disabled = false;
  $('retrieval-submit').textContent = '开始检索';
  $('retrieval-results').replaceChildren();
  $('retrieval-results').setAttribute('aria-busy', 'false');
  $('retrieval-error').textContent = '';
  $('retrieval-status').textContent = '';
  $('retrieval-retry').hidden = true;
}

function openRetrieval() {
  if (state.user?.role !== 'admin') return;
  resetRetrieval();
  $('retrieval-dialog').showModal();
  $('retrieval-query').focus();
}

function retrievalScore(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '未提供';
}

function renderRetrieval(data, compared) {
  const scoreNames = { vector_score: '向量分数', bm25_score: '关键词分数', rrf_score: '融合分数', rerank_score: '重排序分数' };
  const timingNames = { vector_ms: '向量检索', bm25_ms: '关键词检索', fusion_ms: '融合', rerank_ms: '重排序', total_ms: '总耗时' };
  const timings = Object.entries(timingNames).filter(([key]) => Number.isFinite(data.timings?.[key]))
    .map(([key, label]) => `${label} ${data.timings[key]} ms`).join(' · ');
  const results = data.results || [];
  let html = `<p class="kb-file-info">测试问题：${esc(data.query)}</p>` +
    (timings ? `<p class="account-hint">${esc(timings)}</p>` : '') +
    '<p class="account-hint">不同类型的分数量纲不同，请分别比较。命中片段不代表它一定能回答问题。</p>';
  html += results.length ? results.map(result => {
    const scores = Object.entries(scoreNames).filter(([key]) => result[key] != null)
      .map(([key, label]) => `<span>${label} ${retrievalScore(result[key])}</span>`).join('');
    const sources = (result.sources || []).map(source => ({ vector: '向量', bm25: '关键词' }[source] || source)).join('、');
    return `<article class="chunk-card">
      <div class="chunk-meta">第 ${esc(result.rank)} 条 · ${esc(result.file_name || '来源未知')}${result.page != null ? ' · 页码 ' + esc(result.page) : ''}${result.sheet_name ? ' · 工作表 ' + esc(result.sheet_name) : ''}</div>
      ${sources ? `<p class="account-hint">召回来源：${esc(sources)}${result.hit_both ? '（两路均命中）' : ''}</p>` : ''}
      <div class="retrieval-scores">${scores || '<span>未提供分数</span>'}</div>
      <div class="chunk-content">${esc(result.content || '')}</div>
      ${result.parent_content ? `<details><summary>展开父片段上下文</summary><div class="chunk-content">${esc(result.parent_content)}</div></details>` : ''}
    </article>`;
  }).join('') : '<p class="s-empty">没有命中片段。可以换一种问法，或检查文档是否已完成索引。</p>';
  if (compared && data.comparison) {
    const names = { vector_only: '纯向量检索', hybrid: '向量 + 关键词', hybrid_rerank: '混合检索 + 重排序' };
    const types = { vector: '向量', bm25: '关键词', rerank: '重排序' };
    html += '<section class="retrieval-comparison"><h3>三种方案对比</h3><p class="account-hint">对照排序和片段内容；不同方案的分数不能直接比较大小。</p>';
    for (const [key, name] of Object.entries(names)) {
      const items = data.comparison[key] || [];
      html += `<h3>${name}</h3>` + (items.length ? '<ol>' + items.map(item =>
        `<li><b>${esc(item.file_name || '来源未知')}</b> · ${esc(types[item.score_type] || '检索')}分数 ${retrievalScore(item.score)}<div class="chunk-content">${esc(item.content || '')}</div></li>`
      ).join('') + '</ol>' : '<p class="account-hint">没有命中片段</p>');
    }
    html += '</section>';
  }
  $('retrieval-results').innerHTML = html;
}

async function runRetrieval(retry = false) {
  if (retrievalView.busy || state.user?.role !== 'admin' || !$('retrieval-dialog').open) return;
  const input = retry ? retrievalView.last : {
    query: $('retrieval-query').value.trim(),
    top_k: Number($('retrieval-top-k').value),
    compare: $('retrieval-compare').checked,
  };
  if (!input) return;
  if (!input.query || input.query.length > 2000 || !Number.isInteger(input.top_k) || input.top_k < 1 || input.top_k > 20) {
    $('retrieval-error').textContent = '请输入 1～2000 个字符的问题，返回条数应为 1～20 的整数';
    return;
  }
  const token = state.token;
  const request = ++retrievalView.request;
  const current = () => request === retrievalView.request && token === state.token && $('retrieval-dialog').open;
  retrievalView.last = { ...input };
  retrievalView.busy = true;
  retrievalView.abort = new AbortController();
  $('retrieval-submit').disabled = true;
  $('retrieval-submit').textContent = '检索中…';
  $('retrieval-error').textContent = '';
  $('retrieval-retry').hidden = true;
  $('retrieval-status').textContent = input.compare ? '正在检索并对比三种方案…' : '正在检索资料…';
  $('retrieval-results').replaceChildren();
  $('retrieval-results').setAttribute('aria-busy', 'true');
  const form = new FormData();
  form.append('query', input.query);
  form.append('top_k', input.top_k);
  form.append('compare', input.compare);
  try {
    const data = await api('/api/kb/search', { method: 'POST', body: form, signal: retrievalView.abort.signal });
    if (!current()) return;
    renderRetrieval(data, input.compare);
    $('retrieval-status').textContent = `检索完成，命中 ${data.results?.length || 0} 条片段`;
  } catch (error) {
    if (!current() || error.name === 'AbortError') return;
    $('retrieval-error').textContent = '检索失败：' + error.message;
    $('retrieval-status').textContent = '';
    $('retrieval-retry').hidden = false;
  } finally {
    if (current()) {
      retrievalView.busy = false;
      retrievalView.abort = null;
      $('retrieval-submit').disabled = false;
      $('retrieval-submit').textContent = '开始检索';
      $('retrieval-results').setAttribute('aria-busy', 'false');
    }
  }
}

// ==================================================================
//  八、事件绑定
// ==================================================================

// ---- 登录 / 注册 ----
$('login-btn').onclick = doLogin;
$('register-btn').onclick = doRegister;
$('logout-btn').onclick = () => doLogout(false);
$('account-btn').onclick = openAccountDialog;
$('profile-form').onsubmit = saveProfile;
$('password-form').onsubmit = savePassword;
$('account-close').onclick = () => { if (!accountBusy) $('account-dialog').close(); };
$('account-dialog').oncancel = e => { if (accountBusy) e.preventDefault(); };
$('account-dialog').onclose = clearAccountForms;
$('doc-prev').onclick = () => loadDocs(docsPage - 1);
$('doc-next').onclick = () => loadDocs(docsPage + 1);
$('doc-filter').onsubmit = e => { e.preventDefault(); applyDocFilters(); };
$('doc-status').onchange = applyDocFilters;
$('doc-filter-clear').onclick = () => { $('doc-filter').reset(); applyDocFilters(); };
$('upload-clear').onclick = () => { if (!uploads.busy) { uploads.items = []; renderUploads(); } };
$('retrieval-open').onclick = openRetrieval;
$('retrieval-close').onclick = () => $('retrieval-dialog').close();
$('retrieval-dialog').onclose = resetRetrieval;
$('retrieval-form').onsubmit = e => { e.preventDefault(); runRetrieval(); };
$('retrieval-retry').onclick = () => runRetrieval(true);
$('chunk-prev').onclick = () => loadChunks(kbView.page - 1);
$('chunk-next').onclick = () => loadChunks(kbView.page + 1);
$('kb-retry').onclick = () => loadChunks();
$('kb-close').onclick = () => { if (!kbView.rebuilding) $('kb-dialog').close(); };
$('kb-dialog').oncancel = e => { if (kbView.rebuilding) e.preventDefault(); };
$('kb-dialog').onclose = () => {
  kbView.request++;
  kbView.doc = null;
  kbView.loading = false;
  $('chunk-list').replaceChildren();
};
$('kb-reindex').onclick = () => {
  $('kb-reindex-confirm').hidden = false;
  $('kb-reindex-ok').focus();
};
$('kb-reindex-cancel').onclick = () => { $('kb-reindex-confirm').hidden = true; $('kb-reindex').focus(); };
$('kb-reindex-ok').onclick = reindexDocument;

$('password').onkeydown = e => { if (e.key === 'Enter') doLogin(); };
$('username').onkeydown = e => { if (e.key === 'Enter') $('password').focus(); };
$('reg-username').onkeydown = e => { if (e.key === 'Enter') $('reg-password').focus(); };
$('reg-password').onkeydown = e => { if (e.key === 'Enter') $('reg-password2').focus(); };
$('reg-password2').onkeydown = e => { if (e.key === 'Enter') doRegister(); };

$('to-register').onclick = () => {
  showView('register');
  $('reg-username').focus();
};
$('to-login').onclick = () => {
  showView('login');
  $('username').focus();
};

// ---- 会话列表：一个监听器管全部（事件委托） ----
// 【为什么要用事件委托】
// 会话列表是动态生成的，每次刷新都会重建 DOM。
// 如果给每一行单独绑定点击事件，刷新后就全丢了，得重新绑一遍。
// 委托的做法是：只在父容器上绑一次，靠 e.target 判断点到了谁。
// 不管列表怎么重建，这个监听器都一直在。
$('session-list').onclick = e => {
  const item = e.target.closest('.session-item');
  if (!item) return;

  const id = Number(item.dataset.id);
  const session = state.sessions.find(s => s.id === id);
  if (!session) return;

  const actBtn = e.target.closest('.s-act');
  if (actBtn) {
    e.stopPropagation();
    const act = actBtn.dataset.act;
    if (act === 'rename') openRenameModal(session);
    else if (act === 'pin') togglePin(session);
    else if (act === 'del') deleteSession(session);
    return;
  }

  switchSession(id);
};

$('new-chat-btn').onclick = () => newChat();

// ---- 输入框 ----
$('send-btn').onclick = send;

// 停止生成：拿出"取消令牌"执行取消即可，
// 剩下的（把已生成的部分留住、补上操作按钮）都由 streamAnswer 的收尾段处理
$('stop-btn').onclick = () => {
  if (state.abort) state.abort.abort();
};

// ---- 回答上的所有点击：一个监听器管全部（事件委托） ----
// 【为什么这里也必须用委托，不能挨个绑】
// 两个原因：
//   1. 回答是**逐字刷新**的，每来一个字，正文区的 DOM 就整个重建一次
//   2. 切会话时整片消息区会被清空重建
// 给按钮单独绑事件，赶上这两件事就全丢了（点了没反应，还很难查）。
// 绑在 #messages 这个父容器上则永远有效 —— 不管里面被重建多少遍。
$('messages').onclick = e => {
  // 正文里的引用标注 [1]
  const ref = e.target.closest('.cite-ref');
  if (ref) return jumpToCitation(ref);

  // 信息条右侧的四个操作按钮
  const btn = e.target.closest('.meta-actions .act-btn');
  if (!btn) return;

  const wrap = btn.closest('.msg.ai');
  if (!wrap) return;

  switch (btn.dataset.act) {
    case 'copy':  copyAnswer(wrap, btn);          break;
    case 'regen': regenerate(wrap);               break;
    case 'up':    submitFeedback(wrap, 1, btn);   break;
    case 'down':  submitFeedback(wrap, -1, btn);  break;
  }
};

$('question').onkeydown = e => {
  // Enter 发送，Shift+Enter 换行
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
};

$('question').oninput = function () {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 132) + 'px';
};

// ---- 知识库折叠 ----
$('kb-head').onclick = () => $('kb-section').classList.toggle('collapsed');

// ---- 上传：点击 + 拖拽 ----
$('upload-area').onclick = () => $('file-input').click();

$('file-input').onchange = e => {
  uploadFiles(e.target.files);
  e.target.value = '';   // 清空，这样连续上传同一个文件也能触发
};

{
  const ua = $('upload-area');
  ua.ondragover = e => { e.preventDefault(); if (!uploads.busy) ua.classList.add('dragover'); };
  ua.ondragleave = () => ua.classList.remove('dragover');
  ua.ondrop = e => {
    e.preventDefault();
    ua.classList.remove('dragover');
    uploadFiles(e.dataTransfer.files);
  };
}

// ---- 重命名弹窗：点遮罩空白处关闭 ----
$('rename-mask').onclick = e => {
  if (e.target === $('rename-mask')) $('rename-mask').classList.remove('show');
};
$('rename-cancel').onclick = () => $('rename-mask').classList.remove('show');

// ---- 窄屏侧栏开关 ----
$('menu-btn').onclick = () => $('sidebar').classList.toggle('open');

// ==================================================================
//  九、启动
// ==================================================================
(async function init() {
  if (!state.token) {
    showView('login');
    $('username').focus();
    return;
  }
  // 本地存着令牌 → 试着用它换回用户信息
  try {
    state.user = await api('/api/auth/me');
    await enterApp();
  } catch {
    // 令牌失效，doLogout 已经在 api() 里被调用过了
    showView('login');
  }
})();
