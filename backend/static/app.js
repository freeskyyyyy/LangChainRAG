/* ==================================================================
   服饰知识助手 · 前端逻辑
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

  if (res.status === 401) {
    // 令牌过期或失效 → 退回登录页
    doLogout(true);
    throw new Error('登录已过期，请重新登录');
  }

  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }

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
function md(text) {
  let html = esc(text);

  // 代码块（``` 包裹）
  html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  // 粗体
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // 行内代码
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // 引用标注 [1] → 可点击的小标签
  // data-cite 里存的是引用编号，点它就能跳到下面第 N 张引用卡片。
  // 用 data- 属性而不是把编号写进 onclick，是因为回答里可能有用户
  // 自己输入的 [1]，走属性 + 事件委托更干净，也不会拼出可执行的代码。
  html = html.replace(
    /\[(\d+)\]/g,
    '<span class="cite-ref" data-cite="$1" title="点击查看第 $1 条出处">[$1]</span>'
  );
  // 无序列表
  html = html.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');
  html = html.replace(/<\/ul>\s*<ul>/g, '');

  // 按空行切段落
  html = html.split(/\n{2,}/).map(p => {
    p = p.trim();
    if (!p) return '';
    // 已经是块级元素的，不要再包 <p>
    if (p.startsWith('<ul>') || p.startsWith('<pre')) return p;
    return '<p>' + p.replace(/\n/g, '<br>') + '</p>';
  }).join('');

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

  await loadSessions();
  newChat(false);          // 默认停在一个空白的新对话上

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
      <h2>🌿 你好，我是服饰知识助手</h2>
      <p>
        我会先在你的知识库里查找资料，再基于资料回答，<br>
        并标出每句话来自哪份文件。资料里没有的，我会直接说不知道。
      </p>
      <div class="sugg-grid">
        <button class="sugg">我身高168体重120斤，穿什么尺码</button>
        <button class="sugg">黄皮适合穿什么颜色的衣服</button>
        <button class="sugg">真丝连衣裙怎么洗</button>
        <button class="sugg">羽绒服可以机洗吗</button>
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
  answerEl.innerHTML = md(m.content);

  if (m.citations && m.citations.length) {
    renderCitations(wrap.querySelector('.citations'), m.citations);
  }

  // 历史消息底部同样显示当时的耗时和缓存情况，外加操作按钮
  const tags = [];
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
            answerEl.innerHTML = md(answerText) + '<span class="cursor"></span>';
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
      answerEl.innerHTML = md(answerText);
    }

    const tags = [];
    if (stopped) {
      // 半截回答也要保留 —— 用户按停止通常正是因为"该说的已经说完了"，
      // 把已经生成的部分丢掉反而更让人恼火
      tags.push('<span class="meta-tag">⏹ 已停止</span>');
    }
    if (donePayload) {
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
  try {
    const [s, health] = await Promise.all([
      api('/api/kb/stats'),
      api('/api/health'),
    ]);
    $('kb-stat').innerHTML =
      `<b>文档</b> ${s.documents.total} 份 &nbsp;·&nbsp; ` +
      `<b>片段</b> ${s.content.vector_chunks} 个<br>` +
      `<b>总字数</b> ${s.content.total_chars.toLocaleString()}<br>` +
      `<b>模型</b> ${esc(health.llm_model)}<br>` +
      `<b>缓存命中率</b> ${esc(s.semantic_cache.hit_rate || '0%')}`;
  } catch (e) {
    $('kb-stat').textContent = '统计加载失败：' + e.message;
  }
}

async function loadDocs() {
  if (!state.user || state.user.role !== 'admin') return;
  try {
    const data = await api('/api/kb/documents?page=1&page_size=50');
    const list = $('doc-list');

    if (!data.items.length) {
      list.innerHTML = '<div class="s-empty" style="padding:14px">还没有上传任何资料</div>';
      return;
    }

    list.innerHTML = data.items.map(d => {
      const icon = { indexed: '📄', parsing: '⏳', pending: '⏳', failed: '⚠️' }[d.status] || '📄';
      const color = d.status === 'failed' ? 'color:var(--danger)' : '';
      return `<div class="doc-item" style="${color}">
        <span>${icon}</span>
        <span class="name" title="${esc(d.file_name)}">${esc(d.file_name)}</span>
        <span style="font-size:11px;color:var(--text-3)">${d.chunk_count}段</span>
        <span class="del" data-id="${d.id}" data-name="${esc(d.file_name)}">×</span>
      </div>`;
    }).join('');

    // 用事件委托绑定删除：只绑一次，比给每个 × 单独绑更省
    list.querySelectorAll('.del').forEach(el => {
      el.onclick = () => removeDoc(Number(el.dataset.id), el.dataset.name);
    });
  } catch (e) {
    $('doc-list').innerHTML =
      `<div class="s-empty" style="color:var(--danger)">加载失败：${esc(e.message)}</div>`;
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

async function uploadFiles(files) {
  if (!files.length) return;
  for (const f of files) {
    const form = new FormData();
    form.append('file', f);
    toast(`正在上传解析「${f.name}」…`, 60000);
    try {
      const r = await api('/api/kb/documents/upload', { method: 'POST', body: form });
      toast(`✅ 「${f.name}」入库成功：${r.chunk_count} 个片段`, 3200);
    } catch (e) {
      toast(`❌ 「${f.name}」失败：${e.message}`, 5200);
    }
  }
  await loadDocs();
  await loadStats();
}

// ==================================================================
//  八、事件绑定
// ==================================================================

// ---- 登录 / 注册 ----
$('login-btn').onclick = doLogin;
$('register-btn').onclick = doRegister;
$('logout-btn').onclick = () => doLogout(false);

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
  ua.ondragover = e => { e.preventDefault(); ua.classList.add('dragover'); };
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
