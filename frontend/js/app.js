/* ============================================================================
   AI 代码导师 · 前端交互逻辑

   本文件用原生 JS 编写，不依赖任何框架。整体分成这些块：

     1. 配置与全局状态        —— 接口地址、当前代码、各次结果
     2. 通用工具函数          —— 请求封装、提示条、转义、确认弹窗
     2.5 「有没有代码」的状态管理（需求一）
     3. 代码展示与语法高亮    —— highlight.js 的调用与降级（需求四）
     3.5 代码在线编辑         —— 编辑 / 保存 / 取消、行号、快捷键
     4. 上传与分析            —— 语言识别 + 结构解析
     5. 三个 AI 功能          —— 检测 / 注释 / 改错
     6. 本地项目库            —— 扫描、浏览、编辑保存、替换、删除
     7. 历史记录
     8~10. 面板拖动、窄屏适配、三个拖拽区（需求二 / 需求三）
     11. 页面初始化与事件绑定

   与后端的对接关系：
     /api/v1/tutor 下：
      GET  /status               页面加载时查一次，决定 AI 按钮是否可用
      POST /analyze              上传代码 -> 识别语言 + 解析结构（不用大模型）
      POST /check                AI 检测
      POST /comment              AI 生成注释
      POST /fix                  AI 自动改错
      GET  /history              历史记录列表
      GET  /history/{id}         历史记录详情
      DELETE /history/{id}       删除记录
     /api/v1/library 下（本地项目库，不需要大模型）：
      GET  /scan                 扫描本地文件夹并按语言分类
      GET  /file                 读取某个文件的代码内容（带 sha256，用于乐观锁）
      PUT  /file                 覆盖保存（在线编辑保存 / 用本机文件替换）
      DELETE /file               删除（默认移进 .trash 回收站，可找回）
      GET  /status               项目库配置（代码该放哪个文件夹、是否允许写入）
     /api/v1/files 下（拖拽入库）：
      POST /upload               把文件存到服务器的代码文件夹
      POST /scan                 按语言归档到 c/ java/ python/，并写入台账
   ========================================================================== */

'use strict';

/* ==========================================================================
   1. 配置与全局状态
   ========================================================================== */

/**
 * AI 导师接口地址。
 * 页面与后端同源（都从 127.0.0.1:8000 提供），因此用相对路径即可。
 * 如果以后把前端单独部署到别的域名，把这里改成完整地址，
 * 例如 'http://127.0.0.1:8000/api/v1/tutor'（后端已开启 CORS）。
 */
const API_BASE = '/api/v1/tutor';

/**
 * 本地项目库接口地址。
 * 单独一个常量而不是拼字符串，是为了以后换部署方式时一眼能看清有几处地址。
 */
const LIBRARY_BASE = '/api/v1/library';

/**
 * 文件管理接口地址（拖拽入库 / 按语言归档 / 台账）。
 * 拖进「② 本地项目库」的文件会走这里：
 *   1) POST /upload  把文件存到服务器的代码文件夹
 *   2) POST /scan    按语言归档到 c/ java/ python/，并写入 SQLite 台账
 */
const FILES_BASE = '/api/v1/files';

/**
 * 模型与 API Key 接口地址。
 * 页面顶部「模型设置」卡片会用到：
 *   GET  /models/list     有哪些模型可选（不含 Key）
 *   POST /auth/set_key    把用户填的 Key 存进服务端内存会话
 *   POST /auth/clear_key  清除会话里的 Key
 */
const MODELS_BASE = '/api/v1/models';
const AUTH_BASE = '/api/v1/auth';

/** 全局状态：当前正在处理的代码，以及各个接口返回的结果 */
const state = {
  // ---- 代码是否已载入：占位提示与三个 AI 按钮的显隐都由它决定（见 setHasCode）----
  hasCode: false,
  filename: '',        // 当前代码的文件名
  code: '',            // 当前代码内容（编辑时跟着 textarea 实时更新）
  savedCode: '',       // 上一次"载入或保存"时的内容，用来判断有没有改动
  language: '',        // 后端识别出的语言标识（c / java / python …）
  languageLabel: '',   // 语言展示名
  aiAvailable: false,  // 后端 .env 里是否配置了大模型
  backendModel: '',    // 后端默认模型名（用于徽章上显示）
  history: [],         // 历史记录
  libraryFiles: [],    // 项目库扫描到的文件
  libraryFilter: '',   // 当前选中的语言筛选（'' 表示全部）
  libraryRoots: [],    // 项目库根目录（含绝对路径），供「复制路径」按钮使用
  // ---- 项目库写入相关的状态 ----
  allowWrite: true,    // 后端是否允许编辑/替换/删除（LIBRARY__ALLOW_WRITE）
  trashDirName: '.trash',  // 回收站目录名，删除后的提示语里要用
  // 当前预览的代码来自项目库里的哪个文件；不是库文件时为 null。
  // 在线编辑保存要靠它决定"写回哪个文件"，没它就只能改预览。
  libraryTarget: null, // { relPath, root, sha256 }
  replaceTarget: null, // 点「替换」时记下的目标文件，等学生选完本机文件再用
  // ---- 在线编辑状态 ----
  editing: false,      // 是否处于编辑模式
  dirty: false,        // 编辑后有没有未保存的改动
  // ---- 模型设置（选模型 + 填自己的 API Key）----
  // 注意：这三个都只放在**内存**里，绝不写进 localStorage。
  // 页面一刷新就没了，需要重新填 Key —— 这是刻意的隐私取舍。
  models: [],          // /models/list 返回的模型清单（不含 Key）
  modelId: '',         // 当前选中的模型 ID，例如 deepseek
  sessionId: '',       // 服务端返回的会话号；后续 AI 请求放在 X-Session-Id 头里
  modelConnected: false,  // 是否已经"保存并测试"成功过
  modelLabel: '',      // 已连接模型的展示名（徽章与提示语里用）
  modelName: '',       // 服务商实际返回的模型名（可能和请求名不同）
  modelTtlMinutes: 30,    // 后端给的 Key 有效期（分钟），提示语里用得到
  // 面板尺寸（需求三）：被用户拖过之后记在这里，并写入 localStorage
  layout: {},
};

/** 常用的 DOM 元素引用，统一取一次，避免到处 querySelector */
const el = {
  // 模型设置（选模型 + 填自己的 API Key）
  modelCard: document.getElementById('modelCard'),
  modelSelect: document.getElementById('modelSelect'),
  modelApiKey: document.getElementById('modelApiKey'),
  toggleKeyVisible: document.getElementById('toggleKeyVisible'),
  modelHint: document.getElementById('modelHint'),
  modelStatus: document.getElementById('modelStatus'),
  modelStatusText: document.getElementById('modelStatusText'),
  saveModelKey: document.getElementById('saveModelKey'),
  clearModelKey: document.getElementById('clearModelKey'),
  refreshModels: document.getElementById('refreshModels'),

  // 左侧栏
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  pasteName: document.getElementById('pasteName'),
  pasteCode: document.getElementById('pasteCode'),
  pasteBtn: document.getElementById('pasteBtn'),
  classifyCard: document.getElementById('classifyCard'),
  classifyList: document.getElementById('classifyList'),
  structureBox: document.getElementById('structureBox'),
  structureList: document.getElementById('structureList'),
  historyList: document.getElementById('historyList'),
  refreshHistory: document.getElementById('refreshHistory'),

  // 本地项目库
  libraryHint: document.getElementById('libraryHint'),
  libraryChips: document.getElementById('libraryChips'),
  libraryList: document.getElementById('libraryList'),
  libraryMore: document.getElementById('libraryMore'),
  refreshLibrary: document.getElementById('refreshLibrary'),
  libraryDropzone: document.getElementById('libraryDropzone'),
  // 项目库文件夹路径：常显，回答"我的代码存在哪个文件夹"（需求：本地项目库位置可见）
  libraryPathValue: document.getElementById('libraryPathValue'),
  copyLibraryPath: document.getElementById('copyLibraryPath'),

  // 顶部状态
  aiBadge: document.getElementById('aiBadge'),
  langBadge: document.getElementById('langBadge'),

  // 代码区
  codeTitle: document.getElementById('codeTitle'),
  codeLangBadge: document.getElementById('codeLangBadge'),
  // 占位提示：显隐靠加/减 hidden 类，不改它的文字
  codePlaceholder: document.getElementById('code-placeholder'),
  codeDropzone: document.getElementById('codeDropzone'),
  codeBlock: document.getElementById('codeBlock'),
  codeContent: document.getElementById('codeContent'),
  clearCodeBtn: document.getElementById('clearCodeBtn'),
  // 在线编辑
  codeEditBtn: document.getElementById('codeEditBtn'),
  codeSaveBtn: document.getElementById('codeSaveBtn'),
  codeCancelBtn: document.getElementById('codeCancelBtn'),
  codeDirtyBadge: document.getElementById('codeDirtyBadge'),
  editorWrap: document.getElementById('editorWrap'),
  editorGutter: document.getElementById('editorGutter'),
  codeEditor: document.getElementById('codeEditor'),

  // 替换文件用的一次性选择器 + 确认弹窗
  replaceInput: document.getElementById('replaceInput'),
  confirmModal: document.getElementById('confirmModal'),
  confirmTitle: document.getElementById('confirmTitle'),
  confirmText: document.getElementById('confirmText'),
  confirmOkBtn: document.getElementById('confirmOkBtn'),
  confirmCancelBtn: document.getElementById('confirmCancelBtn'),

  // 面板拖动（需求三）：三根分隔条 + 需要测量尺寸的几个容器
  resizerLeft: document.getElementById('resizerLeft'),
  resizerRight: document.getElementById('resizerRight'),
  resizerCode: document.getElementById('resizerCode'),
  sidebar: document.querySelector('.sidebar'),
  resultPane: document.querySelector('.pane:last-of-type'),
  codeCard: document.getElementById('codeCard'),
  codeActions: document.getElementById('codeActions'),

  // 按钮
  btnCheck: document.getElementById('btnCheck'),
  btnComment: document.getElementById('btnComment'),
  btnFix: document.getElementById('btnFix'),
  actionHint: document.getElementById('actionHint'),

  // 结果区
  resultEmpty: document.getElementById('resultEmpty'),
  resultCheck: document.getElementById('resultCheck'),
  resultComment: document.getElementById('resultComment'),
  resultFix: document.getElementById('resultFix'),

  scoreCircle: document.getElementById('scoreCircle'),
  scoreValue: document.getElementById('scoreValue'),
  scoreLevel: document.getElementById('scoreLevel'),
  scoreSummary: document.getElementById('scoreSummary'),
  issueList: document.getElementById('issueList'),
  highlightBox: document.getElementById('highlightBox'),
  highlightList: document.getElementById('highlightList'),

  commentSummary: document.getElementById('commentSummary'),
  commentCode: document.getElementById('commentCode'),

  fixSummary: document.getElementById('fixSummary'),
  fixNoError: document.getElementById('fixNoError'),
  fixBody: document.getElementById('fixBody'),
  changeList: document.getElementById('changeList'),
  fixCode: document.getElementById('fixCode'),

  toast: document.getElementById('toast'),
};

/* ==========================================================================
   2. 通用工具函数
   ========================================================================== */

/**
 * 弹出底部提示条。
 * @param {string} message 提示文字
 * @param {'info'|'ok'|'error'} kind 类型，决定颜色
 */
let toastTimer = null;
function toast(message, kind = 'info') {
  el.toast.textContent = message;
  el.toast.className = 'toast' + (kind === 'error' ? ' toast--error' : kind === 'ok' ? ' toast--ok' : '');
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  // 2.6 秒后自动隐藏，不用学生手动关
  toastTimer = setTimeout(() => { el.toast.hidden = true; }, 2600);
}

/**
 * 调用后端接口的统一入口。
 * 负责：拼地址、发请求、解析 JSON、把后端的错误提示转成中文抛出。
 *
 * 顺手做一件事：只要页面上已经拿到会话号（用户填过 Key），
 * 就自动带上 `X-Session-Id` 请求头——这样后面所有 AI 调用都会用
 * "用户选的那个模型 + 用户自己那把 Key"，不需要每个调用点各写一遍。
 *
 * @param {string} path   接口路径，例如 '/check'
 * @param {object} options fetch 的选项；body 可直接传普通对象
 * @param {string} base   接口前缀，默认是 AI 导师接口；项目库接口传 LIBRARY_BASE
 * @returns {Promise<object>} 后端返回的 JSON
 */
async function api(path, options = {}, base = API_BASE) {
  const url = base + path;

  // 会话号只放在内存里（state.sessionId），绝不写 localStorage：
  // 它等价于"临时凭据"，泄露出去别人就能用你的 Key 调模型。
  const headers = { ...(options.headers || {}) };
  if (state.sessionId) headers['X-Session-Id'] = state.sessionId;

  const response = await fetch(url, { ...options, headers });

  // 204 表示删除成功，没有响应体
  if (response.status === 204) return {};

  let data = null;
  try {
    data = await response.json();
  } catch (err) {
    throw new Error(`后端返回的内容无法解析（HTTP ${response.status}）`);
  }

  if (!response.ok) {
    // FastAPI 的错误格式是 { detail: "..." }，直接透传给学生看
    const detail = data && data.detail ? data.detail : `请求失败（HTTP ${response.status}）`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data;
}

/**
 * 把「代码文本 + 文件名」组装成 FormData，
 * 后端三个 AI 接口和 analyze 都支持这种提交方式。
 *
 * 顺便带上当前选中的模型 ID：这样后端就知道该用哪个模型，
 * 而 Key 走 `X-Session-Id` 请求头（见 api()），不在这里重复传。
 */
function codeFormData() {
  const form = new FormData();
  form.append('code', state.code);
  form.append('filename', state.filename);
  // 带上用户选的模型（Key 不必带：它已经在服务端会话里，走 X-Session-Id 头）
  if (state.modelId) form.append('model_id', state.modelId);
  return form;
}

/** 转义 HTML，防止代码里的 < > & 破坏页面结构 */
function escapeHtml(text) {
  return String(text == null ? '' : text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/** 把换行符转成 <br>，用于展示短文本 */
function nl2br(text) {
  return escapeHtml(text).replace(/\n/g, '<br>');
}

/* ==========================================================================
   2.6 确认弹窗
   --------------------------------------------------------------------------
   删除文件、放弃未保存的修改这类操作都不可逆，必须先让用户确认。
   为什么不用浏览器自带的 confirm()：
     · 它会把整个页面冻住（连动画都停），样式也没法改，跟页面风格脱节；
     · 它返回的是同步的 true/false，没法用 await 编排，行为测试也不好写。
   这里用 Promise 包一层，调用处写 `if (!(await showConfirm({...}))) return;` 即可。
   ========================================================================== */

/** 当前等待用户点击的 resolve 函数；没有弹窗时为 null */
let confirmResolver = null;

/**
 * 弹出确认框，等待用户选择。
 *
 * @param {object} options
 * @param {string} options.title    标题
 * @param {string} options.text     正文（**允许含 HTML**，调用方必须自己转义过）
 * @param {string} [options.okText] 确定按钮文字
 * @param {boolean} [options.danger] true = 确定按钮用红色（危险操作）
 * @returns {Promise<boolean>} 用户点「确定」为 true，点「取消」/遮罩/Esc 为 false
 */
function showConfirm({ title, text, okText = '确定', danger = false }) {
  el.confirmTitle.textContent = title;
  el.confirmText.innerHTML = text;
  el.confirmOkBtn.textContent = okText;
  // 危险操作把确定按钮变红：手快的人也能在点下去之前"看见"后果
  el.confirmOkBtn.classList.toggle('btn--danger', Boolean(danger));
  el.confirmModal.classList.remove('hidden');

  // 焦点落在「取消」上：直接回车不会误删文件（默认操作要选安全的那个）
  if (typeof el.confirmCancelBtn.focus === 'function') el.confirmCancelBtn.focus();

  return new Promise((resolve) => {
    confirmResolver = resolve;
  });
}

/** 关闭确认框并返回用户的选择（重复调用时后一次无效，避免重复 resolve） */
function closeConfirm(result) {
  el.confirmModal.classList.add('hidden');
  const resolve = confirmResolver;
  confirmResolver = null;
  if (resolve) resolve(Boolean(result));
}

/**
 * 有未保存的修改时先问一句，避免换文件/清空把学生写的东西默默丢掉。
 * @returns {Promise<boolean>} true = 可以继续（没有改动，或用户确认丢弃）
 */
async function confirmDiscardChanges() {
  if (!state.dirty) return true;
  return showConfirm({
    title: '有还没保存的修改',
    text: '当前编辑的内容还没有保存，继续操作会丢掉这些修改。',
    okText: '丢掉修改继续',
    danger: true,
  });
}

/** 文件扩展名（小写，含点）；没有扩展名时返回空字符串 */
function fileExtension(name) {
  const text = String(name || '');
  const index = text.lastIndexOf('.');
  return index <= 0 ? '' : text.slice(index).toLowerCase();
}

/**
 * 「现在能不能用 AI」的唯一判断。
 *
 * 两个来源，满足一个就行：
 *   1. `state.aiAvailable`   —— 后端 `.env` 里配好了 Key；
 *   2. `state.modelConnected`—— 用户在页面「模型设置」里填了 Key 并测通了。
 *
 * 为什么要有它：以前只认第 1 条，于是"页面上刚填好 Key、右上角还写着
 * AI 未启用、三个按钮还是灰的"——这正是用户反馈的问题。
 * 现在按钮与徽章都走这一个函数，不会再出现两边不一致。
 */
function aiReady() {
  return Boolean(state.aiAvailable || state.modelConnected);
}

/**
 * 刷新右上角的 AI 状态徽章（三种状态）。
 *
 * · 已连接（用户自己的 Key）→ 绿色：`AI 已就绪 · DeepSeek 官方`
 * · 后端已配置 Key          → 绿色：`AI 已就绪 · deepseek-chat`
 * · 都没有                  → 灰色：`AI 未启用`
 */
function renderAiBadge() {
  if (state.modelConnected) {
    // 优先显示用户自己连上的那个模型（他更关心"我现在用的是哪个"）
    const label = state.modelLabel || state.modelName || '已连接';
    el.aiBadge.textContent = `AI 已就绪 · ${label}`;
    el.aiBadge.className = 'badge badge--ok';
    return;
  }
  if (state.aiAvailable) {
    el.aiBadge.textContent = `AI 已就绪 · ${state.backendModel || ''}`.trim();
    el.aiBadge.className = 'badge badge--ok';
    return;
  }
  el.aiBadge.textContent = 'AI 未启用';
  el.aiBadge.className = 'badge badge--off';
}

/**
 * 按钮进入/退出「处理中」状态的统一封装。
 *
 * 三个 AI 按钮可点的条件是**同时**满足：
 *   1. 当前不在处理中（避免连点导致重复计费）
 *   2. AI 可用——后端配了 Key，或用户在页面上填了自己的 Key（见 aiReady）
 *   3. 当前**有代码**——没代码时点了只会拿到"代码内容为空"的报错，
 *      不如直接禁用，学生一眼就知道"得先放代码进来"
 *
 * 把三条判断收在这一个函数里，调用点就不必各写一遍，
 * 也不会出现"AI 可用但还没代码，按钮却是亮的"这种状态不一致。
 */
function setBusy(busy, hintText = '') {
  const clickable = !busy && aiReady() && state.hasCode;
  el.btnCheck.disabled = !clickable;
  el.btnComment.disabled = !clickable;
  el.btnFix.disabled = !clickable;
  el.actionHint.textContent = hintText;
}

/* ==========================================================================
   2.5 「有没有代码」的统一状态管理（需求一）
   --------------------------------------------------------------------------
   为什么要有这么一个函数：
   "占位提示该不该显示"这件事，表面上只是中间区域的一个样式问题，
   但它其实取决于好几个动作——拖文件进来、粘贴、点项目库里的文件、点清空——
   如果每个动作里各写一遍 `placeholder.hidden = true/false`，
   只要漏掉一处就会出现"代码都显示出来了，'还没有代码'还压在下面"这种 bug
   （这就是之前真实踩到的坑）。

   现在的做法：
     · 只认 `state.hasCode` 这一个来源；
     · 所有会改变"有没有代码"的地方，都必须调用 setHasCode()；
     · 显隐一律通过加/减 CSS 类实现，不修改提示文字、不直接写 style.display，
       样式与逻辑彻底分开（需求一第 2、5 条）。
   ========================================================================== */

/**
 * 切换「当前是否有代码」这个状态，并同步所有相关的界面元素。
 *
 * @param {boolean} hasCode true = 已有代码（占位提示隐藏）
 */
function setHasCode(hasCode) {
  state.hasCode = Boolean(hasCode);

  // ① 占位提示 / ② 代码块 / ③ 在线编辑器：三个东西互斥，统一在一个函数里算，
  //    否则很容易出现"编辑器和代码块同时显示"这种叠影（见 applyCodeVisibility）
  applyCodeVisibility();

  // ④ 语言徽章：没有代码时不该显示"Python"这种残留标签
  el.codeLangBadge.classList.toggle('hidden', !state.hasCode || !state.languageLabel);

  // ⑤ 三个 AI 按钮：setBusy 里已经把"有没有代码"算进可点条件了，这里调用一次即可
  setBusy(false);
}

/**
 * 统一计算「占位提示 / 只读代码块 / 在线编辑器」谁该显示。
 *
 * 这三个元素的显隐条件互相牵连（有代码才显示代码块，编辑时又要让位给编辑器），
 * 分散在三处各写一遍必然会出现漏改，所以收敛到这里。
 * 显隐一律通过加/减 `hidden` 类实现，不直接改 style.display。
 */
function applyCodeVisibility() {
  el.codePlaceholder.classList.toggle('hidden', state.hasCode || state.editing);
  el.codeBlock.classList.toggle('hidden', !state.hasCode || state.editing);
  el.editorWrap.classList.toggle('hidden', !state.editing);
  refreshEditButtons();
}

/**
 * 刷新「编辑 / 保存 / 取消」三个按钮的显隐与可用状态。
 *
 * 只读模式（LIBRARY__ALLOW_WRITE=false）下：
 *   · 项目库里的文件：编辑按钮直接禁用，鼠标悬停能看到原因；
 *   · 只是拖进来/粘贴的代码（不落盘）：照样可以改着玩，保存时只更新预览。
 */
function refreshEditButtons() {
  const isLibraryFile = Boolean(state.libraryTarget);
  const readOnly = isLibraryFile && !state.allowWrite;

  el.codeEditBtn.hidden = state.editing;
  el.codeSaveBtn.hidden = !state.editing;
  el.codeCancelBtn.hidden = !state.editing;

  el.codeEditBtn.disabled = !state.hasCode || readOnly;
  el.codeEditBtn.title = readOnly
    ? '项目库当前是只读模式（LIBRARY__ALLOW_WRITE=false），不能修改文件'
    : '在线编辑这段代码';

  // 未保存标记：只有真的改过内容才显示，否则"进去又出来"也会闪一个警告，很吵
  el.codeDirtyBadge.classList.toggle('hidden', !state.editing || !state.dirty);
}

/**
 * 清空当前代码，让占位提示重新出现（需求一第 4 条）。
 *
 * 清空的不只是代码本身，还有跟着它产生的一切：
 *   · 文件名 / 语言 → 否则下次载入别的文件时会串味
 *   · 代码区标题与徽章
 *   · 编辑状态与「这段代码来自哪个文件」的记认
 *   · 左侧「③ 自动分类结果」卡片（那是上一份代码的结论，留着会误导）
 *   · 右侧结果区（回到空状态）
 *
 * @param {{silent?: boolean}} [options] silent=true 时不弹"已清空代码"提示
 *        （删除文件后顺手清空预览时用，否则会把删除结果那条提示顶掉）
 */
function clearCode(options = {}) {
  // 注意：这个函数会被直接当作 click 回调（el.clearCodeBtn.addEventListener），
  // 那时第一个参数是事件对象，没有 silent 属性，所以这里必须判类型而不是判真假。
  const silent = Boolean(options) && options.silent === true;

  state.filename = '';
  state.code = '';
  state.savedCode = '';
  state.language = '';
  state.languageLabel = '';
  state.libraryTarget = null;

  // 退出编辑模式并清掉"未保存"标记，否则清空后还挂着一个红点
  state.editing = false;
  setDirty(false);
  el.codeEditor.value = '';
  el.codeDropzone.classList.remove('is-editing');

  cancelPendingHighlight();   // 停掉大文件没做完的分批高亮，避免往空代码里写入旧片段
  el.codeContent.textContent = '';
  el.codeContent.className = '';
  el.codeTitle.textContent = '代码预览';

  // 上一份代码的分类结论作废
  el.classifyCard.hidden = true;
  el.structureBox.hidden = true;

  // 右侧结果区回到空状态
  el.resultEmpty.hidden = false;
  el.resultCheck.hidden = true;
  el.resultComment.hidden = true;
  el.resultFix.hidden = true;

  setHasCode(false);
  if (!silent) toast('已清空代码', 'ok');
}

/* ==========================================================================
   2.7 模型设置（选模型 + 填自己的 API Key）
   --------------------------------------------------------------------------
   这一块解决的是"一个服务只能用一个模型、Key 只能写死在 .env 里"的问题：

     GET  /api/v1/models/list   → 下拉框的选项（不含任何 Key）
     POST /api/v1/auth/set_key  → 把 Key 存进**服务端内存会话**，返回会话号
     POST /api/v1/auth/clear_key→ 清除会话（退出 / 换一把 Key）

   隐私设计（有意为之，别改成 localStorage）：
     · Key 只在提交那一刻经过一次网络，之后**前端不再保存它**；
     · 前端只记住会话号，而且放在 JS 变量里 —— 刷新页面就没了，
       需要重新填一次；这是"宁可多填一次，也不把凭据留在浏览器里"的取舍；
     · 后端那边也是内存存储 + 30 分钟不用自动清除（见 api_key_manager.py）。
   ========================================================================== */

/** 状态提示区的三种样式（css 里用颜色区分） */
const MODEL_STATUS_CLASS = { ok: 'is-ok', error: 'is-error', busy: 'is-busy' };

/**
 * 更新「状态提示」那一行。
 *
 * @param {string} text  要说的话（中文，面向学生）
 * @param {'idle'|'ok'|'error'|'busy'} kind 状态种类，决定小圆点的颜色
 */
function setModelStatus(text, kind = 'idle') {
  el.modelStatusText.textContent = text;
  el.modelStatus.classList.remove('is-ok', 'is-error', 'is-busy');
  if (MODEL_STATUS_CLASS[kind]) el.modelStatus.classList.add(MODEL_STATUS_CLASS[kind]);
  state.modelConnected = kind === 'ok';
}

/** 取当前下拉框选中的模型对象（找不到时返回 null） */
function currentModel() {
  return state.models.find((item) => item.id === state.modelId) || null;
}

/**
 * 拉取模型清单并填充下拉框。
 *
 * 失败也不算致命：页面其余功能（项目库、文件分类、在线编辑）都不依赖模型，
 * 所以这里只把状态提示改成"读取失败"，不抛异常打断整页初始化。
 */
async function loadModels() {
  setModelStatus('正在读取模型清单…', 'busy');
  try {
    const data = await api('/list', {}, MODELS_BASE);
    state.models = data.models || [];

    // 后端没配任何模型时，说清楚"去哪儿配"，而不是给一个空下拉框
    if (!state.models.length) {
      el.modelSelect.innerHTML = '<option value="">（后端还没有配置模型）</option>';
      el.modelHint.textContent = '请在 .env 里配置 LLM__MODELS__<ID>__* 之后点右上角 ↻ 重新读取。';
      setModelStatus('未配置模型：AI 功能暂不可用', 'error');
      return;
    }

    // 尽量保留用户已经选过的模型；没有就选后端标的默认模型
    const keep = state.modelId && state.models.some((m) => m.id === state.modelId);
    state.modelId = keep ? state.modelId : (data.default_model_id || state.models[0].id);

    renderModelOptions();
    updateModelHint();
    setModelStatus('请选择模型并填写 API Key（也可以直接用后端配好的）', 'idle');
  } catch (err) {
    el.modelSelect.innerHTML = '<option value="">（读取失败）</option>';
    setModelStatus(`读取模型清单失败：${err.message}`, 'error');
  }
}

/** 渲染下拉框选项：展示名 + 「免 Key / 后端已有 Key」这类提示 */
function renderModelOptions() {
  el.modelSelect.innerHTML = state.models.map((item) => {
    // 选项文字里带上关键信息，学生不用点开就知道哪个能直接用
    const tag = item.requires_api_key
      ? (item.has_default_key ? '· 后端已有 Key' : '· 需要填 Key')
      : '· 本地模型免 Key';
    const selected = item.id === state.modelId ? ' selected' : '';
    return `<option value="${escapeHtml(item.id)}"${selected}>${escapeHtml(item.label)} ${tag}</option>`;
  }).join('');
}

/**
 * 按当前模型更新下面的说明文字与输入框状态。
 *
 * 两个特殊情况要处理好：
 *   · 本地模型（Ollama）：不需要 Key，把输入框禁掉并说明原因；
 *   · 后端已经配了 Key：告诉学生"不填也能用"，避免他以为必须先填。
 */
function updateModelHint() {
  const model = currentModel();
  if (!model) {
    el.modelHint.textContent = '';
    return;
  }

  const lines = [`模型名：${model.model_name}`, `接口地址：${model.base_url}`];
  if (model.description) lines.push(model.description);

  if (!model.requires_api_key) {
    lines.push('这是本地模型，不需要 API Key。');
  } else if (model.has_default_key) {
    lines.push('后端已经配好 Key，不填也能用；想用自己的 Key 就填在下面。');
  } else {
    lines.push('这个模型需要 API Key，请粘贴你自己的 Key。');
  }

  // 后端只配了一个模型时，必须说清"为什么只有一个"，并给出加模型的办法。
  // 不加这一段的话，下拉框里孤零零一个选项很容易被理解成
  // 「这个项目就只支持这一个模型」——而其实是 .env 里没写多模型那一段。
  if (state.models.length === 1) {
    lines.push(
      state.models[0].id === 'default'
        ? '后端现在用的是「单模型」写法（.env 里的 LLM__MODEL），所以只有这 1 个模型可选。'
        : '后端目前只配置了这 1 个模型。',
      '想让下拉框里多出几个（DeepSeek / 通义千问 / 本地 Ollama…）：'
        + '在 .env 里加一段 LLM__MODELS__<ID>__*，重启服务后点本卡片标题右边的 ↻ 重新读取。',
    );
  }

  el.modelHint.textContent = lines.join('\n');
  el.modelApiKey.disabled = !model.requires_api_key;
  el.modelApiKey.placeholder = model.requires_api_key
    ? '粘贴你的 Key（如 sk-…）'
    : '本地模型不需要 Key';
}

/**
 * 「保存并测试」：先存 Key，再真的调一次 AI 检测来验证它能用。
 *
 * 为什么必须"测试"：只保存的话，Key 写错了要等学生点了「AI 检测」才发现；
 * 这里用一段 3 行的代码调一次 `/tutor/check`，成功才显示"已连接"。
 * 测试失败时把后端的中文提示原样展示出来（例如"API Key 无效或已过期"）。
 */
async function saveAndTestModel() {
  const model = currentModel();
  if (!model) {
    setModelStatus('请先选择一个模型', 'error');
    return;
  }

  const apiKey = el.modelApiKey.value.trim();
  if (model.requires_api_key && !apiKey) {
    setModelStatus('请先填入 API Key（本地模型才不需要）', 'error');
    el.modelApiKey.focus();
    return;
  }

  setModelStatus('正在保存并测试…', 'busy');
  el.saveModelKey.disabled = true;

  try {
    // ---- 第 1 步：把 Key 存进服务端内存会话 ----
    const saved = await api('/set_key', {
      method: 'POST',
      body: JSON.stringify({ model_id: model.id, api_key: apiKey }),
      headers: { 'Content-Type': 'application/json' },
    }, AUTH_BASE);

    state.sessionId = saved.session.session_id;   // 只放内存
    if (saved.ttl_seconds) {
      state.modelTtlMinutes = Math.round(saved.ttl_seconds / 60);
    }

    // 存完就把输入框清掉：Key 已经交给服务端了，前端不留副本
    el.modelApiKey.value = '';

    // ---- 第 2 步：拿它真调一次 AI 检测，验证 Key 真的能用 ----
    const probe = new FormData();
    probe.append('code', 'def add(a, b):\n    return a + b\n');
    probe.append('filename', '_key_check.py');
    probe.append('model_id', model.id);

    const result = await api('/check', { method: 'POST', body: probe });

    // 判断依据：`/tutor/check` 在模型不可用时会**直接返回错误**（503/502/504），
    // 所以"能走到这一行"本身就说明模型真的被调用了。
    // 下面这个 ai_available 判断是一道防线：万一以后接口改成"降级返回 200 +
    // ai_available=false"，这里也不会误报"已连接"骗学生。
    if (result.ai_available === false) {
      setModelStatus(
        `${model.label} 已保存，但这次没有真正调用模型；请检查 Key 是否正确`,
        'error',
      );
      return;
    }

    // 响应里的 model 是服务商实际用的模型名（可能与请求名不同，如实显示）
    const used = result.model || model.model_name;
    setModelStatus(`已连接 ${model.label}（${used}）`, 'ok');
    toast(`已连接 ${model.label}`, 'ok');

    // 连上之后要做三件事，缺一件就会出现"填了 Key 但右上角还写着 AI 未启用"：
    //   1. 记下"用户自己的 Key 已连上"（不覆盖 aiAvailable，那是后端的事实）；
    //   2. 刷新右上角徽章；
    //   3. 让三个 AI 按钮可点（之前因为没配 Key 是灰的）。
    state.modelConnected = true;
    state.modelLabel = model.label;
    state.modelName = used;
    renderAiBadge();
    setBusy(false);
  } catch (err) {
    // 后端已经把它们翻译成中文了（Key 无效 / 超时 / 连不上…），直接展示
    setModelStatus(`连接失败：${err.message}`, 'error');
    toast('模型连接失败，请检查 Key 或换一个模型', 'error');
  } finally {
    el.saveModelKey.disabled = false;
  }
}

/** 「清除 Key」：清掉服务端会话 + 清掉前端内存里的痕迹 */
async function clearModelKey() {
  if (!state.sessionId) {
    el.modelApiKey.value = '';
    setModelStatus('当前没有保存的 Key', 'idle');
    return;
  }

  setModelStatus('正在清除…', 'busy');
  try {
    await api('/clear_key', {
      method: 'POST',
      body: JSON.stringify({ session_id: state.sessionId }),
      headers: { 'Content-Type': 'application/json' },
    }, AUTH_BASE);
    setModelStatus('已清除 Key（需要重新填写才能使用 AI 功能）', 'idle');
    toast('已清除 API Key', 'ok');
  } catch (err) {
    // 清除失败也要把本地状态清掉：会话可能已经过期了
    setModelStatus(`清除失败：${err.message}`, 'error');
  } finally {
    // 无论服务端结果如何，前端一定不留：会话号与输入框都清空
    state.sessionId = '';
    el.modelApiKey.value = '';
    // 连线状态一并复位：徽章退回"后端有没有配 Key"的实际情况，
    // 三个 AI 按钮也跟着变灰（没别的地方能用了）
    state.modelConnected = false;
    state.modelLabel = '';
    state.modelName = '';
    renderAiBadge();
    setBusy(false);
  }
}

/** 绑定模型设置区的所有交互 */
function bindModelSettings() {
  // 换模型：更新说明文字，并提示"改了模型要重新保存并测试"
  el.modelSelect.addEventListener('change', () => {
    state.modelId = el.modelSelect.value;
    updateModelHint();

    // 换了模型，"上一个模型验证通过"这个结论就不成立了：
    // 撤掉已连接标记，免得右上角还挂着"AI 已就绪 · 旧模型"骗人。
    // （后端自己配了 Key 时 aiReady() 仍为 true，按钮不会因此变灰。）
    if (state.modelConnected) {
      state.modelConnected = false;
      state.modelLabel = '';
      state.modelName = '';
      renderAiBadge();
      setBusy(false);
    }
    setModelStatus('模型已切换，点「保存并测试」后生效', 'idle');
  });

  // 显示 / 隐藏 Key：只是把 type 在 password 与 text 之间切一下
  el.toggleKeyVisible.addEventListener('click', () => {
    const showing = el.modelApiKey.type === 'text';
    el.modelApiKey.type = showing ? 'password' : 'text';
    el.toggleKeyVisible.textContent = showing ? '显示' : '隐藏';
  });

  el.saveModelKey.addEventListener('click', saveAndTestModel);
  el.clearModelKey.addEventListener('click', clearModelKey);
  el.refreshModels.addEventListener('click', loadModels);

  // 在输入框里按回车 = 点「保存并测试」，省一次鼠标移动
  el.modelApiKey.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      saveAndTestModel();
    }
  });
}

/* ==========================================================================
   3. 代码展示与语法高亮（大文件分批高亮 —— 需求四）
   ========================================================================== */

/** 语言标识 -> highlight.js 的语言名（两者基本一致，Java 需要小写） */
function hljsLang(language) {
  const map = { c: 'c', java: 'java', python: 'python', javascript: 'javascript', go: 'go' };
  return map[language] || 'plaintext';
}

/**
 * 超过这个行数就启用「分批高亮」。
 * 500 行是需求里给的阈值：再多的代码一次性高亮会让页面卡住几百毫秒，
 * 学生看到的就是"拖进去以后页面不动了"。
 */
const LAZY_HIGHLIGHT_THRESHOLD = 500;

/** 每批高亮多少行（一次处理太多同样会卡，太小又会让批次数变多） */
const LAZY_HIGHLIGHT_CHUNK = 300;

/**
 * 当前正在进行的分批高亮任务。
 * 用它做"取消"：学生连着拖几个文件时，前一个文件还没高亮完，
 * 如果不取消就会往**新代码**里写入**旧代码**的高亮片段（页面直接花掉）。
 */
let highlightJob = null;

/** 取消未完成的分批高亮（切换文件、清空代码时都要调） */
function cancelPendingHighlight() {
  if (highlightJob) {
    highlightJob.cancelled = true;
    highlightJob = null;
  }
}

/** 让出一帧/空闲时段执行任务：优先用 requestIdleCallback，不支持则退回 setTimeout */
function runWhenIdle(callback) {
  if (typeof window.requestIdleCallback === 'function') {
    window.requestIdleCallback(callback, { timeout: 300 });
  } else {
    // Safari 老版本没有 requestIdleCallback；用 setTimeout 也能达到"不阻塞界面"的效果
    window.setTimeout(callback, 16);
  }
}

/**
 * 把代码渲染到指定的 <code> 元素里，并做语法高亮。
 *
 * 三种情况：
 *   1. 高亮库没加载 / 代码很短（≤ 500 行）→ 一次性高亮；
 *   2. 代码很长（> 500 行）→ **先只高亮前 500 行**，其余部分立刻以纯文本显示
 *      （内容完整可见、可复制，绝不是"留白等加载"），
 *      再用 requestIdleCallback 分批把剩余部分逐块换成高亮 HTML；
 *   3. 高亮过程中抛错 → 保住已经渲染出来的纯文本，不白屏。
 *
 * 为什么要分批而不是"干脆不高亮"：几千行的文件一次性高亮会让主线程卡住，
 * 分批之后每一批都很短，界面始终能响应滚动和点击（需求四第 2 条）。
 *
 * @param {HTMLElement} target 目标 <code> 元素
 * @param {string} code 代码文本
 * @param {string} language 语言标识
 */
function renderCode(target, code, language) {
  cancelPendingHighlight();                    // 先取消上一份代码没做完的任务
  target.className = '';                       // 清掉上一次的高亮 class
  target.textContent = code;                   // 先放纯文本（同时完成转义）

  if (typeof hljs === 'undefined') return;     // 高亮库没加载成功，直接用纯文本

  const lines = code.split('\n');
  const lang = hljsLang(language);

  // ---- 小文件：一次性高亮，跟以前一样 ----
  if (lines.length <= LAZY_HIGHLIGHT_THRESHOLD) {
    try {
      target.innerHTML = hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
      target.className = 'hljs';
    } catch (err) {
      // 语言不被支持等情况：保持纯文本即可，不打扰学生
      console.warn('语法高亮失败，已降级为纯文本：', err);
    }
    return;
  }

  // ---- 大文件：分批高亮 ----
  renderLargeCode(target, lines, lang);
}

/**
 * 大文件的分批高亮。
 *
 * 结构：把内容切成若干块，每块用一个 <span> 承载，块与块之间插一个换行文本节点
 *      （这样行号/换行结构完全不变，复制出来还是原来的代码）。
 *      第一块（前 500 行）立刻高亮；其余块先设为纯文本，随后在空闲时段逐块升级。
 *
 * @param {HTMLElement} target 目标 <code> 元素
 * @param {string[]} lines     代码按行切好的数组
 * @param {string} lang        highlight.js 的语言名
 */
function renderLargeCode(target, lines, lang) {
  const job = { cancelled: false };
  highlightJob = job;

  // 第一块的结束位置：不是死板的 500 行，而是要避开"跨行的块注释/三引号字符串"，
  // 否则切在半截注释里，后面那半会被当成代码着色，出现一片莫名其妙的颜色。
  const firstEnd = safeSplitPoint(lines, 0, LAZY_HIGHLIGHT_THRESHOLD, lang);

  const chunkTexts = [];
  const chunkNodes = [];

  const headText = lines.slice(0, firstEnd).join('\n');
  chunkTexts.push(headText);
  const headNode = document.createElement('span');
  chunkNodes.push(headNode);

  for (let start = firstEnd; start < lines.length; start += LAZY_HIGHLIGHT_CHUNK) {
    const end = safeSplitPoint(lines, start, LAZY_HIGHLIGHT_CHUNK, lang);
    chunkTexts.push(lines.slice(start, end).join('\n'));
    chunkNodes.push(document.createElement('span'));
  }

  // ---- 第一块**立刻**高亮 ----
  // 需求说的是"先只高亮前 500 行"：所以首块必须当场出效果，
  // 学生拖进来马上就看到一块正常着色的代码，而不是"整页都是黑白、等一会儿才变色"。
  try {
    headNode.innerHTML = hljs.highlight(headText, { language: lang, ignoreIllegals: true }).value;
  } catch (err) {
    console.warn('首块高亮失败，该块保持纯文本：', err);
    headNode.textContent = headText;
  }

  // ---- 其余块先按纯文本铺好 ----
  // 内容是完整的：能看、能滚动、能复制，只是暂时没有颜色。
  for (let index = 1; index < chunkNodes.length; index += 1) {
    chunkNodes[index].textContent = chunkTexts[index];
  }

  target.textContent = '';
  chunkNodes.forEach((node, index) => {
    if (index > 0) target.appendChild(document.createTextNode('\n'));
    target.appendChild(node);
  });
  target.className = 'hljs';

  if (chunkNodes.length > 1) {
    toast(`文件较大（${lines.length} 行），已先显示全部内容，语法高亮将分批完成`, 'info');
  }

  /** 从第二块开始，逐个在空闲时段高亮；每做完一块就检查有没有被取消 */
  let index = 1;
  const step = () => {
    if (job.cancelled) return;                 // 学生已经换了别的文件，停止写入
    if (index >= chunkNodes.length) {
      highlightJob = null;
      return;
    }
    const node = chunkNodes[index];
    const text = chunkTexts[index];
    try {
      node.innerHTML = hljs.highlight(text, { language: lang, ignoreIllegals: true }).value;
    } catch (err) {
      // 某一块失败就保持纯文本，不影响其它块
      console.warn('分批高亮某一块失败，该块保持纯文本：', err);
    }
    index += 1;
    runWhenIdle(step);                          // 让出主线程，继续下一块
  };
  if (index < chunkNodes.length) runWhenIdle(step);
}

/**
 * 找一个"安全的分块边界"：尽量切在 size 行处，但不要切在跨行的注释/字符串中间。
 *
 * 为什么需要它：高亮是按块独立计算的，如果一块正好从 `/* 注释` 的中间开始，
 * 高亮器不知道上一块还没结束，就会把注释内容当成代码着色，屏幕上出现一片
 * 莫名其妙的颜色（这是"分块高亮"最典型的副作用）。
 *
 * 做法很朴素：从期望位置往后找，直到这一段文本里的块注释与三引号字符串都成对闭合。
 * 最多多找 200 行就放弃（真遇到几千行没闭合的注释，那代码本身也有问题）。
 *
 * @param {string[]} lines 全部代码行
 * @param {number} start   本块起始行下标
 * @param {number} size    期望的块大小（行数）
 * @param {string} lang    highlight.js 语言名
 * @returns {number} 本块的结束行下标（不含）
 */
function safeSplitPoint(lines, start, size, lang) {
  const desired = Math.min(start + size, lines.length);
  if (desired >= lines.length) return lines.length;

  // Python 用三引号字符串，C/Java 用 /* */；两种都要检查
  const checkBlockComment = lang === 'c' || lang === 'java';
  const checkTripleQuote = lang === 'python';

  for (let end = desired; end < Math.min(desired + 200, lines.length); end += 1) {
    const text = lines.slice(start, end).join('\n');

    if (checkBlockComment) {
      // 去掉 // 行注释后再数 /* 与 */ 的个数，成对才说明没切在注释里
      const withoutLineComments = text.replace(/\/\/[^\n]*/g, '');
      const opened = (withoutLineComments.match(/\/\*/g) || []).length;
      const closed = (withoutLineComments.match(/\*\//g) || []).length;
      if (opened !== closed) continue;
    }
    if (checkTripleQuote) {
      const doubleQuote = (text.match(/"""/g) || []).length;
      const singleQuote = (text.match(/'''/g) || []).length;
      if (doubleQuote % 2 !== 0 || singleQuote % 2 !== 0) continue;
    }
    return end;
  }
  return desired;    // 找不到安全点就按原计划切，保证不会死循环
}

/* ==========================================================================
   3.5 代码在线编辑
   --------------------------------------------------------------------------
   为什么要有它：学生看完 AI 的检测结论，最自然的动作就是"那我改一下"。
   如果只能改本机文件再重新拖进来，中间要多走三四步，思路早断了。

   设计要点：
     · 编辑器就是一个 <textarea>，语法高亮在保存/退出编辑后照常生效；
       没有引入 CodeMirror 之类的重型编辑器——学生机可能没网，
       而且原生输入法的中文行为最稳。
     · 编辑过程中 `state.code` 实时跟着走，所以三个 AI 按钮可以直接
       分析"你正在写的这一版"，不必先保存。
     · 「保存」的落点分两种，且**必须对学生说清楚**：
         1) 项目库里的文件 → PUT /library/file 真的写回磁盘；
         2) 拖进来/粘贴的代码 → 只更新预览，不写任何文件（提示里讲明白）。
   ========================================================================== */

/** 把「有没有未保存的改动」反映到界面上的小标记 */
function setDirty(dirty) {
  state.dirty = Boolean(dirty);
  el.codeDirtyBadge.classList.toggle('hidden', !state.editing || !state.dirty);
}

/**
 * 进入编辑模式。
 * 编辑器的初始内容取 `state.code`（也就是当前预览的这一份）。
 */
function startEdit() {
  if (!state.hasCode) {
    toast('先放一段代码进来再编辑', 'info');
    return;
  }
  if (state.libraryTarget && !state.allowWrite) {
    // 正常情况下按钮已经禁用了，这里是双保险（例如键盘直接触发）
    toast('项目库当前是只读模式，不能修改文件', 'info');
    return;
  }

  state.savedCode = state.code;
  el.codeEditor.value = state.code;
  state.editing = true;
  setDirty(false);
  el.codeDropzone.classList.add('is-editing');
  applyCodeVisibility();
  syncEditorGutter();

  if (typeof el.codeEditor.focus === 'function') el.codeEditor.focus();
}

/**
 * 退出编辑模式并还原成"载入时的那一份"。
 * 有未保存改动时先确认，避免手滑把写了一半的代码扔掉。
 */
async function cancelEdit() {
  if (!state.editing) return;

  if (state.dirty) {
    const ok = await showConfirm({
      title: '放弃这些修改？',
      text: '这段修改还没有保存，放弃之后就找不回来了。',
      okText: '放弃修改',
      danger: true,
    });
    if (!ok) return;
  }

  el.codeEditor.value = state.savedCode;
  state.code = state.savedCode;
  state.editing = false;
  setDirty(false);
  el.codeDropzone.classList.remove('is-editing');

  // 回到高亮的只读视图
  renderCode(el.codeContent, state.code, state.language);
  applyCodeVisibility();
  toast('已放弃修改', 'info');
}

/**
 * 保存编辑内容。
 *
 * 两种落点（见本节的说明），失败时**保持编辑模式**——
 * 学生写了半天的东西不能因为一次网络错误就没了。
 */
async function saveEdit() {
  if (!state.editing) return;

  const code = el.codeEditor.value;
  state.code = code;

  // ---- 情况 1：这是项目库里的文件，真的写回磁盘 ----
  if (state.libraryTarget && state.allowWrite) {
    const payload = {
      path: state.libraryTarget.relPath,
      root: state.libraryTarget.root || null,
      code,
    };
    // 带上传入打开时的指纹：文件若被别处改过，后端会返回 409 拦下来
    if (state.libraryTarget.sha256) payload.expected_sha256 = state.libraryTarget.sha256;

    el.codeSaveBtn.disabled = true;
    try {
      const data = await api('/file', {
        method: 'PUT',
        body: JSON.stringify(payload),
        headers: { 'Content-Type': 'application/json' },
      }, LIBRARY_BASE);

      // 更新乐观锁基准：下一次保存要跟"这次写进去的内容"比
      state.libraryTarget.sha256 = data.sha256 || '';
      state.savedCode = code;
      state.editing = false;
      setDirty(false);
      el.codeDropzone.classList.remove('is-editing');

      renderCode(el.codeContent, code, state.language);
      applyCodeVisibility();

      // 原文件不是 UTF-8（例如 Windows 记事本存的 GBK）时如实说明：
      // 保存会统一写成 UTF-8，中文不会乱，但别的程序看到的编码变了。
      const sourceEncoding = state.libraryTarget.encoding || 'utf-8';
      const encodingNote = sourceEncoding.startsWith('utf-8')
        ? ''
        : `（原文件是 ${sourceEncoding} 编码，已按 UTF-8 保存，中文不会丢）`;
      toast((data.message || '已保存') + encodingNote, 'ok');

      await refreshAfterWrite();
      return;
    } catch (err) {
      // 409 的提示语本身已经把"怎么办"说清楚了（重新打开文件再改），原样透出
      toast(err.message, 'error');
      return;    // 保持编辑模式，别把内容弄丢
    } finally {
      el.codeSaveBtn.disabled = false;
    }
  }

  // ---- 情况 2：只是预览的代码（拖进来 / 粘贴），没有对应文件 ----
  state.savedCode = code;
  state.editing = false;
  setDirty(false);
  el.codeDropzone.classList.remove('is-editing');
  renderCode(el.codeContent, code, state.language);
  applyCodeVisibility();
  toast('已更新预览（这段代码不在项目库里，没有写进任何文件）', 'info');

  // 行数/函数个数这些要跟着变，重新分析一次
  try {
    await analyzeCurrent();
    renderCode(el.codeContent, code, state.language);
  } catch (err) {
    toast(err.message, 'error');
  }
}

/** 保存/删除后统一刷新：项目库列表 + 分类结果卡片 */
async function refreshAfterWrite() {
  await loadLibrary();
  try {
    await analyzeCurrent();
    renderCode(el.codeContent, state.code, state.language);
  } catch (err) {
    console.warn('保存后重新分析失败：', err);
  }
}

/**
 * 同步左侧行号槽。
 *
 * 行号值的意义：AI 的报告里说的是"第 8 行有问题"，
 * 编辑时能看到行号才能立刻对上（否则还要自己数）。
 * 只在行数变化时重建文本，滚动时只平移，避免每次按键都拼一遍大字符串。
 */
function syncEditorGutter() {
  const lines = el.codeEditor.value.split('\n').length;
  if (el.editorGutter.dataset.lines !== String(lines)) {
    el.editorGutter.dataset.lines = String(lines);
    let text = '';
    for (let index = 1; index <= lines; index += 1) text += index + '\n';
    el.editorGutter.textContent = text;
  }
  el.editorGutter.style.transform = `translateY(${-el.codeEditor.scrollTop}px)`;
}

/** 在光标处插入文本（Tab 缩进、回车自动缩进都用它） */
function insertAtEditorCursor(text) {
  const area = el.codeEditor;
  // 桩环境/极端情况下 selectionStart 可能是 undefined，退化成"追加到末尾"
  const start = area.selectionStart ?? area.value.length;
  const end = area.selectionEnd ?? start;
  area.value = area.value.slice(0, start) + text + area.value.slice(end);
  area.selectionStart = start + text.length;
  area.selectionEnd = start + text.length;
  onEditorInput();
}

/** 编辑器内容变化：同步到 state.code，并更新"未保存"标记 */
function onEditorInput() {
  state.code = el.codeEditor.value;
  setDirty(state.code !== state.savedCode);
  syncEditorGutter();
}

/**
 * 编辑器键盘快捷键。
 * 这三条是写代码时的肌肉记忆，浏览器默认行为都很碍事，所以必须拦下来：
 *   Tab   → 插入 4 个空格（默认行为是焦点跳走，写代码时完全没法用）
 *   Ctrl+S→ 保存（默认行为是"保存整个网页"）
 *   Esc   → 放弃修改
 *   回车  → 自动补上当前行的缩进
 */
function onEditorKeydown(event) {
  const area = el.codeEditor;

  if (event.key === 'Tab') {
    event.preventDefault();
    insertAtEditorCursor('    ');
    return;
  }

  if ((event.ctrlKey || event.metaKey) && String(event.key).toLowerCase() === 's') {
    event.preventDefault();
    saveEdit();
    return;
  }

  if (event.key === 'Escape') {
    event.preventDefault();
    cancelEdit();
    return;
  }

  if (event.key === 'Enter') {
    const start = area.selectionStart ?? area.value.length;
    const before = area.value.slice(0, start);
    const lineStart = before.lastIndexOf('\n') + 1;
    const indent = (before.slice(lineStart).match(/^[ \t]*/) || [''])[0];
    if (indent) {
      // 只在有缩进时才接管：否则会破坏浏览器自己的换行行为
      event.preventDefault();
      insertAtEditorCursor('\n' + indent);
    }
  }
}

/** 绑定编辑器的输入与快捷键 */
function bindEditor() {
  el.codeEditor.addEventListener('input', onEditorInput);
  el.codeEditor.addEventListener('keydown', onEditorKeydown);
  // 滚动时行号要跟着走，否则滚两屏以后行号就完全对不上了
  el.codeEditor.addEventListener('scroll', syncEditorGutter);
  el.codeEditBtn.addEventListener('click', startEdit);
  el.codeSaveBtn.addEventListener('click', saveEdit);
  el.codeCancelBtn.addEventListener('click', cancelEdit);
}

/* ==========================================================================
   4. 上传与分析
   ========================================================================== */

/**
 * 把选中的代码送到后端分析（识别语言 + 解析结构）。
 * 这一步不调用大模型，所以即使没配置 AI 也能用。
 */
async function analyzeCurrent() {
  const data = await api('/analyze', { method: 'POST', body: codeFormData() });

  state.language = data.language;
  state.languageLabel = data.language_label;

  // ---- 更新代码区标题与语言徽章 ----
  el.codeTitle.textContent = data.filename;
  el.codeLangBadge.textContent = data.language_label;
  el.codeLangBadge.hidden = false;

  // ---- 渲染分类结果卡片 ----
  el.classifyList.innerHTML = [
    ['文件', escapeHtml(data.filename)],
    ['语言', `${escapeHtml(data.language_label)}（${escapeHtml(data.language)}）`],
    ['行数', data.line_count],
    ['字符', data.char_count],
    ['函数 / 类', data.symbols.length],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  el.classifyCard.hidden = false;

  // ---- 渲染代码结构清单 ----
  if (data.symbols.length) {
    el.structureList.innerHTML = data.symbols.map((s) => `
      <li>
        <code>${escapeHtml(s.qualified_name)}</code>
        <span>${escapeHtml(s.kind)} · L${s.start_line}-${s.end_line}</span>
      </li>`).join('');
    el.structureBox.hidden = false;
  } else {
    el.structureBox.hidden = true;
  }

  // ---- 把后端的提示（例如语法错误、语言不支持）转达给学生 ----
  (data.notes || []).forEach((note) => toast(note, 'info'));
  if (data.parse_error) {
    toast('代码有语法错误，仍可使用 AI 检测', 'error');
  }

  return data;
}

/**
 * 处理一个代码文件/文本：展示 → 分析 → 刷新历史。
 *
 * 拖拽（三个区域）、点击选择文件、粘贴、点项目库条目，最终都走这里，
 * 因此"占位提示消失"只需要在这一处调用 setHasCode(true) 就够了
 * ——这正是把它做成统一状态的好处。
 *
 * @param {string} filename 文件名（决定语言识别）
 * @param {string} code     代码内容
 * @param {object} [options]
 * @param {{relPath: string, root: string, sha256?: string}} [options.library]
 *        这段代码来自项目库里的哪个文件。传了它，「保存」才会真的写回磁盘；
 *        不传就是"只在预览里"的代码（拖进来的文件、粘贴的代码）。
 */
async function loadCode(filename, code, options = {}) {
  // 换文件前先确认：当前若有没保存的修改，不能默默丢掉
  if (!(await confirmDiscardChanges())) return;

  state.filename = filename;
  state.code = code;
  state.savedCode = code;
  state.libraryTarget = options.library || null;

  // 新载入的代码一律从只读视图开始，编辑模式要重新点「编辑」
  state.editing = false;
  setDirty(false);
  el.codeDropzone.classList.remove('is-editing');
  el.codeEditor.value = '';

  // 先立刻把代码显示出来，不用等后端返回，学生感觉更快。
  // setHasCode(true) 会同时：隐藏占位提示、显示代码块、启用三个 AI 按钮。
  setHasCode(true);
  el.codeTitle.textContent = filename;
  renderCode(el.codeContent, code, '');   // 语言未知，先不高亮（也顺手清掉上一次的高亮）

  try {
    const data = await analyzeCurrent();
    // 语言识别出来了，重新高亮一次
    renderCode(el.codeContent, code, state.language);
    toast(`已识别为 ${data.language_label}`, 'ok');
  } catch (err) {
    toast(err.message, 'error');
  }
  await loadHistory();
}

/* ==========================================================================
   5. 三个 AI 功能
   ========================================================================== */

/** 切换结果区显示哪一块 */
function showResult(which) {
  el.resultEmpty.hidden = true;
  el.resultCheck.hidden = which !== 'check';
  el.resultComment.hidden = which !== 'comment';
  el.resultFix.hidden = which !== 'fix';
}

/** ---- 5.1 AI 检测 ---- */
/**
 * AI 调用失败后的状态自检。
 *
 * 后端说"请先在网页上输入 API Key"时，说明**会话过期了**（默认 30 分钟）
 * 或者用户刚在别处清了 Key。这时右上角不能还挂在"AI 已就绪"上骗人，
 * 顺手把它改回"未启用"，学生一看就知道该重新填。
 *
 * @param {string} message 后端返回的错误信息
 */
function syncBadgeAfterAiError(message) {
  if (!state.modelConnected) return;
  if (!String(message || '').includes('请先在网页上输入 API Key')) return;
  state.modelConnected = false;
  state.modelLabel = '';
  state.modelName = '';
  renderAiBadge();
  setModelStatus('Key 已过期或被清除，请重新填写后再试', 'error');
}

/**
 * 统一的 AI 调用失败处理：弹提示 + 必要时复位顶部徽章。
 * @param {Error} err 捕获到的错误
 */
function handleAiError(err) {
  toast(err.message, 'error');
  syncBadgeAfterAiError(err.message);
}

async function runCheck() {
  setBusy(true, 'AI 正在检查代码…');
  showResult('check');

  try {
    const data = await api('/check', { method: 'POST', body: codeFormData() });

    // 评分圆环：用 CSS 变量控制 conic-gradient 的百分比
    el.scoreCircle.style.setProperty('--pct', data.score);
    el.scoreValue.textContent = Math.round(data.score);
    el.scoreLevel.textContent = data.level;
    el.scoreSummary.textContent = data.summary || '（模型没有给出总体评价）';

    // 问题列表：按严重程度着色，并显示行号与修改建议
    if (data.issues.length) {
      el.issueList.innerHTML = data.issues.map((issue) => `
        <li class="issue issue--${escapeHtml(issue.severity)}">
          <div class="issue__head">
            <span class="issue__title">${escapeHtml(issue.title)}</span>
            ${issue.line ? `<span class="issue__line">第 ${issue.line} 行</span>` : ''}
          </div>
          ${issue.detail ? `<p class="issue__detail">${nl2br(issue.detail)}</p>` : ''}
          ${issue.suggestion ? `<pre class="issue__suggestion">${escapeHtml(issue.suggestion)}</pre>` : ''}
        </li>`).join('');
    } else {
      el.issueList.innerHTML =
        '<li class="issue issue--info"><div class="issue__head">' +
        '<span class="issue__title">没有发现明显问题 🎉</span></div></li>';
    }

    // 亮点
    if ((data.highlights || []).length) {
      el.highlightList.innerHTML = data.highlights.map((h) => `<li>${escapeHtml(h)}</li>`).join('');
      el.highlightBox.hidden = false;
    } else {
      el.highlightBox.hidden = true;
    }

    toast(`检测完成，得分 ${Math.round(data.score)} 分`, 'ok');
  } catch (err) {
    handleAiError(err);
    el.scoreSummary.textContent = '检测失败：' + err.message;
  } finally {
    setBusy(false);
    await loadHistory();
  }
}

/** ---- 5.2 生成注释 ---- */
async function runComment() {
  setBusy(true, 'AI 正在写注释…');
  showResult('comment');

  try {
    const data = await api('/comment', { method: 'POST', body: codeFormData() });
    el.commentSummary.textContent = data.summary || '已生成中文注释';
    renderCode(el.commentCode, data.commented_code, state.language);
    toast('注释生成完成', 'ok');
  } catch (err) {
    handleAiError(err);
    el.commentSummary.textContent = '生成失败：' + err.message;
    el.commentCode.textContent = '';
  } finally {
    setBusy(false);
    await loadHistory();
  }
}

/** ---- 5.3 自动改错 ---- */
async function runFix() {
  setBusy(true, 'AI 正在检查并修正代码…');
  showResult('fix');

  try {
    const data = await api('/fix', { method: 'POST', body: codeFormData() });
    el.fixSummary.textContent = data.summary || '';

    if (!data.had_error) {
      // 代码没问题时给出正向反馈，而不是硬凑一堆"修改"
      el.fixNoError.hidden = false;
      el.fixBody.hidden = true;
      toast('代码没有发现错误 👍', 'ok');
    } else {
      el.fixNoError.hidden = true;
      el.fixBody.hidden = false;

      // 逐条修改说明：左右并排显示"改前 / 改后"，直观对比
      el.changeList.innerHTML = data.changes.map((c) => `
        <li class="change">
          <div class="issue__head">
            <span class="issue__title">${escapeHtml(c.reason || '修改')}</span>
            ${c.line ? `<span class="issue__line">第 ${c.line} 行</span>` : ''}
          </div>
          <div class="change__diff">
            <div class="change__side change__side--before">${escapeHtml(c.original) || '（无）'}</div>
            <div class="change__side change__side--after">${escapeHtml(c.fixed) || '（无）'}</div>
          </div>
        </li>`).join('');

      renderCode(el.fixCode, data.fixed_code, state.language);
      toast(`已修正 ${data.changes.length} 处问题`, 'ok');
    }
  } catch (err) {
    handleAiError(err);
    el.fixSummary.textContent = '改错失败：' + err.message;
    el.fixBody.hidden = true;
  } finally {
    setBusy(false);
    await loadHistory();
  }
}

/* ==========================================================================
   6. 本地项目库
   --------------------------------------------------------------------------
   面向同学的真实痛点：一学期下来 .c / .java / .py 作业散在各个文件夹里，
   自己都想不起来放哪儿了。这一块负责：
     - 扫描后端配置好的文件夹（后端只允许扫它自己配置的目录，前端无法指定任意路径）
     - 按语言自动分类，生成筛选按钮
     - 点一个文件就把它读进中间代码区，接着就能用三个 AI 功能
     - 对单个文件做「替换 / 删除」：都要二次确认，删除默认进 .trash 可找回
   ========================================================================== */

/** 把字节数转成好读的写法（学生看得懂 KB / MB 就行） */
function formatSize(bytes) {
  if (bytes == null) return '—';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

/** 渲染语言筛选按钮：全部 + 各语言（带文件数） */
function renderLibraryChips(counts, labels, total) {
  const items = [['', '全部', total]].concat(
    Object.keys(counts).map((name) => [name, labels[name] || name, counts[name]])
  );

  el.libraryChips.innerHTML = items.map(([value, label, count]) => `
    <button class="chip${value === state.libraryFilter ? ' is-active' : ''}"
            data-lang="${escapeHtml(value)}">
      ${escapeHtml(label)} <span class="chip__count">${count}</span>
    </button>`).join('');
}

/** 渲染文件列表 */
function renderLibraryFiles(files) {
  if (!files.length) {
    el.libraryList.innerHTML =
      '<li class="files__empty">这个文件夹里还没有找到代码文件</li>';
    return;
  }

  el.libraryList.innerHTML = files.map((file) => {
    // 行数未知（文件过大）时不编数字，老实显示提示
    const meta = file.too_large
      ? `${formatSize(file.size_bytes)} · 文件过大，未统计行数`
      : `${file.line_count} 行 · ${formatSize(file.size_bytes)}`;

    // 只读模式（LIBRARY__ALLOW_WRITE=false）下干脆不渲染这两个按钮，
    // 而不是渲染出来再禁用——点了没反应比"根本没有这个入口"更让人困惑。
    const actions = state.allowWrite
      ? `<div class="files__actions">
           <button class="btn btn--mini" type="button" data-act="replace"
                   title="用本机上的另一个文件覆盖它">替换</button>
           <button class="btn btn--mini btn--danger" type="button" data-act="delete"
                   title="删除（默认移到 ${escapeHtml(state.trashDirName)} 目录，可找回）">删除</button>
         </div>`
      : '';

    return `
      <li class="files__item" data-path="${escapeHtml(file.rel_path)}"
          data-root="${escapeHtml(file.root)}">
        <div class="files__row">
          <span class="files__name" title="${escapeHtml(file.rel_path)}">
            ${escapeHtml(file.filename)}
          </span>
          <span class="badge badge--accent">${escapeHtml(file.language_label)}</span>
        </div>
        <div class="files__meta">${escapeHtml(meta)}</div>
        ${actions}
      </li>`;
  }).join('');
}

/**
 * 渲染「项目库文件夹」这一行（常显）。
 *
 * 为什么常显：这是使用中被问得最多的问题——"我的代码到底存在哪个文件夹？"。
 * 之前只在"一个文件都没扫到"时才提示路径，结果有文件的时候反而看不到关键信息。
 *
 * @param {Array<{name: string, path: string, exists: boolean}>} roots /scan 返回的根目录
 */
function renderLibraryPath(roots) {
  state.libraryRoots = roots || [];

  if (!state.libraryRoots.length) {
    el.libraryPathValue.textContent = '后端还没有配置项目库目录';
    el.libraryPathValue.title = '后端还没有配置项目库目录';
    el.libraryPathValue.classList.add('is-missing');
    return;
  }

  // 默认只配了一个目录，直接把完整路径显示出来；
  // 配了多个时显示第一个 + "等 N 个"，完整清单放在 title 里，避免把左侧栏撑爆。
  const first = state.libraryRoots[0];
  const extra = state.libraryRoots.length > 1 ? ` 等 ${state.libraryRoots.length} 个文件夹` : '';
  el.libraryPathValue.textContent = first.path + extra;
  el.libraryPathValue.title = state.libraryRoots
    .map((root) => root.path + (root.exists ? '' : '（目录不存在）'))
    .join('\n');

  // 目录不存在时用醒目的样式标出来：学生据此就知道该去建那个文件夹
  el.libraryPathValue.classList.toggle('is-missing', !first.exists);
}

/**
 * 复制项目库文件夹路径到剪贴板。
 * 学生拿到路径后可以直接粘到文件资源管理器的地址栏，把作业拖进那个文件夹。
 */
async function copyLibraryPath() {
  const first = state.libraryRoots[0];
  const text = first ? first.path : '';

  if (!text) {
    toast('还没读到项目库路径，先点右上角 ↻ 重新扫描', 'info');
    return;
  }

  try {
    // 剪贴板 API 只在安全上下文可用；本地 127.0.0.1 属于安全上下文，正常能用。
    if (typeof navigator === 'undefined' || !navigator.clipboard) {
      throw new Error('当前浏览器不支持剪贴板 API');
    }
    await navigator.clipboard.writeText(text);
    toast('路径已复制，可粘到文件资源管理器地址栏打开', 'ok');
  } catch (err) {
    // 复制失败不影响主流程：路径本身就显示在旁边，学生手动选中复制即可
    console.warn('复制项目库路径失败：', err);
    toast('复制失败，请手动选中路径复制', 'info');
  }
}

/**
 * 扫描本地项目库并渲染。
 * @param {string} language 语言筛选，'' 表示全部
 */
async function loadLibrary(language = state.libraryFilter) {
  state.libraryFilter = language;
  el.libraryHint.textContent = '正在扫描本地代码…';

  try {
    const query = language ? `?language=${encodeURIComponent(language)}` : '';
    const data = await api('/scan' + query, {}, LIBRARY_BASE);

    state.libraryFiles = data.files;
    // 后端是否允许写入：决定列表里显不显示「替换 / 删除」，以及「编辑」能不能点
    state.allowWrite = data.allow_write !== false;
    if (data.trash_dir_name) state.trashDirName = data.trash_dir_name;

    renderLibraryChips(data.language_counts, data.language_labels, data.total_files);
    renderLibraryFiles(data.files);
    renderLibraryReadOnlyHint();
    refreshEditButtons();

    // 常显的文件夹路径：无论扫到几个文件都要告诉学生代码放在哪
    const roots = data.roots || [];
    renderLibraryPath(roots);

    // 提示文字告诉学生代码是从哪个文件夹扫出来的，方便ta们把新作业放进去
    const missing = roots.filter((root) => !root.exists);
    if (data.total_files) {
      el.libraryHint.textContent =
        `共 ${data.total_files} 个文件 · ${data.total_lines} 行` +
        (roots.length ? ` · 来自 ${roots.map((r) => r.name).join('、')}` : '');
    } else if (language) {
      // 筛没了：不是文件夹空，而是这个语言下没有文件
      el.libraryHint.textContent = '这个语言下暂时没有文件，点「全部」看看其它的';
    } else {
      // 路径已经常显在上面那一行里了，这里只要指个方向，不必再重复一遍长路径
      el.libraryHint.textContent =
        '还没有找到代码文件。把你写的 .c / .java / .py 放进上面那个文件夹，再点右上角 ↻';
    }

    // 有些配置的文件夹不存在时明确提示，而不是让列表莫名其妙为空
    if (missing.length) {
      toast(`有 ${missing.length} 个项目库目录不存在：${missing[0].path}`, 'info');
    }
    if (data.truncated) {
      el.libraryMore.textContent = `文件太多，只列出了前 ${data.files.length} 个`;
      el.libraryMore.hidden = false;
    } else {
      el.libraryMore.hidden = true;
    }
  } catch (err) {
    el.libraryHint.textContent = '项目库不可用';
    el.libraryList.innerHTML =
      `<li class="files__empty">读取失败：${escapeHtml(err.message)}</li>`;
  }
}

/**
 * 打开项目库里的一个文件：读内容 -> 载入代码区 -> 自动分析。
 *
 * 注意把 rel_path / sha256 一起交给 loadCode：前者告诉「保存」写回哪个文件，
 * 后者是乐观锁基准——别人（或学生自己在记事本里）改过这个文件时，
 * 保存会被后端用 409 拦下，而不是默默覆盖。
 *
 * @param {string} relPath 相对项目库根目录的路径
 * @param {string} root    根目录标识
 */
async function openLibraryFile(relPath, root) {
  el.libraryHint.textContent = `正在读取 ${relPath} …`;
  try {
    const params = new URLSearchParams({ path: relPath });
    if (root) params.append('root', root);
    const data = await api('/file?' + params.toString(), {}, LIBRARY_BASE);

    // 用相对路径作为文件名，历史记录里就能看出这份作业来自哪个文件夹
    await loadCode(data.rel_path, data.code, {
      library: {
        relPath: data.rel_path,
        root: root || '',
        sha256: data.sha256 || '',
        // 记下原编码：保存时统一写成 UTF-8，如果原来是 GBK 要如实告诉学生
        encoding: data.encoding || 'utf-8',
      },
    });

    if (data.replaced) {
      toast('这个文件里有些字符无法识别，已用替代字符显示', 'info');
    }
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    await loadLibrary();
  }
}

/* --------------------------------------------------------------------------
   6.3 替换 / 删除
   --------------------------------------------------------------------------
   两个操作都会改动学生磁盘上的真实文件，所以都守两条规矩：
     1. 先弹确认框，把**具体是哪个文件**写清楚再让学生点确定；
     2. 结果如实反馈——替换成没成、删除是进了回收站还是彻底删掉，
        以及"去哪能把文件找回来"。
   -------------------------------------------------------------------------- */

/** 项目库只读时的说明条（只读就明确写出来，别让学生以为是按钮坏了） */
function renderLibraryReadOnlyHint() {
  const hint = document.getElementById('libraryReadOnlyHint');
  if (!hint) return;
  hint.classList.toggle('hidden', state.allowWrite);
  hint.textContent = state.allowWrite
    ? ''
    : '项目库当前是只读模式（LIBRARY__ALLOW_WRITE=false）：'
      + '可以浏览、可以用 AI 分析，但不能编辑 / 替换 / 删除文件。';
}

/** 从列表项里取出对应的文件对象（按钮上不会带全部字段，回查一次最稳） */
function findLibraryFile(relPath) {
  return state.libraryFiles.find((item) => item.rel_path === relPath) || null;
}

/**
 * 点「替换」：记下目标文件并唤起本机文件选择框。
 *
 * 替换的语义是"用本机另一个文件的内容，覆盖项目库里这个文件"，
 * 文件名与位置都不变——所以下面要校验扩展名一致，避免出现
 * `linked_list.c` 里装着 Python 代码这种让语言识别错乱的情况。
 */
function requestReplace(target) {
  if (!state.allowWrite) {
    toast('项目库当前是只读模式，不能替换文件', 'info');
    return;
  }
  state.replaceTarget = target;
  // 清空 value：连续两次选同一个文件时，不清空就不会再触发 change 事件
  el.replaceInput.value = '';
  el.replaceInput.click();
}

/**
 * 学生选完本机文件后的处理：校验 → 确认 → PUT 覆盖。
 * @param {HTMLInputElement} input 隐藏的文件选择框
 */
async function performReplace(input) {
  const chosen = input.files && input.files[0];
  const target = state.replaceTarget;
  state.replaceTarget = null;          // 用完就清，避免下次误用到旧目标
  if (!chosen || !target) return;

  const targetExt = fileExtension(target.filename || target.rel_path);
  const chosenExt = fileExtension(chosen.name);
  if (targetExt !== chosenExt) {
    toast(
      `替换要选同类型文件：${target.rel_path} 需要 ${targetExt || '同扩展名'} 文件，`
      + `你选的是 ${chosenExt || '无扩展名'} 文件；新增文件请拖到上面的虚线框里`,
      'error',
    );
    return;
  }

  let text = '';
  try {
    text = await chosen.text();
  } catch (err) {
    toast('读取本机文件失败：' + err.message, 'error');
    return;
  }

  const ok = await showConfirm({
    title: '替换这个文件？',
    text: `原文件：<code>${escapeHtml(target.rel_path)}</code><br>`
      + `新内容来自：<code>${escapeHtml(chosen.name)}</code>（${formatSize(chosen.size)}）<br>`
      + '替换后原内容会被覆盖，且无法从回收站找回。',
    okText: '替换',
    danger: true,
  });
  if (!ok) return;

  const payload = { path: target.rel_path, root: target.root || null, code: text };
  // 如果这个文件正好是当前打开的那一个，带上乐观锁基准，防止盖掉别处的改动
  if (state.libraryTarget && state.libraryTarget.relPath === target.rel_path
      && state.libraryTarget.sha256) {
    payload.expected_sha256 = state.libraryTarget.sha256;
  }

  try {
    const data = await api('/file', {
      method: 'PUT',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    }, LIBRARY_BASE);
    toast(`已用 ${chosen.name} 替换 ${data.rel_path}`, 'ok');
    await loadLibrary();

    // 替换的正是当前预览的文件 → 重新读一遍，否则屏幕上还是旧内容
    if (state.libraryTarget && state.libraryTarget.relPath === target.rel_path) {
      await openLibraryFile(target.rel_path, target.root);
    }
  } catch (err) {
    toast('替换失败：' + err.message, 'error');
  }
}

/**
 * 点「删除」：二次确认 → DELETE。
 *
 * 默认是"移进回收站"而不是真删，所以确认文案里要讲清楚两件事：
 * 文件去哪了、怎么找回来。彻底删除需要学生自己去 .trash 里删，
 * 页面上不提供——少一个"一键永久删除"的入口，就少一批误删的作业。
 */
async function requestDelete(target) {
  if (!state.allowWrite) {
    toast('项目库当前是只读模式，不能删除文件', 'info');
    return;
  }

  const ok = await showConfirm({
    title: '删除这个文件？',
    text: `要删除的是：<code>${escapeHtml(target.rel_path)}</code>（${formatSize(target.size_bytes)}）<br>`
      + `删除后会移到项目库文件夹里的 <code>${escapeHtml(state.trashDirName)}</code> 目录，`
      + '需要时还能自己找回来。',
    okText: '删除',
    danger: true,
  });
  if (!ok) return;

  const params = new URLSearchParams({ path: target.rel_path });
  if (target.root) params.append('root', target.root);

  try {
    const data = await api('/file?' + params.toString(), { method: 'DELETE' }, LIBRARY_BASE);
    toast(data.message || '已删除', 'ok');

    // 删掉的正好是当前预览的文件 → 顺手清空代码区，
    // 否则屏幕上还留着一个已经不存在的文件，再点保存就会报错
    if (state.libraryTarget && state.libraryTarget.relPath === target.rel_path) {
      clearCode({ silent: true });
    }
    await loadLibrary();
  } catch (err) {
    toast('删除失败：' + err.message, 'error');
  }
}

/**
 * 项目库列表的点击总入口（事件委托）。
 *
 * 顺序很重要：**先判断点的是不是「替换 / 删除」按钮**，是就执行动作并返回；
 * 否则才当成"打开这个文件"。反过来的话，点删除会先弹出一遍文件内容。
 */
function onLibraryListClick(event) {
  const actionBtn = event.target.closest('[data-act]');
  if (actionBtn) {
    const row = actionBtn.closest('.files__item');
    if (!row) return;
    const relPath = row.dataset.path;
    const target = findLibraryFile(relPath) || {
      rel_path: relPath,
      root: row.dataset.root || '',
      filename: relPath,
      size_bytes: 0,
    };
    if (actionBtn.dataset.act === 'replace') requestReplace(target);
    else if (actionBtn.dataset.act === 'delete') requestDelete(target);
    return;
  }

  const item = event.target.closest('.files__item');
  if (item) openLibraryFile(item.dataset.path, item.dataset.root);
}

/** 绑定「替换」文件选择框与确认弹窗（弹窗是全局的，只绑一次） */
function bindFileActions() {
  el.replaceInput.addEventListener('change', () => performReplace(el.replaceInput));

  el.confirmOkBtn.addEventListener('click', () => closeConfirm(true));
  el.confirmCancelBtn.addEventListener('click', () => closeConfirm(false));
  // 点遮罩也算取消：这是所有弹窗的通用直觉
  el.confirmModal.addEventListener('click', (event) => {
    if (event.target === el.confirmModal) closeConfirm(false);
  });
  // Esc 关闭（编辑器里的 Esc 由编辑器自己处理，两者不会打架：
  // 弹窗打开时焦点在弹窗里，编辑器收不到这个事件）
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !el.confirmModal.classList.contains('hidden')) {
      event.preventDefault();
      closeConfirm(false);
    }
  });
}

/* ==========================================================================
   7. 历史记录
   ========================================================================== */

/** 操作类型 -> 中文名，用于历史列表上的小标签 */
const ACTION_LABEL = { analyze: '分析', check: '检测', comment: '注释', fix: '改错' };

/** 从后端拉取历史记录并渲染 */
async function loadHistory() {
  try {
    const data = await api('/history?limit=20');
    state.history = data.items;

    if (!data.items.length) {
      el.historyList.innerHTML = '<li class="history__empty">还没有记录，先上传一段代码试试</li>';
      return;
    }

    el.historyList.innerHTML = data.items.map((item) => {
      // 评分只有"检测"操作才有，其它操作显示操作类型
      const extra = item.score != null
        ? `${Math.round(item.score)} 分`
        : ACTION_LABEL[item.action] || item.action;
      const time = new Date(item.created_at).toLocaleString('zh-CN', {
        month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
      });
      return `
        <li class="history__item" data-id="${item.id}">
          <div class="history__row">
            <span class="history__name">${escapeHtml(item.filename)}</span>
            <span class="badge badge--accent">${extra}</span>
          </div>
          <div class="history__meta">${escapeHtml(item.language)} · ${time}</div>
        </li>`;
    }).join('');
  } catch (err) {
    el.historyList.innerHTML =
      `<li class="history__empty">读取失败：${escapeHtml(err.message)}</li>`;
  }
}

/** 点击历史记录：把当时的原始代码重新载入编辑区 */
async function openHistoryItem(recordId) {
  try {
    const detail = await api(`/history/${recordId}`);
    await loadCode(detail.filename, detail.code);
    toast('已载入历史记录中的代码', 'ok');
  } catch (err) {
    toast(err.message, 'error');
  }
}

/* ==========================================================================
   11. 页面初始化与事件绑定
   ========================================================================== */

/** 查询后端状态：AI 是否可用、支持哪些语言 */
async function loadStatus() {
  try {
    const data = await api('/status');
    state.aiAvailable = data.ai_available;

    // 语言支持徽章
    const langs = Object.values(data.supported_languages || {});
    el.langBadge.textContent = langs.length ? `支持 ${langs.join(' / ')}` : '语言支持未知';

    // AI 状态徽章：走统一的 renderAiBadge()，它会考虑"用户是否在页面上连了 Key"
    state.backendModel = data.model || '';
    renderAiBadge();

    if (aiReady()) {
      setBusy(false);
    } else {
      // 后端没配 Key 也不算"没法用"：页面上填一把就能连上（见模型设置卡片）
      el.btnCheck.disabled = true;
      el.btnComment.disabled = true;
      el.btnFix.disabled = true;
      el.actionHint.textContent = '后端未配置大模型；可在左侧「模型设置」里填自己的 API Key';
      toast('还没配置模型：可在左侧「模型设置」里选择模型并填入 API Key', 'info');
    }
  } catch (err) {
    el.aiBadge.textContent = '后端未连接';
    el.aiBadge.className = 'badge badge--off';
    el.langBadge.textContent = '服务不可用';
    toast('连接后端失败：' + err.message, 'error');
  }
}

/* ==========================================================================
   8. 面板尺寸拖动（需求三）
   --------------------------------------------------------------------------
   学生看代码时经常需要"左边收窄一点、代码区大一点"，固定三栏很难受。
   这里用原生 JS 实现三根分隔条：

     垂直分隔条 ①  左栏 ｜ 代码区      → 改 --sidebar-w
     垂直分隔条 ②  代码区 ｜ 结果区    → 改 --result-w
     水平分隔条    代码区 ｜ 按钮区    → 改 --code-height

   最小宽度限制（需求指定，防止拖到看不见内容）：
     左侧栏 ≥ 200px ｜ 中间代码区 ≥ 400px ｜ 右侧结果区 ≥ 280px

   实现要点：
     · 宽度写进 CSS 变量，栅格用 var() 引用，JS 只改变量，不拼栅格字符串；
     · 用 Pointer Events（pointerdown/move/up + setPointerCapture），
       鼠标、触摸屏、触控笔一套代码全兼容，比 mousedown/mousemove 省心；
     · 拖动过程给 body 加类，靠 CSS 把整页鼠标样式锁成 col-resize/row-resize，
       否则鼠标移到别的元素上会变回箭头，看起来像"拖断了"；
     · 尺寸记到 localStorage，刷新后还是学生习惯的样子。
   ========================================================================== */

/** 各面板的最小宽度/高度（需求三第 4 条明确指定的数值） */
const PANEL_MIN = {
  sidebar: 200,   // 左侧栏最小宽度
  main: 400,      // 中间代码区最小宽度
  result: 280,    // 右侧结果区最小宽度
  code: 120,      // 代码区最小高度（需求没规定，取一个还能看到几行代码的值）
};

/** 分隔条占位宽度，必须与 CSS 里的 --resizer 保持一致（用于算可用空间） */
const RESIZER_SIZE = 8;

/** 单侧最多占布局宽度的比例：防止把一边拉到满、另一边只剩最小宽度 */
const PANEL_MAX_RATIO = 0.6;

/** 面板尺寸的本地存储键（刷新后沿用） */
const LAYOUT_STORAGE_KEY = 'ai-tutor.layout';

/**
 * 把期望宽度夹到「最小值 ~ 可用上限」之间（纯函数，方便单测）。
 *
 * @param {string} kind         'sidebar' 或 'result'
 * @param {number} desired      用户拖到的位置算出来的期望宽度
 * @param {number} layoutWidth  布局总宽度
 * @param {number} otherWidth   另一侧面板当前宽度
 * @returns {number} 夹好的宽度（整数像素）
 *
 * 上限怎么算：总宽减去两根分隔条、减去"另一侧"、再减去中间栏的最小宽度 400，
 * 剩下的才是这一侧能用的最大值。这样无论怎么拖，中间栏都不会被挤到 400 以下。
 * 两种上限取较小值：一个是"按比例"，一个是"给中间栏留够"。
 */
function clampPanelWidth(kind, desired, layoutWidth, otherWidth) {
  const min = PANEL_MIN[kind];
  const available = layoutWidth - RESIZER_SIZE * 2;

  const byRatio = available * PANEL_MAX_RATIO;
  const byMiddle = available - PANEL_MIN.main - otherWidth;
  const max = Math.max(min, Math.min(byRatio, byMiddle));

  return Math.round(Math.min(Math.max(desired, min), max));
}

/**
 * 把期望高度夹到「最小高度 ~ 卡片里剩下的空间」之间（纯函数）。
 *
 * @param {number} desired       期望高度
 * @param {number} cardHeight    代码卡片的总高度
 * @param {number} actionsHeight 底部按钮区高度
 * @returns {number} 夹好的高度
 */
function clampCodeHeight(desired, cardHeight, actionsHeight) {
  // 留出：标题行、按钮区、分隔条本身、一点呼吸空间
  const reserved = actionsHeight + 70;
  const max = Math.max(PANEL_MIN.code, cardHeight - reserved);
  return Math.round(Math.min(Math.max(desired, PANEL_MIN.code), max));
}

/** 读回上次保存的面板尺寸（没有或格式不对就用 CSS 里的默认值） */
function loadSavedLayout() {
  try {
    const raw = localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (!raw) return null;
    const saved = JSON.parse(raw);
    return saved && typeof saved === 'object' ? saved : null;
  } catch (err) {
    // localStorage 可能被禁用（隐私模式），读不到就用默认值，不影响使用
    console.warn('读取面板尺寸失败，使用默认布局：', err);
    return null;
  }
}

/** 保存面板尺寸（失败也不影响使用） */
function saveLayout() {
  try {
    localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(state.layout));
  } catch (err) {
    console.warn('保存面板尺寸失败：', err);
  }
}

/**
 * 应用某个面板的宽度：写 CSS 变量 + 记到 state。
 *
 * @param {string} kind   'sidebar' 或 'result'
 * @param {number} width  像素宽度
 */
function applyPanelWidth(kind, width) {
  state.layout[kind] = width;
  const variable = kind === 'sidebar' ? '--sidebar-w' : '--result-w';
  // 写在 :root 上，栅格里的 var() 会立刻生效（不用重排 JS 里的样式表）
  document.documentElement.style.setProperty(variable, `${width}px`);
}

/** 应用代码区高度：写 CSS 变量并标记"已经手动调过高度了" */
function applyCodeHeight(height) {
  state.layout.code = height;
  document.documentElement.style.setProperty('--code-height', `${height}px`);
  el.codeDropzone.classList.add('is-height-fixed');
}

/**
 * 绑定一根分隔条的拖动（三根共用这一份逻辑）。
 *
 * @param {HTMLElement} handle 分隔条元素
 * @param {string} axis        'x' = 左右拖动；'y' = 上下拖动
 * @param {string} kind        'sidebar' / 'result' / 'code'
 */
function bindResizer(handle, axis, kind) {
  if (!handle) return;   // 窄屏时元素可能不存在，容错处理

  handle.addEventListener('pointerdown', (event) => {
    event.preventDefault();          // 防止拖动时选中文字
    handle.setPointerCapture(event.pointerId);   // 鼠标移出分隔条也能继续收到事件

    const layoutRect = document.querySelector('.layout').getBoundingClientRect();
    const startPos = axis === 'x' ? event.clientX : event.clientY;

    // 记录拖动开始时的大小与参照物尺寸，用"位移量"而不是"绝对坐标"来算新尺寸，
    // 这样在窗口滚动、布局变化时也不会跳
    const startWidth = axis === 'x'
      ? (kind === 'sidebar'
          ? el.sidebar.getBoundingClientRect().width
          : el.resultPane.getBoundingClientRect().width)
      : el.codeDropzone.getBoundingClientRect().height;

    const startCardHeight = el.codeCard.getBoundingClientRect().height;
    const actionsHeight = el.codeActions ? el.codeActions.getBoundingClientRect().height : 0;
    const otherWidth = axis === 'x'
      ? (kind === 'sidebar'
          ? el.resultPane.getBoundingClientRect().width
          : el.sidebar.getBoundingClientRect().width)
      : 0;

    document.body.classList.add(axis === 'x' ? 'is-resizing-col' : 'is-resizing-row');
    handle.classList.add('is-dragging');

    /**
     * 拖动方向：+1 = 面板随鼠标同向变宽，-1 = 反向（鼠标往右拖，面板变窄）。
     *
     * 两根垂直分隔条的方向**天生相反**，这是最容易写错的地方：
     *   · 左侧栏在分隔条**左边**：分隔条往右拖 → 左栏变宽（同向，+1）
     *   · 结果区在分隔条**右边**：分隔条往右拖 → 右栏变窄（反向，-1），
     *     因为它是被"挤"掉的那一边。写成同向的话，拖起来就会觉得
     *     "面板根本不跟着鼠标走"，甚至越拖越反。
     * 水平分隔条同理：代码区在分隔条上方，往下拖 → 变高（同向，+1）。
     */
    const direction = kind === 'result' ? -1 : 1;

    /** 拖动过程中：按位移算出新尺寸并夹到合法范围 */
    const onMove = (moveEvent) => {
      const current = axis === 'x' ? moveEvent.clientX : moveEvent.clientY;
      const delta = (current - startPos) * direction;

      if (axis === 'x') {
        const desired = startWidth + delta;
        applyPanelWidth(kind, clampPanelWidth(kind, desired, layoutRect.width, otherWidth));
      } else {
        const desired = startWidth + delta;
        applyCodeHeight(clampCodeHeight(desired, startCardHeight, actionsHeight));
      }
    };

    /** 松开：解除捕获、恢复鼠标样式、把尺寸存起来 */
    const onUp = (upEvent) => {
      handle.releasePointerCapture?.(upEvent.pointerId);
      handle.removeEventListener('pointermove', onMove);
      handle.removeEventListener('pointerup', onUp);
      handle.removeEventListener('pointercancel', onUp);
      document.body.classList.remove('is-resizing-col', 'is-resizing-row');
      handle.classList.remove('is-dragging');
      saveLayout();
    };

    handle.addEventListener('pointermove', onMove);
    handle.addEventListener('pointerup', onUp);
    handle.addEventListener('pointercancel', onUp);
  });

  // 键盘也能调：聚焦分隔条后按方向键，每次 20px（无障碍 + 触控板不好微调时用）
  handle.addEventListener('keydown', (event) => {
    const step = event.shiftKey ? 60 : 20;
    const layoutRect = document.querySelector('.layout').getBoundingClientRect();

    if (axis === 'x') {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      const current = kind === 'sidebar'
        ? el.sidebar.getBoundingClientRect().width
        : el.resultPane.getBoundingClientRect().width;
      const other = kind === 'sidebar'
        ? el.resultPane.getBoundingClientRect().width
        : el.sidebar.getBoundingClientRect().width;
      // 右侧栏的方向是反的：按 ← 应该是"变宽"（把分隔条往左推）
      const direction = kind === 'sidebar' ? (event.key === 'ArrowRight' ? 1 : -1)
                                           : (event.key === 'ArrowLeft' ? 1 : -1);
      applyPanelWidth(kind, clampPanelWidth(kind, current + step * direction, layoutRect.width, other));
      saveLayout();
    } else {
      if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
      event.preventDefault();
      const current = el.codeDropzone.getBoundingClientRect().height;
      const direction = event.key === 'ArrowDown' ? 1 : -1;
      applyCodeHeight(clampCodeHeight(
        current + step * direction,
        el.codeCard.getBoundingClientRect().height,
        el.codeActions ? el.codeActions.getBoundingClientRect().height : 0,
      ));
      saveLayout();
    }
  });

  // 双击分隔条 = 恢复默认尺寸（拖歪了不用手动量）
  handle.addEventListener('dblclick', () => {
    if (kind === 'code') {
      el.codeDropzone.classList.remove('is-height-fixed');
      document.documentElement.style.removeProperty('--code-height');
      delete state.layout.code;
    } else {
      const fallback = kind === 'sidebar' ? 300 : 380;
      applyPanelWidth(kind, fallback);
    }
    saveLayout();
    toast('已恢复默认尺寸', 'ok');
  });
}

/** 绑定三根分隔条，并恢复上次保存的尺寸 */
function bindResizers() {
  bindResizer(el.resizerLeft, 'x', 'sidebar');
  bindResizer(el.resizerRight, 'x', 'result');
  bindResizer(el.resizerCode, 'y', 'code');

  const saved = loadSavedLayout();
  if (!saved) return;
  if (typeof saved.sidebar === 'number') applyPanelWidth('sidebar', saved.sidebar);
  if (typeof saved.result === 'number') applyPanelWidth('result', saved.result);
  // 代码区高度只在用户真的调过之后才恢复，否则保持自适应的默认行为
  if (typeof saved.code === 'number') applyCodeHeight(saved.code);
}

/* ==========================================================================
   9. 屏幕很窄时自动收起分隔条
   --------------------------------------------------------------------------
   CSS 已经在 1280px / 860px 两个断点把分隔条隐藏了，但 JS 里的
   "另一侧宽度"之类的计算仍然会用到它们。窗口缩小到分隔条不可见时，
   把用户拖出来的固定宽度清掉，让布局回到自适应，避免出现
   "左栏 500px + 无法拖动"这种卡死的状态。
   ========================================================================== */
function watchViewportForLayout() {
  const narrow = window.matchMedia('(max-width: 1280px)');

  /** 窄屏时清除固定宽度，宽屏恢复用户设置 */
  const sync = (isNarrow) => {
    if (isNarrow) {
      document.documentElement.style.removeProperty('--sidebar-w');
      document.documentElement.style.removeProperty('--result-w');
    } else {
      if (typeof state.layout.sidebar === 'number') applyPanelWidth('sidebar', state.layout.sidebar);
      if (typeof state.layout.result === 'number') applyPanelWidth('result', state.layout.result);
    }
  };

  sync(narrow.matches);
  narrow.addEventListener('change', (event) => sync(event.matches));
}

/* ==========================================================================
   10. 三个拖拽区（需求一 + 需求二）
   --------------------------------------------------------------------------
   页面上有三个能接收文件的地方，语义**刻意不同**，必须让学生分得清：

   ┌────────────────────────┬──────────────────────────────────────────────┐
   │ 区域                   │ 拖进去会发生什么                              │
   ├────────────────────────┼──────────────────────────────────────────────┤
   │ A ①「选择你的代码」      │ 只预览 + 分析（不入库）。相当于"打开看看"       │
   │ B ②「本地项目库」        │ **入库**：上传到服务器 → 按语言归档 → 更新列表  │
   │ C ③「代码预览区」        │ 只预览 + 分析（不入库）。相当于"拖进来读一读"   │
   └────────────────────────┴──────────────────────────────────────────────┘

   A 与 C 行为一样、只是位置不同——学生在中间看代码时顺手拖进去最自然，
   不用绕回左上角；两者的区别只在于"哪个区域高亮"。

   为什么不能拖哪儿都入库：
   入库会**真的在服务器上创建文件**、还会按语言搬进 c/ java/ python/ 三个目录。
   如果预览区也入库，学生只是想看一眼别人的代码，项目库里就多出一份，
   列表很快就乱了；要交作业必须明确拖到 B 区。这是需求二第 2 条的意图。
   ========================================================================== */

/**
 * 给一个元素绑定"拖拽接收"行为（三个区域共用）。
 *
 * @param {HTMLElement} target      接收拖拽的元素
 * @param {Function}    onFileDrop  松开鼠标时的处理函数，参数是 File 对象
 *
 * 关键细节：
 *   · dragenter / dragover 必须 preventDefault()，否则浏览器会**直接打开这个文件**，
 *     页面就被替换掉了（这是新手最常踩的坑）；
 *   · dragleave 在拖到子元素上时也会触发，所以高亮状态用"进/出都重算"的方式，
 *     简单加个类就够用，不必做复杂的计数。
 */
function bindDropTarget(target, onFileDrop) {
  ['dragenter', 'dragover'].forEach((type) => {
    target.addEventListener(type, (event) => {
      event.preventDefault();
      target.classList.add('is-over');
    });
  });

  ['dragleave', 'drop'].forEach((type) => {
    target.addEventListener(type, (event) => {
      event.preventDefault();
      target.classList.remove('is-over');
    });
  });

  target.addEventListener('drop', async (event) => {
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) await onFileDrop(file);
  });
}

/** 绑定「① 选择你的代码」上传区：点击选择文件 + 拖拽放入（只预览） */
function bindDropzone() {
  // 点击整个区域 -> 触发隐藏的文件输入框
  el.dropzone.addEventListener('click', () => el.fileInput.click());

  el.fileInput.addEventListener('change', async (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    // 交给后端读文件内容：用 /analyze 的上传能力，前端不做编码猜测
    await uploadFile(file);
    el.fileInput.value = '';   // 清空，方便重复选同一个文件
  });

  // 拖拽 A 区：只预览，不入库
  bindDropTarget(el.dropzone, uploadFile);
}

/** 绑定「③ 代码预览区」：拖进来只预览，**不触发入库**（需求二第 2 条） */
function bindCodeDropzone() {
  bindDropTarget(el.codeDropzone, uploadFile);
}

/** 把文件内容读成文本并交给 loadCode 处理（只预览，不写服务器） */
async function uploadFile(file) {
  try {
    const text = await file.text();
    if (!text.trim()) {
      toast('这个文件是空的', 'error');
      return;
    }
    // loadCode 内部会调用 setHasCode(true)，占位提示随之消失
    await loadCode(file.name, text);
  } catch (err) {
    toast('读取文件失败：' + err.message, 'error');
  }
}

/* ---------- 需求二：拖进「② 本地项目库」= 入库 ---------- */

/** 允许入库的扩展名（与后端保持一致；前端先拦一道，提示更即时） */
const ARCHIVABLE_SUFFIXES = ['.c', '.h', '.java', '.py'];

/**
 * 判断文件名的扩展名是否允许入库。
 * @param {string} name 文件名
 * @returns {boolean}
 */
function isArchivable(name) {
  const lowered = String(name || '').toLowerCase();
  return ARCHIVABLE_SUFFIXES.some((suffix) => lowered.endsWith(suffix));
}

/**
 * 把文件加入本地项目库（需求二第 1 条）。
 *
 * 完整流程（每一步失败都有明确提示，不让学生猜）：
 *   1. 前端先做两道快速检查：扩展名是否支持、项目库里是否已有同名文件；
 *   2. POST /files/upload 把文件存到服务器的代码文件夹
 *      —— 后端还会按「同名 + 内容相同」再判一次重复，重复时返回 409；
 *   3. POST /files/scan 让后端按语言归档（.py → python/、.c/.h → c/、.java → java/），
 *      并把文件名/语言/路径/大小/入库时间写进 SQLite 台账；
 *   4. 刷新左侧项目库列表（GET /files/list）与语言筛选按钮；
 *   5. 同时把代码显示到中间预览区，并更新「③ 自动分类结果」。
 *
 * @param {File} file 拖入的文件
 */
async function addToLibrary(file) {
  // ---- 1a. 扩展名检查：不支持的文件早点说，不要白白上传一遍 ----
  if (!isArchivable(file.name)) {
    toast(`只支持 ${ARCHIVABLE_SUFFIXES.join(' / ')} 文件，${file.name} 不在范围内`, 'error');
    return;
  }

  // ---- 1b. 前端去重：项目库里已经有同名文件就先提示 ----
  //（真正的判重以后端为准——同名但内容不同的文件后端会另存为新名字，
  //  所以这里只做"同名就提醒一下"，不直接拦截）
  const sameName = state.libraryFiles.find((item) => item.filename === file.name);

  try {
    // ---- 2. 上传到服务器 ----
    const form = new FormData();
    form.append('file', file, file.name);
    const uploaded = await api('/upload', { method: 'POST', body: form }, FILES_BASE);

    // ---- 3. 按语言归档 + 写台账 ----
    const scanned = await api('/scan', { method: 'POST', body: JSON.stringify({}) ,
      headers: { 'Content-Type': 'application/json' } }, FILES_BASE);

    // 提示语分层：有重名 → 说清"已存在/已改名"；否则报归档结果
    if (uploaded.renamed) {
      toast(`项目库里已有同名文件，已另存为 ${uploaded.filename}`, 'info');
    } else if (sameName) {
      toast(`文件已存在：${sameName.rel_path}（已是最新）`, 'info');
    } else {
      toast(`${uploaded.filename} 已加入项目库，本次共归档 ${scanned.moved} 个文件`, 'ok');
    }

    // ---- 4. 刷新项目库列表 ----
    await loadLibrary();

    // ---- 5. 同时显示代码并更新「自动分类结果」----
    // 顺带把"这份代码对应项目库里的哪个文件"记下来：这样学生拖进来之后
    // 直接点「编辑」改两下再「保存」，就能写回磁盘，中间不用再点一次列表。
    const text = await file.text();
    if (text.trim()) {
      await loadCode(uploaded.filename, text, {
        library: libraryTargetForArchive(scanned, uploaded),
      });
    }
  } catch (err) {
    // 后端对"文件已存在"返回 409，detail 是「文件已存在：<路径>」
    if (String(err.message).includes('文件已存在')) {
      toast(err.message, 'info');
      return;
    }
    toast('加入项目库失败：' + err.message, 'error');
  }
}

/**
 * 归档完成后，找出这份文件在**项目库列表**里的位置，作为可保存的目标。
 *
 * 为什么要绕这一下：上传落在 `CLASSIFIER__ROOT` 下，而项目库列表来自
 * `LIBRARY__ROOTS`。默认两者是同一个目录（见 config.py 的说明），
 * 但配置成不同目录时，归档路径就不在项目库里，这时**不能**声称可以保存——
 * 否则学生点保存只会拿到"文件不存在"的报错。
 *
 * @returns {{relPath: string, root: string, sha256: string}|null}
 */
function libraryTargetForArchive(scanned, uploaded) {
  const archived = (scanned.files || []).find(
    (item) => item.filename === uploaded.filename || item.target_path === uploaded.rel_path
  );
  const relPath = archived ? archived.target_path : uploaded.rel_path;
  const inLibrary = state.libraryFiles.some((item) => item.rel_path === relPath);
  if (!inLibrary) return null;
  const rootName = (state.libraryRoots[0] && state.libraryRoots[0].name) || '';
  return { relPath, root: rootName, sha256: '' };
}

/** 绑定「② 本地项目库」的拖拽区（拖进来 = 入库） */
function bindLibraryDropzone() {
  bindDropTarget(el.libraryDropzone, addToLibrary);
}

/** 绑定所有按钮 */
function bindActions() {
  el.btnCheck.addEventListener('click', runCheck);
  el.btnComment.addEventListener('click', runComment);
  el.btnFix.addEventListener('click', runFix);
  el.refreshHistory.addEventListener('click', loadHistory);

  // ---- 本地项目库 ----
  // 重新扫描：学生把新作业拖进文件夹后点一下就刷新了
  el.refreshLibrary.addEventListener('click', () => loadLibrary());

  // 复制项目库文件夹路径：粘到文件资源管理器地址栏就能直接打开那个文件夹
  el.copyLibraryPath.addEventListener('click', copyLibraryPath);

  // 语言筛选按钮是动态生成的，用事件委托绑定
  el.libraryChips.addEventListener('click', (event) => {
    const chip = event.target.closest('.chip');
    if (!chip) return;
    // 再点一次已选中的按钮就取消筛选，回到全部
    const next = chip.dataset.lang === state.libraryFilter ? '' : chip.dataset.lang;
    loadLibrary(next);
  });

  // 文件列表用事件委托：列表项是动态生成的，逐个绑定会随刷新丢失。
  // 点击的总入口在 onLibraryListClick 里——它会先判断点的是不是「替换 / 删除」。
  el.libraryList.addEventListener('click', onLibraryListClick);

  // 粘贴代码（左侧折叠区）：直接用文本调用分析流程
  el.pasteBtn.addEventListener('click', async () => {
    const code = el.pasteCode.value;
    const name = (el.pasteName.value || 'untitled.py').trim();
    if (!code.trim()) {
      toast('请先粘贴代码', 'error');
      return;
    }
    await loadCode(name, code);
  });

  // ---- 清空代码：占位提示会重新出现（需求一第 4 条）----
  el.clearCodeBtn.addEventListener('click', clearCode);

  // 历史列表用事件委托：列表项是动态生成的，逐个绑定会随刷新丢失
  el.historyList.addEventListener('click', (event) => {
    const item = event.target.closest('.history__item');
    if (item) openHistoryItem(item.dataset.id);
  });
}

/**
 * 绑定「粘贴到中间区域」（需求一第 3 条的第三种触发方式）。
 *
 * 场景：学生从别处复制了一段代码，鼠标停在中间代码区直接 Ctrl+V 就想看效果。
 * 之前必须先展开左侧的折叠区、粘进 textarea、再点按钮，多三步。
 *
 * 两个必须处理的边界：
 *   1. 焦点在输入框/文本域里时**不要抢**——那是学生在往粘贴框里输入，
 *      抢过来会让他输不进去（判断 event.target.tagName）；
 *   2. 只在剪贴板里是**文本**时处理，复制文件时的 files 不归这里管。
 */
function bindPasteToCodePane() {
  document.addEventListener('paste', async (event) => {
    const tag = (event.target && event.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;   // 别抢输入框里的粘贴

    const text = event.clipboardData && event.clipboardData.getData('text');
    if (!text || !text.trim()) return;

    event.preventDefault();
    // 粘贴的内容没有文件名，给个默认名让后端能识别语言（默认按 Python 处理）
    const name = guessNameFromPaste(text);
    await loadCode(name, text);
    toast(`已载入粘贴的代码（${name}）`, 'ok');
  });
}

/**
 * 猜一个文件名给粘贴进来的代码用。
 *
 * 为什么要猜：后端靠**文件后缀**识别语言，粘贴的内容没有文件名，
 * 全都叫 untitled.py 的话，一段 Java 代码会被当成 Python 解析，结果全错。
 * 这里用几个非常明显的特征做判断（够用且不会误判），
 * 认不出来就按 Python 处理——初学者用得最多。
 *
 * @param {string} text 粘贴的代码
 * @returns {string} 文件名
 */
function guessNameFromPaste(text) {
  const head = text.slice(0, 2000);
  if (/#include\s*<|int\s+main\s*\(/.test(head)) return 'pasted.c';
  if (/public\s+(class|static)|System\.out\.print|import\s+java\./.test(head)) {
    return 'Pasted.java';
  }
  if (/^\s*(def|class)\s+\w+|print\(|import\s+\w+$/m.test(head)) return 'pasted.py';
  return 'pasted.py';
}

/** 页面入口 */
async function init() {
  // ---- 三根分隔条：拖动改面板尺寸（需求三）----
  bindResizers();
  watchViewportForLayout();

  // ---- 三个拖拽区各绑各的（语义不同，见「10. 三个拖拽区」的说明）----
  bindDropzone();          // A：① 选择你的代码 —— 只预览
  bindLibraryDropzone();   // B：② 本地项目库   —— 入库 + 归档 + 刷新列表
  bindCodeDropzone();      // C：③ 代码预览区   —— 只预览
  bindPasteToCodePane();   // 直接往中间区域粘贴代码

  // ---- 在线编辑 + 替换/删除（都要二次确认，见第 3.5 与 6.3 节）----
  bindEditor();
  bindFileActions();

  // ---- 模型设置（选模型 + 填自己的 API Key）----
  bindModelSettings();

  bindActions();

  // 初始状态：还没有代码 → 占位提示显示、代码块隐藏、三个 AI 按钮禁用。
  // 这一步不能省：页面刷新后要让 JS 的状态与 HTML 的初始样子对齐，
  // 否则可能出现"看起来有代码但其实 state 是空的"这种不一致。
  setHasCode(false);

  // 并行请求：状态、历史、项目库、模型清单互不依赖，一起发更快。
  // 项目库放在最后且不 await 失败——它不可用也不该影响整页启动。
  await Promise.all([loadStatus(), loadHistory(), loadLibrary(''), loadModels()]);
}

document.addEventListener('DOMContentLoaded', init);
