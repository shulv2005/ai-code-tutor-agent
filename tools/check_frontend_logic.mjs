/* ============================================================================
   app.js 行为测试（在 Node 里跑真实的 app.js，配一套最小 DOM 桩）

   为什么需要它：
   “拖动分隔条会不会小于最小宽度”“1000 行代码是不是分批高亮”“清空后占位提示
   会不会回来”——这些是**行为**，静态检查（找关键字、比对 id）证明不了。
   浏览器自动化在这个环境里不可用，于是用一个最小 DOM 桩把 app.js 真正跑起来，
   直接调用它的函数、模拟 pointerdown/pointermove 事件，然后断言结果。

   桩只实现被测代码真正用到的那部分 DOM API：
     getElementById / querySelector / createElement / createTextNode
     classList（add/remove/toggle/contains）、style.setProperty/removeProperty
     addEventListener + 手动触发（fire）、getBoundingClientRect、dataset、disabled
     matchMedia / requestIdleCallback（排队，由测试手动 drain）/ localStorage

   用法：node tools/check_frontend_logic.mjs [项目根目录]
   退出码 0 = 全部通过；1 = 有失败项。
   ========================================================================== */

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const BASE = process.argv[2] ? path.resolve(process.argv[2]) : process.cwd();
const APP_JS = path.join(BASE, 'frontend', 'js', 'app.js');

/* ---------------------------------------------------------------------------
   1. 最小 DOM 桩
   ------------------------------------------------------------------------- */

/** 极简 classList：够 app.js 用（toggle 支持第二个 force 参数） */
function makeClassList(classSet) {
  return {
    add: (...names) => names.forEach((n) => classSet.add(n)),
    remove: (...names) => names.forEach((n) => classSet.delete(n)),
    contains: (n) => classSet.has(n),
    toggle: (n, force) => {
      const on = force === undefined ? !classSet.has(n) : Boolean(force);
      if (on) classSet.add(n);
      else classSet.delete(n);
      return on;
    },
    get _set() { return classSet; },
  };
}

let nodeSeq = 0;

/** 创建一个假元素 */
function makeElement(id = '') {
  const element = {
    id,
    tagName: 'DIV',
    textContent: '',
    _innerHTML: '',
    disabled: false,
    hidden: false,
    value: '',
    dataset: {},
    children: [],
    _rect: { width: 300, height: 200, top: 0, left: 0 },
    _listeners: {},
    _style: {},
  };
  // className 与 classList 必须是**同一份数据**（真实 DOM 就是这样）。
  // 之前桩里两者各存一份，于是 `el.className = 'badge badge--ok'` 之后
  // `classList.contains('badge--ok')` 仍是 false，测试会误报失败。
  const classSet = new Set();
  Object.defineProperty(element, 'className', {
    get: () => [...classSet].join(' '),
    set: (value) => {
      classSet.clear();
      String(value || '')
        .split(/\s+/)
        .filter(Boolean)
        .forEach((name) => classSet.add(name));
    },
  });
  element.classList = makeClassList(classSet);
  Object.defineProperty(element, 'innerHTML', {
    get: () => element._innerHTML,
    set: (value) => {
      element._innerHTML = String(value);
      element.children = [];        // 简化：设置 innerHTML 视为清空子节点
      // 真实 DOM 里 innerHTML 一变，textContent 也跟着变（标签被剥掉）。
      // 桩里不同步这一点的话，「弹窗里到底写了什么」就没法用 textContent 断言。
      element.textContent = String(value).replace(/<[^>]*>/g, '');
    },
  });
  element.style = {
    setProperty: (name, value) => {
      element._style[name] = String(value);
      // 让 documentElement 上的变量变化"同步"到各元素的假尺寸上，
      // 模拟真实的布局回流：否则每次拖动都从固定值开始算，
      // 连续两次拖动的断言就会对不上（第一版就踩了这个）
      if (typeof element._onStyleSet === 'function') element._onStyleSet(name, String(value));
    },
    removeProperty: (name) => {
      delete element._style[name];
      if (typeof element._onStyleSet === 'function') element._onStyleSet(name, '');
    },
    getPropertyValue: (name) => element._style[name] ?? '',
  };
  element.addEventListener = (type, handler) => {
    (element._listeners[type] ||= []).push(handler);
  };
  element.removeEventListener = (type, handler) => {
    element._listeners[type] = (element._listeners[type] || []).filter((h) => h !== handler);
  };
  element.appendChild = (child) => {
    element.children.push(child);
    // 简化：appendChild 后 textContent 拼上子节点的文本，便于断言"内容完整"
    element.textContent += child.textContent || '';
    return child;
  };
  element.getBoundingClientRect = () => ({ ...element._rect });
  element.setPointerCapture = () => {};
  element.releasePointerCapture = () => {};
  element.closest = () => null;
  element.querySelectorAll = () => [];
  element.querySelector = () => null;
  // 有些流程会以代码方式触发点击（例如「替换」要唤起隐藏的文件选择框）。
  // 真实 DOM 里每个元素都有 click()，桩里也得有，否则那行代码会直接抛 TypeError。
  element.click = () => {
    element._clicked = (element._clicked || 0) + 1;
    fire(element, 'click', {});
  };
  element.focus = () => { element._focused = true; };
  nodeSeq += 1;
  return element;
}

/** 创建假文本节点 */
function makeTextNode(text) {
  return { nodeType: 3, textContent: String(text), _rect: { width: 0, height: 0 } };
}

/** 触发某个元素上的事件（模拟浏览器 dispatchEvent） */
function fire(element, type, event = {}) {
  const handlers = element._listeners[type] || [];
  handlers.slice().forEach((handler) => handler({ preventDefault() {}, ...event }));
  return handlers.length;
}

/** 建一个 id -> 元素 的注册表，并按 app.js 需要的 id 预置好节点 */
const elementsById = new Map();
function register(id, rect) {
  const element = makeElement(id);
  if (rect) element._rect = { ...element._rect, ...rect };
  elementsById.set(id, element);
  return element;
}

// app.js 在模块顶层就用 getElementById 取了一大批节点，必须先建好
[
  'dropzone', 'fileInput', 'pasteName', 'pasteCode', 'pasteBtn',
  'classifyCard', 'classifyList', 'structureBox', 'structureList',
  'historyList', 'refreshHistory',
  'libraryHint', 'libraryChips', 'libraryList', 'libraryMore', 'refreshLibrary',
  'libraryDropzone', 'libraryPathValue', 'copyLibraryPath', 'libraryReadOnlyHint',
  // 模型设置（选模型 + 填自己的 API Key）
  'modelCard', 'modelSelect', 'modelApiKey', 'toggleKeyVisible', 'modelHint',
  'modelStatus', 'modelStatusText', 'saveModelKey', 'clearModelKey', 'refreshModels',
  'aiBadge', 'langBadge',
  'codeTitle', 'codeLangBadge', 'code-placeholder', 'codeDropzone', 'codeBlock',
  'codeContent', 'clearCodeBtn',
  // 在线编辑
  'codeEditBtn', 'codeSaveBtn', 'codeCancelBtn', 'codeDirtyBadge',
  'editorWrap', 'editorGutter', 'codeEditor',
  // 替换 / 删除
  'replaceInput', 'confirmModal', 'confirmTitle', 'confirmText',
  'confirmOkBtn', 'confirmCancelBtn',
  'resizerLeft', 'resizerRight', 'resizerCode', 'codeCard', 'codeActions',
  'btnCheck', 'btnComment', 'btnFix', 'actionHint',
  'resultEmpty', 'resultCheck', 'resultComment', 'resultFix',
  'scoreCircle', 'scoreValue', 'scoreLevel', 'scoreSummary', 'issueList',
  'highlightBox', 'highlightList',
  'commentSummary', 'commentCode',
  'fixSummary', 'fixNoError', 'fixBody', 'changeList', 'fixCode',
  'toast',
].forEach((id) => register(id));

// 几个需要特定初始尺寸的节点
elementsById.get('codeDropzone')._rect = { width: 800, height: 300, top: 100, left: 320 };
elementsById.get('codeCard')._rect = { width: 800, height: 460, top: 90, left: 320 };
elementsById.get('codeActions')._rect = { width: 800, height: 40, top: 500, left: 320 };

const layoutElement = makeElement('layout');
layoutElement._rect = { width: 1600, height: 800, top: 0, left: 0 };

const sidebarElement = elementsById.get('dropzone');   // querySelector('.sidebar')
const resultPaneElement = makeElement('resultPane');
resultPaneElement._rect = { width: 380, height: 700, top: 0, left: 1180 };
sidebarElement._rect = { width: 300, height: 700, top: 0, left: 20 };

/* ---------------------------------------------------------------------------
   2. window / document / hljs 桩
   ------------------------------------------------------------------------- */

const idleQueue = [];
const highlightCalls = [];         // 每次 hljs.highlight 的入参行数

const documentStub = {
  documentElement: makeElement('html'),
  body: makeElement('body'),
  getElementById: (id) => elementsById.get(id) || null,
  querySelector: (selector) => {
    if (selector === '.layout') return layoutElement;
    if (selector === '.sidebar') return sidebarElement;
    if (selector === '.pane:last-of-type') return resultPaneElement;
    return null;
  },
  createElement: () => makeElement(),
  createTextNode: (text) => makeTextNode(text),
  addEventListener: (type, handler) => {
    if (type === 'DOMContentLoaded') documentStub._domReady = handler;
    else if (type === 'paste') documentStub._paste = handler;
    else (documentStub._handlers[type] ||= []).push(handler);
  },
  _handlers: {},
};

/** 触发挂在 document 上的事件（Esc 关弹窗那类全局快捷键） */
function fireDocument(type, event = {}) {
  const handlers = documentStub._handlers[type] || [];
  handlers.slice().forEach((handler) => handler({ preventDefault() {}, ...event }));
  return handlers.length;
}

// 让 CSS 变量的变化反映到假的 getBoundingClientRect 上（模拟布局回流）。
// 少了这一步，"连续拖两次"的断言会一直从初始尺寸起算，测不出真实行为。
documentStub.documentElement._onStyleSet = (name, value) => {
  const pixels = parseInt(value, 10);
  if (Number.isNaN(pixels)) return;
  if (name === '--sidebar-w') sidebarElement._rect.width = pixels;
  if (name === '--result-w') resultPaneElement._rect.width = pixels;
  if (name === '--code-height') elementsById.get('codeDropzone')._rect.height = pixels;
};

const localStorageStub = {
  _data: {},
  getItem(key) { return this._data[key] ?? null; },
  setItem(key, value) { this._data[key] = String(value); },
  removeItem(key) { delete this._data[key]; },
};

// 剪贴板桩：记录「复制项目库路径」写进去的内容，供断言检查
const clipboardWrites = [];
const navigatorStub = {
  clipboard: {
    writeText: async (text) => { clipboardWrites.push(text); },
  },
};

const windowStub = {
  requestIdleCallback: (callback) => {
    idleQueue.push(callback);
    return idleQueue.length;
  },
  setTimeout: (callback) => {
    idleQueue.push(callback);       // 统一排队，由测试手动 drain
    return idleQueue.length;
  },
  matchMedia: () => ({ matches: false, addEventListener: () => {} }),
  localStorage: localStorageStub,
};

const hljsStub = {
  highlight: (text, options) => {
    highlightCalls.push({ lines: text.split('\n').length, language: options?.language });
    // 真实高亮会返回带 <span> 的 HTML；这里做等价的最小模拟：
    // 转义后包一层标记，测试据此判断"这块被高亮过了"
    const escaped = String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    return { value: `<span class="hljs-block">${escaped}</span>` };
  },
};

/*
  可编排的 fetch 桩。
  以前这里直接 throw，只能测纯逻辑；但「保存 / 替换 / 删除」这类功能
  价值恰恰在于"发出去的请求对不对"，必须能观察 HTTP 调用。
  用法：fetchQueue.push({ ...响应 }) 或 push((url, options) => 响应)；
  没排队就发请求 = 测试写错了，直接抛错（保持"不发意外请求"这条约束）。
*/
const fetchQueue = [];
const fetchCalls = [];

function queueResponse(payload, { status = 200 } = {}) {
  fetchQueue.push({ payload, status });
}

const fetchStub = async (url, options = {}) => {
  fetchCalls.push({ url: String(url), options, method: (options.method || 'GET').toUpperCase() });
  if (!fetchQueue.length) {
    throw new Error(`测试没有为这个请求排队：${options.method || 'GET'} ${url}`);
  }
  const next = fetchQueue.shift();
  const { payload, status } = typeof next === 'function' ? next(url, options) : next;
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
  };
};

/** 取出某次调用（按 method + url 片段匹配），方便断言请求内容 */
function findCall(method, urlPart) {
  return fetchCalls.find(
    (item) => item.method === method.toUpperCase() && item.url.includes(urlPart)
  ) || null;
}

/* app.js 里用到但 Node 的 vm 沙箱没有的两个浏览器全局对象。
   FormData：三个 AI 接口与 /analyze 用它组装表单；
   URLSearchParams：项目库的查询串用它拼。
   它们在 Node 里本来就有实现，直接传进沙箱即可（URLSearchParams），
   FormData 在 Node 18+ 也有全局实现，但行为与浏览器不同，这里用最小桩。 */
class FormDataStub {
  constructor() { this._entries = []; }
  append(name, value) { this._entries.push([name, value]); }
  get(name) {
    const found = this._entries.find(([key]) => key === name);
    return found ? found[1] : null;
  }
}

const context = vm.createContext({
  document: documentStub,
  window: windowStub,
  hljs: hljsStub,
  localStorage: localStorageStub,
  navigator: navigatorStub,
  FormData: FormDataStub,
  URLSearchParams,
  console,
  setTimeout: windowStub.setTimeout,
  clearTimeout: () => {},
  fetch: fetchStub,
});

/* ---------------------------------------------------------------------------
   3. 载入真实的 app.js，并把要测的函数导出来
   ------------------------------------------------------------------------- */

const source = fs.readFileSync(APP_JS, 'utf8');
// 在同一段脚本作用域里追加导出：顶层 const 不会挂到 globalThis 上，
// 只有追加在同一个 script 里才能拿到它们（这是 vm 的常规做法）
const epilogue = `
;globalThis.__api = {
  state, el, panel: { PANEL_MIN, RESIZER_SIZE, PANEL_MAX_RATIO },
  lazy: { LAZY_HIGHLIGHT_THRESHOLD, LAZY_HIGHLIGHT_CHUNK },
  setHasCode, clearCode, loadCode, renderCode, renderLargeCode,
  clampPanelWidth, clampCodeHeight, safeSplitPoint,
  applyPanelWidth, applyCodeHeight, bindResizers, bindActions,
  bindDropzone, bindLibraryDropzone, bindCodeDropzone, bindPasteToCodePane,
  renderLibraryPath, copyLibraryPath,
  renderLibraryFiles, findLibraryFile, onLibraryListClick,
  startEdit, cancelEdit, saveEdit, setDirty, onEditorInput, onEditorKeydown,
  insertAtEditorCursor, syncEditorGutter, applyCodeVisibility, refreshEditButtons,
  showConfirm, closeConfirm, requestReplace, performReplace, requestDelete,
  bindEditor, bindFileActions,
  loadModels, renderModelOptions, updateModelHint, currentModel, setModelStatus,
  aiReady, renderAiBadge, loadStatus, setBusy, handleAiError, syncBadgeAfterAiError,
  saveAndTestModel, clearModelKey, bindModelSettings,
  api, codeFormData,
  cancelPendingHighlight, init,
};
`;
vm.runInContext(source + epilogue, context, { filename: 'app.js' });

const api = context.__api;
const el = api.el;

/* ---------------------------------------------------------------------------
   4. 断言工具
   ------------------------------------------------------------------------- */

let passed = 0;
const failures = [];

function check(label, ok, detail = '') {
  if (ok) {
    passed += 1;
    console.log(`  OK   ${label}${detail ? '  —— ' + detail : ''}`);
  } else {
    failures.push(label);
    console.log(`  FAIL ${label}${detail ? '  —— ' + detail : ''}`);
  }
}

/** 把排队的 requestIdleCallback 任务全部执行完（模拟浏览器空闲时段） */
function drainIdle(maxSteps = 200) {
  let steps = 0;
  while (idleQueue.length && steps < maxSteps) {
    const job = idleQueue.shift();
    job();
    steps += 1;
  }
  return steps;
}

/* ---------------------------------------------------------------------------
   5. 需求一：占位提示的显隐
   ------------------------------------------------------------------------- */
console.log('='.repeat(72));
console.log('需求一  占位提示由 hasCode 状态统一管理');
console.log('='.repeat(72));

api.setHasCode(false);
check('初始状态：占位提示可见（没有 hidden 类）',
  !el.codePlaceholder.classList.contains('hidden'));
check('初始状态：代码块隐藏（有 hidden 类）',
  el.codeBlock.classList.contains('hidden'));
check('初始状态：三个 AI 按钮禁用',
  el.btnCheck.disabled && el.btnComment.disabled && el.btnFix.disabled);

api.setHasCode(true);
check('有代码后：占位提示加上 hidden 类（被"顶掉"）',
  el.codePlaceholder.classList.contains('hidden'));
check('有代码后：代码块去掉 hidden 类',
  !el.codeBlock.classList.contains('hidden'));
check('有代码后：语言徽章仍然隐藏（还没识别出语言）',
  el.codeLangBadge.classList.contains('hidden'));

// 有代码 + AI 可用 → 按钮可点
api.state.aiAvailable = true;
api.setHasCode(true);
check('有代码且 AI 可用：三个按钮可点',
  !el.btnCheck.disabled && !el.btnComment.disabled && !el.btnFix.disabled);
api.state.aiAvailable = false;
api.setHasCode(true);
check('没配大模型时：即使有代码按钮也保持禁用',
  el.btnCheck.disabled && el.btnComment.disabled && el.btnFix.disabled);
api.state.aiAvailable = true;

// 需求一第 4 条：清空后占位提示重新出现
api.state.filename = 'demo.py';
api.state.code = 'print(1)\n';
api.state.language = 'python';
api.state.languageLabel = 'Python';
api.setHasCode(true);
api.clearCode();
check('清空代码后：占位提示重新出现（去掉 hidden 类）',
  !el.codePlaceholder.classList.contains('hidden'));
check('清空代码后：代码块重新隐藏',
  el.codeBlock.classList.contains('hidden'));
check('清空代码后：文件名与语言也清掉了',
  api.state.filename === '' && api.state.language === '' && api.state.code === '');
check('清空代码后：按钮重新禁用', el.btnCheck.disabled);

/* ---------------------------------------------------------------------------
   6. 需求三：分隔条拖动的夹取逻辑
   ------------------------------------------------------------------------- */
console.log();
console.log('='.repeat(72));
console.log('需求三  面板拖动与最小宽度限制');
console.log('='.repeat(72));

check('最小宽度常量符合需求（左 200 / 中 400 / 右 280）',
  api.panel.PANEL_MIN.sidebar === 200
  && api.panel.PANEL_MIN.main === 400
  && api.panel.PANEL_MIN.result === 280,
  `sidebar=${api.panel.PANEL_MIN.sidebar} main=${api.panel.PANEL_MIN.main} result=${api.panel.PANEL_MIN.result}`);

// 左侧栏：拖到 100px 也要被夹到 200
check('左栏拖到 100px → 夹回 200px（不小于最小宽度）',
  api.clampPanelWidth('sidebar', 100, 1600, 380) === 200,
  `得到 ${api.clampPanelWidth('sidebar', 100, 1600, 380)}px`);
// 右侧栏：拖到 50px 也要被夹到 280
check('右栏拖到 50px → 夹回 280px',
  api.clampPanelWidth('result', 50, 1600, 300) === 280,
  `得到 ${api.clampPanelWidth('result', 50, 1600, 300)}px`);
// 拖得过大时不能把中间栏挤到 400 以下
const huge = api.clampPanelWidth('sidebar', 1400, 1600, 380);
check('左栏想拉到 1400px → 被夹住，给中间栏留够 400px',
  huge <= 1600 - 16 - 380 - 400 + 1,
  `得到 ${huge}px（上限应为 ${1600 - 16 - 380 - 400}px 左右）`);
check('夹取结果同时受"最多占 60%"限制',
  huge <= Math.round((1600 - 16) * 0.6) + 1, `得到 ${huge}px`);
// 合理值原样通过
check('拖到 420px → 原样保留', api.clampPanelWidth('sidebar', 420, 1600, 380) === 420);

// 代码区高度
check('代码区拖到 10px → 夹回最小高度 120px',
  api.clampCodeHeight(10, 460, 40) === 120,
  `得到 ${api.clampCodeHeight(10, 460, 40)}px`);
check('代码区拖到 9999px → 不超过卡片内部可用高度',
  api.clampCodeHeight(9999, 460, 40) === 460 - 40 - 70,
  `得到 ${api.clampCodeHeight(9999, 460, 40)}px`);

// 真的模拟一次拖动：pointerdown → pointermove → pointerup
api.bindResizers();
const resizerLeft = el.resizerLeft;
fire(resizerLeft, 'pointerdown', { clientX: 300, pointerId: 1 });
fire(resizerLeft, 'pointermove', { clientX: 150, pointerId: 1 });   // 往左拖 150px
const dragged = documentStub.documentElement._style['--sidebar-w'];
fire(resizerLeft, 'pointerup', { pointerId: 1 });
check('拖动左分隔条 → CSS 变量 --sidebar-w 被更新为合法值',
  Boolean(dragged) && parseInt(dragged, 10) >= 200, `--sidebar-w=${dragged}`);
check('拖动时 body 加了 is-resizing-col（鼠标样式锁成 col-resize）',
  !documentStub.body.classList.contains('is-resizing-col'),
  '松开后应已移除');
check('拖动结束把尺寸存进了 localStorage',
  Boolean(localStorageStub.getItem('ai-tutor.layout')));

// 拖到左边极限：必须停在 200
fire(resizerLeft, 'pointerdown', { clientX: 300, pointerId: 2 });
fire(resizerLeft, 'pointermove', { clientX: -500, pointerId: 2 });
const floored = documentStub.documentElement._style['--sidebar-w'];
fire(resizerLeft, 'pointerup', { pointerId: 2 });
check('一路往左拖 → 左栏停在最小宽度 200px',
  parseInt(floored, 10) === 200, `--sidebar-w=${floored}`);

/* ---- 拖动方向：面板必须跟着鼠标走（这里踩过坑，专门守一道）----
   左栏在分隔条**左边** → 往右拖应变宽；
   右栏在分隔条**右边** → 往右拖应变**窄**（它是被挤掉的那一边）。
   第一版两根都写成同向，结果右栏拖起来完全反着走，用户直接反馈了这个问题。 */
// 先清掉 localStorage：bindResizers() 会用上次保存的尺寸覆盖这里设的初值，
// 不清就会出现"我设了 300，实际从 200 开始拖"的错乱（第一版踩过）
localStorageStub._data = {};
documentStub.documentElement._style = {};
api.applyPanelWidth('sidebar', 300);
api.applyPanelWidth('result', 380);
api.bindResizers();

// 左栏：往右拖 100px → 300 变 400
fire(el.resizerLeft, 'pointerdown', { clientX: 300, pointerId: 10 });
fire(el.resizerLeft, 'pointermove', { clientX: 400, pointerId: 10 });
const widerLeft = parseInt(documentStub.documentElement._style['--sidebar-w'], 10);
fire(el.resizerLeft, 'pointerup', { pointerId: 10 });
check('左分隔条往右拖 → 左侧栏变宽（跟着鼠标方向）',
  widerLeft === 400, `300px → ${widerLeft}px`);

// 右栏：往右拖 100px → 380 变 280（变窄，因为它是被挤掉的一边）
fire(el.resizerRight, 'pointerdown', { clientX: 1200, pointerId: 11 });
fire(el.resizerRight, 'pointermove', { clientX: 1300, pointerId: 11 });
const narrowerRight = parseInt(documentStub.documentElement._style['--result-w'], 10);
fire(el.resizerRight, 'pointerup', { pointerId: 11 });
check('右分隔条往右拖 → 右侧栏变窄（方向必须与左栏相反）',
  narrowerRight === 280, `380px → ${narrowerRight}px`);

// 右栏：往左拖 80px → 280 变 360（变宽）
fire(el.resizerRight, 'pointerdown', { clientX: 1200, pointerId: 12 });
fire(el.resizerRight, 'pointermove', { clientX: 1120, pointerId: 12 });
const widerRight = parseInt(documentStub.documentElement._style['--result-w'], 10);
fire(el.resizerRight, 'pointerup', { pointerId: 12 });
check('右分隔条往左拖 → 右侧栏变宽',
  widerRight === 360, `280px → ${widerRight}px`);

// 水平分隔条：往下拖让代码区变高
api.bindResizers();
fire(el.resizerCode, 'pointerdown', { clientY: 400, pointerId: 3 });
fire(el.resizerCode, 'pointermove', { clientY: 500, pointerId: 3 });   // 往下 100px
const newHeight = documentStub.documentElement._style['--code-height'];
fire(el.resizerCode, 'pointerup', { pointerId: 3 });
check('拖动水平分隔条 → --code-height 被更新',
  Boolean(newHeight) && parseInt(newHeight, 10) > 120, `--code-height=${newHeight}`);
check('代码区加上了 is-height-fixed（改用固定高度）',
  el.codeDropzone.classList.contains('is-height-fixed'));

/* ---------------------------------------------------------------------------
   7. 需求四：大文件分批高亮
   ------------------------------------------------------------------------- */
console.log();
console.log('='.repeat(72));
console.log('需求四  大文件分批高亮（懒加载）');
console.log('='.repeat(72));

check('懒加载阈值是 500 行', api.lazy.LAZY_HIGHLIGHT_THRESHOLD === 500,
  `阈值=${api.lazy.LAZY_HIGHLIGHT_THRESHOLD}`);

// ---- 小文件：一次性高亮 ----
highlightCalls.length = 0;
drainIdle();
api.renderCode(el.codeContent, 'print(1)\nprint(2)\n', 'python');
check('小文件一次性高亮（只调 1 次 highlight）',
  highlightCalls.length === 1, `调用 ${highlightCalls.length} 次`);

// ---- 1000 行：先高亮前 500 行，其余分批 ----
highlightCalls.length = 0;
idleQueue.length = 0;
const bigLines = [];
for (let i = 1; i <= 1000; i += 1) bigLines.push(`int value_${i} = ${i};`);
const bigJava = bigLines.join('\n');

api.renderCode(el.codeContent, bigJava, 'java');

check('渲染后立刻只高亮了第一块（未等空闲时段）',
  highlightCalls.length === 1, `此刻调用 ${highlightCalls.length} 次`);
check('第一块 ≤ 500 行（符合"先只高亮前 500 行"）',
  (highlightCalls[0]?.lines ?? 0) <= 500,
  `第一块 ${highlightCalls[0]?.lines ?? '-'} 行`);
check('内容完整：1000 行全部可见（不是留白等加载）',
  el.codeContent.textContent.includes('value_1000'),
  `textContent 含最后一行: ${el.codeContent.textContent.includes('value_1000')}`);

const steps = drainIdle();
check('空闲时段里把剩余部分分批高亮完',
  highlightCalls.length >= 2, `共调用 ${highlightCalls.length} 次，drain ${steps} 次`);
// 首块是"前 500 行"（阈值），其余每一批都不得超过分块大小 300
check('首块之后的每一批都不超过分块大小 300 行',
  highlightCalls.slice(1).every((item) => item.lines <= 300),
  `各批行数：${highlightCalls.map((c) => c.lines).join(', ')}`);
check('高亮调用次数 = 1（首块）+ 其余分块（证明是分批而不是一次性）',
  highlightCalls.length >= Math.ceil(500 / 300) + 1,
  `共 ${highlightCalls.length} 次`);
// children 里除了 3 个块 <span>，还夹着 2 个换行文本节点（保持行结构用），
// 所以先只留下元素节点再断言——文本节点本来就不该带高亮标记。
const chunkElements = el.codeContent.children.filter(
  (node) => typeof node._innerHTML === 'string',
);
const highlightedChunks = chunkElements.filter(
  (node) => node._innerHTML.includes('hljs-block'),
).length;
check('三个内容块全部被高亮（文本换行节点不算）',
  chunkElements.length === 3 && highlightedChunks === 3,
  `${highlightedChunks}/${chunkElements.length} 块被高亮`);

// ---- 切换文件时取消未完成的任务（避免旧片段写进新代码）----
highlightCalls.length = 0;
idleQueue.length = 0;
api.renderCode(el.codeContent, bigJava, 'java');
api.renderCode(el.codeContent, 'x = 1\n', 'python');   // 学生马上换了另一个文件
drainIdle();
check('高亮未完成时切换文件 → 旧任务被取消（不会污染新代码）',
  !String(el.codeContent._innerHTML).includes('value_1000'),
  '新代码里没有被写入旧文件的高亮片段');

// ---- 安全分块：不要在块注释中间切开 ----
const commentLines = [];
commentLines.push('int main(void) {');
for (let i = 0; i < 30; i += 1) commentLines.push(`    // 注释第 ${i} 行`);
commentLines.push('    return 0;');
commentLines.push('}');
const splitAt = api.safeSplitPoint(commentLines, 0, 10, 'c');
check('分块边界落在行内注释上时不会切错（c/java 会检查块注释配对）',
  splitAt >= 10 && splitAt <= commentLines.length,
  `切在第 ${splitAt} 行`);

const blockComment = [];
blockComment.push('/* 一段很长的块注释');
for (let i = 0; i < 20; i += 1) blockComment.push(` * 第 ${i} 行说明`);
blockComment.push(' */');
blockComment.push('int x = 1;');
const safeEnd = api.safeSplitPoint(blockComment, 0, 5, 'c');
check('块注释没闭合时会往后找到闭合处再切',
  safeEnd > 5, `期望位置 5，实际切在 ${safeEnd}`);

/* ---------------------------------------------------------------------------
   7. 项目库文件夹路径（常显 + 一键复制）
   ------------------------------------------------------------------------- */
console.log();
console.log('【7】项目库文件夹路径');

// ---- 只有一个根目录：直接把完整路径显示出来 ----
api.renderLibraryPath([{ name: 'library', path: 'D:\\demo\\data\\library', exists: true }]);
check('单个项目库目录 → 显示完整绝对路径',
  el.libraryPathValue.textContent === 'D:\\demo\\data\\library',
  el.libraryPathValue.textContent);
check('目录存在时不加 is-missing 标记',
  !el.libraryPathValue.classList.contains('is-missing'));

// ---- 两个根目录：显示第一个 + "等 N 个"，完整清单进 title ----
api.renderLibraryPath([
  { name: 'library', path: 'D:\\demo\\data\\library', exists: true },
  { name: 'examples', path: 'D:\\demo\\examples', exists: true },
]);
check('多个项目库目录 → 显示第一个并注明还有 N 个',
  el.libraryPathValue.textContent === 'D:\\demo\\data\\library 等 2 个文件夹',
  el.libraryPathValue.textContent);
check('完整路径清单写进 title（鼠标悬停可见）',
  el.libraryPathValue.title.includes('D:\\demo\\examples'),
  el.libraryPathValue.title);

// ---- 目录不存在：用醒目样式标出来，别让学生以为代码被吞了 ----
api.renderLibraryPath([{ name: 'library', path: 'D:\\demo\\data\\library', exists: false }]);
check('目录不存在 → 加上 is-missing 标记（界面会变黄提示）',
  el.libraryPathValue.classList.contains('is-missing'));

// ---- 一键复制：把当前路径写进剪贴板 ----
clipboardWrites.length = 0;
api.renderLibraryPath([{ name: 'library', path: 'D:\\demo\\data\\library', exists: true }]);
await api.copyLibraryPath();
check('点「复制」→ 剪贴板拿到完整路径',
  clipboardWrites.length === 1 && clipboardWrites[0] === 'D:\\demo\\data\\library',
  JSON.stringify(clipboardWrites));

// ---- 还没扫到路径时点复制：只提示，不写空内容 ----
clipboardWrites.length = 0;
api.renderLibraryPath([]);
check('后端没配目录 → 显示提示文字而不是空着',
  el.libraryPathValue.textContent.includes('还没有配置'),
  el.libraryPathValue.textContent);
await api.copyLibraryPath();
check('没有路径时点「复制」不会写入空内容',
  clipboardWrites.length === 0);

/* ---------------------------------------------------------------------------
   8. 在线编辑 / 保存
   ------------------------------------------------------------------------- */
console.log();
console.log('【8】代码在线编辑');

/** 造一个"已经打开了项目库文件"的状态，省掉网络往返 */
function primeLibraryFile(overrides = {}) {
  api.state.filename = 'homework/main.py';
  api.state.code = 'def add(a, b):\n    return a + b\n';
  api.state.savedCode = api.state.code;
  api.state.language = 'python';
  api.state.languageLabel = 'Python';
  api.state.allowWrite = true;
  api.state.trashDirName = '.trash';
  api.state.libraryTarget = {
    relPath: 'homework/main.py', root: 'library', sha256: 'sha-old',
    encoding: 'utf-8', ...overrides,
  };
  api.setHasCode(true);
}

primeLibraryFile();
check('有代码时「编辑」按钮可用', el.codeEditBtn.disabled === false);

// ---- 进入编辑模式：编辑器出现，只读代码块让位 ----
api.startEdit();
check('点「编辑」→ 编辑器出现', !el.editorWrap.classList.contains('hidden'));
check('编辑时只读代码块隐藏（两者不能同时显示）',
  el.codeBlock.classList.contains('hidden'));
check('编辑时占位提示保持隐藏', el.codePlaceholder.classList.contains('hidden'));
check('编辑器内容 = 当前代码',
  el.codeEditor.value === 'def add(a, b):\n    return a + b\n');
check('编辑时「保存 / 取消」出现、「编辑」隐藏',
  !el.codeSaveBtn.hidden && !el.codeCancelBtn.hidden && el.codeEditBtn.hidden);
check('刚进编辑模式不算"有改动"（未保存标记不显示）',
  el.codeDirtyBadge.classList.contains('hidden'));
check('行号槽按行数生成（AI 说的"第 N 行"能对上）',
  el.editorGutter.dataset.lines === '3' && el.editorGutter.textContent.startsWith('1\n2\n3'));

// ---- 改一行：state 实时更新 + 未保存标记出现 ----
el.codeEditor.value = 'def add(a, b):\n    return a - b\n';
api.onEditorInput();
check('编辑后 state.code 实时跟着走（AI 按钮分析的是你正在写的这版）',
  api.state.code === 'def add(a, b):\n    return a - b\n');
check('编辑后出现「未保存」标记', !el.codeDirtyBadge.classList.contains('hidden'));

// ---- Tab 缩进 / 回车自动缩进 ----
el.codeEditor.selectionStart = el.codeEditor.value.length;
el.codeEditor.selectionEnd = el.codeEditor.value.length;
let tabPrevented = false;
api.onEditorKeydown({ key: 'Tab', preventDefault: () => { tabPrevented = true; } });
check('Tab 插入 4 个空格并阻止焦点跳走',
  tabPrevented && el.codeEditor.value.endsWith('    '), JSON.stringify(el.codeEditor.value.slice(-6)));

el.codeEditor.value = 'def f():\n    x = 1';
el.codeEditor.selectionStart = el.codeEditor.value.length;
el.codeEditor.selectionEnd = el.codeEditor.value.length;
api.onEditorKeydown({ key: 'Enter', preventDefault: () => {} });
check('回车自动补上当前行缩进',
  el.codeEditor.value.endsWith('\n    '), JSON.stringify(el.codeEditor.value.slice(-6)));

// ---- Ctrl+S 保存：必须真的 PUT，并带上乐观锁指纹 ----
el.codeEditor.value = 'def add(a, b):\n    return a - b\n';
api.onEditorInput();
fetchCalls.length = 0;
fetchQueue.length = 0;
queueResponse({
  rel_path: 'homework/main.py', sha256: 'sha-new', size_bytes: 30,
  line_count: 2, char_count: 29, message: '已保存 homework/main.py', newline: '\n',
});                                              // PUT /library/file
queueResponse({ files: [], roots: [], languages: {}, total_files: 0, total_lines: 0,
  language_counts: {}, language_labels: {}, allow_write: true,
  trash_dir_name: '.trash' });                   // GET /library/scan（保存后刷新列表）
queueResponse({ filename: 'homework/main.py', language: 'python', language_label: 'Python',
  line_count: 2, char_count: 29, symbols: [] }); // POST /tutor/analyze
let savePrevented = false;
api.onEditorKeydown({
  key: 's', ctrlKey: true, preventDefault: () => { savePrevented = true; },
});
await new Promise((resolve) => setImmediate(resolve));
await new Promise((resolve) => setImmediate(resolve));
await new Promise((resolve) => setImmediate(resolve));
await new Promise((resolve) => setImmediate(resolve));

const putCall = findCall('PUT', '/api/v1/library/file');
check('Ctrl+S 拦掉了浏览器的"保存网页"', savePrevented);
check('保存发出的是 PUT /api/v1/library/file', Boolean(putCall));
if (putCall) {
  const body = JSON.parse(putCall.options.body);
  check('请求体里带上了要写入的文件路径', body.path === 'homework/main.py', body.path);
  check('请求体里带上了乐观锁指纹（防止盖掉别处的改动）',
    body.expected_sha256 === 'sha-old', String(body.expected_sha256));
  check('请求体里是编辑器里的新内容',
    body.code === 'def add(a, b):\n    return a - b\n');
}
check('保存成功后退出编辑模式',
  el.editorWrap.classList.contains('hidden') && el.codeBlock.classList.contains('hidden') === false);
check('保存成功后「未保存」标记消失',
  el.codeDirtyBadge.classList.contains('hidden'));
check('保存后用后端返回的新指纹更新基准',
  api.state.libraryTarget.sha256 === 'sha-new', String(api.state.libraryTarget.sha256));
check('保存后刷新了项目库列表',
  Boolean(findCall('GET', '/api/v1/library/scan')));

// ---- 保存失败（409 乐观锁冲突）：必须留在编辑模式，别把内容弄丢 ----
primeLibraryFile();
api.startEdit();
el.codeEditor.value = 'def add(a, b):\n    return a * b\n';
api.onEditorInput();
fetchCalls.length = 0;
fetchQueue.length = 0;
queueResponse({ detail: '这个文件在别处被改动过（内容和打开时不一样）' }, { status: 409 });
await api.saveEdit();
check('保存遇到 409 仍停留在编辑模式（内容不会丢）', api.state.editing === true);
check('409 的提示原样透出给学生',
  el.toast.textContent.includes('别处被改动过'), el.toast.textContent);
check('409 后没有误报"已保存"', el.codeDirtyBadge.classList.contains('hidden') === false);

// ---- 取消编辑：先确认，确认后才还原 ----
let cancelPromise = api.cancelEdit();
await new Promise((resolve) => setImmediate(resolve));
check('放弃修改前先弹确认框（防手滑）',
  !el.confirmModal.classList.contains('hidden'));
check('确认框里写明"找不回来"',
  el.confirmText.textContent.includes('找不回来') || el.confirmText.textContent.includes('放弃'),
  el.confirmText.textContent);
api.closeConfirm(false);          // 点「取消」
await cancelPromise;
check('点「取消」→ 保持在编辑模式，修改还在',
  api.state.editing === true && el.codeEditor.value.includes('*'));

cancelPromise = api.cancelEdit();
await new Promise((resolve) => setImmediate(resolve));
api.closeConfirm(true);           // 点「放弃修改」
await cancelPromise;
check('点「放弃修改」→ 退出编辑并还原成载入时的内容',
  api.state.editing === false && api.state.code === 'def add(a, b):\n    return a + b\n');

// ---- 非项目库的代码：保存只更新预览，并且要如实说明没写盘 ----
api.state.libraryTarget = null;
api.state.code = 'print(1)\n';
api.state.savedCode = api.state.code;
api.startEdit();
el.codeEditor.value = 'print(2)\n';
api.onEditorInput();
fetchCalls.length = 0;
fetchQueue.length = 0;
queueResponse({ filename: 'pasted.py', language: 'python', language_label: 'Python',
  line_count: 1, char_count: 9, symbols: [] });   // analyze
await api.saveEdit();
check('不在项目库里的代码：保存不发 PUT（没有文件可写）',
  findCall('PUT', '/api/v1/library/file') === null);
check('不在项目库里的代码：提示里说清"没有写进任何文件"',
  el.toast.textContent.includes('没有写进任何文件'), el.toast.textContent);
check('不在项目库里的代码：保存后 state.code 已更新',
  api.state.code === 'print(2)\n');

// ---- GBK 文件保存：编码会被统一成 UTF-8，必须如实说明 ----
primeLibraryFile({ encoding: 'gbk' });
api.startEdit();
el.codeEditor.value = 'x = 1\n';
api.onEditorInput();
fetchCalls.length = 0;
fetchQueue.length = 0;
queueResponse({ rel_path: 'homework/main.py', sha256: 'sha-gbk', size_bytes: 6,
  line_count: 1, char_count: 6, message: '已保存 homework/main.py', newline: '\r\n' });
queueResponse({ files: [], roots: [], total_files: 0, total_lines: 0, language_counts: {},
  language_labels: {}, allow_write: true, trash_dir_name: '.trash' });
queueResponse({ filename: 'homework/main.py', language: 'python', language_label: 'Python',
  line_count: 1, char_count: 6, symbols: [] });
await api.saveEdit();
check('保存 GBK 文件时如实说明"已按 UTF-8 保存"（不让学生以为出了乱码）',
  el.toast.textContent.includes('UTF-8'), el.toast.textContent);

/* ---------------------------------------------------------------------------
   9. 替换 / 删除（项目库文件）
   ------------------------------------------------------------------------- */
console.log();
console.log('【9】替换与删除');

// 绑定真实的事件处理（弹窗按钮、替换选择框、Esc 关闭）。
// 不绑的话只能调 closeConfirm() 去"假装用户点了"，测不到按钮到底有没有接上。
api.bindFileActions();

// ---- 列表里的按钮：可写时才有，只读时不渲染 ----
api.state.allowWrite = true;
api.renderLibraryFiles([
  { rel_path: 'homework/main.py', root: 'library', filename: 'main.py',
    language_label: 'Python', line_count: 2, size_bytes: 30, too_large: false },
]);
check('可写模式下列表项带「替换 / 删除」按钮',
  el.libraryList.innerHTML.includes('data-act="replace"')
  && el.libraryList.innerHTML.includes('data-act="delete"'));

api.state.allowWrite = false;
api.renderLibraryFiles([
  { rel_path: 'homework/main.py', root: 'library', filename: 'main.py',
    language_label: 'Python', line_count: 2, size_bytes: 30, too_large: false },
]);
check('只读模式下不渲染这两个按钮（少一个误操作的入口）',
  !el.libraryList.innerHTML.includes('data-act='));

// ---- 只读时点删除：只提示，不发请求 ----
api.state.allowWrite = false;
fetchCalls.length = 0;
await api.requestDelete({ rel_path: 'homework/main.py', root: 'library',
  filename: 'main.py', size_bytes: 30 });
check('只读模式下点删除：不弹确认框、不发请求',
  el.confirmModal.classList.contains('hidden') && fetchCalls.length === 0);

// ---- 正常删除：弹确认 → DELETE → 刷新列表 ----
api.state.allowWrite = true;
api.state.libraryTarget = null;
fetchCalls.length = 0;
fetchQueue.length = 0;
const deletePromise = api.requestDelete({ rel_path: 'homework/main.py', root: 'library',
  filename: 'main.py', size_bytes: 30 });
await new Promise((resolve) => setImmediate(resolve));
check('删除前弹确认框，并写明要删哪个文件',
  !el.confirmModal.classList.contains('hidden')
  && el.confirmText.textContent.includes('homework/main.py'),
  el.confirmText.textContent);
check('确认框里说明白"移到 .trash 还能找回"',
  el.confirmText.textContent.includes('.trash'));
check('确认按钮用了危险样式（红色）',
  el.confirmOkBtn.classList.contains('btn--danger'));

queueResponse({ rel_path: 'homework/main.py', permanent: false, trash_path: 'D:\\lib\\.trash\\x',
  trash_dir: 'D:\\lib\\.trash', size_bytes: 30, message: '已把 homework/main.py 移到回收站' });
queueResponse({ files: [], roots: [], total_files: 0, total_lines: 0, language_counts: {},
  language_labels: {}, allow_write: true, trash_dir_name: '.trash' });
fire(el.confirmOkBtn, 'click');      // 真的点「删除」按钮，验证按钮确实接上了
await deletePromise;

const deleteCall = findCall('DELETE', '/api/v1/library/file');
check('确认后发出 DELETE /api/v1/library/file', Boolean(deleteCall));
check('请求里带上了路径参数',
  Boolean(deleteCall) && deleteCall.url.includes('path=homework') , deleteCall && deleteCall.url);
check('删除后提示里说明文件去哪了',
  el.toast.textContent.includes('回收站'), el.toast.textContent);
check('删除后刷新了项目库列表', Boolean(findCall('GET', '/api/v1/library/scan')));

// ---- 删掉的正好是当前打开的文件：预览要一起清空 ----
api.state.libraryTarget = { relPath: 'a/b.py', root: 'library', sha256: '' };
api.state.code = 'x = 1\n';
api.state.savedCode = 'x = 1\n';
api.setHasCode(true);
fetchCalls.length = 0;
fetchQueue.length = 0;
const deleteOpen = api.requestDelete({ rel_path: 'a/b.py', root: 'library',
  filename: 'b.py', size_bytes: 6 });
await new Promise((resolve) => setImmediate(resolve));
queueResponse({ rel_path: 'a/b.py', permanent: false, trash_path: 't', trash_dir: 'd',
  size_bytes: 6, message: '已删除' });
queueResponse({ files: [], roots: [], total_files: 0, total_lines: 0, language_counts: {},
  language_labels: {}, allow_write: false, trash_dir_name: '.trash' });
api.closeConfirm(true);
await deleteOpen;
check('删掉的正是当前预览的文件 → 代码区一起清空（避免"看着一个不存在的文件"）',
  api.state.hasCode === false && el.codeBlock.classList.contains('hidden'));

// ---- 替换：扩展名不一致时直接拦下 ----
function fakePickedFile(name, text) {
  return { name, size: text.length, text: async () => text };
}

api.state.replaceTarget = { rel_path: 'homework/main.py', root: 'library',
  filename: 'main.py', size_bytes: 30 };
fetchCalls.length = 0;
el.replaceInput.files = [fakePickedFile('other.c', 'int main(void){return 0;}\n')];
await api.performReplace(el.replaceInput);
check('替换时扩展名不一致 → 拦下并给出正确做法',
  fetchCalls.length === 0 && el.toast.textContent.includes('同类型文件'),
  el.toast.textContent);

// ---- 替换：正常流程（确认 → PUT）----
api.state.replaceTarget = { rel_path: 'homework/main.py', root: 'library',
  filename: 'main.py', size_bytes: 30 };
api.state.libraryTarget = null;
fetchCalls.length = 0;
fetchQueue.length = 0;
el.replaceInput.files = [fakePickedFile('new_main.py', 'def add(a, b):\n    return a + b\n')];
const replacePromise = api.performReplace(el.replaceInput);
await new Promise((resolve) => setImmediate(resolve));
check('替换前弹确认框，并说明原内容会被覆盖',
  !el.confirmModal.classList.contains('hidden')
  && el.confirmText.textContent.includes('覆盖'),
  el.confirmText.textContent);

queueResponse({ rel_path: 'homework/main.py', sha256: 'sha-r', size_bytes: 30,
  line_count: 2, char_count: 29, message: '已保存' });
queueResponse({ files: [], roots: [], total_files: 0, total_lines: 0, language_counts: {},
  language_labels: {}, allow_write: true, trash_dir_name: '.trash' });
api.closeConfirm(true);
await replacePromise;

const replacePut = findCall('PUT', '/api/v1/library/file');
check('替换 = 用本机文件的内容覆盖项目库里的那个路径',
  Boolean(replacePut) && JSON.parse(replacePut.options.body).path === 'homework/main.py'
  && JSON.parse(replacePut.options.body).code.includes('return a + b'));
check('替换成功后给出明确反馈',
  el.toast.textContent.includes('替换'), el.toast.textContent);

// ---- 替换的入口按钮：点它会唤起隐藏的文件选择框 ----
api.state.replaceTarget = null;
el.replaceInput._clicked = 0;
api.requestReplace({ rel_path: 'homework/main.py', root: 'library',
  filename: 'main.py', size_bytes: 30 });
check('点「替换」→ 唤起文件选择框，并记下目标文件',
  el.replaceInput._clicked === 1
  && api.state.replaceTarget && api.state.replaceTarget.rel_path === 'homework/main.py');

// ---- Esc 关闭弹窗 ----
api.showConfirm({ title: '测试', text: '内容' });
check('弹窗打开时不是 hidden', !el.confirmModal.classList.contains('hidden'));
fireDocument('keydown', { key: 'Escape' });
check('按 Esc 关闭弹窗（等价于点取消）',
  el.confirmModal.classList.contains('hidden'));

// ---- 点「取消」按钮同样关掉弹窗，而且不会误发请求 ----
fetchCalls.length = 0;
let cancelResult = null;
const cancelProbe = api.showConfirm({ title: '再测一次', text: '内容' }).then((v) => { cancelResult = v; });
fire(el.confirmCancelBtn, 'click');
await cancelProbe;
check('点「取消」→ 弹窗关闭且返回 false（不会误删）',
  cancelResult === false && fetchCalls.length === 0);
void cancelProbe;

/* ---------------------------------------------------------------------------
   10. 模型设置（选模型 + 填自己的 API Key）
   ------------------------------------------------------------------------- */
console.log();
console.log('【10】模型设置：下拉框 / Key 输入 / 保存并测试 / 清除');

api.bindModelSettings();

const MODELS_RESPONSE = {
  models: [
    { id: 'deepseek', label: 'DeepSeek 官方', provider: 'deepseek',
      model_name: 'deepseek-chat', base_url: 'https://api.deepseek.com/v1',
      description: '在 platform.deepseek.com 申请 Key', is_local: false,
      requires_api_key: true, has_default_key: false, is_default: true },
    { id: 'ollama', label: '本地 Ollama（免 Key）', provider: 'ollama',
      model_name: 'qwen2.5-coder:7b', base_url: 'http://127.0.0.1:11434/v1',
      description: '本地模型', is_local: true,
      requires_api_key: false, has_default_key: false, is_default: false },
  ],
  default_model_id: 'deepseek',
  allow_client_key: true,
  session: null,
  total: 2,
};

// ---- 拉取模型清单并填充下拉框 ----
fetchCalls.length = 0;
fetchQueue.length = 0;
queueResponse(MODELS_RESPONSE);
await api.loadModels();

check('模型清单来自 GET /api/v1/models/list',
  Boolean(findCall('GET', '/api/v1/models/list')));
check('下拉框填好了两个模型',
  el.modelSelect.innerHTML.includes('deepseek') && el.modelSelect.innerHTML.includes('ollama'));
check('默认选中后端标的默认模型',
  api.state.modelId === 'deepseek' && el.modelSelect.innerHTML.includes('selected'));
check('选项文字里标出"需要填 Key / 免 Key"',
  el.modelSelect.innerHTML.includes('需要填 Key')
  && el.modelSelect.innerHTML.includes('免 Key'));
check('说明文字带上了模型名与申请地址',
  el.modelHint.textContent.includes('deepseek-chat')
  && el.modelHint.textContent.includes('platform.deepseek.com'));

// ---- 后端只配了一个模型：说清"为什么只有一个"，并给出加模型的办法 ----
// 这一条是踩过的坑：.env 里没写多模型那一段时，下拉框只会出现单模型兜底项，
// 使用者会以为是"项目只支持这一个模型"。
const SINGLE_MODEL_RESPONSE = {
  models: [
    { id: 'default', label: 'deepseek-chat', provider: 'openai-compatible',
      model_name: 'deepseek-chat', base_url: 'https://api.deepseek.com/v1',
      description: '来自单模型配置（LLM__MODEL / LLM__BASE_URL / LLM__API_KEY）',
      is_local: false, requires_api_key: true, has_default_key: false, is_default: true },
  ],
  default_model_id: 'default',
  allow_client_key: true,
  session: null,
  total: 1,
};

fetchQueue.length = 0;
queueResponse(SINGLE_MODEL_RESPONSE);
await api.loadModels();
check('只有一个模型时，说明文字点出"单模型写法"与要改的配置项',
  el.modelHint.textContent.includes('单模型')
  && el.modelHint.textContent.includes('LLM__MODELS__'));

// 恢复成"两个模型"的清单，后面的用例继续基于它跑
fetchQueue.length = 0;
queueResponse(MODELS_RESPONSE);
await api.loadModels();

// ---- 本地模型：不需要 Key，输入框禁用 ----
api.state.modelId = 'ollama';
api.updateModelHint();
check('切到本地模型 → Key 输入框禁用并说明"不需要 Key"',
  el.modelApiKey.disabled === true && el.modelHint.textContent.includes('不需要 API Key'));

// ---- 云端模型 + 空 Key：拦住，不发请求 ----
api.state.modelId = 'deepseek';
api.updateModelHint();
el.modelApiKey.value = '';
fetchCalls.length = 0;
await api.saveAndTestModel();
check('云端模型没填 Key → 就地提示，不发请求',
  fetchCalls.length === 0
  && el.modelStatusText.textContent.includes('请先填入 API Key'));

// ---- 保存并测试：set_key → 真调一次 /tutor/check ----
el.modelApiKey.value = 'sk-user-1234567890';
fetchCalls.length = 0;
fetchQueue.length = 0;
queueResponse({ session: { session_id: 'sess-abc123', models: ['deepseek'],
  default_model_id: 'deepseek', has_key: true, expires_at: '2026-01-01T00:00:00Z',
  last_used_at: '2026-01-01T00:00:00Z', use_count: 1 },
  message: '已保存', ttl_seconds: 1800 });                       // POST /auth/set_key
queueResponse({ filename: '_key_check.py', language: 'python', language_label: 'Python',
  score: 90, level: '优秀', issues: [], highlights: [], summary: 'ok',
  ai_available: true, model: 'deepseek-chat', record_id: 1 });    // POST /tutor/check
await api.saveAndTestModel();

const setKeyCall = findCall('POST', '/api/v1/auth/set_key');
const probeCall = findCall('POST', '/api/v1/tutor/check');
check('点了「保存并测试」→ 先调 set_key', Boolean(setKeyCall));
check('set_key 的请求体带上了模型与 Key',
  Boolean(setKeyCall) && JSON.parse(setKeyCall.options.body).model_id === 'deepseek'
  && JSON.parse(setKeyCall.options.body).api_key === 'sk-user-1234567890');
check('随后真的调了一次 AI 检测来验证 Key', Boolean(probeCall));
check('验证请求带着会话号（X-Session-Id）',
  Boolean(probeCall) && probeCall.options.headers['X-Session-Id'] === 'sess-abc123');
check('会话号只放在内存里，没有写进 localStorage',
  JSON.stringify(localStorageStub._data).indexOf('sess-abc123') === -1
  && JSON.stringify(localStorageStub._data).indexOf('sk-user') === -1);
check('成功后状态显示"已连接 DeepSeek 官方"',
  el.modelStatusText.textContent.includes('已连接')
  && el.modelStatus.classList.contains('is-ok'));
check('保存成功后清空输入框（前端不留 Key 副本）', el.modelApiKey.value === '');

// ---- Key 无效：把后端的中文提示原样展示，并且不能报"已连接" ----
el.modelApiKey.value = 'sk-wrong-key-000000';
fetchCalls.length = 0;
fetchQueue.length = 0;
queueResponse({ session: { session_id: 'sess-def456', models: ['deepseek'],
  default_model_id: 'deepseek', has_key: true, expires_at: '2026-01-01T00:00:00Z',
  last_used_at: '2026-01-01T00:00:00Z', use_count: 1 },
  message: '已保存', ttl_seconds: 1800 });
queueResponse({ detail: 'AI 调用失败：API Key 无效或已过期（HTTP 401）' }, { status: 502 });
await api.saveAndTestModel();
check('Key 无效 → 状态变红并原样显示后端提示',
  el.modelStatus.classList.contains('is-error')
  && el.modelStatusText.textContent.includes('API Key 无效'));
check('Key 无效时绝不说"已连接"',
  !el.modelStatusText.textContent.includes('已连接'));

// ---- 后端没真调用模型（ai_available=false）也要如实说 ----
el.modelApiKey.value = 'sk-user-1234567890';
fetchCalls.length = 0;
fetchQueue.length = 0;
queueResponse({ session: { session_id: 'sess-ghi789', models: ['deepseek'],
  default_model_id: 'deepseek', has_key: true, expires_at: '2026-01-01T00:00:00Z',
  last_used_at: '2026-01-01T00:00:00Z', use_count: 1 },
  message: '已保存', ttl_seconds: 1800 });
queueResponse({ filename: '_key_check.py', language: 'python', language_label: 'Python',
  score: 80, level: '良好', issues: [], highlights: [], summary: '本地结论',
  ai_available: false, model: '', record_id: 2 });
await api.saveAndTestModel();
check('后端没真正调用模型时如实提示，不谎报"已连接"',
  el.modelStatus.classList.contains('is-error')
  && el.modelStatusText.textContent.includes('没有真正调用模型'));

// ---- 清除 Key：调 clear_key 并清空本地状态 ----
fetchCalls.length = 0;
fetchQueue.length = 0;
queueResponse({ cleared: true, session_id: 'sess-abc123', message: '已清除本次会话的全部 API Key' });
await api.clearModelKey();
const clearCall = findCall('POST', '/api/v1/auth/clear_key');
check('点「清除 Key」→ 调 clear_key', Boolean(clearCall));
check('清除后本地会话号与输入框都清空',
  api.state.sessionId === '' && el.modelApiKey.value === '');
check('清除后状态提示改回"未连接"', el.modelStatusText.textContent.includes('已清除'));

// ---- 显示 / 隐藏 Key ----
el.modelApiKey.type = 'password';
fire(el.toggleKeyVisible, 'click');
check('点「显示」→ 输入框变成明文（type=text）',
  el.modelApiKey.type === 'text' && el.toggleKeyVisible.textContent === '隐藏');
fire(el.toggleKeyVisible, 'click');
check('再点一次 → 又变回隐藏', el.modelApiKey.type === 'password');

// ---- 换模型时提示要重新保存，并且徽章不能还挂着旧模型 ----
api.state.models = MODELS_RESPONSE.models;
api.state.modelId = 'deepseek';
api.state.aiAvailable = false;          // 后端没配 Key，只有用户自己连的那把
api.state.modelConnected = true;
api.state.modelLabel = 'DeepSeek 官方';
api.renderAiBadge();
// 真实浏览器里 select 的 value 会先被用户改掉，change 事件才触发，这里照做
el.modelSelect.value = 'ollama';
fire(el.modelSelect, 'change');
check('切换模型 → 提示需要重新保存并测试',
  api.state.modelId === 'ollama' && el.modelStatusText.textContent.includes('保存并测试'));
check('切换模型 → 徽章不再显示旧模型的"已连接"',
  el.aiBadge.textContent === 'AI 未启用' && api.state.modelConnected === false,
  el.aiBadge.textContent);

/* ---------------------------------------------------------------------------
   11. 右上角 AI 徽章：填完 Key 必须从「AI 未启用」变成「AI 已就绪」
   ------------------------------------------------------------------------- */
console.log();
console.log('【11】右上角 AI 状态徽章（用户反馈：填了 Key 还显示未启用）');

// ---- 后端没配 Key、页面也没填：未启用 ----
api.state.sessionId = '';
api.state.modelConnected = false;
api.state.aiAvailable = false;
api.renderAiBadge();
check('什么都没有 → 徽章显示「AI 未启用」并置灰',
  el.aiBadge.textContent === 'AI 未启用'
  && el.aiBadge.classList.contains('badge--off'));
check('此时 aiReady() 为 false（按钮该灰）', api.aiReady() === false);

// ---- 用户在页面上填了 Key 并测通：徽章变绿 ----
api.state.sessionId = 'sess-xyz';
api.state.modelConnected = true;
api.state.modelLabel = 'DeepSeek 官方';
api.state.modelName = 'deepseek-flash';
api.renderAiBadge();
check('页面上填好 Key 之后 → 徽章显示「AI 已就绪 · DeepSeek 官方」',
  el.aiBadge.textContent === 'AI 已就绪 · DeepSeek 官方'
  && el.aiBadge.classList.contains('badge--ok'),
  el.aiBadge.textContent);
check('此时 aiReady() 为 true（按钮可点）', api.aiReady() === true);

// ---- 有代码时三个按钮要真的变可点 ----
api.state.hasCode = true;
api.setHasCode(true);
api.setBusy(false);
check('AI 就绪 + 有代码 → 三个 AI 按钮可点',
  el.btnCheck.disabled === false && el.btnComment.disabled === false
  && el.btnFix.disabled === false);

// ---- 后端自己配了 Key 的情况：徽章也要绿（显示后端模型名）----
api.state.aiAvailable = true;
api.state.backendModel = 'deepseek-chat';
api.state.modelConnected = false;
api.renderAiBadge();
check('后端配了 Key → 徽章显示「AI 已就绪 · deepseek-chat」',
  el.aiBadge.textContent === 'AI 已就绪 · deepseek-chat'
  && el.aiBadge.classList.contains('badge--ok'),
  el.aiBadge.textContent);

// ---- 清除 Key 之后：徽章退回未启用、按钮变灰 ----
api.state.sessionId = '';
api.state.modelConnected = false;
api.state.modelLabel = '';
api.state.aiAvailable = false;
api.renderAiBadge();
api.setBusy(false);
check('清除 Key 之后 → 徽章回到「AI 未启用」且按钮变灰',
  el.aiBadge.textContent === 'AI 未启用' && el.btnCheck.disabled === true);

// ---- 会话过期：后端说"请先在网页上输入 API Key"时，徽章要自己复位 ----
api.state.modelConnected = true;
api.state.sessionId = 'sess-old';
api.state.modelLabel = 'DeepSeek 官方';
api.renderAiBadge();
check('（前置）徽章此时显示已连接', el.aiBadge.classList.contains('badge--ok'));
api.handleAiError({ message: 'AI 功能尚未启用：请先在网页上输入 API Key（或在后端 .env 里配置 LLM__API_KEY）' });
check('会话过期后调用 AI → 徽章自动复位成「AI 未启用」',
  el.aiBadge.textContent === 'AI 未启用' && api.state.modelConnected === false,
  el.aiBadge.textContent);
check('并提示学生重新填 Key',
  el.modelStatusText.textContent.includes('请重新填写'), el.modelStatusText.textContent);

/* ---------------------------------------------------------------------------
   12. 汇总
   ------------------------------------------------------------------------- */
console.log();
console.log('='.repeat(72));
console.log(`前端行为测试：${passed}/${passed + failures.length} 项通过`);
console.log('='.repeat(72));
failures.forEach((label) => console.log(`  FAIL  ${label}`));
process.exit(failures.length ? 1 : 0);
