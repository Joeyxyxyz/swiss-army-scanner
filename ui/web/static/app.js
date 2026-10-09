'use strict';
/*
 * Swiss Army Scanner web console. Plain JS: no build step and no CDN (the bench may be offline).
 * Every action is an engine job: POST /api/jobs {verb, args} -> poll GET /api/jobs/<id>.
 * The page passes library ids, never objects; the server resolves them on its single engine worker.
 */

// =================================================================== utilities
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clone = (o) => (o == null ? o : JSON.parse(JSON.stringify(o)));
const LS = {
  get(k, d) { try { const v = localStorage.getItem('sas.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('sas.' + k, JSON.stringify(v)); } catch (e) { /* storage blocked: per-viewer convenience only */ } },
};

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, v);
    }
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
// DOM append() would print "null": optional children (cond ? x : null) are dropped instead
const clean = (kids) => kids.flat(Infinity).filter((k) => k != null && k !== false);
function add(el, ...kids) { el.append(...clean(kids)); return el; }
function put(el, ...kids) { el.replaceChildren(...clean(kids)); return el; }
const on = (sel, ev, fn) => { const el = typeof sel === 'string' ? $(sel) : sel; if (el) el.addEventListener(ev, fn); return el; };

function fmt(v, d = 2) {
  if (v == null || (typeof v === 'number' && !Number.isFinite(v))) return '—';
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && a < Math.pow(10, -d)) return v.toPrecision(2);
  return v.toFixed(d);
}
const num = (v) => (v === '' || v == null ? NaN : Number(v));
const shortTime = (iso) => (iso ? String(iso).replace('T', ' ').slice(5, 19) : '');
const isAbs = (p) => /^([a-zA-Z]:|[\\/])/.test(p || '');
function fileUrl(rel, download) {
  if (!rel || typeof rel !== 'string' || isAbs(rel)) return null;
  return '/files/' + rel.split('/').map(encodeURIComponent).join('/') + (download ? '?download=1' : '');
}
function absPath(rel) {
  const lib = S.status && S.status.paths && S.status.paths.library;
  if (!rel) return '';
  if (isAbs(rel) || !lib) return rel;
  const sep = lib.includes('\\') ? '\\' : '/';
  return lib.replace(/[\\/]+$/, '') + sep + rel.split('/').join(sep);
}
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const tag = (t, cls) => h('span', { class: 'tag ' + (cls || '') }, t);
const baseName = (p) => String(p || '').split(/[\\/]/).pop();

function kv(rows) {
  return h('dl', { class: 'kv' }, rows.filter(Boolean).map(([k, v, u, d]) => [
    h('dt', null, k),
    h('dd', null, v instanceof Node ? v : fmt(v, d == null ? 3 : d), u ? h('span', { class: 'u' }, u) : null),
  ]));
}
function metric(k, v, unit, cls, sub, d = 2) {
  return h('div', { class: 'metric ' + (cls || '') },
    h('div', { class: 'k' }, k),
    h('div', { class: 'v' }, typeof v === 'number' ? fmt(v, d) : (v == null ? '—' : v), unit ? h('span', { class: 'u' }, unit) : null),
    sub ? h('div', { class: 's' }, sub) : null);
}
function warningsList(ws) {
  return ws && ws.length ? h('ul', { class: 'warnings' }, ws.map((w) => h('li', null, w))) : null;
}
function fmtCell(v) {
  if (v == null) return '—';
  if (typeof v === 'number') return fmt(v);
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (Array.isArray(v)) return v.map((x) => (typeof x === 'number' ? fmt(x, 1) : JSON.stringify(x))).join(', ');
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}
function table(cols, rows, o = {}) {
  return h('div', { class: 'tablewrap' }, h('table', { class: 't' },
    h('thead', null, h('tr', null, cols.map((c) => h('th', { class: c.n ? 'n' : '' }, c.label)))),
    h('tbody', null, rows.map((r, i) => h('tr', { class: o.rowClass ? o.rowClass(r, i) : '' },
      cols.map((c) => h('td', { class: c.n ? 'n' : '' }, c.f ? c.f(r[c.k], r, i) : fmtCell(r[c.k]))))))));
}
function verdictBadge(v) {
  const cls = { PASS: 'ok', FAIL: 'bad', FLAG: 'bad', INDETERMINATE: 'na' }[v] || 'na';
  return h('span', { class: 'verdict ' + cls }, v || '—');
}
// a sensor's state: PRESENT / ABSENT are both normal answers (the output is on or off);
// INDETERMINATE means the sensor could not decide and must not be read as either
const SENSOR_CLS = { PRESENT: 'ok', ABSENT: 'info', INDETERMINATE: 'warn', ALIGNED: 'ok', NOT_ALIGNED: 'bad' };
function sensorBadge(v) {
  return h('span', { class: 'verdict big ' + (SENSOR_CLS[v] || 'na') }, (v || '—').replace('_', ' '));
}
/** Why a verdict is INDETERMINATE (inspect's "cannot PASS: ...", a sensor's "INDETERMINATE: ..."), for next to its badge. */
function verdictWhy(verdict, warnings) {
  if (verdict !== 'INDETERMINATE') return null;
  const w = (warnings || []).find((x) => /cannot PASS|^INDETERMINATE\b/i.test(x));
  return w ? h('span', { class: 'why' }, w.replace(/^INDETERMINATE:\s*/i, '')) : null;
}
function fileLink(rel, text, download) {
  const url = fileUrl(rel, download);
  if (!url) return rel ? h('span', { class: 'mono small' }, rel) : null;
  return h('a', { href: url, target: download ? null : '_blank', download: download ? baseName(rel) : null }, text || baseName(rel));
}
function figure(rel, caption) {
  const url = fileUrl(rel);
  if (!url) return null;
  return h('figure', null,
    h('img', { src: url, alt: caption, loading: 'lazy', onclick: () => openImage(rel, caption) }),
    h('figcaption', null, h('span', null, caption), h('a', { href: url, target: '_blank' }, 'open')));
}
function figures(...pairs) {
  const f = pairs.filter((p) => p && p[0]).map(([rel, cap]) => figure(rel, cap)).filter(Boolean);
  return f.length ? h('div', { class: 'imgs' }, f) : null;
}
function thumb(rel, title) {
  const url = fileUrl(rel);
  return h('div', { class: 'thumb', onclick: () => url && openImage(rel, title), title: url ? 'click to enlarge' : '' },
    url ? h('img', { src: url, alt: title || '', loading: 'lazy' }) : h('span', { class: 'muted small' }, 'no preview'));
}

// =================================================================== chrome
function toast(msg, cls = '', detail = '') {
  const box = $('#toasts');
  add(box, h('div', { class: 'toast ' + cls }, h('div', null, msg), detail ? h('div', { class: 'd' }, detail) : null));
  while (box.children.length > 5) box.firstChild.remove();
  const t = box.lastChild;
  setTimeout(() => t.remove(), cls === 'bad' ? 9000 : 4000);
}
function openImage(rel, title) {
  const url = fileUrl(rel);
  if (!url) return;
  $('#modalImg').src = url;
  put($('#modalTitle'), title || '', ' · ', h('a', { href: url, target: '_blank' }, baseName(rel)));
  $('#modal').hidden = false;
}
function closeModal() { $('#modal').hidden = true; $('#modalImg').removeAttribute('src'); }
function errorBox(job) {
  const notBuilt = job.error_type === 'NotImplementedError';
  return h('div', { class: 'alert ' + (notBuilt ? 'info' : 'bad') },
    h('b', null, notBuilt ? 'Not built yet' : `${job.verb || 'Action'} failed${job.error_type ? ' · ' + job.error_type : ''}`),
    h('div', { class: 'msg' }, job.error || 'unknown error'),
    job.hint ? h('div', { class: 'hint' }, job.hint) : null);
}
function listNote(r) {
  if (!r) return null;
  if (r.error) {
    const nb = r.error_type === 'NotImplementedError';
    return h('div', { class: 'alert ' + (nb ? 'info' : 'bad') }, h('b', null, nb ? 'Not built yet' : 'Could not list'),
      h('div', { class: 'msg' }, r.error), r.hint ? h('div', { class: 'hint' }, r.hint) : null);
  }
  if (r.stale) return h('div', { class: 'alert warn' }, r.note || 'Worker busy: showing the last listing.');
  return null;
}

// =================================================================== server access
async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body !== undefined ? 'POST' : 'GET'), headers: {} };
  if (opts.body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(opts.body);
  }
  let res;
  try { res = await fetch(path, init); } catch (e) { throw new Error('UI server unreachable: is it still running?'); }
  const txt = await res.text();
  let data = null;
  try { data = txt ? JSON.parse(txt) : null; } catch (e) { data = { error: txt }; }
  if (!res.ok) {
    const err = new Error((data && data.error) || `${res.status} ${res.statusText}`);
    err.status = res.status; err.data = data;
    throw err;
  }
  return data;
}

const done = (job) => !!job && job.status === 'done';

/** Submit a verb, show progress in `out`, disable `btn`, resolve with the finished job. */
async function runJob(verb, args = {}, o = {}) {
  const btns = [].concat(o.btn || []).filter(Boolean);
  btns.forEach((b) => { b.disabled = true; b.classList.add('busy'); });
  let t = null;
  if (o.out) {
    t = h('span', null, `${verb}: submitting…`);
    put(o.out, h('div', { class: 'running' }, h('span', { class: 'spin' }), t));
  }
  const t0 = Date.now();
  let job;
  try {
    job = await api('/api/jobs', { body: { verb, args } });
    while (job.status === 'queued' || job.status === 'running') {
      if (t) {
        const s = ((Date.now() - t0) / 1000).toFixed(1);
        const cur = S.status && S.status.job;
        t.textContent = job.status === 'queued'
          ? `${verb}: queued${cur && cur.id !== job.id ? ' behind ' + cur.verb : ''}… ${s} s`
          : `${verb}: running… ${s} s`;
      }
      await sleep(job.status === 'queued' ? 300 : 400);
      job = await api('/api/jobs/' + job.id);
    }
  } catch (e) {
    job = { verb, status: 'error', error: e.message, error_type: 'Request failed' };
  } finally {
    btns.forEach((b) => { b.disabled = false; b.classList.remove('busy'); });
  }
  if (o.out) put(o.out);
  if (job.status === 'error') {
    if (o.out) add(o.out, errorBox(job));
    // quietError: the caller shows the failure itself (e.g. an optional comparison)
    if (!o.quietError) toast(`${verb} failed`, job.error_type === 'NotImplementedError' ? 'warn' : 'bad', job.error);
  } else {
    if (job.note && o.out) add(o.out, h('div', { class: 'alert info' }, job.note));
    if (!o.quiet) toast(`${verb} done`, 'ok', job.duration_s != null ? `${job.duration_s} s` : '');
  }
  refreshStatus();
  return job;
}

// =================================================================== state + status
const S = {
  status: null, geo: undefined, presets: [], builtins: [], form: null, loaded: null, loadedName: null,
  basesAll: [], bases: [], objects: [], banks: [], materials: [], readings: [], sensors: [],
  grid: null, zones: [], cloudId: null, calSel: new Set(), lastJobId: null, lastSpot: null,
};

function activePresetName() { return S.status && S.status.preset ? S.status.preset.name : null; }
function activeBaseId() {
  const id = LS.get('base.' + S.geo, null);
  const b = S.bases.find((x) => x.id === id && !(x.meta && x.meta.rejected));
  return b ? b.id : null;
}
function setActiveBase(id) {
  LS.set('base.' + S.geo, id);
  fillBaseSelects(); renderBases(); renderHeader();
}

async function refreshStatus() {
  let st;
  try { st = await api('/api/status'); } catch (e) { $('#connBanner').hidden = false; return; }
  $('#connBanner').hidden = true;
  const prevGeo = S.geo;
  S.status = st;
  S.geo = st.geometry_id;
  renderHeader();
  if (prevGeo !== undefined && prevGeo !== st.geometry_id) onGeometryChange();
  syncGrid(st.grid);
  // a job finished (maybe started by an agent over HTTP): refresh what this tab shows
  const cur = st.job && st.job.verb !== 'live_view' ? st.job.id : null;
  if (S.lastJobId && S.lastJobId !== cur) refreshTab();
  S.lastJobId = cur;
  if (!$('#bench').hidden) renderBench(st.bench);
}

function setChip(sel, key, val, cls, title) {
  const el = $(sel);
  put(el, key ? key + ' ' : '', h('b', null, val));
  el.className = 'chip ' + (cls || '');
  if (title) el.title = title;
}

function renderHeader() {
  const st = S.status;
  if (!st) return;
  const mb = $('#modeBadge');
  mb.textContent = st.mock ? 'MOCK' : 'REAL';
  mb.className = 'badge ' + (st.mock ? 'mock' : 'real');
  mb.title = st.mock ? 'Simulated bench: no hardware is touched' : 'Real hardware: camera + LightCrafter over HDMI';
  $('#realBanner').hidden = !!st.mock;
  $('#benchBtn').hidden = !st.mock;
  document.title = (st.mock ? '[MOCK] ' : '[REAL] ') + 'Swiss Army Scanner';
  const p = st.preset;
  setChip('#chipPreset', 'preset', p ? p.name : 'none loaded', p ? '' : 'warn');
  setChip('#chipGeo', 'geometry', st.geometry_id || '—');
  const c = st.calibration;
  if (c && (c.mm_per_px != null || c.lateral_px_per_mm != null)) {
    const parts = [];
    if (c.mm_per_px != null) parts.push(`Z ${fmt(c.mm_per_px, 4)} mm/px`);
    if (c.lateral_px_per_mm != null) parts.push(`XY ${fmt(c.lateral_px_per_mm, 3)} px/mm`);
    setChip('#chipCal', 'cal', parts.join(' · '), 'ok', `calibration ${c.id}${c.r2 != null ? ' · R² ' + fmt(c.r2, 5) : ''}`);
  } else {
    setChip('#chipCal', '', 'uncalibrated · results in px', 'warn', 'No calibration for this geometry: measurements are reported in px (rule 6)');
  }
  const act = S.bases.find((b) => b.id === activeBaseId());
  const lb = st.latest_base;
  if (act) setChip('#chipBase', 'base', act.label, '', act.id);
  else if (lb) setChip('#chipBase', 'base', 'latest: ' + lb.label, '', lb.id);
  else setChip('#chipBase', 'base', 'none: capture one', 'warn');
  const ji = $('#jobInd');
  const q = (st.queued || []).length;
  if (st.job && st.job.verb !== 'live_view') {
    ji.className = 'jobind busy';
    put(ji, h('span', { class: 'spin' }), st.job.verb, q ? ` +${q} queued` : '');
  } else if (st.live && st.live.running) {
    ji.className = 'jobind live';
    put(ji, h('span', { class: 'spin' }), 'live view', q ? ` +${q} queued` : '');
  } else {
    ji.className = 'jobind';
    ji.textContent = q ? `${q} queued` : 'idle';
  }
  renderJobMenu();
  $$('.geoName').forEach((e) => { e.textContent = st.geometry_id || '—'; });
  updateAgentBrief();
}

// ---------- job queue menu (click the job indicator): cancel queued jobs, stop the live view
let jobMenuKey = null;
function renderJobMenu(force) {
  const box = $('#jobMenu');
  if (box.hidden || !S.status) return;
  const st = S.status;
  const cur = st.job;
  const q = st.queued || [];
  const key = JSON.stringify([cur && cur.id, cur && cur.verb, q.map((j) => j.id), st.live && st.live.running]);
  if (key === jobMenuKey && !force) return;         // unchanged: keep the buttons under the mouse
  jobMenuKey = key;
  const liveRunning = st.live && st.live.running;
  const rows = [];
  if (cur) {
    rows.push(h('div', { class: 'jm' }, h('span', { class: 'spin' }), h('b', null, cur.verb), h('span', { class: 'mono small muted' }, ` ${cur.id} · started ${shortTime(cur.started)}`),
      cur.verb === 'live_view' ? h('button', { class: 'small', onclick: () => liveStop().then(refreshStatus) }, 'Stop')
        : h('span', { class: 'muted small' }, 'running: a capture cannot be interrupted')));
  }
  if (liveRunning && (!cur || cur.verb !== 'live_view')) {
    rows.push(h('div', { class: 'jm' }, h('b', null, 'live view'), h('span', { class: 'muted small' }, ' (serving a rig-free job)'),
      h('button', { class: 'small', onclick: () => liveStop().then(refreshStatus) }, 'Stop')));
  }
  q.forEach((j, i) => rows.push(h('div', { class: 'jm' }, h('span', { class: 'muted small' }, `${i + 1}.`), h('b', null, j.verb),
    h('span', { class: 'mono small muted' }, ` ${j.id} · queued ${shortTime(j.created)}`),
    h('button', { class: 'small danger', onclick: () => cancelJob(j) }, 'Cancel'))));
  put(box, h('div', { class: 'cardhead' }, h('b', null, 'Engine jobs'), h('button', { class: 'ghost small', onclick: () => { box.hidden = true; } }, '×')),
    rows.length ? rows : h('div', { class: 'muted small' }, 'Idle: nothing running or queued.'),
    h('div', { class: 'muted small' }, 'One job at a time on the rig. Queued jobs can be cancelled before they start.'));
}
async function cancelJob(j) {
  try {
    await api(`/api/jobs/${encodeURIComponent(j.id)}/cancel`, { body: {} });
    toast(`${j.verb} cancelled`, 'ok', j.id);
  } catch (e) { toast(`could not cancel ${j.verb}`, 'bad', e.message); }
  refreshStatus();
}

function onGeometryChange() {
  S.grid = LS.get('grid.' + S.geo, null);
  S.calSel.clear();
  S.lastSpot = null;                  // spots, sensors and the measured spot scale are per geometry
  refreshBases(); refreshObjects(); renderGrid();
  if (curTab === 'sensors') { refreshSensors(); renderSensPick(); }
}

// =================================================================== tabs
const TABS = {
  setup: { show: showSetup },
  focus: { show: showFocus, hide: () => { Live.polling = false; stopLiveOnLeave(); } },
  bases: { show: () => refreshBases() },
  scan3d: { show: () => { refreshBases(); refreshObjects(); Cloud.draw(); } },
  measure: { show: () => { refreshObjects(); refreshCals(); } },
  defects: { show: () => { refreshObjects(); refreshBanks(); } },
  codes: { show: () => refreshCodeReadings() },
  spectral: { show: () => { refreshReadings(); refreshMaterials(); } },
  grid: { show: () => { refreshBases(); refreshMaps(); renderGrid(); refreshAtlases(); } },
  sensors: { show: () => { refreshObjects(); refreshSensors(); renderSensPick(); } },
  cell: { show: () => { refreshObjects(); refreshCellPickers(); } },
  outputs: { show: () => refreshOutputs() },
  log: { show: () => refreshLog(), hide: () => {} },
};
let curTab = null;
function showTab(name) {
  if (!TABS[name]) name = 'setup';
  if (curTab && curTab !== name && TABS[curTab].hide) TABS[curTab].hide();
  curTab = name;
  $$('.tab').forEach((s) => s.classList.toggle('on', s.id === 'tab-' + name));
  $$('#tabs a').forEach((a) => a.classList.toggle('on', a.dataset.tab === name));
  LS.set('tab', name);
  TABS[name].show();
}
function refreshTab() { if (curTab && curTab !== 'setup' && curTab !== 'focus') TABS[curTab].show(); }
function go(name) { if (location.hash !== '#' + name) location.hash = name; else showTab(name); }

// =================================================================== library lists
async function listKind(kind, geo) {
  try { return await api(`/api/library/${kind}` + (geo ? `?geometry_id=${encodeURIComponent(geo)}` : '')); } catch (e) { return { items: [], error: e.message }; }
}

async function refreshBases() {
  const r = await listKind('bases');
  S.basesAll = r.items || [];
  S.bases = S.basesAll.filter((b) => b.geometry_id === S.geo);
  S.basesNote = r;
  renderBases(); fillBaseSelects(); renderHeader();
}

async function refreshObjects() {
  if (!S.geo) return;
  const r = await listKind('objects', S.geo);
  S.objects = (r.items || []).slice().reverse();          // newest first
  S.objectsNote = r;
  renderObjects(); fillObjectSelects(); renderCalList();
}

function fillBaseSelects() {
  const act = activeBaseId();
  $$('select[data-bases]').forEach((sel) => {
    const keep = sel.dataset.touched ? sel.value : (act || '');
    put(sel, h('option', { value: '' }, 'latest usable base'),
      S.bases.slice().reverse().map((b) => h('option', { value: b.id, disabled: !!(b.meta && b.meta.rejected) },
        `${b.label} · ${shortTime(b.created)}${b.meta && b.meta.rejected ? ' (rejected)' : ''}`)));
    sel.value = [...sel.options].some((o) => o.value === keep && !o.disabled) ? keep : '';
  });
}
const objLabel = (o) => `${o.label} · ${shortTime(o.created)}${o.known_dims ? ' · known dims' : ''}`;
/**
 * Which scan an object picker shows after the list is refreshed (pure; tests/test_ui_server.py).
 * ids: newest first. forced: the scan this page just captured (every picker but the golden
 * follows it). fresh: the newest scan that appeared since the last refresh, e.g. captured by an
 * agent (pickers the operator has not set by hand follow it). The golden is a deliberate choice.
 */
function objectSelectValue({ selId, current, touched, ids, fresh, forced }) {
  const golden = selId === 'cmpGolden';
  if (!golden && forced && ids.includes(forced)) return forced;
  if (!golden && fresh && !touched && ids.includes(fresh)) return fresh;
  if (current && ids.includes(current)) return current;
  if (golden && ids.length > 1) return ids[1];
  return ids[0] || '';
}
function fillObjectSelects() {
  const ids = S.objects.map((o) => o.id);
  const seen = S.objIdsSeen;
  const fresh = seen ? ids.find((id) => !seen.has(id)) || null : null;
  const forced = S.selectScan && ids.includes(S.selectScan) ? S.selectScan : null;
  $$('select[data-objects]').forEach((sel) => {
    const current = sel.value;
    put(sel, ...(S.objects.length ? S.objects.map((o) => h('option', { value: o.id }, objLabel(o)))
      : [h('option', { value: '' }, 'no scans at this geometry')]));
    sel.value = objectSelectValue({ selId: sel.id, current, touched: !!sel.dataset.touched, ids, fresh, forced });
    if (forced && sel.id !== 'cmpGolden') delete sel.dataset.touched;
    if (sel.id === 'profScan' && sel.value !== Prof.id) loadProfileImage();
  });
  if (forced) S.selectScan = null;
  S.objIdsSeen = new Set(ids);
}
const objById = (id) => S.objects.find((o) => o.id === id);
/** "label · time · id" of a scan, for result headers: every result says which scan it used. */
function scanRef(id) {
  const o = objById(id);
  return h('span', { class: 'scanref', title: id },
    h('b', null, o ? o.label : '(scan not in the list)'), o ? h('span', { class: 'muted small' }, ` · ${shortTime(o.created)}`) : null,
    ' ', h('span', { class: 'mono small muted' }, id));
}
function resultHead(...parts) { return h('div', { class: 'resulthead' }, ...parts); }

async function deleteEntry(kind, e) {
  const id = e.id || e.name;
  const what = { bases: 'base', objects: 'scan', calibrations: 'calibration', banks: 'bank', maps: 'projector map', spectral: 'spectral reading', readings: 'code reading', materials: 'material (every learned version)', sensors: 'sensor (every taught version)' }[kind] || kind;
  if (!confirm(`Move ${what} "${e.label || e.name || id}" to trash?\n\n${id}\n\nNothing is destroyed: it moves to library/trash/ and can be moved back.`)) return;
  const job = await runJob('delete', { kind, id, reason: 'deleted from the web UI' }, {});
  if (done(job)) {
    toast(`${what} moved to trash`, 'ok', job.result && job.result.trashed_to);
    if (kind === 'bases') refreshBases();
    if (kind === 'objects') { if (S.cloudId === id) { S.cloudId = null; Cloud.setData(null); $('#cloudTitle').textContent = 'no scan selected'; } refreshObjects(); }
    if (kind === 'banks') refreshBanks();
    if (kind === 'maps') refreshMaps();
    if (kind === 'spectral') refreshReadings();
    if (kind === 'calibrations') refreshCals();
    if (kind === 'readings') refreshCodeReadings();
    if (kind === 'materials') refreshMaterials();
    if (kind === 'sensors') refreshSensors();
  }
}

// =================================================================== 1 SETUP
const FORM = [
  { title: 'Camera', fields: [
    { p: 'camera.exposure_us', label: 'Exposure', type: 'int', unit: 'µs', min: 1, hint: 'the scan assistant recommends one' },
    { p: 'camera.gain_db', label: 'Gain', type: 'num', unit: 'dB', step: 0.1 },
    { p: 'camera.average', label: 'Average', type: 'int', unit: 'frames', min: 1, hint: 'frames per pattern; noise ~ 1/√N, time ~ N' },
    { p: 'camera.mono', label: 'Mono camera', type: 'bool' },
  ] },
  { title: 'Projection', fields: [
    { p: 'projection.monitor_offset.0', label: 'Monitor offset X', type: 'int', unit: 'px', hint: 'EVM display top-left on the Windows desktop' },
    { p: 'projection.monitor_offset.1', label: 'Monitor offset Y', type: 'int', unit: 'px' },
    { p: 'projection.duty', label: 'Duty (software ND)', type: 'range', min: 0.05, max: 1, step: 0.01 },
    { p: 'projection.gamma', label: 'Gamma precompensation', type: 'num', step: 0.05, min: 0.1, hint: '1 = off; Measure gamma recommends a value' },
    { p: 'projection.invert', label: 'Invert', type: 'bool', hint: 'bench HDMI lands inverted; decode measures polarity' },
    { p: 'projection.rotate90', label: 'Rotate 90°', type: 'bool', geo: true, hint: 'turns the fringes; must run across the baseline' },
    { p: 'projection.push_twice', label: 'Push twice', type: 'bool', hint: 'the DMD shows HDMI one frame behind' },
    { p: 'projection.settle_s', label: 'Settle', type: 'num', unit: 's', step: 0.01, min: 0 },
    { p: 'projection.min_hold_s', label: 'Min hold', type: 'num', unit: 's', step: 0.01, min: 0, hint: 'pumped hold per pattern' },
    { p: 'projection.warmup_s', label: 'Warm-up', type: 'num', unit: 's', step: 0.05, min: 0 },
    { p: 'projection.scheme', label: 'Scheme', type: 'select', options: ['bw', 'red', 'green', 'blue', 'rgb'], hint: 'mono camera: bw is maximum contrast' },
  ] },
  { title: 'LED currents', custom: 'led' },
  { title: 'Rig geometry', geo: true, fields: [
    { p: 'rig.projector_side', label: 'Projector side (seen from behind the camera)', type: 'segmented', options: ['left', 'right', 'above', 'below'], wide: true },
    { p: 'rig.baseline_mm', label: 'Baseline', type: 'num', unit: 'mm', nullable: true, min: 0, hint: 'camera lens to projector lens' },
    { p: 'rig.standoff_mm', label: 'Projection distance (standoff)', type: 'num', unit: 'mm', nullable: true, min: 0, hint: 'rig to the base / wall' },
    { p: 'rig.throw_ratio', label: 'Throw ratio', type: 'num', step: 0.01, min: 0.1, hint: 'LightCrafter 4500: 1.4 (DLPU011E)' },
  ] },
  { title: 'Decode', fields: [
    { p: 'decode.conf_min', label: 'Modulation floor (conf_min)', type: 'num', step: 0.5, min: 0, unit: 'counts' },
    { p: 'decode.contrast_min', label: 'Contrast floor (contrast_min)', type: 'num', step: 0.5, min: 0, unit: 'counts' },
  ] },
  { title: 'Pattern bases', fields: [
    { p: 'basis:graycode|hybrid.bits', label: 'Gray-code bits', type: 'int', min: 1, max: 12 },
    { p: 'basis:graycode|hybrid.coarse_bits', label: 'Coarse bits', type: 'int', min: 1, max: 12, nullable: true, placeholder: 'all bits',
      hint: 'decode only the N coarsest gray-code planes; single-mirror stripes keep ~1 % contrast through the optics' },
    { p: 'basis:graycode|hybrid.complementary', label: 'Complementary gray code', type: 'bool', hint: 'pattern + inverse per bit (normal-vs-inverse decode); doubles the gray-code frames' },
    { p: 'basis:graycode|hybrid.min_bit_contrast', label: 'Min bit contrast', type: 'num', unit: 'grey levels', step: 0.5, min: 0, nullable: true, placeholder: 'default',
      hint: 'complementary decode: normal-vs-inverse floor per bit' },
    { p: 'basis:phaseshift|hybrid.steps', label: 'Phase steps', type: 'int', min: 3, max: 16 },
    { p: 'basis:phaseshift|hybrid.period_px', label: 'Fringe period', type: 'int', unit: 'DMD px', min: 4 },
    { p: 'basis:*.axis', label: 'Coded axis', type: 'select', options: ['', 'x', 'y'], optionLabels: { '': 'default (x)', x: 'x: vertical stripes', y: 'y: horizontal stripes' }, geo: true,
      hint: 'x codes the projector column; must run across the baseline (like rotate 90°)' },
  ] },
  { title: 'Tolerance', fields: [
    { p: 'tolerance.ch_fail_mm', label: 'CH fail trigger', type: 'num', unit: 'mm', step: 0.05, min: 0, hint: 'Inspect FAILs blobs at or above this height' },
    { p: 'tolerance.dent_fail_um', label: 'Dent fail', type: 'num', unit: 'µm', step: 1, min: 0, off: true, hint: 'not used yet: Inspect judges every blob against the CH trigger' },
    { p: 'tolerance.min_defect_px', label: 'Min defect area (compare)', type: 'int', unit: 'px', min: 1, hint: 'compare to golden only; Inspect uses the Min area on the Defects tab' },
  ] },
];
const FIELD = Object.fromEntries(FORM.flatMap((s) => (s.fields || []).map((f) => [f.p, { ...f, geo: f.geo || s.geo }])));
const GEO_PATHS = Object.values(FIELD).filter((f) => f.geo).map((f) => f.p);
const LED_LIMITS = { burst: { r: 180, g: 255, b: 230 }, continuous: { r: 120, g: 180, b: 160 } };

// 'basis:graycode|hybrid.bits' -> the first basis of those names; 'basis:*.axis' -> every stripe basis
const STRIPE_BASES = ['graycode', 'phaseshift', 'hybrid'];
const basisNames = (s) => (s === '*' ? STRIPE_BASES : s.split('|'));
function findBasis(obj, names) { return (obj && obj.bases || []).find((b) => names.includes(b.name)); }
function getPath(obj, p) {
  if (p.startsWith('basis:')) {
    const [names, key] = p.slice(6).split('.');
    const b = findBasis(obj, basisNames(names));
    return b ? b[key] : undefined;
  }
  return p.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
}
function setPath(obj, p, v) {
  if (p.startsWith('basis:')) {
    const [names, key] = p.slice(6).split('.');
    const targets = names === '*' ? (obj && obj.bases || []).filter((b) => STRIPE_BASES.includes(b.name)) : [findBasis(obj, basisNames(names))].filter(Boolean);
    // blank = the basis default: drop the key rather than write null
    targets.forEach((b) => { if (v == null || v === '') delete b[key]; else b[key] = v; });
    return;
  }
  const ks = p.split('.');
  let o = obj;
  for (let i = 0; i < ks.length - 1; i++) {
    if (o[ks[i]] == null) o[ks[i]] = /^\d+$/.test(ks[i + 1]) ? [] : {};
    o = o[ks[i]];
  }
  o[ks[ks.length - 1]] = v;
}

function buildField(f) {
  const wrap = h('div', { class: 'fld', 'data-path': f.p, style: f.wide ? { gridColumn: '1 / -1' } : null });
  add(wrap, h('div', { class: 'lab' }, h('span', null, f.label), f.unit ? h('span', { class: 'u' }, f.unit) : null));
  let ctl;
  if (f.type === 'bool') {
    ctl = h('label', { class: 'toggle' }, h('input', { type: 'checkbox', 'data-p': f.p }), h('span', { class: 'sw' }), h('span', { class: 'st' }));
  } else if (f.type === 'select') {
    ctl = h('select', { 'data-p': f.p }, f.options.map((o) => h('option', { value: o }, (f.optionLabels && f.optionLabels[o]) || o)));
  } else if (f.type === 'segmented') {
    ctl = h('div', { class: 'seg', 'data-p': f.p }, f.options.map((o) => h('button', { type: 'button', 'data-v': o }, o)));
  } else if (f.type === 'range') {
    ctl = h('div', { class: 'pair' },
      h('input', { type: 'range', min: f.min, max: f.max, step: f.step, 'data-p': f.p }),
      h('input', { type: 'number', min: f.min, max: f.max, step: f.step, 'data-p': f.p }));
  } else {
    ctl = h('input', { type: 'number', step: f.step || (f.type === 'int' ? 1 : 'any'), min: f.min, max: f.max, 'data-p': f.p, placeholder: f.placeholder || (f.nullable ? 'not set' : '') });
  }
  add(wrap, ctl);
  if (f.off) wrap.classList.add('off');
  if (f.hint) add(wrap, h('div', { class: 'hint' }, f.hint));
  return wrap;
}
function buildLed() {
  return h('fieldset', { class: 'fs', id: 'ledFs' },
    h('legend', null, 'LED currents (as set in the LightCrafter GUI)'),
    h('div', { class: 'fields' },
      h('div', { class: 'fld' }, h('div', { class: 'lab' }, h('span', null, 'Record LED currents')),
        h('label', { class: 'toggle' }, h('input', { type: 'checkbox', id: 'ledOn' }), h('span', { class: 'sw' }), h('span', { class: 'st' }))),
      h('div', { class: 'fld' }, h('div', { class: 'lab' }, h('span', null, 'Mode')),
        h('select', { id: 'ledMode' }, h('option', { value: 'continuous' }, 'continuous (average)'), h('option', { value: 'burst' }, 'burst (short)'))),
      ['r', 'g', 'b'].map((c) => h('div', { class: 'fld', 'data-led': c }, h('div', { class: 'lab' }, h('span', null, { r: 'Red', g: 'Green', b: 'Blue' }[c]), h('span', { class: 'u', id: 'ledLim' + c }, '')),
        h('input', { type: 'number', min: 0, step: 1, id: 'led' + c, placeholder: 'not set' })))),
    h('div', { class: 'muted small', style: { marginTop: '6px' } }, 'Thermal limits (rule 3): burst R180 G255 B230 · continuous R120 G180 B160. The engine refuses a preset over the limit.'));
}
function buildForm() {
  const form = $('#presetForm');
  put(form, ...FORM.map((sec) => {
    if (sec.custom === 'led') return buildLed();
    return h('fieldset', { class: 'fs' + (sec.geo ? ' geo' : ''), 'data-sec': sec.title },
      h('legend', null, sec.title, sec.geo ? tag('geometry-bound', 'warn') : null),
      h('div', { class: 'fields' }, sec.fields.map((f) => buildField({ ...f, geo: f.geo || sec.geo }))));
  }));
  form.addEventListener('submit', (e) => e.preventDefault());    // no inline handler: the CSP forbids them
  form.addEventListener('input', onFormInput);
  form.addEventListener('change', onFormInput);
  form.addEventListener('click', (e) => {
    const b = e.target.closest('.seg[data-p] button');
    if (!b) return;
    const seg = b.parentElement;
    setField(seg.dataset.p, b.dataset.v);
  });
}
function readCtl(el) {
  if (el.type === 'checkbox') return el.checked;
  if (el.tagName === 'SELECT') return el.value;
  const s = String(el.value).trim();
  if (s === '') return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : s;
}
function onFormInput(e) {
  const el = e.target;
  if (!S.form) return;
  if (el.dataset && el.dataset.p) {
    setPath(S.form, el.dataset.p, readCtl(el));
    $$(`[data-p="${el.dataset.p}"]`, $('#presetForm')).forEach((o) => { if (o !== el && o.type !== 'checkbox' && o.tagName === 'INPUT') o.value = el.value; });
  } else if (['ledOn', 'ledMode', 'ledr', 'ledg', 'ledb'].includes(el.id)) {
    readLed();
  }
  el.closest('.fld') && el.closest('.fld').classList.remove('bad');
  formChanged();
}
function setField(p, v) {
  if (!S.form) return;
  if (p === 'name' || p === 'geometry_id') S.form[p] = v; else setPath(S.form, p, v);
  fillForm();
}
function readLed() {
  if (!$('#ledOn').checked) { S.form.projection.led = null; return; }
  const led = { mode: $('#ledMode').value };
  for (const c of ['r', 'g', 'b']) { const v = readCtl($('#led' + c)); if (v != null) led[c] = v; }
  S.form.projection.led = led;
}
function fillForm() {
  const f = S.form;
  if (!f) return;
  $$('#presetForm [data-p]').forEach((el) => {
    const v = getPath(f, el.dataset.p);
    const missing = el.dataset.p.startsWith('basis:') && v === undefined && !findBasis(f, basisNames(el.dataset.p.slice(6).split('.')[0]));
    const off = !!(FIELD[el.dataset.p] && FIELD[el.dataset.p].off);
    el.disabled = missing || off;
    if (missing) el.title = "not in this preset's bases";
    else if (off) el.title = 'not used by the engine yet';
    if (el.classList.contains('seg')) {
      $$('button', el).forEach((b) => b.classList.toggle('on', b.dataset.v === v));
    } else if (el.type === 'checkbox') {
      el.checked = !!v;
      const st = el.parentElement.querySelector('.st');
      if (st) st.textContent = v ? 'on' : 'off';
    } else {
      el.value = v == null ? '' : v;
    }
  });
  const led = f.projection && f.projection.led;
  $('#ledOn').checked = !!led;
  $('#ledOn').parentElement.querySelector('.st').textContent = led ? 'on' : 'off';
  $('#ledMode').value = (led && led.mode) || 'continuous';
  for (const c of ['r', 'g', 'b']) $('#led' + c).value = led && led[c] != null ? led[c] : '';
  $('#saveName').value = f.name || '';
  $('#saveGeo').value = f.geometry_id || '';
  formChanged();
}
function isDirty() { return !!S.form && JSON.stringify(S.form) !== JSON.stringify(S.loaded); }
function formChanged() {
  if (!S.form) return;
  $$('#presetForm [data-p]').forEach((el) => {
    if (el.type === 'checkbox') { const st = el.parentElement.querySelector('.st'); if (st) st.textContent = el.checked ? 'on' : 'off'; }
  });
  $('#ledOn').parentElement.querySelector('.st').textContent = $('#ledOn').checked ? 'on' : 'off';
  $$('#presetForm .fld[data-path]').forEach((w) => {
    const p = w.dataset.path;
    w.classList.toggle('changed', JSON.stringify(getPath(S.form, p)) !== JSON.stringify(getPath(S.loaded, p)));
  });
  // LED: live check against the thermal limits (the server validates too)
  const led = S.form.projection && S.form.projection.led;
  const lim = LED_LIMITS[(led && led.mode) || $('#ledMode').value] || LED_LIMITS.continuous;
  for (const c of ['r', 'g', 'b']) {
    $('#ledLim' + c).textContent = `max ${lim[c]}`;
    const v = led ? led[c] : null;
    $('#led' + c).classList.toggle('invalid', v != null && (typeof v !== 'number' || v < 0 || v > lim[c]));
    $('#led' + c).disabled = !led;
  }
  $('#ledMode').disabled = !led;
  $('#dirtyBadge').hidden = !isDirty();
  renderDerived();
  renderGeoSuggest();
  renderNameHint();
}
function renderDerived() {
  const f = S.form;
  if (!f) return;
  const r = f.rig || {};
  const b = num(r.baseline_mm), s = num(r.standoff_mm), t = num(r.throw_ratio);
  const nominal = b > 0 && s > 0 && t > 0 ? (s * s) / (t * 912 * b) : null;
  const angle = b > 0 && s > 0 ? (Math.atan2(b, s) * 180) / Math.PI : null;
  const width = s > 0 && t > 0 ? s / t : null;
  const lr = r.projector_side === 'left' || r.projector_side === 'right';
  const trig = f.tolerance && num(f.tolerance.ch_fail_mm);
  const mock = S.status && S.status.mock;
  put($('#derived'), ...kv([
    ['Nominal height per relief px', nominal, 'mm/px (estimate)', 3],
    ['Relief px per mm', nominal ? 1 / nominal : null, 'px/mm', 3],
    ['Triangulation angle', angle, '°', 1],
    ['Projected field width', width, 'mm', 0],
    ['DMD column on the wall', width ? width / 912 : null, 'mm', 3],
    [`${trig > 0 ? fmt(trig, 2) : '?'} mm trigger ≈`, nominal && trig > 0 ? trig / nominal : null, 'relief px', 2],
    ['Depth moves fringes along', h('span', null, lr ? 'camera X: fringes VERTICAL' : 'camera Y: fringes HORIZONTAL'), ''],
    ['Effective geometry id', h('span', null, (mock ? 'mock-' : '') + (f.geometry_id || '')), ''],
  ]).childNodes);
  $('#derived').after($('#derivedNote') || h('p', { id: 'derivedNote', class: 'muted small' }, 'Tape-measure estimates, for orientation only. Calibrate to get mm.'));
}
function suggestGeo(g) {
  const m = /^(.*?)(\d+)$/.exec(g || '');
  return m ? m[1] + (Number(m[2]) + 1) : (g || 'geometry') + '_v2';
}
function renderGeoSuggest() {
  const box = $('#geoSuggest');
  put(box);
  if (!S.form || !S.loaded) return;
  const changed = GEO_PATHS.filter((p) => JSON.stringify(getPath(S.form, p)) !== JSON.stringify(getPath(S.loaded, p)));
  if (!changed.length) return;
  if (S.form.geometry_id !== S.loaded.geometry_id) {
    add(box, h('div', { class: 'alert ok' }, h('b', null, 'New geometry id'),
      `Geometry-bound settings changed and the geometry_id is now ${S.form.geometry_id}. Capture a new base (and calibrate) after loading it.`));
    return;
  }
  const sug = suggestGeo(S.form.geometry_id);
  add(box, h('div', { class: 'alert warn' }, h('b', null, 'Geometry-bound settings changed'),
    h('div', null, `${changed.map((p) => (FIELD[p] ? FIELD[p].label : p)).join(', ')} changed. Bases and calibrations are only valid at the geometry they were captured at, so this should be a new geometry_id.`),
    h('div', { class: 'row' }, h('button', { type: 'button', onclick: () => setField('geometry_id', sug) }, `Use ${sug}`))));
}
function renderNameHint() {
  const name = S.form && S.form.name;
  const builtin = name && S.builtins.includes(name);
  const exists = name && S.presets.includes(name);
  $('#saveName').classList.toggle('invalid', !!(builtin || (exists && !$('#saveOverwrite').checked && isDirty())));
  $('#saveName').title = builtin ? 'built-in presets are read-only: save under a new name'
    : (exists ? 'a user preset with this name exists: pick a new name or tick replace' : '');
}
function clearInvalid() { $$('#presetForm .fld.bad').forEach((e) => e.classList.remove('bad')); $$('#saveName, #saveGeo, #ledFs').forEach((e) => e.classList.remove('invalid')); }
function markInvalid(msg) {
  const re = /\b(camera|projection|rig|decode|tolerance)\.([a-z_0-9]+)/g;
  let m;
  while ((m = re.exec(msg))) $$(`#presetForm .fld[data-path^="${m[1]}.${m[2]}"]`).forEach((f) => f.classList.add('bad'));
  const ch = /LED ([RGB])=/.exec(msg);
  if (ch) $$(`#ledFs .fld[data-led="${ch[1].toLowerCase()}"]`).forEach((f) => f.classList.add('bad'));
  else if (/\bLED\b|led[.\s]/.test(msg)) $$('#ledFs .fld').forEach((f) => f.classList.add('bad'));
  if (/preset\.name|exists|built-in/.test(msg)) $('#saveName').classList.add('invalid');
  if (/geometry_id/.test(msg)) $('#saveGeo').classList.add('invalid');
  if (/basis|bases/.test(msg)) $$('#presetForm .fld[data-path^="basis:"]').forEach((f) => f.classList.add('bad'));
}

async function refreshPresets(select) {
  try {
    const r = await api('/api/presets');
    S.presets = r.presets;
    S.builtins = (r.tiers || []).filter((t) => t.tier === 'builtin').map((t) => t.name);
    const sel = $('#presetSel');
    const keep = select || sel.value || r.active;
    put(sel, ...r.presets.map((n) => h('option', { value: n }, n + (n === r.active ? '  (active)' : ''))));
    if (r.presets.includes(keep)) sel.value = keep;
    $('#presetDir').textContent = r.dir;
    renderNameHint();
    renderPresetLib(r);
  } catch (e) { toast('could not list presets', 'bad', e.message); }
}
/** Preset library: both tiers, select / delete user presets, add from any preset. */
function renderPresetLib(r) {
  const tiers = r.tiers || [];
  const base = $('#paBase');
  const keep = base.value;
  put(base, ...r.presets.map((n) => h('option', { value: n }, n)));
  if (r.presets.includes(keep)) base.value = keep;
  put($('#presetTable'), h('div', { class: 'tablewrap' }, h('table', { class: 't' },
    h('thead', null, h('tr', null, ...['name', 'tier', 'geometry', 'stereo cal', 'description', ''].map((t) => h('th', null, t)))),
    h('tbody', null, ...tiers.map((t) => h('tr', null,
      h('td', null, t.name + (t.active ? ' (selected)' : '') + (t.name === r.active ? ' (loaded)' : '')),
      h('td', null, t.tier === 'builtin' ? 'built-in (read-only)' : 'user'),
      h('td', null, t.geometry_id || '?'),
      h('td', null, t.calibrated ? 'yes' : 'none'),
      h('td', { class: 'small' }, t.valid === false ? `INVALID: ${t.error}` : (t.description || '')),
      h('td', null,
        h('button', { class: 'small', disabled: t.active || t.valid === false, onclick: (e) => selectPreset(t.name, e.currentTarget) }, 'Select'),
        t.tier === 'user' ? h('button', { class: 'small', disabled: t.active, onclick: (e) => deletePreset(t.name, e.currentTarget) }, 'Delete') : null)))))));
}
function parseOverrides(text) {
  const out = {};
  for (const part of String(text || '').split(',').map((x) => x.trim()).filter(Boolean)) {
    const i = part.indexOf('=');
    if (i < 1) throw new Error(`override "${part}" is not key=value`);
    out[part.slice(0, i).trim()] = part.slice(i + 1).trim();
  }
  return out;
}
async function addPreset() {
  const out = $('#presetLibOut');
  let overrides;
  try { overrides = parseOverrides($('#paSet').value); } catch (e) { put(out, h('div', { class: 'alert bad' }, e.message)); return; }
  const args = { name: $('#paName').value.trim(), base: $('#paBase').value, overrides, description: $('#paDesc').value.trim() };
  if ($('#paGeo').value.trim()) args.geometry_id = $('#paGeo').value.trim();
  const job = await runJob('preset_add', args, { btn: $('#paAdd'), out });
  if (!done(job)) return;
  add(out, h('div', { class: 'alert ok' }, h('b', null, `Added user preset ${job.result.name}`), job.result.note || ''), warningsList(job.result.warnings));
  refreshPresets();
}
async function selectPreset(name, btn) {
  const out = $('#presetLibOut');
  const job = await runJob('preset_select', { name }, { btn, out });
  if (!done(job)) return;
  add(out, h('div', { class: 'alert ok' }, h('b', null, `Selected and loaded ${name}`), 'used when no preset is named; a station config\'s own preset still wins'));
  await refreshStatus();
  refreshPresets(name);
}
async function deletePreset(name, btn) {
  const reason = prompt(`Delete user preset "${name}"?\n\nIt moves to the library trash (recoverable). Reason:`);
  if (reason == null) return;
  if (!reason.trim()) { toast('a reason is required', 'bad'); return; }
  const out = $('#presetLibOut');
  const job = await runJob('preset_delete', { name, reason: reason.trim(), confirm: true }, { btn, out });
  if (!done(job)) return;
  add(out, h('div', { class: 'alert ok' }, h('b', null, `Moved ${name} to the trash`), String(job.result.trashed || '')));
  refreshPresets();
}
async function openEditor(name) {
  if (!name) return;
  if (isDirty() && S.loadedName !== name && !confirm('Discard unsaved edits in the editor?')) return;
  try {
    const d = await api('/api/presets/' + encodeURIComponent(name));
    S.form = clone(d.preset);
    S.loaded = clone(d.preset);
    S.loadedName = name;
    $('#editorName').textContent = name + (d.active ? ' (active)' : '');
    put($('#presetWarn'), ...(d.warnings && d.warnings.length ? [h('div', { class: 'alert warn' }, h('b', null, 'Preset warnings'), warningsList(d.warnings))] : []));
    clearInvalid();
    fillForm();
    put($('#saveOut'));
  } catch (e) {
    put($('#presetWarn'), h('div', { class: 'alert bad' }, h('b', null, `Could not open ${name}`), h('div', { class: 'msg' }, e.message)));
  }
}
async function loadPreset(name, out = $('#presetLoadOut'), btn = $('#presetLoad')) {
  const job = await runJob('load_preset', { name }, { btn, out });
  if (!done(job)) return job;
  const p = job.result;
  add(out, h('div', { class: 'alert ok' }, h('b', null, `Active preset: ${p.name}`),
    `geometry ${p.geometry_id} · exposure ${p.camera.exposure_us} µs · average ${p.camera.average} · calibration ${p.calibration ? p.calibration.id : 'none (px)'}`),
    warningsList(p.warnings));
  await refreshStatus();
  refreshPresets(name);
  refreshBases(); refreshObjects();
  return job;
}
async function savePreset(andLoad) {
  const out = $('#saveOut');
  put(out);
  clearInvalid();
  if (!S.form) { add(out, h('div', { class: 'alert warn' }, 'Open a preset in the editor first.')); return; }
  S.form.name = $('#saveName').value.trim();
  S.form.geometry_id = $('#saveGeo').value.trim();
  if (S.builtins.includes(S.form.name)) {
    add(out, h('div', { class: 'alert bad' }, h('b', null, `"${S.form.name}" is a built-in preset (read-only)`), 'Save your edits as a user preset under a new name.'));
    $('#saveName').classList.add('invalid');
    return;
  }
  const overwrite = $('#saveOverwrite').checked;
  if (overwrite && S.presets.includes(S.form.name) &&
      !confirm(`Replace user preset "${S.form.name}"?\n\nThe old version moves to the library trash. Scans already captured keep their recorded settings, but everything loaded from this preset later uses the new settings.`)) return;
  const btns = [$('#saveBtn'), $('#saveLoadBtn')];
  btns.forEach((b) => { b.disabled = true; });
  try {
    // a USER preset in this library, copied from the preset the editor opened (built-ins
    // are never written)
    const r = await api('/api/presets', { body: { preset: S.form, base: S.loadedName, overwrite } });
    add(out, h('div', { class: 'alert ok' }, h('b', null, `Saved user preset ${r.saved}`), h('div', { class: 'msg' }, r.path), r.note || ''), warningsList(r.warnings || []));
    S.loaded = clone(S.form);
    S.loadedName = r.saved;
    $('#editorName').textContent = r.saved;
    $('#saveOverwrite').checked = false;
    await refreshPresets(r.saved);
    formChanged();
    if (andLoad) await loadPreset(r.saved, out, $('#saveLoadBtn'));
    else if (r.saved === activePresetName()) {
      // the file changed but the engine still holds the settings it loaded earlier
      add(out, h('div', { class: 'alert warn' }, h('b', null, `"${r.saved}" is the active preset: scans still use the settings loaded earlier`),
        h('div', null, 'The file on disk changed, the engine did not. Load it to make the new settings active.'),
        h('div', { class: 'row' }, h('button', { class: 'primary small', onclick: (e) => loadPreset(r.saved, out, e.currentTarget) }, `Load ${r.saved} now`))));
    }
  } catch (e) {
    add(out, h('div', { class: 'alert bad' }, h('b', null, 'Not saved: the engine rejected the preset'), h('div', { class: 'msg' }, e.message)));
    markInvalid(e.message);
  } finally {
    btns.forEach((b) => { b.disabled = false; });
  }
}
function deepMerge(t, p) {
  for (const [k, v] of Object.entries(p || {})) {
    if (v && typeof v === 'object' && !Array.isArray(v) && t[k] && typeof t[k] === 'object' && !Array.isArray(t[k])) deepMerge(t[k], v);
    else t[k] = clone(v);
  }
  return t;
}
/** Put a recommended preset patch into the Setup form (never saved automatically). */
async function applyPatch(patch, source) {
  const active = activePresetName();
  if (!S.form || (!isDirty() && S.loadedName !== active)) await openEditor(active);
  if (!S.form) return;
  deepMerge(S.form, patch);
  if (S.form.name === S.loadedName) S.form.name = `${S.loadedName}_tuned`;
  fillForm();
  go('setup');
  put($('#saveOut'), h('div', { class: 'alert info' }, h('b', null, `Applied from ${source}`),
    h('div', { class: 'msg' }, JSON.stringify(patch)),
    h('div', { class: 'hint' }, 'Review the highlighted fields, then Save & load. Nothing changes on the rig until a preset is loaded.')));
}
function showSetup() {
  refreshPresets().then(() => { if (!S.form && activePresetName()) openEditor(activePresetName()); });
}

// =================================================================== 2 FOCUS / SIGNAL
const Live = { polling: false, frame: -1, loading: false };
function liveSettings() {
  return { pattern: $('#livePattern').value, colour: $('#liveColour').value, level: Number($('#liveLevel').value), period_px: Number($('#livePeriod').value) };
}
async function liveStart() {
  try {
    await api('/api/live/start', { body: liveSettings() });
    pollLive();
  } catch (e) { toast('live view failed to start', 'bad', e.message); }
}
async function liveStop() {
  try { await api('/api/live/stop', { body: {} }); } catch (e) { toast('could not stop live view', 'bad', e.message); }
}
/** Leaving the Focus tab stops the live view: nobody watches it elsewhere, and it holds the rig lock. */
function stopLiveOnLeave() {
  const st = S.status;
  const pending = st && ((st.live && st.live.running) || [st.job, ...(st.queued || [])].some((j) => j && j.verb === 'live_view'));
  if (pending) api('/api/live/stop', { body: {} }).then(() => toast('live view stopped', 'ok', 'it stops when you leave the Focus tab')).catch(() => {});
}
let liveTimer = null;
function liveSettingsChanged() {
  $('#liveLevelV').textContent = $('#liveLevel').value;
  LS.set('live', liveSettings());
  clearTimeout(liveTimer);
  liveTimer = setTimeout(() => {
    if (S.status && S.status.live && S.status.live.running) api('/api/live/settings', { body: liveSettings() }).catch(() => {});
  }, 120);
}
async function pollLive() {
  if (Live.polling) return;
  Live.polling = true;
  let idle = 0;
  try {
    while (Live.polling && curTab === 'focus') {
      let st;
      try { st = await api('/api/live/state'); } catch (e) { await sleep(1000); continue; }
      renderLive(st);
      const queued = S.status && (S.status.queued || []).some((j) => j.verb === 'live_view');
      if (st.running || queued) idle = 0; else if (++idle > 15) break;
      await sleep(st.running ? 180 : 400);
    }
  } finally { Live.polling = false; }
}
function renderLive(st) {
  const m = st.metrics || {};
  const f = m.focus, peak = st.focus_peak;
  $('#focusVal').textContent = f != null ? fmt(f, f >= 100 ? 0 : 1) : '—';
  $('#focusPeak').textContent = peak ? fmt(peak, peak >= 100 ? 0 : 1) : '—';
  const frac = f != null && peak > 0 ? Math.min(1, f / peak) : 0;
  const bar = $('#focusBar');
  bar.style.width = (frac * 100).toFixed(1) + '%';
  bar.style.background = frac > 0.9 ? cssVar('--ok') : frac > 0.7 ? cssVar('--warn') : cssVar('--bad');
  put($('#liveStats'), ...kv([
    ['State', h('span', null, st.running ? tag('running', 'ok') : tag('stopped'))],
    ['Frame', st.frame, ''],
    ['Rate', st.fps, 'fps', 1],
    ['Mean (lit)', m.mean, 'counts', 1],
    ['Saturated', m.saturated_pct, '% of lit px', 2],
    ['Lit area', m.lit_pct, '% of frame', 1],
    ['Pattern', h('span', null, `${st.settings.pattern} · ${st.settings.colour} · level ${st.settings.level}`)],
  ]).childNodes);
  if (st.error) add($('#liveStats'), h('dt', null, 'Error'), h('dd', { style: { color: cssVar('--bad') } }, st.error));
  if (st.has_image && st.frame !== Live.frame && !Live.loading) {
    Live.loading = true;
    Live.frame = st.frame;
    const img = new Image();
    img.onload = () => { $('#liveImg').src = img.src; Live.loading = false; };
    img.onerror = () => { Live.loading = false; };
    img.src = `/api/live.jpg?f=${st.frame}&t=${Date.now()}`;
  }
  if (!st.running && st.stopped_because) add($('#liveStats'), h('dt', null, 'Stopped'), h('dd', null, st.stopped_because));
  $('#liveEmpty').hidden = !!st.has_image;
  $('#liveEmpty').textContent = st.running ? 'Starting…' : 'Live view is off.';
  $('#liveStart').disabled = st.running;
  $('#liveStop').disabled = !st.running;
}
function showFocus() {
  const saved = LS.get('live', null);
  if (saved) {
    $('#livePattern').value = saved.pattern || 'flood';
    $('#liveColour').value = saved.colour || 'white';
    if (saved.level != null) $('#liveLevel').value = saved.level;
    if (saved.period_px) $('#livePeriod').value = saved.period_px;
  } else if (S.status && S.status.preset) {
    $('#liveLevel').value = Math.round(255 * S.status.preset.projection.duty);
  }
  $('#liveLevelV').textContent = $('#liveLevel').value;
  api('/api/live/state').then((st) => { renderLive(st); if (st.running) pollLive(); }).catch(() => {});
}

async function checkSignal() {
  const out = $('#sigOut');
  const job = await runJob('check_signal', {}, { btn: $('#sigBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  const qcls = { excellent: 'ok', good: 'ok', usable: 'warn', marginal: 'bad', unusable: 'bad' }[r.quality] || '';
  add(out, 
    h('div', { class: 'row' }, h('span', { class: 'verdict ' + (r.ok ? 'ok' : 'bad') }, r.ok ? 'READY TO SCAN' : 'NOT READY'), tag(r.quality, qcls)),
    h('div', { class: 'tiles2' },
      metric('gc_conf', r.gc_conf, 'counts', qcls, 'fringe modulation, median over valid px'),
      metric('valid', r.valid_pct, '%', r.valid_pct > 20 ? 'ok' : r.valid_pct > 1 ? 'warn' : 'bad', 'camera px that decode'),
      metric('focus ratio', r.focus_ratio, '', r.focus_ratio >= 0.7 ? 'ok' : r.focus_ratio >= 0.4 ? 'warn' : 'bad', r.focus_hint, 3),
      metric('coded axis', (r.coded_axis || '?').toUpperCase(), '', r.axis_ok === true ? 'ok' : r.axis_ok === false ? 'bad' : '',
        r.axis_ok === true ? 'OK: fringes cross the baseline' : r.axis_ok === false ? 'WRONG: toggle rotate90' : 'not determined'),
      metric('contrast', r.contrast_mean, 'counts', '', 'white − black, mean', 1)),
    warningsList(r.warnings),
    h('div', { class: 'muted small' }, 'gc_conf bands: below 8 unusable · 8–18 marginal · 18–30 usable · 30–120 good · 120 and up excellent'));
  if (r.axis_ok === false && S.status.preset) {
    const rot = !!S.status.preset.projection.rotate90;
    add(out, h('div', { class: 'row' }, h('button', { onclick: () => applyPatch({ projection: { rotate90: !rot } }, 'check signal (coded axis)') }, `Set rotate90 = ${!rot} in the Setup form`)));
  }
}

async function measureGamma() {
  const out = $('#gammaOut');
  const job = await runJob('measure_gamma', { n_levels: Number($('#gammaN').value) }, { btn: $('#gammaBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  const cv = h('canvas', { class: 'plot' });
  const res = r.residual_gamma;
  add(out, h('div', { class: 'tiles2' }, metric('gamma', r.gamma, '', '', 'projector vs the value sent over HDMI (the precomp to use)', 3),
    metric('residual gamma', res, '', res == null ? '' : Math.abs(res - 1) < 0.05 ? 'ok' : 'warn',
      res == null ? 'not measured' : `the preset chain as driven now${Math.abs(res - 1) < 0.05 ? ': linear' : ' (1 = linear)'}`, 3),
    metric('R²', r.r2, '', r.r2 > 0.99 ? 'ok' : 'warn', 'fit quality', 4)),
    cv, warningsList(r.warnings));
  const lv = r.levels || [], resp = r.response || [];
  let mx = 1;
  lv.forEach((v) => { if (v > mx) mx = v; });
  const fx = Array.from({ length: 51 }, (_, i) => i / 50);
  plotXY(cv, [
    { x: lv.map((v) => v / mx), y: resp, kind: 'points', color: cssVar('--accent'), label: 'measured' },
    { x: fx, y: fx.map((x) => Math.pow(x, r.gamma)), color: cssVar('--warn'), label: `x^${fmt(r.gamma, 2)}` },
  ], { xLabel: 'drive level (normalised)', yLabel: 'camera response (normalised)' });
  if (r.recommended && Object.keys(r.recommended).length) {
    add(out, h('div', { class: 'row' }, h('span', { class: 'mono small' }, JSON.stringify(r.recommended)),
      h('button', { onclick: () => applyPatch(r.recommended, 'measure gamma') }, 'Apply to Setup form')));
  }
  if (r.path) add(out, h('div', { class: 'small' }, 'report: ', fileLink(r.path)));
}

async function runTune() {
  const out = $('#tuneOut');
  const args = {};
  const ex = $('#tuneExp').value.trim(), du = $('#tuneDuty').value.trim();
  if (ex) args.exposures = ex;
  if (du) args.duties = du;
  const job = await runJob('tune', args, { btn: $('#tuneBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  const rec = r.recommended || {};
  const recExp = rec.camera && rec.camera.exposure_us;
  const recDuty = rec.projection && rec.projection.duty;
  const steps = r.steps || [];
  const isRec = (s) => s.exposure_us === recExp && (recDuty == null || s.duty == null || s.duty === recDuty);
  const cv = h('canvas', { class: 'plot' });
  add(out, 
    h('div', { class: 'row' }, h('span', { class: 'verdict ' + (r.ok ? 'ok' : 'bad') }, r.ok ? 'RECOMMENDATION READY' : 'NO GOOD SETTING'),
      Object.keys(rec).length ? h('button', { class: 'primary', onclick: () => applyPatch(rec, 'scan assistant') }, 'Apply recommended to Setup form') : null),
    Object.keys(rec).length ? h('div', { class: 'mono small' }, 'recommended patch: ' + JSON.stringify(rec)) : null,
    r.advice && r.advice.length ? h('ul', { class: 'plain' }, r.advice.map((a) => h('li', null, a))) : null,
    table([
      { k: 'exposure_us', label: 'Exposure µs', n: 1 },
      { k: 'duty', label: 'Duty', n: 1, f: (v) => (v == null ? 'preset' : fmt(v, 2)) },
      { k: 'gc_conf', label: 'gc_conf', n: 1 },
      { k: 'quality', label: 'Quality', f: (v) => (v ? tag(v, { excellent: 'ok', good: 'ok', usable: 'warn' }[v] || 'bad') : '—') },
      { k: 'valid_pct', label: 'Valid %', n: 1 },
      { k: 'saturated_pct', label: 'Saturated %', n: 1 },
      { k: 'focus_ratio', label: 'Focus ratio', n: 1, f: (v) => fmt(v, 3) },
      { k: 'coded_axis', label: 'Axis', f: (v, s) => h('span', null, (v || '?').toUpperCase(), ' ', s.axis_ok === false ? tag('wrong', 'bad') : s.axis_ok ? tag('ok', 'ok') : '') },
      { k: 'frame_locked', label: 'Frame-locked', f: (v) => (v == null ? '—' : v ? tag('locked', 'ok') : tag('not locked', 'warn')) },
    ], steps, { rowClass: (s) => (isRec(s) ? 'hl' : '') }),
    steps.some((s) => s.frame_locked != null) ? h('div', { class: 'muted small' },
      `Frame-locked: the exposure is a whole number of HDMI video frames (${fmt(r.video_frame_us, 1)} µs each), so every frame sees the same DMD duty. Unlocked exposures can beat against the frame rate.`) : null,
    cv,
    r.path ? h('div', { class: 'small' }, 'report: ', fileLink(r.path)) : null);
  const duties = [...new Set(steps.map((s) => s.duty))];
  const colors = [cssVar('--accent'), cssVar('--ok'), cssVar('--warn'), cssVar('--bad'), cssVar('--info')];
  const series = [];
  if (duties.length > 1) {
    duties.forEach((d, i) => {
      const ss = steps.filter((s) => s.duty === d);
      series.push({ x: ss.map((s) => s.exposure_us), y: ss.map((s) => s.gc_conf), color: colors[i % colors.length], label: `gc_conf, duty ${fmt(d, 2)}` });
    });
  } else {
    series.push({ x: steps.map((s) => s.exposure_us), y: steps.map((s) => s.gc_conf), color: colors[0], label: 'gc_conf (counts)' });
    series.push({ x: steps.map((s) => s.exposure_us), y: steps.map((s) => s.valid_pct), color: colors[1], label: 'valid %' });
    series.push({ x: steps.map((s) => s.exposure_us), y: steps.map((s) => s.saturated_pct), color: colors[3], label: 'saturated %' });
  }
  plotXY(cv, series, { logx: true, xLabel: 'exposure (µs, log)', yLabel: 'counts / %', y0: 0 });
}

// =================================================================== 3 BASES
function renderBases() {
  const box = $('#basesList');
  const all = $('#basesAll').checked;
  const list = (all ? S.basesAll : S.bases).slice().reverse();
  const act = activeBaseId();
  const latest = S.status && S.status.latest_base && S.status.latest_base.id;
  const note = listNote(S.basesNote);
  put(box, ...[note].filter(Boolean), ...(list.length ? list.map((b) => baseTile(b, act, latest))
    : [h('div', { class: 'empty' }, `No bases for ${S.geo || 'this geometry'} yet. Capture one with the scene empty.`)]));
}
function baseTile(b, act, latest) {
  const m = b.meta || {};
  const rej = m.rejected;
  const other = b.geometry_id !== S.geo;
  return h('div', { class: 'tile' + (b.id === act ? ' active' : '') + (rej ? ' rejected' : '') },
    thumb(b.preview, `base ${b.label}`),
    h('div', { class: 'body' },
      h('div', { class: 'title' }, b.label,
        b.id === act ? tag('active', 'ok') : null,
        b.id === latest && !act ? tag('latest', 'info') : null,
        rej ? tag('rejected', 'bad') : null,
        other ? tag('other geometry', 'warn') : null),
      h('div', { class: 'mono small muted' }, b.id),
      h('div', { class: 'small' }, `${shortTime(b.created)} · valid ${fmt(m.valid_pct, 1)} %`),
      other ? h('div', { class: 'small mono' }, b.geometry_id) : null,
      rej ? h('div', { class: 'small', style: { color: cssVar('--bad') } }, `rejected: ${rej}`) : null),
    h('div', { class: 'acts' },
      !rej && !other ? h('button', { class: 'small', onclick: () => setActiveBase(b.id === act ? null : b.id) }, b.id === act ? 'Unset' : 'Use for scans') : null,
      !rej ? h('button', { class: 'small', onclick: () => rejectBase(b) }, 'Reject…') : null,
      h('button', { class: 'small danger', onclick: () => deleteEntry('bases', b) }, 'Delete…')));
}
async function captureBase() {
  if (!confirm(`Is the scene EMPTY and nothing moved?\n\nThis base becomes the datum for every object scan at ${S.geo}. Capture it with no object in view, and with the camera, projector and focus exactly as they will be for the objects.`)) return;
  const out = $('#baseOut');
  const job = await runJob('capture_base', { label: $('#baseLabel').value }, { btn: $('#baseCapture'), out });
  if (!done(job)) return;
  const b = job.result;
  const m = b.meta || {};
  const ws = b.warnings || [];
  // an auto-rejected base (almost nothing decoded) is kept but never used: say so loudly
  add(out, h('div', { class: 'alert ' + (m.rejected ? 'bad' : ws.length ? 'warn' : 'ok') },
    h('b', null, m.rejected ? `Base "${b.label}" captured but auto-REJECTED: scans will not use it` : `Base "${b.label}" captured`),
    `valid ${fmt(m.valid_pct, 1)} % · ${b.geometry_id} · decode ${(m.basis_names || []).join(' + ')}`),
  warningsList(ws));
  refreshBases();
}
async function rejectBase(b) {
  const reason = prompt(`Why is base "${b.label}" unusable? (for example: bumped the tripod, object in view)`);
  if (!reason) return;
  const job = await runJob('reject_base', { base_id: b.id, reason }, {});
  if (done(job)) refreshBases();
}

// =================================================================== 4 3D SCAN
function renderObjects() {
  const box = $('#objList');
  const note = listNote(S.objectsNote);
  put(box, ...[note].filter(Boolean), ...(S.objects.length ? S.objects.map(objItem)
    : [h('div', { class: 'empty' }, 'No scans at this geometry yet.')]));
}
function objItem(o) {
  const m = o.meta || {};
  const kd = o.known_dims;
  const ws = m.warnings || [];
  return h('div', { class: 'item' + (o.id === S.cloudId ? ' sel' : '') },
    thumb(o.preview, `relief ${o.label}`),
    h('div', null,
      h('div', { class: 'title' }, o.label, kd ? tag('known dims', 'info') : null, m.zone ? tag('zone ' + m.zone) : null,
        ws.length ? h('span', { title: ws.join('\n') }, tag(`${ws.length} warning${ws.length > 1 ? 's' : ''}`, 'warn')) : null),
      h('div', { class: 'meta mono' }, o.id),
      h('div', { class: 'meta' }, `${shortTime(o.created)} · valid ${fmt(m.valid_pct, 1)} % · overlap ${fmt(m.overlap_px, 0)} px · base ${o.base_id}`),
      kd ? h('div', { class: 'meta' }, 'known: ' + Object.entries(kd).map(([k, v]) => `${k} ${v}`).join(', ') + ' mm') : null,
      h('div', { class: 'acts' },
        h('button', { class: 'small', onclick: () => showCloud(o) }, '3D'),
        h('button', { class: 'small', onclick: () => pickProfile(o) }, 'Profile'),
        h('button', { class: 'small', onclick: (e) => reconstruct(o, 'ply', e.currentTarget) }, '.ply'),
        h('button', { class: 'small danger', onclick: () => deleteEntry('objects', o) }, 'Delete…'))));
}
/**
 * Zone text as sent to the engine: pinned to the spacing of the grid this page shows
 * ('E6' -> 'E6@96'). Without it the engine uses the LAST projected grid, which another
 * browser, the CLI or an agent may have projected at a different spacing.
 */
function zoneForEngine(zone, grid) {
  const raw = String(zone || '').trim();
  // Surfer atlas regions ('atlas:part/part') are names, not cells: sent as typed, never pinned
  if (/^atlas:/i.test(raw)) return 'atlas:' + raw.slice(6).trim();
  const z = raw.toUpperCase();
  if (!z || z.includes('@')) return z;
  const sp = grid && grid.grid && grid.grid.spacing_px;
  return sp ? `${z}@${sp}` : z;
}
async function captureObject(o) {
  const kd = {};
  for (const k of ['H', 'W', 'L']) if (o[k] !== '' && o[k] != null && Number.isFinite(Number(o[k]))) kd[k] = Number(o[k]);
  const args = { label: o.label, base_id: o.base_id || null, known_dims: Object.keys(kd).length ? kd : null, zone: o.zone ? zoneForEngine(o.zone, S.grid) : null };
  const job = await runJob('capture_object', args, { btn: o.btn, out: o.out });
  if (!done(job)) return job;
  const s = job.result;
  const m = s.meta || {};
  const ws = m.warnings || [];
  add(o.out, h('div', { class: 'alert ' + (ws.length ? 'warn' : 'ok') }, h('b', null, `Captured "${s.label}"`),
    `valid ${fmt(m.valid_pct, 1)} % · overlap with the base ${fmt(m.overlap_px, 0)} px · datum ${fmt(m.datum_px, 3)} px${m.zone ? ' · zone ' + m.zone : ''}`),
    warningsList(ws),
    h('div', { class: 'row' }, h('button', { onclick: () => { go('scan3d'); showCloud(s); } }, 'View in 3D'),
      fileUrl(s.preview) ? h('button', { class: 'ghost', onclick: () => openImage(s.preview, `relief ${s.label}`) }, 'Relief preview') : null),
    h('div', { class: 'muted small' }, 'Measure, Defects, bank and profile pickers now point at this scan.'));
  S.selectScan = s.id;          // the next list refresh points every scan picker (but the golden) here
  await refreshObjects();
  return job;
}
async function reconstruct(o, fmtName, btn) {
  const job = await runJob('reconstruct', { object_id: o.id, fmt: fmtName }, { btn: btn || null });
  if (!done(job)) return;
  const rel = job.result;
  put($('#cloudLinks'), h('span', { class: 'small' }, `${o.label}: `), fileLink(rel, `download ${baseName(rel)}`, true));
  toast(`${fmtName.toUpperCase()} written`, 'ok', absPath(rel));
}

// ---------- point-cloud viewer (canvas, orthographic, z-buffered points)
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
const LUT32 = (() => {
  const lut = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (VIRIDIS.length - 1), k = Math.min(VIRIDIS.length - 2, Math.floor(t)), f = t - k;
    const c = VIRIDIS[k].map((v, j) => Math.round(v + (VIRIDIS[k + 1][j] - v) * f));
    lut[i] = (255 << 24) | (c[2] << 16) | (c[1] << 8) | c[0];
  }
  return lut;
})();
function color32(css, fallback = 0xff110d0a) {
  const m = /^#([0-9a-f]{6})$/i.exec(css || '');
  if (!m) return fallback;
  const n = parseInt(m[1], 16);
  return ((255 << 24) | ((n & 0xff) << 16) | (n & 0xff00) | ((n >> 16) & 0xff)) >>> 0;
}
class CloudView {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.n = 0;
    this.zexMul = 1;
    this.psize = 2;
    this.reset(false);
    let drag = null;
    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 };
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) { this.panX += dx; this.panY += dy; } else {
        this.yaw += dx * 0.01;
        this.pitch = Math.max(-Math.PI / 2, Math.min(0.15, this.pitch + dy * 0.01));
      }
      this.draw();
    });
    const end = () => { drag = null; };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoom = Math.max(0.1, Math.min(60, this.zoom * Math.exp(-e.deltaY * 0.0015)));
      this.draw();
    }, { passive: false });
    canvas.addEventListener('dblclick', () => this.reset());
    if (window.ResizeObserver) new ResizeObserver(() => this.draw()).observe(canvas);
  }
  reset(draw = true) {
    this.yaw = -0.45; this.pitch = -0.95; this.zoom = 1; this.panX = 0; this.panY = 0;
    if (draw) this.draw();
  }
  setData(d) {
    this.meta = d;
    const P = (d && d.points) || [];
    const n = P.length;
    this.n = n;
    if (!n) { this.draw(); return; }
    const x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
    for (let i = 0; i < n; i++) { x[i] = P[i][0]; y[i] = P[i][1]; z[i] = P[i][2]; }
    const lo = d.bounds.min, hi = d.bounds.max;
    this.cx = (lo[0] + hi[0]) / 2;
    this.cy = (lo[1] + hi[1]) / 2;
    // a part is often ~1 % of the points: a tight percentile window would clip it away
    const zs = Float32Array.from(z).sort();
    const q = (p) => zs[Math.min(n - 1, Math.max(0, Math.floor(p * (n - 1))))];
    this.zlo = q(0.002); this.zhi = q(0.999);
    if (!(this.zhi - this.zlo > 1e-6)) this.zhi = this.zlo + 1;
    this.cz = (this.zlo + this.zhi) / 2;
    this.xyExt = Math.max(hi[0] - lo[0], hi[1] - lo[1], 1e-6);
    // true scale only when both are mm; camera px vs relief (projector) px are different units
    this.zexAuto = d.units.xy === 'mm' && d.units.z === 'mm' ? 1 : (0.3 * this.xyExt) / (this.zhi - this.zlo);
    const col = new Uint32Array(n), span = this.zhi - this.zlo;
    for (let i = 0; i < n; i++) {
      const t = Math.max(0, Math.min(1, (z[i] - this.zlo) / span));
      col[i] = LUT32[(t * 255) | 0];
    }
    Object.assign(this, { x, y, z, col });
    this.draw();
  }
  get zex() { return (this.zexAuto || 1) * this.zexMul; }
  draw() {
    const c = this.c, dpr = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.round(c.clientWidth * dpr)), H = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== W || c.height !== H || !this.img) {
      c.width = W; c.height = H;
      this.img = this.ctx.createImageData(W, H);
      this.buf = new Uint32Array(this.img.data.buffer);
      this.zb = new Float32Array(W * H);
    }
    this.buf.fill(color32(cssVar('--cloud-bg')));
    if (!this.n) { this.ctx.putImageData(this.img, 0, 0); this.legend(); return; }
    this.zb.fill(-Infinity);
    const cyw = Math.cos(this.yaw), syw = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const s = ((0.8 * Math.min(W, H)) / this.xyExt) * this.zoom;
    const ox = W / 2 + this.panX * dpr, oy = H / 2 + this.panY * dpr;
    const zex = this.zex, ps = Math.max(1, Math.round(this.psize * dpr));
    const { x, y, z, col, buf, zb, cx, cy, cz } = this;
    for (let i = 0; i < this.n; i++) {
      const X = x[i] - cx, Y = -(y[i] - cy), Z = (z[i] - cz) * zex;     // image y points down: flip
      const xr = X * cyw - Y * syw, yr = X * syw + Y * cyw;
      const ys = yr * cp - Z * sp, d = yr * sp + Z * cp;
      const px = (ox + xr * s) | 0, py = (oy - ys * s) | 0;
      if (px < 0 || py < 0 || px + ps > W || py + ps > H) continue;
      for (let dy = 0; dy < ps; dy++) {
        let idx = (py + dy) * W + px;
        for (let dx = 0; dx < ps; dx++, idx++) if (d > zb[idx]) { zb[idx] = d; buf[idx] = col[i]; }
      }
    }
    this.ctx.putImageData(this.img, 0, 0);
    // axis gizmo
    const g = this.ctx, L = 26 * dpr, gx = 44 * dpr, gy = H - 44 * dpr;
    g.lineWidth = 2 * dpr;
    g.font = `${11 * dpr}px ${cssVar('--mono') || 'monospace'}`;
    [[1, 0, 0, 'X', '#ff6b6b'], [0, 1, 0, 'Y', '#3ccf8a'], [0, 0, 1, 'Z', '#4ca3ff']].forEach(([X, Y, Z, lab, colr]) => {
      const xr = X * cyw - Y * syw, yr = X * syw + Y * cyw, ys = yr * cp - Z * sp;
      g.strokeStyle = colr; g.fillStyle = colr;
      g.beginPath(); g.moveTo(gx, gy); g.lineTo(gx + xr * L, gy - ys * L); g.stroke();
      g.fillText(lab, gx + xr * (L + 8 * dpr) - 4 * dpr, gy - ys * (L + 8 * dpr) + 4 * dpr);
    });
    this.legend();
  }
  legend() {
    const el = $('#cloudLegend');
    if (!this.n || !this.meta) { el.hidden = true; return; }
    const u = this.meta.units;
    el.hidden = false;
    put(el, 
      h('div', null, `height (${u.z}${u.z === 'px' ? ', relief' : ''})`),
      h('div', { class: 'bar', style: { background: `linear-gradient(90deg, ${VIRIDIS.map((c) => `rgb(${c.join(',')})`).join(',')})` } }),
      h('div', { class: 'ends' }, h('span', null, fmt(this.zlo, 2)), h('span', null, fmt(this.zhi, 2))),
      h('div', null, `z × ${fmt(this.zex, this.zex < 10 ? 2 : 0)}${u.xy !== u.z ? ' (xy ' + u.xy + ')' : ''}`));
  }
}
let Cloud = null;
async function showCloud(o) {
  S.cloudId = o.id;
  S.cloudObj = o;
  $('#cloudTitle').textContent = `${o.label} (${o.id})`;
  $$('#objList .item').forEach((el) => el.classList.toggle('sel', el.querySelector('.meta.mono') && el.querySelector('.meta.mono').textContent === o.id));
  const info = $('#cloudInfo');
  put(info, h('span', { class: 'running' }, h('span', { class: 'spin' }), 'loading points…'));
  try {
    const d = await api(`/api/pointcloud/${encodeURIComponent(o.id)}?max=${$('#cloudMax').value}`);
    if (S.cloudId !== o.id) return;
    Cloud.reset(false);
    Cloud.setData(d);
    $('#cloudZexV').textContent = zexLabel();
    put(info, `${d.n.toLocaleString()} of ${d.n_total.toLocaleString()} points · xy ${d.units.xy} · z ${d.units.z} · tier ${d.tier}`,
      d.calibration_id ? ` · calibration ${d.calibration_id}` : ' · uncalibrated (relief px; no mm without a calibration for this geometry)');
  } catch (e) {
    put(info, h('div', { class: 'alert bad' }, h('b', null, 'Could not load the point cloud'), h('div', { class: 'msg' }, e.message)));
  }
}
function zexLabel() { return Number($('#cloudZex').value) === 0 ? `auto ×${fmt(Cloud.zex, Cloud.zex < 10 ? 2 : 0)}` : `×${fmt(Cloud.zex, Cloud.zex < 10 ? 2 : 0)}`; }

// ---------- image picker: points / rectangles / polygons in image (camera) px
class Picker {
  constructor(root, o = {}) {
    this.root = root;
    this.img = $('img', root);
    this.cv = $('canvas', root);
    this.ctx = this.cv.getContext('2d');
    this.mode = o.mode || 'none';
    this.onShape = o.onShape || (() => {});
    this.pts = []; this.poly = []; this.rect = null; this.shapes = []; this.drag = null;
    this.img.addEventListener('load', () => {
      this.cv.width = this.img.naturalWidth; this.cv.height = this.img.naturalHeight;
      root.classList.add('has');
      this.draw();
    });
    this.img.addEventListener('error', () => root.classList.remove('has'));
    this.cv.addEventListener('pointerdown', (e) => {
      if (this.mode !== 'rect') return;
      this.cv.setPointerCapture(e.pointerId);
      this.drag = this.at(e); this.rect = [this.drag, this.drag]; this.draw();
    });
    this.cv.addEventListener('pointermove', (e) => {
      if (this.mode === 'rect' && this.drag) { this.rect = [this.drag, this.at(e)]; this.draw(); }
    });
    this.cv.addEventListener('pointerup', (e) => {
      if (this.mode !== 'rect' || !this.drag) return;
      const a = this.drag, b = this.at(e);
      this.drag = null;
      if (Math.abs(a[0] - b[0]) < 4 || Math.abs(a[1] - b[1]) < 4) { this.rect = null; this.draw(); return; }
      this.rect = [a, b];
      this.draw();
      this.onShape({ type: 'rect', pts: this.rect });
    });
    this.cv.addEventListener('click', (e) => {
      const p = this.at(e);
      if (this.mode === 'line') {
        if (this.pts.length >= 2) this.pts = [];
        this.pts.push(p);
        this.draw();
        this.onShape({ type: 'line', pts: this.pts.slice() });
      } else if (this.mode === 'poly') {
        this.poly.push(p);
        this.draw();
      } else if (this.mode === 'pick') {
        this.onShape({ type: 'point', pts: [p] });
      }
    });
    this.cv.addEventListener('dblclick', (e) => {
      if (this.mode !== 'poly') return;
      e.preventDefault();
      // a double-click also fired two clicks: drop the duplicate vertex
      while (this.poly.length > 1) {
        const [a, b] = this.poly.slice(-2);
        if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 6 * this.scale()) this.poly.pop(); else break;
      }
      if (this.poly.length >= 3) this.onShape({ type: 'poly', pts: this.poly.slice() });
      this.poly = [];
      this.draw();
    });
  }
  scale() { const r = this.cv.getBoundingClientRect(); return r.width ? this.cv.width / r.width : 1; }
  at(e) {
    const r = this.cv.getBoundingClientRect();
    return [Math.round(((e.clientX - r.left) / r.width) * this.cv.width), Math.round(((e.clientY - r.top) / r.height) * this.cv.height)];
  }
  set(url) {
    if (!url) { this.root.classList.remove('has'); this.img.removeAttribute('src'); return; }
    if (this.img.getAttribute('src') !== url) { this.root.classList.remove('has'); this.img.src = url; } else this.draw();
  }
  clear() { this.pts = []; this.poly = []; this.rect = null; this.draw(); }
  draw() {
    const g = this.ctx, k = this.scale();
    g.clearRect(0, 0, this.cv.width, this.cv.height);
    g.lineWidth = 2 * k;
    g.font = `${13 * k}px ${cssVar('--mono') || 'monospace'}`;
    const accent = cssVar('--accent'), warn = cssVar('--warn');
    const path = (pts, close) => { g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); if (close) g.closePath(); };
    const dot = (p, c) => { g.fillStyle = c; g.beginPath(); g.arc(p[0], p[1], 4 * k, 0, Math.PI * 2); g.fill(); };
    for (const s of this.shapes) {
      g.strokeStyle = s.color || warn;
      g.fillStyle = (s.color || warn) + '33';
      if (s.type === 'line') { path(s.pts, false); g.stroke(); s.pts.forEach((p) => dot(p, s.color || warn)); } else if (s.type === 'mark') {
        g.lineWidth = 3 * k; g.beginPath(); g.arc(s.pts[0][0], s.pts[0][1], 16 * k, 0, Math.PI * 2); g.stroke(); g.lineWidth = 2 * k;
      } else { path(s.pts, true); g.fill(); g.stroke(); }
      if (s.label) { g.fillStyle = s.color || warn; g.fillText(s.label, s.pts[0][0] + 6 * k, s.pts[0][1] - 6 * k); }
    }
    g.strokeStyle = accent;
    if (this.mode === 'line' && this.pts.length) { if (this.pts.length === 2) { path(this.pts, false); g.stroke(); } this.pts.forEach((p, i) => { dot(p, accent); g.fillText(i ? 'p1' : 'p0', p[0] + 7 * k, p[1] - 7 * k); }); }
    if (this.mode === 'rect' && this.rect) { const [a, b] = this.rect; g.setLineDash([6 * k, 4 * k]); g.strokeRect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])); g.setLineDash([]); }
    if (this.mode === 'poly' && this.poly.length) { path(this.poly, false); g.stroke(); this.poly.forEach((p) => dot(p, accent)); }
  }
}
const rectPoly = ([a, b]) => { const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]); return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]; };

// ---------- 2D profile
const Prof = { picker: null, id: null };
function loadProfileImage() {
  const o = objById($('#profScan').value);
  Prof.id = o ? o.id : null;
  Prof.picker.shapes = [];
  Prof.picker.clear();
  Prof.picker.set(o ? fileUrl(o.preview) : null);
}
function pickProfile(o) {
  $('#profScan').value = o.id;
  $('#profScan').dataset.touched = '1';
  loadProfileImage();
  $('#profPick').scrollIntoView({ behavior: 'smooth', block: 'center' });
}
async function runProfile() {
  const out = $('#profOut');
  const id = $('#profScan').value;
  if (!id) { put(out, h('div', { class: 'alert warn' }, 'Capture a scan first.')); return; }
  const mode = $('#profMode').value;
  const args = { object_id: id, width_px: Number($('#profWidth').value), axis: mode === 'points' ? 'auto' : mode };
  if (mode === 'points') {
    if (Prof.picker.pts.length < 2) { put(out, h('div', { class: 'alert warn' }, 'Click two points on the relief image, or choose an automatic line.')); return; }
    args.p0 = Prof.picker.pts[0]; args.p1 = Prof.picker.pts[1];
  }
  const job = await runJob('profile', args, { btn: $('#profBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  plotXY($('#profPlot'), [{ x: r.distance || [], y: r.height || [], color: cssVar('--accent'), label: 'height' }],
    { xLabel: `distance along the line (${r.units_distance})`, yLabel: `height (${r.units_height})` });
  const st = r.stats || {};
  add(out,
    resultHead('Profile of ', scanRef(job.args.object_id)),
    h('div', { class: 'row' }, tag(r.tier || '', r.units_height === 'mm' ? 'ok' : 'warn'), tag(`height ${r.units_height}`), tag(`distance ${r.units_distance}`),
      h('span', { class: 'small muted' }, `p0 ${JSON.stringify(r.p0)} → p1 ${JSON.stringify(r.p1)} · width ${r.width_px} px`)),
    kv(Object.entries(st).filter(([k]) => k !== 'units').map(([k, v]) => [k.replace(/_/g, ' '),
      typeof v === 'object' && v !== null ? h('span', null, JSON.stringify(v)) : v, profileStatUnit(k, r), /^n_|_n_/.test(k) ? 0 : 3])),
    warningsList(r.warnings),
    h('div', { class: 'row small' }, job.links.csv ? fileLink(job.links.csv, 'profile CSV', true) : null, job.links.png ? fileLink(job.links.png, 'profile PNG') : null));
  if (r.p0 && r.p1) { Prof.picker.shapes = [{ type: 'line', pts: [r.p0, r.p1], color: cssVar('--warn') }]; Prof.picker.draw(); }
}

/** Units for a profile stat: counts and sigmas are not heights. */
function profileStatUnit(k, r) {
  if (/^n_|_n_/.test(k)) return 'samples';
  if (/sigma/.test(k)) return 'σ';
  if (/distance|length/.test(k)) return r.units_distance;
  return r.units_height;
}

// =================================================================== 5 MEASURE
function renderCalList() {
  const box = $('#calList');
  const cands = S.objects.filter((o) => o.known_dims);
  put(box, ...(cands.length ? cands.map((o) => h('label', { class: 'item check' },
    h('input', { type: 'checkbox', checked: S.calSel.has(o.id), onchange: (e) => { if (e.target.checked) S.calSel.add(o.id); else S.calSel.delete(o.id); } }),
    thumb(o.preview, o.label),
    h('div', null, h('div', { class: 'title' }, o.label), h('div', { class: 'meta' }, Object.entries(o.known_dims).map(([k, v]) => `${k} ${v}`).join(' · ') + ' mm'),
      h('div', { class: 'meta mono' }, o.id))))
    : [h('div', { class: 'empty' }, 'No scans with known dimensions at this geometry. Capture one with known H/W/L in the 3D scan tab.')]));
}
async function calibrate() {
  const out = $('#calOut');
  const ids = [...S.calSel].filter((id) => objById(id));
  if (!ids.length) { put(out, h('div', { class: 'alert warn' }, 'Tick at least one scan with known dimensions.')); return; }
  const job = await runJob('calibrate', { object_ids: ids }, { btn: $('#calBtn'), out });
  if (!done(job)) return;
  const c = job.result;
  add(out, h('div', { class: 'tiles2' },
    metric('Z scale', c.mm_per_px, 'mm/px', c.mm_per_px != null ? 'ok' : '', 'height per relief px', 5),
    metric('Z intercept', c.intercept_mm, 'mm', '', '', 3),
    metric('XY scale', c.lateral_px_per_mm, 'px/mm', c.lateral_px_per_mm != null ? 'ok' : '', 'camera px per mm', 4),
    metric('R²', c.r2, '', c.r2 == null ? '' : c.r2 > 0.999 ? 'ok' : 'warn', c.r2 == null ? 'one height: fit through origin' : 'depth fit', 5),
    metric('samples', c.n_samples, '', '', '')),
  kv([['Calibration id', h('span', null, c.id)], ['Bound to geometry', h('span', null, c.geometry_id)]]),
  h('div', { class: 'note' }, `Measurements at ${c.geometry_id} are now reported in mm. The calibration is only valid at this geometry (rule 6).`));
  refreshCals();
}
/** Calibrations in the library for this geometry; the newest is the one measurements use. */
async function refreshCals() {
  if (!S.geo) return;
  const r = await listKind('calibrations', S.geo);
  const cals = (r.items || []).slice().reverse();
  const act = S.status && S.status.calibration;
  const inLib = act && cals.some((c) => c.id === act.id);
  put($('#calLib'), ...[listNote(r)].filter(Boolean), cals.length ? table([
    { k: 'id', label: 'Calibration', f: (v) => h('span', { class: 'mono small' }, v) },
    { k: 'mm_per_px', label: 'Z mm/px', n: 1, f: (v) => fmt(v, 5) },
    { k: 'intercept_mm', label: 'Z intercept mm', n: 1, f: (v) => fmt(v, 3) },
    { k: 'lateral_px_per_mm', label: 'XY px/mm', n: 1, f: (v) => fmt(v, 4) },
    { k: 'r2', label: 'R²', n: 1, f: (v) => fmt(v, 5) },
    { k: 'n_samples', label: 'Samples', n: 1 },
    { k: 'id', label: '', f: (v, c) => h('span', null, act && act.id === v ? tag('in use', 'ok') : null, ' ',
      h('button', { class: 'small danger', onclick: () => deleteEntry('calibrations', { ...c, label: v }) }, 'Delete…')) },
  ], cals) : h('div', { class: 'muted small' }, `No calibration in the library for ${S.geo}.`),
  act && !inLib ? h('div', { class: 'muted small' }, `In use: ${act.id} (from the preset, not the library).`) : null,
  cals.length ? h('div', { class: 'muted small' }, 'Measurements use the newest calibration. Delete (move to trash) a bad fit to fall back to the one before it.') : null);
}
async function measure() {
  const out = $('#measOut');
  const id = $('#measScan').value;
  if (!id) { put(out, h('div', { class: 'alert warn' }, 'Capture a scan first.')); return; }
  const job = await runJob('measure', { object_id: id }, { btn: $('#measBtn'), out });
  if (!done(job)) return;
  const d = job.result;
  const used = job.args.object_id;
  const o = objById(used) || {};
  const known = o.known_dims || {};
  const rows = ['H', 'W', 'L'].map((k) => ({
    dim: { H: 'H (height)', W: 'W (major footprint)', L: 'L (minor footprint)' }[k],
    v: d[k], u: (d.unit_by_dim && d.unit_by_dim[k]) || d.units, known: known[k],
    err: d.error ? d.error[k] : null, pct: d.error_pct ? d.error_pct[k] : null,
  }));
  add(out,
    resultHead('Measured ', scanRef(used)),
    h('div', { class: 'row' }, tag(d.units, d.units === 'mm' ? 'ok' : d.units === 'mixed' ? 'warn' : 'bad'), tag(d.tier)),
    table([
      { k: 'dim', label: 'Dimension' },
      { k: 'v', label: 'Measured', n: 1, f: (v) => fmt(v, 3) },
      { k: 'u', label: 'Units' },
      { k: 'known', label: 'Known (mm)', n: 1, f: (v) => fmt(v, 2) },
      { k: 'err', label: 'Error', n: 1, f: (v, r) => (v == null ? '—' : `${v > 0 ? '+' : ''}${fmt(v, 3)} ${r.u}`) },
      { k: 'pct', label: 'Error %', n: 1, f: (v) => (v == null ? '—' : `${v > 0 ? '+' : ''}${fmt(v, 2)} %`) },
    ], rows),
    kv([['Blob pixels', d.n_px, 'px', 0], ['Raw relief (median)', d.relief_px, 'px', 3]]),
    warningsList(d.warnings));
}

// =================================================================== 6 DEFECTS
async function inspect() {
  const out = $('#inspOut');
  const id = $('#inspScan').value;
  if (!id) { put(out, h('div', { class: 'alert warn' }, 'Capture a scan first.')); return; }
  const job = await runJob('inspect', { object_id: id, n_sigma: Number($('#inspSigma').value), min_area_px: Number($('#inspArea').value),
    bg_window: Number($('#inspWin').value), whole_field: $('#inspWhole').checked }, { btn: $('#inspBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  const u = r.units;
  add(out,
    resultHead('Inspected ', scanRef(job.args.object_id)),
    h('div', { class: 'row' }, verdictBadge(r.verdict), verdictWhy(r.verdict, r.warnings), tag(u, u === 'mm' ? 'ok' : 'warn'), tag(r.tier), h('span', { class: 'muted small' }, `${(r.defects || []).length} blob(s)`)),
    h('div', { class: 'split even' },
      h('div', null,
        kv([['Noise σ', r.noise_sigma, u, 4], ['Detection threshold', r.threshold, u, 4], ['FAIL trigger', r.trigger, r.trigger != null ? 'mm' : '', 3],
          ['Inspected region', r.region_px, 'px', 0]]),
        warningsList(r.warnings),
        r.defects && r.defects.length ? table([
          { k: 'i', label: '#', n: 1, f: (v, d, i) => i + 1 },
          { k: 'height', label: `Height (${u})`, n: 1, f: (v) => fmt(v, 4) },
          { k: 'peak', label: `Peak (${u})`, n: 1, f: (v) => fmt(v, 4) },
          { k: 'snr', label: 'SNR', n: 1, f: (v) => fmt(v, 1) },
          { k: 'area_px', label: 'Area px', n: 1 },
          { k: 'area_mm2', label: 'Area mm²', n: 1, f: (v) => fmt(v, 2) },
          { k: 'centroid', label: 'Centroid (x, y)' },
          { k: 'exceeds_trigger', label: '≥ trigger', f: (v) => (v == null ? '—' : v ? tag('FAIL', 'bad') : tag('below', 'ok')) },
        ], r.defects, { rowClass: (d) => (d.exceeds_trigger ? 'err' : '') }) : h('div', { class: 'empty' }, 'No defect blobs above the threshold.')),
      figures([job.links.annotated_png, 'annotated relief (defects boxed)'])));
}
async function compare() {
  const out = $('#cmpOut');
  const a = $('#cmpScan').value, g = $('#cmpGolden').value;
  if (!a || !g) { put(out, h('div', { class: 'alert warn' }, 'Pick a scan and a golden scan.')); return; }
  const args = { object_id: a, golden_id: g };
  if ($('#cmpMm').value !== '') args.fail_mm = Number($('#cmpMm').value);
  if ($('#cmpPx').value !== '') args.fail_px = Number($('#cmpPx').value);
  const job = await runJob('compare', args, { btn: $('#cmpBtn'), out });
  if (!done(job)) return;
  const v = job.result;
  const L = job.links || {};
  add(out,
    resultHead('Compared ', scanRef(job.args.object_id), h('span', { class: 'muted' }, ' against golden '), scanRef(job.args.golden_id)),
    a === g ? h('div', { class: 'alert warn' }, 'The scan was compared with itself.') : null,
    h('div', { class: 'row' }, verdictBadge(v.result), tag(v.units, v.units === 'mm' ? 'ok' : 'warn'), tag(v.tier)),
    h('div', { class: 'split even' },
      h('div', null,
        kv([['Max |deviation|', v.max_abs_dev, v.units, 4], ['99th pct |deviation|', v.p99_abs_dev, v.units, 4], ['Tolerance band', v.tolerance, v.units, 3],
          ['Pixels over the band', v.n_fail_px, 'px', 0], ['Pixels compared', v.n_compared_px, 'px', 0]]),
        warningsList(v.warnings),
        L.json ? h('div', { class: 'small' }, 'result: ', fileLink(L.json)) : null),
      figures([L.png, 'deviation map (where the scan departs from the golden)'])));
}
async function refreshBanks() {
  const r = await listKind('banks', S.geo);
  S.banks = r.items || [];
  put($('#bankNames'), ...S.banks.map((b) => h('option', { value: b.name })));
  const box = $('#bankList');
  const members = (b) => h('details', null, h('summary', null, `${b.n_scans} good scan${b.n_scans === 1 ? '' : 's'}`),
    (b.scan_ids || []).length ? h('ul', { class: 'plain members' }, b.scan_ids.map((id) => h('li', null, scanRef(id), ' ',
      h('button', { class: 'small ghost', onclick: () => bankRemove(b, id) }, 'Remove…'))))
      : h('div', { class: 'muted small' }, 'no members'));
  put(box, ...[listNote(r)].filter(Boolean), S.banks.length ? table([
    { k: 'name', label: 'Bank' }, { k: 'n_scans', label: 'Good scans', f: (v, b) => members(b) }, { k: 'zone', label: 'Zone' },
    { k: 'name', label: '', f: (v, b) => h('span', null, h('button', { class: 'small ghost', onclick: () => { $('#bankName').value = b.name; } }, 'use'), ' ',
      h('button', { class: 'small danger', onclick: () => deleteEntry('banks', b) }, 'Delete…')) },
  ], S.banks) : null);
}
async function bankRemove(b, id) {
  const o = objById(id);
  const reason = prompt(`Remove scan "${o ? o.label : id}" from bank "${b.name}"?\n\n${id}\n\nThe scan itself is kept; the bank records the removal (append-only). Why?`, 'not a good part');
  if (reason == null) return;
  const out = $('#bankOut');
  const job = await runJob('bank_remove', { object_id: id, bank: b.name, reason: reason || 'removed from the web UI' }, { out });
  if (!done(job)) return;
  const info = job.result;
  add(out, resultHead('Removed ', scanRef(id), h('span', { class: 'muted' }, ` from bank "${info.name}"`)),
    h('div', { class: 'alert ok' }, `Bank "${info.name}" now holds ${info.n_scans} good scan${info.n_scans === 1 ? '' : 's'}.`), warningsList(info.warnings));
  refreshBanks();
}
async function bankAdd() {
  const out = $('#bankOut');
  const id = $('#bankScan').value;
  if (!id) { put(out, h('div', { class: 'alert warn' }, 'Capture a scan first.')); return; }
  const job = await runJob('bank_add', { object_id: id, bank: $('#bankName').value || 'default' }, { btn: $('#bankAdd'), out });
  if (!done(job)) return;
  const b = job.result;
  add(out, resultHead('Added ', scanRef(job.args.object_id), h('span', { class: 'muted' }, ` to bank "${b.name}"`)),
    h('div', { class: 'alert ok' }, h('b', null, `Bank "${b.name}" now holds ${b.n_scans} good scan${b.n_scans === 1 ? '' : 's'}`),
      `${b.geometry_id}${b.zone ? ' · zone ' + b.zone : ''}`), warningsList(b.warnings));
  refreshBanks();
}
async function bankCheck() {
  const out = $('#bankOut');
  const id = $('#bankScan').value;
  if (!id) { put(out, h('div', { class: 'alert warn' }, 'Capture a scan first.')); return; }
  const job = await runJob('bank_check', { object_id: id, bank: $('#bankName').value || 'default', k_sigma: Number($('#bankK').value),
    min_area_px: Number($('#bankArea').value) }, { btn: $('#bankCheck'), out });
  if (!done(job)) return;
  const v = job.result;
  const an = v.anomalies || [];
  add(out,
    resultHead('Checked ', scanRef(job.args.object_id), h('span', { class: 'muted' }, ` against bank "${v.bank}"`)),
    h('div', { class: 'row' }, verdictBadge(v.result), tag(v.units, v.units === 'mm' ? 'ok' : 'warn'), v.tier ? tag(v.tier) : null,
      h('span', { class: 'muted small' }, `${an.length} anomal${an.length === 1 ? 'y' : 'ies'} · ${v.n_ref} good reference scan(s)`)),
    kv([['Worst |dev| / spread', v.max_z, 'σ', 2], ['k (threshold)', v.k_sigma, 'σ', 1], ['Pixels judged', v.n_compared_px, 'px', 0],
      ['Overlap with the reference', v.overlap_pct, '%', 1], ['Noise floor', v.noise_floor, v.units, 4], v.zone ? ['Zone', h('span', null, v.zone)] : null]),
    warningsList(v.warnings),
    an.length ? table([
      { k: 'area_px', label: 'Area px', n: 1 }, { k: 'max_z', label: 'max |z| σ', n: 1, f: (x) => fmt(x, 1) },
      { k: 'mean_dev', label: 'Mean dev', n: 1, f: (x, r) => `${fmt(x, 4)} ${r.units || v.units}` }, { k: 'centroid', label: 'Centroid' }, { k: 'bbox', label: 'Bbox' },
    ], an) : null,
    v.missing && v.missing.length ? h('div', null, h('h4', null, 'Missing (present in every good part, absent here)'),
      table([{ k: 'area_px', label: 'Area px', n: 1 }, { k: 'centroid', label: 'Centroid' }, { k: 'bbox', label: 'Bbox' }], v.missing)) : null,
    figures([job.links.png, 'anomaly map']));
}

// =================================================================== 7 CODES
async function readCodes() {
  const out = $('#codeOut');
  const args = { label: $('#codeLabel').value || 'codes' };
  if ($('#codeLevels').value.trim()) args.levels = $('#codeLevels').value.trim();
  const job = await runJob('read_codes', args, { btn: $('#codeBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  const codes = r.codes || [];
  add(out, 
    h('div', { class: 'row' }, h('span', { class: 'verdict ' + (codes.length ? 'ok' : 'na') }, `${codes.length} CODE${codes.length === 1 ? '' : 'S'}`),
      tag(r.polarity_inverted ? 'polarity inverted (measured)' : 'polarity normal'), h('span', { class: 'muted small' }, 'illuminations tried: ' + (r.illuminations || []).join(', '))),
    codes.length ? table([
      { k: 'kind', label: 'Kind' }, { k: 'data', label: 'Data', f: (v) => h('span', { class: 'mono' }, v) }, { k: 'decoder', label: 'Decoder' },
      { k: 'illumination', label: 'Illumination' }, { k: 'corners', label: 'Corners (camera px)', f: (v) => (v ? v.map((p) => `(${p.map((x) => Math.round(x)).join(',')})`).join(' ') : '—') },
    ], codes) : null,
    warningsList(r.warnings),
    !codes.length && S.status && S.status.mock ? h('div', { class: 'note' }, 'Mock mode: place a QR label on the simulated bench (Sim bench, top right) first.') : null,
    figures([job.links.frame_png, 'full-on frame']));
  refreshCodeReadings();
}
async function refreshCodeReadings() {
  const r = await listKind('readings');
  const rows = (r.items || []).slice().reverse();
  put($('#codeList'), ...[listNote(r)].filter(Boolean), rows.length ? table([
    { k: 'id', label: 'Reading', f: (v) => h('span', { class: 'mono small' }, v) },
    { k: 'codes', label: 'Codes', f: (v) => ((v || []).length ? (v || []).map((c) => `${c.kind || ''} ${c.data || ''}`.trim()).join(' · ') : h('span', { class: 'muted' }, 'none')) },
    { k: 'preset', label: 'Preset', f: (v, x) => h('span', { class: 'small' }, `${v || ''}${x.mock ? ' (mock)' : ''}`) },
    { k: 'path', label: '', f: (v) => (v && fileUrl(v + '/full_on.png') ? h('a', { href: '#', onclick: (e) => { e.preventDefault(); openImage(v + '/full_on.png', 'full-on frame'); } }, 'frame') : '') },
    { k: 'id', label: '', f: (v, x) => h('button', { class: 'small danger', onclick: () => deleteEntry('readings', { ...x, label: v }) }, 'Delete…') },
  ], rows) : h('div', { class: 'empty' }, 'No code readings yet.'));
}

// =================================================================== 8 SPECTRAL
const Spec = { picker: null, roi: null };
function readingImage(r) { return r && (r.preview || r.colour_png); }
async function refreshReadings(select) {
  const r = await listKind('spectral');
  S.readings = (r.items || []).slice().reverse();
  const sel = $('#specReading');
  const keep = select || sel.value;
  put(sel, ...(S.readings.length ? S.readings.map((x) => h('option', { value: x.id }, `${x.label} · ${shortTime(x.created)}${x.geometry_id !== S.geo ? ' (other geometry)' : ''}`))
    : [h('option', { value: '' }, r.error ? 'spectral mode unavailable' : 'no readings yet')]));
  if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
  if (r.error) put($('#specOut'), listNote(r));
  showReading();
}
function showReading() {
  const rd = S.readings.find((x) => x.id === $('#specReading').value);
  Spec.picker.set(rd ? fileUrl(readingImage(rd)) : null);
}
async function spectralCapture() {
  const out = $('#specOut');
  const job = await runJob('spectral_capture', { label: $('#specLabel').value || 'spectral' }, { btn: $('#specBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  add(out, kv([['Reading', h('span', null, r.id)], ['Bands', h('span', null, (r.bands || []).join(', '))], ['Valid', r.valid_pct, '%', 1],
    ['Noise σ', r.noise_sigma, 'counts', 2], ['White reference', r.white_ref, 'counts', 1]]), warningsList(r.warnings));
  refreshReadings(r.id);
}
async function refreshMaterials() {
  const r = await listKind('materials');
  S.materials = r.items || [];
  const box = $('#matList');
  const swatch = (sig) => {
    if (!sig || sig.length !== 3) return null;
    const m = Math.max(...sig, 1e-6);
    const [rr, gg, bb] = sig.map((v) => Math.round((255 * v) / m));
    return h('span', { style: { display: 'inline-block', width: '14px', height: '14px', borderRadius: '3px', verticalAlign: 'middle', background: `rgb(${rr},${gg},${bb})`, border: '1px solid ' + cssVar('--line2') } });
  };
  put(box, ...[listNote(r)].filter(Boolean), S.materials.length ? table([
    { k: 'name', label: '', f: (v) => h('input', { type: 'checkbox', class: 'matPick', value: v }) },
    { k: 'name', label: 'Material', f: (v, m) => h('span', null, swatch(m.signature), ' ', v) },
    { k: 'signature', label: 'Signature R/G/B', f: (v) => (v || []).map((x) => fmt(x, 3)).join(' / ') },
    { k: 'brightness', label: 'Brightness', n: 1, f: (v) => fmt(v, 3) },
    { k: 'n_px', label: 'Pixels', n: 1 },
    { k: 'geometry_id', label: 'Learned at', f: (v, m) => h('span', { class: 'small' }, `${v || ''}${m.mock ? ' (mock)' : ''}`) },
    { k: 'name', label: '', f: (v, m) => h('button', { class: 'small danger', title: 'moves every learned version of this material to trash', onclick: () => deleteEntry('materials', m) }, 'Delete…') },
  ], S.materials) : h('div', { class: 'empty' }, 'No materials learned yet.'));
}
function deleteReading() {
  const rd = S.readings.find((x) => x.id === $('#specReading').value);
  if (!rd) { toast('pick a spectral reading first', 'warn'); return; }
  deleteEntry('spectral', rd);
}
async function learnMaterial() {
  const out = $('#matOut');
  const rid = $('#specReading').value;
  if (!rid) { put(out, h('div', { class: 'alert warn' }, 'Capture a spectral reading first.')); return; }
  if (!Spec.roi) { put(out, h('div', { class: 'alert warn' }, 'Drag a rectangle on the reading, inside one surface.')); return; }
  const name = $('#matName').value.trim();
  if (!name) { put(out, h('div', { class: 'alert warn' }, 'Name the material.')); return; }
  const [a, b] = Spec.roi;
  const roi = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  const job = await runJob('learn_material', { reading_id: rid, name, roi }, { btn: $('#matLearn'), out });
  if (!done(job)) return;
  const m = job.result;
  add(out, h('div', { class: 'alert ok' }, h('b', null, `Learned "${m.name}"`),
    `signature ${(m.signature || []).map((x) => fmt(x, 3)).join(' / ')} · brightness ${fmt(m.brightness, 3)} · ${m.n_px} px`), warningsList(m.warnings));
  refreshMaterials();
}
async function classify() {
  const out = $('#clsOut');
  const rid = $('#specReading').value;
  if (!rid) { put(out, h('div', { class: 'alert warn' }, 'Pick a spectral reading.')); return; }
  const mats = $$('.matPick:checked').map((c) => c.value);
  const args = { reading_id: rid, max_distance: Number($('#clsDist').value) };
  if (mats.length) args.materials = mats;
  const job = await runJob('classify', args, { btn: $('#clsBtn'), out });
  if (!done(job)) return;
  const c = job.result;
  const cov = Object.entries(c.coverage_pct || {}).sort((x, y) => y[1] - x[1]).map(([name, pct]) => ({ name, pct }));
  add(out, 
    figures([c.map_png, 'classification map']),
    table([{ k: 'name', label: 'Material' }, { k: 'pct', label: 'Coverage % of valid px', n: 1, f: (v) => fmt(v, 2) }], cov),
    c.blobs && c.blobs.length ? table([{ k: 'name', label: 'Region' }, { k: 'area_px', label: 'Area px', n: 1 }, { k: 'centroid', label: 'Centroid' }, { k: 'bbox', label: 'Bbox' }], c.blobs) : null,
    kv([['Valid', c.valid_pct, '%', 1], ['Max distance', c.max_distance, '', 2]]),
    warningsList(c.warnings),
    c.json_path ? h('div', { class: 'small' }, 'result: ', fileLink(c.json_path)) : null);
}
async function separateLight() {
  const out = $('#sepOut');
  const job = await runJob('separate_light', { label: $('#sepLabel').value || 'light', square_px: Number($('#sepSq').value), shifts: Number($('#sepShifts').value) },
    { btn: $('#sepBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  add(out, 
    h('div', { class: 'tiles2' },
      metric('global fraction', r.global_fraction_median, '', r.global_fraction_median > 0.3 ? 'warn' : 'ok', 'median, lit px', 3),
      metric('direct', r.direct_median, 'counts', '', 'median', 1), metric('global', r.global_median, 'counts', '', 'median', 1),
      metric('lit', r.lit_pct, '%', '', '', 1), metric('checker contrast', r.checker_contrast, '', '', 'direct / white', 3)),
    figures([r.direct_png, 'direct (single bounce)'], [r.global_png, 'global (inter-reflection + scatter)']),
    warningsList(r.warnings));
}

// =================================================================== 9 GRID & ZONES
const Grid = { picker: null, zonePicker: null, view: 'overlay' };
function gridImage(g, view) {
  if (!g) return null;
  const order = { overlay: ['overlay_png', 'camera_png'], camera: ['camera_png', 'overlay_png'], dmd: ['dmd_png'] }[view] || ['camera_png'];
  for (const k of order) if (g[k]) return g[k];
  return null;
}
function colName(i) { let s = ''; i += 1; while (i > 0) { const m = (i - 1) % 26; s = String.fromCharCode(65 + m) + s; i = Math.floor((i - 1) / 26); } return s; }
function colIndex(letters) { let n = 0; for (const ch of letters.toUpperCase()) n = n * 26 + (ch.charCodeAt(0) - 64); return n - 1; }
function gridCells(g) {
  if (!g) return [];
  if (g.cells_visible && g.cells_visible.length) return g.cells_visible.map((c) => (Array.isArray(c) ? colName(c[0]) + (c[1] + 1) : String(c)));
  const spec = g.grid || {};
  const cols = spec.col_labels || [], rows = spec.row_labels || [];
  return rows.flatMap((r) => cols.map((c) => c + r));
}
/** Zone text -> set of cell names, blocks expanded (display only; the engine parses for real). */
function zoneCells(text, spec) {
  const out = new Set();
  const norm = String(text || '').replace(/@\s*\d+\s*$/, '').replace(/\s*:\s*/g, ':');
  for (const part of norm.split(/[,;\s]+/).filter(Boolean)) {
    if (part.toLowerCase() === 'all' && spec) { gridCells({ grid: spec }).forEach((c) => out.add(c)); continue; }
    const ends = part.split(':').map((e) => /^([A-Za-z]{1,3})(\d{1,3})$/.exec(e));
    if (ends.some((m) => !m)) continue;
    if (ends.length === 1) { out.add(ends[0][1].toUpperCase() + Number(ends[0][2])); continue; }
    const [a, b] = ends;
    const c0 = colIndex(a[1]), c1 = colIndex(b[1]), r0 = Number(a[2]), r1 = Number(b[2]);
    for (let c = Math.min(c0, c1); c <= Math.max(c0, c1); c++) for (let r = Math.min(r0, r1); r <= Math.max(r0, r1); r++) out.add(colName(c) + r);
  }
  return out;
}
/**
 * The engine names zones against the newest grid projected for this geometry, by anyone.
 * This page keeps showing the grid it has (zones it sends are pinned to that grid's spacing),
 * adopts the engine's grid when it has none, and says when a newer one exists.
 */
function syncGrid(eng) {
  S.engineGrid = eng || null;
  if (!S.grid && S.geo) S.grid = LS.get('grid.' + S.geo, null);
  if (!S.grid && eng) {
    S.grid = eng;
    LS.set('grid.' + S.geo, eng);
    if (curTab === 'grid') renderGrid();
    if (curTab === 'sensors') renderSensPick();
  }
  renderGridNotice();
}
let gridNoticeKey = null;
function renderGridNotice() {
  const box = $('#gridNotice');
  const e = S.engineGrid, g = S.grid;
  const differs = !!(e && g && e.json_path && g.json_path !== e.json_path);
  const key = differs ? `${e.json_path}|${g.json_path}` : '';
  if (key === gridNoticeKey) return;
  gridNoticeKey = key;
  if (!differs) { put(box); return; }
  const es = e.grid && e.grid.spacing_px, gs = g.grid && g.grid.spacing_px;
  put(box, h('div', { class: 'alert ' + (es !== gs ? 'warn' : 'info') },
    h('b', null, 'A newer grid was projected'),
    h('div', null, `The engine's newest grid for this geometry (${es} DMD px, ${shortTime(e.created)}${e.label ? ', "' + e.label + '"' : ''}) is not the photo shown here (${gs} DMD px). ` +
      `Zones this page sends are pinned to the photo shown ("…@${gs}"), so they still mean the cells you see.`),
    h('div', { class: 'row' }, h('button', { class: 'small', onclick: () => { S.grid = e; LS.set('grid.' + S.geo, e); renderGrid(); renderGridNotice(); zoneTextChanged(); } }, 'Show the newest grid'))));
}
function renderGrid() {
  const g = S.grid;
  Grid.picker.set(fileUrl(gridImage(g, Grid.view)));
  $$('#gridView button').forEach((b) => b.classList.toggle('on', b.dataset.v === Grid.view));
  const spec = g && g.grid;
  $('#gridPhotoInfo').textContent = spec ? `${spec.n_cols} × ${spec.n_rows} cells of ${spec.spacing_px} DMD px` + (g.cells_visible && g.cells_visible.length ? ` · ${g.cells_visible.length} visible` : '') + ' · click the photo to pick cells' : '';
  const zone = zoneCells($('#zoneText').value, spec);
  // mark the chosen cells: camera views via the map's cell centres, the DMD view by geometry
  const onDmd = Grid.view === 'dmd' || !gridImage(g, 'overlay');
  Grid.picker.shapes = !spec ? [] : [...zone].flatMap((name) => {
    if (onDmd && g.dmd_png && Grid.view === 'dmd') {
      const m = /^([A-Z]+)(\d+)$/.exec(name);
      if (!m) return [];
      const x0 = colIndex(m[1]) * spec.spacing_px, y0 = (Number(m[2]) - 1) * spec.spacing_px, s = spec.spacing_px;
      return [{ type: 'poly', pts: [[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]], color: cssVar('--accent') }];
    }
    const c = g.cell_centers && g.cell_centers[name];
    return c ? [{ type: 'mark', pts: [c], color: cssVar('--accent') }] : [];
  });
  Grid.picker.draw();
  // mini-map of the cells, laid out like the photo (rows top to bottom); unseen cells dimmed
  const box = $('#gridCells');
  if (spec && spec.col_labels && spec.row_labels) {
    const seen = g.cells_visible && g.cells_visible.length ? new Set(gridCells(g)) : null;
    box.style.gridTemplateColumns = `repeat(${spec.n_cols}, minmax(0, 1fr))`;
    put(box, ...spec.row_labels.flatMap((r) => spec.col_labels.map((c) => {
      const name = c + r;
      return h('button', { class: zone.has(name) ? 'on' : '', disabled: !!(seen && !seen.has(name)), title: seen && !seen.has(name) ? 'not seen by the camera' : name, onclick: () => toggleCell(name) }, name);
    })));
  } else {
    box.style.gridTemplateColumns = '';
    put(box);
  }
  Grid.zonePicker.set(fileUrl(g && (g.camera_png || g.overlay_png)));
  updateAgentBrief();
}
/**
 * The grid cell under a click on a grid image: the DMD view by geometry, a camera view by the
 * nearest cell centre (those need a projector map). -> { cell } or { error } (pure).
 */
function cellAt(p, g, view) {
  const spec = g && g.grid;
  if (!spec) return { error: 'Project the grid first (Grid & zones tab).' };
  if (view === 'dmd') {
    const c = Math.floor(p[0] / spec.spacing_px), r = Math.floor(p[1] / spec.spacing_px);
    if (c >= 0 && r >= 0 && c < spec.n_cols && r < spec.n_rows) return { cell: (spec.col_labels[c] || colName(c)) + (spec.row_labels[r] || r + 1) };
    return { error: 'That is outside the grid.' };
  }
  let best = Infinity, cell = null;
  for (const [name, xy] of Object.entries(g.cell_centers || {})) {
    if (!xy) continue;
    const d = Math.hypot(xy[0] - p[0], xy[1] - p[1]);
    if (d < best) { best = d; cell = name; }
  }
  return cell ? { cell } : { error: 'Picking cells on the camera photo needs a projector map: capture one, or switch the view to "sent to DMD".' };
}
/** A click on the grid photo -> the cell under it. */
function gridClick(p) {
  if (!(S.grid && S.grid.grid)) return;
  const r = cellAt(p, S.grid, Grid.view);
  if (r.error) { toast(r.error, 'warn'); return; }
  toggleCell(r.cell);
}
function toggleCell(c) {
  const parts = $('#zoneText').value.split(/[,;]+/).map((s) => s.trim()).filter(Boolean);
  const i = parts.findIndex((p) => p.toUpperCase() === c);
  if (i >= 0) parts.splice(i, 1); else parts.push(c);
  $('#zoneText').value = parts.join(',');
  zoneTextChanged();
  renderGrid();
}
// lenient: a hint only. The engine parses zones and explains what is wrong.
const ZONE_TOKEN = '(all|[A-Z]{1,3}\\d{1,3}(\\s*:\\s*[A-Z]{1,3}\\d{1,3})?)';
const ATLAS_RE = /^\s*atlas:\s*[\w.-]+\s*\/\s*[\w.-]+\s*$/i;
const ZONE_RE = new RegExp(`^\\s*${ZONE_TOKEN}([\\s,;]+${ZONE_TOKEN})*[\\s,;]*(@\\s*\\d+)?\\s*$`, 'i');
function zoneTextChanged() {
  const v = $('#zoneText').value;
  const atlas = ATLAS_RE.test(v);
  const ok = !v.trim() || ZONE_RE.test(v) || atlas;
  $('#zoneText').classList.toggle('invalid', !ok);
  $('#zoneCheck').textContent = !v.trim() ? 'Click cells on the photo or under it to build a zone.'
    : atlas ? `Atlas region ${zoneForEngine(v, S.grid)}: the engine resolves it from the imported atlas (Surfer, below).`
    : ok ? `Zone ${zoneForEngine(v, S.grid)}: ${zoneCells(v, S.grid && S.grid.grid).size || '?'} cell(s) of the grid shown. The engine checks that they exist.`
      : 'Expected something like C4, C4:D6 or B2,E5. The engine will explain what is wrong.';
  updateAgentBrief();
}
async function captureMap() {
  const out = $('#mapOut');
  if (!confirm('Capture a projector map now?\n\nThe rig must stay fixed and the room light steady. It maps the CURRENT scene; capture a new one after moving the camera, projector or focus.')) return;
  const job = await runJob('capture_map', { label: $('#mapLabel').value || 'map' }, { btn: $('#mapBtn'), out });
  if (!done(job)) return;
  const m = job.result;
  add(out, h('div', { class: 'alert ok' }, h('b', null, `Projector map ${m.id}`), `${m.geometry_id} · DMD ${(m.dmd_shape || []).join(' × ')}`),
    warningsList(m.warnings), figures([m.preview, 'projector map preview']));
  refreshMaps();
}
async function refreshMaps() {
  const r = await listKind('maps', S.geo);
  const maps = (r.items || []).slice().reverse();
  put($('#mapList'), ...[listNote(r)].filter(Boolean), maps.length ? table([
    { k: 'id', label: 'Map', f: (v) => h('span', { class: 'mono small' }, v) }, { k: 'created', label: 'Created', f: shortTime },
    { k: 'preview', label: '', f: (v) => (v ? h('a', { href: '#', onclick: (e) => { e.preventDefault(); openImage(v, 'projector map'); } }, 'preview') : '') },
    { k: 'id', label: '', f: (v, m) => h('button', { class: 'small danger', onclick: () => deleteEntry('maps', m) }, 'Delete…') },
  ], maps) : (r.error ? null : h('div', { class: 'muted small' }, 'No projector map for this geometry yet: zone scans are lit but not cut.')));
}
async function projectGrid() {
  const out = $('#gridOut');
  const job = await runJob('project_grid', { spacing_px: Number($('#gridSpacing').value), label: 'grid', hold_s: Number($('#gridHold').value) },
    { btn: $('#gridBtn'), out });
  if (!done(job)) return;
  S.grid = job.result;
  LS.set('grid.' + S.geo, S.grid);
  renderGridNotice();
  zoneTextChanged();
  const spec = S.grid.grid || {};
  add(out, h('div', { class: 'alert ok' }, h('b', null, 'Grid projected and photographed'),
    `${spec.n_cols || '?'} columns (${(spec.col_labels || []).slice(0, 1)}–${(spec.col_labels || []).slice(-1)}) × ${spec.n_rows || '?'} rows (1–${spec.n_rows || '?'})`),
  warningsList(S.grid.warnings));
  renderGrid();
}
async function scanZone() {
  const zone = $('#zoneText').value.trim();
  if (!zone) { put($('#zoneOut'), h('div', { class: 'alert warn' }, 'Enter a zone like C4, C4:D6, B2,E5 or atlas:part/part (or click cells on the photo).')); return; }
  await captureObject({ label: $('#zoneLabel').value || 'zone-part', base_id: $('#zoneBase').value, H: $('#zoneH').value, zone,
    btn: $('#zoneScan'), out: $('#zoneOut') });
}
function updateAgentBrief() {
  const st = S.status;
  if (!st) return;
  const g = S.grid;
  const port = (st.server && st.server.port) || location.port;
  const base = `http://127.0.0.1:${port}`;
  const zone = zoneForEngine($('#zoneText').value.trim() || 'C4:D5', g);
  const spec = g && g.grid;
  const baseId = activeBaseId();
  const args = { label: 'part', zone, ...(baseId ? { base_id: baseId } : {}) };
  const lines = [
    'Swiss Army Scanner: zone scan brief',
    `Server: ${base} (mode ${st.mock ? 'MOCK' : 'REAL'}, preset ${st.preset ? st.preset.name : 'none'}, geometry ${st.geometry_id || '?'})`,
  ];
  if (g) {
    if (g.overlay_png) lines.push(`Grid photo with cell labels: ${absPath(g.overlay_png)}`);
    if (g.camera_png) lines.push(`Camera photo of the projected grid: ${absPath(g.camera_png)}`);
    if (g.overlay_png || g.camera_png) lines.push(`  (also over HTTP: ${base}${fileUrl(g.overlay_png || g.camera_png)})`);
    if (spec) lines.push(`Grid: ${spec.spacing_px} DMD px cells; columns ${(spec.col_labels || []).slice(0, 1)}-${(spec.col_labels || []).slice(-1)} left to right, rows 1-${spec.n_rows} top to bottom, as seen in the photo.`);
    const cells = g.cells_visible && g.cells_visible.length ? gridCells(g) : [];
    if (spec && cells.length === spec.n_cols * spec.n_rows) lines.push(`The camera sees all ${cells.length} cells.`);
    else if (cells.length) lines.push(`Cells the camera sees: ${cells.join(', ')}`);
    else lines.push('No projector map yet, so the photo has no camera-space labels and zone scans are lit but not cut.');
  } else {
    lines.push('(No grid photo yet: project the grid first in the Grid & zones tab.)');
  }
  lines.push(
    'Zone syntax: "C4" = one cell, "C4:D6" = inclusive block, "B2,E5" = union.' +
      (spec ? ` Keep the "@${spec.spacing_px}": it pins the names to the ${spec.spacing_px} px grid in this photo (someone may project another grid later).` : ''),
    baseId ? `Base (the empty-scene datum the operator chose): ${baseId}` : 'Base: the newest usable base for this geometry.',
    'To scan the zone the part is in:',
    `  POST ${base}/api/jobs   (Content-Type: application/json)`,
    `  ${JSON.stringify({ verb: 'capture_object', args })}`,
    `  then poll GET ${base}/api/jobs/<id> until status is "done" or "error".`,
    `  Python: engine.api.capture_object(preset, base, "part", zone="${zone}")`,
    `All verbs and their args: GET ${base}/api/verbs`,
    'Rules: a person does the physical steps (placing parts, focus). Never move the rig between the base and the object. Check "units": px means uncalibrated.');
  $('#agentBrief').value = lines.join('\n');
}
function renderZones() {
  const z = S.zones;
  Grid.zonePicker.shapes = z.map((s) => ({ type: 'poly', pts: s.polygon, label: s.name, color: cssVar('--warn') }));
  Grid.zonePicker.draw();
  put($('#zoneList'), z.length ? table([
    { k: 'name', label: 'Zone', f: (v, s) => h('input', { value: v, class: 'w10', oninput: (e) => { s.name = e.target.value; renderZonesShapes(); } }) },
    { k: 'pattern', label: 'Pattern', f: (v, s) => h('select', { onchange: (e) => { s.pattern = e.target.value; } }, ['flood', 'outline', 'grid', 'stripes', 'checker'].map((p) => { const o = h('option', { value: p }, p); if (p === v) o.selected = true; return o; })) },
    { k: 'level', label: 'Level', f: (v, s) => h('input', { type: 'number', value: v, min: 0, max: 255, class: 'w5', oninput: (e) => { s.level = Number(e.target.value); } }) },
    { k: 'polygon', label: 'Points', n: 1, f: (v) => v.length },
    { k: 'name', label: '', f: (v, s) => h('button', { class: 'small ghost', onclick: () => { S.zones.splice(S.zones.indexOf(s), 1); renderZones(); } }, 'remove') },
  ], z) : h('div', { class: 'empty' }, 'No zones drawn yet.'));
}
function renderZonesShapes() {
  Grid.zonePicker.shapes = S.zones.map((s) => ({ type: 'poly', pts: s.polygon, label: s.name, color: cssVar('--warn') }));
  Grid.zonePicker.draw();
}
function addZoneShape(shape) {
  const poly = shape.type === 'rect' ? rectPoly(shape.pts) : shape.pts;
  let n = S.zones.length + 1;
  while (S.zones.some((z) => z.name === `Z${n}`)) n++;
  S.zones.push({ name: `Z${n}`, polygon: poly.map((p) => [Math.round(p[0]), Math.round(p[1])]), space: 'camera',
    pattern: $('#zonePattern').value, level: Number($('#zoneLevel').value) });
  Grid.zonePicker.rect = null;
  renderZones();
}
async function projectZones() {
  const out = $('#zonesOut');
  if (!S.zones.length) { put(out, h('div', { class: 'alert warn' }, 'Draw at least one zone on the photo.')); return; }
  const job = await runJob('project_zones', { zones: S.zones, label: 'zones' }, { btn: $('#zoneProject'), out });
  if (!done(job)) return;
  const r = job.result;
  add(out, warningsList(r.warnings), figures([r.camera_png, 'camera: zones lit'], [r.overlay_png, 'overlay'], [r.dmd_png, 'sent to the DMD']));
}

// =================================================================== 10 SENSORS
// Virtual sensors (engine/sense/virtual.py): the DMD draws the spot, the camera is the receiver.
const Sens = { picker: null, view: 'camera', clickTo: 'spotAt', clickAs: 'cell', scaleGeo: null };

const AT_POINT_RE = /^(dmd|cam|camera)\s*[:=]?\s*\(?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*\)?$/i;
/**
 * Where a spot / sensor goes, as typed -> the engine's `at` (pure; tests/test_ui_server.py).
 * 'h8' -> 'H8@96', 'G7:H9' -> 'G7:H9@96': pinned to the grid shown, like zones.
 * 'dmd 456,570' -> {dmd: [456, 570]} (DMD px); 'cam 700,500' -> {camera: [700, 500]} (camera px).
 */
function parseAt(text, grid) {
  const t = String(text || '').trim();
  if (!t) return null;
  const m = AT_POINT_RE.exec(t);
  if (m) {
    const pt = [Number(m[2]), Number(m[3])];
    return m[1].toLowerCase() === 'dmd' ? { dmd: pt } : { camera: pt };
  }
  return zoneForEngine(t, grid);
}
/** An `at` as the engine stores it ('H8@96', {cell}, {zone}, {dmd}, {camera}) -> short text. */
function atText(at) {
  if (at == null) return '—';
  if (typeof at === 'string') return at;
  const xy = (p) => p.map((v) => fmt(v, 1)).join(', ');
  if (at.cell) return at.cell + (at.spacing_px ? `@${at.spacing_px}` : '');
  if (at.zone) return `zone ${at.zone}${at.spacing_px ? '@' + at.spacing_px : ''}`;
  if (at.camera) return `cam ${xy(at.camera)}${at.dmd ? ` (dmd ${xy(at.dmd)})` : ''}`;
  if (at.dmd) return `dmd ${xy(at.dmd)}`;
  return JSON.stringify(at);
}
const xyText = (p) => (Array.isArray(p) && p.length === 2 ? h('span', { class: 'mono' }, `${fmt(p[0], 1)}, ${fmt(p[1], 1)}`) : h('span', { class: 'muted' }, '—'));
/** A spot's drawn size: DMD columns x rows (DLP4500 diamond array: rows at half the column pitch). */
const dmdSize = (s) => (s ? `${fmt(s.diameter_dmd_px, 1)} × ${fmt(s.diameter_dmd_rows, 1)}` : '—');
const warnBox = (out, msg) => put(out, h('div', { class: 'alert warn' }, msg));

function rememberScale() {
  if (S.geo) LS.set('spotK.' + S.geo, $('#spotK').value);
}
/** The measured scale belongs to the geometry it was measured at: restore it once per geometry. */
function restoreScale() {
  if (!S.geo || Sens.scaleGeo === S.geo) return;
  Sens.scaleGeo = S.geo;
  $('#spotK').value = LS.get('spotK.' + S.geo, '') || '';
}

async function spotSize() {
  const out = $('#spotSizeOut');
  const d = num($('#spotD').value), k = num($('#spotK').value);
  if (!(d > 0)) { warnBox(out, 'Enter the spot diameter at the part, in mm.'); return; }
  const measured = k > 0;
  const job = await runJob('spot_size', measured ? { diameter_mm: d, mm_per_dmd_px: k } : { diameter_mm: d },
    { btn: $('#spotSizeBtn'), out, quiet: true });
  if (!done(job)) return;
  let nominal = measured ? null : job.result, nomErr = null;
  if (measured) {           // the nominal one too, for comparison (it needs rig.standoff_mm)
    const nj = await runJob('spot_size', { diameter_mm: d }, { quiet: true, quietError: true });
    if (done(nj)) nominal = nj.result; else nomErr = nj.error;
  }
  const meas = measured ? job.result : null;
  const cell = (s, text) => (s ? text(s) : h('span', { class: 'muted small' }, s === null && nomErr ? 'not available' : '—'));
  const cols = [{ k: 'what', label: '' },
    { k: 'nominal', label: 'Nominal (rig geometry)', f: (v) => v }, { k: 'measured', label: 'Measured (your beam)', f: (v) => v }];
  const row = (what, text) => ({ what, nominal: cell(nominal, text), measured: meas ? text(meas) : h('span', { class: 'muted small' }, 'enter a measured scale') });
  const used = meas || nominal;
  add(out,
    h('div', { class: 'row' }, h('span', { class: 'verdict ok' }, `${dmdSize(used)} DMD px`), tag(`${used.source} scale`, used.source === 'measured' ? 'ok' : 'warn'),
      h('span', { class: 'muted small' }, `a ${fmt(d, 2)} mm spot: columns × rows`)),
    table(cols, [
      row('mm per DMD column at the part', (s) => h('span', { class: 'mono' }, fmt(s.mm_per_dmd_px, 4))),
      row('Spot on the DMD, columns × rows', (s) => h('b', null, dmdSize(s))),
      row('From', (s) => h('span', { class: 'small' }, s.source === 'measured' ? 'your measurement' : `standoff ${fmt(s.standoff_mm, 0)} mm ÷ throw ratio ${fmt(s.throw_ratio, 2)} ÷ 912`)),
    ]),
    meas && nominal ? h('div', { class: 'note' }, `Your beam: ${fmt(meas.mm_per_dmd_px, 4)} mm per column, ${fmt(100 * (meas.mm_per_dmd_px / nominal.mm_per_dmd_px - 1), 1)} % vs nominal. Projected spots and taught sensors use the measured scale.`) : null,
    nomErr ? h('div', { class: 'alert info' }, h('b', null, 'Nominal scale not available'), h('div', { class: 'msg' }, nomErr)) : null,
    h('div', { class: 'muted small' }, 'The DLP4500 is a diamond array: its rows sit at half the column pitch, so a round spot is drawn with twice as many rows as columns.'),
    warningsList(used.warnings));
  rememberScale();
}
function useMeasuredScale() {
  const mm = num($('#spotMeasMm').value), px = num($('#spotMeasPx').value);
  if (!(mm > 0) || !(px > 0)) { toast('Enter the measured diameter (mm) and the drawn width (DMD columns)', 'warn'); return; }
  $('#spotK').value = (mm / px).toPrecision(5);
  rememberScale();
  spotSize();
}

async function projectSpot() {
  const out = $('#spotOut');
  const at = parseAt($('#spotAt').value, S.grid);
  if (!at) { warnBox(out, 'Where? Type a grid cell (H8), dmd col,row or cam x,y, or click the grid photo below.'); return; }
  const d = num($('#spotD').value), k = num($('#spotK').value), level = num($('#spotLevel').value);
  if (!(d > 0)) { warnBox(out, 'Enter the spot diameter in step 1.'); return; }
  const args = { at, diameter_mm: d, shape: $('#spotShape').value, label: $('#spotLabel').value.trim() || 'spot' };
  if (k > 0) args.mm_per_dmd_px = k;
  if (Number.isFinite(level)) args.level = level;
  const job = await runJob('project_spot', args, { btn: $('#spotBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  S.lastSpot = r;
  if (r.spec && r.spec.diameter_dmd_px != null) $('#spotMeasPx').value = fmt(r.spec.diameter_dmd_px, 2);
  add(out, renderSpot(r));
  renderSensPick();
}
function renderSpot(r) {
  const sp = r.spec || {};
  return [
    h('div', { class: 'row' }, h('span', { class: 'verdict ' + (r.found ? 'ok' : 'bad') }, r.found ? 'SPOT FOUND' : 'NO SPOT'),
      tag(`${sp.source || '?'} scale`, sp.source === 'measured' ? 'ok' : 'warn'),
      r.diameter_mm != null ? tag('mm: lateral calibration', 'ok') : tag('camera px only', 'warn')),
    h('div', { class: 'split even' },
      h('div', null,
        kv([
          ['At', h('span', { class: 'mono' }, atText(r.at))],
          ['DMD target', xyText(r.target_dmd), 'DMD px [col, row]'],
          ['Drawn', h('span', null, dmdSize(sp)), `DMD px, cols × rows = ${fmt(sp.diameter_mm, 2)} mm`],
          ['Centroid', xyText(r.centroid_px), 'camera px'],
          ['Expected', xyText(r.expected_px), r.map_id ? 'camera px (projector map)' : 'camera px (needs a projector map)'],
          r.offset_px ? ['Offset', xyText(r.offset_px), 'camera px (centroid − expected)'] : null,
          ['Diameter', r.diameter_px, 'camera px (half-max area)', 2],
          r.diameter_mm != null ? ['Diameter', r.diameter_mm, 'mm (lateral calibration)', 3] : null,
          r.extent_px ? ['Extent', h('span', null, `${r.extent_px[0]} × ${r.extent_px[1]}`), 'camera px (w × h)'] : null,
          ['Peak', r.peak_counts, 'counts above dark', 1],
          ['SNR', r.snr, 'peak / noise σ', 1],
          r.saturated_px ? ['Saturated', r.saturated_px, 'px', 0] : null,
        ]),
        warningsList(r.warnings),
        h('div', { class: 'row small' }, r.photo_png ? fileLink(r.photo_png, 'photo') : null, r.json_path ? fileLink(r.json_path, 'report') : null)),
      figures([r.overlay_png, 'spot: circle = measured, cross = expected'], [r.photo_png, 'camera photo'])),
  ];
}

async function teachPresence() {
  const out = $('#presOut');
  const name = $('#presName').value.trim();
  const at = parseAt($('#presAt').value, S.grid);
  if (!name) { warnBox(out, 'Name the sensor.'); return; }
  if (!at) { warnBox(out, 'Where? A grid cell (H8), a zone (G7:H9), dmd col,row or cam x,y, or click the photo above with "click sets: sensor".'); return; }
  const method = $('#presMethod').value;
  const d = num($('#spotD').value), k = num($('#spotK').value), thr = num($('#presThr').value), minH = num($('#presMinH').value);
  // clearing the field is a physical, human-only step: the engine refuses to teach until the operator says it is done
  if (!confirm(`Is the sensor's field EMPTY?\n\nSensor "${name}" (${method}) at ${atText(at)} records the EMPTY scene as its reference at ${S.geo}. ` +
    'Take every part out of the light first: that is a physical step only you can do.\n\nOK: the field is empty, teach now.\nCancel: do not teach.')) return;
  const args = { name, at, method, empty_confirmed: true };
  if (d > 0) args.diameter_mm = d;
  if (k > 0) args.mm_per_dmd_px = k;
  if (thr > 0) args.threshold = thr;
  if (minH > 0) args.min_height_mm = minH;
  const job = await runJob('teach_presence', args, { btn: $('#presTeach'), out });
  if (!done(job)) return;
  const t = job.result;
  const trig = t.method === 'bgs' ? `PRESENT when a surface is more than ${fmt(t.min_height_mm, 1)} mm above the taught background`
    : `PRESENT when the returned light changes by more than ${fmt(100 * t.threshold, 0)} %`;
  add(out, h('div', { class: 'alert ok' }, h('b', null, `Sensor "${t.name}" taught (${t.method}, ${t.mode})`), trig),
    kv([
      ['At', h('span', { class: 'mono' }, atText(t.at))],
      ['Reference (empty)', t.reference_signal, 'counts per lit px', 1],
      ['Lit area', t.n_px, 'camera px', 0],
      ['Receiver window', h('span', { class: 'mono' }, (t.window || []).join(', ')), 'camera px [x0, y0, x1, y1)'],
      t.spot ? ['Spot', h('span', null, dmdSize(t.spot)), `DMD px = ${fmt(t.spot.diameter_mm, 2)} mm (${t.spot.source} scale)`] : null,
      t.jacobian ? ['Jacobian', h('span', { class: 'mono small' }, JSON.stringify(t.jacobian)), 'camera px per DMD px'] : null,
    ]),
    warningsList(t.warnings), figures([t.photo_png, 'teach photo: green box = receiver window']));
  refreshSensors();
}

async function checkPresence(name, btn) {
  const out = $('#presCheckOut');
  const job = await runJob('check_presence', { name }, { btn, out });
  if (!done(job)) return;
  const r = job.result;
  const pct = (v) => (v == null ? null : 100 * v);
  add(out,
    resultHead('Sensor ', h('b', null, r.name), h('span', { class: 'muted small' }, ` · ${r.method} · ${shortTime(r.created)}`)),
    h('div', { class: 'row' }, sensorBadge(r.verdict), verdictWhy(r.verdict, r.warnings)),
    kv(r.method === 'bgs' ? [
      ['Height', r.height_mm, `mm (${r.height_source || 'no mm scale'})`, 2],
      ['Trigger', r.min_height_mm, 'mm: |height| above the taught background', 1],
      ['Relief', r.relief_px, 'DMD px', 3],
      ['Spot shift', r.shift_px, 'camera px along the baseline', 2],
      ['Off-axis', r.off_axis_px, 'camera px across the baseline', 2],
      ['Centroid', xyText(r.centroid_px), 'camera px'],
      ['Taught centroid', xyText(r.reference_centroid_px), 'camera px'],
      r.mm_per_relief_px != null ? ['Scale', r.mm_per_relief_px, 'mm per relief px', 4] : null,
    ] : [
      ['Signal', r.signal, 'counts per lit px', 1],
      ['Reference (empty)', r.reference_signal, 'counts per lit px', 1],
      ['Change', pct(r.relative_change), '%', 1],
      ['Trigger', pct(r.threshold), '%: |change| above it = PRESENT', 0],
      ['Saturated', pct(r.saturated_fraction), '% of lit px', 1],
    ]),
    warningsList(r.warnings),
    figures([r.photo_png, 'receiver: green = window, cyan = taught spot, yellow = now']),
    r.json_path ? h('div', { class: 'small' }, 'report: ', fileLink(r.json_path)) : null);
}

async function refreshSensors() {
  if (!S.geo) return;
  const r = await listKind('sensors', S.geo);
  S.sensors = r.items || [];
  put($('#alNames'), ...S.sensors.filter((s) => s.kind === 'alignment').map((s) => h('option', { value: s.name })));
  const trigger = (s) => (s.kind !== 'presence' ? `pose ${fmt(s.angle_deg, 1)}°`
    : s.method === 'bgs' ? `> ${fmt(s.min_height_mm, 1)} mm` : `± ${fmt(100 * s.threshold, 0)} %`);
  put($('#sensList'), ...[listNote(r)].filter(Boolean), S.sensors.length ? table([
    { k: 'name', label: 'Sensor', f: (v) => h('b', null, v) },
    { k: 'kind', label: 'Type', f: (v, s) => h('span', null, v === 'presence' ? tag(s.method || '?', 'info') : tag('alignment'), s.mode === 'zone' ? ' zone' : '') },
    { k: 'at', label: 'At', f: (v, s) => h('span', { class: 'mono small' }, s.kind === 'presence' ? atText(v) : `cam ${(s.centroid_px || []).map((x) => fmt(x, 0)).join(', ')}`) },
    { k: 'reference_signal', label: 'Reference', n: 1, f: (v, s) => (s.kind === 'presence' ? `${fmt(v, 1)} counts` : '—') },
    { k: 'threshold', label: 'Trigger', f: (v, s) => trigger(s) },
    { k: 'created', label: 'Taught', f: (v, s) => `${shortTime(v)}${s.versions > 1 ? ` · v${s.versions}` : ''}` },
    { k: 'name', label: '', f: (v, s) => h('span', null,
      s.kind === 'presence' ? h('button', { class: 'small primary', onclick: (e) => checkPresence(s.name, e.currentTarget) }, 'Check')
        : h('button', { class: 'small', onclick: () => { $('#alName').value = s.name; $('#alName').scrollIntoView({ behavior: 'smooth', block: 'center' }); } }, 'Use'),
      ' ', h('button', { class: 'small danger', onclick: () => deleteEntry('sensors', s) }, 'Delete…')) },
  ], S.sensors) : h('div', { class: 'empty' }, `No sensors taught at ${S.geo} yet.`),
  S.sensors.length ? h('div', { class: 'muted small' }, 'A sensor is valid only at the geometry it was taught at. Teaching again adds a version; the newest is used.') : null);
}

function alArgs(out) {
  const name = $('#alName').value.trim(), id = $('#alScan').value;
  if (!name) { warnBox(out, 'Name the alignment sensor.'); return null; }
  if (!id) { warnBox(out, 'Capture a scan of the part first (3D scan tab).'); return null; }
  return { name, object_id: id };
}
async function teachAlignment() {
  const out = $('#alOut');
  const args = alArgs(out);
  if (!args) return;
  const job = await runJob('teach_alignment', args, { btn: $('#alTeach'), out });
  if (!done(job)) return;
  const a = job.result;
  add(out, resultHead('Taught ', h('b', null, `"${a.name}"`), h('span', { class: 'muted' }, ' from '), scanRef(job.args.object_id)),
    kv([['Centroid', xyText(a.centroid_px), 'camera px'], ['Angle', a.angle_deg, '° (major axis; x right, y down)', 2],
      ['Extents', h('span', null, (a.extents_px || []).map((v) => fmt(v, 1)).join(' × ')), 'camera px (major × minor)'],
      ['Area', a.area_px, 'camera px', 0], ['Axis ratio', a.axis_ratio, '(the angle needs ≥ 1.15)', 2]]),
    warningsList(a.warnings));
  refreshSensors();
}
async function checkAlignment() {
  const out = $('#alOut');
  const args = alArgs(out);
  if (!args) return;
  const tmm = num($('#alTolMm').value), tpx = num($('#alTolPx').value), tdeg = num($('#alTolDeg').value);
  if (tmm > 0) args.tol_mm = tmm;
  if (tpx > 0) args.tol_px = tpx;
  if (tdeg > 0) args.tol_deg = tdeg;
  const job = await runJob('check_alignment', args, { btn: $('#alCheck'), out });
  if (!done(job)) return;
  const r = job.result;
  const mm = r.position_units === 'mm';
  const judged = (v) => (v == null ? tag('not judged', 'warn') : v ? tag('ok', 'ok') : tag('out', 'bad'));
  add(out, resultHead('Checked ', scanRef(job.args.object_id), h('span', { class: 'muted' }, ` against "${r.name}"`)),
    h('div', { class: 'row' }, sensorBadge(r.verdict), verdictWhy(r.verdict, r.warnings), tag(`position in ${r.position_units}`, mm ? 'ok' : 'warn')),
    h('div', { class: 'split even' },
      h('div', null,
        kv([
          mm ? ['dx', r.dx_mm, 'mm', 3] : null,
          mm ? ['dy', r.dy_mm, 'mm', 3] : null,
          mm ? ['Offset', h('span', null, fmt(r.offset_mm, 3), ' ', judged(r.position_ok)), `mm (tolerance ${fmt(r.tol_mm, 2)} mm)`] : null,
          ['dx', r.dx_px, 'camera px', 2],
          ['dy', r.dy_px, 'camera px', 2],
          mm ? null : ['Offset', h('span', null, fmt(r.offset_px, 2), ' ', judged(r.position_ok)),
            r.tol_px != null ? `camera px (tolerance ${fmt(r.tol_px, 1)} px)` : 'camera px (no px tolerance given: not judged)'],
          ['dθ', h('span', null, fmt(r.dtheta_deg, 2), ' ', judged(r.angle_ok)), `° (tolerance ${fmt(r.tol_deg, 1)}°)`],
          ['Area vs taught', r.area_ratio, '×', 3],
          mm ? ['Lateral scale', r.lateral_px_per_mm, 'camera px per mm', 3] : null,
        ]),
        warningsList(r.warnings),
        r.json_path ? h('div', { class: 'small' }, 'report: ', fileLink(r.json_path)) : null),
      figures([r.overlay_png, 'pose: cyan = taught, yellow = this scan'])));
}

// ---------- aim: click the grid photo to fill the spot / sensor "At" field
function sensImage() { return gridImage(S.grid, Sens.view === 'dmd' ? 'dmd' : 'overlay'); }
function renderSensPick() {
  if (!Sens.picker) return;
  restoreScale();
  const g = S.grid;
  Sens.picker.set(fileUrl(sensImage()));
  $$('#sensView button').forEach((b) => b.classList.toggle('on', b.dataset.v === Sens.view));
  $$('#sensClickTo button').forEach((b) => b.classList.toggle('on', b.dataset.v === Sens.clickTo));
  $$('#sensClickAs button').forEach((b) => b.classList.toggle('on', b.dataset.v === Sens.clickAs));
  const spec = g && g.grid;
  $('#sensPickInfo').textContent = spec ? `${spec.n_cols} × ${spec.n_rows} cells of ${spec.spacing_px} DMD px` +
    (g.map_id ? '' : ' · no projector map: aim on "sent to DMD", or capture a map') : '';
  const sp = S.lastSpot;
  const at = sp && (Sens.view === 'dmd' ? sp.target_dmd : sp.centroid_px);
  Sens.picker.shapes = at ? [{ type: 'mark', pts: [at], color: cssVar('--warn'), label: 'last spot' }] : [];
  Sens.picker.draw();
}
function sensClick(p) {
  let text;
  if (Sens.clickAs === 'point') {
    if (Sens.view !== 'dmd' && !(S.grid && S.grid.map_id)) toast('A camera point needs a projector map', 'warn', 'Capture one on the Grid & zones tab, or aim on the "sent to DMD" view.');
    text = `${Sens.view === 'dmd' ? 'dmd' : 'cam'} ${p[0]},${p[1]}`;
  } else {
    const r = cellAt(p, S.grid, Sens.view === 'dmd' ? 'dmd' : 'camera');
    if (r.error) { toast(r.error, 'warn'); return; }
    text = r.cell;
  }
  $('#' + Sens.clickTo).value = text;
  $('#sensPickInfo').textContent = `${Sens.clickTo === 'spotAt' ? 'spot' : 'sensor'} at ${text}`;
}

// =================================================================== 11 OUTPUT BIN
async function refreshOutputs() {
  const kind = $('#outKind').value;
  const st = S.status;
  if (st && $('#outKind').options.length <= 1) {
    (st.output_kinds || []).forEach((k) => add($('#outKind'), h('option', { value: k }, k)));
    $('#outKind').value = kind;
  }
  if (st) {
    const P = st.paths || {};
    const copyBtn = (p) => h('button', { class: 'small ghost', onclick: () => copyText(p) }, 'copy');
    // every folder api.paths() reports (the server's presets folder replaces the engine default)
    const first = ['library', 'output', 'output_index', 'bases', 'objects', 'calibrations', 'readings', 'spectral', 'banks', 'maps', 'sessions', 'trash'];
    const keys = [...first, ...Object.keys(P).filter((k) => !first.includes(k))].filter((k) => P[k] && k !== 'presets');
    put($('#outPaths'), ...keys.flatMap((k) => [h('dt', null, k.replace('_', ' ')), h('dd', null, P[k], ' ', copyBtn(P[k]))]),
      h('dt', null, 'presets'), h('dd', null, st.presets_dir || '', ' ', copyBtn(st.presets_dir || '')));
  }
  let r;
  try { r = await api('/api/outputs' + (kind ? '?kind=' + encodeURIComponent(kind) : '')); } catch (e) { r = { items: [], error: e.message }; }
  const rows = (r.items || []).slice().reverse();
  $('#outCount').textContent = `${rows.length} file${rows.length === 1 ? '' : 's'}`;
  const link = (rel) => {
    const ext = String(rel).split('.').pop().toLowerCase();
    const title = absPath(rel);
    if (ext === 'png' || ext === 'jpg' || ext === 'jpeg') return h('a', { href: '#', title, onclick: (e) => { e.preventDefault(); openImage(rel, title); } }, baseName(rel));
    if (ext === 'ply' || ext === 'xyz') return h('span', { title }, fileLink(rel, baseName(rel), true));
    return h('span', { title }, fileLink(rel, baseName(rel)), ' ', fileLink(rel, '⤓', true));
  };
  const pathCell = (rel) => h('span', { class: 'abspath' }, h('span', { class: 'mono small muted' }, absPath(rel)), ' ',
    h('button', { class: 'small ghost', title: 'copy the absolute path', onclick: () => copyText(absPath(rel)) }, 'copy'));
  put($('#outTable'), ...[listNote(r)].filter(Boolean), rows.length ? table([
    { k: 'time', label: 'Time', f: shortTime }, { k: 'kind', label: 'Kind', f: (v) => tag(v) }, { k: 'label', label: 'Label' },
    { k: 'file', label: 'File', f: link }, { k: 'file', label: 'Absolute path', f: pathCell },
    { k: 'geometry_id', label: 'Geometry', f: (v) => h('span', { class: 'mono small' }, v) },
    { k: 'source', label: 'Source', f: (v) => h('span', { class: 'mono small muted' }, v) },
  ], rows) : h('div', { class: 'empty' }, 'Nothing in the output bin yet.'));
}
function copyText(t) {
  const fallback = () => { const ta = h('textarea', null, t); add(document.body, ta); ta.select(); try { document.execCommand('copy'); } catch (e) { /* ignore */ } ta.remove(); };
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(() => toast('copied', 'ok'), fallback); else { fallback(); toast('copied', 'ok'); }
}

// =================================================================== 12 LOG
let logTimer = null;
async function refreshLog() {
  let r;
  // the server filters BEFORE taking the last n, so browsing can't crowd real actions out
  const hide = $('#logHideList').checked;
  try { r = await api('/api/log?n=' + $('#logN').value + (hide ? '&exclude=list_library' : '')); } catch (e) { put($('#logTable'), h('div', { class: 'alert bad' }, e.message)); return; }
  $('#logPath').textContent = r.path || '';
  let recs = (r.records || []).slice().reverse();
  if (hide) recs = recs.filter((x) => x.verb !== 'list_library');
  const summary = (x) => {
    if (!x.ok) return h('span', { style: { color: cssVar('--bad') } }, x.error || 'error');
    const res = x.result;
    if (res == null) return '';
    if (typeof res !== 'object') return String(res).slice(0, 160);
    if (Array.isArray(res)) return `${res.length} item(s)`;
    const keys = ['type', 'id', 'label', 'name', 'verdict', 'result', 'gc_conf', 'valid_pct', 'quality', 'units', 'mm_per_px', 'H', 'W', 'L',
      'frames', 'focus_peak', 'duration_s'];
    return keys.filter((k) => res[k] != null && typeof res[k] !== 'object').map((k) => `${k}=${typeof res[k] === 'number' ? fmt(res[k], 3) : res[k]}`).join(' · ').slice(0, 220);
  };
  const tbody = h('tbody');
  recs.forEach((x) => {
    const row = h('tr', { class: 'click' + (x.ok ? '' : ' err') },
      h('td', { class: 'mono small' }, shortTime(x.ts)), h('td', null, x.verb), h('td', null, x.ok ? tag('ok', 'ok') : tag('error', 'bad')),
      h('td', { class: 'n' }, fmt(x.duration_s, 2)), h('td', { class: 'small' }, summary(x)));
    row.addEventListener('click', () => {
      const nx = row.nextSibling;
      if (nx && nx.classList.contains('detail')) { nx.remove(); return; }
      row.after(h('tr', { class: 'detail' }, h('td', { colspan: 5 }, h('pre', { class: 'json' }, JSON.stringify(x, null, 2)))));
    });
    add(tbody, row);
  });
  put($('#logTable'), recs.length ? h('div', { class: 'tablewrap' }, h('table', { class: 't' },
    h('thead', null, h('tr', null, h('th', null, 'Time'), h('th', null, 'Verb'), h('th', null, ''), h('th', { class: 'n' }, 's'), h('th', null, 'Result / error'))), tbody))
    : h('div', { class: 'empty' }, 'No records.'));
}
function setLogAuto() {
  clearInterval(logTimer);
  if ($('#logAuto').checked) logTimer = setInterval(() => { if (curTab === 'log') refreshLog(); }, 4000);
}

// =================================================================== sim bench (mock only)
function renderBench(b) {
  if (!b) { $('#benchState').textContent = 'real rig: no simulated bench'; return; }
  const lines = [];
  (b.objects || []).forEach((o) => lines.push(`box H${o.h_mm} W${o.w_mm} L${o.l_mm} mm @ (${o.center_mm.join(', ')}) ${o.angle_deg}°${o.albedo != null ? ` · albedo ${o.albedo}` : ''}`));
  (b.labels || []).forEach((l) => lines.push(`QR "${l.data}" ${l.size_mm} mm @ (${l.center_mm.join(', ')})`));
  put($('#benchState'), ...(lines.length ? lines.map((l) => h('div', null, l)) : [h('div', { class: 'muted' }, 'empty (like a base scene)')]));
}
async function benchDo(args, btn) {
  const job = await runJob(args ? 'mock_place' : 'mock_clear', args || {}, { btn, quiet: true });
  if (done(job)) { renderBench(job.result); toast(args ? 'placed on the simulated bench' : 'simulated bench cleared', 'ok'); }
}

// =================================================================== plotting
function niceTicks(lo, hi, n = 6) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toFixed(12)));
  return out;
}
function tickLabel(v, step) {
  if (Math.abs(v) >= 10000) return (v / 1000).toFixed(0) + 'k';
  const d = Math.max(0, Math.min(6, -Math.floor(Math.log10(step || 1))));
  return v.toFixed(d);
}
function plotXY(cv, series, o = {}) {
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth || 600, H = cv.clientHeight || 280;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const muted = cssVar('--muted'), grid = cssVar('--line'), frame = cssVar('--line2'), mono = cssVar('--mono') || 'monospace';
  const tx = o.logx ? (v) => Math.log10(v) : (v) => v;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of series) {
    for (let i = 0; i < s.x.length; i++) {
      const X = s.x[i], Y = s.y[i];
      if (X == null || Y == null || !Number.isFinite(X) || !Number.isFinite(Y) || (o.logx && X <= 0)) continue;
      const t = tx(X);
      if (t < x0) x0 = t; if (t > x1) x1 = t; if (Y < y0) y0 = Y; if (Y > y1) y1 = Y;
    }
  }
  g.font = `11px ${mono}`;
  if (!Number.isFinite(x0)) { g.fillStyle = muted; g.fillText('no valid samples', 12, 20); return; }
  if (o.y0 != null) y0 = Math.min(y0, o.y0);
  if (x1 === x0) { x0 -= 1; x1 += 1; }
  if (y1 === y0) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.06; y1 += pad; if (o.y0 == null || y0 < o.y0) y0 -= pad;
  const P = { l: 60, r: 14, t: 14, b: 40 };
  const X = (t) => P.l + ((t - x0) / (x1 - x0)) * (W - P.l - P.r);
  const Y = (v) => H - P.b - ((v - y0) / (y1 - y0)) * (H - P.t - P.b);
  g.lineWidth = 1;
  const yt = niceTicks(y0, y1, 5), ys = yt.length > 1 ? yt[1] - yt[0] : 1;
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (const v of yt) { const py = Math.round(Y(v)) + 0.5; g.strokeStyle = grid; g.beginPath(); g.moveTo(P.l, py); g.lineTo(W - P.r, py); g.stroke(); g.fillStyle = muted; g.fillText(tickLabel(v, ys), P.l - 6, py); }
  let xt;
  if (o.logx) {
    xt = [];
    for (let e = Math.floor(x0); e <= Math.ceil(x1); e++) for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (Math.log10(v) >= x0 - 1e-9 && Math.log10(v) <= x1 + 1e-9) xt.push(v); }
  } else xt = niceTicks(x0, x1, 6);
  const xs = !o.logx && xt.length > 1 ? xt[1] - xt[0] : 1;
  g.textAlign = 'center'; g.textBaseline = 'top';
  for (const v of xt) { const px = Math.round(X(tx(v))) + 0.5; g.strokeStyle = grid; g.beginPath(); g.moveTo(px, P.t); g.lineTo(px, H - P.b); g.stroke(); g.fillStyle = muted; g.fillText(o.logx ? tickLabel(v, v) : tickLabel(v, xs), px, H - P.b + 5); }
  g.strokeStyle = frame; g.strokeRect(P.l + 0.5, P.t + 0.5, W - P.l - P.r, H - P.t - P.b);
  if (o.xLabel) { g.fillStyle = muted; g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillText(o.xLabel, P.l + (W - P.l - P.r) / 2, H - 3); }
  if (o.yLabel) { g.save(); g.translate(12, P.t + (H - P.t - P.b) / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = muted; g.fillText(o.yLabel, 0, 0); g.restore(); }
  g.save();
  g.beginPath(); g.rect(P.l, P.t, W - P.l - P.r, H - P.t - P.b); g.clip();
  for (const s of series) {
    g.strokeStyle = s.color; g.fillStyle = s.color; g.lineWidth = 1.6;
    if (s.kind === 'points') {
      for (let i = 0; i < s.x.length; i++) {
        const xv = s.x[i], yv = s.y[i];
        if (xv == null || yv == null || !Number.isFinite(xv) || !Number.isFinite(yv) || (o.logx && xv <= 0)) continue;
        g.beginPath(); g.arc(X(tx(xv)), Y(yv), 3, 0, Math.PI * 2); g.fill();
      }
    } else {
      g.beginPath();
      let pen = false;
      for (let i = 0; i < s.x.length; i++) {
        const xv = s.x[i], yv = s.y[i];
        if (xv == null || yv == null || !Number.isFinite(xv) || !Number.isFinite(yv) || (o.logx && xv <= 0)) { pen = false; continue; }
        const px = X(tx(xv)), py = Y(yv);
        if (pen) g.lineTo(px, py); else g.moveTo(px, py);
        pen = true;
      }
      g.stroke();
    }
  }
  g.restore();
  const labeled = series.filter((s) => s.label);
  if (labeled.length > 1) {
    g.textAlign = 'left'; g.textBaseline = 'middle';
    labeled.forEach((s, i) => { const y = P.t + 10 + i * 15; g.fillStyle = s.color; g.fillRect(P.l + 10, y - 4, 12, 8); g.fillStyle = muted; g.fillText(s.label, P.l + 27, y); });
  }
}

// =================================================================== init
function bind() {
  on('#themeBtn', 'click', () => {
    const t = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = t;
    LS.set('theme', t);
    Cloud.draw();
    renderZonesShapes();
  });
  on('#benchBtn', 'click', () => { $('#bench').hidden = !$('#bench').hidden; if (!$('#bench').hidden) renderBench(S.status && S.status.bench); });
  on('#benchClose', 'click', () => { $('#bench').hidden = true; });
  on('#benchValidation', 'click', (e) => benchDo({ object: 'validation' }, e.currentTarget));
  on('#benchBox', 'click', (e) => benchDo({ object: 'box', h_mm: $('#bxH').value, w_mm: $('#bxW').value, l_mm: $('#bxL').value,
    x_mm: $('#bxX').value, y_mm: $('#bxY').value, angle_deg: $('#bxA').value, albedo: $('#bxAlb').value }, e.currentTarget));
  on('#benchLabel', 'click', (e) => benchDo({ object: 'label', data: $('#qrData').value, x_mm: $('#qrX').value, y_mm: $('#qrY').value, size_mm: $('#qrS').value }, e.currentTarget));
  on('#benchClear', 'click', (e) => benchDo(null, e.currentTarget));
  on('#modal', 'click', (e) => { if (e.target.id === 'modal') closeModal(); });
  on('#modalClose', 'click', closeModal);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeModal(); $('#bench').hidden = true; $('#jobMenu').hidden = true; } });
  on('#jobInd', 'click', () => { const m = $('#jobMenu'); m.hidden = !m.hidden; renderJobMenu(true); });
  on('#jobInd', 'keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#jobInd').click(); } });

  // setup
  on('#presetEdit', 'click', () => openEditor($('#presetSel').value));
  on('#presetLoad', 'click', () => {
    const name = $('#presetSel').value;
    if (!name) return;
    loadPreset(name);
  });
  on('#paAdd', 'click', addPreset);
  on('#saveBtn', 'click', () => savePreset(false));
  on('#saveLoadBtn', 'click', () => savePreset(true));
  on('#revertBtn', 'click', () => { if (S.loaded) { S.form = clone(S.loaded); clearInvalid(); fillForm(); put($('#saveOut')); } });
  on('#saveName', 'input', (e) => { if (S.form) { S.form.name = e.target.value.trim(); formChanged(); } });
  on('#saveGeo', 'input', (e) => { if (S.form) { S.form.geometry_id = e.target.value.trim(); formChanged(); } });
  on('#saveOverwrite', 'change', renderNameHint);

  // focus
  on('#liveStart', 'click', liveStart);
  on('#liveStop', 'click', liveStop);
  ['#livePattern', '#liveColour', '#liveLevel', '#livePeriod'].forEach((s) => on(s, 'input', liveSettingsChanged));
  on('#focusReset', 'click', () => api('/api/live/settings', { body: { reset_peak: true } }).catch(() => {}));
  on('#sigBtn', 'click', checkSignal);
  on('#gammaBtn', 'click', measureGamma);
  on('#tuneBtn', 'click', runTune);

  // bases
  on('#baseCapture', 'click', captureBase);
  on('#basesRefresh', 'click', refreshBases);
  on('#basesAll', 'change', renderBases);
  $$('select[data-bases]').forEach((s) => on(s, 'change', () => { s.dataset.touched = '1'; }));
  // a scan picked by hand stays picked when other clients add scans (own captures still move it)
  $$('select[data-objects]').forEach((s) => on(s, 'change', () => { s.dataset.touched = '1'; }));

  // 3D scan
  on('#objCapture', 'click', (e) => captureObject({ label: $('#objLabel').value, base_id: $('#objBase').value, H: $('#objH').value, W: $('#objW').value,
    L: $('#objL').value, zone: $('#objZone').value.trim().toUpperCase(), btn: e.currentTarget, out: $('#objOut') }));
  on('#objRefresh', 'click', refreshObjects);
  Cloud = new CloudView($('#cloud'));
  on('#cloudMax', 'change', () => { if (S.cloudObj) showCloud(S.cloudObj); });
  on('#cloudZex', 'input', (e) => { Cloud.zexMul = Math.pow(10, Number(e.target.value)); $('#cloudZexV').textContent = zexLabel(); Cloud.draw(); });
  on('#cloudSize', 'input', (e) => { Cloud.psize = Number(e.target.value); Cloud.draw(); });
  on('#cloudReset', 'click', () => Cloud.reset());
  on('#cloudPly', 'click', (e) => { if (S.cloudObj) reconstruct(S.cloudObj, 'ply', e.currentTarget); else toast('pick a scan first', 'warn'); });
  on('#cloudXyz', 'click', (e) => { if (S.cloudObj) reconstruct(S.cloudObj, 'xyz', e.currentTarget); else toast('pick a scan first', 'warn'); });
  Prof.picker = new Picker($('#profPick'), { mode: 'line', onShape: (s) => {
    $('#profMode').value = 'points';
    $('#profHint').textContent = s.pts.length === 2 ? `Line p0 (${s.pts[0].join(', ')}) → p1 (${s.pts[1].join(', ')}) camera px` : 'Click the second point.';
  } });
  on('#profScan', 'change', loadProfileImage);
  on('#profBtn', 'click', runProfile);

  // measure
  on('#calRefresh', 'click', () => { refreshObjects(); refreshCals(); });
  on('#calBtn', 'click', calibrate);
  on('#measBtn', 'click', measure);

  // defects
  on('#inspBtn', 'click', inspect);
  on('#cmpBtn', 'click', compare);
  on('#bankAdd', 'click', bankAdd);
  on('#bankCheck', 'click', bankCheck);
  on('#bankRefresh', 'click', refreshBanks);
  if (S.status && S.status.preset) $('#cmpMm').placeholder = `preset ${S.status.preset.tolerance.ch_fail_mm}`;

  // codes
  on('#codeBtn', 'click', readCodes);
  on('#codeRefresh', 'click', refreshCodeReadings);

  // spectral
  on('#specBtn', 'click', spectralCapture);
  on('#specRefresh', 'click', () => refreshReadings());
  on('#specDelete', 'click', deleteReading);
  on('#specReading', 'change', showReading);
  Spec.picker = new Picker($('#specPick'), { mode: 'rect', onShape: (s) => {
    Spec.roi = s.pts;
    const [a, b] = s.pts;
    $('#specRoiText').textContent = `ROI x ${Math.min(a[0], b[0])}–${Math.max(a[0], b[0])}, y ${Math.min(a[1], b[1])}–${Math.max(a[1], b[1])} (${Math.abs(b[0] - a[0])} × ${Math.abs(b[1] - a[1])} camera px)`;
  } });
  on('#matLearn', 'click', learnMaterial);
  on('#matRefresh', 'click', refreshMaterials);
  on('#clsBtn', 'click', classify);
  on('#sepBtn', 'click', separateLight);

  // grid & zones
  Grid.picker = new Picker($('#gridPick'), { mode: 'pick', onShape: (s) => gridClick(s.pts[0]) });
  Grid.zonePicker = new Picker($('#zonePick'), { mode: 'rect', onShape: addZoneShape });
  on('#mapBtn', 'click', captureMap);
  on('#mapRefresh', 'click', refreshMaps);
  on('#gridBtn', 'click', projectGrid);
  on('#gridView', 'click', (e) => { const b = e.target.closest('button'); if (b) { Grid.view = b.dataset.v; renderGrid(); } });
  on('#gridEnlarge', 'click', () => { const rel = gridImage(S.grid, Grid.view); if (rel) openImage(rel, 'grid photo'); });
  on('#zoneText', 'input', () => { zoneTextChanged(); renderGrid(); });
  on('#zoneScan', 'click', scanZone);
  on('#agentCopy', 'click', () => copyText($('#agentBrief').value));
  on('#zoneTool', 'click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    Grid.zonePicker.mode = b.dataset.v;
    Grid.zonePicker.clear();
    $$('#zoneTool button').forEach((x) => x.classList.toggle('on', x === b));
  });
  on('#zoneClear', 'click', () => { S.zones = []; renderZones(); });
  on('#zoneProject', 'click', projectZones);
  on('#atlasImport', 'click', importAtlas);
  on('#atlasRefresh', 'click', refreshAtlases);
  on('#floodBtn', 'click', floodVerify);
  on('#glareBtn', 'click', antiGlare);
  on('#moBtn', 'click', multiOrient);

  // cylinder & 3D, fast mode
  on('#meshBtn', 'click', exportMesh);
  on('#unwrapBtn', 'click', cellUnwrap);
  on('#dentBtn', 'click', cellDents);
  on('#mapsBtn', 'click', bakeMaps);
  on('#chkBtn', 'click', checkerScan);
  on('#flashBtn', 'click', makeFlashPack);

  // sensors
  Sens.picker = new Picker($('#sensPick'), { mode: 'pick', onShape: (s) => sensClick(s.pts[0]) });
  on('#spotSizeBtn', 'click', spotSize);
  on('#spotMeasUse', 'click', useMeasuredScale);
  on('#spotK', 'change', rememberScale);
  on('#spotBtn', 'click', projectSpot);
  on('#presTeach', 'click', teachPresence);
  on('#sensRefresh', 'click', refreshSensors);
  on('#alTeach', 'click', teachAlignment);
  on('#alCheck', 'click', checkAlignment);
  for (const [sel, key] of [['#sensView', 'view'], ['#sensClickTo', 'clickTo'], ['#sensClickAs', 'clickAs']]) {
    on(sel, 'click', (e) => { const b = e.target.closest('button'); if (b) { Sens[key] = b.dataset.v; renderSensPick(); } });
  }
  on('#sensEnlarge', 'click', () => { const rel = sensImage(); if (rel) openImage(rel, 'grid photo'); });

  // outputs + log
  on('#outKind', 'change', refreshOutputs);
  on('#outRefresh', 'click', refreshOutputs);
  on('#logRefresh', 'click', refreshLog);
  on('#logN', 'change', refreshLog);
  on('#logHideList', 'change', refreshLog);
  on('#logAuto', 'change', setLogAuto);

  window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));
}

// =================================================================== 11 CELL & 3D
const optNum = (sel) => { const v = num($(sel).value); return Number.isFinite(v) ? v : null; };
/** Links a job returned: every PNG as a figure, captioned by its file name. */
function linkFigures(links) {
  return figures(...Object.entries(links || {}).filter(([, v]) => /\.png$/i.test(v || '')).map(([k, v]) => [v, k.replace(/\.png$/i, '').replace(/_/g, ' ')]));
}
/** Why a cell-dent verdict is INDETERMINATE: the engine's own words (rule 6, arc coverage, ...). */
function cellWhy(verdict, warnings) {
  if (verdict !== 'INDETERMINATE') return null;
  const own = verdictWhy(verdict, warnings);
  if (own) return own;
  const ws = warnings || [];
  const w = ws.find((x) => /verdict|rule 6|calibrat/i.test(x)) || ws[0];
  return w ? h('span', { class: 'why' }, w) : h('span', { class: 'why' }, 'the engine could not decide (see warnings)');
}
async function exportMesh() {
  const out = $('#meshOut');
  const kind = $('#meshKind').value;
  const args = { object_id: $('#cellScan').value, kind, fmt: $('#meshFmt').value, stride: Number($('#meshStride').value) || 2 };
  if (kind === 'cylinder' && optNum('#cellRadius') != null) args.radius_mm = optNum('#cellRadius');
  const job = await runJob('export_mesh', args, { btn: $('#meshBtn'), out });
  if (!done(job)) return;
  const r = job.result, m = r.meta || {};
  const units = (m.units && m.units.xyz) || '?';
  add(out, h('div', { class: 'alert ' + (m.finite_vertices === false ? 'bad' : 'ok') },
    h('b', null, `${kind} mesh (${r.fmt.toUpperCase()})`), ` ${fmt(m.n_vertices, 0)} vertices · ${fmt(m.n_triangles, 0)} triangles · units ${units}`),
  kv([['finite vertices', m.finite_vertices == null ? '—' : (m.finite_vertices ? 'yes' : 'NO')], ['to scale', m.to_scale == null ? '—' : (m.to_scale ? 'yes (mm)' : 'no')],
    m.arc_deg != null ? ['seen arc', m.arc_deg, 'deg', 1] : null, m.cell_height != null ? ['end-face height', m.cell_height, units, 2] : null,
    m.radius != null ? ['radius', m.radius, units, 3] : null, ['calibration', h('span', { class: 'mono small' }, m.calibration_id || '—')]]),
  m.note ? h('div', { class: 'muted small' }, m.note) : null,
  units !== 'mm' ? h('div', { class: 'alert warn' }, 'Uncalibrated: coordinates are not mm (rule 6). The shape is right, the scale is not.') : null,
  h('div', { class: 'row' }, fileLink(r.file, `Download ${baseName(r.file)}`, true), r.meta_file ? fileLink(r.meta_file, 'meta JSON') : null),
  h('div', { class: 'muted small mono' }, absPath(r.file)));
}
async function cellUnwrap() {
  const out = $('#unwrapOut');
  const job = await runJob('cell_unwrap', { object_id: $('#cellScan').value, radius_mm: optNum('#cellRadius') }, { btn: $('#unwrapBtn'), out });
  if (!done(job)) return;
  const u = job.result;
  add(out, h('div', { class: 'tiles2' },
    metric('seen arc', u.arc_deg, 'deg', u.arc_deg < 90 ? 'warn' : '', 'open arc only', 1),
    metric('end-face height', u.cell_height, u.units, '', '', 2),
    metric('radius', u.radius, u.units, '', u.units === 'mm' ? 'fitted' : 'relief px', 3),
    metric('fit rms', u.fit && u.fit.rms, u.units, '', '', 4)),
  figures([job.links && job.links.png, 'dent heatmap (deviation from the fitted can)']),
  warningsList(u.warnings));
}
async function cellDents() {
  const out = $('#dentOut');
  const job = await runJob('cell_dents', { object_id: $('#cellScan').value, radius_mm: optNum('#cellRadius'), n_sigma: optNum('#dentSigma'),
    min_area_bins: optNum('#dentArea'), depth_trigger_mm: optNum('#dentTrigger') }, { btn: $('#dentBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  const u = r.units;
  add(out, h('div', { class: 'row' }, verdictBadge(r.verdict), cellWhy(r.verdict, r.warnings)),
    kv([['seen arc', r.arc_deg, 'deg', 1], ['noise σ', r.noise_sigma, u, 4], ['threshold', r.threshold, u, 4], ['trigger', r.trigger, r.trigger == null ? '' : u, 3]]),
    (r.dents || []).length ? table([
      { k: 'theta_deg', label: 'θ (deg)', n: true, f: (v) => fmt(v, 1) }, { k: 'z', label: `z (${u})`, n: true, f: (v) => fmt(v, 2) },
      { k: 'depth', label: `depth (${u})`, n: true, f: (v) => fmt(v, 3) }, { k: 'area', label: `area (${u}²)`, n: true, f: (v) => fmt(v, 2) },
      { k: 'snr', label: 'SNR', n: true, f: (v) => fmt(v, 1) },
      { k: 'exceeds_trigger', label: 'over trigger', f: (v) => (v == null ? '—' : (v ? tag('yes', 'bad') : 'no')) },
    ], r.dents) : h('div', { class: 'muted small' }, 'No dents found on the seen arc.'),
    figures([job.links && job.links.png, 'dents on the unwrapped cylinder']),
    warningsList(r.warnings));
}
async function refreshCellPickers() {
  const [sp, li] = await Promise.all([listKind('spectral'), listKind('lights')]);
  const specs = (sp.items || []).slice().reverse(), lights = (li.items || []).slice().reverse();
  const fill = (sel, items, none) => {
    const keep = $(sel).value;
    put($(sel), h('option', { value: '' }, none), items.map((x) => h('option', { value: x.id }, `${x.label || x.id} · ${shortTime(x.created)}`)));
    if ([...$(sel).options].some((o) => o.value === keep)) $(sel).value = keep;
  };
  fill('#mapsSpectral', specs, '— no albedo —');
  fill('#mapsGolden', specs, '— no dust —');
  fill('#mapsLight', lights, '— no subsurface —');
}
const MAP_ORDER = ['height', 'cavity', 'normal', 'validity', 'density', 'albedo', 'subsurface', 'dust'];
async function bakeMaps() {
  const out = $('#mapsOut');
  const job = await runJob('bake_maps', { object_id: $('#cellScan').value, spectral_id: $('#mapsSpectral').value || null,
    light_id: $('#mapsLight').value || null, golden_id: $('#mapsGolden').value || null }, { btn: $('#mapsBtn'), out });
  if (!done(job)) return;
  const m = job.result, png = m.png_paths || {};
  add(out, h('div', { class: 'alert ' + (m.calibrated ? 'ok' : 'warn') }, h('b', null, 'Map stack baked'),
    m.calibrated ? ' height in mm' : ' height in relief px (no depth calibration, rule 6)'),
  figures(...MAP_ORDER.filter((n) => png[n]).map((n) => [png[n], `${n}${m.units && m.units[n] ? ' · ' + m.units[n] : ''}`])),
  warningsList(m.warnings));
}
async function checkerScan() {
  const out = $('#chkOut');
  const job = await runJob('checker_scan', { square_px: Number($('#chkSquare').value), shifts: Number($('#chkShifts').value) }, { btn: $('#chkBtn'), out });
  if (!done(job)) return;
  const c = job.result, md = c.median || {};
  add(out, h('div', { class: 'tiles2' },
    metric('corners', c.n_corners, '', c.n_corners ? 'ok' : 'bad', `${c.n_cells} cells`, 0),
    metric('camera px / square', c.cam_square_px, 'px', '', `${c.square_px} DMD px squares`, 1),
    metric('stretch', md.stretch, '', '', 'median, vs ' + (c.stretch_ref || 'median'), 3),
    metric('contrast', md.contrast, '', '', 'median, 0..1', 3),
    metric('anisotropy', md.anisotropy, '', '', 'mag_x / mag_y', 3),
    metric('skew', md.skew_deg, 'deg', '', 'median', 2)),
  figures([(c.png_paths || {}).stretch, 'stretch'], [(c.png_paths || {}).contrast, 'contrast']),
  warningsList(c.warnings));
}

// =================================================================== Surfer (grid & zones)
function regionArg(text) {
  const t = String(text || '').trim();
  const m = /^\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*$/.exec(t);
  if (m) return m.slice(1).map(Number);
  return /^atlas:/i.test(t) ? 'atlas:' + t.slice(6).trim() : zoneForEngine(t, S.grid);
}
function parseCarriers(text) {
  const parts = String(text || '').split(/[;|]/).map((s) => s.trim()).filter(Boolean);
  if (!parts.length) return null;
  const out = parts.map((p) => p.split(/[,\s]+/).filter(Boolean).map(Number));
  if (out.some((c) => (c.length !== 2 && c.length !== 3) || c.some((v) => !Number.isFinite(v)))) throw new Error('carriers are fx,fy[,z] separated by ;');
  return out;
}
async function refreshAtlases() {
  const r = await listKind('atlases', S.geo);
  const rows = (r.items || []).flatMap((a) => (a.regions || []).map((g) => ({ ...g, atlas: a.name, step_px: a.step_px })));
  put($('#atlasList'), listNote(r), rows.length ? table([
    { k: 'spec', label: 'Zone spec', f: (v) => h('a', { href: '#', class: 'mono', onclick: (e) => { e.preventDefault(); $('#surfRegion').value = v; } }, v) },
    { k: 'rect', label: 'DMD rect' }, { k: 'zone', label: 'Grid', f: (v) => v || '—' }, { k: 'label', label: 'Atlas label' },
    { k: 'spec', label: '', f: (v) => h('button', { class: 'small ghost', onclick: () => { $('#zoneText').value = v; zoneTextChanged(); } }, 'Use for zone scan') },
  ], rows) : (r.error ? null : h('div', { class: 'muted small' }, 'No atlas imported for this geometry yet.')));
}
async function importAtlas() {
  const out = $('#atlasOut');
  const job = await runJob('import_atlas', { path: $('#atlasPath').value.trim(), name: $('#atlasName').value.trim() }, { btn: $('#atlasImport'), out });
  if (!done(job)) return;
  const r = job.result;
  add(out, h('div', { class: 'alert ok' }, h('b', null, `Atlas "${r.name}" imported`), ` ${((r.atlas && r.atlas.regions) || []).length} region(s); the source file was only read.`),
    h('div', { class: 'muted small mono' }, absPath(r.file)), warningsList(r.atlas && r.atlas.warnings));
  refreshAtlases();
}
async function floodVerify() {
  const out = $('#floodOut');
  const job = await runJob('flood_verify', { region: regionArg($('#surfRegion').value) }, { btn: $('#floodBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  add(out, h('div', { class: 'row' }, h('span', { class: 'verdict ' + (r.lands ? 'ok' : 'bad') }, r.lands ? 'LANDS' : 'MISSES'), h('span', null, r.verdict)),
    kv([['region', h('span', { class: 'mono' }, r.region)], ['lit area', r.lit_area_px, 'camera px', 0], ['lit bbox', h('span', { class: 'mono' }, fmtCell(r.lit_bbox))],
      ['lit mean', r.lit_mean_counts, 'counts', 1], ['outside mean', r.outside_mean_counts, 'counts', 1], r.expected_overlap != null ? ['map overlap', r.expected_overlap, '', 3] : null]),
    figures([r.camera_png, 'camera photo'], [r.diff_png, 'lit − dark'], [r.dmd_png, 'sent to DMD']), warningsList(r.warnings));
}
async function antiGlare() {
  const out = $('#glareOut');
  let carriers;
  try { carriers = parseCarriers($('#glareCarriers').value); } catch (e) { warnBox(out, e.message); return; }
  const job = await runJob('anti_glare_sweep', { region: regionArg($('#surfRegion').value), carriers }, { btn: $('#glareBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  add(out, h('div', { class: 'alert ' + (r.lock ? 'ok' : 'warn') }, h('b', null, r.lock ? `Lock fx ${fmt(r.lock[0])}, fy ${fmt(r.lock[1])}, z ${fmt(r.lock[2])}` : 'No lock'), ' ', r.wording),
    (r.scores || []).length ? table(Object.keys(r.scores[0]).map((k) => ({ k, label: k, n: typeof r.scores[0][k] === 'number' })), r.scores) : null,
    kv([['flat flood clipping', 100 * (r.baseline_clip_frac || 0), '%', 2]]), linkFigures(job.links), warningsList(r.warnings));
}
async function multiOrient() {
  const out = $('#moOut');
  const ors = $('#moOrients').value.split(/[,\s]+/).map((s) => s.trim().toLowerCase()).filter(Boolean);
  const job = await runJob('multi_orientation_scan', { region: regionArg($('#surfRegion').value), orientations: ors, period_px: Number($('#moPeriod').value) || 24 },
    { btn: $('#moBtn'), out });
  if (!done(job)) return;
  const r = job.result;
  add(out, kv(Object.entries(r.stats || {}).map(([k, v]) => [k.replace(/_/g, ' '), typeof v === 'object' && v !== null ? h('span', { class: 'mono small' }, fmtCell(v)) : v])),
    h('div', { class: 'muted small' }, r.note), linkFigures(job.links), warningsList(r.warnings));
}

// =================================================================== fast mode (focus & signal)
async function makeFlashPack() {
  const out = $('#flashOut');
  const job = await runJob('make_flash_pack', { axes: [$('#flashAxis').value], video_port: $('#flashVideo').checked }, { btn: $('#flashBtn'), out });
  if (!done(job)) return;
  const r = job.result, m = r.manifest || {};
  add(out, h('div', { class: 'alert ok' }, h('b', null, 'Pack written (files only; the rig was not touched)')),
    kv([['folder', h('span', { class: 'mono small' }, r.folder)], ['files', (r.files || []).length, '', 0],
      ...Object.entries(m).slice(0, 8).map(([k, v]) => [k.replace(/_/g, ' '), typeof v === 'number' ? v : h('span', { class: 'mono small' }, String(v))])]),
    h('div', { class: 'alert info' }, r.doc_summary), h('div', { class: 'muted small' }, 'Full procedure: ', h('span', { class: 'mono' }, r.doc), ' and procedure.txt in the pack folder.'));
}

async function statusLoop() {
  for (;;) {
    await sleep(document.hidden ? 5000 : 1500);
    await refreshStatus();
  }
}

async function init() {
  document.documentElement.dataset.theme = LS.get('theme', 'dark');
  // the header wraps on narrow windows: sticky nav / side panels follow its real height
  const hdr = () => document.documentElement.style.setProperty('--hdr', $('.top').offsetHeight + 'px');
  hdr();
  if (window.ResizeObserver) new ResizeObserver(hdr).observe($('.top'));
  buildForm();
  bind();
  renderZones();
  await refreshStatus();
  S.grid = LS.get('grid.' + S.geo, null);
  if (S.status && S.status.preset) $('#cmpMm').placeholder = `preset ${S.status.preset.tolerance.ch_fail_mm}`;
  await refreshBases();
  showTab(location.hash.slice(1) || LS.get('tab', 'setup'));
  statusLoop();
}
document.addEventListener('DOMContentLoaded', init);
