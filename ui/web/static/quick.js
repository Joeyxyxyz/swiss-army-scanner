// Operator page (/quick). Every button and key sends one command to the station command bus
// (POST /api/cmd/<name>, engine/control/commands.py), the same path a PLC / HMI gateway uses;
// the page then follows the bus state and the job it started. No inline handlers (CSP).
'use strict';

const $ = (sel) => document.querySelector(sel);
const state = { setupBusy: false, mode: null, stations: [], bus: null, runJobId: null, homeJobId: null,
  shownRun: null, shownHome: null };

async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    opt.headers['Content-Type'] = 'application/json';
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok && r.status !== 409) throw new Error(data.error || `${r.status}`);
  return data;
}

function lamp(text, cls) {
  const el = $('#qLamp');
  el.textContent = text;
  el.className = `qlamp lamp-${cls}`;
}

function steps(lights, camera, action) {
  $('#qStepLights').className = lights || '';
  $('#qStepCamera').className = camera || '';
  $('#qStepAction').className = action || '';
}

function confirmed() { return state.mode === 'mock' || $('#qConfirm').checked; }

function refreshButtons() {
  const b = state.bus || { state: 'READY' };
  const s = b.state;
  const idle = s === 'READY' || s === 'ARMED';
  const st = state.stations.find((x) => x.config === $('#qStation').value);
  const usable = st && !st.error;
  $('#qConfirmRow').hidden = state.mode !== 'real';
  $('#qStart').disabled = !idle || !usable || !confirmed();
  $('#qAbort').disabled = false;                       // ABORT is never disabled
  $('#qHome').disabled = !(idle || s === 'FAULT' || s === 'ABORTING');
  $('#qArm').disabled = !(s === 'READY' || s === 'ARMED') || !usable || (s === 'READY' && !confirmed());
  $('#qArm').textContent = s === 'ARMED' ? 'DISARM' : 'ARM';
  $('#qArm').classList.toggle('on', s === 'ARMED');
  $('#qTrigger').disabled = s !== 'ARMED';
  $('#qStation').disabled = !(idle || s === 'FAULT');
  $('#qBusState').textContent = s;
  for (const id of ['#qGrid', '#qZones', '#qLive', '#qSignal']) {
    $(id).disabled = !(idle || s === 'FAULT') || state.setupBusy;       // setup only when idle
  }
  for (const el of document.querySelectorAll('#qSignals span')) {
    el.classList.toggle('on', !!(b.signals && b.signals[el.dataset.sig]));
  }
}

function showStation() {
  const st = state.stations.find((s) => s.config === $('#qStation').value);
  let txt = '';
  if (st && st.error) txt = `Not usable: ${st.error}`;
  else if (st) {
    txt = `${st.job_type}${st.measure ? ' / ' + st.measure : ''}, part ${st.part_label}`
      + (st.threshold_mm != null ? `, threshold ${st.threshold_mm} mm` : '')
      + (st.preset ? `, preset ${st.preset}` : '');
  }
  $('#qStationInfo').textContent = txt;
  $('#qZone').textContent = (st && st.zone) || 'whole field';
  refreshButtons();
}

async function loadStations() {
  const d = await api('GET', '/api/stations');
  state.mode = d.mode;
  state.stations = d.stations || [];
  const badge = $('#qMode');
  badge.textContent = d.mode === 'mock' ? 'MOCK' : 'REAL RIG';
  badge.className = `badge ${d.mode === 'mock' ? 'mock' : 'real'}`;
  const sel = $('#qStation');
  sel.textContent = '';
  for (const s of state.stations) {
    const o = document.createElement('option');
    o.value = s.config;
    o.textContent = s.error ? `${s.config} (invalid)` : `${s.station_id} - ${s.part_label}`;
    sel.appendChild(o);
  }
  await pollOnce();
  if (state.bus && state.bus.workstate) sel.value = state.bus.workstate;
  showStation();
}

async function send(name, extra) {
  const body = { source: 'quick', ...(extra || {}) };
  if (['start', 'trigger', 'arm'].includes(name)) body.operator_confirmed = confirmed();
  try {
    const r = await api('POST', `/api/cmd/${name}`, body);
    state.bus = r.bus;
    trackJobs(r.jobs);
    if (!r.accepted) $('#qReason').textContent = `${r.command}: ${r.message}`;
    else if (name === 'start' || name === 'trigger') {
      lamp('RUNNING', 'running');
      steps('active', '', '');
      $('#qReason').textContent = '';
    } else if (name === 'abort') $('#qReason').textContent = r.message;
    else if (name === 'home') { $('#qRigStatus').textContent = 'checking...'; $('#qRigMsg').textContent = r.message; }
    if (name === 'start' || name === 'trigger' || name === 'arm') {
      if (r.accepted && state.mode === 'real' && name !== 'arm') $('#qConfirm').checked = false;
    }
  } catch (e) {
    $('#qConn').hidden = false;
  }
  refreshButtons();
}

function trackJobs(jobs) {
  for (const [id, kind] of Object.entries(jobs || {})) {
    if (kind === 'run') state.runJobId = id;
    else state.homeJobId = id;
  }
}

function fmtNum(v, unit) {
  return (v === null || v === undefined) ? '-' : `${Number(v).toFixed(3)} ${unit}`;
}

function showRun(job) {
  const r = job.result || {};
  const verdict = job.status === 'error' ? (/Cancelled/.test(job.error_type || '') ? 'ABORTED' : 'INVALID')
    : (r.result || '-');
  const cls = { PASS: 'pass', FAIL: 'fail', INVALID: 'invalid', ABORTED: 'aborted' }[verdict] || 'ready';
  lamp(verdict, cls);
  $('#qVerdict').textContent = verdict;
  const dm = r.deviation_mm || {};
  $('#qNumLabel').textContent = dm.method ? `Max deviation (${dm.method})` : 'Max deviation';
  $('#qNumber').textContent = r.deviation_units === 'mm' ? fmtNum(dm.max, 'mm')
    : (dm.max != null ? `${dm.max} ${r.deviation_units || ''}` : '-');
  $('#qTime').textContent = (r.finished || job.finished || '-').replace('T', ' ');
  $('#qRunId').textContent = r.run_id || '-';
  const reasons = [];
  if (job.status === 'error') reasons.push(job.error || 'error');
  if (r.invalid_reason) reasons.push(r.invalid_reason);
  for (const w of r.warnings || []) if (/abort/i.test(w)) reasons.push(w);
  $('#qReason').textContent = reasons.join(' | ');
  const links = job.links || {};
  const a = $('#qResultLink');
  a.hidden = !links.result_json;
  if (links.result_json) a.href = links.result_json;
  const img = $('#qPreview');
  img.hidden = !links.preview;
  if (links.preview) img.src = `${links.preview}?t=${Date.now()}`;
  steps('done', 'done', verdict === 'ABORTED' ? '' : 'done');
}

function showRig(job) {
  if (job.status === 'error') {
    $('#qRigStatus').textContent = 'ERROR';
    $('#qRigMsg').textContent = job.error || '';
    return;
  }
  const r = job.result || {};
  $('#qRigStatus').textContent = r.status || '-';
  $('#qRigMsg').textContent = [r.message, ...(r.warnings || [])].filter(Boolean).join(' | ');
}

async function pollOnce() {
  const b = await api('GET', '/api/cmd/state');
  state.bus = b;
  trackJobs(b.jobs);
  if (state.runJobId && state.shownRun !== state.runJobId) {
    const j = await api('GET', `/api/jobs/${state.runJobId}`);
    if (j.status === 'running') { lamp('RUNNING', 'running'); steps('done', 'active', ''); }
    else if (j.status === 'done' || j.status === 'error') { showRun(j); state.shownRun = j.id; }
  }
  if (state.homeJobId && state.shownHome !== state.homeJobId) {
    const j = await api('GET', `/api/jobs/${state.homeJobId}`);
    if (j.status === 'done' || j.status === 'error') { showRig(j); state.shownHome = j.id; }
  }
  if (b.state === 'HOMING') lamp('HOMING', 'running');
  else if (b.state === 'FAULT' && $('#qLamp').textContent === 'HOMING') lamp('FAULT', 'invalid');
  else if ((b.state === 'READY' || b.state === 'ARMED') && $('#qLamp').textContent === 'HOMING') {
    lamp(b.state, 'ready');
  }
  $('#qConn').hidden = true;
  refreshButtons();
}

async function loop() {
  for (;;) {
    try { await pollOnce(); } catch (e) { $('#qConn').hidden = false; }
    await new Promise((res) => setTimeout(res, 500));
  }
}

// ---- setup helpers (Lights / Camera): ordinary rig jobs, only while the station is idle
async function setupJob(verb, args, msgEl) {
  state.setupBusy = true;
  refreshButtons();
  $(msgEl).textContent = `${verb}...`;
  try {
    let j = await api('POST', '/api/jobs', { verb, args: args || {} });
    while (j.status === 'queued' || j.status === 'running') {
      await new Promise((res) => setTimeout(res, 400));
      j = await api('GET', `/api/jobs/${j.id}`);
    }
    $(msgEl).textContent = j.status === 'error' ? `${verb}: ${j.error}` : `${verb}: done`;
    return j;
  } catch (e) {
    $(msgEl).textContent = String(e.message || e);
    return null;
  } finally {
    state.setupBusy = false;
    refreshButtons();
  }
}

async function projectGrid() {
  const j = await setupJob('project_grid', {}, '#qSetupMsg');
  const img = $('#qGridImg');
  const m = j && j.status === 'done' && JSON.stringify(j.result || {}).match(/"([^"]+\.png)"/);
  const link = m && !/^[A-Za-z]:|^\//.test(m[1]) ? m[1] : null;     // library-relative only
  img.hidden = !link;
  if (link) img.src = `/files/${link}`;
}

async function showZone() {
  const st = state.stations.find((x) => x.config === $('#qStation').value);
  if (!st || !st.zone) { $('#qSetupMsg').textContent = 'this work state uses the whole field (no job.zone)'; return; }
  $('#qSetupMsg').textContent = `zone ${st.zone}: lighting it is a placeholder; use Grid & zones in the advanced UI`;
}

let liveTimer = null;
async function toggleLive() {
  if (liveTimer) {
    clearInterval(liveTimer);
    liveTimer = null;
    $('#qLive').textContent = 'Live preview';
    await api('POST', '/api/live/stop', {}).catch(() => {});
    return;
  }
  try {
    await api('POST', '/api/live/start', { pattern: 'flood', max_s: 300 });
    $('#qLive').textContent = 'Stop preview';
    const img = $('#qLiveImg');
    img.hidden = false;
    liveTimer = setInterval(() => { img.src = `/api/live.jpg?t=${Date.now()}`; }, 500);
  } catch (e) {
    $('#qCamMsg').textContent = String(e.message || e);
  }
}

async function signalCheck() {
  const j = await setupJob('check_signal', {}, '#qCamMsg');
  const r = j && j.result;
  if (r && typeof r === 'object' && 'valid_pct' in r) {
    $('#qCamMsg').textContent = `signal ${r.quality || ''}: valid ${Number(r.valid_pct).toFixed(1)} %, `
      + `gc_conf ${Number(r.gc_conf).toFixed(1)}${r.ok ? ' - ok to scan' : ' - fix the setup'}`;
  }
}

function onKey(ev) {
  const t = ev.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA'
            || t.isContentEditable)) {
    if (ev.key !== 'Escape') return;                // Esc = ABORT, even from a field
  }
  if (ev.ctrlKey || ev.altKey || ev.metaKey) return;
  const k = ev.key;
  const s = state.bus ? state.bus.state : 'READY';
  let handled = true;
  if (k === ' ' || k === 'Enter') { if (!$('#qStart').disabled) send('start'); }
  else if (k === 'Escape') send('abort');
  else if (k === 'h' || k === 'H') send('home');
  else if (k === 't' || k === 'T') send('trigger');
  else if (k === 'a' || k === 'A') send(s === 'ARMED' ? 'disarm' : 'arm');
  else if (/^[1-9]$/.test(k)) selectWorkstate(Number(k));
  else handled = false;
  if (handled) ev.preventDefault();
}

async function selectWorkstate(arg) {
  const r = await api('POST', '/api/cmd/workstate', { workstate: arg, source: 'quick' });
  state.bus = r.bus;
  if (r.accepted) $('#qStation').value = r.bus.workstate;
  else $('#qReason').textContent = `${r.command}: ${r.message}`;
  showStation();
}

document.addEventListener('DOMContentLoaded', () => {
  $('#qStation').addEventListener('change', () => selectWorkstate($('#qStation').value));
  $('#qConfirm').addEventListener('change', refreshButtons);
  $('#qStart').addEventListener('click', () => send('start'));
  $('#qAbort').addEventListener('click', () => send('abort'));
  $('#qHome').addEventListener('click', () => send('home'));
  $('#qTrigger').addEventListener('click', () => send('trigger'));
  $('#qArm').addEventListener('click', () => send(state.bus && state.bus.state === 'ARMED' ? 'disarm' : 'arm'));
  $('#qGrid').addEventListener('click', projectGrid);
  $('#qZones').addEventListener('click', showZone);
  $('#qLive').addEventListener('click', toggleLive);
  $('#qSignal').addEventListener('click', signalCheck);
  document.addEventListener('keydown', onKey);
  loadStations().catch(() => { $('#qConn').hidden = false; }).finally(loop);
});
