// Operator page (/quick). Every button and key sends one command to the station command bus
// (POST /api/cmd/<name>, engine/control/commands.py), the same path a PLC / HMI gateway uses;
// the page then follows the bus state and the job it started. No inline handlers (CSP).
'use strict';

const $ = (sel) => document.querySelector(sel);
const state = { setupBusy: false, mode: null, stations: [], bus: null, runJobId: null, homeJobId: null,
  shownRun: null, shownHome: null, shownWorkstate: null, loadingStations: false, lampDisp: null };

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

function lamp(text, cls, disp) {
  // disp: the disposition the lamp now shows (null: a state such as READY / RUNNING / HOMING)
  const el = $('#qLamp');
  el.textContent = text;
  el.className = `qlamp lamp-${cls}`;
  state.lampDisp = disp || null;
}

// Dispositions (engine/disposition.py): CONTINUE / REJECT / PULL (quality inspection). A run
// without a disposition (one that never wrote its result) is a doubt: PULL, never a pass.
const PULL_TEXT = 'PULL \u2013 quality inspection';
const DISPOSITIONS = ['CONTINUE', 'REJECT', 'PULL'];

function dispositionOf(job) {
  const r = job.result || {};
  const d = r.disposition;
  if (!DISPOSITIONS.includes(d)) return 'PULL';
  if (d === 'CONTINUE' && (job.status !== 'done' || r.result !== 'PASS')) return 'PULL';
  return d;
}

function pullReasons(job, verdict) {
  // the reasons a PULL names (result.json pull_reasons), else why the run has no result
  const r = job.result || {};
  const out = (r.pull_reasons || []).map((p) => {
    const where = [p.zone, p.check].filter(Boolean).join('/');
    return `${where ? where + ': ' : ''}${p.reason}${p.message ? ' (' + p.message + ')' : ''}`;
  });
  if (!out.length) out.push(job.error || r.invalid_reason || `result ${verdict}`);
  return out;
}

function showDisposition(job, verdict) {
  const d = dispositionOf(job);
  if (d === 'CONTINUE') lamp(verdict, 'pass', d);
  else if (d === 'REJECT') lamp(verdict === 'FAIL' ? 'FAIL' : 'REJECT', 'fail', d);
  else lamp(PULL_TEXT, 'pull', d);
  $('#qDisposition').textContent = d === 'PULL' ? 'PULL (quality inspection)' : d;
  const box = $('#qPull');
  box.textContent = '';
  box.hidden = d !== 'PULL';
  if (d === 'PULL') box.textContent = `Pull reasons: ${pullReasons(job, verdict).join(' | ')}`;
  return d;
}

// The lamp reads the same signals as the line (the station bus, engine/control/commands.py):
// whatever a run's job says, the lamp never shows a PASS the bus does not give - an ABORT
// after the run's verdict, a START refused on an unusable work state, a crashed job, ...
// The bus's disposition of the last cycle, while it is idle (null: none, or busy / FAULT).
function busDisposition(b) {
  if (!b || !(b.state === 'READY' || b.state === 'ARMED') || !b.last_result) return null;
  return DISPOSITIONS.includes(b.last_disposition) ? b.last_disposition : 'PULL';
}

function busWhy(b) {
  // why the bus says so: the refused START / TRIGGER that set it, else its last result
  const h = (b.history || [])[(b.history || []).length - 1];
  if (h && !h[3] && (h[2] === 'START' || h[2] === 'TRIGGER') && h[4]) return `${h[2]} refused: ${h[4]}`;
  return `station: result ${b.last_result}`
    + (b.last_result === 'ABORTED' ? ' (ABORT pressed before the cycle ended)' : '');
}

function applyBus(b) {
  const d = busDisposition(b);
  if (d === null || d === state.lampDisp) return;
  if (d === 'CONTINUE') {
    // a PASS only when the PASS output is on and the lamp shows no run of its own
    if (state.lampDisp !== null || !(b.signals && b.signals.PASS)) return;
    lamp('PASS', 'pass', d);
  } else if (d === 'REJECT') lamp('FAIL', 'fail', d);
  else lamp(PULL_TEXT, 'pull', d);
  $('#qDisposition').textContent = d === 'PULL' ? 'PULL (quality inspection)' : d;
  const box = $('#qPull');
  box.hidden = d !== 'PULL';
  box.textContent = d === 'PULL' ? `Pull reasons: ${busWhy(b)}` : '';
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
  const shown = $('#qStation').value;
  const st = state.stations.find((x) => x.config === shown);
  // usable = valid when listed, not marked unusable by the bus since, and the one the bus
  // runs (START never runs anything but what this page shows)
  const usable = st && !st.error && !(b.unusable && b.unusable[shown])
    && (!state.bus || b.workstate === shown);
  $('#qConfirmRow').hidden = state.mode !== 'real';
  $('#qStart').disabled = !idle || !usable || !confirmed();
  $('#qAbort').disabled = false;                       // ABORT is never disabled
  $('#qHome').disabled = !(idle || s === 'FAULT' || s === 'ABORTING');
  $('#qArm').disabled = s === 'ARMED' ? false : (s !== 'READY' || !usable || !confirmed());   // DISARM: always
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
  const shown = $('#qStation').value;
  const st = state.stations.find((s) => s.config === shown);
  const bad = state.bus && state.bus.unusable && state.bus.unusable[shown];
  let txt = '';
  if (state.bus && state.bus.workstate && !st) txt = `Not usable: ${state.bus.workstate} is not in the station list`;
  else if (st && st.error) txt = `Not usable: ${st.error}`;
  else if (bad) txt = `Not usable: ${bad}`;
  else if (st && st.job_type === 'recipe') {
    // a recipe work state: zones + the checks recognised there; the recipe pins its preset
    txt = `recipe ${st.recipe}: zones ${(st.zones || []).join(', ')} (${(st.checks || []).join(', ')})`
      + (st.preset ? `, preset ${st.preset}` : '')
      + (st.feedback ? ', verdict projected on the part' : '');
  } else if (st) {
    txt = `${st.job_type}${st.measure ? ' / ' + st.measure : ''}, part ${st.part_label}`
      + (st.threshold_mm != null ? `, threshold ${st.threshold_mm} mm` : '')
      + (st.preset ? `, preset ${st.preset}` : '');
  }
  $('#qStationInfo').textContent = txt;
  $('#qZone').textContent = (st && st.zone) || 'whole field';
  refreshButtons();
}

async function fetchStations() {
  // the work-state list (GET /api/stations also re-reads the configs on the server; the bus
  // never switches its selected work state because of that)
  state.loadingStations = true;
  try {
    const d = await api('GET', '/api/stations');
    state.mode = d.mode;
    state.stations = d.stations || [];
    const badge = $('#qMode');
    badge.textContent = d.mode === 'mock' ? 'MOCK' : 'REAL RIG';
    badge.className = `badge ${d.mode === 'mock' ? 'mock' : 'real'}`;
    const sel = $('#qStation');
    sel.textContent = '';
    state.stations.forEach((s, i) => {
      const o = document.createElement('option');
      o.value = s.config;
      o.textContent = `${i + 1}: ` + (s.error ? `${s.config} (invalid)` : `${s.station_id} - ${s.part_label}`);
      sel.appendChild(o);
    });
  } finally {
    state.loadingStations = false;
  }
}

function stationListChanged(b) {
  const names = state.stations.map((s) => s.config);
  return JSON.stringify(b.workstates || []) !== JSON.stringify(names);
}

async function loadStations() {
  await fetchStations();
  await pollOnce();
}

function syncWorkstate(b) {
  // the selector always shows the bus's work state (a PLC, another page or a key may have
  // changed it): what the operator confirms and what START runs are the same work state.
  // A change of work state clears the confirmation (it was for the other one).
  const sel = $('#qStation');
  const ws = b.workstate || '';
  if (sel.value !== ws) sel.value = ws;          // a work state missing from the list shows blank
  if (ws !== state.shownWorkstate) {
    if (state.shownWorkstate !== null) $('#qConfirm').checked = false;
    state.shownWorkstate = ws;
  }
  showStation();                                 // its info, usability and the buttons
}

async function send(name, extra) {
  const body = { source: 'quick', ...(extra || {}) };
  if (['start', 'trigger', 'arm'].includes(name)) {
    body.operator_confirmed = confirmed();
    body.workstate = $('#qStation').value;       // refused if the bus has another selected
  }
  try {
    const r = await api('POST', `/api/cmd/${name}`, body);
    state.bus = r.bus;
    trackJobs(r.jobs);
    syncWorkstate(r.bus);
    if (!r.accepted) {
      $('#qReason').textContent = `${r.command}: ${r.message}`;
      applyBus(r.bus);                           // e.g. a refused START: the bus PULLs
    }
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
  const disp = showDisposition(job, verdict);   // the lamp: PASS / FAIL / PULL (amber)
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
  showZones(r, disp);
  const links = job.links || {};
  const fl = $('#qFeedbackLink');
  fl.hidden = !links.feedback;
  if (links.feedback) fl.href = links.feedback;
  const a = $('#qResultLink');
  a.hidden = !links.result_json;
  if (links.result_json) a.href = links.result_json;
  const img = $('#qPreview');
  img.hidden = !links.preview;
  if (links.preview) img.src = `${links.preview}?t=${Date.now()}`;
  steps('done', 'done', verdict === 'ABORTED' ? '' : 'done');
}

function showZones(r, disp) {
  // recipe runs: one line per zone (verdict + the first failing / invalid check's message);
  // under PULL the zones that caused it are marked pull
  const ul = $('#qZoneList');
  ul.textContent = '';
  const zones = r.zones || [];
  ul.hidden = zones.length === 0;
  const pulled = new Set((r.pull_reasons || []).map((p) => p.zone).filter(Boolean));
  for (const z of zones) {
    const li = document.createElement('li');
    li.className = disp === 'PULL' && pulled.has(z.id) ? 'z-pull'
      : `z-${String(z.verdict || '').toLowerCase()}`;
    const b = document.createElement('b');
    b.textContent = `${z.id} ${z.verdict}`;
    li.appendChild(b);
    const bad = (z.checks || []).find((c) => c.verdict !== 'PASS');
    li.appendChild(document.createTextNode(bad ? `${bad.id}: ${bad.message}` : (z.checks || []).map((c) => c.id).join(', ')));
    ul.appendChild(li);
  }
  const fb = r.feedback || null;
  let t = '';
  if (fb && fb.rendered) t = `Verdict painted on the part (${fb.hold_s} s; START / HOME / ABORT clears it)`
    + (disp === 'PULL' ? ': amber frame = PULL' : '');
  else if (fb && fb.enabled && fb.refused) t = `Not projected: ${fb.refused}`;
  $('#qFeedback').textContent = t;
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

function showWarnings(b) {
  // the server's own warnings (e.g. the selected preset no longer exists: nothing loaded)
  const ws = b.warnings || [];
  $('#qWarn').hidden = !ws.length;
  $('#qWarn').textContent = ws.join(' | ');
}

function showPreset(b) {
  // the preset START runs: the work state's own, else the loaded one (run_preset)
  const run = b.run_preset || b.active_preset;
  $('#qPreset').textContent = run || 'none loaded';
  $('#qPresetLoaded').textContent = run && b.active_preset && b.active_preset !== run
    ? `(loaded now: ${b.active_preset}; the run loads ${run})` : '';
}

async function pollOnce() {
  const b = await api('GET', '/api/cmd/state');
  state.bus = b;
  if (!state.loadingStations && (b.state === 'READY' || b.state === 'FAULT') && stationListChanged(b)) {
    await fetchStations();                         // configs added / removed / renamed
  }
  syncWorkstate(b);
  showPreset(b);
  showWarnings(b);
  trackJobs(b.jobs);
  if (state.runJobId && state.shownRun !== state.runJobId) {
    const j = await api('GET', `/api/jobs/${state.runJobId}`);
    const ended = j.status === 'done' || j.status === 'error';
    // a run is shown once the bus has heard it end (an ABORT may still turn it ABORTED)
    if (j.status === 'running' || (ended && (b.state === 'RUNNING' || b.state === 'ABORTING'))) {
      lamp('RUNNING', 'running');
      steps('done', 'active', '');
    } else if (ended) { showRun(j); state.shownRun = j.id; }
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
  applyBus(b);                                   // the lamp never contradicts the outputs
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

// Space / Enter activate a focused control (a button left focused by a click, a summary,
// a link): never turn that into START, and never cancel the control's own action
const ACTIVATABLE = 'button, summary, a[href], [role="button"], [tabindex]';

function onKey(ev) {
  const t = ev.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA'
            || t.isContentEditable)) {
    if (ev.key !== 'Escape') return;                // Esc = ABORT, even from a field
  }
  if (ev.ctrlKey || ev.altKey || ev.metaKey) return;
  const k = ev.key;
  if ((k === ' ' || k === 'Enter') && t && t.closest && t.closest(ACTIVATABLE)) return;
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
  try {
    const r = await api('POST', '/api/cmd/workstate', { workstate: arg, source: 'quick' });
    state.bus = r.bus;
    $('#qReason').textContent = r.accepted ? (/disarmed/.test(r.message) ? r.message : '')
      : `${r.command}: ${r.message}`;
    syncWorkstate(r.bus);              // accepted or refused: show the bus's work state
    showPreset(r.bus);
  } catch (e) {
    $('#qConn').hidden = false;
  }
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
