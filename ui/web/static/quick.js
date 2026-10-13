// Operator page (/quick). Every button and key sends one command to the station command bus
// (POST /api/cmd/<name>, engine/control/commands.py), the same path a PLC / HMI gateway uses;
// the page then follows the bus state and the job it started. No inline handlers (CSP).
//
// Words: the STATION CONFIGURATION (stations/*.json) is what the station knows: its parameters
// and the PROFILE LIBRARY SET it can call on (long tube / short tube). The PROFILE selector
// picks the part variant. The WORK STATES are the steps of the configuration's flow: what the
// line's sensor says, what the DLP says, and the permission that lets the line continue.
//
// Operating modes (the switch at the top; the bus holds the selected one: POST /api/cmd/mode):
// STATION is all of the above. MEASURE is an instrument: a triangulation preset, what to
// measure, a region or two picked camera px -> POST /api/measure -> a table of values with
// their uncertainty and a neutral MEASURED / NOT MEASURABLE (never a verdict, never the lamp's
// PASS / FAIL). QUICK CHECK is a preset + one check + a threshold -> POST /api/quick-check ->
// the lamp shows CONTINUE / REJECT / PULL. Outside Station mode the bus starts no cycle.
// The station's keys (1-9, P, T, A, N, K) work in Station mode only: in the other modes the
// station selectors are hidden, and a key must never change what is hidden. The picker of
// Measure / Quick check always shows the LOADED preset (the bus's active_preset: the one HOME
// checks and the measurement runs with). The operator's confirmation of the physical setup
// is for one setup: a mode change or another preset clears it.
'use strict';

const $ = (sel) => document.querySelector(sel);
const state = { setupBusy: false, mode: null, stations: [], bus: null, runJobId: null, homeJobId: null,
  shownRun: null, shownHome: null, shownConfig: null, shownProfile: null, loadingStations: false,
  lampDisp: null, flowKey: null, mockKey: null, profileKey: null, cycleKey: null, resultShown: false,
  // operating mode (null until the first bus state), its job, its pickers, the picture clicks
  opMode: null, modeJobId: null, shownModeJob: null, rigKey: null, recipeKey: null,
  pick: { n: 0, corner: null }, presetBusy: false, shownPreset: null, rigAsked: null };

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

function notice(text) {
  // the last command the bus refused (or what an accepted one changed): it stays until the
  // next command is accepted - the polls never overwrite it
  const el = $('#qNotice');
  el.textContent = text || '';
  el.hidden = !text;
}

function clearChecks() {
  // nothing of an earlier DLP check stays on the panel (its zones, its picture, what was painted)
  $('#qFeedback').textContent = '';
  $('#qZoneList').textContent = '';
  $('#qZoneList').hidden = true;
  $('#qFeedbackLink').hidden = true;
  $('#qPreview').hidden = true;
}

function clearResult() {
  // the bus has forgotten the last cycle: neither the lamp nor the panel keeps it
  for (const id of ['#qVerdict', '#qDisposition', '#qNumber', '#qTime', '#qRunId']) $(id).textContent = '-';
  $('#qReason').textContent = '';
  $('#qPull').textContent = '';
  $('#qPull').hidden = true;
  $('#qResultLink').hidden = true;
  $('#qValues').hidden = true;
  $('#qValuesBody').textContent = '';
  $('#qValidity').textContent = '';
  $('#qValuesLink').hidden = true;
  $('#qPlyLink').hidden = true;
  clearChecks();
  steps('', '', '');
  state.resultShown = false;
  state.cycleKey = null;
}

// Dispositions (engine/disposition.py): CONTINUE / REJECT / PULL (quality inspection). A run
// without a disposition (one that never wrote its result) is a doubt: PULL, never a pass.
const PULL_TEXT = 'PULL – quality inspection';
const DISPOSITIONS = ['CONTINUE', 'REJECT', 'PULL'];

function dispositionOf(job) {
  const r = job.result || {};
  const d = r.disposition;
  if (!DISPOSITIONS.includes(d)) return 'PULL';
  if (d === 'CONTINUE' && (job.status !== 'done' || r.result !== 'PASS')) return 'PULL';
  return d;
}

function reasonText(p) {
  const where = [p.step, p.zone, p.check].filter(Boolean).join('/');
  return `${where ? where + ': ' : ''}${p.reason}${p.message ? ' (' + p.message + ')' : ''}`;
}

function pullReasons(job, verdict) {
  // the reasons a PULL names (result.json pull_reasons), else why the run has no result
  const r = job.result || {};
  const out = (r.pull_reasons || []).map(reasonText);
  if (!out.length) out.push(job.error || r.invalid_reason || `result ${verdict}`);
  return out;
}

function showDisposition(job, verdict) {
  const d = dispositionOf(job);
  state.resultShown = true;
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

// The cycle of a flow (the bus walks its work states): null for none / a single-job config.
function flowCycle(b) {
  const c = b && b.cycle;
  return c && !c.legacy ? c : null;
}

// The lamp reads the same signals as the line (the station bus, engine/control/commands.py):
// whatever a run's job says, the lamp never shows a PASS the bus does not give - an ABORT
// after the run's verdict, a START refused on an unusable configuration, a crashed job, ...
// The bus's disposition of the last cycle, while it is idle (null: none, or busy / FAULT).
function busDisposition(b) {
  if (!b || !(b.state === 'READY' || b.state === 'ARMED') || !b.last_result) return null;
  return DISPOSITIONS.includes(b.last_disposition) ? b.last_disposition : 'PULL';
}

function busWhy(b) {
  // why the bus says so: the refused START / TRIGGER that set it, the cycle's own pull
  // reasons (which work state, what), else its last result
  const h = (b.history || [])[(b.history || []).length - 1];
  if (h && !h[3] && (h[2] === 'START' || h[2] === 'TRIGGER') && h[4]) return `${h[2]} refused: ${h[4]}`;
  const c = flowCycle(b);
  if (c && !c.active && (c.pull_reasons || []).length) return c.pull_reasons.map(reasonText).join(' | ');
  return `station: result ${b.last_result}`
    + (b.last_result === 'ABORTED' ? ' (ABORT pressed before the cycle ended)' : '');
}

function applyBus(b) {
  // Measure / Quick check: the bus stands aside (no cycle, no PASS): the lamp is the mode's own
  if (opMode(b) !== 'station') return;
  const d = busDisposition(b);
  if (d === null) {
    // idle with no result and no cycle: the bus has forgotten the last one (another profile or
    // configuration, HOME, ARM, a START / TRIGGER that started nothing). The lamp and the
    // panel never keep a PASS the outputs no longer give.
    const idle = b && (b.state === 'READY' || b.state === 'ARMED');
    if (idle && !b.last_result && !b.cycle && state.resultShown) {
      lamp(b.state, 'ready');
      clearResult();
    }
    return;
  }
  state.resultShown = true;
  if (d === state.lampDisp) return;
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
  const c = flowCycle(b);
  // usable = valid when listed, not marked unusable by the bus since, and the one the bus
  // runs (START never runs anything but what this page shows)
  const usable = st && !st.error && !(b.unusable && b.unusable[shown])
    && (!state.bus || b.station_config === shown);
  const waitStep = s === 'WAITING' && c && c.waiting === 'step';
  const m = state.opMode || 'station';
  // a measurement / quick check of this page (or another client) is queued or running
  const modeBusy = (b.mode_jobs || []).length > 0
    || !!(state.modeJobId && state.shownModeJob !== state.modeJobId);
  const free = (s === 'READY' || s === 'FAULT') && !modeBusy;
  $('#qConfirmRow').hidden = state.mode !== 'real' || m === 'measure';
  // Station: START begins a cycle; Quick check: START runs the one check
  $('#qStart').disabled = m === 'station' ? (!idle || !usable || !confirmed()) : (!free || !confirmed());
  $('#qMeasure').disabled = !free;
  for (const k of MODES) $(MODE_BTN[k]).disabled = !(idle || s === 'FAULT') || modeBusy;
  $('#qRigPreset').disabled = !free;
  $('#qAbort').disabled = false;                       // ABORT is never disabled
  $('#qHome').disabled = !(idle || s === 'FAULT' || s === 'ABORTING');
  $('#qArm').disabled = s === 'ARMED' ? false : (s !== 'READY' || !usable || !confirmed());   // DISARM: always
  $('#qArm').textContent = s === 'ARMED' ? 'DISARM' : 'ARM';
  $('#qArm').classList.toggle('on', s === 'ARMED');
  $('#qTrigger').disabled = !(s === 'ARMED' || waitStep);
  $('#qStep').disabled = !waitStep;
  $('#qStep').classList.toggle('on', !!waitStep);
  $('#qAck').disabled = !(s === 'WAITING' && b.prompt);
  $('#qStation').disabled = !(idle || s === 'FAULT');
  $('#qProfile').disabled = !(idle || s === 'FAULT');
  $('#qBusState').textContent = s;
  for (const id of ['#qGrid', '#qZones', '#qLive', '#qSignal', '#qPaint', '#qZoneShow']) {
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
  if (state.bus && state.bus.station_config && !st) txt = `Not usable: ${state.bus.station_config} is not in the station list`;
  else if (st && st.error) txt = `Not usable: ${st.error}`;
  else if (bad) txt = `Not usable: ${bad}`;
  else if (st && st.job_type === 'flow') {
    // a configuration with a profile library set and a flow of work states
    txt = `${st.station_id}: ${(st.profiles || []).length} profile(s) (${(st.profiles || []).map((p) => p.name).join(', ')}), `
      + `${(st.flow || []).length} work state(s)` + (st.description ? ` - ${st.description}` : '');
  } else if (st && st.job_type === 'recipe') {
    // a single-job configuration verified by a recipe: zones + the checks recognised there
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
  // the station configurations (GET /api/stations also re-reads the files on the server; the
  // bus never switches its selected configuration or profile because of that)
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
      o.textContent = `${i + 1}: ` + (s.error ? `${s.config} (invalid)`
        : (s.job_type === 'flow' ? `${s.station_id} - ${(s.flow || []).length} work states`
          : `${s.station_id} - ${s.part_label}`));
      sel.appendChild(o);
    });
  } finally {
    state.loadingStations = false;
  }
}

function stationListChanged(b) {
  const names = state.stations.map((s) => s.config);
  return JSON.stringify(b.station_configs || []) !== JSON.stringify(names);
}

async function loadStations() {
  await fetchStations();
  await pollOnce();
}

function syncProfile(b) {
  // the profile selector lists the selected configuration's profile library set and always
  // shows the bus's active profile
  const sel = $('#qProfile');
  const profs = b.profiles || [];
  const key = JSON.stringify(profs.map((p) => [p.name, p.title]));
  if (key !== state.profileKey) {
    state.profileKey = key;
    sel.textContent = '';
    profs.forEach((p, i) => {
      const o = document.createElement('option');
      o.value = p.name;
      o.textContent = `${i + 1}: ${p.title || p.name}`;
      sel.appendChild(o);
    });
  }
  const want = b.profile || '';
  if (sel.value !== want) sel.value = want;
}

function syncConfig(b) {
  // the selectors always show the bus's station configuration and profile (a PLC, another
  // page or a key may have changed them): what the operator confirms and what START runs are
  // the same. A change of either clears the confirmation (it was for the other one).
  const sel = $('#qStation');
  const cfg = b.station_config || '';
  if (sel.value !== cfg) sel.value = cfg;        // a configuration missing from the list shows blank
  syncProfile(b);
  const prof = b.profile || '';
  if (cfg !== state.shownConfig || prof !== state.shownProfile) {
    if (state.shownConfig !== null) $('#qConfirm').checked = false;
    state.shownConfig = cfg;
    state.shownProfile = prof;
  }
  showStation();                                 // its info, usability and the buttons
}

// ---- work states: the steps of the selected configuration's flow

const OUTCOME_TEXT = { PENDING: '', CONFIRMED: 'CONFIRMED', CORRECTED: 'CORRECTED',
  NOT_MET: 'NOT MET', DEFECT: 'DEFECT', DOUBT: 'DOUBT', SKIPPED: 'SKIPPED', ABORTED: 'ABORTED' };

function sayBool(v) { return v === true ? 'true' : (v === false ? 'false' : 'unknown'); }

function chip(cls, label, value, extra) {
  const el = document.createElement('span');
  el.className = `qchip ${cls}`;
  const k = document.createElement('i');
  k.textContent = label;
  const v = document.createElement('b');
  v.textContent = value;
  el.appendChild(k);
  el.appendChild(v);
  if (extra) Object.assign(el.dataset, extra);
  return el;
}

function flowRows(b) {
  // the open (or last) cycle's work states when it belongs to what is selected now, else
  // the flow as the configuration defines it (nothing verified yet)
  const c = b.cycle;
  if (c && (c.steps || []).length && c.profile === b.profile
      && (!c.station_config || c.station_config === b.station_config)) {
    return c.steps.map((r) => ({ ...r, live: !!c.active && r.index === c.index }));
  }
  return (b.flow || []).map((s) => ({ ...s, outcome: 'PENDING', sensor_inputs: {}, observed: {},
    permission_granted: false, actions: [], message: '', live: false }));
}

function renderFlow(b) {
  const rows = flowRows(b);
  const inputs = b.inputs || {};
  const perms = b.permissions || {};
  // MOCK: what the simulated line's sensors say (reported for every cycle); a claim the bus
  // holds now wins. A claim belongs to one cycle: between cycles a real line's are unknown.
  const sim = (state.mode === 'mock' && b.mock && b.mock.inputs) || {};
  const says = (s) => (Object.prototype.hasOwnProperty.call(inputs, s) ? inputs[s] : sim[s]);
  const key = JSON.stringify([rows, inputs, perms, b.state, sim]);
  if (key === state.flowKey) return;
  state.flowKey = key;
  const ol = $('#qFlow');
  ol.textContent = '';
  for (const r of rows) {
    const li = document.createElement('li');
    const o = String(r.outcome || 'PENDING');
    li.className = `qws o-${o.toLowerCase()}${r.live ? ' current' : ''}`;
    const head = document.createElement('div');
    head.className = 'qws-head';
    const n = document.createElement('span');
    n.className = 'qws-n';
    n.textContent = String(r.index);
    const t = document.createElement('b');
    t.textContent = r.title || r.id;
    const out = document.createElement('span');
    out.className = 'qws-out';
    out.textContent = r.live && o === 'PENDING'
      ? (b.state === 'RUNNING' ? 'VERIFYING' : 'waits for STEP') : (OUTCOME_TEXT[o] || o);
    head.appendChild(n);
    head.appendChild(t);
    head.appendChild(out);
    li.appendChild(head);
    const chips = document.createElement('div');
    chips.className = 'qchips';
    const expects = r.expects || {};
    const names = Object.keys(expects);
    const verified = o !== 'PENDING' && o !== 'SKIPPED';
    if (names.length) {
      for (const s of names) {
        // what the sensor said when the scan started; before that, what it says now
        const said = verified ? (r.sensor_inputs || {})[s] : says(s);
        const seen = (r.observed || {})[s];
        const bad = verified && said !== seen;
        chips.appendChild(chip(`sensor${bad ? ' bad' : ''}`, 'sensor says', `${s} ${sayBool(said)}`));
        chips.appendChild(chip(`dlp${bad ? ' bad' : ''}`, 'DLP says', verified ? `${s} ${sayBool(seen)}` : '-'));
      }
    } else {
      chips.appendChild(chip('sensor none', 'sensor says', '-'));
      const d = r.dlp_result || {};
      chips.appendChild(chip('dlp', 'DLP says', verified ? (d.result || '-') : '-'));
    }
    if (r.permit) {
      const on = perms[r.permit] === true;
      chips.appendChild(chip(`permit ${on ? 'on' : 'off'}`, 'permission', `${r.permit} ${on ? 'GRANTED' : 'withheld'}`));
    } else chips.appendChild(chip('permit none', 'permission', '-'));
    li.appendChild(chips);
    const acts = (r.actions || []).map((a) => a.action + (a.disposition ? ' ' + a.disposition : '')
      + (a.action === 'prompt' ? (a.acknowledged ? ' (acknowledged)' : ' (waits for ACK)') : '')).join(' → ');
    // why a CONFIRMED work state's permission is not (or no longer) granted
    const note = r.permit && r.permission_note ? `${r.permit} ${r.permission_note}` : '';
    const msg = [r.message, note, acts].filter(Boolean).join(' | ');
    if (msg) {
      const m = document.createElement('div');
      m.className = 'qws-msg small';
      m.textContent = msg;
      li.appendChild(m);
    }
    ol.appendChild(li);
  }
  // the line's sensor claims (POST /api/io/inputs); MOCK: click one to toggle the simulated sensor
  const box = $('#qInputs');
  box.textContent = '';
  const sigs = b.signal_names || [];
  for (const s of sigs) {
    box.appendChild(chip(`sensor${state.mode === 'mock' ? ' toggle' : ''}`, s, sayBool(says(s)),
      { signal: s, value: String(says(s) === true) }));
  }
  if (!sigs.length) box.appendChild(chip('sensor none', 'none', 'this flow expects no sensor'));
  // the permissions as the line reads them (PERMIT_<NAME> outputs)
  const pbox = $('#qPermits');
  pbox.textContent = '';
  for (const name of Object.keys(perms)) {
    const el = document.createElement('span');
    el.textContent = `PERMIT_${name}`;
    el.className = perms[name] === true ? 'on' : '';
    el.dataset.sig = `PERMIT_${name}`;
    pbox.appendChild(el);
  }
}

function renderPrompt(b) {
  const p = b.prompt || null;
  $('#qPrompt').hidden = !p;
  $('#qPromptTitle').textContent = p ? `${p.index}. ${p.title}:` : '';
  $('#qPromptMsg').textContent = p ? p.message : '';
  $('#qPromptDetail').textContent = p ? [p.outcome, p.detail].filter(Boolean).join(' - ') : '';
}

function renderMock(b) {
  // MOCK only: the simulated line (what the simulated human places, what the sensors say)
  const m = b.mock || null;
  $('#qMock').hidden = !m || !(m.cases || []).length;
  if (!m) return;
  const key = JSON.stringify([m.cases, m.case, m.scene]);
  if (key === state.mockKey) return;
  state.mockKey = key;
  const sel = $('#qMockCase');
  sel.textContent = '';
  const first = document.createElement('option');
  first.value = '';
  first.textContent = m.scene ? `scene ${m.scene}` : '(default case)';
  sel.appendChild(first);
  for (const c of m.cases || []) {
    const o = document.createElement('option');
    o.value = c.name;
    o.textContent = c.name;
    o.title = c.description || '';
    sel.appendChild(o);
  }
  sel.value = m.case || '';
}

function showCycle(b) {
  // a flow's cycle: the lamp shows where it is; its end (PASS / FAIL / PULL) comes from the
  // bus outputs (applyBus), never from a single work state's own verdict
  const c = flowCycle(b);
  if (!c) return false;
  state.resultShown = true;
  const ck = c.cycle_id || c.started || '';
  if (ck !== state.cycleKey) {                   // another cycle: nothing of the last one's checks
    state.cycleKey = ck;
    clearChecks();
  }
  const n = (c.steps || []).length;
  const row = (c.steps || [])[c.index - 1] || {};
  if (c.active) {
    const at = `${c.index}/${n}`;
    if (b.state === 'ABORTING') lamp('ABORTING', 'running');
    else if (b.state === 'RUNNING') lamp(`VERIFYING ${at}`, 'running');
    else if (c.waiting === 'ack') lamp('ACK NEEDED', 'invalid');
    else lamp(`STEP ${at}`, 'wait');
    $('#qDisposition').textContent = '-';
    $('#qPull').hidden = true;
  }
  $('#qVerdict').textContent = c.result || (c.active ? 'cycle open' : '-');
  if (!c.active && c.disposition) {
    $('#qDisposition').textContent = c.disposition === 'PULL' ? 'PULL (quality inspection)' : c.disposition;
  }
  $('#qNumLabel').textContent = 'Work state';
  $('#qNumber').textContent = row.id ? `${c.index}/${n} ${row.title || row.id}: ${OUTCOME_TEXT[row.outcome] || 'pending'}` : '-';
  $('#qTime').textContent = String(c.finished || c.started || '-').replace('T', ' ');
  $('#qRunId').textContent = c.cycle_id || '-';
  const k = c.kickout;
  // how the cycle ended: a kickout action, or the cycle itself (unconfirmed, timeout, abort)
  $('#qReason').textContent = k
    ? `${k.by && k.by !== 'kickout' ? `ended ${k.disposition} (${k.by})` : `kickout ${k.disposition}`}: ${k.reason}`
      + (k.detail && k.detail !== k.reason && !String(k.reason).includes(k.detail) ? ` - ${k.detail}` : '')
    : (row.message || '');
  const lc = b.last_cycle;
  const a = $('#qResultLink');
  const link = lc && lc.cycle_id === c.cycle_id && lc.links ? lc.links.result_json : null;
  a.hidden = !link;
  if (link) a.href = link;
  return true;
}

function showStepJob(job) {
  // the DLP check of one work state: its zones, its picture and the fault image - never the lamp
  const r = job.result || {};
  showZones(r, r.disposition);
  // what was painted on the part is the cycle's project_fault action, not the check's own verdict
  const c = flowCycle(state.bus) || {};
  const row = (c.steps || []).find((s) => s.id === (r.job || {}).work_state) || {};
  const pf = (row.actions || []).find((a) => a.action === 'project_fault');
  $('#qFeedback').textContent = !pf ? ''
    : (pf.requested ? 'Fault sent to the projector: painted on the part (the next verify, HOME or ABORT clears it)'
      : `Not painted: ${pf.why || 'nothing to paint'}`);
  const links = job.links || {};
  const fl = $('#qFeedbackLink');
  fl.hidden = !links.feedback;
  if (links.feedback) fl.href = links.feedback;
  const img = $('#qPreview');
  img.hidden = !links.preview;
  if (links.preview) img.src = `${links.preview}?t=${Date.now()}`;
}

// ---- operating modes: Measure | Quick check | Station

const MODES = ['measure', 'quick_check', 'station'];
const MODE_BTN = { measure: '#qModeMeasure', quick_check: '#qModeQuick', station: '#qModeStation' };
const MODE_INFO = {
  measure: 'an instrument: numbers with units and uncertainty. No tolerance, no verdict, nothing on the line\'s outputs',
  quick_check: 'a triangulation preset + one check: CONTINUE / REJECT / PULL. No sensors, no permissions',
  station: 'station configuration, profile, work states: the line\'s cycle',
};
const MODE_ACTION = {
  measure: 'MEASURE: scan and report the numbers',
  quick_check: 'START: run the one check',
  station: 'run the cycle: START, then each work state is verified (STEP)',
};

function opMode(b) {
  // the bus's operating mode (a bus that says none runs cycles: station)
  const m = b && b.operating_mode;
  return MODES.includes(m) ? m : 'station';
}

function applyMode(b) {
  // the page always shows the bus's mode (another client or a PLC may have changed it)
  const m = opMode(b);
  for (const k of MODES) $(MODE_BTN[k]).classList.toggle('on', k === m);
  if (m === state.opMode) return;
  const first = state.opMode === null;
  state.opMode = m;
  $('#qPaneStation').hidden = m !== 'station';
  $('#qStationExtras').hidden = m !== 'station';
  $('#qPaneRig').hidden = m === 'station';
  $('#qPaneMeasure').hidden = m !== 'measure';
  $('#qPaneQuick').hidden = m !== 'quick_check';
  $('#qMeasure').hidden = m !== 'measure';
  $('#qStart').hidden = m === 'measure';
  $('#qMeasImg').hidden = true;
  $('#qModeInfo').textContent = MODE_INFO[m];
  $('#qActionText').textContent = MODE_ACTION[m];
  $('#qMockRow').hidden = state.mode !== 'mock';
  if (!first) {                                  // another mode: nothing of the last one stays
    const s = b && b.state ? b.state : 'READY';
    lamp(s, s === 'FAULT' ? 'invalid' : 'ready');
    clearResult();
    $('#qReason').textContent = '';
    notice('');
    $('#qConfirm').checked = false;              // the confirmation was for the other mode's setup
  }
  $('#qNumLabel').textContent = m === 'measure' ? 'Measured' : 'Max deviation';
  measFields();
  quickFields();
  if (m !== 'station') loadModes().catch(() => {});
}

async function selectMode(mode) {
  try {
    const r = await api('POST', '/api/cmd/mode', { mode, source: 'quick' });
    state.bus = r.bus;
    applyMode(r.bus);
    syncConfig(r.bus);
    renderCycle(r.bus);
    notice(r.accepted ? '' : `${r.command}: ${r.message}`);
  } catch (e) {
    $('#qConn').hidden = false;
  }
  refreshButtons();
}

function fillSelect(sel, items, keyName, key) {
  // rebuild a picker only when its list changed (the selection survives the polls)
  if (state[keyName] === key) return;
  state[keyName] = key;
  const keep = sel.value;
  sel.textContent = '';
  for (const [value, text] of items) {
    const o = document.createElement('option');
    o.value = value;
    o.textContent = text;
    sel.appendChild(o);
  }
  if (keep) sel.value = keep;
}

async function loadModes() {
  // the pickers of Measure / Quick check: the triangulation presets (mount, calibration)
  // and the recipes
  const d = await api('GET', '/api/modes');
  const presets = (d.presets || []).filter((p) => p.valid !== false);
  fillSelect($('#qRigPreset'), presets.map((p) => [p.name, p.name
    + (p.mount === 'mobile' ? ' [mobile]' : '') + (p.calibrated ? '' : ' (no stereo calibration)')]),
  'rigKey', JSON.stringify(presets.map((p) => [p.name, p.mount, p.calibrated])));
  if (d.loaded_preset && !state.presetBusy && $('#qRigPreset').value !== d.loaded_preset) {
    $('#qRigPreset').value = d.loaded_preset;
  }
  const recipes = d.recipes || [];
  fillSelect($('#qQuickRecipe'), recipes.map((r) => [r, r]), 'recipeKey', JSON.stringify(recipes));
}

async function selectRigPreset(name) {
  // the preset is LOADED: HOME (the rig check) and the measurement are about the same rig.
  // Another preset is another rig setup: what the operator confirmed was not this one
  if (!name) return;
  $('#qConfirm').checked = false;
  state.presetBusy = true;                       // the polls leave the picker alone meanwhile
  try {
    const j = await api('POST', `/api/presets/${encodeURIComponent(name)}/load`, {});
    notice(j.error ? `preset ${name}: ${j.error}` : '');
    state.presetBusy = false;
    await loadModes();
    await pollOnce();
  } catch (e) {
    notice(`preset ${name}: ${e.message || e}`);
  } finally {
    state.presetBusy = false;
  }
  refreshButtons();
}

function syncRigPreset(b) {
  // the picker always shows the loaded preset (a PLC, another page or a station
  // configuration may have loaded another one): what MEASURE / START runs and what HOME
  // checks is what the operator sees. A change of it clears the confirmation.
  const loaded = b.active_preset || '';
  if (loaded !== state.shownPreset) {
    if (state.shownPreset !== null) {
      $('#qConfirm').checked = false;
      loadZoneSets('');               // another preset: its geometry's zone sets, none picked
    }
    state.shownPreset = loaded;
  }
  if (state.opMode === 'station' || state.presetBusy || !loaded) return;
  const sel = $('#qRigPreset');
  if (sel.value === loaded) return;
  const listed = (sel.options ? Array.from(sel.options) : []).some((o) => o.value === loaded);
  if (listed) sel.value = loaded;
  else if (state.rigAsked !== loaded) {          // the list changed: read it again (once)
    state.rigAsked = loaded;
    loadModes().catch(() => {});
  }
}

function showRigInfo(b) {
  // the loaded preset's rig: fixed (trusted until a rig check says MOVED / UNKNOWN) or
  // mobile (no mm until verified in this session: HOME)
  const r = b.rig || null;
  let t = `Loaded: ${b.active_preset || 'none'}`;
  if (r) {
    t += ` - ${r.mount} rig`;
    if (!r.ok) t += ` - NO mm now: ${r.message}. HOME runs the rig check.`;
    else if (r.mount === 'mobile') t += ', verified for this session';
  }
  $('#qRigInfo').textContent = t;
  $('#qRigInfo').classList.toggle('qwarn', !!(r && !r.ok));
}

function measFields() {
  // only the inputs the chosen measurement reads
  const w = $('#qMeasWhat').value || 'flatness';
  $('#qMeasRoiRow').hidden = w === 'endface';
  $('#qMeasRoiLabel').textContent = w === 'step' ? 'Region A (camera px)'
    : (w === 'distance' ? 'Plane region (camera px, optional)' : 'Region (camera px)');
  $('#qMeasRoi2Row').hidden = w !== 'step';
  $('#qMeasP1Row').hidden = w !== 'distance';
  $('#qMeasP2Row').hidden = w !== 'distance';
  $('#qMeasGoldenRow').hidden = w !== 'deviation';
  $('#qMeasDiameterRow').hidden = w !== 'endface';
  state.pick = { n: 0, corner: null };
}

function quickFields() {
  const c = $('#qQuickCheck').value || 'flatness';
  $('#qQuickThresholdRow').hidden = c === 'recipe';
  $('#qQuickRoiRow').hidden = c === 'recipe';
  $('#qQuickGuardRow').hidden = c === 'recipe';
  $('#qQuickGoldenRow').hidden = c !== 'golden';
  $('#qQuickRecipeRow').hidden = c !== 'recipe';
}

function numList(text, n, what) {
  // "x0,y0,x1,y1" -> numbers; empty -> null; anything else is refused here, never guessed
  const parts = String(text || '').split(/[,;\s]+/).filter(Boolean);
  if (!parts.length) return null;
  const v = parts.map(Number);
  if (v.length !== n || v.some((x) => !Number.isFinite(x))) {
    throw new Error(`${what}: ${n} numbers, comma-separated (camera px), got "${text}"`);
  }
  return v;
}

function numOrNull(text, what) {
  const t = String(text || '').trim();
  if (!t) return null;
  const v = Number(t);
  if (!Number.isFinite(v)) throw new Error(`${what}: "${text}" is not a number`);
  return v;
}

function measureArgs() {
  // what POST /api/measure gets: only what the chosen measurement reads
  const what = $('#qMeasWhat').value || 'flatness';
  const a = { what, source: 'quick', scan_kind: $('#qMeasScan').value || 'metric' };
  if ($('#qRigPreset').value) a.preset = $('#qRigPreset').value;
  const roi = what === 'endface' ? null : numList($('#qMeasRoi').value, 4, 'region');
  if (roi) a.roi = roi;
  if (what === 'step') {
    const roi2 = numList($('#qMeasRoi2').value, 4, 'region B');
    if (roi2) a.roi2 = roi2;
  }
  if (what === 'distance') {
    const p1 = numList($('#qMeasP1').value, 2, 'point 1');
    const p2 = numList($('#qMeasP2').value, 2, 'point 2');
    if (p1) a.p1 = p1;
    if (p2) a.p2 = p2;
  }
  if (what === 'deviation' && $('#qMeasGolden').value.trim()) a.golden = $('#qMeasGolden').value.trim();
  if (what === 'endface') {
    const d = numOrNull($('#qMeasDiameter').value, 'expected diameter');
    if (d !== null) a.face_diameter_mm = d;
  }
  const rep = numOrNull($('#qMeasRepeats').value, 'repeats');
  if (rep !== null && rep !== 1) a.repeats = rep;
  // a painted zone set picked in Lights: only its zones are lit, and they are the region
  if ($('#qZoneSet').value && what !== 'endface') a.zone = `zoneset:${$('#qZoneSet').value}`;
  if (state.mode === 'mock' && $('#qMockPart').value.trim()) a.mock_scene = $('#qMockPart').value.trim();
  return a;
}

function quickArgs() {
  // what POST /api/quick-check gets
  const check = $('#qQuickCheck').value || 'flatness';
  const a = { check, source: 'quick', operator_confirmed: confirmed() };
  if (check === 'recipe') {
    if (!$('#qQuickRecipe').value) throw new Error('pick the recipe');
    a.recipe = $('#qQuickRecipe').value;
  } else {
    const t = numOrNull($('#qQuickThreshold').value, 'threshold');
    if (t === null) throw new Error('give the threshold in mm');
    a.threshold_mm = t;
    const roi = numList($('#qQuickRoi').value, 4, 'region');
    if (roi) a.roi = roi;
    const gb = numOrNull($('#qQuickGuard').value, 'guard band');
    if (gb) a.guard_band_pct = gb;
    if ($('#qRigPreset').value) a.preset = $('#qRigPreset').value;
    if (check === 'golden') {
      if (!$('#qQuickGolden').value.trim()) throw new Error('name the golden scan');
      a.golden = $('#qQuickGolden').value.trim();
    }
  }
  if (state.mode === 'mock' && $('#qMockPart').value.trim()) a.mock_part = $('#qMockPart').value.trim();
  return a;
}

async function sendModeJob(path, build, busyText, label) {
  // one measurement / quick check as a job of its mode; the page then follows the job
  let args;
  try { args = build(); } catch (e) { notice(`${label}: ${e.message || e}`); return; }
  try {
    const j = await api('POST', path, args);
    if (j.error || !j.id) { notice(`${label}: ${j.error || 'refused'}`); return; }
    notice('');
    if (j.bus) state.bus = j.bus;
    state.modeJobId = j.id;
    state.shownModeJob = null;
    clearResult();
    $('#qReason').textContent = '';
    lamp(busyText, 'running');
    if (path === '/api/quick-check' && state.mode === 'real') $('#qConfirm').checked = false;
  } catch (e) {
    notice(`${label}: ${e.message || e}`);
  }
  refreshButtons();
}

function sendMeasure() {
  if ($('#qMeasure').disabled) return Promise.resolve();
  return sendModeJob('/api/measure', measureArgs, 'MEASURING', 'MEASURE');
}

function sendQuick() {
  if ($('#qStart').disabled) return Promise.resolve();
  return sendModeJob('/api/quick-check', quickArgs, 'RUNNING', 'START');
}

function fmtValue(v, unit) {
  return `${Number(v).toFixed(unit === 'mm' ? 4 : 3)} ${unit}`;
}

function showMeasure(job) {
  // a measurement: a neutral MEASURED / NOT MEASURABLE and its values. Never a verdict:
  // the lamp shows no PASS / FAIL / PULL here, whatever the document says
  if (state.opMode !== 'measure') return;        // the mode was left: nothing of it is shown
  const r = job.result || {};
  state.resultShown = true;
  const ok = job.status === 'done' && r.status === 'MEASURED';
  lamp(ok ? 'MEASURED' : 'NOT MEASURABLE', ok ? 'measured' : 'notmeasurable');
  $('#qVerdict').textContent = 'none (a measurement decides nothing)';
  $('#qDisposition').textContent = '-';
  $('#qPull').hidden = true;
  $('#qNumLabel').textContent = 'Measured';
  $('#qNumber').textContent = ok ? `${r.what}: ${(r.values || []).length} value(s), ${r.repeats} scan(s)` : '-';
  $('#qTime').textContent = String(r.finished || job.finished || '-').replace('T', ' ');
  $('#qRunId').textContent = r.run_id || '-';
  $('#qReason').textContent = ok ? ''
    : (job.status === 'error' ? (job.error || 'error')
      : `${r.reason || 'not measurable'}: ${r.message || ''}` + (r.how_to_fix ? ` | fix: ${r.how_to_fix}` : ''));
  $('#qValidity').textContent = ok ? (r.validity || '') : '';
  const body = $('#qValuesBody');
  body.textContent = '';
  const vals = ok ? (r.values || []) : [];
  $('#qValues').hidden = vals.length === 0;
  for (const v of vals) {
    const u = v.uncertainty || {};
    const tr = document.createElement('tr');
    if (v.near_resolution_limit) tr.className = 'near';
    const cells = [v.label, fmtValue(v.value, v.units),
      u.value == null ? '-' : `± ${fmtValue(u.value, u.units || v.units)}`,
      (u.method || '-') + ((u.n_repeats || 1) > 1 ? ` ×${u.n_repeats}` : ''),
      String(v.n_points)];
    for (const c of cells) {
      const td = document.createElement('td');
      td.textContent = c;
      tr.appendChild(td);
    }
    if ((v.notes || []).length) tr.title = v.notes.join(' | ');
    body.appendChild(tr);
    for (const n of v.notes || []) {
      const nr = document.createElement('tr');
      nr.className = 'note';
      const td = document.createElement('td');
      td.colSpan = cells.length;
      td.textContent = n;
      nr.appendChild(td);
      body.appendChild(nr);
    }
  }
  const links = job.links || {};
  for (const [id, key] of [['#qResultLink', 'result_json'], ['#qValuesLink', 'values_csv'], ['#qPlyLink', 'ply']]) {
    $(id).hidden = !links[key];
    if (links[key]) $(id).href = links[key];
  }
  const img = $('#qMeasImg');                    // the picture the next clicks pick camera px on
  if (links.preview) {
    img.hidden = false;
    img.src = `${links.preview}?t=${Date.now()}`;
  }
}

function previewPx(ev) {
  // a click on the measurement's picture -> camera px (the picture is the camera image,
  // drawn smaller: scaled back by its natural size)
  const img = $('#qMeasImg');
  const w = img.clientWidth || img.naturalWidth || 1;
  const h = img.clientHeight || img.naturalHeight || 1;
  return [Math.round(ev.offsetX * (img.naturalWidth || w) / w),
    Math.round(ev.offsetY * (img.naturalHeight || h) / h)];
}

function onPreviewClick(ev) {
  // distance: the clicks are point 1, point 2, point 1 ...; anything else: two clicks are the
  // corners of a region (step: region A, then region B)
  const [x, y] = previewPx(ev);
  const what = $('#qMeasWhat').value || 'flatness';
  const pk = state.pick;
  let said;
  if (what === 'distance') {
    const second = pk.n % 2 === 1;
    $(second ? '#qMeasP2' : '#qMeasP1').value = `${x},${y}`;
    pk.n += 1;
    said = `point ${second ? 2 : 1} = camera px (${x}, ${y})`;
  } else if (pk.corner === null) {
    pk.corner = [x, y];
    said = `corner at camera px (${x}, ${y}): click the opposite corner`;
  } else {
    const [cx, cy] = pk.corner;
    const roi = [Math.min(cx, x), Math.min(cy, y), Math.max(cx, x) + 1, Math.max(cy, y) + 1];
    const b = what === 'step' && pk.n % 2 === 1;
    $(b ? '#qMeasRoi2' : '#qMeasRoi').value = roi.join(',');
    pk.corner = null;
    pk.n += 1;
    said = `region${what === 'step' ? (b ? ' B' : ' A') : ''} = camera px ${roi.join(',')}`;
  }
  $('#qMeasHint').textContent = said;
}

async function send(name, extra) {
  const body = { source: 'quick', ...(extra || {}) };
  if (['start', 'trigger', 'arm'].includes(name)) {
    body.operator_confirmed = confirmed();
    body.config = $('#qStation').value;          // refused if the bus has another selected
    if ($('#qProfile').value) body.profile = $('#qProfile').value;
  }
  try {
    const r = await api('POST', `/api/cmd/${name}`, body);
    state.bus = r.bus;
    trackJobs(r.jobs);
    syncConfig(r.bus);
    renderCycle(r.bus);
    notice(r.accepted ? '' : `${r.command}: ${r.message}`);
    if (!r.accepted) {
      applyBus(r.bus);                           // e.g. a refused START: the bus PULLs, or forgets a PASS
      $('#qReason').textContent = `${r.command}: ${r.message}`;
    }
    else if ((name === 'start' || name === 'trigger') && !flowCycle(r.bus)) {
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

function renderCycle(b) {
  renderFlow(b);
  renderPrompt(b);
  renderMock(b);
  return showCycle(b);
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
  state.resultShown = true;
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
  // the preset START runs: the configuration's / profile's own, else the loaded one (run_preset)
  const run = b.run_preset || b.active_preset;
  $('#qPreset').textContent = run || 'none loaded';
  $('#qPresetLoaded').textContent = run && b.active_preset && b.active_preset !== run
    ? `(loaded now: ${b.active_preset}; the run loads ${run})` : '';
}

async function pollOnce() {
  const b = await api('GET', '/api/cmd/state');
  state.bus = b;
  if (b.mode) state.mode = b.mode;
  applyMode(b);
  syncRigPreset(b);
  showRigInfo(b);
  if (!state.loadingStations && (b.state === 'READY' || b.state === 'FAULT') && stationListChanged(b)) {
    await fetchStations();                         // configs added / removed / renamed
  }
  syncConfig(b);
  showPreset(b);
  showWarnings(b);
  trackJobs(b.jobs);
  const inFlow = renderCycle(b);
  if (state.runJobId && state.shownRun !== state.runJobId) {
    const j = await api('GET', `/api/jobs/${state.runJobId}`);
    const ended = j.status === 'done' || j.status === 'error';
    if (inFlow) {
      // one work state's DLP check: its zones and picture; the lamp follows the cycle
      if (ended) { showStepJob(j); state.shownRun = j.id; }
    } else if (j.status === 'running' || (ended && (b.state === 'RUNNING' || b.state === 'ABORTING'))) {
      // a run is shown once the bus has heard it end (an ABORT may still turn it ABORTED)
      lamp('RUNNING', 'running');
      steps('done', 'active', '');
    } else if (ended) { showRun(j); state.shownRun = j.id; }
  }
  if (state.homeJobId && state.shownHome !== state.homeJobId) {
    const j = await api('GET', `/api/jobs/${state.homeJobId}`);
    if (j.status === 'done' || j.status === 'error') { showRig(j); state.shownHome = j.id; }
  }
  if (state.modeJobId && state.shownModeJob !== state.modeJobId) {
    // the measurement / quick check this page started: shown when its job ends
    const j = await api('GET', `/api/jobs/${state.modeJobId}`);
    if (j.status === 'done' || j.status === 'error') {
      // shown only while the page is still in the job's own mode (another client may have
      // switched the bus meanwhile: Station's lamp is the bus's, never a quick check's)
      if (j.verb === 'measure_run') showMeasure(j);
      else if (state.opMode === 'quick_check') showRun(j);     // PASS / FAIL / PULL
      state.shownModeJob = state.modeJobId;
    }
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
async function setupJob(verb, args, msgEl, busyText) {
  state.setupBusy = true;
  refreshButtons();
  $(msgEl).textContent = busyText || `${verb}...`;
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

// ---- painted zone sets (engine/zones): drawn on the part with the mouse, in projector px
async function loadZoneSets(pick) {
  // the sets painted at the loaded preset's geometry; `pick`: the one to select afterwards
  let d;
  try { d = await api('GET', '/api/zonesets'); } catch (e) { return; }
  state.zoneSets = d.sets || [];
  const sel = $('#qZoneSet');
  const want = pick !== undefined ? pick : sel.value;
  sel.textContent = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = state.zoneSets.length ? '(none: the whole field)' : '(none painted yet)';
  sel.appendChild(none);
  for (const z of state.zoneSets) {
    const o = document.createElement('option');
    o.value = z.name;
    o.textContent = `${z.name} - ${z.n_zones} zone(s)` + (z.valid ? '' : ' (invalid)');
    sel.appendChild(o);
  }
  sel.value = state.zoneSets.some((z) => z.name === want) ? want : '';
  showZoneSet();
}

function showZoneSet() {
  // the picked set: its zones and its picture (the photo of the zones on the part, else the
  // DMD-space preview)
  const z = (state.zoneSets || []).find((x) => x.name === $('#qZoneSet').value);
  const img = $('#qZoneImg');
  const link = z && (z.camera || z.preview);
  img.hidden = !link;
  if (link) img.src = `/files/${link}?t=${encodeURIComponent(z.created || '')}`;
  $('#qZoneInfo').textContent = !z ? ''
    : (z.valid ? `${z.name}: ` + (z.zones || []).map((q) => q.id + (q.name ? ` ${q.name}` : '')).join(', ')
      + ` (painted ${String(z.created || '?').replace('T', ' ')}; projector px, valid at this rig placement). `
      + 'MEASURE lights and measures only these zones.'
      + ((z.rig_changes || []).length ? ` CHECK (Show zones) or paint again: since it was painted, ${z.rig_changes.slice(0, 3).join('; ')}.` : '')
      : `${z.name} is not usable: ${z.error}`);
  // Measure mode shows what MEASURE will be limited to (the Lights panel may be closed)
  $('#qMeasZone').textContent = z
    ? `Zone set ${z.name} is picked (Lights): only its zones are lit and measured.` : '';
}

async function paintZones() {
  // a new set (a name typed), or the picked set edited (its zones loaded, the old one to trash)
  const typed = $('#qZoneName').value.trim();
  const picked = $('#qZoneSet').value;
  if (!typed && !picked) {
    $('#qSetupMsg').textContent = 'name the zone set to paint (New set name), or pick one to edit';
    return;
  }
  const args = typed && typed !== picked ? { name: typed } : { name: picked, source: picked, replace: true };
  const j = await setupJob('zone_paint', args, '#qSetupMsg',
    state.mode === 'mock' ? 'painting (MOCK: a scripted demo session)...'
      : 'painting: draw on the part with the mouse - SPACE saves, ESC cancels (ABORT ends it)');
  const r = j && j.status === 'done' ? j.result : null;
  if (!r) return;
  if (r.saved) {
    $('#qSetupMsg').textContent = `zone set ${r.name} saved: ${r.n_zones} zone(s)`;
    $('#qZoneName').value = '';
    await loadZoneSets(r.name);
  } else {
    $('#qSetupMsg').textContent = `zone set ${args.name} not saved: ${r.reason || 'cancelled'}`;
  }
}

async function showZonesOnPart() {
  const name = $('#qZoneSet').value;
  if (!name) { $('#qSetupMsg').textContent = 'pick a zone set first (or paint one)'; return; }
  const j = await setupJob('zone_show', { zones: name, hold_s: 5 }, '#qSetupMsg');
  const r = j && j.status === 'done' ? j.result : null;
  if (!r) return;
  const seen = (r.zones || []).filter((z) => z.seen).length;
  $('#qSetupMsg').textContent = `${name}: ${seen} of ${(r.zones || []).length} zone(s) land in the camera image`
    + ((r.warnings || []).length ? ` - ${r.warnings.join('; ')}` : '');
  const link = r.overlay_png || r.camera_png;
  if (link && !/^[A-Za-z]:|^\//.test(link)) {                      // library-relative only
    $('#qZoneImg').hidden = false;
    $('#qZoneImg').src = `/files/${link}`;
  }
}

async function showZone() {
  const st = state.stations.find((x) => x.config === $('#qStation').value);
  if (!st || !st.zone) { $('#qSetupMsg').textContent = 'this station configuration uses the whole field (no job.zone)'; return; }
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
  const mode = state.opMode || 'station';
  let handled = true;
  if (k === ' ' || k === 'Enter') {              // the mode's own action
    if (mode === 'measure') sendMeasure();
    else if (mode === 'quick_check') sendQuick();
    else if (!$('#qStart').disabled) send('start');
  } else if (k === 'm' || k === 'M') {           // M = MEASURE, in Measure mode only
    if (mode === 'measure') sendMeasure();
    else handled = false;
  } else if (k === 'Escape') send('abort');
  else if (k === 'h' || k === 'H') send('home');
  else if (mode !== 'station') handled = false;  // the station's keys: Station mode only (its
  // selectors and buttons are hidden in the other modes - a digit typed after clicking the
  // measurement picture must never select a station configuration and load its preset)
  else if (k === 't' || k === 'T') send('trigger');
  else if (k === 'a' || k === 'A') send(s === 'ARMED' ? 'disarm' : 'arm');
  else if (k === 'n' || k === 'N') { if (!$('#qStep').disabled) send('step'); }   // the next work state
  else if (k === 'k' || k === 'K') { if (!$('#qAck').disabled) send('ack'); }     // acknowledge the prompt
  else if (k === 'p' || k === 'P') nextProfile();
  else if (/^[1-9]$/.test(k)) selectConfig(Number(k));
  else handled = false;
  if (handled) ev.preventDefault();
}

async function selectConfig(arg) {
  try {
    const r = await api('POST', '/api/cmd/config', { config: arg, source: 'quick' });
    state.bus = r.bus;
    syncConfig(r.bus);                 // accepted or refused: show the bus's configuration
    showPreset(r.bus);
    renderCycle(r.bus);
    applyBus(r.bus);                   // another configuration: the last cycle is forgotten
    const said = r.accepted ? (/disarmed/.test(r.message) ? r.message : '') : `${r.command}: ${r.message}`;
    if (said || !flowCycle(r.bus)) $('#qReason').textContent = said;
    notice(said);                      // (the cycle's own line is redrawn by every poll)
  } catch (e) {
    $('#qConn').hidden = false;
  }
  showStation();
}

async function selectProfile(arg) {
  try {
    const r = await api('POST', '/api/cmd/profile', { profile: arg, source: 'quick' });
    state.bus = r.bus;
    syncConfig(r.bus);                 // accepted or refused: show the bus's profile
    showPreset(r.bus);
    renderCycle(r.bus);
    applyBus(r.bus);                   // another profile: the last cycle is forgotten
    const said = r.accepted ? (/disarmed/.test(r.message) ? r.message : '') : `${r.command}: ${r.message}`;
    if (said || !flowCycle(r.bus)) $('#qReason').textContent = said;
    notice(said);
  } catch (e) {
    $('#qConn').hidden = false;
  }
  refreshButtons();
}

function nextProfile() {
  const b = state.bus || {};
  const names = (b.profiles || []).map((p) => p.name);
  if (names.length < 2) return;
  selectProfile(names[(names.indexOf(b.profile) + 1) % names.length]);
}

async function setInput(signal, value) {
  // MOCK only: the operator page sets the SIMULATED line's sensor, which reports for every
  // cycle (the real rig's claims come from the line: POST /api/io/inputs from its gateway,
  // for every part)
  if (state.mode !== 'mock') return;
  try {
    const r = await api('POST', '/api/mock/case', { inputs: { [signal]: value } });
    state.bus = r.bus;
    renderCycle(r.bus);
  } catch (e) {
    $('#qFlowMsg').textContent = String(e.message || e);
  }
}

async function setMockCase(name) {
  try {
    const r = await api('POST', '/api/mock/case', name ? { case: name } : {});
    state.bus = r.bus;
    $('#qFlowMsg').textContent = name ? `simulated line: ${name}` : '';
    renderCycle(r.bus);
  } catch (e) {
    $('#qFlowMsg').textContent = String(e.message || e);
  }
}

function onInputClick(ev) {
  const el = ev.target && ev.target.closest ? ev.target.closest('[data-signal]') : null;
  if (el) setInput(el.dataset.signal, el.dataset.value !== 'true');
}

document.addEventListener('DOMContentLoaded', () => {
  $('#qStation').addEventListener('change', () => selectConfig($('#qStation').value));
  $('#qProfile').addEventListener('change', () => selectProfile($('#qProfile').value));
  $('#qMockCase').addEventListener('change', () => setMockCase($('#qMockCase').value));
  $('#qInputs').addEventListener('click', onInputClick);
  $('#qConfirm').addEventListener('change', refreshButtons);
  for (const k of MODES) $(MODE_BTN[k]).addEventListener('click', () => selectMode(k));
  $('#qRigPreset').addEventListener('change', () => selectRigPreset($('#qRigPreset').value));
  $('#qMeasWhat').addEventListener('change', measFields);
  $('#qQuickCheck').addEventListener('change', quickFields);
  $('#qMeasure').addEventListener('click', sendMeasure);
  $('#qMeasImg').addEventListener('click', onPreviewClick);
  $('#qStart').addEventListener('click', () => (state.opMode === 'quick_check' ? sendQuick() : send('start')));
  $('#qAbort').addEventListener('click', () => send('abort'));
  $('#qHome').addEventListener('click', () => send('home'));
  $('#qStep').addEventListener('click', () => send('step'));
  $('#qAck').addEventListener('click', () => send('ack'));
  $('#qTrigger').addEventListener('click', () => send('trigger'));
  $('#qArm').addEventListener('click', () => send(state.bus && state.bus.state === 'ARMED' ? 'disarm' : 'arm'));
  $('#qGrid').addEventListener('click', projectGrid);
  $('#qZones').addEventListener('click', showZone);
  $('#qPaint').addEventListener('click', paintZones);
  $('#qZoneShow').addEventListener('click', showZonesOnPart);
  $('#qZoneSet').addEventListener('change', showZoneSet);
  // zone sets belong to the loaded preset's geometry: re-read when the panel opens, when
  // another preset is loaded and after painting
  $('#qLights').addEventListener('toggle', () => { if ($('#qLights').open) loadZoneSets(); });
  loadZoneSets();
  $('#qLive').addEventListener('click', toggleLive);
  $('#qSignal').addEventListener('click', signalCheck);
  document.addEventListener('keydown', onKey);
  loadStations().catch(() => { $('#qConn').hidden = false; }).finally(loop);
});
