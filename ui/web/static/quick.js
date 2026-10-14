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
// is for one setup: a mode change, another preset or another zone selection clears it.
//
// Scan zones (the panel above 1 Lights): the zone sets painted at the loaded preset's rig
// placement, each with an on / off switch and one switch per zone. What is switched on is
// the ZONE SELECTION; the server stores it (it survives a reload and a restart) and sends it
// with every bus state (zone_selection), together with the painter that holds the rig
// (painter) and how the last painter session ended (last_paint) - so every page, also one
// just reloaded, shows the same rows, the same "Painting ..." and the same SAVED / NOT SAVED.
// ONE function (zoneScope) says what the run button inspects in the mode, measurement and
// station configuration shown, and what it does with the selection: the bold line and the
// sentence above the run buttons, the save message and the requests all come from it. A run
// that uses the selection sends "selection@<stamp>" (the selection exactly as this page
// showed it): the server refuses a stamp that is not the stored one any more.
'use strict';

const $ = (sel) => document.querySelector(sel);
const state = { setupBusy: false, mode: null, stations: [], bus: null, runJobId: null, homeJobId: null,
  shownRun: null, shownHome: null, shownConfig: null, shownProfile: null, loadingStations: false,
  lampDisp: null, flowKey: null, mockKey: null, profileKey: null, cycleKey: null, resultShown: false,
  // operating mode (null until the first bus state), its job, its pickers, the picture clicks
  opMode: null, modeJobId: null, shownModeJob: null, rigKey: null, recipeKey: null,
  pick: { n: 0, corner: null }, presetBusy: false, shownPreset: null, rigAsked: null,
  // Scan zones: the sets of GET /api/zonesets (photo, who uses them), what the rows were
  // drawn from, the open question, the page's own zone job, the message under the buttons
  zoneSets: [], zoneOthers: [], zoneKey: null, zoneSetsSig: null, zoneStamp: null, ask: null,
  zoneBusy: false, zoneMsg: null, paintKey: null, painting: null,
  // a change of the selection is on its way to the server (its answer is not here yet)
  selBusy: false };

async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    opt.headers['Content-Type'] = 'application/json';
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok && r.status !== 409) {
    const e = new Error(data.error || data.message || `${r.status}`);
    e.data = data;                                 // a refusal's own answer (its bus, its selection)
    throw e;
  }
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
  // which area the last result was about, its zones and their warnings: gone with it
  $('#qInspected').textContent = '';
  $('#qZoneCounts').textContent = '';
  $('#qZoneCounts').hidden = true;
  $('#qZoneWarn').textContent = '';
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
  // the painter (or a zone show) holds the rig and the projector display: nothing starts
  const pOn = painterOn();
  // what the run button would do with the zone selection: a measurement / quick check that
  // could only be refused (a selection that is unknown or unusable, a measurement that has
  // no meaning across several zones) is not offered - the sentence beside the button says why
  const sc = zoneScope();
  $('#qConfirmRow').hidden = state.mode !== 'real' || m === 'measure';
  // Station: START begins a cycle; Quick check: START runs the one check
  $('#qStart').disabled = m === 'station' ? (!idle || !usable || !confirmed() || pOn)
    : (!free || !confirmed() || !sc.ok || pOn);
  $('#qMeasure').disabled = !free || !sc.ok || pOn;
  for (const k of MODES) $(MODE_BTN[k]).disabled = !(idle || s === 'FAULT') || modeBusy;
  $('#qRigPreset').disabled = !free || pOn;
  $('#qAbort').disabled = false;                       // ABORT is never disabled
  $('#qHome').disabled = !(idle || s === 'FAULT' || s === 'ABORTING');
  $('#qArm').disabled = s === 'ARMED' ? false : (s !== 'READY' || !usable || !confirmed() || pOn);   // DISARM: always
  $('#qArm').textContent = s === 'ARMED' ? 'DISARM' : 'ARM';
  $('#qArm').classList.toggle('on', s === 'ARMED');
  $('#qTrigger').disabled = !(s === 'ARMED' || waitStep);
  $('#qStep').disabled = !waitStep;
  $('#qStep').classList.toggle('on', !!waitStep);
  $('#qAck').disabled = !(s === 'WAITING' && b.prompt);
  $('#qStation').disabled = !(idle || s === 'FAULT');
  $('#qProfile').disabled = !(idle || s === 'FAULT');
  $('#qBusState').textContent = s;
  for (const id of ['#qGrid', '#qZones', '#qLive', '#qSignal']) {
    $(id).disabled = !(idle || s === 'FAULT') || state.setupBusy || pOn;    // setup only when idle
  }
  // the zones a station inspects are fixed while it is ARMED or in a cycle: painting, deleting
  // and the switches only while READY / FAULT with no job of a mode or of the zones going
  const zfree = zonesFree();
  for (const id of ['#qPaint', '#qZoneDelete', '#qZoneShow']) $(id).disabled = !zfree || state.setupBusy;
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
  $('#qZone').textContent = st && st.zone_from_selection
    ? `the selected zones (${(zsel() || {}).label || 'not known'})` : ((st && st.zone) || 'whole field');
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
    renderZones();                               // another mode: another use of the zones
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
  // the zone selection exactly as this page shows it (also "nothing on = the whole field"):
  // the server refuses a stamp that is not the stored selection's any more. A measurement
  // that does not use the selection (the end face) sends no zone
  const sc = zoneScope();
  if (!sc.ok) throw new Error(sc.scope);
  if (sc.uses) a.zone = `selection@${sc.stamp}`;
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
  // flatness / golden: the zone selection as this page shows it (a recipe has its own zones)
  const sc = zoneScope();
  if (!sc.ok) throw new Error(sc.scope);
  if (sc.uses) a.zone = `selection@${sc.stamp}`;
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
  showInspected(r);                              // which area these numbers are about
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
    // a configuration that inspects the selected zones: the selection this page shows
    // (refused if the stored one is another by now)
    const sc = zoneScope();
    if (sc.uses && sc.stamp) body.zones = sc.stamp;
  }
  try {
    const r = await api('POST', `/api/cmd/${name}`, body);
    if (!r.bus) {                                // a refusal that is not the bus's own reply
      notice(`${name.toUpperCase()}: ${r.message || r.error || 'refused'}`);
      refreshButtons();
      return;
    }
    state.bus = r.bus;
    trackJobs(r.jobs);
    syncConfig(r.bus);
    renderCycle(r.bus);
    renderZones();
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
  // lit through painted zones: the verdict of each zone (the one that failed is named)
  if (r.zone_selection && r.zone_selection.summary) reasons.push(`zones: ${r.zone_selection.summary}`);
  for (const w of r.warnings || []) if (/abort/i.test(w)) reasons.push(w);
  $('#qReason').textContent = reasons.join(' | ');
  showInspected(r);                              // which area this verdict is about
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

function inspectedText(r) {
  // which area a result is about, from the run's OWN record - never from what is switched on
  // now: the zones it was lit through, the whole field, a station configuration's own zone
  const z = r.zone_selection || null;
  if (z) {
    if (!z.text) return 'Inspected: whole field (no zone set was switched on)';
    const keys = (z.by_zone || []).map((x) => x.key);
    return `Inspected: ${keys.length ? keys.join(', ') : z.text}`
      + (z.source === 'selection' ? ' (the zone selection'
        + (z.selected && z.selected.time ? ` of ${String(z.selected.time).replace('T', ' ')}` : '') + ')' : '');
  }
  if (r.operating_mode === 'measure') {
    const sp = r.spec || {};
    if (sp.what === 'endface') return 'Inspected: whole field (the end-face measurement)';
    return `Inspected: ${sp.zone || 'whole field'}` + (sp.roi ? `, inside region ${sp.roi.join(',')}` : '');
  }
  const job = r.job || {};
  if (job.type === 'recipe' || job.recipe) return `Inspected: recipe ${job.recipe || '?'}'s own zones`;
  const zone = (r.capture || {}).zone || job.zone;
  const file = job.config && r.operating_mode !== 'quick_check' ? `station configuration ${job.config}, ` : '';
  return `Inspected: ${file}${zone ? `its own zone ${zone}` : 'whole field'}`;
}

function showInspected(r) {
  $('#qInspected').textContent = r && (r.run_id || r.zone_selection) ? inspectedText(r) : '';
  const z = (r && r.zone_selection) || {};
  const ul = $('#qZoneCounts');
  ul.textContent = '';
  const rows = z.by_zone || [];
  ul.hidden = rows.length === 0;
  for (const x of rows) {
    // one line per zone: its verdict (a quick check / station run), its points, how much of it was seen
    const li = document.createElement('li');
    const v = x.verdict || '';
    li.className = v ? `z-${v.toLowerCase()}` : '';
    const b = document.createElement('b');
    b.textContent = `${x.key}${x.name ? ` ${x.name}` : ''}${v ? ` ${v}` : ''}`;
    li.appendChild(b);
    const bits = [];
    if (x.max_mm != null) bits.push(`max ${Number(x.max_mm).toFixed(3)} mm`);
    bits.push(x.points == null ? 'points not known' : `${x.points} points`);
    if (x.coverage_pct != null) bits.push(`${Number(x.coverage_pct).toFixed(0)} % of the zone seen`);
    if (x.reason) bits.push(x.reason);
    const t = document.createElement('span');
    t.textContent = bits.join(', ');
    li.appendChild(t);
    ul.appendChild(li);
  }
  const notes = [...(z.warnings || [])];
  if (z.outside) notes.push(`${z.outside} point(s) decoded outside every zone (reported, not judged)`);
  $('#qZoneWarn').textContent = notes.join(' | ');
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
  renderZones();                                 // the Scan zones panel, the line above the run buttons
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

function contactLost() {
  $('#qConn').hidden = false;
  const p = state.painting;
  if (p && !(state.zoneMsg && state.zoneMsg.lost)) {
    // the painter was open when the server stopped answering: what it did is not known here
    state.zoneMsg = { warn: true, lost: true,
      text: `NOT KNOWN: lost contact with the server while painting ${p.name}. Look at the list when it is back: `
        + `${p.name} is new only if its painted time is after ${p.at || 'now'}.` };
    renderZoneMsg();
  }
}

async function loop() {
  for (;;) {
    try { await pollOnce(); } catch (e) { contactLost(); }
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
    let fails = 0;
    while (j.status === 'queued' || j.status === 'running') {
      await new Promise((res) => setTimeout(res, 400));
      try {                                      // one failed poll is not the job's end
        j = await api('GET', `/api/jobs/${j.id}`);
        fails = 0;
      } catch (e) {
        fails += 1;
        if (fails > 75) throw e;
      }
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

// ---- Scan zones: painted zone sets (engine/zones), drawn on the part with the mouse, in
// projector px; which of them are switched on is the zone selection (stored by the server)

function busHas(key) {
  return !!state.bus && Object.prototype.hasOwnProperty.call(state.bus, key);
}

function zsel() {
  // the zone selection as the server sent it: an object; null = no preset is loaded;
  // undefined = this bus state says nothing about it (never read as "nothing is on")
  return busHas('zone_selection') ? state.bus.zone_selection : undefined;
}

function painterOn() {
  // the painter or a zone show holds the rig (from the bus state: true in every tab and
  // after a reload), or this page's own zone job has not reported back yet
  return !!(state.bus && state.bus.painter) || state.zoneBusy;
}

function zonesFree() {
  // painting, deleting and the switches: only while the station is READY / FAULT (what a
  // station inspects is fixed while it is ARMED or in a cycle) and no measurement, quick
  // check, painter, zone show or setup job of this page (grid, signal check) is queued or
  // running: the server would refuse the change, or keep it waiting behind that job
  const b = state.bus || { state: 'READY' };
  const modeBusy = (b.mode_jobs || []).length > 0
    || !!(state.modeJobId && state.shownModeJob !== state.modeJobId);
  return (b.state === 'READY' || b.state === 'FAULT') && !modeBusy && !painterOn() && !state.setupBusy;
}

function hm(t) {
  // "2026-01-02T09:14:02" -> "09:14"
  const m = /T(\d\d:\d\d)/.exec(String(t || ''));
  return m ? m[1] : '';
}

function when(t) {
  // "2026-01-02T09:14:02" -> "2026-01-02 09:14": with its date (a message may be days old)
  const m = /^(\d{4}-\d\d-\d\d)T(\d\d:\d\d)/.exec(String(t || ''));
  return m ? `${m[1]} ${m[2]}` : hm(t);
}

function plural(n, word) { return `${n} ${word}${n === 1 ? '' : 's'}`; }

function selProblem(s) {
  // why the selection cannot be used by a run (null: it can)
  if (s === undefined) return 'the server has not said which zone sets are switched on';
  if (s === null) return 'no preset is loaded';
  if (s.state === 'unknown' || !s.ok) {
    return ((s.problems || [])[0] || {}).message || 'a zone set that is switched on cannot be used';
  }
  return null;
}

// What the run button inspects, and what it does with the zone selection - in the mode, the
// measurement / check and the station configuration this page shows now. Everything the page
// says or sends about zones comes from here:
//   line   the bold line above the run buttons: what the run inspects
//   scope  the sentence beside it: what the run button does (or why it will be refused)
//   uses   the run lights and inspects the SELECTED zones (it sends selection@<stamp>)
//   ok     false: a run could only be refused - MEASURE / START (Quick check) stay off
//   warn   the line is shown in the warning colour
function zoneScope() {
  const b = state.bus || {};
  const mode = state.opMode || 'station';
  const s = zsel();
  const known = !!s && s.state !== 'unknown';
  const lbl = known ? s.label : (s === null ? 'no preset loaded' : 'NOT KNOWN');
  const on = known && !s.empty;
  const bad = selProblem(s);
  const chk = known && (s.check || []).length ? s.check.join(' | ') : '';
  const n = known ? (s.n_zones_on || 0) : 0;
  const many = n > 1;
  const these = many ? 'these zones' : 'this zone';
  const out = { line: `Zones: ${lbl}`, scope: '', uses: true, ok: true, warn: !!bad || !!chk,
    stamp: known ? s.stamp : null, n, label: lbl };
  const unused = (line, text) => {               // this run does not use the selection
    out.uses = false;
    out.line = line;
    out.warn = false;
    out.stamp = null;
    out.scope = text + (s && bad ? ` The selection has a problem (${bad}); this run does not use it.`
      : (on ? ` The selected zones (${lbl}) are NOT used.` : ''));
    return out;
  };
  const refuse = (text) => { out.ok = false; out.warn = true; out.scope = text; return out; };
  const region = (id) => $(id).value.trim();
  const fix = s && s.state === 'problem' ? ' Switch that set off or paint it again.' : '';

  if (mode === 'measure') {
    const what = $('#qMeasWhat').value || 'flatness';
    if (what === 'endface') {
      return unused('End face: whole field', 'The end-face measurement uses the whole field (it finds the rim on the fully lit part).');
    }
    if (bad) {
      return refuse(s === null ? 'Load a preset first: MEASURE is off.'
        : `This run will be refused: ${bad}.${fix || ' MEASURE stays off until the selection is known.'}`);
    }
    const roi = region('#qMeasRoi');
    if (!on) {
      out.scope = `MEASURE uses the whole field${roi ? `, inside Region ${roi}` : ''}.`;
      return out;
    }
    if (what === 'step') {
      return refuse('Step height needs region A and region B, and the selected zones cannot say which is which: switch the zones off and type Region A and Region B.');
    }
    if (many && (what === 'dims' || what === 'deviation')) {
      return refuse(`${what === 'dims' ? 'The bounding dimensions are of ONE part' : 'The deviation from the golden is ONE compare'}: with several zones on MEASURE is off. Leave one zone on, or switch the zones off.`);
    }
    if (what === 'distance') {
      if (many && !$('#qMeasP2').value.trim()) {
        return refuse('Point 1 to the plane of the region needs ONE zone: with several zones on give Point 2, or leave one zone on.');
      }
      out.scope = `MEASURE lights only ${these}: pick the points inside ${many ? 'them' : 'it'}.`
        + (many ? '' : ' Point 1 is also measured to the plane of this zone.');
    } else if (what === 'flatness' || what === 'cylinder') {
      const one = what === 'flatness' ? 'plane' : 'cylinder';
      out.scope = many ? `MEASURE lights only these zones and measures each one on its own (one ${one} per zone, named set/zone).`
        : 'MEASURE lights and measures only this zone.';
    } else if (what === 'dims') out.scope = 'MEASURE lights only this zone and measures the part inside it.';
    else out.scope = 'MEASURE lights only this zone; the golden must have been scanned through the same zone.';
    if (roi) out.scope += ` Only the part of ${these} inside Region ${roi} is measured.`;
    if (chk) out.scope += ` Check them first (Show selected zones): since they were painted, ${chk}.`;
    return out;
  }

  if (mode === 'quick_check') {
    const c = $('#qQuickCheck').value || 'flatness';
    if (c === 'recipe') {
      const r = $('#qQuickRecipe').value || '?';
      return unused(`Recipe ${r}: its own zones`, `START runs recipe ${r} with its own zones.`);
    }
    if (bad) {
      return refuse(s === null ? 'Load a preset first: START is off.'
        : `This run will be refused: ${bad}.${fix || ' START stays off until the selection is known.'}`);
    }
    const roi = region('#qQuickRoi');
    if (!on) {
      out.scope = `START checks the whole field${roi ? `, inside Region ${roi}` : ''}.`;
      return out;
    }
    out.scope = c === 'golden'
      ? `START lights only ${these} and compares what ${many ? 'they show' : 'it shows'} with the golden, which must have been scanned through the same zones (else the part is PULLed).`
      : (many ? 'START lights only these zones and checks each one against the limit on its own; the result names a zone that fails.'
        : 'START lights and checks only this zone.');
    out.scope += ' A zone the camera does not see well enough PULLs the part.';
    if (roi) out.scope += ` Only the part of ${these} inside Region ${roi} is checked.`;
    if (chk) out.scope += ` START will PULL until they were shown on the part (Show selected zones) or painted again: since they were painted, ${chk}.`;
    return out;
  }

  // Station: a run inspects what its configuration says, always. The selection is used only
  // when the configuration's own job says "zone": "selection"
  const st = state.stations.find((x) => x.config === $('#qStation').value);
  if (!st || st.error) return unused('Station zone: no usable configuration selected', '');
  const file = st.config;
  if (!st.zone_from_selection) {
    let line = `Station zone: ${st.zone || 'whole field'} (${file})`;
    let text = `START runs ${file} as written: ${st.zone ? `its own zone ${st.zone}` : 'the whole field'}.`;
    if (st.job_type === 'flow') {
      line = `Station zones: each work state's recipe (${file})`;
      text = `START runs ${file} as written: each work state is checked by its recipe's own zones.`;
    } else if (st.job_type === 'recipe') {
      line = `Station zones: recipe ${st.recipe}'s own (${file})`;
      text = `START runs ${file} as written: recipe ${st.recipe} with its own zones.`;
    }
    unused(line, text);
    if (on && !bad) out.scope += ' To inspect them, use Quick check or Measure, or a station configuration whose job says "zone": "selection".';
    return out;
  }
  const pulled = 'START will be refused and the part PULLed';
  const run = b.run_preset || '';
  const loaded = b.active_preset || '';
  if (run && run !== loaded) {
    return refuse(`${file} inspects the selected zones with preset ${run}, and ${loaded ? `preset ${loaded}` : 'no preset'} is loaded: ${pulled}. Select the station configuration again (that loads ${run}).`);
  }
  if (bad) return refuse(`${file} inspects the selected zones and the selection cannot be used (${bad}): ${pulled}.${fix}`);
  if (!on) return refuse(`${file} inspects the selected zones and none is switched on: ${pulled}. Switch a zone set on.`);
  if (chk) {
    return refuse(`${file} inspects the selected zones, and since they were painted: ${chk}. ${pulled} until they were shown on the part (Show selected zones) or painted again.`);
  }
  out.scope = `START runs ${file}, which inspects the SELECTED zones: ${lbl}, each one against the limit on its own. `
    + 'Changing the selection changes what this station inspects; it is fixed while the station is ARMED or in a cycle.';
  return out;
}

function zoneSay(text, warn) {
  // what the last zone action did (stays until the next one)
  state.zoneMsg = { text: text || '', warn: !!warn };
  renderZoneMsg();
}

function paintingText(name) {
  return state.mode === 'mock'
    ? `Painting ${name} (MOCK: a scripted demo session)...`
    : `Painting ${name}: draw on the part with the mouse. SPACE on the projector saves, ESC cancels. ABORT here ends it and saves nothing.`;
}

function waitingText(name) {
  // the painter job is queued behind another job: it has not opened yet
  return `Waiting to paint ${name}: another job still holds the scanner. The painter opens on the projector when that job ends; nothing is drawn yet. ABORT here drops it.`;
}

function paintNow(lp, fresh) {
  // what is true of the saved set NOW, from the selection the server stores - never from
  // what was so when it was saved (it may have been switched off, painted again or deleted
  // since, or belong to another rig placement than the one this page shows). fresh: this
  // page saw the session end; else the message is history (a reload, a second screen)
  if (lp.on !== true && lp.on !== false) return lp.selection_note || 'The zone selection was not changed.';
  const s = zsel();
  if (!s || s.state === 'unknown') return '';     // not known: nothing is said about its switch
  if (lp.geometry_id && s.geometry_id && lp.geometry_id !== s.geometry_id) {
    return `That was at another rig placement${lp.preset ? ` (preset ${lp.preset})` : ''}: it is not one of the sets listed here.`;
  }
  const r = (s.sets || []).find((x) => x.set === lp.name);
  if (!r || r.found === false) return 'It is not in the list any more: it was deleted since.';
  if (lp.created && r.created && r.created !== lp.created) return 'It was painted again since.';
  if (!r.on) {
    if (!fresh) return 'It is switched OFF now: switch it on in the list to select it.';
    return lp.replaced ? 'It stays switched OFF, as before: switch it on in the list to select it.'
      : 'It is NOT switched on: switch it on in the list to select it.';
  }
  const all = r.n_on === r.n_zones;
  if (!fresh) return `It is switched ON now${all ? '' : ` (${r.n_on} of ${r.n_zones} zones)`}.`;
  let t = lp.replaced ? 'It stays switched ON.' : 'It is switched ON.';
  if (!all) t += ` ${r.n_on} of ${r.n_zones} zones are on.`;
  else if (lp.switches_reset) t += ` All ${r.n_zones} zones are on (the per-zone switches were reset: the zones were painted again).`;
  return `${t} ${zoneScope().scope}`.trim();      // what the run button does with it, now
}

function paintText(lp, fresh) {
  // how the last painter session ended, from its outcome (never by reading the engine's
  // English). What happened then is said with its date and time; what is so now: paintNow
  const n = lp.name || 'the set';
  const at = when(lp.time) ? ` ${when(lp.time)}` : '';
  const head = fresh ? '' : 'Last painter session - ';
  switch (lp.outcome) {
    case 'saved': {
      const zs = (lp.zones || []).map((z) => z.id + (z.name ? ` ${z.name}` : '')).join(', ');
      let t = `${head}SAVED${at}: zone set ${n}${lp.replaced ? ' replaced' : ''}, ${plural(lp.n_zones || 0, 'zone')}: ${zs}.`;
      if (lp.replaced) t += ' The earlier one is in the library trash.';
      t += ` ${paintNow(lp, fresh)}`;
      if (!lp.photo) t += ' No photo of the zones could be taken (the drawing is shown instead).';
      if (lp.rig_error) t += ` The rig was NOT given back cleanly after the save (${lp.rig_error}): check the camera and the projector before the next scan.`;
      return t.replace(/\s+/g, ' ').trim();
    }
    case 'cancelled': return `${head}NOT SAVED${at}: ${n} - the painter was cancelled (ESC). Nothing was written.`;
    case 'aborted': return `${head}NOT SAVED${at}: ${n} - ABORT ended the painter. Nothing was written.`;
    case 'idle_timeout': return `${head}NOT SAVED${at}: ${n} - no input for ${lp.idle_timeout_s || 600} s, so the rig was given back. Nothing was written.`;
    case 'script_ended': return `${head}NOT SAVED${at}: ${n} - the scripted session ended without SPACE.`;
    case 'dropped': return `${head}NOT SAVED${at}: ABORT came before the painter started.`;
    default: return `${head}NOT SAVED${at}: the painter did not start - ${lp.reason || 'refused'}.`;
  }
}

function renderZoneMsg() {
  const m = state.zoneMsg;
  const el = $('#qZoneResult');
  el.textContent = !m ? '' : (m.paint ? paintText(m.paint, !!m.fresh) : m.text);
  el.classList.toggle('qwarn', !!(m && (m.warn || (m.paint && (m.paint.outcome !== 'saved' || m.paint.rig_error)))));
  if (m && m.reveal) {
    // how the painter ended, in view the moment it is known (the operator scrolled down to
    // the Paint zones button; the page's scroll padding keeps the line above the run buttons)
    m.reveal = false;
    if (typeof el.scrollIntoView === 'function') el.scrollIntoView({ block: 'nearest' });
  }
}

function renderPainter(b) {
  // "Painting ..." while the painter holds the rig, then how it ended: both from the bus
  // state, so a reload, a second screen or one failed poll never loses the save message
  const p = b.painter || null;
  if (p && p.verb === 'zone_paint') {
    state.painting = { name: p.name, at: hm(p.since) };
    // queued: another job still holds the scanner, nothing is on the projector yet
    state.zoneMsg = { text: p.state === 'queued' ? waitingText(p.name) : paintingText(p.name),
      warn: false, painting: true };
    return;
  }
  state.painting = null;
  const lp = b.last_paint || null;
  const key = lp ? `${lp.time}|${lp.name}|${lp.outcome}` : '';
  if (key === state.paintKey) {
    if (state.zoneMsg && state.zoneMsg.painting) state.zoneMsg = lp ? { paint: lp, fresh: true, reveal: true } : null;
    return;
  }
  // the first bus state of this page: the session ended before the page was loaded, so it
  // is shown as history. Later ones this page saw end
  const first = state.paintKey === null;
  state.paintKey = key;
  if (!lp) return;
  state.zoneMsg = { paint: lp, fresh: !first, reveal: !first };
  if (lp.outcome === 'saved') {
    if (!first) $('#qZoneName').value = '';
    loadZoneSets(lp.name).catch(() => {});      // listed, viewed, its photo shown
  }
}

function focusedSwitch(box) {
  // the zone switch that has the keyboard focus -> { set, zone } (null: none has it)
  const a = document.activeElement;
  if (!a || !a.dataset || !a.dataset.set || typeof box.contains !== 'function' || !box.contains(a)) return null;
  return { set: a.dataset.set, zone: a.dataset.zone || '' };
}

function refocusSwitch(box, had, made) {
  // the rows were rebuilt, so the switch that had the focus is gone: the same switch gets it
  // back. Left alone, the focus would fall to the page, where Space / Enter is the run
  // button - a second Space meant for the switch would start a run. A switch that is gone or
  // cannot be pressed now: the Scan zones panel takes the focus (Space / Enter start nothing
  // there)
  if (!had) return;
  const to = made.find((c) => c.dataset.set === had.set && (c.dataset.zone || '') === had.zone);
  const el = to && !to.disabled ? to : $('#qScanZones');
  if (typeof el.focus === 'function') el.focus();
}

function renderZoneRows(b) {
  // one row per zone set of this rig placement: its switch, one switch per zone; sets that
  // are on first. Drawn from the bus state's zone_selection (what is STORED), so a refused
  // click never stays checked
  const s = zsel();
  const free = zonesFree();
  const key = JSON.stringify([s === undefined ? '?' : s, state.zoneOthers, free]);
  if (key === state.zoneKey) return;
  state.zoneKey = key;
  const box = $('#qZoneRows');
  const had = focusedSwitch(box);
  const made = [];
  box.textContent = '';
  const geo = $('#qZoneGeo');
  if (s === undefined) {
    geo.textContent = 'Which zone sets are switched on is NOT KNOWN: the server has not said.';
    refocusSwitch(box, had, made);
    return;
  }
  if (s === null) {
    geo.textContent = 'Load a preset first: zone sets belong to a rig placement.';
    refocusSwitch(box, had, made);
    return;
  }
  const rows = [...(s.sets || [])].sort((x, y) => (Number(!!y.on) - Number(!!x.on)) || (x.set < y.set ? -1 : 1));
  let head = `Zone sets at this rig placement (preset ${s.preset}). Switch a set on to select it; the line above the run button says what the next run does with the selection.`;
  if (s.state === 'unknown') head = `The stored zone selection cannot be read (${selProblem(s)}). Nothing is known to be on: switch the sets on again.`;
  else if (s.time) head += ` Last changed ${String(s.time).replace('T', ' ')}${s.by ? ` by ${s.by}` : ''}.`;
  if (!rows.length) {
    const o = state.zoneOthers || [];
    const n = o.reduce((a, x) => a + (x.n_sets || 0), 0);
    const ps = [...new Set(o.flatMap((x) => x.presets || []))];
    head = n ? `No zone set at this rig placement (preset ${s.preset}). ${plural(n, 'set')} ${n === 1 ? 'was' : 'were'} painted `
      + `${ps.length ? `with preset ${ps.join(', ')}` : 'elsewhere'} (another rig placement): load that preset to use ${n === 1 ? 'it' : 'them'}.`
      : 'No zone set painted at this rig placement yet. Name one and press Paint zones.';
  }
  geo.textContent = head;
  const check = (set, zone, on, off, tip) => {
    const c = document.createElement('input');
    c.type = 'checkbox';
    c.checked = !!on;
    c.disabled = !!off;
    c.title = tip;
    c.dataset.set = set;
    if (zone) c.dataset.zone = zone;
    made.push(c);
    return c;
  };
  for (const r of rows) {
    const gone = r.found === false;
    const row = document.createElement('div');
    row.className = `qzrow${r.on ? ' on' : ''}${gone || (r.on && !r.usable) ? ' bad' : ''}${r.on && (r.rig_changes || []).length ? ' check' : ''}`;
    row.dataset.set = r.set;
    const lab = document.createElement('label');
    lab.className = 'qzset';
    // a set that cannot be used can only be switched OFF
    lab.appendChild(check(r.set, '', r.on, !free || (!r.usable && !r.on),
      'On: this set\'s zones are selected. Off: the set is left out. What a run does with the selection is said above the run button.'));
    const name = document.createElement('b');
    name.textContent = r.set;
    lab.appendChild(name);
    const cnt = document.createElement('span');
    cnt.textContent = gone ? 'NOT FOUND (deleted or renamed)' : plural(r.n_zones, 'zone');
    lab.appendChild(cnt);
    row.appendChild(lab);
    const zs = document.createElement('span');
    zs.className = 'qzzones';
    for (const z of r.zones || []) {
      const zl = document.createElement('label');
      zl.appendChild(check(r.set, z.id, z.on, !free || !r.usable,
        'Off: this one zone is left out; the rest of the set stays on.'));
      const t = document.createElement('span');
      t.textContent = z.id + (z.name ? ` ${z.name}` : '');
      zl.appendChild(t);
      zs.appendChild(zl);
    }
    row.appendChild(zs);
    const notes = [];
    if (gone) notes.push(`${r.set}: ${r.error || 'not in the library any more'}. Switch it off to take it out of the selection.`);
    else if (!r.usable) notes.push(`not usable: ${r.error || 'it cannot be read'}`);
    if (!gone && (r.rig_changes || []).length) notes.push(`CHECK: since it was painted, ${r.rig_changes.slice(0, 3).join('; ')} - show it on the part (Show selected zones) or paint it again.`);
    if (notes.length) {
      const nt = document.createElement('span');
      nt.className = 'qznote';
      nt.textContent = notes.join(' ');
      row.appendChild(nt);
    }
    box.appendChild(row);
  }
  refocusSwitch(box, had, made);
}

function renderZones() {
  // the Scan zones panel and the line above the run buttons, from the bus state
  const b = state.bus || {};
  renderPainter(b);
  renderZoneRows(b);
  const sc = zoneScope();
  $('#qZoneLine').textContent = sc.line;
  $('#qZoneLine').classList.toggle('qwarn', !!sc.warn);
  $('#qRunScope').textContent = sc.scope;
  $('#qMeasZone').textContent = state.opMode === 'measure' ? sc.scope : '';   // (repeated in the Measure pane)
  renderZoneMsg();
  const s = zsel();
  // another selection: what the operator confirmed was not this one
  const stamp = s ? s.stamp : null;
  if (state.zoneStamp !== null && stamp !== state.zoneStamp) $('#qConfirm').checked = false;
  state.zoneStamp = stamp;
  // the picker, the photos and who uses a set (GET /api/zonesets): re-read when the sets changed
  const sig = s ? JSON.stringify([s.geometry_id, (s.sets || []).map((r) => [r.set, r.created])]) : '-';
  if (sig !== state.zoneSetsSig) {
    const firstTime = state.zoneSetsSig === null;
    state.zoneSetsSig = sig;
    if (!firstTime) loadZoneSets().catch(() => {});
  }
}

async function loadZoneSets(pick) {
  // the sets painted at the loaded preset's geometry (their photo, who uses them) for the
  // picker; `pick`: the one to view afterwards
  let d;
  try { d = await api('GET', '/api/zonesets'); } catch (e) { return; }
  state.zoneSets = d.sets || [];
  state.zoneOthers = d.others || [];
  if (state.bus && d.selection !== undefined) state.bus.zone_selection = d.selection;
  const sel = $('#qZoneSet');
  const want = pick !== undefined ? pick : sel.value;
  sel.textContent = '';
  const none = document.createElement('option');
  none.value = '';
  none.textContent = '(a new set: type its name)';
  sel.appendChild(none);
  for (const z of state.zoneSets) {
    const o = document.createElement('option');
    o.value = z.name;
    o.textContent = `${z.name} - ${plural(z.n_zones, 'zone')}` + (z.valid ? '' : ' (invalid)');
    sel.appendChild(o);
  }
  sel.value = state.zoneSets.some((z) => z.name === want) ? want : '';
  showZoneSet();
  state.zoneKey = null;
  renderZones();
}

function usedBy(z) {
  // who names the set (their inspected area changes with it)
  const u = (z && z.used_by) || {};
  const bits = [];
  if ((u.recipes || []).length) bits.push(`recipe${u.recipes.length === 1 ? '' : 's'} ${u.recipes.join(', ')}`);
  if ((u.stations || []).length) bits.push(`station configuration${u.stations.length === 1 ? '' : 's'} ${u.stations.join(', ')}`);
  return bits.join(' and ');
}

function showZoneSet() {
  // the viewed set: its zones and its picture (the photo of the zones on the part, else the
  // DMD-space preview). It does not decide what a run uses: the switches do
  const z = (state.zoneSets || []).find((x) => x.name === $('#qZoneSet').value);
  const img = $('#qZoneImg');
  const link = z && (z.camera || z.preview);
  img.hidden = !link;
  if (link) img.src = `/files/${link}?t=${encodeURIComponent(z.created || '')}`;
  const used = usedBy(z);
  $('#qZoneInfo').textContent = !z ? ''
    : (z.valid ? `Viewing ${z.name}: ` + (z.zones || []).map((q) => q.id + (q.name ? ` ${q.name}` : '')).join(', ')
      + ` (painted ${String(z.created || '?').replace('T', ' ')}; projector px, valid at this rig placement).`
      + (used ? ` Used by ${used}.` : '')
      + ((z.rig_changes || []).length ? ` CHECK (Show selected zones) or paint again: since it was painted, ${z.rig_changes.slice(0, 3).join('; ')}.` : '')
      : `${z.name} is not usable: ${z.error}`);
}

function focusOn(sel) {
  const el = $(sel);
  if (el && typeof el.focus === 'function') el.focus();
}

function openAsk(a) {
  // one question at a time, in the page (no browser dialog): the focus goes to the answer
  // that changes nothing (delete: to the reason). a.from: the button that asked (the focus
  // goes back to it when the question closes)
  state.ask = a;
  const box = $('#qZoneAsk');
  box.hidden = false;
  $('#qZoneAskText').textContent = a.text;
  $('#qZoneReasonRow').hidden = !a.reason;
  $('#qZoneReason').value = '';
  $('#qZoneAskYes').textContent = a.yes;
  focusOn(a.reason ? '#qZoneReason' : '#qZoneAskNo');
  // the whole question in view (never half under the run buttons that stay at the bottom)
  if (typeof box.scrollIntoView === 'function') box.scrollIntoView({ block: 'nearest' });
}

function closeAsk(to) {
  // the question goes away. When the focus was on one of its answers it moves to `to`
  // (default: the button that asked; a button that is switched off: the Scan zones panel).
  // Hiding the question would drop the focus on the page, where Space / Enter is the run
  // button
  const a = state.ask;
  state.ask = null;
  const box = $('#qZoneAsk');
  const act = document.activeElement;
  const inside = !!a && !!act && typeof box.contains === 'function' && box.contains(act);
  box.hidden = true;
  if (!inside) return;
  const back = to || a.from || '#qScanZones';
  focusOn($(back) && !$(back).disabled ? back : '#qScanZones');
}

async function postSelection(sets, said) {
  // the whole selection as it should be stored, with the stamp of the one this page showed:
  // the answer always carries the STORED selection, and the rows are redrawn from it. Until
  // the answer is here the page says so, and takes no second change and no run key
  const s = zsel();
  state.selBusy = true;
  zoneSay('Changing the zone selection...');
  try {
    const r = await api('POST', '/api/zones/selection', { sets, was: s ? s.stamp : undefined, source: 'quick' });
    if (r.bus) state.bus = r.bus;
    zoneSay(r.accepted ? said : `NOT changed: ${r.message || r.error || 'refused'}`, !r.accepted);
  } catch (e) {
    if (e.data && e.data.bus) state.bus = e.data.bus;
    zoneSay(`NOT changed: ${e.message || e}`, true);
  }
  state.selBusy = false;
  state.zoneKey = null;
  renderZones();
  refreshButtons();
}

function storedEntries() {
  // what is stored, as the entries POST /api/zones/selection takes
  const s = zsel();
  return ((s && s.sets) || []).map((r) => ({ set: r.set, on: !!r.on, ids: r.ids || null }));
}

function toggleSet(name, on) {
  closeAsk();
  if (!zsel()) return Promise.resolve();
  const sets = storedEntries().map((e) => (e.set === name ? { ...e, on: !!on } : e));
  return postSelection(sets, `${name} is switched ${on ? 'ON' : 'OFF'}.`);
}

function toggleZone(name, id, on) {
  closeAsk();
  const s = zsel();
  const r = ((s && s.sets) || []).find((x) => x.set === name);
  if (!r) return Promise.resolve();
  const ids = (r.zones || []).filter((z) => (z.id === id ? on : z.on)).map((z) => z.id);
  let entry;
  let said = `${name}/${id} is switched ${on ? 'on' : 'off'}.`;
  if (!ids.length) {
    if (!r.on) {                                 // an OFF set keeps at least one zone to come back with
      zoneSay(`${name}: at least one zone stays switched on in a set (the set itself is off).`, true);
      state.zoneKey = null;
      renderZones();
      return Promise.resolve();
    }
    // the last zone of a set that is on: the set is switched off, its switches stay
    entry = { set: name, on: false, ids: r.ids || null };
    said = `${name}/${id} was the last zone that was on: ${name} is switched OFF.`;
  } else entry = { set: name, on: !!r.on, ids: ids.length === (r.zones || []).length ? null : ids };
  return postSelection(storedEntries().map((e) => (e.set === name ? entry : e)), said);
}

function onZoneRowChange(ev) {
  // one listener on the rows: a set's switch, or one zone's
  const t = ev && ev.target;
  if (!t || !t.dataset || !t.dataset.set) return;
  if (state.selBusy) {
    // the last change is not answered yet: this one is not taken (the rows are redrawn from
    // what the server stores when the answer is here)
    t.checked = !t.checked;
    return;
  }
  if (t.dataset.zone) toggleZone(t.dataset.set, t.dataset.zone, !!t.checked);
  else toggleSet(t.dataset.set, !!t.checked);
}

async function zoneJob(verb, args, busyText) {
  // one zone job of this page (show, delete), followed to its end. A poll that fails is
  // asked again: one lost answer is not the job's end
  state.zoneBusy = true;
  refreshButtons();
  zoneSay(busyText);
  try {
    let j = await api('POST', '/api/jobs', { verb, args: args || {} });
    let fails = 0;
    while (j.status === 'queued' || j.status === 'running') {
      await new Promise((res) => setTimeout(res, 400));
      try {
        j = await api('GET', `/api/jobs/${j.id}`);
        fails = 0;
      } catch (e) {
        fails += 1;
        if (fails > 75) throw e;                 // half a minute without an answer
      }
    }
    return j;
  } catch (e) {
    zoneSay(`NOT KNOWN: lost contact with the server (${e.message || e}). Look at the list of zone sets.`, true);
    return null;
  } finally {
    state.zoneBusy = false;
    refreshButtons();
  }
}

async function startPaint(args) {
  // the painter job; the page then only follows the bus state (painter, last_paint)
  closeAsk('#qScanZones');                       // (Paint zones is switched off in a moment)
  state.zoneBusy = true;
  refreshButtons();
  zoneSay(paintingText(args.name));
  try {
    const j = await api('POST', '/api/jobs', { verb: 'zone_paint', args });
    if (j.error || !j.id) zoneSay(`NOT SAVED: the painter did not start - ${j.error || 'refused'}.`, true);
    else state.painting = { name: args.name, at: hm(j.created) };
  } catch (e) {
    zoneSay(`NOT SAVED: the painter did not start - ${e.message || e}.`, true);
  }
  state.zoneBusy = false;
  try { await pollOnce(); } catch (e) { $('#qConn').hidden = false; }
  refreshButtons();
}

function paintZones() {
  // Paint zones: a NEW set needs its name first; an existing one (typed, or the viewed set)
  // is painted over only after the question (the set as it is goes to the library trash
  // when the painter saves)
  closeAsk();
  const typed = $('#qZoneName').value.trim();
  const name = typed || $('#qZoneSet').value;
  if (!name) {
    zoneSay('Name the new zone set first (New set name), or pick a set to edit.', true);
    focusOn('#qZoneName');
    return Promise.resolve();
  }
  const z = (state.zoneSets || []).find((x) => x.name === name);
  if (!z) return startPaint({ name, select: true });
  const used = usedBy(z);
  openAsk({ kind: 'paint', name, from: '#qPaint', yes: `Replace ${name}`,
    args: { name, source: name, replace: true, select: true },
    text: `Zone set ${name} exists (${plural(z.n_zones, 'zone')}, painted ${String(z.created || '?').replace('T', ' ')}). `
      + 'Paint over it? The painter opens with its zones. When you save with SPACE the set as it is now moves to the library trash (it can be brought back). ESC in the painter leaves it unchanged.'
      + (used ? ` It is used by ${used}: what they inspect changes with it.` : '') });
  return Promise.resolve();
}

function deleteZoneSet() {
  // Delete set: the viewed set, to the library trash - after a question, with a reason. A set
  // that is switched on is switched off first (a run never loses its zones behind its back)
  closeAsk();
  const name = $('#qZoneSet').value;
  if (!name) { zoneSay('Pick the set to delete under Set to view or edit.', true); return; }
  const s = zsel();
  const r = ((s && s.sets) || []).find((x) => x.set === name);
  if (r && r.on) { zoneSay(`${name} is switched on. Switch it off first, then delete it.`, true); return; }
  const z = (state.zoneSets || []).find((x) => x.name === name) || {};
  const used = usedBy(z);
  openAsk({ kind: 'delete', name, reason: true, from: '#qZoneDelete', yes: `Move ${name} to the trash`,
    text: `Move zone set ${name} (${plural(z.n_zones || 0, 'zone')}) to the library trash? It can be brought back from there. Give a reason (it is recorded).`
      + (used ? ` It is used by ${used}: the delete is refused while they name it.` : '') });
}

async function confirmAsk() {
  // the Yes button of the open question
  const a = state.ask;
  if (!a) return;
  if (a.kind === 'paint') { await startPaint(a.args); return; }
  const reason = $('#qZoneReason').value.trim();
  if (!reason) {
    $('#qZoneAskText').textContent = `${a.text} Type a reason first.`;
    focusOn('#qZoneReason');
    return;
  }
  closeAsk('#qScanZones');                       // (Delete set is switched off in a moment)
  const j = await zoneJob('zone_delete', { name: a.name, reason, confirm: true }, `Moving ${a.name} to the library trash...`);
  if (!j) return;
  if (j.status === 'done') zoneSay(`DELETED: ${a.name} is in the library trash.`);
  else zoneSay(`NOT deleted: ${j.error || 'refused'}`, true);
  await loadZoneSets('');
  try { await pollOnce(); } catch (e) { $('#qConn').hidden = false; }
}

async function showZonesOnPart() {
  // Show selected zones: every switched-on zone projected on the part and photographed
  closeAsk();
  const s = zsel();
  if (!s || s.state === 'unknown') { zoneSay(`Not shown: ${selProblem(s)}.`, true); return; }
  if (s.empty) { zoneSay('No zone set is switched on: there is nothing to show.'); return; }
  if (!s.ok) { zoneSay(`Not shown: ${selProblem(s)}. Switch that set off or paint it again.`, true); return; }
  const j = await zoneJob('zone_show', { zones: `selection@${s.stamp}`, hold_s: 5 },
    `Showing ${s.label} on the part for 5 s...`);
  if (!j) return;
  const r = j.status === 'done' ? j.result : null;
  if (!r) { zoneSay(`Not shown: ${j.error || 'refused'}`, true); return; }
  const zs = r.zones || [];
  const miss = zs.filter((z) => !z.seen).map((z) => z.key || z.id);
  zoneSay(miss.length
    ? `Shown: ${zs.length - miss.length} of ${zs.length}. NOT SEEN: ${miss.join(', ')} (outside the camera's view, in shadow, or on nothing).`
    : `Shown: ${zs.length} of ${plural(zs.length, 'zone')} land${zs.length === 1 ? 's' : ''} in the camera image. Look at the part: an outline that does not sit where it should means paint that set again.`,
  miss.length > 0);
  const link = r.overlay_png || r.camera_png;
  if (link && !/^[A-Za-z]:|^\//.test(link)) {                      // library-relative only
    $('#qZoneImg').hidden = false;
    $('#qZoneImg').src = `/files/${link}`;
  }
  try { await pollOnce(); } catch (e) { $('#qConn').hidden = false; }    // a CHECK it answered is gone
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
  const field = !!t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA'
                        || t.isContentEditable);
  if (state.ask) {
    // a question of the Scan zones panel is open: the keys answer it and start nothing.
    // Esc closes it (and still sends ABORT: Esc is ABORT everywhere); Enter in the reason
    // box presses the Yes button; Space / Enter elsewhere press only a focused button
    if (ev.key === 'Escape') {
      closeAsk();
      zoneSay('Cancelled (Esc): nothing was changed. ABORT was sent too.');
      send('abort');
      ev.preventDefault();
    } else if (ev.key === 'Enter' && t && t.id === 'qZoneReason') {
      confirmAsk();
      ev.preventDefault();
    }
    return;
  }
  if (painterOn()) {
    // the painter (or a zone show) holds the rig: its keys - P polygon, N name, H legend,
    // digits ... - belong to the projector display. Here only Esc works: ABORT
    if (ev.key === 'Escape') { send('abort'); ev.preventDefault(); }
    return;
  }
  if (field) {
    if (ev.key !== 'Escape') return;                // Esc = ABORT, even from a field
  }
  if (ev.ctrlKey || ev.altKey || ev.metaKey) return;
  const k = ev.key;
  if ((k === ' ' || k === 'Enter') && t && t.closest && t.closest(ACTIVATABLE)) return;
  if ((k === ' ' || k === 'Enter') && state.selBusy) {
    // a zone switch was just pressed and its change is not answered yet: a second Space /
    // Enter is meant for the switches, never for the run button
    ev.preventDefault();
    return;
  }
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
    renderZones();                     // another configuration: another zone, another use
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
    renderZones();
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
  // what the run button does with the zones depends on the measurement / check and on the
  // region typed: the line above the run buttons follows them at once
  const rescope = () => { renderZones(); refreshButtons(); };
  $('#qMeasWhat').addEventListener('change', () => { measFields(); rescope(); });
  $('#qQuickCheck').addEventListener('change', () => { quickFields(); rescope(); });
  for (const id of ['#qMeasRoi', '#qMeasP2', '#qQuickRoi']) $(id).addEventListener('input', rescope);
  $('#qQuickRecipe').addEventListener('change', rescope);
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
  $('#qZoneDelete').addEventListener('click', deleteZoneSet);
  $('#qZoneSet').addEventListener('change', () => { closeAsk(); showZoneSet(); });
  $('#qZoneRows').addEventListener('change', onZoneRowChange);       // every switch, one listener
  $('#qZoneAskYes').addEventListener('click', confirmAsk);
  $('#qZoneAskNo').addEventListener('click', () => { closeAsk(); zoneSay('Cancelled: nothing was changed.'); });
  $('#qZoneImg').addEventListener('click', () => $('#qZoneImg').classList.toggle('big'));
  // zone sets belong to the loaded preset's geometry: read now, and again whenever the bus
  // state shows another set list (another preset, a set painted or deleted - by anyone)
  loadZoneSets();
  $('#qLive').addEventListener('click', toggleLive);
  $('#qSignal').addEventListener('click', signalCheck);
  document.addEventListener('keydown', onKey);
  loadStations().catch(() => { $('#qConn').hidden = false; }).finally(loop);
});
