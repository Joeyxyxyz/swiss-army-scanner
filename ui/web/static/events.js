// Event log page (/static/events.html): the inspection event log of one month as a plain
// table, newest first. The engine appends the rows (engine/eventlog.py); this page only reads
// the csv files through the library file route:
//   /files/events/inspections_v1_<YYYY-MM>.csv           one row per inspection
//   /files/events/pending/inspections_v1_<YYYY-MM>.csv   rows still waiting to be written
// Nothing is sent to the server. No inline handlers (CSP), no external scripts or fonts.
// The whole month is read (tens of MB late in a busy month: a few seconds). Whoever wants
// only the row of the inspection that just ended reads /files/events/last_v1.csv instead:
// the header and that one row.
//
// Words: Inspection no. = the run id. Part ID = the code read from the part. Program = what
// was checked. Judgment = PASS / FAIL / PULL ("-" for a measurement, which decides nothing).
// An inspection can have more than one row (written again when something happened to the
// run after its result, e.g. ABORT after the verdict): the last one counts.
'use strict';

const $ = (sel) => document.querySelector(sel);
const FILES = '/files/';
const EVENTS = 'events/';
const PAGE = 500;                       // rows shown at once (a month can hold many thousands)
const state = { month: null, rows: [], shown: PAGE, waiting: 0 };

function parseCsv(text) {
  // RFC 4180: fields in double quotes may hold commas and doubled quotes -> rows of cells.
  // Whole stretches between the special characters are taken at once: a month's file is
  // tens of MB, and one character at a time takes seconds on it.
  const rows = [];
  const s = text.charCodeAt(0) === 0xFEFF ? text.slice(1) : text;
  const special = /[",\r\n]/g;
  let row = [], cell = '', i = 0;
  while (i < s.length) {
    special.lastIndex = i;
    const m = special.exec(s);
    const j = m ? m.index : s.length;
    if (j > i) cell += s.slice(i, j);
    i = j + 1;
    if (!m) break;
    if (m[0] === '"') {                                     // up to the closing quote
      for (;;) {
        const q = s.indexOf('"', i);
        if (q < 0) { cell += s.slice(i); i = s.length; break; }
        cell += s.slice(i, q);
        i = q + 1;
        if (s[i] !== '"') break;
        cell += '"';                                        // "" inside quotes: one quote
        i += 1;
      }
    } else if (m[0] === ',') { row.push(cell); cell = ''; }
    else {
      if (m[0] === '\r' && s[i] === '\n') i += 1;
      row.push(cell); cell = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    }
  }
  if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

function plain(cell) {
  // the log puts an apostrophe before text a spreadsheet would take for a formula: not shown
  if (!cell) return '';
  return cell.charCodeAt(0) === 39 && /^'[=+\-@]/.test(cell) ? cell.slice(1) : cell;
}

function rowsOf(text) {
  // csv text -> one object per row, keyed by the header line
  const lines = parseCsv(text);
  const head = lines.shift() || [];
  return lines.map((cells) => {
    const o = {};
    for (let i = 0; i < head.length; i += 1) o[head[i]] = plain(cells[i]);
    return o;
  });
}

function lastRows(rows) {
  // one row per inspection: the last one written (rows are in the order they were written)
  const last = new Map();
  rows.forEach((r, i) => last.set(r.inspection_id || `#${i}`, r));
  return Array.from(last.values());
}

function monthNow() {
  return new Date().toISOString().slice(0, 7);              // the files are kept by UTC month
}

function validMonth(m) {
  return /^\d{4}-(0[1-9]|1[0-2])$/.test(m || '');
}

function shiftMonth(month, by) {
  const d = new Date(Date.UTC(Number(month.slice(0, 4)), Number(month.slice(5, 7)) - 1 + by, 1));
  return d.toISOString().slice(0, 7);
}

function fileUrl(path) {
  // a file cell -> the address /files/ serves it at; null when the file is outside the
  // library (a full path: the server does not serve it)
  if (!path || /^([A-Za-z]:|[\\/])/.test(path)) return null;
  return FILES + path.split('/').map(encodeURIComponent).join('/');
}

function localTime(iso) {
  const d = new Date(iso);
  if (!iso || Number.isNaN(d.getTime())) return iso || '';
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} `
    + `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined && text !== null) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

function link(text, path, download) {
  // a link to a library file; a file outside the library is named, not linked
  const url = fileUrl(path);
  if (!url) {
    const s = el('span', text, 'muted');
    s.title = `${path} (outside the library: not served by this page)`;
    return s;
  }
  const a = el('a', text);
  a.href = download ? `${url}?download=1` : url;
  a.target = '_blank';
  a.rel = 'noopener';
  return a;
}

const JUDGMENT_CLASS = { PASS: 'tag ok', FAIL: 'tag bad', PULL: 'tag warn' };

function rowEl(r) {
  const tr = document.createElement('tr');
  const time = el('td', localTime(r.time), 'evtime');
  time.title = r.time || '';
  if (r.waiting) {
    const wait = el('span', 'waiting', 'tag warn');
    wait.title = 'Not in the file of this month yet: the file was open in another program, '
      + 'or the log was busy';
    time.appendChild(document.createTextNode(' '));
    time.appendChild(wait);
  }
  tr.appendChild(time);
  const id = el('td', r.inspection_id, 'evid mono');
  if (r.mode === 'MOCK') {
    const mock = el('span', 'MOCK', 'tag warn');
    mock.title = 'A simulated inspection: no real part was measured';
    id.appendChild(document.createTextNode(' '));
    id.appendChild(mock);
  }
  tr.appendChild(id);
  tr.appendChild(el('td', r.part_id));
  tr.appendChild(el('td', r.part));
  tr.appendChild(el('td', r.program));
  const j = el('td');
  j.appendChild(el('span', r.judgment, JUDGMENT_CLASS[r.judgment] || 'muted'));
  tr.appendChild(j);
  tr.appendChild(el('td', r.reason, 'evwhy'));
  const vals = el('td', null, 'evvals');
  (r.values ? r.values.split('; ') : []).forEach((v) => vals.appendChild(el('div', v)));
  if (Number(r.n_out_of_limit) > 0) {
    vals.appendChild(el('span', `${r.n_out_of_limit} out of limit`, 'tag bad'));
  }
  tr.appendChild(vals);
  const files = el('td', null, 'evfiles');
  if (r.image_file) files.appendChild(link('image', r.image_file));
  if (r.profile_file) files.appendChild(link('profile', r.profile_file));
  if (r.data_3d_file) files.appendChild(link('3D data', r.data_3d_file, true));
  if (r.result_file) files.appendChild(link('result', r.result_file));
  tr.appendChild(files);
  return tr;
}

function matches(r, needle) {
  if (!needle) return true;
  return ['inspection_id', 'part_id', 'part', 'program', 'judgment', 'reason', 'values', 'time']
    .some((k) => String(r[k] || '').toLowerCase().includes(needle));
}

function render() {
  const needle = ($('#evFilter').value || '').trim().toLowerCase();
  const rows = state.rows.filter((r) => matches(r, needle));
  const body = $('#evBody');
  body.textContent = '';
  rows.slice(0, state.shown).forEach((r) => body.appendChild(rowEl(r)));
  const n = state.rows.length;
  $('#evCount').textContent = needle ? `${rows.length} of ${n} inspection(s)` : `${n} inspection(s)`;
  const empty = $('#evEmpty');
  empty.hidden = rows.length > 0;
  empty.textContent = n ? 'No row contains that text.' : `No inspections recorded in ${state.month}.`;
  const more = $('#evMore');
  more.hidden = rows.length <= state.shown;
  more.textContent = `Show ${Math.min(PAGE, rows.length - state.shown)} more`;
  const w = $('#evWaiting');
  w.hidden = !state.waiting;
  w.textContent = `${state.waiting} row${state.waiting === 1 ? '' : 's'} waiting: not in the file `
    + 'of this month yet (the file was open in another program, or the log was busy). They are '
    + 'written at the next inspection; if the file is open in another program, close it there.';
}

async function fetchRows(path) {
  // the rows of a csv file of the library; [] when there is no such file
  const r = await fetch(FILES + path, { cache: 'no-store' });
  if (r.status === 404) return [];
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return rowsOf(await r.text());
}

async function load(month) {
  if (!validMonth(month)) return;
  state.month = month;
  state.shown = PAGE;
  $('#evMonth').value = month;
  const name = `inspections_v1_${month}.csv`;
  const err = $('#evErr');
  $('#evCount').textContent = 'reading ...';                // a full month can take seconds
  try {
    const rows = await fetchRows(EVENTS + name);
    // rows that could not be written yet (the file open in another program, the log busy):
    // shown too
    const waiting = (await fetchRows(`${EVENTS}pending/${name}`))
      .map((r) => ({ ...r, waiting: true }));
    if (state.month !== month) return;                      // another month was picked meanwhile
    // one row per inspection (its last), newest first; the file is in the order the rows
    // were written, and a waiting row was written after the rows in the file
    state.rows = lastRows(rows.concat(waiting))
      .sort((a, b) => (a.time < b.time ? 1 : a.time > b.time ? -1 : 0));
    state.waiting = waiting.length;
    err.hidden = true;
  } catch (e) {
    state.rows = [];
    state.waiting = 0;
    err.textContent = `The event log could not be read (${e.message}).`;
    err.hidden = false;
  }
  for (const [id, file] of [['#evCsv', name], ['#evValuesCsv', `inspection_values_v1_${month}.csv`]]) {
    const a = $(id);
    a.href = `${FILES}${EVENTS}${file}?download=1`;
    a.hidden = !state.rows.length;
  }
  render();
}

function init() {
  const pick = $('#evMonth');
  pick.addEventListener('change', () => load(pick.value));
  $('#evPrev').addEventListener('click', () => load(shiftMonth(state.month, -1)));
  $('#evNext').addEventListener('click', () => load(shiftMonth(state.month, 1)));
  $('#evReload').addEventListener('click', () => load(state.month));
  $('#evFilter').addEventListener('input', () => { state.shown = PAGE; render(); });
  $('#evMore').addEventListener('click', () => { state.shown += PAGE; render(); });
  const asked = new URLSearchParams(window.location.search).get('month');
  load(validMonth(asked) ? asked : monthNow());
}

document.addEventListener('DOMContentLoaded', init);
