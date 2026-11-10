'use strict';
/* SMESH replay. Every count and every orb's brightness comes from
   data/replay.js (exported from a recorded run). Particle paths and timing
   are interpretive; their numbers are not. */
(() => {
  const R = window.SMESH_REPLAY;
  if (!R) return;
  const $ = id => document.getElementById(id);
  const fmt = n => Number(n || 0).toLocaleString('en-US');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');

  const SHORT = {
    'crypto.market_beats_model': 'Kalshi beats the vol model',
    'crypto.blend_beats_market': 'A blend beats Kalshi',
    'crypto.edge_strategy_profits': 'Edge strategy profits',
    'crypto.odds_calibrated': 'Crypto odds calibrated',
    'macro.kalshi_beats_last_print': 'Beats the last print',
    'macro.kalshi_beats_climatology': 'Beats climatology',
    'macro.surprise_moves_spy': 'Surprise moves SPY',
    'macro.surprise_moves_tlt': 'Surprise moves TLT'
  };
  const short = id => SHORT[id] || id.split('.').pop().replace(/_/g, ' ');
  const findings = (R.findings || []).map(([track, f]) => ({ track, ...f }));
  const finalStrength = Object.fromEntries(findings.map(f => [f.id, f.strength]));
  const hyps = Object.keys(R.signals || {}).sort((a, b) => a.localeCompare(b));
  const CONSENSUS = 0.30;

  /* ------------------------------------------------------------ the replay */
  const canvas = $('mesh');
  const ctx = canvas.getContext('2d');
  const instrument = document.querySelector('.instrument');
  const play = $('play-toggle');
  const tabs = [...document.querySelectorAll('[data-phase]')];

  const crypto = (R.windows || []).filter(w => w.track === 'crypto');
  const macro = (R.windows || []).filter(w => w.track === 'macro');
  const steps = Math.max(crypto.length, macro.length, 1);
  const GATHER = 14, STEP = 1.9, CONS = 9;
  const durations = [GATHER, steps * STEP, CONS];

  const NODES = {
    'venue.kalshi': [.05, .17, 'Kalshi', 'venue'], 'venue.massive': [.05, .45, 'Massive', 'venue'], 'venue.kraken': [.05, .73, 'Kraken', 'venue'],
    'scout.kalshi': [.22, .17, 'scout', 'agent'], 'scout.massive': [.22, .45, 'scout', 'agent'], 'scout.kraken': [.22, .73, 'scout', 'agent'],
    'analyst.crypto': [.40, .30, 'crypto analyst', 'analyst'], 'analyst.macro': [.40, .60, 'macro analyst', 'analyst'],
    'walker': [.575, .38, 'walker', 'agent'], 'skeptic': [.575, .74, 'skeptic', 'agent']
  };
  const ROUTES = {
    'venue.kalshi': ['scout.kalshi', ['analyst.crypto', 'analyst.macro']],
    'venue.massive': ['scout.massive', ['analyst.crypto', 'analyst.macro']],
    'venue.kraken': ['scout.kraken', ['analyst.crypto']]
  };

  let width = 0, height = 0, phase = 0, t = 0, previous = 0, frame = 0, visible = true;
  let paused = reduced.matches, particles = [], serial = 0, spawnedTo = -1;
  const shown = Object.fromEntries(hyps.map(h => [h, 0]));
  const proposed = new Set();

  const hash = n => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
  const P = id => {
    if (NODES[id]) return { x: NODES[id][0] * width, y: NODES[id][1] * height };
    const i = hyps.indexOf(id), n = Math.max(hyps.length, 1);
    const gap = hyps.length > 4 ? .03 : 0, half = Math.ceil(n / 2);
    const y = .1 + (i / Math.max(n - 1, 1)) * .76 + (i >= half ? gap : -gap) * .5;
    return { x: (width < 600 ? .78 : .76) * width, y: y * height };
  };
  const bez = (a, b, u) => {
    const mx = (a.x + b.x) / 2, k = 1 - u;
    return {
      x: k * k * k * a.x + 3 * k * k * u * mx + 3 * k * u * u * mx + u * u * u * b.x,
      y: k * k * k * a.y + 3 * k * k * u * a.y + 3 * k * u * u * b.y + u * u * u * b.y
    };
  };
  function track(a, b, alpha, dash) {
    const mx = (a.x + b.x) / 2;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.bezierCurveTo(mx, a.y, mx, b.y, b.x, b.y);
    ctx.strokeStyle = `rgba(235,235,235,${alpha})`; ctx.lineWidth = .65; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]);
  }
  function glyph(x, y, kind, size, alpha) {
    ctx.save(); ctx.translate(x, y);
    ctx.strokeStyle = ctx.fillStyle = `rgba(245,245,242,${alpha})`; ctx.lineWidth = 1; ctx.beginPath();
    if (kind === 'contradict') { ctx.moveTo(-size, -size); ctx.lineTo(size, size); ctx.moveTo(size, -size); ctx.lineTo(-size, size); ctx.stroke(); }
    else if (kind === 'dampen' || kind === 'cache') { ctx.arc(0, 0, size * 1.15, 0, Math.PI * 2); ctx.stroke(); }
    else if (kind === 'prediction') { ctx.rect(-size * .8, -size * .8, size * 1.6, size * 1.6); ctx.fill(); }
    else { ctx.arc(0, 0, size, 0, Math.PI * 2); ctx.fill(); }
    ctx.restore();
  }

  // A particle moves along one or more segments; `bounce` turns back.
  function spawn(at, path, kind, dur, extra) {
    particles.push({ id: serial++, start: at, path, kind, dur, ...(extra || {}) });
  }
  function position(p, u) {
    if (p.bounce) {
      const v = u < .5 ? u * 1.3 : (1 - u) * 1.3;
      return bez(P(p.path[0]), P(p.path[1]), Math.min(v, .65));
    }
    const segs = p.path.length - 1, s = Math.min(Math.floor(u * segs), segs - 1);
    return bez(P(p.path[s]), P(p.path[s + 1]), u * segs - s);
  }

  /* Gather: spawn rates follow the recorded fetches per time bucket. */
  const venues = Object.keys(R.gather.fetches_by_venue || {});
  const totals = R.gather.fetch_totals || {};
  const allFetches = venues.reduce((s, v) => s + R.gather.fetches_by_venue[v].reduce((a, b) => a + b, 0), 0);
  const scale = Math.min(1, 300 / Math.max(allFetches, 1));
  const carry = {};
  function gatherSpawn(bucket) {
    const at = bucket * GATHER / R.gather.buckets;
    for (const v of venues) {
      const tot = totals[v] || {}, n = Object.values(tot).reduce((a, b) => a + b, 0) || 1;
      const pLimited = (tot.rate_limited || 0) / n, pCache = (tot.cache || 0) / n;
      carry[v] = (carry[v] || 0) + R.gather.fetches_by_venue[v][bucket] * scale;
      while (carry[v] >= 1) {
        carry[v] -= 1;
        const id = `venue.${v}`, r = hash(serial), [scout, analysts] = ROUTES[id] || [];
        if (!scout) continue;
        const analyst = analysts[Math.floor(hash(serial + 7) * analysts.length)];
        const when = at + hash(serial + 3) * GATHER / R.gather.buckets;
        if (r < pLimited) spawn(when, [id, scout], 'refused', 2.2, { bounce: true });
        else if (r < pLimited + pCache) spawn(when, [scout, analyst], 'cache', 2.2);
        else spawn(when, [id, scout, analyst], 'data', 3.4);
      }
    }
  }

  /* Walk: the recorded signal events of each window, in order. */
  const signalAt = (id, w) => {
    let v = null;
    for (const e of R.signals[id]) { if (e.window <= w) v = e.intensity; }
    return v;
  };
  function walkSpawn(s) {
    const w = s + 1, at = s * STEP;
    const cw = crypto[s], mw = macro[s];
    if (cw) for (let i = 0; i < Math.min(cw.n || 0, 14); i++) spawn(at + i * .045, ['analyst.crypto', 'walker'], 'prediction', .9);
    if (mw) for (let i = 0; i < 4; i++) spawn(at + i * .08, ['analyst.macro', 'walker'], 'prediction', .9);
    for (const id of hyps) {
      R.signals[id].forEach((e, i) => {
        const late = at + .75 + (i % 5) * .06;
        if (e.kind === 'emit' && s === 0) spawn(at + .2, [id.startsWith('crypto') ? 'analyst.crypto' : 'analyst.macro', id], 'reinforce', 1.1, { proposes: id });
        else if (e.window === w && e.kind === 'reinforce') spawn(late, ['walker', id], 'reinforce', .95);
        else if (e.window === w && e.kind === 'contradict') spawn(late, ['walker', id], 'contradict', .95);
        else if (e.window === w && e.kind === 'dampen') spawn(late + .15, ['skeptic', id], 'dampen', 1.05);
      });
    }
  }

  function targets() {
    if (phase === 0) return Object.fromEntries(hyps.map(h => [h, 0]));
    if (phase === 2) return finalStrength;
    const s = Math.min(Math.floor(t / STEP), steps - 1);
    return Object.fromEntries(hyps.map(h => [h, signalAt(h, s + (t % STEP > .8 ? 1 : 0)) ?? 0]));
  }

  function spawnUpTo() {
    if (phase === 0) {
      const b = Math.min(Math.floor(t / (GATHER / R.gather.buckets)) + 1, R.gather.buckets - 1);
      while (spawnedTo < b) gatherSpawn(++spawnedTo);
    } else if (phase === 1) {
      const s = Math.min(Math.floor(t / STEP), steps - 1);
      while (spawnedTo < s) walkSpawn(++spawnedTo);
    }
  }

  function label(text, x, y, align, color, size) {
    ctx.fillStyle = color || '#929292'; ctx.font = `${size || 11}px Archivo, Helvetica, sans-serif`;
    ctx.textAlign = align || 'center'; ctx.fillText(text, x, y);
  }

  function draw() {
    if (!width) return;
    ctx.clearRect(0, 0, width, height);
    const small = width < 600;

    // Tracks.
    for (const [v, [scout, analysts]] of Object.entries(ROUTES)) {
      track(P(v), P(scout), .16);
      for (const a of analysts) track(P(scout), P(a), .1);
    }
    track(P('analyst.crypto'), P('walker'), .14); track(P('analyst.macro'), P('walker'), .14);
    for (const h of hyps) { track(P('walker'), P(h), .07); track(P('skeptic'), P(h), .05, [2, 5]); }

    // Nodes.
    for (const [id, [, , name, kind]] of Object.entries(NODES)) {
      const p = P(id);
      ctx.strokeStyle = '#a4a4a4'; ctx.lineWidth = 1; ctx.beginPath();
      if (kind === 'venue') ctx.rect(p.x - 5, p.y - 5, 10, 10); else ctx.arc(p.x, p.y, kind === 'analyst' ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = '#080808'; ctx.fill(); ctx.stroke();
      if (kind === 'analyst') { ctx.beginPath(); ctx.arc(p.x, p.y, 10, 0, Math.PI * 2); ctx.strokeStyle = '#3a3a3a'; ctx.stroke(); }
      if (kind === 'venue') label(name, p.x, p.y - 14, 'center', '#e8e8e4', small ? 10 : 12);
      else if (!small || kind !== 'agent' || id === 'walker' || id === 'skeptic') label(name, p.x, p.y + 22, 'center', '#8a8a8a', small ? 9 : 11);
    }

    // Particles.
    particles = particles.filter(p => t - p.start < p.dur);
    for (const p of particles) {
      const u = (t - p.start) / p.dur; if (u < 0) continue;
      const pos = position(p, u), fade = Math.min(1, u * 10, (1 - u) * 10);
      for (let j = 4; j >= 1; j--) { const q = position(p, Math.max(0, u - j * .012)); glyph(q.x, q.y, p.kind, small ? 1 : 1.4, fade * (.1 - j * .018)); }
      const base = p.kind === 'refused' ? .55 : .95;
      glyph(pos.x, pos.y, p.kind === 'refused' ? 'contradict' : p.kind, small ? 1.6 : 2.2, fade * base);
      if (p.proposes && u > .95) proposed.add(p.proposes);
    }

    // The field.
    const tg = targets();
    const fx = P(hyps[0] || 'walker').x;
    label('FIELD', fx, P(hyps[0] || 'walker').y - 26, 'center', '#6f6f6f', 10);
    for (const h of hyps) {
      shown[h] += ((tg[h] || 0) - shown[h]) * .06;
      const p = P(h), v = Math.max(0, Math.min(1, shown[h]));
      const live = phase === 2 || proposed.has(h) || v > .01;
      const r = 3 + v * (small ? 12 : 17);
      if (phase >= 1) {
        ctx.beginPath(); ctx.arc(p.x, p.y, 3 + CONSENSUS * (small ? 12 : 17), 0, Math.PI * 2);
        ctx.strokeStyle = '#2c2c2c'; ctx.setLineDash([2, 3]); ctx.stroke(); ctx.setLineDash([]);
      }
      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 2.2);
      g.addColorStop(0, `rgba(245,245,242,${live ? .08 + v * .5 : .03})`); g.addColorStop(1, 'rgba(245,245,242,0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, r * 2.2, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(245,245,242,${live ? .15 + v * .8 : .05})`; ctx.fill();
      ctx.strokeStyle = h.startsWith('crypto') ? '#9a9a9a' : '#6d6d6d'; ctx.stroke();
      const winner = phase === 2 && (finalStrength[h] || 0) >= CONSENSUS;
      if (winner) {
        const a = t * .5 + hyps.indexOf(h);
        ctx.beginPath(); ctx.arc(p.x, p.y, r + 6, a, a + .9); ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1.2; ctx.stroke(); ctx.lineWidth = 1;
      }
      if (!small) {
        const txt = short(h) + (phase >= 1 && live ? `  ${v.toFixed(2)}` : '');
        label(txt, p.x + 28, p.y + 4, 'left', winner ? '#f5f5f2' : (live ? '#9d9d9d' : '#555'), 11);
      }
    }
  }

  const stories = [
    'Scouts read three venues through a client that can only GET market data. Refused requests back off and retry; settled data comes from cache.',
    'Each window tests every claim on data it has not seen. Agreement restores a signal; contradiction and the skeptic’s doubt dim it.',
    'Nothing kept the weak claims alive. What still glows past the dashed ring is the mesh’s answer.'
  ];
  function counts() {
    const sum = k => venues.reduce((s, v) => s + ((totals[v] || {})[k] || 0), 0);
    const preds = (R.metrics.crypto?.test_predictions || 0) + macro.length;
    const consensus = findings.filter(f => f.consensus).length;
    const strongest = findings.reduce((m, f) => Math.max(m, f.strength), 0);
    return [
      [[sum('ok') + sum('rate_limited') + sum('error'), 'requests'], [sum('cache'), 'from cache'], [sum('rate_limited'), 'refused, retried']],
      [[crypto.length + macro.length, 'windows'], [preds, 'out-of-sample predictions'], [(R.paper_trades || []).length, 'paper trades']],
      [[hyps.length, 'hypotheses'], [consensus, 'reached consensus'], [strongest.toFixed(2), 'strongest signal']]
    ][phase];
  }
  function setPhase(i, keepTime) {
    phase = i; if (!keepTime) t = 0;
    particles = []; spawnedTo = -1;
    if (phase === 0) { proposed.clear(); hyps.forEach(h => shown[h] = 0); }
    if (phase >= 1) hyps.forEach(h => proposed.add(h));
    tabs.forEach((b, j) => b.setAttribute('aria-pressed', String(j === phase)));
    $('phase-story').textContent = stories[phase];
    counts().forEach(([n, l], j) => { $(`count-${'abc'[j]}`).textContent = typeof n === 'string' ? n : fmt(n); $(`label-${'abc'[j]}`).textContent = l; });
    spawnUpTo(); draw();
  }
  function status() {
    if (phase === 1) $('phase-time').textContent = `window ${Math.min(Math.floor(t / STEP) + 1, steps)} of ${steps}`;
    else if (phase === 0) $('phase-time').textContent = `${(R.run.duration_ms / 1000 / 60).toFixed(1)} min of fetching, compressed`;
    else $('phase-time').textContent = 'final strengths';
  }
  function tick(now) {
    frame = 0;
    if (paused || !visible || document.hidden) { previous = 0; return; }
    if (previous) t += Math.min((now - previous) / 1000, .06);
    previous = now;
    if (t >= durations[phase]) { setPhase((phase + 1) % 3); }
    spawnUpTo(); draw(); status();
    frame = requestAnimationFrame(tick);
  }
  function schedule() { if (frame) cancelAnimationFrame(frame); frame = 0; previous = 0; if (!paused && visible && !document.hidden) frame = requestAnimationFrame(tick); }
  function syncPlay() {
    play.setAttribute('aria-label', paused ? 'Play animation' : 'Pause animation');
    play.querySelector('span').textContent = paused ? 'Play' : 'Pause';
    play.querySelector('path').setAttribute('d', paused ? 'M6 4 15 10 6 16Z' : 'M7 5v10M13 5v10');
  }
  tabs.forEach(b => b.addEventListener('click', () => { setPhase(Number(b.dataset.phase)); status(); }));
  play.addEventListener('click', () => { paused = !paused; syncPlay(); schedule(); });
  reduced.addEventListener('change', () => { paused = reduced.matches; syncPlay(); schedule(); });
  document.addEventListener('visibilitychange', schedule);
  new IntersectionObserver(e => { visible = e[0].isIntersecting; schedule(); }, { threshold: 0 }).observe(canvas);
  new ResizeObserver(() => {
    width = canvas.clientWidth; height = canvas.clientHeight; const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
  }).observe(canvas);

  // Reduced motion: show the conclusion, still.
  if (reduced.matches) { setPhase(2); for (const h of hyps) shown[h] = finalStrength[h] || 0; draw(); }
  else setPhase(0);
  syncPlay(); status(); schedule();
  $('run-id').textContent = `SMESH / ${String(R.run.id || 'RUN').toUpperCase()}`;

  /* ------------------------------------------------------------ the study */
  const consensus = findings.filter(f => f.consensus).length;
  $('consensus-count').textContent = `${consensus} of ${findings.length} reached consensus`;
  const body = $('findings');
  for (const f of [...findings].sort((a, b) => b.strength - a.strength)) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<th scope="row">${f.consensus ? '<span class="consensus-mark">' : ''}${f.label}${f.consensus ? '</span>' : ''}<small>${f.track}</small></th>`
      + `<td>${f.confirmations} / ${f.contradictions}</td>`
      + `<td><span class="bar${f.consensus ? '' : ' weak'}" style="--v:${Math.min(f.strength, 1)}"><i></i>${f.strength.toFixed(2)}</span></td>`;
    body.appendChild(tr);
  }
  const started = R.run.started ? new Date(R.run.started) : null;
  const c = R.metrics.crypto || {}, m = R.metrics.macro || {};
  $('study-scope').textContent = `Run ${R.run.id}${started ? `, ${started.toISOString().slice(0, 10)}` : ''}: ${fmt(R.run.events)} journal events over ${(R.run.duration_ms / 60000).toFixed(1)} minutes. `
    + `${fmt(c.samples)} crypto samples in ${fmt(c.windows)} windows; ${fmt(m.releases)} macro releases in ${fmt(m.windows)} windows. `
    + 'One recorded period, not a forecast. Paper trades were never sent anywhere.';
  const sha = R.run.sha256 || {};
  $('hashes').textContent = `Sources: timeline.jsonl sha256 ${String(sha['timeline.jsonl'] || '').slice(0, 16)}…, metrics.json sha256 ${String(sha['metrics.json'] || '').slice(0, 16)}…`;

  /* ------------------------------------------------------------ the charts */
  const NS = 'http://www.w3.org/2000/svg';
  const el = (parent, name, attrs, text) => {
    const n = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    parent.appendChild(n); return n;
  };
  const reveal = (svg, nodes) => {
    nodes.forEach(n => { n.style.opacity = 0; n.style.transition = 'opacity .5s'; });
    new IntersectionObserver((e, o) => {
      if (!e[0].isIntersecting) return; o.disconnect();
      nodes.forEach((n, i) => setTimeout(() => { n.style.opacity = 1; }, reduced.matches ? 0 : 60 * i));
    }, { threshold: .3 }).observe(svg);
  };

  (function calibration() {
    const svg = $('chart-calibration'), L = 44, T = 14, W = 340, H = 240;
    const x = p => L + p * W, y = p => T + H - p * H;
    el(svg, 'line', { x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: '#444', 'stroke-dasharray': '3 4' });
    el(svg, 'line', { x1: L, y1: y(0), x2: L + W, y2: y(0), stroke: '#333' });
    el(svg, 'line', { x1: L, y1: T, x2: L, y2: y(0), stroke: '#333' });
    for (const v of [0, .5, 1]) {
      el(svg, 'text', { x: x(v), y: y(0) + 16, 'text-anchor': 'middle' }, v.toFixed(1));
      el(svg, 'text', { x: L - 8, y: y(v) + 3, 'text-anchor': 'end' }, v.toFixed(1));
    }
    el(svg, 'text', { x: L + W / 2, y: y(0) + 32, 'text-anchor': 'middle' }, 'predicted probability');
    el(svg, 'text', { x: 12, y: T + H / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${T + H / 2})` }, 'happened');
    const pts = [];
    for (const b of c.calibration_model || []) {
      const s = 2 + Math.min(Math.sqrt(b.n) * .5, 5);
      pts.push(el(svg, 'rect', { x: x(b.predicted) - s, y: y(b.observed) - s, width: s * 2, height: s * 2, stroke: '#cfcfcb', fill: 'none' }));
    }
    for (const b of c.calibration_market || []) {
      pts.push(el(svg, 'circle', { cx: x(b.predicted), cy: y(b.observed), r: 2 + Math.min(Math.sqrt(b.n) * .5, 5), fill: '#f5f5f2', stroke: 'none' }));
    }
    reveal(svg, pts);
  })();

  (function surprises() {
    const svg = $('chart-surprise'), L = 74, W = 300, Z = 8;
    const series = ['CPI', 'Payrolls', 'Fed'], rowY = { CPI: 60, Payrolls: 140, Fed: 220 };
    const x = z => L + (Math.max(-Z, Math.min(Z, z)) + Z) / (2 * Z) * W;
    el(svg, 'line', { x1: x(0), y1: 26, x2: x(0), y2: 252, stroke: '#444', 'stroke-dasharray': '3 4' });
    for (const s of series) {
      el(svg, 'line', { x1: L, y1: rowY[s], x2: L + W, y2: rowY[s], stroke: '#262626' });
      el(svg, 'text', { x: L - 12, y: rowY[s] + 3, 'text-anchor': 'end' }, s);
    }
    for (const v of [-Z, 0, Z]) el(svg, 'text', { x: x(v), y: 272, 'text-anchor': 'middle' }, v === 0 ? 'as expected' : `${v > 0 ? '+' : '−'}${Math.abs(v)}${Math.abs(v) === Z ? '+' : ''}`);
    el(svg, 'text', { x: L + W / 2, y: 290, 'text-anchor': 'middle' }, 'print minus Kalshi’s expectation, in threshold steps');
    const pts = [];
    (R.releases || []).forEach((r, i) => {
      const cx = x(r.z), cy = rowY[r.series] + (hash(i) - .5) * 26;
      if (r.series === 'CPI') pts.push(el(svg, 'circle', { cx, cy, r: 3, fill: '#f5f5f2', stroke: 'none' }));
      else if (r.series === 'Payrolls') pts.push(el(svg, 'rect', { x: cx - 3, y: cy - 3, width: 6, height: 6, fill: '#f5f5f2', stroke: 'none' }));
      else pts.push(el(svg, 'path', { d: `M${cx} ${cy - 4}l4 7h-8z`, fill: '#f5f5f2', stroke: 'none' }));
    });
    reveal(svg, pts);
  })();
})();
