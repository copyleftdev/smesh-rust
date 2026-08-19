/* Shared drawing utilities for the film. Everything here is a pure function of
   time so any frame can be rendered in isolation and in any order. */

const W = 1920, H = 1080;

const PALETTE = {
  ground:  '#0A0C07',
  soil:    '#101408',
  bone:    '#EDEBE0',
  dim:     '#9BA184',
  muted:   '#6F765C',
  myc:     '#7FBFA6',
  signal:  '#E8B04B',
  danger:  '#E0637E',
  latency: '#E5905A',
  errors:  '#E0637E',
  saturation: '#D9B23F',
  traces:  '#56BFA9',
  deploys: '#8B9CE4',
};

const CONCERNS = [
  { id: 'latency',    label: 'response times', color: PALETTE.latency },
  { id: 'errors',     label: 'errors',         color: PALETTE.errors },
  { id: 'saturation', label: 'capacity',       color: PALETTE.saturation },
  { id: 'traces',     label: 'retries',        color: PALETTE.traces },
  { id: 'deploys',    label: 'releases',       color: PALETTE.deploys },
];

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const easeOut = t => 1 - Math.pow(1 - clamp(t), 3);
const easeIn = t => Math.pow(clamp(t), 3);
const easeInOut = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Progress of a cue that starts at `at` seconds and lasts `dur`. */
const cue = (t, at, dur = 1) => clamp((t - at) / dur);

/** A seeded generator, so every render of a frame produces the same picture. */
function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

function hexA(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${clamp(alpha)})`;
}

function text(ctx, str, x, y, o = {}) {
  const {
    size = 32, weight = 400, family = 'Archivo', color = PALETTE.bone,
    align = 'center', baseline = 'alphabetic', alpha = 1, tracking = 0, upper = false,
  } = o;
  if (alpha <= 0.002) return;
  ctx.save();
  ctx.globalAlpha = clamp(alpha);
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px ${family}, sans-serif`;
  ctx.textBaseline = baseline;
  const s = upper ? str.toUpperCase() : str;

  if (!tracking) {
    ctx.textAlign = align;
    ctx.fillText(s, x, y);
  } else {
    // Manual tracking: canvas has no letter-spacing everywhere yet.
    const chars = [...s];
    const widths = chars.map(c => ctx.measureText(c).width);
    const total = widths.reduce((a, b) => a + b, 0) + tracking * (chars.length - 1);
    let cx = align === 'center' ? x - total / 2 : align === 'right' ? x - total : x;
    ctx.textAlign = 'left';
    chars.forEach((c, i) => { ctx.fillText(c, cx, y); cx += widths[i] + tracking; });
  }
  ctx.restore();
}

/** Word-wrapped paragraph; returns the y after the last line. */
function paragraph(ctx, str, x, y, maxWidth, o = {}) {
  const { size = 28, lineHeight = 1.5, align = 'center' } = o;
  ctx.save();
  ctx.font = `${o.weight || 400} ${size}px ${o.family || 'Archivo'}, sans-serif`;
  const words = str.split(' ');
  const lines = [];
  let line = '';
  for (const w of words) {
    const test = line ? line + ' ' + w : w;
    if (ctx.measureText(test).width > maxWidth && line) { lines.push(line); line = w; }
    else line = test;
  }
  if (line) lines.push(line);
  ctx.restore();
  lines.forEach((l, i) => text(ctx, l, x, y + i * size * lineHeight, { ...o, size, align }));
  return y + lines.length * size * lineHeight;
}

/** Reveal a string character by character. */
function typeOn(str, p) {
  const n = Math.floor(clamp(p) * str.length + 0.0001);
  return str.slice(0, n);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Soft vignette so the frame reads as cinema rather than a web page. */
function vignette(ctx, strength = 0.55) {
  const g = ctx.createRadialGradient(W / 2, H / 2, H * 0.25, W / 2, H / 2, H * 0.95);
  g.addColorStop(0, 'rgba(0,0,0,0)');
  g.addColorStop(1, `rgba(0,0,0,${strength})`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
}

/** Very light film grain, seeded per frame index. */
function grain(ctx, frame, amount = 0.022) {
  const r = rng(frame * 2654435761);
  ctx.save();
  ctx.globalAlpha = amount;
  for (let i = 0; i < 900; i++) {
    ctx.fillStyle = r() > 0.5 ? '#ffffff' : '#000000';
    ctx.fillRect(r() * W, r() * H, 2, 2);
  }
  ctx.restore();
}

function fadeToBlack(ctx, a) {
  if (a <= 0) return;
  ctx.fillStyle = `rgba(0,0,0,${clamp(a)})`;
  ctx.fillRect(0, 0, W, H);
}

/** An eyebrow + rule used as the standard scene caption. */
function caption(ctx, label, alpha, y = 120) {
  if (alpha <= 0.002) return;
  text(ctx, label, W / 2, y, {
    size: 19, family: 'IBM Plex Mono', color: PALETTE.myc,
    tracking: 7, upper: true, alpha,
  });
  ctx.save();
  ctx.globalAlpha = clamp(alpha) * 0.4;
  ctx.strokeStyle = PALETTE.myc;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(W / 2 - 130, y + 22);
  ctx.lineTo(W / 2 + 130, y + 22);
  ctx.stroke();
  ctx.restore();
}

/** Draw a node in the house style: soft ring, solid core. */
function node(ctx, x, y, r, color, o = {}) {
  const { alpha = 1, ringAlpha = 0.4, glow = 0, label = null, labelAlpha = 1, sub = null } = o;
  if (alpha <= 0.002) return;
  ctx.save();
  ctx.globalAlpha = clamp(alpha);

  if (glow > 0) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r * 6);
    g.addColorStop(0, hexA(color, 0.34 * glow));
    g.addColorStop(1, hexA(color, 0));
    ctx.fillStyle = g;
    ctx.fillRect(x - r * 6, y - r * 6, r * 12, r * 12);
  }

  ctx.strokeStyle = hexA(color, ringAlpha);
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.arc(x, y, r * 2.6, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();

  if (label) {
    text(ctx, label, x, y + r * 2.6 + 34, {
      size: 21, family: 'IBM Plex Mono', color: PALETTE.bone, alpha: labelAlpha,
    });
  }
  if (sub) {
    text(ctx, sub, x, y + r * 2.6 + 58, {
      size: 16, family: 'IBM Plex Mono', color: PALETTE.muted, alpha: labelAlpha,
    });
  }
  ctx.restore();
}

function link(ctx, a, b, o = {}) {
  const { alpha = 0.3, color = PALETTE.muted, width = 1.4, dash = null } = o;
  if (alpha <= 0.002) return;
  ctx.save();
  ctx.globalAlpha = clamp(alpha);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.restore();
}

function packet(ctx, a, b, p, color, radius = 7) {
  const x = lerp(a.x, b.x, p), y = lerp(a.y, b.y, p);
  ctx.save();
  const g = ctx.createRadialGradient(x, y, 0, x, y, radius * 4);
  g.addColorStop(0, hexA(color, 0.5));
  g.addColorStop(1, hexA(color, 0));
  ctx.fillStyle = g;
  ctx.fillRect(x - radius * 4, y - radius * 4, radius * 8, radius * 8);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/** Points of a regular polygon, used for every mesh layout in the film. */
function ring(cx, cy, r, n, rot = -Math.PI / 2) {
  return Array.from({ length: n }, (_, i) => {
    const a = rot + (i * Math.PI * 2) / n;
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  });
}
