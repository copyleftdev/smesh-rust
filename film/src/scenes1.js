/* Scenes 1-4: the cold open, the forest mechanism, the reveal, the problem. */

/* ---------- shared organic growth ---------- */

/** Build a branching root system once; draw it progressively by time. */
function buildRoots(seed, ox, oy, angle, depth, len, spread) {
  const r = rng(seed);
  const segs = [];
  (function grow(x, y, ang, d, l, t0) {
    if (d <= 0) return;
    const nx = x + Math.cos(ang) * l;
    const ny = y + Math.sin(ang) * l;
    const dur = 0.10 + r() * 0.06;
    segs.push({ x1: x, y1: y, x2: nx, y2: ny, t0, t1: t0 + dur, w: d * 0.85, depth: d });
    const branches = d > 2 ? (r() > 0.35 ? 2 : 3) : 2;
    for (let i = 0; i < branches; i++) {
      const spreadAmt = (r() - 0.5) * spread;
      grow(nx, ny, ang + spreadAmt, d - 1, l * (0.66 + r() * 0.2), t0 + dur * (0.7 + r() * 0.3));
    }
  })(ox, oy, angle, depth, len, 0);
  return segs;
}

const ROOTS_L = buildRoots(1337, 640, 512, Math.PI / 2 + 0.34, 7, 116, 1.18);
const ROOTS_R = buildRoots(9021, 1290, 512, Math.PI / 2 - 0.34, 7, 116, 1.18);

/** The tip nearest the centre line, so the two systems join where they end. */
function innerTip(segs, towardX) {
  let best = null, bestScore = Infinity;
  for (const g of segs) {
    if (g.depth > 2) continue;
    const score = Math.abs(g.x2 - towardX) - g.y2 * 0.55;
    if (score < bestScore) { bestScore = score; best = g; }
  }
  return best ? { x: best.x2, y: best.y2, t: best.t1 } : { x: towardX, y: 800, t: 0.8 };
}
const TIP_L = innerTip(ROOTS_L, 980);
const TIP_R = innerTip(ROOTS_R, 980);

function drawRoots(ctx, segs, p, color, alpha) {
  ctx.save();
  ctx.lineCap = 'round';
  for (const s of segs) {
    const local = clamp((p - s.t0) / (s.t1 - s.t0));
    if (local <= 0) continue;
    ctx.globalAlpha = clamp(alpha * (0.32 + s.depth / 9));
    ctx.strokeStyle = color;
    ctx.lineWidth = s.w;
    ctx.beginPath();
    ctx.moveTo(s.x1, s.y1);
    ctx.lineTo(lerp(s.x1, s.x2, local), lerp(s.y1, s.y2, local));
    ctx.stroke();
  }
  ctx.restore();
}

/** Drifting spores; ambience that makes the soil feel alive. */
function spores(ctx, t, alpha, count = 70) {
  const r = rng(4242);
  ctx.save();
  for (let i = 0; i < count; i++) {
    const bx = r() * W, by = 420 + r() * 620, sp = 0.25 + r() * 0.6, ph = r() * 100;
    const x = (bx + t * sp * 14) % W;
    const y = by + Math.sin(t * 0.5 + ph) * 16;
    const tw = 0.28 + 0.72 * (0.5 + 0.5 * Math.sin(t * 1.3 + ph));
    ctx.globalAlpha = clamp(alpha * tw * 0.5);
    ctx.fillStyle = PALETTE.myc;
    ctx.beginPath();
    ctx.arc(x, y, 1.5 + r() * 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

/** A low, cold sky. Without it the tree silhouettes have nothing to sit against. */
function sky(ctx, horizon, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = clamp(alpha);
  const g = ctx.createLinearGradient(0, 0, 0, horizon);
  g.addColorStop(0, '#07090C');
  g.addColorStop(0.62, '#0C1014');
  g.addColorStop(1, '#18201C');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, horizon);
  ctx.restore();
}

function soilGround(ctx, horizon, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  const g = ctx.createLinearGradient(0, horizon - 120, 0, H);
  g.addColorStop(0, '#0A0C07');
  g.addColorStop(0.35, '#0E1209');
  g.addColorStop(1, '#05060A');
  ctx.fillStyle = g;
  ctx.fillRect(0, horizon - 120, W, H - horizon + 120);
  ctx.globalAlpha = alpha * 0.5;
  ctx.strokeStyle = hexA(PALETTE.myc, 0.18);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, horizon);
  ctx.lineTo(W, horizon);
  ctx.stroke();
  ctx.restore();
}

/** A simple, believable tree silhouette. */
function tree(ctx, x, groundY, scale, seed, alpha) {
  const r = rng(seed);
  ctx.save();
  ctx.globalAlpha = clamp(alpha);
  ctx.strokeStyle = '#03040A';
  ctx.fillStyle = '#03040A';
  ctx.lineCap = 'round';
  (function branch(bx, by, ang, len, wdt, d) {
    if (d === 0) return;
    const nx = bx + Math.cos(ang) * len, ny = by + Math.sin(ang) * len;
    ctx.lineWidth = wdt;
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(nx, ny);
    ctx.stroke();
    const n = d > 3 ? 2 : 3;
    for (let i = 0; i < n; i++) {
      branch(nx, ny, ang + (r() - 0.5) * 0.95, len * (0.68 + r() * 0.16), wdt * 0.66, d - 1);
    }
  })(x, groundY, -Math.PI / 2, 92 * scale, 15 * scale, 6);
  ctx.restore();
}

/* ---------- 1. cold open ---------- */

SCENES.roots = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);

  const p = cue(t, 0.6, 9.0);
  // Deliberately no trees here. The cold open is the hidden half of the
  // forest; showing the canopy would give away the reveal in the next scene.
  sky(ctx, 512, cue(t, 0.2, 2.5) * 0.8);
  soilGround(ctx, 512, cue(t, 0.2, 2.5));
  spores(ctx, t, cue(t, 1.5, 3) * 0.9);

  drawRoots(ctx, ROOTS_L, p, PALETTE.myc, 0.82);
  drawRoots(ctx, ROOTS_R, p, PALETTE.myc, 0.82);

  // The two systems find each other and a signal crosses.
  const joinP = cue(t, 9.4, 1.8);
  if (joinP > 0) {
    const a = TIP_L, b = TIP_R;
    ctx.save();
    ctx.globalAlpha = joinP * 0.85;
    ctx.strokeStyle = PALETTE.myc;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    const sagX = (a.x + b.x) / 2, sagY = Math.max(a.y, b.y) + 78;
    ctx.quadraticCurveTo(sagX, sagY, lerp(a.x, b.x, easeOut(joinP)), lerp(a.y, b.y, easeOut(joinP)));
    ctx.stroke();
    ctx.restore();
    node(ctx, a.x, a.y, 5, PALETTE.myc, { alpha: joinP, glow: 0.8 });
    node(ctx, b.x, b.y, 5, PALETTE.myc, { alpha: joinP, glow: 0.8 });

    const pulseP = cue(t, 11.4, 2.6);
    if (pulseP > 0 && pulseP < 1) packet(ctx, a, b, easeInOut(pulseP), PALETTE.signal, 9);
  }

  text(ctx, typeOn('Under every forest,', cue(t, 3.4, 1.5)), W / 2, 232, {
    size: 60, weight: 600, alpha: cue(t, 3.4, 0.8),
  });
  text(ctx, typeOn('there is a second network.', cue(t, 5.0, 1.8)), W / 2, 306, {
    size: 60, weight: 600, alpha: cue(t, 5.0, 0.8), color: PALETTE.myc,
  });
  text(ctx, 'no tree is in charge of it', W / 2, 386, {
    size: 21, family: 'IBM Plex Mono', color: PALETTE.muted,
    tracking: 6, upper: true, alpha: cue(t, 12.8, 1.6) * 0.9,
  });

  vignette(ctx, 0.6);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 1.6));
  fadeToBlack(ctx, cue(t, seg.dur - 1.0, 1.0) * 0.55);
};

/* ---------- 2. the forest mechanism ---------- */

const FOREST_TREES = [
  { x: 300, s: 1.00, seed: 11 }, { x: 700, s: 1.22, seed: 22 },
  { x: 1150, s: 0.94, seed: 33 }, { x: 1560, s: 1.12, seed: 44 },
];
const FOREST_ROOTS = FOREST_TREES.map(tr => ({ x: tr.x, y: 700 }));

SCENES.forest = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);
  sky(ctx, 620, 1);
  soilGround(ctx, 620, 1);
  spores(ctx, t + 20, 0.65);

  const intro = cue(t, 0, 1.6);
  FOREST_TREES.forEach((tr, i) => tree(ctx, tr.x, 620, tr.s, tr.seed, intro * cue(t, i * 0.16, 1)));

  // The underground network, drawn as gentle catenaries.
  for (let i = 0; i < FOREST_ROOTS.length - 1; i++) {
    const a = FOREST_ROOTS[i], b = FOREST_ROOTS[i + 1];
    ctx.save();
    ctx.globalAlpha = cue(t, 1.2 + i * 0.25, 1.2) * 0.55;
    ctx.strokeStyle = PALETTE.myc;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.quadraticCurveTo((a.x + b.x) / 2, a.y + 130, b.x, b.y);
    ctx.stroke();
    ctx.restore();
  }
  FOREST_ROOTS.forEach((n, i) =>
    node(ctx, n.x, n.y, 8, PALETTE.myc, { alpha: cue(t, 1.0 + i * 0.2, 1), glow: 0.5 }));

  const beat = (label, body, at, colour) => {
    const a = cue(t, at, 0.9) * (1 - cue(t, at + 7.2, 1.0));
    if (a <= 0.002) return;
    text(ctx, label, W / 2, 176, {
      size: 20, family: 'IBM Plex Mono', color: colour, tracking: 8, upper: true, alpha: a,
    });
    paragraph(ctx, body, W / 2, 236, 1180, { size: 42, weight: 500, alpha: a, lineHeight: 1.35 });
  };

  beat('one', 'A tree in trouble releases a signal into the network.', 2.2, PALETTE.signal);
  beat('two', 'The signal fades as it travels, and fades as time passes.', 10.4, PALETTE.signal);
  beat('three', 'When a second tree senses the same threat, the two reinforce each other.', 18.6, PALETTE.myc);

  // Beat one: a pulse leaves the second tree.
  const p1 = cue(t, 3.6, 3.4);
  if (p1 > 0 && p1 < 1) packet(ctx, FOREST_ROOTS[1], FOREST_ROOTS[2], easeInOut(p1), PALETTE.signal, 10);

  // Beat two: the same pulse, visibly weakening as it goes.
  const p2 = cue(t, 12.0, 4.2);
  if (p2 > 0 && p2 < 1) {
    const a = FOREST_ROOTS[1], b = FOREST_ROOTS[3];
    const x = lerp(a.x, b.x, p2), y = lerp(a.y, b.y, p2);
    const strength = 1 - p2;
    ctx.save();
    ctx.globalAlpha = strength;
    packet(ctx, a, b, p2, PALETTE.signal, 3 + 8 * strength);
    ctx.restore();
    text(ctx, `${Math.round(strength * 100)}%`, x, y - 40, {
      size: 19, family: 'IBM Plex Mono', color: PALETTE.signal, alpha: strength * 0.9,
    });
  }

  // Beat three: two sources converge and the merged signal is brighter.
  const p3 = cue(t, 20.0, 3.6);
  if (p3 > 0 && p3 < 1) {
    const mid = { x: 925, y: 762 };
    packet(ctx, FOREST_ROOTS[1], mid, easeInOut(clamp(p3 * 1.6)), PALETTE.signal, 9);
    packet(ctx, FOREST_ROOTS[2], mid, easeInOut(clamp(p3 * 1.6)), PALETTE.signal, 9);
    if (p3 > 0.62) {
      const b = cue(p3, 0.62, 0.22);
      node(ctx, mid.x, mid.y, 10 + 12 * b, PALETTE.myc, { alpha: 1, glow: 1.6 * b });
      text(ctx, 'reinforced', mid.x, mid.y + 92, {
        size: 22, family: 'IBM Plex Mono', color: PALETTE.myc, tracking: 5, upper: true, alpha: b,
      });
    }
  }

  const outro = cue(t, seg.dur - 5.6, 1.4);
  if (outro > 0) {
    paragraph(ctx, 'Coordination is not something the forest does. It is something the forest grows.',
      W / 2, 902, 1280, { size: 36, weight: 500, alpha: outro * (1 - cue(t, seg.dur - 0.9, 0.9)), color: PALETTE.bone });
  }

  vignette(ctx, 0.62);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 1.0));
};

/* ---------- 3. the reveal ---------- */

SCENES.reveal = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);

  // The organic layout resolves into a geometric one.
  const morph = easeInOut(cue(t, 0.3, 2.6));
  const organic = FOREST_ROOTS.map(n => ({ x: n.x, y: n.y - 40 }));
  const geo = ring(W / 2, 640, 250, 4, -Math.PI / 2);
  const pts = organic.map((o, i) => ({
    x: lerp(o.x, geo[i].x, morph),
    y: lerp(o.y, geo[i].y, morph),
  }));

  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      link(ctx, pts[i], pts[j], { alpha: 0.16 + 0.3 * morph, color: PALETTE.myc, width: 1.3 });
    }
  }
  pts.forEach(p => node(ctx, p.x, p.y, 9, PALETTE.myc, { glow: 0.7 }));

  const packP = (t * 0.42) % 1;
  packet(ctx, pts[0], pts[2], packP, PALETTE.signal, 6 * morph);
  packet(ctx, pts[1], pts[3], (packP + 0.5) % 1, PALETTE.signal, 6 * morph);

  const titleA = cue(t, 2.4, 1.2);
  text(ctx, 'SMESH', W / 2, 300, {
    size: 168, weight: 700, alpha: titleA, tracking: 22 * (1 - easeOut(titleA)) + 12,
  });
  text(ctx, 'signal diffusion for distributed agents', W / 2, 372, {
    size: 25, family: 'IBM Plex Mono', color: PALETTE.myc,
    tracking: 6, upper: true, alpha: cue(t, 3.6, 1.2),
  });

  paragraph(ctx, 'Software agents that coordinate the way a forest does.',
    W / 2, 972, 1200, { size: 34, weight: 500, alpha: cue(t, 6.4, 1.2), color: PALETTE.dim });

  vignette(ctx, 0.55);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 0.9));
};

/* ---------- 4. the problem ---------- */

SCENES.problem = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);
  caption(ctx, 'the thing in the middle', cue(t, 0.4, 1.2));

  const leftC = { x: 520, y: 600 }, rightC = { x: 1400, y: 600 };
  const spokes = ring(leftC.x, leftC.y, 210, 7);
  const meshPts = ring(rightC.x, rightC.y, 210, 7);

  const appear = cue(t, 1.0, 1.4);
  const failP = cue(t, 13.5, 1.1);      // the hub dies
  const deadP = cue(t, 15.0, 1.0);      // spokes go dark
  const reroute = cue(t, 20.0, 1.6);    // the mesh routes around

  // Hub and spoke.
  spokes.forEach((s, i) => {
    link(ctx, leftC, s, { alpha: appear * (0.42 - 0.34 * deadP), color: PALETTE.muted });
    node(ctx, s.x, s.y, 8, PALETTE.dim, { alpha: appear * (1 - 0.72 * deadP), ringAlpha: 0.22 });
    const pp = ((t * 0.55 + i * 0.14) % 1);
    if (failP < 0.5) packet(ctx, s, leftC, pp, hexA(PALETTE.dim, 1), 4 * appear);
  });
  const hubColor = failP > 0 ? PALETTE.danger : PALETTE.signal;
  node(ctx, leftC.x, leftC.y, 19 + 5 * Math.sin(t * 3) * (failP > 0 ? 1 : 0), hubColor, {
    alpha: appear, glow: 0.9 + failP, ringAlpha: 0.5,
  });
  text(ctx, 'coordinator', leftC.x, leftC.y + 96, {
    size: 21, family: 'IBM Plex Mono', color: failP > 0 ? PALETTE.danger : PALETTE.dim, alpha: appear,
  });
  if (deadP > 0) {
    text(ctx, 'everything stops', leftC.x, leftC.y + 300, {
      size: 30, weight: 600, color: PALETTE.danger, alpha: deadP,
    });
  }

  // The mesh, which has no middle to lose.
  const appear2 = cue(t, 2.4, 1.4);
  meshPts.forEach((a, i) => {
    meshPts.forEach((b, j) => {
      if (j <= i) return;
      const adjacent = Math.abs(i - j) === 1 || Math.abs(i - j) === meshPts.length - 1 || (i + 3) % meshPts.length === j;
      if (!adjacent) return;
      const lost = reroute > 0 && (i === 2 || j === 2);
      link(ctx, a, b, { alpha: appear2 * (lost ? 0.08 : 0.38 + 0.25 * reroute), color: PALETTE.myc });
    });
  });
  meshPts.forEach((p, i) => {
    const lost = reroute > 0 && i === 2;
    node(ctx, p.x, p.y, 9, lost ? PALETTE.danger : PALETTE.myc, {
      alpha: appear2 * (lost ? 0.35 : 1), glow: lost ? 0.2 : 0.55,
    });
  });
  if (reroute > 0) {
    const rp = (t * 0.5) % 1;
    packet(ctx, meshPts[1], meshPts[3], rp, PALETTE.signal, 6);
    text(ctx, 'the signal routes around it', rightC.x, rightC.y + 300, {
      size: 30, weight: 600, color: PALETTE.myc, alpha: reroute,
    });
  }
  text(ctx, 'mesh', rightC.x, rightC.y + 96, {
    size: 21, family: 'IBM Plex Mono', color: PALETTE.dim, alpha: appear2,
  });

  const cost = cue(t, 6.6, 1.2) * (1 - cue(t, 12.6, 1.0));
  if (cost > 0.002) {
    paragraph(ctx, 'The thing you pay for. The thing you scale. The thing that pages you at 3am.',
      W / 2, 918, 1300, { size: 34, weight: 500, alpha: cost, color: PALETTE.dim });
  }

  vignette(ctx, 0.6);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 0.8));
};
