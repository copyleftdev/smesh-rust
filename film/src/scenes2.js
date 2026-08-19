/* Scenes 5-9b: the three primitives, the wire, and the demo setup. */

/* ---------- 5. decay ---------- */

SCENES.primitive_decay = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);
  caption(ctx, 'mechanism one', cue(t, 0.3, 1.0));
  text(ctx, 'Decay', W / 2, 216, { size: 78, weight: 700, alpha: cue(t, 0.6, 1.0) });

  const x0 = 380, x1 = 1540, yBase = 800, hgt = 380;
  const draw = easeOut(cue(t, 1.6, 5.0));

  // Axes.
  ctx.save();
  ctx.globalAlpha = cue(t, 1.2, 1.0) * 0.45;
  ctx.strokeStyle = PALETTE.muted;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, yBase - hgt); ctx.lineTo(x0, yBase); ctx.lineTo(x1, yBase);
  ctx.stroke();
  ctx.restore();
  text(ctx, 'intensity', x0 - 28, yBase - hgt / 2, {
    size: 17, family: 'IBM Plex Mono', color: PALETTE.muted, align: 'right', alpha: cue(t, 1.4, 1),
  });
  text(ctx, 'time', (x0 + x1) / 2, yBase + 42, {
    size: 17, family: 'IBM Plex Mono', color: PALETTE.muted, alpha: cue(t, 1.4, 1),
  });

  const decayAt = u => Math.exp(-3.1 * u);

  // The curve, and the area under it.
  ctx.save();
  ctx.globalAlpha = 0.09 * draw;
  ctx.fillStyle = PALETTE.signal;
  ctx.beginPath();
  ctx.moveTo(x0, yBase);
  for (let u = 0; u <= draw; u += 0.004) ctx.lineTo(lerp(x0, x1, u), yBase - decayAt(u) * hgt);
  ctx.lineTo(lerp(x0, x1, draw), yBase);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = PALETTE.signal;
  ctx.lineWidth = 3;
  ctx.globalAlpha = 0.95;
  ctx.beginPath();
  for (let u = 0; u <= draw; u += 0.004) {
    const px = lerp(x0, x1, u), py = yBase - decayAt(u) * hgt;
    u === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.restore();

  // The signal riding its own decay.
  if (draw > 0 && draw < 1) {
    const px = lerp(x0, x1, draw), py = yBase - decayAt(draw) * hgt;
    node(ctx, px, py, 4 + 9 * decayAt(draw), PALETTE.signal, { glow: decayAt(draw) });
    text(ctx, `${Math.round(decayAt(draw) * 100)}%`, px, py - 46, {
      size: 22, family: 'IBM Plex Mono', color: PALETTE.signal,
    });
  }

  // Where it stops mattering.
  const thr = cue(t, 7.2, 1.0);
  if (thr > 0) {
    ctx.save();
    ctx.globalAlpha = thr * 0.55;
    ctx.strokeStyle = PALETTE.danger;
    ctx.setLineDash([7, 7]);
    ctx.beginPath();
    ctx.moveTo(x0, yBase - 0.1 * hgt);
    ctx.lineTo(x1, yBase - 0.1 * hgt);
    ctx.stroke();
    ctx.restore();
    text(ctx, 'below this, the signal is gone', x1, yBase - 0.1 * hgt - 22, {
      size: 19, family: 'IBM Plex Mono', color: PALETTE.danger, align: 'right', alpha: thr,
    });
  }

  paragraph(ctx, 'Nothing has to clean up. Stale work removes itself.',
    W / 2, 962, 1200, { size: 38, weight: 600, alpha: cue(t, 11.5, 1.2), color: PALETTE.bone });

  vignette(ctx, 0.55);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 0.7));
};

/* ---------- 6. reinforcement ---------- */

SCENES.primitive_reinforce = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);
  caption(ctx, 'mechanism two', cue(t, 0.3, 1.0));
  text(ctx, 'Reinforcement', W / 2, 216, { size: 78, weight: 700, alpha: cue(t, 0.6, 1.0) });

  const claim = { x: W / 2, y: 596 };
  const agents = ring(W / 2, 596, 300, 5);
  const arriveAt = [2.6, 4.6, 6.6, 8.4, 10.0];
  let witnesses = 0;

  agents.forEach((a, i) => {
    const arrived = cue(t, arriveAt[i], 1.1);
    if (arrived > 0.98) witnesses++;
    const c = CONCERNS[i].color;
    node(ctx, a.x, a.y, 10, c, { alpha: 0.25 + 0.75 * arrived, glow: 0.5 * arrived });
    link(ctx, a, claim, { alpha: 0.5 * arrived, color: c, width: 1.6 });
    const travel = cue(t, arriveAt[i], 1.0);
    if (travel > 0 && travel < 1) packet(ctx, a, claim, easeInOut(travel), c, 7);
  });

  const conf = clamp(witnesses / 5);
  node(ctx, claim.x, claim.y, 16 + 20 * conf, PALETTE.myc, { alpha: 1, glow: 0.5 + conf * 1.6 });
  text(ctx, `${witnesses}`, claim.x, claim.y + 12, {
    size: 40, weight: 700, family: 'IBM Plex Mono', color: PALETTE.ground,
  });
  text(ctx, witnesses === 1 ? 'witness' : 'witnesses', claim.x, claim.y + 116, {
    size: 22, family: 'IBM Plex Mono', color: PALETTE.myc, tracking: 5, upper: true, alpha: cue(t, 2.6, 1),
  });

  // Confidence bar.
  const bx = 660, bw = 600, by = 856;
  ctx.save();
  ctx.globalAlpha = cue(t, 2.2, 1.0);
  ctx.strokeStyle = hexA(PALETTE.muted, 0.5);
  ctx.lineWidth = 1;
  roundRect(ctx, bx, by, bw, 16, 8);
  ctx.stroke();
  ctx.fillStyle = PALETTE.myc;
  roundRect(ctx, bx, by, Math.max(6, bw * conf), 16, 8);
  ctx.fill();
  ctx.restore();
  text(ctx, 'confidence', bx - 22, by + 13, {
    size: 18, family: 'IBM Plex Mono', color: PALETTE.muted, align: 'right', alpha: cue(t, 2.2, 1),
  });

  paragraph(ctx, 'Not a duplicate. A second witness.',
    W / 2, 962, 1100, { size: 38, weight: 600, alpha: cue(t, 8.0, 1.2), color: PALETTE.bone });

  vignette(ctx, 0.55);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 0.7));
};

/* ---------- 7. content addressing ---------- */

SCENES.primitive_address = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);
  caption(ctx, 'mechanism three', cue(t, 0.3, 1.0));
  text(ctx, 'Addressed by content', W / 2, 216, { size: 74, weight: 700, alpha: cue(t, 0.6, 1.0) });

  const converge = easeInOut(cue(t, 6.4, 2.4));
  const midX = W / 2;
  const lx = lerp(520, midX, converge), rx = lerp(1400, midX, converge);
  const y = 560;

  const side = (x, i, evidence, at) => {
    const a = cue(t, at, 1.0);
    if (a <= 0.002) return;
    const c = CONCERNS[i].color;
    node(ctx, x, y, 12, c, { alpha: a * (1 - converge * 0.35), glow: 0.6 * a });
    text(ctx, CONCERNS[i].id, x, y + 62, {
      size: 22, family: 'IBM Plex Mono', color: c, alpha: a * (1 - converge),
    });
    text(ctx, evidence, x, y + 94, {
      size: 18, family: 'IBM Plex Mono', color: PALETTE.muted, alpha: a * (1 - converge),
    });
  };
  side(lx, 0, 'latency up 3x', 1.2);
  side(rx, 2, 'pool at 98%', 2.4);

  // Different evidence, identical claim, therefore identical address.
  const claimA = cue(t, 3.6, 1.2);
  const boxW = 460, boxH = 96;
  [[lx, 0], [rx, 2]].forEach(([x, i]) => {
    const a = claimA * (1 - converge * 0.2);
    if (a <= 0.002) return;
    ctx.save();
    ctx.globalAlpha = a;
    ctx.strokeStyle = hexA(CONCERNS[i].color, 0.6);
    ctx.lineWidth = 1.4;
    roundRect(ctx, x - boxW / 2, y + 140, boxW, boxH, 5);
    ctx.stroke();
    ctx.restore();
    text(ctx, '{"subject":"checkout-api",', x, y + 178, {
      size: 20, family: 'IBM Plex Mono', color: PALETTE.bone, alpha: a,
    });
    text(ctx, '"claim":"degraded"}', x, y + 208, {
      size: 20, family: 'IBM Plex Mono', color: PALETTE.bone, alpha: a,
    });
  });

  const hashA = cue(t, 5.2, 1.2);
  if (hashA > 0.002) {
    [[lx], [rx]].forEach(([x]) => {
      text(ctx, 'sha-256', x, y + 286, {
        size: 16, family: 'IBM Plex Mono', color: PALETTE.muted, tracking: 4, upper: true, alpha: hashA * (1 - converge),
      });
      text(ctx, '10757c4a01affa2d', x, y + 324, {
        size: 30, family: 'IBM Plex Mono', color: PALETTE.myc, alpha: hashA,
      });
    });
  }

  if (converge > 0.85) {
    const b = cue(converge, 0.85, 0.15);
    node(ctx, midX, y, 22, PALETTE.myc, { alpha: b, glow: 1.8 * b });
    text(ctx, 'one claim, two witnesses', midX, y + 62, {
      size: 26, family: 'IBM Plex Mono', color: PALETTE.myc, tracking: 4, upper: true, alpha: b,
    });
  }

  paragraph(ctx, 'Same conclusion, same address. They find each other without ever talking.',
    W / 2, 972, 1320, { size: 36, weight: 600, alpha: cue(t, 12.6, 1.2), color: PALETTE.bone });

  vignette(ctx, 0.55);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 0.7));
};

/* ---------- 8. the wire ---------- */

SCENES.quic = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);
  caption(ctx, 'the wire', cue(t, 0.3, 1.0));
  text(ctx, 'QUIC', W / 2, 224, { size: 84, weight: 700, alpha: cue(t, 0.5, 1.0), tracking: 6 });
  text(ctx, 'encrypted, peer to peer, no broker', W / 2, 274, {
    size: 21, family: 'IBM Plex Mono', color: PALETTE.myc, tracking: 5, upper: true, alpha: cue(t, 1.2, 1.0),
  });

  const pts = ring(W / 2, 630, 260, 5);
  const LINKS = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [4, 0]];

  // The broker that is not there.
  const ghost = (1 - cue(t, 8.0, 2.0)) * cue(t, 3.0, 1.2);
  if (ghost > 0.004) {
    ctx.save();
    ctx.globalAlpha = ghost * 0.4;
    ctx.setLineDash([6, 8]);
    ctx.strokeStyle = PALETTE.danger;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(W / 2, 630, 46, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    text(ctx, 'no broker', W / 2, 638, {
      size: 21, family: 'IBM Plex Mono', color: PALETTE.danger, alpha: ghost * 0.85,
    });
  }

  LINKS.forEach(([i, j], k) => {
    const a = cue(t, 1.6 + k * 0.28, 0.9);
    link(ctx, pts[i], pts[j], { alpha: a * 0.5, color: PALETTE.myc, width: 1.8 });
    if (a > 0.9) {
      const mid = { x: (pts[i].x + pts[j].x) / 2, y: (pts[i].y + pts[j].y) / 2 };
      text(ctx, 'TLS 1.3', mid.x, mid.y - 10, {
        size: 14, family: 'IBM Plex Mono', color: PALETTE.myc, alpha: cue(t, 4.4, 1.2) * 0.7,
      });
    }
  });

  pts.forEach((p, i) => node(ctx, p.x, p.y, 11, CONCERNS[i].color, {
    alpha: cue(t, 1.2 + i * 0.18, 0.9), glow: 0.6, label: CONCERNS[i].id,
    labelAlpha: cue(t, 2.0 + i * 0.18, 0.9),
  }));

  LINKS.forEach(([i, j], k) => {
    const p = ((t * 0.34 + k * 0.17) % 1);
    const fwd = k % 2 === 0;
    packet(ctx, pts[fwd ? i : j], pts[fwd ? j : i], p, PALETTE.signal, 5 * cue(t, 3.0, 1));
  });

  paragraph(ctx, 'Nothing in the middle to buy, to scale, or to lose.',
    W / 2, 992, 1200, { size: 38, weight: 600, alpha: cue(t, 13.0, 1.2), color: PALETTE.bone });

  vignette(ctx, 0.55);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 0.7));
};

/* ---------- 9. the setup ---------- */

const SERVICES = ['edge-gateway', 'checkout-api', 'payments-api', 'inventory-svc', 'session-store', 'notification-worker'];

SCENES.setup = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);
  caption(ctx, 'the run you are about to see', cue(t, 0.3, 1.0));

  // The fleet under observation.
  const svcY = 268;
  text(ctx, 'one fleet of services', W / 2, 208, {
    size: 22, family: 'IBM Plex Mono', color: PALETTE.dim, tracking: 4, upper: true, alpha: cue(t, 0.8, 1),
  });
  SERVICES.forEach((s, i) => {
    const a = cue(t, 1.2 + i * 0.14, 0.8);
    const x = 250 + i * 285;
    ctx.save();
    ctx.globalAlpha = a * 0.85;
    ctx.strokeStyle = hexA(PALETTE.muted, 0.5);
    roundRect(ctx, x - 128, svcY, 256, 52, 4);
    ctx.stroke();
    ctx.restore();
    text(ctx, s, x, svcY + 33, { size: 19, family: 'IBM Plex Mono', color: PALETTE.dim, alpha: a });
  });

  // Five processes, five ports, five blind spots.
  const agentY = 640;
  text(ctx, 'five separate programs', W / 2, 470, {
    size: 22, family: 'IBM Plex Mono', color: PALETTE.myc, tracking: 4, upper: true, alpha: cue(t, 4.0, 1),
  });
  CONCERNS.forEach((c, i) => {
    const a = cue(t, 4.6 + i * 0.5, 0.9);
    const x = 288 + i * 336;
    node(ctx, x, agentY, 13, c.color, { alpha: a, glow: 0.7 * a });
    text(ctx, c.id, x, agentY + 66, { size: 24, family: 'IBM Plex Mono', color: c.color, alpha: a });
    text(ctx, `watches ${c.label}`, x, agentY + 98, {
      size: 18, family: 'IBM Plex Mono', color: PALETTE.muted, alpha: a,
    });
    text(ctx, `127.0.0.1:930${i + 1}`, x, agentY + 128, {
      size: 16, family: 'IBM Plex Mono', color: PALETTE.muted, alpha: a * cue(t, 8.0, 1.2) * 0.75,
    });

    // Each one is walled off from the others.
    const wall = cue(t, 11.5, 1.6);
    if (wall > 0.004 && i < CONCERNS.length - 1) {
      ctx.save();
      ctx.globalAlpha = wall * 0.4;
      ctx.strokeStyle = PALETTE.danger;
      ctx.setLineDash([5, 9]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x + 168, agentY - 76);
      ctx.lineTo(x + 168, agentY + 146);
      ctx.stroke();
      ctx.restore();
    }
  });

  const blind = cue(t, 13.0, 1.3);
  paragraph(ctx, 'None of them can see what the others see.',
    W / 2, 900, 1200, { size: 40, weight: 600, alpha: blind, color: PALETTE.danger });

  const warn = cue(t, 17.5, 1.3);
  paragraph(ctx, 'And something is about to go wrong.',
    W / 2, 968, 1200, { size: 36, weight: 500, alpha: warn, color: PALETTE.bone });

  vignette(ctx, 0.55);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 0.7));
};

/* ---------- 9b. five windows ---------- */

SCENES.windows = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);

  const bx = 460, by = 250, bw = 1000, bh = 620;
  const a = cue(t, 0.2, 1.2);

  ctx.save();
  ctx.globalAlpha = a * 0.9;
  ctx.fillStyle = '#0E1109';
  ctx.fillRect(bx, by, bw, bh);
  ctx.strokeStyle = hexA(PALETTE.muted, 0.45);
  ctx.lineWidth = 1.4;
  ctx.strokeRect(bx, by, bw, bh);
  ctx.restore();

  // The fire nobody can see.
  const fire = cue(t, 1.0, 2.0);
  ctx.save();
  ctx.globalAlpha = fire * (0.5 + 0.5 * Math.sin(t * 5));
  const g = ctx.createRadialGradient(bx + bw * 0.62, by + bh * 0.66, 0, bx + bw * 0.62, by + bh * 0.66, 300);
  g.addColorStop(0, hexA(PALETTE.danger, 0.55));
  g.addColorStop(1, hexA(PALETTE.danger, 0));
  ctx.fillStyle = g;
  ctx.fillRect(bx, by, bw, bh);
  ctx.restore();

  // Five windows, each showing only smoke.
  CONCERNS.forEach((c, i) => {
    const wa = cue(t, 1.4 + i * 0.4, 0.8);
    const wx = bx + 90 + i * 176, wy = by + 190;
    ctx.save();
    ctx.globalAlpha = wa;
    ctx.fillStyle = hexA(c.color, 0.13);
    ctx.fillRect(wx, wy, 120, 168);
    ctx.strokeStyle = hexA(c.color, 0.75);
    ctx.lineWidth = 1.6;
    ctx.strokeRect(wx, wy, 120, 168);
    ctx.restore();
    text(ctx, c.id, wx + 60, wy + 200, {
      size: 17, family: 'IBM Plex Mono', color: c.color, alpha: wa,
    });
    const smoke = cue(t, 4.6 + i * 0.2, 1.2);
    text(ctx, 'smoke', wx + 60, wy + 92, {
      size: 19, family: 'IBM Plex Mono', color: PALETTE.dim, alpha: smoke * (0.55 + 0.45 * Math.sin(t * 2 + i)),
    });
  });

  text(ctx, 'None of them can see the fire.', W / 2, 972, {
    size: 44, weight: 600, color: PALETTE.bone, alpha: cue(t, 7.0, 1.2),
  });
  text(ctx, 'Every one of them sees smoke.', W / 2, 1030, {
    size: 32, weight: 500, color: PALETTE.danger, alpha: cue(t, 9.0, 1.2),
  });

  vignette(ctx, 0.6);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 0.7));
  fadeToBlack(ctx, cue(t, seg.dur - 1.2, 1.2));
};
