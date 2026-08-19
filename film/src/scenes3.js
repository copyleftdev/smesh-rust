/* Scenes 16-17b: the payoff, the business case, and the close. */

SCENES.payoff = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);
  caption(ctx, 'what just happened', cue(t, 0.4, 1.2));

  const pts = ring(W / 2, 520, 232, 5);
  const LINKS = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [4, 0]];
  const appear = cue(t, 0.8, 1.2);

  LINKS.forEach(([i, j]) => link(ctx, pts[i], pts[j], { alpha: appear * 0.35, color: PALETTE.myc }));
  pts.forEach((p, i) => {
    const lit = cue(t, 2.0 + i * 0.34, 0.7);
    node(ctx, p.x, p.y, 11, CONCERNS[i].color, {
      alpha: appear, glow: 0.4 + lit * 1.1, label: CONCERNS[i].id, labelAlpha: appear * 0.9,
    });
    if (lit > 0.5) {
      const mid = { x: W / 2, y: 520 };
      link(ctx, p, mid, { alpha: (lit - 0.5) * 1.4 * 0.5, color: CONCERNS[i].color, width: 1.5 });
    }
  });

  const conv = cue(t, 4.4, 1.0);
  if (conv > 0.004) {
    node(ctx, W / 2, 520, 20 + 14 * conv, PALETTE.myc, { alpha: conv, glow: 2.0 * conv });
    text(ctx, 'checkout-api', W / 2, 520 + 8, {
      size: 22, weight: 700, family: 'IBM Plex Mono', color: PALETTE.ground, alpha: conv,
    });
  }

  const lines = [
    ['Not one agent had enough information to be right.', 6.2, PALETTE.bone, 40],
    ['The system was right anyway.', 8.4, PALETTE.myc, 46],
  ];
  lines.forEach(([s, at, colour, size], i) => {
    text(ctx, s, W / 2, 852 + i * 74, {
      size, weight: 600, color: colour, alpha: cue(t, at, 1.2),
    });
  });

  const honest = cue(t, 14.5, 1.2);
  if (honest > 0.004) {
    text(ctx, 'telemetry: synthetic and reproducible   ·   coordination: real processes, real sockets',
      W / 2, 1012, { size: 20, family: 'IBM Plex Mono', color: PALETTE.muted, alpha: honest });
  }

  vignette(ctx, 0.55);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 0.8));
};

SCENES.unlocks = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);
  caption(ctx, 'why it is worth building', cue(t, 0.3, 1.0));

  const items = [
    ['Add agents freely', 'no reconfiguration, no registry to update', 1.6],
    ['Lose agents safely', 'failure does not take the answer with it', 5.0],
    ['No coordinator bill', 'nothing central to scale or pay for', 8.4],
  ];

  items.forEach(([head, sub, at], i) => {
    const a = cue(t, at, 1.1);
    if (a <= 0.004) return;
    const y = 330 + i * 190;
    const slide = (1 - easeOut(a)) * 44;

    ctx.save();
    ctx.globalAlpha = a * 0.65;
    ctx.strokeStyle = PALETTE.myc;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(452 - slide, y - 40);
    ctx.lineTo(452 - slide, y + 42);
    ctx.stroke();
    ctx.restore();

    text(ctx, head, 492 - slide, y, { size: 50, weight: 600, align: 'left', alpha: a });
    text(ctx, sub, 492 - slide, y + 42, {
      size: 24, family: 'IBM Plex Mono', color: PALETTE.muted, align: 'left', alpha: a * 0.9,
    });
  });

  const kicker = cue(t, 13.5, 1.3);
  paragraph(ctx, 'The network gets more reliable as it gets larger.',
    W / 2, 946, 1300, { size: 38, weight: 600, alpha: kicker, color: PALETTE.myc });

  vignette(ctx, 0.55);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 0.7));
};

SCENES.close = (ctx, t, seg, frame) => {
  ctx.fillStyle = PALETTE.ground;
  ctx.fillRect(0, 0, W, H);

  // The mesh dissolves back into the root system it came from.
  const dissolve = easeInOut(cue(t, 5.0, 5.0));
  const geo = ring(W / 2, 600, 250, 5);
  const organic = [
    { x: 470, y: 760 }, { x: 760, y: 690 }, { x: 1020, y: 800 },
    { x: 1320, y: 700 }, { x: 1520, y: 790 },
  ];
  const pts = geo.map((g, i) => ({
    x: lerp(g.x, organic[i].x, dissolve),
    y: lerp(g.y, organic[i].y, dissolve),
  }));

  const LINKS = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [4, 0]];
  LINKS.forEach(([i, j]) => {
    ctx.save();
    ctx.globalAlpha = 0.4 * (1 - cue(t, 12.5, 3.0));
    ctx.strokeStyle = PALETTE.myc;
    ctx.lineWidth = lerp(1.6, 2.6, dissolve);
    ctx.beginPath();
    ctx.moveTo(pts[i].x, pts[i].y);
    const cxm = (pts[i].x + pts[j].x) / 2, cym = (pts[i].y + pts[j].y) / 2 + 110 * dissolve;
    ctx.quadraticCurveTo(cxm, cym, pts[j].x, pts[j].y);
    ctx.stroke();
    ctx.restore();
  });
  pts.forEach((p, i) => node(ctx, p.x, p.y, 9, dissolve > 0.5 ? PALETTE.myc : CONCERNS[i].color, {
    alpha: 1 - cue(t, 12.5, 3.0), glow: 0.6,
  }));
  spores(ctx, t + 60, dissolve * 0.7);

  text(ctx, 'SMESH', W / 2, 300, {
    size: 132, weight: 700, alpha: cue(t, 0.6, 1.4) * (1 - cue(t, 16.0, 2.5)), tracking: 14,
  });

  paragraph(ctx, 'The hard problem is no longer how clever each agent is. It is how they agree.',
    W / 2, 402, 1280, { size: 34, weight: 500, color: PALETTE.dim, alpha: cue(t, 2.4, 1.4) * (1 - cue(t, 16.0, 2.5)) });

  text(ctx, 'running under our feet for four hundred million years', W / 2, 1000, {
    size: 24, family: 'IBM Plex Mono', color: PALETTE.myc, tracking: 4, upper: true,
    alpha: cue(t, 9.5, 1.6) * (1 - cue(t, 17.0, 2.0)),
  });

  vignette(ctx, 0.62);
  grain(ctx, frame);
  fadeToBlack(ctx, 1 - cue(t, 0, 1.0));
  fadeToBlack(ctx, cue(t, seg.dur - 3.4, 3.2));
};
