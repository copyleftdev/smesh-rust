/* Capture the demo section by driving the real replay page.
   The camera is a CSS transform, so type re-rasterises at every focal length
   instead of being upscaled. Replay position is set explicitly per frame, so
   the capture is deterministic and independent of wall-clock playback. */
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const OUT_W = 1920, OUT_H = 1080;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const easeInOut = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const lerpRect = (a, b, t) => ({
  x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t),
  w: lerp(a.w, b.w, t), h: lerp(a.h, b.h, t),
});

/* Framings in wrap-local coordinates: the wrap is pinned to the origin, so
   x is the page coordinate minus the 260px the layout used to centre it.
   Every rect is 16:9 so nothing is ever distorted. */
const SHOT = {
  wide:      { x: -96, y: 0,    w: 1580, h: 889 },
  wideLow:   { x: -46, y: 88,   w: 1492, h: 839 },
  mesh:      { x: -60, y: 596,  w: 890,  h: 501 },
  meshWide:  { x: -130, y: 512, w: 1100, h: 619 },
  claims3:   { x: 690, y: 566,  w: 720,  h: 405 },
  claimTop:  { x: 700, y: 528,  w: 700,  h: 394 },
  claimOne:  { x: 712, y: 552,  w: 668,  h: 376 },
  claimsLow: { x: 690, y: 858,  w: 720,  h: 405 },
  journal:   { x: 40,  y: 1150, w: 1290, h: 726 },
};

/* film-time window -> replay-time window, camera path, and selection. */
/* Replay windows are matched to where the run is actually busy. The events are
   bursty -- almost everything decisive happens between 16.3s and 18.6s -- so
   the dense moments run in heavy slow motion and the dead air is skipped. */
const SHOTS = [
  { seg: 's10_incident',  replay: [0.0, 14.0],   from: 'wide',     to: 'wideLow',   select: null },
  { seg: 's11_mesh',      replay: [14.0, 16.4],  from: 'meshWide', to: 'mesh',      select: null },
  { seg: 's12_claims',    replay: [16.3, 16.9],  from: 'claims3',  to: 'claimTop',  select: null },
  { seg: 's13_consensus', replay: [16.9, 18.6],  from: 'claimTop', to: 'claimOne',  select: 'checkout-api' },
  { seg: 's14_decoys',    replay: [18.6, 30.0],  from: 'claims3',  to: 'claimsLow', select: null },
  { seg: 's15_evidence',  replay: [30.0, 34.2],  from: 'journal',  to: 'journal',   select: null },
];

async function main() {
  const args = Object.fromEntries(process.argv.slice(2).map(a => {
    const [k, ...v] = a.replace(/^--/, '').split('=');
    return [k, v.join('=')];
  }));

  const timeline = JSON.parse(fs.readFileSync('timeline.json', 'utf8'));
  const outDir = args.out || 'frames';
  const probe = args.probe ? args.probe.split(',').map(Number) : null;
  const only = args.only || null;
  fs.mkdirSync(outDir, { recursive: true });

  const byId = Object.fromEntries(timeline.segments.map(s => [s.id, s]));
  const shots = SHOTS.map(s => ({ ...s, ...byId[s.seg] })).filter(s => !only || s.seg === only);

  const browser = await chromium.launch({ args: ['--force-color-profile=srgb', '--hide-scrollbars'] });
  const page = await browser.newPage({
    viewport: { width: OUT_W, height: OUT_H },
    deviceScaleFactor: 1,
  });
  await page.goto('file://' + path.resolve('..', 'five-concerns.html'));
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
  await page.waitForFunction(() => document.querySelectorAll('.claim').length > 0);
  await page.evaluate(() => document.fonts.ready);

  // Take the page off its own clock and prepare it to be moved as a camera.
  await page.evaluate(() => {
    document.body.style.overflow = 'hidden';
    // The wrap is taken out of flow below, so the body collapses; paint the
    // ground on the root too or the frame letterboxes to black.
    document.documentElement.style.background = '#12140F';
    document.body.style.background = '#12140F';
    const wrap = document.querySelector('.wrap');
    wrap.style.transformOrigin = '0 0';
    wrap.style.willChange = 'transform';
    // Pin the wrap so page coordinates are stable under the camera. An
    // absolutely positioned block shrinks to fit, so the width must be stated
    // explicitly or the whole grid collapses to its narrow layout.
    wrap.style.position = 'absolute';
    wrap.style.left = '0px';
    wrap.style.top = '0px';
    wrap.style.margin = '0';
    wrap.style.width = '1400px';
    wrap.style.maxWidth = 'none';

    window.__setReplay = (seconds, total) => {
      const s = document.getElementById('scrub');
      s.value = String(Math.round((seconds / total) * 1000));
      s.dispatchEvent(new Event('input'));
    };
    window.__setCamera = (r) => {
      const k = 1920 / r.w;
      document.querySelector('.wrap').style.transform =
        `translate(${-r.x * k}px, ${-r.y * k}px) scale(${k})`;
    };
    window.__select = (subject) => {
      const cards = [...document.querySelectorAll('.claim')];
      for (const c of cards) {
        const on = c.getAttribute('aria-pressed') === 'true';
        const want = subject && c.querySelector('.claim-subject').textContent === subject;
        if (on !== !!want) c.click();
      }
    };
  });

  const runSeconds = 34.204;
  const frameMs = 1000 / timeline.fps;
  let written = 0;

  for (const shot of shots) {
    const first = Math.ceil(shot.start_ms / frameMs);
    const last = Math.floor((shot.end_ms - 1) / frameMs);
    const a = SHOT[shot.from], b = SHOT[shot.to];

    await page.evaluate(s => window.__select(s), shot.select);

    const indices = probe
      ? probe.filter(ms => ms >= shot.start_ms && ms < shot.end_ms).map(ms => Math.round(ms / frameMs))
      : Array.from({ length: last - first + 1 }, (_, i) => first + i);

    for (const i of indices) {
      const ms = i * frameMs;
      const u = clamp((ms - shot.start_ms) / (shot.end_ms - shot.start_ms), 0, 1);
      const eased = easeInOut(u);
      const replay = lerp(shot.replay[0], shot.replay[1], u);

      await page.evaluate(([r, t, total]) => {
        window.__setReplay(t, total);
        window.__setCamera(r);
      }, [lerpRect(a, b, eased), replay, runSeconds]);

      await page.screenshot({
        path: path.join(outDir, String(i).padStart(6, '0') + (probe ? `_${shot.seg}.png` : '.jpg')),
        type: probe ? 'png' : 'jpeg',
        quality: probe ? undefined : 92,
      });
      written++;
      if (!probe && written % 250 === 0) {
        process.stdout.write(`  demo: ${written} frames (film ${(ms / 1000).toFixed(1)}s, replay ${replay.toFixed(1)}s)\n`);
      }
    }
    if (!probe) console.log(`  shot ${shot.seg} done (${last - first + 1} frames)`);
  }

  await browser.close();
  console.log(`demo: wrote ${written} frames`);
}

main().catch(e => { console.error(e); process.exit(1); });
