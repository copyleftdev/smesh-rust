/* Deterministic frame capture. Renders the film page at explicit times so any
   frame can be produced in isolation, in parallel, and reproducibly. */
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function main() {
  const args = Object.fromEntries(process.argv.slice(2).map(a => {
    const [k, ...v] = a.replace(/^--/, '').split('=');
    return [k, v.join('=')];
  }));

  const timeline = JSON.parse(fs.readFileSync('timeline.json', 'utf8'));
  const outDir = args.out || 'frames';
  const from = Number(args.from ?? 0);
  const to = Number(args.to ?? timeline.total_ms);
  const stride = Number(args.stride ?? 1);
  const quality = Number(args.quality ?? 92);
  const probe = args.probe ? args.probe.split(',').map(Number) : null;

  fs.mkdirSync(outDir, { recursive: true });

  const browser = await chromium.launch({
    args: ['--force-color-profile=srgb', '--disable-lcd-text', '--hide-scrollbars'],
  });
  const page = await browser.newPage({
    viewport: { width: timeline.width, height: timeline.height },
    deviceScaleFactor: 1,
  });

  await page.goto('file://' + path.resolve('film.html'));
  await page.evaluate(t => window.loadTimeline(t), timeline);
  await page.evaluate(() => window.filmReady());

  const frameMs = 1000 / timeline.fps;

  if (probe) {
    for (const ms of probe) {
      const scene = await page.evaluate(m => window.renderAt(m), ms);
      await page.locator('#c').screenshot({ path: path.join(outDir, `probe_${Math.round(ms)}_${scene}.png`) });
      console.log(`probe ${(ms / 1000).toFixed(1)}s -> ${scene}`);
    }
    await browser.close();
    return;
  }

  const first = Math.ceil(from / frameMs);
  const last = Math.floor(to / frameMs);
  let written = 0;

  for (let i = first; i <= last; i += stride) {
    const ms = i * frameMs;
    await page.evaluate(m => window.renderAt(m), ms);
    await page.locator('#c').screenshot({
      path: path.join(outDir, String(i).padStart(6, '0') + '.jpg'),
      type: 'jpeg',
      quality,
    });
    written++;
    if (written % 250 === 0) {
      process.stdout.write(`  ${outDir}: ${written} frames (t=${(ms / 1000).toFixed(1)}s)\n`);
    }
  }

  await browser.close();
  console.log(`${outDir}: wrote ${written} frames [${first}..${last}]`);
}

main().catch(e => { console.error(e); process.exit(1); });
