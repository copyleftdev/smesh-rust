const { chromium } = require('playwright');
const path = require('path');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await page.goto('file://' + path.resolve('..', 'five-concerns.html'));
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
  await page.waitForFunction(() => document.querySelectorAll('.claim').length > 0);
  await page.evaluate(() => document.fonts.ready);

  const geo = await page.evaluate(() => {
    const r = el => { const b = el.getBoundingClientRect();
      return { x: Math.round(b.x + scrollX), y: Math.round(b.y + scrollY),
               w: Math.round(b.width), h: Math.round(b.height) }; };
    const panels = [...document.querySelectorAll('.panel')];
    return {
      docHeight: document.documentElement.scrollHeight,
      wrap: r(document.querySelector('.wrap')),
      header: r(document.querySelector('header')),
      verdicts: r(document.querySelector('.verdicts')),
      transport: r(panels[0]),
      graph: r(document.querySelector('#graph')),
      meshPanel: r(panels[1]),
      claimsPanel: r(panels[2]),
      claims: [...document.querySelectorAll('.claim')].map(c => ({
        subject: c.querySelector('.claim-subject').textContent, ...r(c) })),
      journal: r(panels[3]),
      duration: (() => { const s = document.getElementById('scrub'); return s.max; })(),
    };
  });
  console.log(JSON.stringify(geo, null, 1));
  await browser.close();
})();
