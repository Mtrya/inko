import { createRequire } from 'node:module';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const require = createRequire(join(homedir(), '.agents', 'skills', 'archify', 'package.json'));
const puppeteer = require('puppeteer-core');

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/chromium',
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1300, height: 1000, deviceScaleFactor: 2 });
  const html = pathToFileURL(join(here, 'pretrain_architecture.html'));
  await page.goto(`${html.href}?theme=light`, { waitUntil: 'networkidle0' });
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => {
    const svg = document.querySelector('svg');
    const children = [...svg.children];
    const legend = children.find((element) => element.textContent === 'Legend');
    if (legend) {
      for (const element of children.slice(children.indexOf(legend))) element.remove();
    }
    svg.setAttribute('viewBox', '0 0 1480 830');
  });
  const svg = await page.$('svg');
  await svg.screenshot({ path: join(root, 'notes', 'assets', 'pretrain_architecture.png') });
} finally {
  await browser.close();
}
