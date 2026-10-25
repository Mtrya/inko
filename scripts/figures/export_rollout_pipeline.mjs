import { createRequire } from 'node:module';
import { mkdir } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const { values } = parseArgs({
  options: {
    output: { type: 'string', default: join(root, 'notes/assets/rollout_pipeline.png') },
    browser: { type: 'string', default: '/usr/bin/chromium' },
    'archify-dir': { type: 'string', default: join(homedir(), '.agents/skills/archify') },
    help: { type: 'boolean', default: false },
  },
});
if (values.help) {
  console.log('Export the rollout pipeline diagram to PNG.\n\n'
    + 'Usage: node scripts/figures/export_rollout_pipeline.mjs '
    + '[--output FILE] [--browser CHROMIUM] [--archify-dir DIR]');
  process.exit(0);
}

const require = createRequire(join(resolve(values['archify-dir']), 'package.json'));
const puppeteer = require('puppeteer-core');
await mkdir(dirname(resolve(values.output)), { recursive: true });
const browser = await puppeteer.launch({
  executablePath: values.browser,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--force-color-profile=srgb'],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1000, height: 800, deviceScaleFactor: 3 });
  const html = pathToFileURL(join(here, 'rollout_pipeline.html'));
  await page.goto(`${html.href}?theme=light`, { waitUntil: 'networkidle0' });
  await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({ content: '.toolbar{display:none!important}' });
  const svg = await page.$('svg');
  await svg.screenshot({ path: values.output });
  console.log(values.output);
} finally {
  await browser.close();
}
