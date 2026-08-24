// CI gate: load every published ONNX variant in REAL Chromium via onnxruntime-web,
// synthesise a line, and check it is finite, audible and fast enough.
//
// Python's onnxruntime and the browser's WASM build do NOT have the same kernels.
// A model that passes the Python asserts can still be unusably slow (or missing a
// kernel entirely) in the browser. This step is what makes "it works in the browser"
// a fact instead of a hope. It also writes the measured winner into manifest.json.
//
//   node tools/browser_check/check.mjs <buildDir> <manifestPath>

import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE       = path.dirname(fileURLToPath(import.meta.url));
const buildDir   = path.resolve(process.argv[2] ?? 'build');
const manifestPath = path.resolve(process.argv[3] ?? 'models-web/manifest.json');
const MAX_RTF    = Number(process.env.QARINA_MAX_RTF ?? 0.5);   // 0.5 = 2x faster than real time
const MIN_RMS    = Number(process.env.QARINA_MIN_RMS ?? 0.005);
const REPS       = Number(process.env.QARINA_REPS ?? 3);

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const webDir   = path.dirname(manifestPath);
const vectors  = JSON.parse(fs.readFileSync(path.join(webDir, 'test_vectors.json'), 'utf8'));
const cfg      = JSON.parse(fs.readFileSync(path.join(webDir, 'config.json'), 'utf8'));
const ids      = vectors.vectors[2].input_ids;            // poora misra — sab se bhaari case

const ortDist = path.join(HERE, 'node_modules', 'onnxruntime-web', 'dist');
const MIME = { '.html':'text/html', '.mjs':'text/javascript', '.js':'text/javascript',
               '.wasm':'application/wasm', '.onnx':'application/octet-stream', '.json':'application/json' };

const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  const file = url.startsWith('/_ort/') ? path.join(ortDist, url.slice(6))
             : url === '/' || url === '/page.html' ? path.join(HERE, 'page.html')
             : path.join(buildDir, url);
  if (url === '/favicon.ico') { res.writeHead(204).end(); return; }
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404).end('nope'); return; }
  res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream',
                       'content-length': fs.statSync(file).size });
  fs.createReadStream(file).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch(
  process.env.QARINA_CHROME ? { executablePath: process.env.QARINA_CHROME } : {});
const page = await browser.newPage();
page.on('console', m => { if (m.type() === 'error') console.log('   [browser]', m.text().slice(0, 400)); });
page.on('pageerror', e => console.log('   [pageerror]', String(e).slice(0, 400)));
await page.goto(`${origin}/page.html`);
await page.waitForFunction(() => window.__ready, null, { timeout: 60_000 });

const bench = { sampling_rate: manifest.sampling_rate, noise_scale: cfg.noise_scale,
                length_scale: cfg.length_scale, noise_scale_w: cfg.noise_scale_w };

console.log(`browser gate — ${ids.length} tokens, best of ${REPS}, max rtf ${MAX_RTF}\n`);
console.log('variant   size MB   load ms   run ms   audio s     rtf      rms   verdict');

const results = {};
let failures = 0;
for (const [name, v] of Object.entries(manifest.variants)) {
  let r;
  try {
    r = await page.evaluate(([f, i, c, n]) => window.bench(f, i, c, n), [v.file, ids, bench, REPS]);
  } catch (e) {
    console.log(`${name.padEnd(9)} ${String(v.size_mb).padStart(7)}   —  BROWSER FAILED: ${String(e).split('\n')[0].slice(0,200)}`);
    failures++; continue;
  }
  const ok = r.finite && r.rms > MIN_RMS;
  const fast = r.rtf <= MAX_RTF;
  if (!ok) failures++;
  console.log(`${name.padEnd(9)} ${String(v.size_mb).padStart(7)} ${String(r.load_ms).padStart(9)} `
            + `${String(r.run_ms).padStart(8)} ${String(r.duration_s).padStart(9)} `
            + `${String(r.rtf).padStart(7)} ${String(r.rms).padStart(8)}   `
            + `${!ok ? 'BROKEN' : fast ? 'ok' : 'too slow'}`);
  results[name] = { ...r, ok, fast };
  Object.assign(manifest.variants[name], { browser_rtf: r.rtf, browser_load_ms: r.load_ms, browser_ok: ok });
}
await browser.close();
server.close();

if (failures) { console.error(`\n${failures} variant(s) do not run correctly in the browser.`); process.exit(1); }

// Default = chhota se chhota variant jo RTF ki had ke andar hai.
// Ye faisla naap kar hota hai, andaza laga kar nahi.
const usable = Object.entries(results).filter(([, r]) => r.ok && r.fast)
  .sort((a, b) => manifest.variants[a[0]].size_mb - manifest.variants[b[0]].size_mb);
if (!usable.length) {
  console.error(`\nNo variant met rtf <= ${MAX_RTF}. Fastest was `
    + Object.entries(results).sort((a,b)=>a[1].rtf-b[1].rtf)[0].join(' @ rtf '));
  process.exit(1);
}
manifest.default = usable[0][0];
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 1) + '\n');
console.log(`\ndefault -> ${manifest.default} `
  + `(${manifest.variants[manifest.default].size_mb} MB, rtf ${results[manifest.default].rtf})`);
