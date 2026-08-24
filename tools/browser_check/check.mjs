// CI gate: load every published ONNX variant in REAL Chromium via onnxruntime-web,
// synthesise a line, and check it is finite, audible, and how fast it actually is.
//
// Python's onnxruntime and the browser's WASM build do NOT have the same kernels.
// A model that passes the Python asserts can still be broken or unusably slow in the
// browser. This step is what makes "it works in the browser" a fact instead of a hope.
//
// Do baatein alag hain, aur inhein alag hi rakha gaya hai:
//   1. TOOTA HUA hai? (load fail / NaN / khali awaz)  -> build FAIL. Yeh correctness hai.
//   2. Kitna tez hai?                                  -> naapo, manifest mein likho,
//      aur agar target se sust ho to WARN karo — fail nahi.
// Kyun: CI runner 2-4 shared vCPU par chalta hai. Us par naapa hua RTF asli user ke
// laptop/phone ka nateeja nahi hai. Raftaar ko hard gate banane ka matlab tha ke ek
// bilkul theek model sirf is liye reject ho jaata ke GitHub ka runner sust tha.
//
//   node tools/browser_check/check.mjs <buildDir> <manifestPath>

import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE         = path.dirname(fileURLToPath(import.meta.url));
const buildDir     = path.resolve(process.argv[2] ?? 'build');
const manifestPath = path.resolve(process.argv[3] ?? 'models-web/manifest.json');
const TARGET_RTF   = Number(process.env.QARINA_TARGET_RTF ?? 1.0);   // CI budget, UX target nahi
const MIN_RMS      = Number(process.env.QARINA_MIN_RMS ?? 0.005);
const REPS         = Number(process.env.QARINA_REPS ?? 3);
const THREADS      = Number(process.env.QARINA_THREADS ?? Math.min(4, os.cpus().length));

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
  if (url === '/favicon.ico') { res.writeHead(204).end(); return; }
  const file = url.startsWith('/_ort/') ? path.join(ortDist, url.slice(6))
             : url === '/' || url === '/page.html' ? path.join(HERE, 'page.html')
             : path.join(buildDir, url);
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404).end('nope'); return; }
  res.writeHead(200, {
    'content-type': MIME[path.extname(file)] ?? 'application/octet-stream',
    'content-length': fs.statSync(file).size,
    // Yeh do headers SharedArrayBuffer khol detay hain -> ORT ko threads miltay hain.
    // Asli site par bhi yehi do headers bhejain, warna browser 1 thread par chalega.
    'cross-origin-opener-policy': 'same-origin',
    'cross-origin-embedder-policy': 'require-corp',
    'cross-origin-resource-policy': 'same-origin',
  });
  fs.createReadStream(file).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch(
  process.env.QARINA_CHROME ? { executablePath: process.env.QARINA_CHROME } : {});
const page = await browser.newPage();
page.on('console',   m => { if (m.type() === 'error') console.log('   [browser]', m.text().slice(0, 400)); });
page.on('pageerror', e => console.log('   [pageerror]', String(e).slice(0, 400)));
await page.goto(`${origin}/page.html?threads=${THREADS}`);
await page.waitForFunction(() => window.__ready, null, { timeout: 60_000 });

const env = await page.evaluate(() => window.__env);
const bench = { sampling_rate: manifest.sampling_rate, noise_scale: cfg.noise_scale,
                length_scale: cfg.length_scale, noise_scale_w: cfg.noise_scale_w };

console.log(`browser gate — ${ids.length} tokens, best of ${REPS}, target rtf ${TARGET_RTF}`);
console.log(`wasm threads: ${env.threads} (isolated=${env.isolated}, cores=${env.hardwareConcurrency})\n`);
console.log('variant   size MB   load ms   run ms   audio s     rtf      rms   verdict');

const results = {};
let broken = 0;
for (const [name, v] of Object.entries(manifest.variants)) {
  let r;
  try {
    r = await page.evaluate(([f, i, c, n]) => window.bench(f, i, c, n), [v.file, ids, bench, REPS]);
  } catch (e) {
    console.log(`${name.padEnd(9)} ${String(v.size_mb).padStart(7)}   —  BROWSER FAILED: ${String(e).split('\n')[0].slice(0,200)}`);
    broken++; continue;
  }
  const ok   = r.finite && r.rms > MIN_RMS;      // correctness — isi par build rukti hai
  const fast = r.rtf <= TARGET_RTF;              // raftaar — sirf ittila
  if (!ok) broken++;
  console.log(`${name.padEnd(9)} ${String(v.size_mb).padStart(7)} ${String(r.load_ms).padStart(9)} `
            + `${String(r.run_ms).padStart(8)} ${String(r.duration_s).padStart(9)} `
            + `${String(r.rtf).padStart(7)} ${String(r.rms).padStart(8)}   `
            + `${!ok ? 'BROKEN' : fast ? 'ok' : 'slow (ci)'}`);
  results[name] = { ...r, ok, fast };
  Object.assign(manifest.variants[name],
    { browser_rtf: r.rtf, browser_load_ms: r.load_ms, browser_ok: ok, browser_threads: env.threads });
}
await browser.close();
server.close();

if (broken) {
  console.error(`\n${broken} variant(s) do not run correctly in the browser.`);
  process.exit(1);                                // <- sirf tootne par fail
}

const working = Object.entries(results).filter(([, r]) => r.ok);
if (!working.length) { console.error('\nNo working variant.'); process.exit(1); }

// Default chunne ka usool: agar koi variant CI budget ke andar hai to un mein sab se
// chhota. Warna sab se TEZ (kyunke tab bandish raftaar hai, size nahi).
const withinBudget = working.filter(([, r]) => r.fast)
  .sort((a, b) => manifest.variants[a[0]].size_mb - manifest.variants[b[0]].size_mb);
const fastest = [...working].sort((a, b) => a[1].rtf - b[1].rtf);

const [pick, pickR] = withinBudget[0] ?? fastest[0];
manifest.default  = pick;
manifest.gate     = { target_rtf: TARGET_RTF, threads: env.threads, within_budget: !!withinBudget.length };
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 1) + '\n');

if (!withinBudget.length) {
  console.log(`::warning::Koi variant CI par rtf <= ${TARGET_RTF} tak nahi pohancha `
    + `(sab se tez: ${fastest[0][0]} @ rtf ${fastest[0][1].rtf}). CI runner 2-4 shared vCPU par hai; `
    + `asli users ke devices tez hain. Model theek hai — sirf raftaar note kar li gayi.`);
}
console.log(`\ndefault -> ${pick} (${manifest.variants[pick].size_mb} MB, rtf ${pickR.rtf})`);
