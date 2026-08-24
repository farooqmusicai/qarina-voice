// Qarina — browser side model loader + tokenizer.
//
// Wazan Release par hain, is liye pehli baar download hota hai aur phir
// Cache Storage mein reh jaata hai. Doosri dafa kholne par network par
// kuch nahi jaata. Cache key mein sha256 hai — naya Release aayega to
// key khud badal jaayegi, purani apne aap saaf ho jaayegi.

const CACHE = 'qarina-models-v1';

/** manifest.json padho (git mein hai, chhota hai, har baar taaza) */
export async function loadManifest(url = './models-web/manifest.json') {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`manifest ${r.status}`);
  return r.json();
}

/**
 * Chuna hua variant laao. Cache mein ho to wahan se, warna Release se.
 * @param {object} manifest  loadManifest() ka nateeja
 * @param {object} [opts]
 * @param {string} [opts.variant]   'fp32' | 'q8' | 'hybrid' — default manifest se
 * @param {(p:{loaded:number,total:number,ratio:number,cached:boolean})=>void} [opts.onProgress]
 * @param {boolean} [opts.verify]   sha256 milao (thora waqt lagta hai, magar yaqeen deta hai)
 * @returns {Promise<ArrayBuffer>}
 */
export async function loadModel(manifest, opts = {}) {
  const name = opts.variant ?? manifest.default;
  const v = manifest.variants?.[name];
  if (!v) throw new Error(`variant "${name}" manifest mein nahi hai`);

  const url = manifest.base_url + v.file;
  const key = `${url}#${v.sha256}`;                 // sha badla = naya cache entry

  const cache = await caches.open(CACHE).catch(() => null);
  if (cache) {
    const hit = await cache.match(key);
    if (hit) {
      opts.onProgress?.({ loaded: v.size_mb * 1e6, total: v.size_mb * 1e6, ratio: 1, cached: true });
      return hit.arrayBuffer();
    }
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`${v.file}: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || Math.round(v.size_mb * 1e6);

  // Stream kar ke progress dikhao — 40 MB par yeh farq karta hai.
  const chunks = [];
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    opts.onProgress?.({ loaded, total, ratio: total ? loaded / total : 0, cached: false });
  }
  const buf = await new Blob(chunks).arrayBuffer();

  if (opts.verify && v.sha256 && crypto?.subtle) {
    const got = [...new Uint8Array(await crypto.subtle.digest('SHA-256', buf))]
      .map(b => b.toString(16).padStart(2, '0')).join('');
    if (got !== v.sha256) throw new Error(`${v.file}: sha256 mismatch — download kharab hai`);
  }

  if (cache) {
    // Purani entries hatao (naya sha aane par purana wazan jagah na ghere)
    for (const k of await cache.keys()) {
      if (k.url.startsWith(manifest.base_url) && k.url !== key) await cache.delete(k);
    }
    await cache.put(key, new Response(buf, {
      headers: { 'content-type': 'application/octet-stream', 'content-length': String(buf.byteLength) },
    })).catch(() => {});                            // quota bhara ho to chalta hai
  }
  return buf;
}

/**
 * Wahi tokenizer jo tools/make_onnx.py mein hai — har harf ka id, beech mein blank(0).
 * models-web/test_vectors.json se milaan karo taake dono taraf ek jaisa rahe.
 */
export function makeTokenizer(vocab, addBlank = true) {
  return function textToIds(text) {
    const kept = [], dropped = [];
    for (const ch of text.toLowerCase()) {
      (ch in vocab ? kept : dropped).push(ch);
    }
    let ids = kept.map(c => vocab[c]);
    if (addBlank) {
      const out = new Array(2 * ids.length + 1).fill(0);
      for (let i = 0; i < ids.length; i++) out[2 * i + 1] = ids[i];
      ids = out;
    }
    return { ids, dropped };
  };
}

/** Float32 audio ko chalne laayak AudioBuffer banao */
export function toAudioBuffer(ctx, samples, sampleRate) {
  const b = ctx.createBuffer(1, samples.length, sampleRate);
  b.getChannelData(0).set(samples);
  return b;
}
