# Qarina — permanent fix for the 100 MB push failure

## Kya hua tha

Run #1 ka export bilkul theek chala. Sirf aakhri step (`git push`) mara:

```
File models-web/model.int8.onnx is 109.07 MB; this exceeds GitHub's file size limit of 100.00 MB
File models-web/model.fp32.onnx is 108.75 MB; this exceeds GitHub's file size limit of 100.00 MB
! [remote rejected] main -> main (pre-receive hook declined)
```

Aur us error ke neeche ek doosra bug chhupa tha: **int8 file fp32 se BARI thi**
(109.07 vs 108.75 MB). Int8 ko 4 guna chhota hona chahiye tha.

## Asal wajah

`tools/make_onnx.py` mein:

```python
op_types_to_quantize=["MatMul", "Gemm"]
```

VITS (`jaywalnut310/vits`) mein har wazan wali layer `nn.Conv1d` hai — attention ke
q/k/v/o projections aur FFN bhi (`vits/attentions.py`), flows, duration predictor,
aur poora HiFi-GAN decoder. Graph mein jo `MatMul` hain woh attention scores hain —
activation × activation, un mein wazan hai hi nahi. Is liye quantizer ko quantize
karne ko kuch mila hi nahi; usne sirf scale/zero-point initializers add kiye aur
file thori bari ho gayi.

## Naapi hui haqeeqat (VITS jaise synthetic model par, 29.5M params / 118 MB)

| variant | size | % of fp32 | Chromium (onnxruntime-web, 1 thread) |
|---|---|---|---|
| fp32 | 118.10 MB | 100 % | 153 ms |
| int8, saare Conv (`+Conv`) | 37.41 MB | **31.7 %** | 829 ms — **5.4× susat** |
| hybrid (decoder fp32) | 76.72 MB | 65.0 % | 317 ms — 2.1× susat |

Do baatein pakki hui:

1. `"Conv"` add karna waqai kaam karta hai — 108.75 MB ka model **~34 MB** ban jaata hai.
2. `ConvInteger` ka WASM kernel **maujood hai** (browser mein chalta hai, output finite
   aur theek hai) — magar WASM mein woh fp32 Conv se kaafi susat hai. TTS compute-bound
   hai, is liye "sab kuch int8" ka faisla andhaa nahi lena chahiye.

## Faisla

Asal masla yeh tha ke **git ki 100 MB wali had model ke design ka faisla kar rahi thi.**
Woh ghalat constraint hai. Isay hata do:

- **Wazan Release asset bantay hain** (2 GB per file, CORS on, `github.token` kaafi hai,
  koi naya secret nahi). Git mein sirf `models-web/manifest.json` rehta hai jo unhein
  point karta hai — sha256 aur size ke saath.
- Ab precision ka faisla sirf **size vs raftaar** par hota hai. CI teeno variant banata
  hai, **asli Chromium mein RTF naapta hai**, aur sab se chhota variant jo had ke andar
  ho wohi `manifest.default` mein likh deta hai. Andaza nahi — naap.
- `tools/guard_no_big_files.sh` har push/PR par chalta hai (25 MB had). Yeh galti
  dobara ho hi nahi sakti.
- `upload-artifact` `if: always()` par hai — publish fail bhi ho jaaye to 3 minute ki
  mehnat mehfooz rehti hai.

## Files

| file | kya badla |
|---|---|
| `tools/make_onnx.py` | `Conv` quantize hota hai; 3 variant; sha256 manifest; `build/` vs `models-web/` alag; woh assert jo yeh bug pakarta |
| `.github/workflows/make-onnx.yml` | Release publish, artifact backup, browser gate, guard; `checkout@v5`/`setup-python@v6` (Node 20 warning khatam) |
| `.github/workflows/guard-repo-size.yml` | har push/PR par size guard |
| `tools/guard_no_big_files.sh` | 25 MB se bari tracked file par fail |
| `tools/browser_check/` | asli Chromium + onnxruntime-web gate, RTF naapta hai, default chunta hai |
| `web/qarina-model.js` | Cache Storage loader (ek dafa download), progress, sha256 verify, tokenizer |
| `.gitignore` | `build/`, `vits/`, `tmp-mms/`, `models-web/*.onnx` |

## Chalane ka tareeqa

```bash
# 1. files daal do, phir:
git rm -r --cached models-web 2>/dev/null || true   # agar purane onnx staged hain
git add -A && git commit -m "weights to Releases; fix Conv quantization; browser gate"
git push

# 2. Actions -> "Qarina ONNX for Browser" -> Run workflow -> tag: models-v1
```

Browser mein:

```js
import { loadManifest, loadModel, makeTokenizer } from './web/qarina-model.js';
import * as ort from 'onnxruntime-web';

const manifest = await loadManifest();
const buf = await loadModel(manifest, { onProgress: p => console.log((p.ratio*100)|0, '%') });
const session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
```

## Ek aur baat

MMS **CC-BY-NC 4.0** hai — non-commercial. `config.json` aur `README.md` mein yeh
likha hua hai; agar Qarina commercial product banega to model ka license alag
dekhna parega, hosting se is ka koi taalluq nahi.
