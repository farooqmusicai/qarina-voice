# Qarina neural voice — MMS Urdu checkpoint -> browser ONNX
# Raasta wahi: asal VITS code (jaywalnut310/vits) + SynthesizerTrn.infer + opset 13
#
# BADLA HUA USOOL (24 Aug 2026):
#   Wazan (.onnx) ab git mein NAHI jaate — GitHub Release par jaate hain.
#   git mein sirf chhoti cheezein: manifest.json, vocab.json, config.json,
#   test_vectors.json, README.md, aur ek chhota sample wav.
#   Is liye 100 MB ki had ab koi masla nahi — precision ka faisla
#   sirf "size vs raftaar" par hota hai, git ki had par nahi.
#
# Banata hai:
#   build/       -> model.fp32.onnx, model.q8.onnx, model.hybrid.onnx  (Release ke liye)
#   models-web/  -> manifest.json, vocab.json, config.json, test_vectors.json, README.md
#   samples/     -> neural-test.wav

import hashlib, json, os, sys, time, urllib.request
import numpy as np

VITS_DIR = os.environ.get("VITS_DIR", "vits")
sys.path.insert(0, VITS_DIR)

import torch
from scipy.io import wavfile
import utils
from models import SynthesizerTrn

ISO   = "urd-script_arabic"
BASE  = f"https://huggingface.co/facebook/mms-tts/resolve/main/models/{ISO}"
TMP   = "tmp-mms"
BUILD = "build"        # bhaari wazan — .gitignore mein hai, Release par jaata hai
WEB   = "models-web"   # halki metadata — git mein rehti hai
for d in (TMP, BUILD, WEB, "samples"):
    os.makedirs(d, exist_ok=True)

def mb(p):  return os.path.getsize(p) / 1e6
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------- 1) checkpoint + config + vocab download ----------
for f in ["G_100000.pth", "config.json", "vocab.txt"]:
    p = os.path.join(TMP, f)
    if not os.path.exists(p):
        print("download:", f)
        urllib.request.urlretrieve(f"{BASE}/{f}", p)
print("checkpoint:", round(mb(os.path.join(TMP, "G_100000.pth")), 1), "MB")

hps = utils.get_hparams_from_file(os.path.join(TMP, "config.json"))
if hps.data.training_files.split(".")[-1] == "uroman":
    raise ValueError("uroman model — Urdu arabic-script hona chahiye tha")

with open(os.path.join(TMP, "vocab.txt"), encoding="utf-8") as f:
    symbols = [line.replace("\n", "") for line in f.readlines()]
sym2id = {s: i for i, s in enumerate(symbols)}
print("vocab:", len(symbols), "symbols | sampling_rate:", hps.data.sampling_rate,
      "| add_blank:", hps.data.add_blank)

# ---------- 2) model load ----------
net_g = SynthesizerTrn(len(symbols),
                       hps.data.filter_length // 2 + 1,
                       hps.train.segment_size // hps.data.hop_length,
                       **hps.model)
net_g.cpu().eval()
utils.load_checkpoint(os.path.join(TMP, "G_100000.pth"), net_g, None)

class OnnxModel(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x, x_length, noise_scale, length_scale, noise_scale_w):
        return self.m.infer(x=x, x_lengths=x_length, noise_scale=noise_scale,
                            length_scale=length_scale, noise_scale_w=noise_scale_w)[0]

# ---------- 3) ONNX export (fp32, opset 13, dynamic axes) ----------
fp32 = os.path.join(BUILD, "model.fp32.onnx")
x = torch.randint(1, len(symbols) - 1, (1, 50), dtype=torch.int64)
args = (x, torch.tensor([50], dtype=torch.int64),
        torch.tensor([0.667], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([0.8], dtype=torch.float32))
kw = dict(opset_version=13,
          input_names=["x", "x_length", "noise_scale", "length_scale", "noise_scale_w"],
          output_names=["y"],
          dynamic_axes={"x": {0: "N", 1: "L"}, "x_length": {0: "N"}, "y": {0: "N", 2: "L"}})
try:
    torch.onnx.export(OnnxModel(net_g), args, fp32, dynamo=False, **kw)
except TypeError:                       # purana torch: dynamo kwarg nahi
    torch.onnx.export(OnnxModel(net_g), args, fp32, **kw)
print("fp32 onnx:", round(mb(fp32), 1), "MB")

# ---------- 4) quantize ----------
# Ahem baat: VITS mein har wazan wali layer nn.Conv1d hai — attention ke q/k/v/o
# aur FFN bhi (vits/attentions.py). Sirf ["MatMul","Gemm"] quantize karne se
# kuch bhi quantize NAHI hota (isi liye purana int8, fp32 se bara nikla tha).
#
# Do variant bantay hain, kyunke ConvInteger chhota to karta hai magar WASM mein
# sust hai — CI ka browser gate dono naapta hai aur default chunta hai:
#   q8     = sab Conv int8      -> sab se chhota,  sab se susat
#   hybrid = decoder fp32 rehta -> darmiyana size, tez (decoder hi asal bojh hai)
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

OPS = ["MatMul", "Gemm", "Conv"]
q8  = os.path.join(BUILD, "model.q8.onnx")
quantize_dynamic(fp32, q8, weight_type=QuantType.QInt8, op_types_to_quantize=OPS)
print("q8 onnx:", round(mb(q8), 1), "MB")

graph = onnx.load(fp32, load_external_data=False).graph
dec_nodes = [n.name for n in graph.node if n.op_type == "Conv" and "/dec/" in n.name]
variants = ["fp32", "q8"]
hyb = os.path.join(BUILD, "model.hybrid.onnx")
if dec_nodes:
    quantize_dynamic(fp32, hyb, weight_type=QuantType.QInt8,
                     op_types_to_quantize=OPS, nodes_to_exclude=dec_nodes)
    print(f"hybrid onnx: {mb(hyb):.1f} MB ({len(dec_nodes)} decoder Conv fp32 chhoray gaye)")
    variants.append("hybrid")
else:
    print("WARNING: '/dec/' naam wale Conv nodes nahi milay — hybrid variant skip. "
          "torch ne node naming badli hogi; upar wala filter dekh lein.")

# Yeh assert wohi bug pakarta hai jo pehli martaba hua tha:
# agar quantize ne kuch nahi kiya to q8 ~= fp32 reh jaata hai.
assert mb(q8) < 0.6 * mb(fp32), (
    f"quantize ne kuch nahi kiya: q8 {mb(q8):.1f} MB vs fp32 {mb(fp32):.1f} MB — "
    "op_types_to_quantize mein 'Conv' hai ya nahi, check karein")

# ---------- 5) vocab.json + config.json (browser tokenizer ke liye) ----------
with open(os.path.join(WEB, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(sym2id, f, ensure_ascii=False, indent=1, sort_keys=True)
cfg = {"iso": ISO, "source": "facebook/mms-tts (Meta MMS research, CC-BY-NC 4.0)",
       "sampling_rate": hps.data.sampling_rate, "add_blank": int(hps.data.add_blank),
       "blank_id": 0, "noise_scale": 0.667, "length_scale": 1.0, "noise_scale_w": 0.8,
       "inputs": ["x", "x_length", "noise_scale", "length_scale", "noise_scale_w"],
       "output": "y"}
with open(os.path.join(WEB, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)

# ---------- 6) tokenizer + test vectors ----------
def text_to_ids(t):
    t = t.lower()
    kept    = [c for c in t if c in sym2id]
    dropped = [c for c in t if c not in sym2id]
    ids = [sym2id[c] for c in kept]
    if hps.data.add_blank:
        out = [0] * (2 * len(ids) + 1); out[1::2] = ids; ids = out
    return ids, dropped

LINES = [
    "سلام", "اردو زبان",
    "مَیں چَلتا ہوں تَنہا یَہاں، چَٹّان سا کھَڑا ہوں مَیں",
    "آسمان جھوٹ نہیں بولتا", "عِشق کے دَرد کا بیمار ہوں",
    "دل ناداں تجھے ہوا کیا ہے", "آخر اس درد کی دوا کیا ہے",
    "ہم کو معلوم ہے جنت کی حقیقت لیکن", "کچھ نہ سمجھے خدا کرے کوئی",
    "پتّا پتّا بوٹا بوٹا حال ہمارا جانے ہے", "ایک دو تین چار پانچ",
    "قرینہ آپ کی خدمت میں حاضر ہے",
]
vec, drop_report = [], {}
for t in LINES:
    ids, dropped = text_to_ids(t)
    vec.append({"text": t, "input_ids": ids})
    for c in dropped:
        drop_report[c] = drop_report.get(c, 0) + 1
with open(os.path.join(WEB, "test_vectors.json"), "w", encoding="utf-8") as f:
    json.dump({"iso": ISO, "add_blank": int(hps.data.add_blank), "vectors": vec},
              f, ensure_ascii=False, indent=1)
print("test vectors:", len(vec), "| vocab se bahar gire huroof:", drop_report or "koi nahi")

# ---------- 7) sanity: har variant, do lambaiyan, awaz ki naap + RTF ----------
import onnxruntime as ort
sr   = hps.data.sampling_rate
MIN0 = float(os.environ.get("QARINA_MIN_DUR0", "0.3"))
MIN2 = float(os.environ.get("QARINA_MIN_DUR2", "2.0"))
paths  = {"fp32": fp32, "q8": q8, "hybrid": hyb}
report = {}
for name in variants:
    path = paths[name]
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for line_i in (0, 2):                       # chhota "سلام" aur poora misra
        ids, _ = text_to_ids(LINES[line_i])
        arr = np.array([ids], dtype=np.int64)
        t0 = time.perf_counter()
        y = sess.run(["y"], {"x": arr,
                             "x_length": np.array([arr.shape[1]], dtype=np.int64),
                             "noise_scale":   np.array([0.667], dtype=np.float32),
                             "length_scale":  np.array([1.0],   dtype=np.float32),
                             "noise_scale_w": np.array([0.8],   dtype=np.float32)})[0]
        wall = time.perf_counter() - t0
        wav  = y[0, 0]
        dur  = len(wav) / sr
        rms  = float(np.sqrt((wav.astype(np.float64) ** 2).mean()))
        print(f"{name} line{line_i}: {dur:.2f}s rms={rms:.4f} wall={wall*1000:.0f}ms rtf={wall/dur:.2f}")
        assert dur > (MIN0 if line_i == 0 else MIN2), f"{name}: awaz bohat chhoti — export kharab"
        assert rms > 0.005, f"{name}: awaz khali — export kharab"
        if line_i == 2:
            report[name] = {"file": os.path.basename(path), "size_mb": round(mb(path), 2),
                            "sha256": sha256(path), "duration_s": round(dur, 2),
                            "rms": round(rms, 4), "py_rtf": round(wall / dur, 3)}
            if name == variants[-1]:
                w = wav / max(1e-9, float(np.abs(wav).max())) * 0.9
                wavfile.write("samples/neural-test.wav", sr, (w * 32767).astype(np.int16))
                print("made samples/neural-test.wav")

# ---------- 8) manifest (git mein rehta hai; browser isi se model dhoondta hai) ----------
tag  = os.environ.get("QARINA_RELEASE_TAG", "models-v1")
repo = os.environ.get("GITHUB_REPOSITORY", "farooqmusicai/qarina-voice")
manifest = {
    "schema": 1, "iso": ISO, "release_tag": tag,
    "base_url": f"https://github.com/{repo}/releases/download/{tag}/",
    "sampling_rate": sr, "add_blank": int(hps.data.add_blank),
    "vocab_size": len(symbols),
    "default": None,                     # browser gate RTF naap kar bharta hai
    "variants": report,
}
with open(os.path.join(WEB, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)

with open(os.path.join(WEB, "README.md"), "w", encoding="utf-8") as f:
    f.write("# Qarina neural voice — browser models (MMS Urdu)\n\n")
    f.write(f"Source: `facebook/mms-tts` models/{ISO} (Meta MMS research, **CC-BY-NC 4.0**) — shukriya Meta.\n\n")
    f.write("Wazan git mein nahi hain — Release `%s` par hain. `manifest.json` unhein point karta hai.\n\n" % tag)
    f.write("| variant | size (MB) | duration | rms | python rtf |\n|---|---|---|---|---|\n")
    for n, r in report.items():
        f.write(f"| {r['file']} | {r['size_mb']:.1f} | {r['duration_s']:.2f}s | {r['rms']:.4f} | {r['py_rtf']:.2f} |\n")
    f.write(f"\nvocab: {len(symbols)} symbols · sampling_rate: {sr} Hz · add_blank: {int(hps.data.add_blank)}\n")
    f.write(f"\nvocab se bahar gire huroof (test lines): {json.dumps(drop_report, ensure_ascii=False)}\n")
print("done — build/ (wazan) aur models-web/ (metadata) tayyar")
