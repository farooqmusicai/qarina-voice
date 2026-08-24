# Qarina neural voice — qadam 2: MMS Urdu checkpoint → browser ONNX
# (GitHub Actions par chalta hai; checkpoint wahan HuggingFace se download hota hai)
# Raasta wahi jo sherpa-onnx/willwade ke sab MMS models ne liya:
#   asal VITS code (jaywalnut310/vits) + SynthesizerTrn.infer + opset 13
# 24 Aug 2026: chhote random VITS par yehi export+quantize+run raasta
#   len 7/19/63 teeno par kamyab janch liya gaya tha (dynamic axes sahih).
# Banata hai: models-web/{model.fp32.onnx, model.int8.onnx, vocab.json,
#   config.json, test_vectors.json, README.md} + samples/neural-int8-test.wav
import json, os, sys, urllib.request
import numpy as np

VITS_DIR = os.environ.get("VITS_DIR", "vits")   # workflow clone karta hai
sys.path.insert(0, VITS_DIR)

import torch
from scipy.io import wavfile
import utils                      # vits/utils.py
from models import SynthesizerTrn

ISO = "urd-script_arabic"
BASE = f"https://huggingface.co/facebook/mms-tts/resolve/main/models/{ISO}"
TMP = "tmp-mms"
OUT = "models-web"
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ---------- 1) checkpoint + config + vocab download ----------
for f in ["G_100000.pth", "config.json", "vocab.txt"]:
    p = os.path.join(TMP, f)
    if not os.path.exists(p):
        print("download:", f)
        urllib.request.urlretrieve(f"{BASE}/{f}", p)
print("checkpoint:", round(os.path.getsize(os.path.join(TMP, "G_100000.pth")) / 1e6, 1), "MB")

hps = utils.get_hparams_from_file(os.path.join(TMP, "config.json"))
if hps.data.training_files.split(".")[-1] == "uroman":
    raise ValueError("uroman model — Urdu arabic-script hona chahiye tha")

with open(os.path.join(TMP, "vocab.txt"), encoding="utf-8") as f:
    symbols = [line.replace("\n", "") for line in f.readlines()]
sym2id = {s: i for i, s in enumerate(symbols)}
print("vocab:", len(symbols), "symbols | sampling_rate:", hps.data.sampling_rate,
      "| add_blank:", hps.data.add_blank)

# ---------- 2) model load ----------
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
net_g.cpu().eval()
utils.load_checkpoint(os.path.join(TMP, "G_100000.pth"), net_g, None)

class OnnxModel(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x, x_length, noise_scale, length_scale, noise_scale_w):
        return self.m.infer(x=x, x_lengths=x_length, noise_scale=noise_scale,
                            length_scale=length_scale, noise_scale_w=noise_scale_w)[0]

# ---------- 3) ONNX export (fp32, opset 13, dynamic axes) ----------
fp32 = os.path.join(OUT, "model.fp32.onnx")
x = torch.randint(1, len(symbols) - 1, (1, 50), dtype=torch.int64)
args = (x, torch.tensor([50], dtype=torch.int64),
        torch.tensor([0.667], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([0.8], dtype=torch.float32))
try:
    torch.onnx.export(OnnxModel(net_g), args, fp32, opset_version=13,
        input_names=["x", "x_length", "noise_scale", "length_scale", "noise_scale_w"],
        output_names=["y"],
        dynamic_axes={"x": {0: "N", 1: "L"}, "x_length": {0: "N"}, "y": {0: "N", 2: "L"}},
        dynamo=False)
except TypeError:  # purana torch: dynamo kwarg nahi
    torch.onnx.export(OnnxModel(net_g), args, fp32, opset_version=13,
        input_names=["x", "x_length", "noise_scale", "length_scale", "noise_scale_w"],
        output_names=["y"],
        dynamic_axes={"x": {0: "N", 1: "L"}, "x_length": {0: "N"}, "y": {0: "N", 2: "L"}})
print("fp32 onnx:", round(os.path.getsize(fp32) / 1e6, 1), "MB")

# ---------- 4) int8 quantize (sirf MatMul/Gemm — awaz mehfooz) ----------
from onnxruntime.quantization import quantize_dynamic, QuantType
int8 = os.path.join(OUT, "model.int8.onnx")
quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8,
                 op_types_to_quantize=["MatMul", "Gemm"])
print("int8 onnx:", round(os.path.getsize(int8) / 1e6, 1), "MB")

# ---------- 5) vocab.json + config.json (browser tokenizer ke liye) ----------
with open(os.path.join(OUT, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(sym2id, f, ensure_ascii=False, indent=1, sort_keys=True)
cfg = {"iso": ISO, "source": "facebook/mms-tts (Meta MMS research, CC-BY-NC 4.0)",
       "sampling_rate": hps.data.sampling_rate, "add_blank": int(hps.data.add_blank),
       "blank_id": 0, "noise_scale": 0.667, "length_scale": 1.0, "noise_scale_w": 0.8,
       "inputs": ["x", "x_length", "noise_scale", "length_scale", "noise_scale_w"],
       "output": "y"}
with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)

# ---------- 6) tokenizer (wahi jo browser JS karega) + test vectors ----------
def text_to_ids(t):
    t = t.lower()
    kept = [c for c in t if c in sym2id]
    dropped = [c for c in t if c not in sym2id]
    ids = [sym2id[c] for c in kept]
    if hps.data.add_blank:
        out = [0] * (2 * len(ids) + 1)
        out[1::2] = ids
        ids = out
    return ids, dropped

LINES = [
    "سلام",
    "اردو زبان",
    "مَیں چَلتا ہوں تَنہا یَہاں، چَٹّان سا کھَڑا ہوں مَیں",
    "آسمان جھوٹ نہیں بولتا",
    "عِشق کے دَرد کا بیمار ہوں",
    "دل ناداں تجھے ہوا کیا ہے",
    "آخر اس درد کی دوا کیا ہے",
    "ہم کو معلوم ہے جنت کی حقیقت لیکن",
    "کچھ نہ سمجھے خدا کرے کوئی",
    "پتّا پتّا بوٹا بوٹا حال ہمارا جانے ہے",
    "ایک دو تین چار پانچ",
    "قرینہ آپ کی خدمت میں حاضر ہے",
]
vec, drop_report = [], {}
for t in LINES:
    ids, dropped = text_to_ids(t)
    vec.append({"text": t, "input_ids": ids})
    for c in dropped:
        drop_report[c] = drop_report.get(c, 0) + 1
with open(os.path.join(OUT, "test_vectors.json"), "w", encoding="utf-8") as f:
    json.dump({"iso": ISO, "add_blank": int(hps.data.add_blank), "vectors": vec},
              f, ensure_ascii=False, indent=1)
print("test vectors:", len(vec), "| vocab se bahar gire huroof:", drop_report or "koi nahi")

# ---------- 7) sanity: fp32+int8, do lambaiyan, awaz ki naap ----------
import onnxruntime as ort
sr = hps.data.sampling_rate
# (janch ke jaali chhote model ke liye QARINA_MIN_DUR0/2 se naram ho sakta hai)
MIN0 = float(os.environ.get("QARINA_MIN_DUR0", "0.3"))
MIN2 = float(os.environ.get("QARINA_MIN_DUR2", "2.0"))
report = []
for name, path in [("fp32", fp32), ("int8", int8)]:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for line_i in (0, 2):  # chhota "سلام" aur poora misra — dynamic axes ki janch
        ids, _ = text_to_ids(LINES[line_i])
        arr = np.array([ids], dtype=np.int64)
        y = sess.run(["y"], {"x": arr, "x_length": np.array([arr.shape[1]], dtype=np.int64),
                             "noise_scale": np.array([0.667], dtype=np.float32),
                             "length_scale": np.array([1.0], dtype=np.float32),
                             "noise_scale_w": np.array([0.8], dtype=np.float32)})[0]
        wav = y[0, 0]
        dur = len(wav) / sr
        rms = float(np.sqrt((wav.astype(np.float64) ** 2).mean()))
        print(f"{name} line{line_i}: {dur:.2f}s rms={rms:.4f}")
        assert dur > (MIN0 if line_i == 0 else MIN2), f"{name}: awaz bohat chhoti — export kharab"
        assert rms > 0.005, f"{name}: awaz khali — export kharab"
        if line_i == 2:
            report.append((name, os.path.getsize(path) / 1e6, dur, rms))
    if name == "int8":
        w = wav / max(1e-9, float(np.abs(wav).max())) * 0.9
        os.makedirs("samples", exist_ok=True)
        wavfile.write("samples/neural-int8-test.wav", sr, (w * 32767).astype(np.int16))
        print("made samples/neural-int8-test.wav")

# ---------- 8) README (saboot ke saath) ----------
with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
    f.write("# Qarina neural voice — browser models (MMS Urdu)\n\n")
    f.write(f"Source: `facebook/mms-tts` models/{ISO} (Meta MMS research, CC-BY-NC 4.0) — shukriya Meta.\n\n")
    f.write("| file | size (MB) | full-line duration | rms |\n|---|---|---|---|\n")
    for name, mb, dur, rms in report:
        f.write(f"| model.{name}.onnx | {mb:.1f} | {dur:.2f}s | {rms:.4f} |\n")
    f.write(f"\nvocab: {len(symbols)} symbols · sampling_rate: {sr} Hz · add_blank: {int(hps.data.add_blank)}\n")
    f.write(f"\nvocab se bahar gire huroof (test lines): {json.dumps(drop_report, ensure_ascii=False)}\n")
    f.write("\nBrowser: onnxruntime-web (WASM); tokenizer = har harf ka id (vocab.json) "
            "+ beech mein blank(0) interleave; janch test_vectors.json se.\n")
print("done — models-web/ tayyar")
