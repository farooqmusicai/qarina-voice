# Piper Urdu voices (aurat + mard) -> browser ONNX
#
# Kyun Piper: Meta MMS ka Urdu model sirf EK mard ki awaz hai. Piper ke paas
# Urdu ki do awazein hain — ur_PK-aegis_female (AURAT) aur ur_PK-fasih (mard).
# Yeh bhi wahi VITS hain, magar:
#   - 22050 Hz  (MMS 16000 Hz)  -> saaf awaz
#   - ~63 MB fp32 (MMS 114 MB)  -> chhota aur tez
#   - raw huroof nahi, eSpeak IPA phonemes lete hain (tools/piper_ids.mjs)
#
# Signature (piper/src/cpp/piper.cpp):
#   input: int64[1,T], input_lengths: int64[1], scales: float32[3]
#   scales = [noise_scale, length_scale, noise_w]
#   output: float32[1,1,N]

import hashlib, json, os, time, urllib.request
import numpy as np
import onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from scipy.io import wavfile

HF   = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
OUT  = "build_piper"
VOICES = {
    "female": "ur/ur_PK/aegis_female/medium/ur_PK-aegis_female-medium",
    "male":   "ur/ur_PK/fasih/medium/ur_PK-fasih-medium",
}
os.makedirs(OUT, exist_ok=True)

def mb(p): return os.path.getsize(p) / 1e6
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def grab(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0: return dst
    print("  <-", url, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "qarina-voice/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    return dst

with open(os.path.join(OUT, "ids.json"), encoding="utf-8") as f:
    REF = json.load(f)                       # {segs:[...], ids:[[...]]}
SEGS, IDS = REF["segs"], REF["ids"]

def decoder_convs(path):
    """HiFi-GAN decoder ke Conv nodes = pehle ConvTranspose ke baad wale sab.
    (Piper ka export MMS jaisa '/dec/' naam nahi deta, is liye dhaancha dekha jaata hai.)"""
    g = onnx.load(path, load_external_data=False).graph
    first = None
    for i, n in enumerate(g.node):
        if n.op_type == "ConvTranspose": first = i; break
    if first is None: return []
    return [n.name for n in g.node[first:] if n.op_type == "Conv"]

def synth(path, ids, cfg, reps=1):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inf = cfg.get("inference", {})
    scales = np.array([inf.get("noise_scale", 0.667),
                       inf.get("length_scale", 1.0),
                       inf.get("noise_w", 0.8)], dtype=np.float32)
    outs, t0 = [], time.time()
    for _ in range(reps):
        outs = []
        for seq in ids:
            x = np.array([seq], dtype=np.int64)
            y = s.run(None, {"input": x,
                             "input_lengths": np.array([x.shape[1]], dtype=np.int64),
                             "scales": scales})[0]
            outs.append(np.asarray(y).reshape(-1))
    wall = (time.time() - t0) / reps
    pcm = np.concatenate([np.concatenate([o, np.zeros(int(0.25 * cfg["audio"]["sample_rate"]), np.float32)])
                          for o in outs])
    dur = len(pcm) / cfg["audio"]["sample_rate"]
    return pcm, wall, dur

report = {"voices": {}}
for name, rel in VOICES.items():
    print(f"\n===== {name}  ({rel.split('/')[-1]}) =====", flush=True)
    base = os.path.join(OUT, name)
    fp32 = grab(f"{HF}/{rel}.onnx",      base + ".fp32.onnx")
    cfgp = grab(f"{HF}/{rel}.onnx.json", base + ".config.json")
    with open(cfgp, encoding="utf-8") as f: cfg = json.load(f)
    sr = cfg["audio"]["sample_rate"]
    print(f"  fp32 {mb(fp32):.1f} MB · {sr} Hz · speakers {cfg.get('num_speakers')}"
          f" · espeak '{cfg.get('espeak',{}).get('voice')}'", flush=True)

    dec = decoder_convs(fp32)
    OPS = ["MatMul", "Gemm", "Conv"]
    made = {"fp32": fp32}

    q8 = base + ".q8.onnx"
    quantize_dynamic(fp32, q8, weight_type=QuantType.QInt8, op_types_to_quantize=OPS)
    assert mb(q8) < 0.6 * mb(fp32), (
        f"quantize ne kuch nahi kiya: q8 {mb(q8):.1f} vs fp32 {mb(fp32):.1f} MB")
    made["q8"] = q8

    if dec:
        hyb = base + ".hybrid.onnx"
        quantize_dynamic(fp32, hyb, weight_type=QuantType.QInt8,
                         op_types_to_quantize=OPS, nodes_to_exclude=dec)
        made["hybrid"] = hyb
        print(f"  hybrid: {len(dec)} decoder Conv fp32 chhoray gaye", flush=True)
    else:
        print("  WARNING: ConvTranspose nahi mila — hybrid skip", flush=True)

    report["voices"][name] = {"model": rel.split("/")[-1], "sample_rate": sr,
                              "inference": cfg.get("inference", {}), "variants": {}}
    for var, p in made.items():
        pcm, wall, dur = synth(p, IDS, cfg)
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
        rtf = wall / dur if dur else 99
        wav = os.path.join(OUT, f"qarina-{name}-{var}.wav")
        wavfile.write(wav, sr, (np.clip(pcm, -1, 1) * 32767).astype(np.int16))
        ok = np.isfinite(pcm).all() and rms > 0.005
        print(f"  {var:7s} {mb(p):6.1f} MB  awaz {dur:5.2f}s  bana {wall:5.2f}s"
              f"  rtf {rtf:5.3f}  rms {rms:.4f}  {'OK' if ok else '*** KHARAB ***'}", flush=True)
        assert ok, f"{name}/{var}: awaz kharab hai (rms {rms:.5f})"
        report["voices"][name]["variants"][var] = {
            "size_mb": round(mb(p), 1), "rtf": round(rtf, 3),
            "rms": round(rms, 4), "sha256": sha256(p), "wav": os.path.basename(wav)}

report["test_text"] = SEGS
with open(os.path.join(OUT, "piper-manifest.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print("\n" + json.dumps(report, ensure_ascii=False, indent=1))
