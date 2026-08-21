# Qarina neural voice — pehla qadam: Meta MMS ki Urdu awaz se namoonay
# (yeh script GitHub Actions par chalta hai; model wahan se download hota hai)
from transformers import VitsModel, AutoTokenizer
import torch, os
import numpy as np
from scipy.io import wavfile

IDS = ["facebook/mms-tts-urd-script_arabic", "facebook/mms-tts-urd"]
model = tok = None
chosen = ""
for mid in IDS:
    try:
        tok = AutoTokenizer.from_pretrained(mid)
        model = VitsModel.from_pretrained(mid)
        chosen = mid
        print("model:", mid)
        break
    except Exception as e:
        print("skip", mid, str(e)[:120])
assert model is not None, "koi MMS Urdu model nahi mila"

LINES = {
    "sample1-chattan": "مَیں چَلتا ہوں تَنہا یَہاں، چَٹّان سا کھَڑا ہوں مَیں",
    "sample2-aasman": "آسمان جھوٹ نہیں بولتا۔ ستارے اپنی چال سے پیغام دیتے ہیں",
    "sample3-ishq": "عِشق کے دَرد کا بیمار ہوں، اِک غَم خوار کی ہے آرزو",
}
os.makedirs("samples", exist_ok=True)
with open("samples/model_id.txt", "w") as f:
    f.write(chosen)
for name, text in LINES.items():
    ins = tok(text, return_tensors="pt")
    with torch.no_grad():
        wav = model(**ins).waveform.squeeze().numpy()
    wav = (wav / max(1e-9, float(np.abs(wav).max())) * 0.9 * 32767).astype(np.int16)
    wavfile.write("samples/%s.wav" % name, model.config.sampling_rate, wav)
    print("made", name, round(len(wav) / model.config.sampling_rate, 1), "sec")
print("done - samples/ tayyar")
