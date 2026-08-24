# Qarina neural voice — browser models (MMS Urdu)

Source: `facebook/mms-tts` models/urd-script_arabic (Meta MMS research, **CC-BY-NC 4.0**) — shukriya Meta.

Wazan git mein nahi hain — Release `models-v1` par hain. `manifest.json` unhein point karta hai.

| variant | size (MB) | duration | rms | python rtf |
|---|---|---|---|---|
| model.fp32.onnx | 114.0 | 3.87s | 0.1697 | 0.26 |
| model.q8.onnx | 38.0 | 3.55s | 0.1715 | 2.53 |
| model.hybrid.onnx | 72.9 | 3.76s | 0.1748 | 0.29 |

vocab: 58 symbols · sampling_rate: 16000 Hz · add_blank: 1

vocab se bahar gire huroof (test lines): {"َ": 8, "،": 1, "ّ": 3, "ِ": 1}
