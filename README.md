# Bhashini-TTS

A fully offline, zero-PyTorch Gujarati Text-to-Speech inference pipeline built on ONNX Runtime. Converts raw Gujarati Unicode text to `.wav` audio using a three-stage model pipeline: **FastSpeech2 Encoder -> FastSpeech2 Decoder -> HiFi-GAN Vocoder**.

***

## Pipeline Overview
```
Gujarati Text
     |
     v
TTSDurAlignPreprocessor   (grapheme to phoneme, numeral expansion)
     |
     v
fs2_encoder.onnx          (text ids -> hidden states + log-durations)
     |
     v
Duration Expansion        (expand hidden states by predicted durations)
     |
     v
fs2_decoder.onnx          (hidden states -> normalised mel spectrogram)
     |
     v
Mel Denorm + x2.3262 Scale
     |
     v
hifigan.onnx              (mel spectrogram -> raw waveform)
     |
     v
22050 Hz WAV Audio
```

***

## Repository Structure
```
Bhashini-TTS/
├── models/
│   ├── fs2_encoder.onnx        # FastSpeech2 encoder (text -> hidden + durations)
│   ├── fs2_decoder.onnx        # FastSpeech2 decoder (hidden -> mel spectrogram)
│   ├── hifigan.onnx            # HiFi-GAN vocoder (mel -> waveform)
│   ├── fs2_encoder_q.onnx      # Quantized encoder (smaller / faster)
│   ├── fs2_decoder_q.onnx      # Quantized decoder
│   ├── hifigan_int8.onnx       # INT8 quantized HiFi-GAN
│   ├── model_fp16.pt           # FP16 PyTorch checkpoint (export reference)
│   ├── config.yaml             # ESPnet2 model config + token list
│   └── feats_stats.npz         # Mel spectrogram normalisation stats (count/sum/sum_square)
├── gu_inference_onnx.ipynb     # End-to-end Kaggle inference notebook
└── README.md
```

***

## Quick Start (Kaggle / Colab)

### 1. Install dependency
```python
!pip install onnxruntime
```

### 2. Download helper files from [smtiitm/Fastspeech2_HS](https://github.com/smtiitm/Fastspeech2_HS)
```python
import os
BASE = "/kaggle/working"
RAW  = "https://github.com/smtiitm/Fastspeech2_HS/raw/main"

files = {
    f"{BASE}/text_preprocess_for_inference.py": f"{RAW}/text_preprocess_for_inference.py",
    f"{BASE}/multilingualcharmap.json":          f"{RAW}/multilingualcharmap.json",
    f"{BASE}/NumberToText.py":                   f"{RAW}/NumberToText.py",
    f"{BASE}/numToText/gujarati.csv":            f"{RAW}/numToText/gujarati.csv",
    f"{BASE}/phone_dict/gujarati":               f"{RAW}/phone_dict/gujarati",
}
for dest, url in files.items():
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.system(f'wget -q "{url}" -O "{dest}"')
```

### 3. Load models and run inference
```python
import numpy as np, yaml, sys
import onnxruntime as ort
from scipy.io.wavfile import write as wav_write

SAMPLING_RATE = 22050
MAX_WAV_VALUE = 32768.0
MODEL_DIR     = "path/to/models"

sys.path.insert(0, BASE)
from text_preprocess_for_inference import TTSDurAlignPreprocessor

sess_enc  = ort.InferenceSession(f"{MODEL_DIR}/fs2_encoder.onnx",  providers=["CPUExecutionProvider"])
sess_dec  = ort.InferenceSession(f"{MODEL_DIR}/fs2_decoder.onnx",  providers=["CPUExecutionProvider"])
sess_hifi = ort.InferenceSession(f"{MODEL_DIR}/hifigan.onnx",      providers=["CPUExecutionProvider"])

with open(f"{MODEL_DIR}/config.yaml") as f:
    config = yaml.safe_load(f)
token2id = {tok: idx for idx, tok in enumerate(config["token_list"])}

stats    = np.load(f"{MODEL_DIR}/feats_stats.npz")
count    = stats["count"]
mel_mean = (stats["sum"] / count).astype(np.float32)
mel_std  = (np.sqrt(stats["sum_square"] / count - mel_mean**2)).astype(np.float32)

preprocessor = TTSDurAlignPreprocessor()

def synthesize(text, out_path="output.wav"):
    tokens, _ = preprocessor.preprocess(text, "gujarati", "male")
    token_ids  = np.array([[token2id.get(c, 1) for c in " ".join(tokens) if c != ' ']], dtype=np.int64)

    hs, d_log  = sess_enc.run(None, {"text_ids": token_ids})
    durations  = np.clip(np.round(np.exp(d_log[0])), 0, None).astype(int)

    hs_exp = np.concatenate(
        [np.repeat(hs[0][i:i+1], durations[i], axis=0) for i in range(hs.shape[1])],
        axis=0
    )[np.newaxis, :].astype(np.float32)

    mel_norm   = sess_dec.run(None, {"hidden_states": hs_exp})[0]
    mel_denorm = (mel_norm * mel_std[None, :, None] + mel_mean[None, :, None]).astype(np.float32)
    mel_scaled = (mel_denorm * 2.3262).astype(np.float32)  # critical scaling factor

    wav   = sess_hifi.run(None, {"mel_spectrogram": mel_scaled})[0].squeeze()
    wav   = wav / (np.abs(wav).max() + 1e-8)
    audio = (wav * MAX_WAV_VALUE).astype(np.int16)
    wav_write(out_path, SAMPLING_RATE, audio)
    print(f"Saved {len(audio)/SAMPLING_RATE:.2f}s -> {out_path}")

synthesize("નમસ્તે, તમે કેમ છો?", "hello.wav")
```

***

## Models

| File | Description | Size |
|------|-------------|------|
| `fs2_encoder.onnx` | FastSpeech2 encoder — text ids to hidden states + log-durations | ~83 MB |
| `fs2_decoder.onnx` | FastSpeech2 decoder — hidden states to normalised mel spectrogram | ~77 MB |
| `hifigan.onnx` | HiFi-GAN vocoder — mel spectrogram to waveform | ~54 MB |
| `fs2_encoder_q.onnx` | Quantized encoder | ~83 MB |
| `fs2_decoder_q.onnx` | Quantized decoder | ~32 MB |
| `hifigan_int8.onnx` | INT8 quantized HiFi-GAN | ~22 MB |
| `model_fp16.pt` | FP16 PyTorch checkpoint | ~73 MB |

> **Note:** All `.onnx` and `.pt` files are tracked via [Git LFS](https://git-lfs.github.com/).

***

## Key Implementation Notes

- **Mel scaling factor x2.3262** — required to match the HiFi-GAN training scale. Skipping this produces muffled or distorted audio.
- **feats_stats.npz format** — uses `count`, `sum`, `sum_square` keys (not `mean`/`std` directly). Mean and std are computed as: `mean = sum/count`, `std = sqrt(sum_square/count - mean^2)`.
- **No PyTorch at inference** — the entire pipeline runs on ONNX Runtime with `CPUExecutionProvider`. GPU (`CUDAExecutionProvider`) also works if available.
- **Preprocessing** — depends on `text_preprocess_for_inference.py` from [smtiitm/Fastspeech2_HS](https://github.com/smtiitm/Fastspeech2_HS) for grapheme-to-phoneme conversion and numeral expansion.

***

## Requirements
```
onnxruntime >= 1.17
numpy
scipy
pyyaml
```

***

## Language and Voice

| Attribute | Value |
|-----------|-------|
| Language | Gujarati (gu) |
| Voice | Male |
| Sample Rate | 22050 Hz |
***
