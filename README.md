# Bhashini-TTS 🗣️

**Offline, on-device Text-to-Speech for Gujarati** — built on FastSpeech2-HS + HiFi-GAN, exported to ONNX with INT8 quantization, targeting Android deployment via ONNX Runtime.

---

## Overview

This repository contains the full inference pipeline for a **privacy-first, low-latency Gujarati TTS system** with both male and female voices. The models are derived from the [IIT Madras FastSpeech2-HS](https://github.com/smtiitm/Fastspeech2_HS) architecture and are optimized for edge/mobile deployment — no internet connection required at inference time.

| Voice | Acoustic Model | Vocoder | Format |
|-------|---------------|---------|--------|
| Gujarati Male | FastSpeech2-HS (Encoder + Decoder) | HiFi-GAN V1 | ONNX INT8 |
| Gujarati Female | FastSpeech2-HS (Encoder + Decoder) | HiFi-GAN V1 | ONNX INT8 |

---

## Repository Structure

```
Bhashini-TTS/
├── model_gu_male/                  # Gujarati male voice assets
│   ├── config.yaml                 # Model configuration
│   ├── feats_stats.npz             # Mel denormalization stats (count/sum/sum_square)
│   ├── energy_stats.npz            # Energy normalization stats
│   ├── pitch_stats.npz             # Pitch normalization stats
│   ├── feats_type                  # Feature type descriptor
│   ├── fs2_encoder_male.onnx       # FastSpeech2 encoder (FP32)
│   ├── fs2_encoder_male_int8.onnx  # FastSpeech2 encoder (INT8 quantized)
│   ├── fs2_decoder_male.onnx       # FastSpeech2 decoder (FP32)
│   ├── fs2_decoder_male_int8.onnx  # FastSpeech2 decoder (INT8 quantized)
│   ├── hifigan_male.onnx           # HiFi-GAN vocoder (FP32)
│   └── hifigan_male_int8.onnx      # HiFi-GAN vocoder (INT8 quantized)
│
├── model_gu_female/                # Gujarati female voice assets
│   ├── config.yaml
│   ├── feats_stats.npz
│   ├── energy_stats.npz
│   ├── pitch_stats.npz
│   ├── feats_type
│   ├── fs2_encoder_female.onnx
│   ├── fs2_encoder_female_int8.onnx
│   ├── fs2_decoder_female.onnx
│   ├── fs2_decoder_female_int8.onnx
│   ├── hifigan_female.onnx
│   └── hifigan_female_int8.onnx
│
├── gu_inference_onnx.ipynb         # Male voice inference notebook
└── gu-female.ipynb                 # Female voice inference notebook
```

---

## Architecture

The TTS pipeline is split into three stages:

```
Text (Gujarati)
     │
     ▼
[G2P + Tokenization]          TTSDurAlignPreprocessor + multilingualcharmap.json
     │
     ▼
[FastSpeech2-HS Encoder]      → phoneme embeddings + variance (pitch, energy, duration)
     │
     ▼
[Length Regulator]            → expands phoneme sequence to frame-level
     │
     ▼
[FastSpeech2-HS Decoder]      → mel spectrogram [1, 80, T]
     │
     ▼
[HiFi-GAN Vocoder]            → raw PCM waveform @ 22050 Hz
```

> The encoder and decoder are exported as **two separate ONNX models** to work around ONNX's lack of native support for the non-differentiable length regulator.

### Key Constants

| Parameter | Value |
|-----------|-------|
| Sample Rate | 22050 Hz |
| Mel Bands | 80 |
| Hidden Dims | 384 |
| Mel Scale Factor | ×2.3262 |
| Mel Layout | Band-major `[1, 80, T]` |
| Stats Format | Online (`count`, `sum`, `sum_square`) |

---

## Inference (Python / Jupyter)

### Prerequisites

```bash
pip install onnxruntime numpy scipy
```

### Minimal Usage

```python
import numpy as np
import onnxruntime as ort

# Load sessions
enc_sess = ort.InferenceSession("model_gu_female/fs2_encoder_female_int8.onnx")
dec_sess = ort.InferenceSession("model_gu_female/fs2_decoder_female_int8.onnx")
voc_sess = ort.InferenceSession("model_gu_female/hifigan_female_int8.onnx")

# Load mel stats for denormalization
stats = np.load("model_gu_female/feats_stats.npz")
count = stats["count"]
mean  = stats["sum"] / count
var   = stats["sum_square"] / count - mean ** 2
std   = np.sqrt(np.maximum(var, 1e-8))

# [Tokenize input text → token_ids, then run encoder → decoder → vocoder]
# See gu-female.ipynb for the full pipeline
```

See **`gu-female.ipynb`** and **`gu_inference_onnx.ipynb`** for complete end-to-end inference examples including G2P preprocessing, duration prediction, mel denormalization, and audio playback.

---

## Android Deployment

The ONNX models are designed for deployment via **ONNX Runtime for Android**.

**Recommended setup:**
- ONNX Runtime Mobile (`onnxruntime-android`) in your Gradle dependencies
- INT8 quantized models for CPU inference (significant latency reduction)
- `AudioTrack` for PCM playback (streaming with overlap for low latency)

**Asset placement in Android project:**
```
app/src/main/assets/
├── config.yaml
├── feats_stats.npz
├── energy_stats.npz
├── pitch_stats.npz
├── multilingualcharmap.json
├── fs2_encoder_female_int8.onnx
├── fs2_decoder_female_int8.onnx
└── hifigan_female_int8.onnx
```

---

