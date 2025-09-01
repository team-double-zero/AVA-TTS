# AVA-TTS

from transformers import AutoProcessor, AutoModelForTextToWaveform
import scipy.io.wavfile
import torch

# 1. 모델과 프로세서 로드
processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModelForTextToWaveform.from_pretrained("suno/bark", torch_dtype=torch.float16, device_map="cpu")

# 2. 입력 텍스트
text = "안녕하세요. Bark로 생성한 샘플 음성입니다."

# 3. 텍스트 → 토큰 변환
inputs = processor(text, return_tensors="pt")

# 4. 모델 추론 (오디오 파형 생성)
with torch.no_grad():
    audio_values = model.generate(**inputs)

# 5. WAV 파일 저장
sample_rate = model.config.sample_rate
scipy.io.wavfile.write("bark_sample.wav", rate=sample_rate, data=audio_values.cpu().numpy())

print("✅ bark_sample.wav 생성 완료!")
# AVA-TTS (Bark minimal, CPU/MPS/CUDA auto, attention_mask & sr fixed)

import torch
import scipy.io.wavfile
from transformers import AutoProcessor, AutoModelForTextToWaveform

# --- device select ---------------------------------------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print(f"[INFO] Using device: {device}")

# dtype: fp16 on CUDA, otherwise fp32 (MPS/CPU are safer with fp32 for Bark)
dtype = torch.float16 if device.type == "cuda" else torch.float32

# --- load model / processor ------------------------------------------------
processor = AutoProcessor.from_pretrained("suno/bark")
# use `dtype` (torch_dtype is deprecated), and no device_map; we move manually
model = AutoModelForTextToWaveform.from_pretrained("suno/bark", dtype=dtype)
model.to(device)
model.eval()

# --- text input ------------------------------------------------------------
text = "안녕하세요. Bark로 생성한 샘플 음성입니다."

# Provide attention_mask to silence warnings; pad to longest for a single string is fine
inputs = processor(
    text,
    return_tensors="pt",
    padding="longest",
    return_attention_mask=True,
)
# move inputs to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# --- inference -------------------------------------------------------------
with torch.no_grad():
    audio_values = model.generate(**inputs)  # shape: (1, num_samples) float

# --- save wav --------------------------------------------------------------
# Bark configs on HF do not expose `sample_rate` in model.config. Get it from processor.
sample_rate = getattr(
    getattr(processor, "feature_extractor", processor),
    "sampling_rate",
    24000,  # Bark default
)

# Convert to int16 PCM for scipy writer
audio = audio_values.squeeze(0).detach().float().cpu().numpy()
audio = (audio * 32767.0).clip(-32768, 32767).astype("int16")

out_path = "bark_sample.wav"
scipy.io.wavfile.write(out_path, rate=sample_rate, data=audio)
print(f"✅ {out_path} 생성 완료! (sr={sample_rate}, device={device}, dtype={dtype})")
# Minimal Bark TTS (CUDA only)

import torch
import scipy.io.wavfile
from transformers import AutoProcessor, AutoModelForTextToWaveform

# --- device: force CUDA ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

# --- load model ------------------------------------------------------------
processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModelForTextToWaveform.from_pretrained("suno/bark", dtype=dtype).to(device).eval()

# --- text input ------------------------------------------------------------
text = "안녕하세요. Bark로 생성한 샘플 음성입니다."
inputs = processor(text, return_tensors="pt", padding="longest", return_attention_mask=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# --- inference -------------------------------------------------------------
with torch.no_grad():
    audio_values = model.generate(**inputs)

# --- save wav --------------------------------------------------------------
sample_rate = getattr(processor, "sampling_rate", 24000)
audio = audio_values.squeeze(0).cpu().numpy()
audio = (audio * 32767.0).clip(-32768, 32767).astype("int16")

scipy.io.wavfile.write("bark_sample.wav", rate=sample_rate, data=audio)
print("✅ bark_sample.wav 생성 완료!")