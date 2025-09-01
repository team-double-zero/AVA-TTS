import torch
import scipy.io.wavfile
from transformers import AutoProcessor, AutoModelForTextToWaveform
import datetime
import os

# --- device: force CUDA ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

# --- load model ------------------------------------------------------------
processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModelForTextToWaveform.from_pretrained("suno/bark", dtype=dtype).to(device).eval()

# --- text input --------------------------------------------------------ßßß----
text = "안녕하세요. Bark로 생성한 샘플 음성입니다."
inputs = processor(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# --- inference -------------------------------------------------------------
with torch.no_grad():
    audio_values = model.generate(**inputs)

# --- save wav --------------------------------------------------------------
sample_rate = getattr(processor, "sampling_rate", getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 24000))
audio = audio_values.squeeze(0).cpu().numpy()
audio = (audio * 32767.0).clip(-32768, 32767).astype("int16")

OUTPUT_DIR = "./sample"
os.makedirs(OUTPUT_DIR, exist_ok=True)
filename_prefix = f"{OUTPUT_DIR}/sample_{datetime.datetime.now():%m%d_%H%M%S}"

scipy.io.wavfile.write(f"{filename_prefix}.wav", rate=sample_rate, data=audio)
print(f"✅ 생성 완료: {filename_prefix}.wav")