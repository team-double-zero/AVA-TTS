import os
import sys
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, lfilter

# --- CUDA ONLY: fail fast if no GPU ---
if not torch.cuda.is_available():
    print("[ERROR] CUDA GPU not available. Run this on a CUDA machine.")
    sys.exit(1)

# use CUDA by default
torch.set_default_device("cuda")
torch.set_num_threads(1)  # not critical on GPU, but keeps env consistent
print(f"[INFO] Using device: {'cuda'} (GPU count={torch.cuda.device_count()})")

# ChatTTS
import ChatTTS  # pip install ChatTTS

# init & load weights
chat = ChatTTS.Chat()
# compile=False starts faster and is more robust across envs
chat.load(compile=False)

# one random speaker embedding (no seed arg in current API)
SPK_EMB = chat.sample_random_speaker()

def _normalize_audio(y: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """
    Normalize audio to target dBFS level.
    """
    rms = np.sqrt(np.mean(y**2))
    if rms == 0:
        return y
    current_dbfs = 20 * np.log10(rms)
    gain_db = target_dbfs - current_dbfs
    gain = 10 ** (gain_db / 20)
    y_norm = y * gain
    y_norm = np.clip(y_norm, -1.0, 1.0)
    return y_norm

def _highpass_filter(y: np.ndarray, sr: int, cutoff: float = 80.0, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth high-pass filter to reduce low-frequency rumble.
    """
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y_filtered = lfilter(b, a, y)
    return y_filtered

def tts(text: str, out_path: str = "sample_cuda.wav") -> str:
    """
    Simplest possible inference on CUDA with stability/quality improvements:
    - Split long text into sentences
    - Use recommended params with lower temperature/top_p
    - Normalize audio to -1 dBFS
    - Apply high-pass filter to reduce low-frequency rumble
    """
    import re

    # Split text into sentences for more stable output
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    if len(sentences) == 0:
        sentences = [text]

    sr = 24000  # ChatTTS default
    wavs = []

    with torch.inference_mode():
        for sent in sentences:
            if not sent:
                continue
            wav_segment = chat.infer(
                sent,
                params_refine_text=ChatTTS.Chat.RefineTextParams(temperature=0.3),
                params_infer_code=ChatTTS.Chat.InferCodeParams(spk_emb=SPK_EMB, temperature=0.3, top_p=0.7),
                use_decoder=False,
                stream=False,
            )
            wavs.append(np.asarray(wav_segment[0], dtype=np.float32))

    # Concatenate all segments
    y = np.concatenate(wavs)

    # Normalize audio to -1 dBFS
    y = _normalize_audio(y, target_dbfs=-1.0)

    # Apply high-pass filter to reduce low-frequency rumble
    y = _highpass_filter(y, sr=sr, cutoff=80.0, order=4)

    sf.write(out_path, y, sr)
    return out_path

if __name__ == "__main__":
    out = tts("Hello, this is a test of ChatTTS running in a CUDA environment. Starting speech synthesis now.", "sample_cuda.wav")
    print("[OK] saved ->", out)