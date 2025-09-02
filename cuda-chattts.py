import os
import sys
import numpy as np
import soundfile as sf
import torch

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

def tts(text: str, out_path: str = "sample_cuda.wav") -> str:
    """
    Simplest possible inference on CUDA.
    """
    with torch.inference_mode():
        wavs = chat.infer(
            text,
            # vanilla params; pass only what's needed
            params_refine_text=ChatTTS.Chat.RefineTextParams(),
            params_infer_code=ChatTTS.Chat.InferCodeParams(spk_emb=SPK_EMB),
            use_decoder=False,   # keeps it stable/fast
            stream=False,
        )

    # ChatTTS returns list[np.ndarray] in float32 range [-1, 1]
    y = np.asarray(wavs[0], dtype=np.float32)
    sr = 24000  # ChatTTS default
    sf.write(out_path, y, sr)
    return out_path

if __name__ == "__main__":
    out = tts("안녕하세요. CUDA 환경에서 ChatTTS를 테스트합니다. 음성 합성을 시작합니다.", "sample_cuda.wav")
    print("[OK] saved ->", out)