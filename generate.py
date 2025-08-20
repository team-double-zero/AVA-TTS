import sys
import json
from pathlib import Path
from datetime import datetime
import os
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager" 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="We detected that you are passing `past_key_values`")
warnings.filterwarnings("ignore", message="We detected that you are passing `past_key_values` as a tuple of tuples")

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS  # type: ignore

def throw_error_and_exit(msg, code=2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def load_request(json_path: Path) -> dict:
    if not json_path.exists():
        throw_error_and_exit(f"JSON not found: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        throw_error_and_exit(f"JSON parse error: {e}")

def main():
    # 인자: JSON 파일 경로 1개
    if len(sys.argv) < 2: throw_error_and_exit(f"Usage: python generate.py <request.json>\n")

    json_path = sys.argv[1]
    if not json_path.endswith(".json"): json_path += ".json"
    data = load_request(Path(json_path))
    print(f"Processing request from: {json_path}")

    OUTPUT = str(json_path).strip(".json")
    SENTENCE = str(data.get("sentence", "Hello, this is a default output of the AVAZON TTS system.")).strip()
    SAMPLE = str(data.get("sample", "no_sample"))
    EXAGGERATION = float(data.get("exaggeration", 0.6))
    WEIGHT = float(data.get("weight", 0.35))
    TEMPERATURE = float(data.get("temperature", 0.8))

    # GPU 사용 권장 (CUDA)
    DEVICE = "cuda"
    model = ChatterboxTTS.from_pretrained(device=DEVICE)
    
    if SAMPLE != "no_sample":
        wav = model.generate(
            SENTENCE,
            audio_prompt_path=f"./sample/{SAMPLE}.wav",
            exaggeration=EXAGGERATION,
            cfg_weight=WEIGHT,
            temperature=TEMPERATURE,
        )
    else:
        wav = model.generate(
            SENTENCE,
            exaggeration=EXAGGERATION,
            cfg_weight=WEIGHT,
            temperature=TEMPERATURE,
        )

    help(model.generate)
    ta.save(f"./output/{OUTPUT}.wav", wav, model.sr)
    print(f"Output saved to: ./output/{OUTPUT}.wav\n\n")

if __name__ == "__main__":
    main()