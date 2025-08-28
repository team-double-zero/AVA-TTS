import json
from pathlib import Path

class AVAChatterboxTTS:
    def __new__(cls, json_path: str | Path) -> str:
        self = super().__new__(cls)
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(self.json_path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # 필수 입력
        sentence: str = cfg.get("sentence", "").strip()
        if not sentence:
            raise ValueError("JSON must contain a non-empty 'sentence' field.")

        # 출력 파일 이름
        sample_name = cfg.get("sample") or "tts_out"
        out_path = self.json_path.parent / f"{sample_name}.wav"

        # 스타일 파라미터
        exaggeration = float(cfg.get("exaggeration", 0.0))
        weight = float(cfg.get("weight", 0.0))
        temperature = float(cfg.get("temperature", 1.0))

        # 실제 합성 실행
        cls._synthesize_with_chatterbox(
            text=sentence,
            output=str(out_path),
            exaggeration=exaggeration,
            weight=weight,
            temperature=temperature,
        )

        # 바로 str 경로 반환
        return str(out_path)

    @staticmethod
    def _synthesize_with_chatterbox(
        *,
        text: str,
        output: str,
        exaggeration: float,
        weight: float,
        temperature: float,
    ):
        try:
            import chatterbox_tts as ctts
        except ImportError as e:
            raise RuntimeError("chatterbox-tts 패키지를 import할 수 없습니다.") from e

        # 실제 엔진 함수명에 맞게 조정 (예시는 synthesize_to_file 가정)
        if hasattr(ctts, "synthesize_to_file"):
            ctts.synthesize_to_file(
                text=text,
                output=output,
                exaggeration=exaggeration,
                weight=weight,
                temperature=temperature,
            )
        else:
            raise NotImplementedError(
                "chatterbox-tts의 합성 함수명을 확인해서 _synthesize_with_chatterbox 수정 필요"
            )
