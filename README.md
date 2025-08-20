# AVA-TTS
tts voice 음성파일을 생성합니다.
보이스 클로닝을 위한 샘플 보이스는 필수는 아니지만

# 디렉토리

### /input
- 입력에 사용될 파라미터를 json 형식으로 전달합니다
  - sentence: tts가 발화할 문장
  - sample: 사용할 샘플 파일명 (.wav 확장자 생략)
  - exaggeration: 절대적인 억양 높이
  - weight: 무게 (억양 높낮이 차이)
  - temperature: 다양성

*sentence 파라미터를 제외한 다른 값은 생략가능하며, 기본값으로 사용됩니다.*

### output
- 생성된 음성 파일이 WAV 형식으로 저장됩니다.

### sample
- 보이스 클로닝에 사용될 샘플 WAV 파일을 보관합니다.

# 설치 및 실행

## 의존성 설치
```
pip install -r requirements.txt
```
## 파일 실행
```
python generate.py <filename>
```
or
```
python generate.py <filename.json>
```
