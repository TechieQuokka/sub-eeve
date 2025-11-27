# EEVE-Korean Subtitle Translator

일본어/영어 자막을 한국어로 번역하는 CLI 도구입니다. EEVE-Korean-Instruct-10.8B Q8_0 모델을 사용합니다.

## 특징

- **고품질 한국어 번역**: 야놀자의 EEVE-Korean 모델 사용 (Chatbot Arena 7위)
- **맥락 인식**: 이전 대화를 고려한 자연스러운 번역
- **배치 처리**: 여러 자막을 한번에 번역하여 일관성 향상
- **순수 한글 출력**: 한자, 일본어, 중국어 문자 제거
- **RTX 3060 12GB 최적화**: Q8_0 양자화로 12GB GPU에서 실행 가능

## 시스템 요구사항

- **GPU**: RTX 3060 12GB 이상 (VRAM 8GB+)
- **OS**: Linux, Windows (WSL2), macOS
- **Python**: 3.8 이상
- **VRAM 사용량**: ~11GB (Q8_0)

## 설치

### 1. 의존성 설치

```bash
pip install llama-cpp-python tqdm
```

**GPU 가속 설치** (CUDA):
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 2. 모델 다운로드

EEVE-Korean-Instruct-10.8B Q8_0 모델을 다운로드합니다:

```bash
# Hugging Face CLI 설치
pip install huggingface-hub

# 모델 다운로드 (약 11GB)
huggingface-cli download heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF \
  --include "*Q8_0.gguf" \
  --local-dir ../models
```

또는 수동 다운로드:
- URL: https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF
- 파일: `EEVE-Korean-Instruct-10.8B-v1.0-Q8_0.gguf`
- 저장 위치: `../models/EEVE-Korean-Instruct-10.8B-v1.0-Q8_0.gguf`

### 3. 모델 경로 설정

`config.py` 파일에서 모델 경로 확인:

```python
MODEL_PATH = "../models/EEVE-Korean-Instruct-10.8B-v1.0-Q8_0.gguf"
```

## 사용법

### 기본 사용 (일본어 → 한국어)

```bash
python translate.py --input subtitle.json --output subtitle_kr.json \
                    --input-lang ja --target-lang ko
```

### 배치 모드 (권장)

여러 자막을 한번에 번역하여 일관성과 속도 향상:

```bash
python translate.py --input subtitle.json --output subtitle_kr.json \
                    --input-lang ja --target-lang ko --batch
```

### 맥락 없이 번역 (빠름)

이전 대화를 고려하지 않고 빠르게 번역:

```bash
python translate.py --input subtitle.json --output subtitle_kr.json \
                    --input-lang ja --target-lang ko --no-context
```

### 사용자 정의 모델 경로

```bash
python translate.py --input subtitle.json --output subtitle_kr.json \
                    --input-lang ja --target-lang ko \
                    --model /path/to/custom-model.gguf
```

## 명령줄 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input`, `-i` | 입력 JSON 자막 파일 (필수) | - |
| `--output`, `-o` | 출력 JSON 파일 (필수) | - |
| `--input-lang` | 원본 언어 (`ja`, `en`, `zh`, `ko`) (필수) | - |
| `--target-lang` | 목표 언어 (보통 `ko`) (필수) | - |
| `--batch` | 배치 번역 모드 활성화 | False |
| `--batch-size` | 배치당 자막 수 | 20 |
| `--no-context` | 맥락 인식 비활성화 | False |
| `--model` | 사용자 정의 모델 경로 | config.py 참조 |
| `--no-progress` | 진행률 표시 비활성화 | False |

## 입력 JSON 형식

```json
{
  "segments": [
    {
      "text": "こんにちは、世界！",
      "start": 0.0,
      "end": 2.5
    },
    {
      "text": "今日はいい天気ですね。",
      "start": 2.5,
      "end": 5.0
    }
  ]
}
```

## 출력 JSON 형식

```json
{
  "segments": [
    {
      "text": "こんにちは、世界！",
      "text_ko": "안녕하세요, 세계!",
      "start": 0.0,
      "end": 2.5
    },
    {
      "text": "今日はいい天気ですね。",
      "text_ko": "오늘은 날씨가 좋네요.",
      "start": 2.5,
      "end": 5.0
    }
  ]
}
```

## 프로젝트 구조

```
sub-eeve/
├── config.py              # 모델 경로 및 설정
├── translate.py           # 메인 CLI 스크립트
├── utils/
│   ├── __init__.py       # 모듈 초기화
│   ├── model_loader.py   # GGUF 모델 로딩
│   ├── translator.py     # 번역 엔진 (프롬프트 최적화)
│   └── json_handler.py   # JSON 파일 입출력
└── README.md             # 이 파일
```

## 설정 (config.py)

주요 설정 파라미터:

```python
# 모델 설정
MODEL_PATH = "../models/EEVE-Korean-Instruct-10.8B-v1.0-Q8_0.gguf"
N_CTX = 8192              # 컨텍스트 창 크기
N_GPU_LAYERS = -1         # GPU 레이어 수 (-1 = 전체)
TEMPERATURE = 0.3         # 낮을수록 결정적

# 번역 설정
CHUNK_SIZE = 20           # 배치당 자막 수
CONTEXT_HISTORY = 3       # 맥락으로 사용할 이전 자막 수
```

## 성능

RTX 3060 12GB 기준:

| 모드 | 속도 | 품질 | VRAM |
|------|------|------|------|
| 단일 + 맥락 | ~2초/자막 | ⭐⭐⭐⭐⭐ | 11GB |
| 배치 + 맥락 | ~0.5초/자막 | ⭐⭐⭐⭐⭐ | 11GB |
| 단일 (맥락 없음) | ~1초/자막 | ⭐⭐⭐⭐ | 11GB |

## 트러블슈팅

### 1. 모델 로딩 실패

```
❌ Model file not found
```

**해결책**: 모델을 다운로드하고 경로 확인
```bash
huggingface-cli download heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF \
  --include "*Q8_0.gguf" --local-dir ../models
```

### 2. VRAM 부족

```
CUDA error: out of memory
```

**해결책**:
- Q6_K 또는 Q5_K 양자화 사용 (~8.5GB / ~7.5GB)
- `config.py`에서 `N_CTX` 감소 (8192 → 4096)

### 3. GPU 가속 안됨

```
llama_init_from_gfile: warning: not loading GPU
```

**해결책**: llama-cpp-python을 CUDA 지원으로 재설치
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 4. 한자/중국어로 번역됨

Solar 모델과 달리 EEVE는 한국어 어휘가 확장되어 이 문제가 거의 없습니다.
만약 발생하면:
- `config.py`에서 `TEMPERATURE` 낮추기 (0.3 → 0.1)
- 프롬프트에서 한글 강조 부분 확인

### 5. 번역 속도가 느림

**해결책**:
- `--batch` 옵션 사용
- `--no-context` 옵션으로 맥락 비활성화
- GPU 가속 활성화 확인

## 다른 모델 사용

### EXAONE-3.5-7.8B (더 빠름, 32K 컨텍스트)

```bash
# 모델 다운로드
huggingface-cli download bartowski/EXAONE-3.5-7.8B-Instruct-GGUF \
  --include "*Q8_0.gguf" --local-dir ../models

# config.py 수정
MODEL_PATH = "../models/EXAONE-3.5-7.8B-Instruct-Q8_0.gguf"
N_CTX = 32768  # 긴 컨텍스트 지원
```

### 더 가벼운 양자화 (Q6_K, Q5_K, Q4_K_M)

```bash
# Q6_K 다운로드 (8.5GB, 품질 98%)
huggingface-cli download heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF \
  --include "*Q6_K.gguf" --local-dir ../models

# Q5_K 다운로드 (7.5GB, 품질 95%)
huggingface-cli download heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF \
  --include "*Q5_K.gguf" --local-dir ../models

# Q4_K_M 다운로드 (6GB, 품질 92%)
huggingface-cli download heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF \
  --include "*Q4_K_M.gguf" --local-dir ../models
```

## 라이선스

- EEVE-Korean 모델: Apache 2.0
- 이 코드: MIT License

## 참고

- [EEVE-Korean 모델](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)
- [GGUF 버전](https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

## Solar 모델과의 비교

| 특징 | Solar 10.7B | EEVE-Korean 10.8B |
|------|-------------|-------------------|
| 한국어 어휘 | 기본 | +8,960 토큰 확장 |
| 한글 출력 | ❌ 한자/중국어 혼입 | ✅ 순수 한글 |
| 번역 품질 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 속도 | 빠름 | 비슷 |
| 한국어 이해 | 보통 | 우수 |

**결론**: EEVE-Korean이 Solar보다 일본어→한국어 번역에 훨씬 우수합니다.
