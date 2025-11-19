# NVIDIA NeMo Curator 데이터 큐레이션 가이드

## 목차
1. [소개](#소개)
2. [환경 설정](#환경-설정)
3. [기본 데이터 큐레이션](#기본-데이터-큐레이션)
4. [합성 데이터 생성](#합성-데이터-생성)
5. [파일 구조](#파일-구조)
6. [실습 가이드](#실습-가이드)

## 소개

NVIDIA NeMo Curator는 생성형 AI 모델 훈련을 위한 대규모, 고품질 데이터셋을 준비하는 오픈 소스 프레임워크입니다. 이 가이드는 NeMo Curator를 사용하여 데이터 큐레이션부터 합성 데이터 생성까지의 전체 워크플로우를 다룹니다.

### 주요 기능
- **데이터 정리 및 통합**: 텍스트 정규화, HTML/URL 제거
- **필터링**: 문서 크기, 품질 기반 필터링
- **PII 식별 및 제거**: 개인 식별 정보 보호
- **합성 데이터 생성**: 주제/부주제, Q&A, 수학 문제 생성
- **품질 평가**: Reward Model을 활용한 데이터 품질 평가

## 환경 설정

### 필수 요구사항
- Python 3.8+
- NVIDIA GPU (권장)
- Dask 및 RAPIDS (대규모 데이터 처리용)

### 설치
```bash
pip install nemo-curator
pip install openai
pip install presidio-analyzer
pip install dask distributed
```

### NVIDIA API 키 설정
```bash
export NVIDIA_API_KEY="your-api-key-here"
export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
```

> **참고**: 무료 API 키는 [build.nvidia.com](https://build.nvidia.com)에서 발급받을 수 있습니다.

## 기본 데이터 큐레이션

### 1. 텍스트 정리 및 통합

데이터셋의 기본적인 정리 작업을 수행합니다:

```python
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers import UnicodeReformatter
from nemo_curator.modules.modify import Modify
from nemo_curator import Sequential

# 데이터셋 로드
dataset = DocumentDataset.read_json("./data", add_filename=True)

# 정리 파이프라인 구성
cleaners = Sequential([
    Modify(QuotationTagUnifier()),  # 따옴표 통일 및 태그 제거
    Modify(UnicodeReformatter()),    # 유니코드 정규화
])

# 실행
cleaned_dataset = cleaners(dataset).persist()
```

#### 주요 정리 작업
- 따옴표 정규화 (`'` → `'`, `"` → `"`)
- HTML/URL/이메일 태그 제거
- 유니코드 문제 해결
- 탭 문자 제거

### 2. 문서 크기 필터링

불완전하거나 너무 짧은 문서를 제거합니다:

```python
from nemo_curator import ScoreFilter
from nemo_curator.filters import WordCountFilter, RepeatingTopNGramsFilter

filters = Sequential([
    ScoreFilter(WordCountFilter(min_words=80)),  # 최소 80단어
    ScoreFilter(IncompleteDocumentFilter()),      # 불완전 문서 제거
    ScoreFilter(RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2)),
])

filtered_dataset = filters(dataset)
```

#### 필터링 기준
- **최소 단어 수**: 80단어 이상
- **문장 완결성**: 마침표, 물음표, 느낌표로 끝나는 문서
- **반복 패턴 제거**: N-gram 반복 비율 제한

### 3. PII (개인 식별 정보) 제거

개인정보 보호를 위한 PII 식별 및 제거:

```python
from nemo_curator.modifiers.pii_modifier import PiiModifier

modifier = PiiModifier(
    supported_entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    anonymize_action="replace",
    device="cpu",
)

redacted_dataset = Modify(modifier)(dataset)
```

#### 지원되는 PII 유형
- `PERSON`: 사람 이름
- `EMAIL_ADDRESS`: 이메일 주소
- `PHONE_NUMBER`: 전화번호
- `ADDRESS`: 주소
- `CREDIT_CARD`: 신용카드 번호
- `US_SSN`: 미국 사회보장번호
- 기타: `IP_ADDRESS`, `URL`, `DATE_TIME` 등

### 4. 다국어 지원

NeMo Curator는 여러 언어를 지원합니다. `languages-config.yml` 파일에서 설정:

```yaml
nlp_engine_name: spacy
models:
  - lang_code: en
    model_name: en_core_web_lg
  - lang_code: es
    model_name: es_core_news_md
  - lang_code: fr
    model_name: fr_core_news_md
```

## 합성 데이터 생성

### 1. OpenAI 클라이언트 설정

```python
from openai import OpenAI
from nemo_curator import OpenAIClient
from nemo_curator.synthetic import NemotronGenerator

# OpenAI 클라이언트 초기화
openai_client = OpenAI(
    base_url=os.environ["NVIDIA_BASE_URL"],
    api_key=os.environ["NVIDIA_API_KEY"],
)

# NeMo Curator 클라이언트 초기화
curator_client = OpenAIClient(openai_client)
generator = NemotronGenerator(curator_client)
```

### 2. 주제 및 부주제 생성

다양한 주제 계층 구조를 생성합니다:

```python
# 매크로 주제 생성
macro_topics = generator.generate_macro_topics(
    model="mistralai/mistral-7b-instruct-v0.3",
    model_kwargs={"temperature": 0.1, "top_p": 0.9, "max_tokens": 1024},
    n_macro_topics=5
)

# 부주제 생성
subtopics = generator.generate_subtopics(
    model="mistralai/mistral-7b-instruct-v0.3",
    model_kwargs={"temperature": 0.1, "top_p": 0.9, "max_tokens": 1024},
    macro_topic=macro_topics[0],
    n_subtopics=3
)
```

#### 프롬프트 커스터마이징

스페인어로 주제 생성:
```python
macro_topics_prompt_spanish = (
    "Genera {n_macro_topics} temas amplios que abarquen diversos aspectos "
    "de nuestra vida diaria, el mundo y la ciencia. Tu respuesta debe ser "
    "únicamente una lista de temas. Por ejemplo: 1. Comida y bebidas. "
    "\n2. Tecnología.\n"
)
```

### 3. Q&A 데이터셋 생성

지도 학습(SFT)을 위한 질문-답변 쌍 생성:

#### 3.1 질문 생성
```python
questions = generator.generate_open_qa_from_topic(
    model=model,
    model_kwargs=model_kwargs,
    topic="Agroecology and Biodiversity Conservation",
    n_openlines=4
)
```

#### 3.2 질문 개선
```python
revised_questions = generator.revise_open_qa(
    model=model,
    model_kwargs=model_kwargs,
    openline=questions[0],
    n_revisions=1
)
```

#### 3.3 답변 생성
```python
dialogue = generator.generate_dialogue(
    openline=revised_question,
    user_model=model,
    user_model_kwargs=model_kwargs,
    assistant_model=model,
    assistant_model_kwargs=model_kwargs,
    n_user_turns=1
)
```

### 4. 수학 문제 생성

특정 주제에 대한 수학 문제와 해답을 생성:

```python
# 수학 문제 생성
math_problems = generator.generate_math_problem(
    model=model,
    topic="Algebra",
    n_openlines=3,
    prompt_template=math_prompt_template
)
```

#### 문제-해답 형식
```
Problem: [문제 설명]
Solution: [해답]
```

### 5. Reward Model을 활용한 품질 평가

생성된 데이터의 품질을 평가합니다:

```python
# Reward Model로 평가
response = openai_client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-70b-reward",
    messages=[
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
)

reward_score = float(response.choices[0].message.content.split(":")[1])
```

#### 품질 기준
- **높은 점수** (>= -20): 고품질, 데이터셋에 포함
- **낮은 점수** (< -20): 저품질, 제외
- 점수 범위는 일반적으로 0 ~ -60 사이

## 파일 구조

```
nvidia_dataset_curation/
├── README.md                          # 본 가이드
├── data_curation_practice.ipynb       # 통합 실습 노트북
│
├── 01_basics_curation.ipynb           # 기본 큐레이션
├── 03_synthetic_data_generation_topics_and_subtopics.ipynb
├── 04_synthetic_data_generation_questions_and_answers.ipynb
├── 05_synthetic_data_generation_math_problems_with_solutions.ipynb
│
├── utils.py                           # 유틸리티 함수
├── languages-config.yml               # 언어 설정
├── lm_tasks.yaml                      # 언어 모델 태스크 정의
└── index1.pdf                         # 참고 문서
```

## 실습 가이드

### 기본 실습 순서

1. **환경 설정 확인**
   - API 키 설정 확인
   - 필요한 라이브러리 import

2. **기본 큐레이션 실습** (`01_basics_curation.ipynb`)
   - 데이터 로드 및 정리
   - 필터링 적용
   - PII 제거

3. **주제 생성 실습** (`03_synthetic_data_generation_topics_and_subtopics.ipynb`)
   - 매크로 주제 생성
   - 부주제 생성
   - 다국어 지원

4. **Q&A 생성 실습** (`04_synthetic_data_generation_questions_and_answers.ipynb`)
   - 질문 생성 및 개선
   - 답변 생성
   - Reward Model 평가

5. **수학 문제 생성 실습** (`05_synthetic_data_generation_math_problems_with_solutions.ipynb`)
   - 수학 부주제 생성
   - 문제-해답 쌍 생성
   - 품질 평가

### 통합 실습 노트북

`data_curation_practice.ipynb`는 전체 워크플로우를 하나의 노트북에서 실습할 수 있도록 구성되어 있습니다.

## 모범 사례

### 데이터 큐레이션
1. **단계별 저장**: 각 큐레이션 단계마다 중간 결과를 저장
2. **병렬 처리**: Dask를 활용한 대규모 데이터 병렬 처리
3. **점진적 필터링**: 엄격한 필터부터 점진적으로 적용

### 합성 데이터 생성
1. **다양성 확보**: 다양한 주제와 형식으로 데이터 생성
2. **품질 검증**: Reward Model로 품질 검증 필수
3. **프롬프트 최적화**: 각 언어와 도메인에 맞는 프롬프트 사용
4. **재시도 로직**: YAML 변환 실패 시 재시도 메커니즘 구현

### 성능 최적화
1. **배치 크기 조정**: GPU 메모리에 맞게 배치 크기 설정
2. **병렬 요청**: 가능한 경우 병렬 API 요청
3. **캐싱**: 반복적인 작업에는 결과 캐싱

## 문제 해결

### 일반적인 오류

#### YamlConversionError
**원인**: LLM 응답이 예상 형식과 다름
**해결**:
- 프롬프트 명확화
- 재시도 횟수 증가
- 커스텀 파서 구현

#### OutOfMemoryError
**원인**: GPU/CPU 메모리 부족
**해결**:
- 배치 크기 감소
- 데이터 파티션 수 증가
- 청크 단위 처리

#### API Rate Limit
**원인**: API 호출 제한 초과
**해결**:
- 요청 간 지연 추가
- 배치 크기 조정
- API 키 로테이션

## 추가 리소스

- [NeMo Curator 공식 문서](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/)
- [NVIDIA API 카탈로그](https://build.nvidia.com)
- [Nemotron-4 340B Technical Report](https://arxiv.org/pdf/2406.11704)
- [Reward Model 리더보드](https://huggingface.co/spaces/allenai/reward-bench)

## 라이선스 및 기여

이 프로젝트는 NVIDIA NeMo Curator를 기반으로 합니다. 자세한 라이선스 정보는 [NeMo Curator GitHub](https://github.com/NVIDIA/NeMo-Curator)를 참조하세요.
