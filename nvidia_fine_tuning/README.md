# NVIDIA NeMo 파인튜닝 가이드

## 목차
1. [소개](#소개)
2. [NeMo Run 프레임워크](#nemo-run-프레임워크)
3. [지속적 사전 학습 (CPT)](#지속적-사전-학습-cpt)
4. [지도 미세 조정 (SFT)](#지도-미세-조정-sft)
5. [직접 선호도 최적화 (DPO)](#직접-선호도-최적화-dpo)
6. [파일 구조](#파일-구조)
7. [실습 가이드](#실습-가이드)

## 소개

NVIDIA NeMo 프레임워크는 대규모 언어 모델(LLM)의 학습 후 처리(Post-Training)를 위한 포괄적인 도구를 제공합니다. 이 가이드는 세 가지 주요 파인튜닝 기법을 다룹니다:

### 학습 파이프라인

```
Base Model → Continued Pretraining → Supervised Fine-Tuning → Alignment (DPO/RLHF) → Production
     ↓               ↓                      ↓                      ↓                  ↓
   General      Domain Knowledge      Task-Specific Skills    Human Preferences   Deployment
  Knowledge      Injection             (Instructions)          (Safety, Style)
```

### 주요 기법

| 기법 | 목적 | 데이터 요구사항 | 학습 시간 | 사용 사례 |
|------|------|----------------|----------|----------|
| **CPT** | 지식 주입 | 대량 (수백만~수십억 토큰) | 길음 | 도메인 적응, 최신 정보 |
| **SFT** | 작업 능력 향상 | 중간 (수천~수만 샘플) | 중간 | 명령 수행, 특정 작업 |
| **DPO** | 선호도 정렬 | 적음 (수백~수천 쌍) | 짧음 | 스타일, 안전성, 품질 |

## NeMo Run 프레임워크

### NeMo Run이란?

NeMo Run은 머신러닝 실험을 구성하고 실행하기 위한 NVIDIA의 프레임워크입니다.

**핵심 역할:**
- **구성 (Configuration)**: 모델, 하이퍼파라미터, 학습 설정 정의
- **실행 (Execution)**: 다양한 컴퓨팅 환경에서 실험 실행
- **관리 (Management)**: 실험, 로그, 체크포인트 추적

### 레시피(Recipe) 시스템

NeMo는 "레시피" 접근 방식을 사용하여 학습을 구성합니다:

```python
from nemo.collections import llm
import nemo_run as run

# 레시피 초기화
recipe = llm.llama32_3b.pretrain_recipe(
    name="my_experiment",
    dir="logs/",
    num_nodes=1,
    num_gpus_per_node=2,
)

# 설정 커스터마이징
recipe.trainer.max_steps = 1000
recipe.optim.config.lr = 1e-5

# 실행
executor = run.LocalExecutor(ntasks_per_node=2)
with run.Experiment("my_experiment") as exp:
    exp.add(recipe, executor=executor)
    exp.run(tail_logs=True)
```

**주요 구성 요소:**
- 모델 아키텍처 및 파라미터
- 분산 학습 설정
- 데이터 로더 구성
- 최적화 및 스케줄러
- 체크포인트 및 로깅

## 지속적 사전 학습 (CPT)

### 개념

지속적 사전 학습(Continued Pretraining, CPT)은 기존 사전 학습된 모델에 새로운 도메인 지식을 주입하는 기술입니다.

**주요 목표:**
1. **지식 주입**: 새로운 사실, 개체, 개념 학습
2. **도메인 적응**: 전문 분야(법률, 의료, 금융) 성능 향상
3. **시간적 업데이트**: 최신 정보 반영

### ChipNeMo 사례 연구

NVIDIA의 [ChipNeMo](https://arxiv.org/abs/2311.00176)는 CPT의 실제 적용 사례입니다:
- 칩 설계라는 매우 전문적인 도메인에 LLM 적응
- CPT를 중간 단계로 사용하여 도메인 지식 습득
- 이후 다운스트림 작업에서 성능 향상 달성

### CPT vs. RAG

| 측면 | CPT | RAG |
|------|-----|-----|
| **지식 저장** | 모델 가중치에 내재화 | 외부 검색 |
| **지연 시간** | 낮음 | 높음 (검색 단계) |
| **데이터 요구량** | 매우 많음 (수백만 토큰) | 적음 |
| **업데이트** | 재학습 필요 | 즉시 가능 |
| **비용** | 높음 (학습 비용) | 낮음 |
| **컨텍스트 길이** | 짧음 | 길음 |

**CPT가 필요한 경우:**
- 정적인 대규모 데이터 존재
- 초저지연 필요
- 컨텍스트 길이 축소 필요
- 고도의 전문성 필요

**RAG가 적합한 경우:**
- 동적 정보 업데이트 필요
- 출처 추적 필요
- 제한된 학습 리소스
- 대부분의 일반 사용 사례

### 데이터 준비

#### 1. JSONL 형식 변환

```python
import json
import os

# 텍스트 문서를 JSONL로 변환
documents = []
for filename in os.listdir("raw_texts"):
    if filename.endswith(".txt"):
        with open(f"raw_texts/{filename}", "r") as f:
            documents.append(f.read())

# NeMo 형식으로 저장
with open("dataset.jsonl", "w") as f:
    for doc in documents:
        f.write(json.dumps({"text": doc}) + "\n")
```

#### 2. 토큰화

```bash
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=dataset.jsonl \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type=llama-checkpoints/tokenizer \
    --output-prefix=tokenized_data \
    --workers=8 \
    --append-eod
```

### CPT 레시피 구성

```python
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

# CPT 레시피 초기화
cpt = llm.llama32_3b.pretrain_recipe(
    name="llama3.2_3b_cpt",
    dir="logs/",
    num_nodes=1,
    num_gpus_per_node=2,
)

# 명령어 튜닝된 모델에서 시작
cpt.resume = run.Config(
    nl.AutoResume,
    restore_config=run.Config(nl.RestoreConfig, path="llama-checkpoints/Llama-3.2-3B-Instruct"),
)

# 토크나이저 설정
tokenizer = run.Config(
    AutoTokenizer,
    pretrained_model_name="llama-checkpoints/tokenizer"
)

# 데이터 설정
cpt.data = run.Config(
    PreTrainingDataModule,
    paths=['tokenized_data'],
    split="100,0,0",  # 전체를 학습용으로
    global_batch_size=4,
    micro_batch_size=1,
    seq_length=4096,
    tokenizer=tokenizer,
)

# 학습 설정
cpt.trainer.max_steps = 1000
cpt.optim.config.lr = 1e-5
cpt.optim.lr_scheduler = None  # 상수 학습률
```

### 파국적 망각 (Catastrophic Forgetting)

**문제:**
- 명령어 튜닝된 모델에 대규모 CPT 수행 시 발생
- 명령 수행 능력 저하
- 대화 일관성 감소

**해결 방법:**
1. **기본 모델 사용**: 명령어 튜닝 전 모델에서 CPT 수행
2. **소규모 데이터**: 신중한 데이터 규모 선택
3. **후속 SFT**: CPT 후 SFT로 능력 복구

## 지도 미세 조정 (SFT)

### 개념

지도 미세 조정(Supervised Fine-Tuning, SFT)은 입력-출력 예시로 모델을 학습하여 특정 작업 능력을 향상시킵니다.

**주요 목표:**
1. **언어 적응**: 대상 언어의 유창성 향상
2. **도메인 전문성**: 전문 용어 사용 개선
3. **문체 제어**: 어조와 스타일 맞춤화
4. **추론 능력**: 추론 패턴 도입

### 데이터 형식

#### Input/Output 형식
```jsonl
{"input": "질문 텍스트", "output": "답변 텍스트"}
{"input": "다음 질문", "output": "다음 답변"}
```

#### 대화 형식 (Llama 3.2)
```python
llama3_2_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}"""
```

### SFT 레시피 구성

```python
from nemo.collections.llm.gpt.data import FineTuningDataModule

# SFT 레시피 초기화
sft = llm.llama32_3b.finetune_recipe(
    name="llama3.2_3b_sft",
    dir="logs/",
    num_nodes=1,
    num_gpus_per_node=2,
)

# 시작 모델 설정
sft.resume = run.Config(
    nl.AutoResume,
    restore_config=run.Config(nl.RestoreConfig, path="llama-checkpoints/Llama-3.2-3B-Instruct"),
)

# 데이터 설정
sft.data = run.Config(
    FineTuningDataModule,
    dataset_root="data/sft",  # training.jsonl, validation.jsonl, test.jsonl
    global_batch_size=16,
    micro_batch_size=1,
    seq_length=4096,
    tokenizer=tokenizer,
    dataset_kwargs={"prompt_template": llama3_2_prompt},
)

# 학습 설정
sft.trainer.max_steps = 500
sft.trainer.val_check_interval = 100
sft.optim.config.lr = 1e-5

# PEFT 제거 (전체 미세 조정)
del sft.peft
```

### 교차 언어 전이 (Cross-lingual Transfer)

**현상:**
- 한 언어로 학습하면 다른 언어에서도 성능 향상
- 다국어 모델의 공유 표현 활용

**예시:**
- 스페인어 수학 문제로 SFT
- 영어 수학 문제 성능도 향상

## 직접 선호도 최적화 (DPO)

### 개념

직접 선호도 최적화(Direct Preference Optimization, DPO)는 인간 선호도에 모델을 정렬하는 기술입니다.

**RLHF vs. DPO:**

| 측면 | RLHF | DPO |
|------|------|-----|
| **보상 모델** | 필요 | 불필요 |
| **복잡도** | 높음 | 낮음 |
| **학습 안정성** | 낮음 | 높음 |
| **구현** | 복잡 | 간단 |
| **확장성** | 어려움 | 쉬움 |

### DPO 작동 원리

**손실 함수:**
```
loss = -log(σ(β · log(R_policy / R_reference)))
```

- `R_policy`: 학습 중인 모델의 점수
- `R_reference`: 고정된 참조 모델의 점수
- `β`: KL 페널티 하이퍼파라미터
- `σ`: 시그모이드 함수

### 선호도 데이터 형식

```jsonl
{
  "prompt": "질문 텍스트",
  "chosen": "선호하는 답변",
  "rejected": "선호하지 않는 답변"
}
```

**예시: 포르투갈어 변형**
```jsonl
{
  "prompt": "Como posso renovar o meu passaporte?",
  "chosen": "Precisas de apresentar o passaporte...",  // PT-PT (유럽)
  "rejected": "Você precisa apresentar o passaporte..."  // PT-BR (브라질)
}
```

### NeMo-RL을 사용한 DPO

#### 설치 및 설정

```bash
cd /workspace/NeMo-RL/
pip install -e .
```

#### DPO 실행

```bash
uv run python examples/run_dpo.py \
  --config="config/dpo.yaml" \
  policy.optimizer.kwargs.lr=5.0e-6 \
  dpo.reference_policy_kl_penalty=0.1 \
  checkpointing.checkpoint_dir="./results" \
  +data.train_data_path="data/train.jsonl" \
  +data.val_data_path="data/val.jsonl"
```

#### 주요 하이퍼파라미터

- **학습률 (lr)**: 5e-6 ~ 1e-5 권장
- **KL 페널티 (β)**: 0.1 ~ 0.5
  - 낮음: 더 공격적인 정렬
  - 높음: 원본 모델에 가까움
- **배치 크기**: GPU 메모리에 맞게 조정

### DPO 모델 변환

```bash
# DCP → Hugging Face 형식 변환
uv run examples/convert_dcp_to_hf.py \
    --config=results/step_100/config.yaml \
    --dcp-ckpt-path=results/step_100/policy/weights \
    --hf-ckpt-path=results/step_100_hf
```

### 사용 사례

**1. 스타일 정렬**
- 공식적 ↔ 친근한 어조
- 간결함 ↔ 상세함
- 기술적 ↔ 비기술적

**2. 안전성 정렬**
- 유해 콘텐츠 거부
- 편향 감소
- 사실성 향상

**3. 언어 변형 정렬**
- 지역 방언 선호
- 문화적 적절성
- 현지화

## 실습 가이드

### 전체 워크플로우

```
1. 환경 설정
   ↓
2. 모델 변환 (HF → NeMo)
   ↓
3. 기준 평가
   ↓
4. CPT (선택적)
   ↓
5. SFT
   ↓
6. DPO (선택적)
   ↓
7. 최종 평가
```

### 1. 환경 설정 (10분)

```bash
# NeMo 설치
pip install nemo-toolkit[all]

# NeMo-RL 설치 (DPO용)
git clone https://github.com/NVIDIA/NeMo-RL.git
cd NeMo-RL && pip install -e .

# 추가 의존성
pip install datasets transformers tensorboard
```

### 2. 모델 변환 (15분)

```python
from nemo.collections import llm

# Hugging Face → NeMo 형식
llm.import_ckpt(
    model=llm.LlamaModel(llm.Llama32Config3B()),
    source="hf://meta-llama/Llama-3.2-3B-Instruct",
    output_path="nemo-checkpoints/Llama-3.2-3B-Instruct"
)
```

### 3. 지속적 사전 학습 (2-6시간)

**노트북**: `01_continued_pretraining.ipynb`

**단계:**
1. 도메인 텍스트 수집
2. JSONL 형식 변환
3. 토큰화
4. CPT 레시피 구성
5. 학습 실행
6. 지식 습득 평가

### 4. 지도 미세 조정 (1-3시간)

**노트북**: `02_supervised_finetuning.ipynb`

**단계:**
1. 입력-출력 쌍 준비
2. 데이터 분할 (train/val/test)
3. SFT 레시피 구성
4. 학습 실행
5. 작업 성능 평가

### 5. DPO 정렬 (30분-2시간)

**노트북**: `03_direct_preference_optimization_alignment.ipynb`

**단계:**
1. 선호도 데이터 수집
2. DPO 구성
3. 학습 실행
4. 모델 변환
5. 선호도 평가

### 통합 실습 노트북

`fine_tuning_practice.ipynb`는 전체 파이프라인을 하나의 노트북에서 실습할 수 있도록 구성되어 있습니다.

## 모범 사례

### 데이터 준비

**1. CPT 데이터**
- 고품질 도메인 텍스트 선별
- 충분한 규모 확보 (최소 수백만 토큰)
- 중복 제거
- 노이즈 필터링

**2. SFT 데이터**
- 다양한 예시 포함
- 고품질 답변 확보
- 명확한 입력-출력 쌍
- 적절한 데이터 분할

**3. DPO 데이터**
- 명확한 선호도 차이
- 일관된 기준 적용
- 다양한 시나리오 포함
- 품질 > 수량

### 하이퍼파라미터 튜닝

**1. 학습률**
- CPT: 1e-5 ~ 5e-5
- SFT: 5e-6 ~ 2e-5
- DPO: 1e-6 ~ 5e-6

**2. 배치 크기**
- GPU 메모리에 맞게 조정
- Gradient accumulation 활용
- Global batch size 일관성 유지

**3. 학습 스텝**
- 과적합 방지
- Validation loss 모니터링
- Early stopping 고려

### 평가

**1. 자동 평가**
- 벤치마크 (MMLU, GSM8K 등)
- 도메인 특화 메트릭
- 정량적 측정

**2. 정성 평가**
- LLM-as-a-Judge
- 인간 평가
- 실제 사용 사례 테스트

**3. 안전성 평가**
- 편향 테스트
- 유해 콘텐츠 검사
- 사실성 확인

## 문제 해결

### 일반적인 오류

#### Out of Memory (OOM)
**증상**: CUDA out of memory
**해결**:
```python
# 배치 크기 감소
config.data.micro_batch_size = 1
config.data.global_batch_size = 4

# Gradient checkpointing 활성화
config.model.config.activations_checkpoint_granularity = "selective"
```

#### 느린 학습
**증상**: 학습 속도 매우 느림
**해결**:
- 데이터 로더 워커 수 증가
- 혼합 정밀도 사용
- 분산 학습 활용

#### 파국적 망각
**증상**: 기존 능력 저하
**해결**:
- 학습률 감소
- 학습 스텝 감소
- 후속 SFT 수행

### 디버깅 팁

**1. TensorBoard 모니터링**
```bash
tensorboard --logdir=logs/ --port=6006
```

**2. 체크포인트 검증**
```python
# 중간 체크포인트 평가
for ckpt in checkpoints:
    evaluate(ckpt)
    compare_to_baseline()
```

**3. 데이터 검증**
```python
# 데이터셋 샘플 확인
with open("training.jsonl") as f:
    for i, line in enumerate(f):
        if i < 5:  # 처음 5개
            print(json.loads(line))
```

## 추가 리소스

### 공식 문서
- [NeMo Framework](https://docs.nvidia.com/nemo-framework/)
- [NeMo-RL GitHub](https://github.com/NVIDIA/NeMo-RL)
- [NeMo Run 문서](https://github.com/NVIDIA/NeMo-Run)

### 논문
- [ChipNeMo](https://arxiv.org/abs/2311.00176)
- [DPO 논문](https://arxiv.org/abs/2305.18290)
- [Llama 3.1](https://arxiv.org/abs/2407.21783)

### 튜토리얼
- [NeMo Tutorials](https://github.com/NVIDIA/NeMo/tree/main/tutorials)
- [DLI 워크숍](https://www.nvidia.com/en-us/training/)

## 라이선스

이 프로젝트는 NVIDIA NeMo Framework를 기반으로 합니다.    
자세한 라이선스 정보는 [NVIDIA 오픈 소스 라이선스](https://github.com/NVIDIA/NeMo/blob/main/LICENSE)를 참조하세요.
