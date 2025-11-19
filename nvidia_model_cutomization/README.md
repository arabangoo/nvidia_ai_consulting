# NVIDIA NeMo 모델 커스터마이제이션 가이드

## 목차
1. [소개](#소개)
2. [아키텍처 개요](#아키텍처-개요)
3. [환경 설정](#환경-설정)
4. [모델 평가](#모델-평가)
5. [모델 커스터마이제이션](#모델-커스터마이제이션)
6. [파일 구조](#파일-구조)
7. [실습 가이드](#실습-가이드)

## 소개

NVIDIA NeMo Microservices는 대규모 언어 모델(LLM)의 평가, 미세 조정 및 배포를 위한 엔터프라이즈급 플랫폼입니다. 이 가이드는 NeMo 마이크로서비스를 사용하여 LLM을 평가하고 커스터마이즈하는 전체 워크플로우를 다룹니다.

### 주요 기능
- **NIM (NVIDIA Inference Microservices)**: 최적화된 모델 추론 서빙
- **NeMo Evaluator**: 다양한 평가 메트릭을 활용한 모델 성능 측정
- **NeMo Customizer**: LoRA 기반 효율적인 미세 조정
- **MLflow 통합**: 실험 추적 및 시각화
- **Data Store & Entity Store**: 데이터셋 및 모델 아티팩트 관리

### 적용 시나리오
- 도메인 특화 모델 개발 (법률, 의료, 금융 등)
- 기존 LLM의 성능 평가 및 벤치마킹
- 효율적인 모델 미세 조정 및 배포
- 인컨텍스트 러닝 vs. 미세 조정 비교 분석

## 아키텍처 개요

### NeMo Microservices 구성 요소

```
                        Client Applications
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ↓               ↓               ↓
              NIM    ←→    NeMo Evaluator    NeMo Customizer
                │               │               │
                └───────────────┼───────────────┘
                                │
                                ↓
                    NeMo Data Store  ←→  NeMo Entity Store
                    (Datasets/Models)      (Metadata)
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ↓               ↓               ↓
           MLflow Tracking  Kubernetes      Storage Layer
```

### 주요 컴포넌트

| 컴포넌트 | 역할 | 네임스페이스 |
|---------|------|-------------|
| **NIM** | 모델 추론 서빙 | `llama3-2-3b-instruct` |
| **NeMo Evaluator** | 모델 평가 | `nemo-evaluator` |
| **NeMo Customizer** | 모델 미세 조정 (LoRA) | `nemo-customizer` |
| **NeMo Data Store** | 데이터셋 및 모델 저장 | `nemo-datastore` |
| **NeMo Entity Store** | 메타데이터 관리 | `nemo-entity-store` |
| **MLflow** | 실험 추적 | `mlflow` |

## 환경 설정

### 필수 요구사항

#### 하드웨어
- **GPU**: NVIDIA GPU (최소 A10, 권장 A100)
- **메모리**: 최소 32GB RAM
- **스토리지**: 최소 100GB

#### 소프트웨어
- Kubernetes 클러스터 (Minikube, K3s, 또는 프로덕션 클러스터)
- kubectl CLI
- Helm 3.x
- Docker
- Python 3.8+

### Minikube 설정

```bash
# Minikube 시작 (GPU 지원)
minikube start --driver=docker --gpus=all --memory=32768 --cpus=8

# NVIDIA Ingress Controller 설치
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
```

### NeMo Microservices 설치

#### 1. NeMo Data Store
```bash
helm install nemo-datastore nemo/nemo-datastore \
  --namespace nemo-datastore --create-namespace
```

#### 2. NeMo Entity Store
```bash
helm install nemo-entity-store nemo/nemo-entity-store \
  --namespace nemo-entity-store --create-namespace
```

#### 3. NIM (Llama 3.2 3B Instruct)
```bash
helm install llama3-2-3b-instruct nemo/nim-llm \
  --set image.repository=nvcr.io/nim/meta/llama-3.2-3b-instruct \
  --set image.tag=latest \
  --namespace llama3-2-3b-instruct --create-namespace
```

#### 4. NeMo Evaluator
```bash
helm install nemo-evaluator nemo/nemo-evaluator \
  --namespace nemo-evaluator --create-namespace
```

#### 5. NeMo Customizer
```bash
helm install nemo-customizer nemo/nemo-customizer \
  --namespace nemo-customizer --create-namespace
```

#### 6. MLflow (실험 추적)
```bash
helm install mlflow nemo/mlflow \
  --namespace mlflow --create-namespace
```

### API 키 설정

NVIDIA API 카탈로그 사용을 위한 API 키 설정:

```bash
export NGC_API_KEY="your-ngc-api-key"
export NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
```

> **참고**: NGC API 키는 [NVIDIA NGC](https://ngc.nvidia.com/)에서 무료로 발급받을 수 있습니다.

### 서비스 확인

```bash
# 모든 서비스 상태 확인
kubectl get pods --all-namespaces

# Ingress 확인
kubectl get ingress --all-namespaces
```

## 모델 평가

### 평가 방법론

NeMo Evaluator는 세 가지 주요 평가 접근 방식을 지원합니다:

#### 1. 벤치마크 평가

표준 데이터셋을 사용한 자동 평가:

**지원 벤치마크:**
- **GSM8K**: 초등학교 수학 문제
- **MMLU**: 다학제 선다형 문제
- **HellaSwag**: 상식 추론
- **TruthfulQA**: 사실성 평가
- **HumanEval**: 코딩 능력

**예시: GSM8K 평가**
```python
import requests

eval_config = {
    "type": "lm_harness",
    "name": "gsm8k_evaluation",
    "tasks": {
        "gsm8k": {
            "type": "gsm8k",
            "params": {
                "limit_samples": 50,
                "temperature": 0.0001,
                "max_tokens": 256
            }
        }
    }
}

response = requests.post(
    f"{eval_url}/v1/evaluations",
    json={
        "model": {"id": "meta/llama-3.2-3b-instruct"},
        "config": eval_config
    }
)
```

#### 2. 유사성 메트릭 평가

생성된 텍스트와 참조 텍스트 간의 유사도 측정:

**지원 메트릭:**
- **BLEU**: 정밀도 기반 n-gram 중첩
- **ROUGE**: 재현율 기반 n-gram 중첩
- **F1 Score**: BLEU와 ROUGE의 조화 평균
- **METEOR**: 형태소 분석 기반 평가
- **BERTScore**: 임베딩 기반 의미론적 유사도

**예시: ROUGE 평가**
```python
eval_config = {
    "type": "similarity_metrics",
    "name": "legal_summary_evaluation",
    "tasks": {
        "legal_summary": {
            "dataset": {
                "files_url": "hf://datasets/default/legal_dataset/test.jsonl"
            },
            "metrics": {
                "rouge": {"type": "rouge"},
                "bleu": {"type": "bleu"},
                "f1": {"type": "f1"}
            }
        }
    }
}
```

#### 3. LLM-as-a-Judge 평가

더 큰 LLM을 사용하여 응답 품질을 평가:

**평가 기준:**
- **Faithfulness**: 충실도 (사실 왜곡 없음)
- **Relevance**: 관련성
- **Coherence**: 일관성
- **Fluency**: 유창성
- **Helpfulness**: 유용성

**예시: Faithfulness 평가**
```python
faithfulness_prompt = """
System: You are an impartial judge assessing the faithfulness of an AI-generated response.

Evaluation Criteria:
1. Correctness: Does the response reflect facts from the ground truth?
2. No Hallucination: Does it introduce false information?
3. Completeness: Does it omit key facts?

Provide a score from 1 (poor) to 5 (perfect) with explanation.
"""

eval_config = {
    "type": "llm_as_judge",
    "name": "faithfulness_evaluation",
    "tasks": {
        "faithfulness": {
            "model": {
                "api_endpoint": {
                    "url": "https://integrate.api.nvidia.com/v1/chat/completions",
                    "model_id": "nvdev/nvidia/llama-3.3-nemotron-super-49b-v1",
                    "api_key": NGC_API_KEY
                }
            },
            "template": {
                "messages": [
                    {"role": "system", "content": faithfulness_prompt},
                    {"role": "user", "content": "Ground truth: {{ item.ground_truth }}\nAnswer: {{ sample.output_text }}"}
                ]
            }
        }
    }
}
```

### 인컨텍스트 러닝 (ICL) vs. Zero-Shot

#### Zero-Shot 평가
모델에 예제 없이 직접 질문:
```
질문: 주 경계를 넘나드는 원격 근무의 법적 의미는 무엇입니까?
```

#### Few-Shot (ICL) 평가
모델에 예제와 함께 질문:
```
예시 1:
질문: 직장 차별을 구성하는 요소는 무엇입니까?
답변: 직장 차별은 보호되는 특성을 기반으로 개인을 다르게 대우할 때 발생합니다...

예시 2:
질문: 저작권법이 소프트웨어에 어떻게 적용됩니까?
답변: 소프트웨어 저작권은 원본 코드를 보호하여...

질문: 주 경계를 넘나드는 원격 근무의 법적 의미는 무엇입니까?
```

### MLflow로 결과 추적

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:30090")
mlflow.set_experiment("legal_domain_evaluation")

with mlflow.start_run(run_name="zero_shot_baseline"):
    mlflow.log_param("approach", "zero_shot")
    mlflow.log_param("model", "llama-3.2-3b-instruct")
    mlflow.log_metric("rouge_score", 0.45)
    mlflow.log_metric("bleu_score", 0.38)
```

## 모델 커스터마이제이션

### LoRA (Low-Rank Adaptation)란?

LoRA는 전체 모델을 재학습하지 않고 효율적으로 미세 조정하는 기술입니다:

#### 기존 미세 조정 vs. LoRA

| 특성 | 전체 미세 조정 | LoRA |
|------|--------------|------|
| 학습 가능 파라미터 | 100% (7B 모델: 7B) | ~1% (7B 모델: 70M) |
| GPU 메모리 | 매우 높음 | 낮음 |
| 학습 시간 | 매우 길음 | 짧음 |
| 스토리지 | 전체 모델 사이즈 | 어댑터만 (수백 MB) |
| 다중 작업 지원 | 각 작업마다 전체 모델 | 어댑터 교체 |

#### LoRA 작동 원리

```python
# 기존 가중치 (고정)
W = W_original  # 예: 4096 x 4096

# LoRA 어댑터 (학습 가능)
LoRA = A @ B
# A: 4096 x 8 (rank=8)
# B: 8 x 4096

# 최종 가중치
W_final = W_original + LoRA
```

**장점:**
- 학습 가능 파라미터 수 99% 감소
- GPU 메모리 사용량 대폭 감소
- 여러 어댑터를 쉽게 교체 가능
- 기존 모델 품질 유지

### 데이터셋 준비

#### 1. 데이터셋 형식

NeMo Customizer는 JSONL 형식을 사용:
```jsonl
{"prompt": "질문 텍스트", "completion": "답변 텍스트"}
{"prompt": "다음 질문 텍스트", "completion": "다음 답변 텍스트"}
```

#### 2. 데이터셋 분할

```python
from datasets import Dataset

# 데이터셋 로드 및 전처리
ds = load_dataset("your-dataset")

# 훈련/검증/테스트 분할
ds = ds.train_test_split(test_size=0.5)
training_data = ds["train"]

ds_test = ds["test"].train_test_split(test_size=0.1)
validation_data = ds_test["train"]
test_data = ds_test["test"]
```

#### 3. Data Store에 업로드

```python
import requests
from huggingface_hub import HfApi

api = HfApi(endpoint=datastore_url)

# 데이터셋 업로드
api.upload_file(
    path_or_fileobj="training.jsonl",
    path_in_repo="training/training.jsonl",
    repo_id="default/legal_dataset",
    repo_type="dataset"
)
```

### LoRA 미세 조정 실행

#### 1. 하이퍼파라미터 설정

```python
training_config = {
    "config": "meta/llama-3.2-3b-instruct",
    "dataset": {"name": "legal_dataset"},
    "hyperparameters": {
        "training_type": "sft",  # Supervised Fine-Tuning
        "finetuning_type": "lora",
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "lora": {
            "adapter_dim": 8,      # LoRA rank
            "adapter_dropout": 0.1,
        },
    },
}
```

**주요 하이퍼파라미터:**
- **adapter_dim (rank)**: 어댑터 차원 (8, 16, 32 등)
  - 낮음 (4-8): 빠르지만 성능 제한적
  - 중간 (16-32): 균형잡힌 선택
  - 높음 (64+): 더 나은 성능, 더 많은 리소스
- **learning_rate**: 학습률 (1e-5 ~ 1e-3)
- **epochs**: 전체 데이터셋을 학습하는 횟수

#### 2. 미세 조정 작업 시작

```python
response = requests.post(
    f"{customizer_url}/v1/customization/jobs",
    headers={"Content-Type": "application/json"},
    json=training_config
)

job_id = response.json()["id"]
print(f"Training job started: {job_id}")
```

#### 3. 학습 모니터링

```python
import time

status = "running"
while status in {"initializing", "running", "created"}:
    job = requests.get(
        f"{customizer_url}/v1/customization/jobs/{job_id}"
    ).json()

    status = job["status"]
    print(f"Status: {status}")

    if status == "completed":
        output_model = job["output_model"]
        print(f"Training completed! Model: {output_model}")
        break

    time.sleep(30)
```

#### 4. 학습 메트릭 분석

```python
import matplotlib.pyplot as plt
import pandas as pd

metrics = job['status_details']['metrics']['metrics']

# Training Loss
train_loss = pd.DataFrame(metrics['train_loss'])
val_loss = pd.DataFrame(metrics['val_loss'])

plt.figure(figsize=(10, 6))
plt.plot(train_loss['step'], train_loss['value'], label='Training Loss')
plt.scatter(val_loss['step'], val_loss['value'], color='red', label='Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.show()

print(f"Final Training Loss: {train_loss['value'].iloc[-1]:.4f}")
print(f"Best Validation Loss: {val_loss['value'].min():.4f}")
```

### 미세 조정된 모델 배포

#### 방법 1: 런타임 어댑터 로드

```python
# 기존 NIM에 LoRA 어댑터 로드
request_body = {
    "model": output_model,  # 예: "default/llama-3.2-3b-legal-lora@cust-xyz"
    "prompt": "법률 질문에 대한 제목을 생성하세요...",
    "temperature": 0.2,
    "max_tokens": 75,
}

response = requests.post(
    f"{nim_url}/v1/completions",
    json=request_body
)

print(response.json()["choices"][0]["text"])
```

#### 방법 2: 새 NIM 컨테이너 시작

```bash
# LoRA 어댑터를 마운트하여 새 NIM 시작
helm install llama3-2-3b-legal nemo/nim-llm \
  --set lora.adapter="default/llama-3.2-3b-legal-lora@cust-xyz" \
  --namespace llama3-2-3b-legal --create-namespace
```

### 성능 비교

| 접근 방식 | ROUGE 점수 | BLEU 점수 | F1 점수 | 설정 시간 | 학습 시간 |
|----------|-----------|----------|---------|----------|----------|
| Zero-Shot | 0.35 | 0.28 | 0.31 | 없음 | 없음 |
| ICL (Few-Shot) | 0.52 | 0.45 | 0.48 | 분 단위 | 없음 |
| LoRA | 0.68 | 0.61 | 0.64 | 시간 단위 | 2-6시간 |

## 파일 구조

```
nvidia_model_cutomization/
├── README.md                          # 본 가이드
├── model_customization_practice.ipynb # 통합 실습 노트북
│
├── 01-NIM-Evaluation.ipynb            # NIM 평가 기초
├── 02-Evaluator_notebook.ipynb        # NeMo Evaluator 사용법
├── 03-Customizer.ipynb                # LoRA 미세 조정
├── 04-Free-Resources.ipynb            # 리소스 정리
│
├── nemo_ms_uninstall.sh               # NeMo MS 제거 스크립트
├── allow-dns-mlflow.yaml              # MLflow DNS 설정
└── index2.pdf                         # 참고 문서
```

## 실습 가이드

### 워크플로우 개요

```
1. 환경 설정 및 확인
   ↓
2. 기본 모델 평가 (Zero-Shot)
   ↓
3. ICL 평가 (Few-Shot)
   ↓
4. 데이터셋 준비 및 업로드
   ↓
5. LoRA 미세 조정
   ↓
6. 미세 조정 모델 평가
   ↓
7. 결과 비교 및 분석
```

### 단계별 가이드

#### 1. 환경 설정 (10분)

```bash
# NeMo 마이크로서비스 설치 스크립트 실행
bash /dli/configs/minikube_configs/nemo_ms_install.sh

# 서비스 상태 확인
kubectl get pods --all-namespaces
kubectl get ingress --all-namespaces

# MLflow 포트 포워딩
kubectl -n mlflow port-forward service/mlflow-tracking 30090:80
```

#### 2. 기본 평가 (30분)

**노트북**: `01-NIM-Evaluation.ipynb` 또는 `model_customization_practice.ipynb` 섹션 1-2

- NIM 상태 확인
- Zero-Shot 추론 테스트
- 벤치마크 평가 (GSM8K)
- LLM-as-a-Judge 평가

#### 3. 데이터셋 기반 평가 (45분)

**노트북**: `02-Evaluator_notebook.ipynb` 또는 `model_customization_practice.ipynb` 섹션 3-4

- 법률 데이터셋 다운로드 및 전처리
- Zero-Shot 모드 평가
- ICL (Few-Shot) 모드 평가
- 유사성 메트릭 비교
- MLflow로 결과 시각화

#### 4. LoRA 미세 조정 (2-3시간)

**노트북**: `03-Customizer.ipynb` 또는 `model_customization_practice.ipynb` 섹션 5-6

- 데이터셋 Data Store 업로드
- 학습 하이퍼파라미터 설정
- 미세 조정 작업 시작
- 학습 진행 모니터링
- 학습 메트릭 분석

#### 5. 미세 조정 모델 평가 (30분)

**노트북**: `model_customization_practice.ipynb` 섹션 7

- 미세 조정 모델 배포
- 성능 테스트
- Zero-Shot/ICL/LoRA 비교
- 최종 결과 분석

#### 6. 리소스 정리

**노트북**: `04-Free-Resources.ipynb`

```bash
# 네임스페이스 삭제
kubectl delete namespace llama3-2-3b-instruct
kubectl delete namespace nemo-customizer
kubectl delete namespace nemo-evaluator
```

### 통합 실습 노트북

`model_customization_practice.ipynb`는 전체 워크플로우를 하나의 노트북에서 단계별로 실습할 수 있도록 구성되어 있습니다.

**구성:**
1. 환경 설정 및 확인
2. NIM 기본 평가
3. 데이터셋 준비
4. Zero-Shot vs. ICL 비교
5. LoRA 미세 조정
6. 결과 비교 및 분석
7. 종합 평가

## 모범 사례

### 평가

1. **다양한 메트릭 사용**
   - 정량적 메트릭 (ROUGE, BLEU)
   - 정성적 평가 (LLM-as-a-Judge)
   - 인간 평가 (프로덕션 전)

2. **적절한 테스트 세트 크기**
   - 개발: 50-100 샘플
   - 검증: 200-500 샘플
   - 프로덕션: 1000+ 샘플

3. **평가 일관성 유지**
   - Temperature를 낮게 설정 (0.0-0.2)
   - 동일한 프롬프트 형식 사용
   - 재현 가능한 시드 설정

### 미세 조정

1. **데이터 품질**
   - 고품질 예제 선별
   - 다양성 확보
   - 노이즈 제거

2. **하이퍼파라미터 튜닝**
   - 작은 LoRA rank로 시작 (8-16)
   - Learning rate 실험 (1e-5 ~ 1e-3)
   - Validation loss 모니터링

3. **과적합 방지**
   - Early stopping 사용
   - Validation set 활용
   - Dropout 적절히 설정

4. **실험 관리**
   - MLflow로 모든 실험 추적
   - 재현 가능한 설정 문서화
   - 버전 관리

### 배포

1. **점진적 롤아웃**
   - 먼저 소수의 사용자에게 테스트
   - A/B 테스팅 수행
   - 모니터링 강화

2. **리소스 최적화**
   - 배치 추론 사용
   - 캐싱 전략 구현
   - 로드 밸런싱

3. **모델 버전 관리**
   - 각 어댑터 버전 추적
   - 롤백 계획 수립
   - 변경 사항 문서화

## 문제 해결

### 일반적인 오류

#### NIM이 시작되지 않음
**증상**: Pod이 CrashLoopBackOff 상태
**원인**: GPU 리소스 부족 또는 이미지 pull 실패
**해결**:
```bash
# GPU 할당 확인
kubectl describe node | grep -A 5 "Allocated resources"

# Pod 로그 확인
kubectl logs -n llama3-2-3b-instruct <pod-name>

# 이미지 pull secret 확인
kubectl get secrets -n llama3-2-3b-instruct
```

#### 미세 조정 작업 실패
**증상**: 작업 상태가 "failed"
**원인**: 잘못된 데이터셋 형식 또는 메모리 부족
**해결**:
```python
# 작업 로그 확인
job = requests.get(f"{customizer_url}/v1/customization/jobs/{job_id}").json()
print(job["status_details"]["error_message"])

# 데이터셋 형식 검증
with open("training.jsonl") as f:
    for line in f:
        data = json.loads(line)
        assert "prompt" in data and "completion" in data
```

#### MLflow 연결 실패
**증상**: MLflow UI에 접근 불가
**원인**: 포트 포워딩 미설정
**해결**:
```bash
# 포트 포워딩 재시작
kubectl -n mlflow port-forward service/mlflow-tracking 30090:80 --address 0.0.0.0
```

#### 평가 작업 느림
**증상**: 평가가 매우 오래 걸림
**원인**: 큰 테스트 세트 또는 복잡한 메트릭
**해결**:
- 테스트 세트 크기 줄이기 (`limit_samples` 사용)
- 배치 크기 증가
- 병렬 처리 활성화

## 추가 리소스

### 공식 문서
- [NeMo Microservices 문서](https://developer.nvidia.com/docs/nemo-microservices/)
- [NeMo Framework 문서](https://docs.nvidia.com/nemo-framework/)
- [NVIDIA NIM 문서](https://docs.nvidia.com/nim/)
- [MLflow 문서](https://mlflow.org/docs/latest/index.html)

### 튜토리얼 및 가이드
- [NeMo Customizer 빠른 시작](https://developer.nvidia.com/docs/nemo-microservices/customization/quickstart.html)
- [NeMo Evaluator 가이드](https://developer.nvidia.com/docs/nemo-microservices/evaluation/index.html)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)

### 관련 도구
- [Hugging Face Hub](https://huggingface.co/)
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [vLLM](https://github.com/vllm-project/vllm)

## 라이선스

이 프로젝트는 NVIDIA NeMo Microservices를 기반으로 합니다. 
자세한 라이선스 정보는 [NVIDIA 소프트웨어 라이선스 계약](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)을 참조하세요.


