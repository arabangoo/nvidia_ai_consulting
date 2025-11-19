# NVIDIA 모델 압축 가이드

## 목차
1. [소개](#소개)
2. [압축 기술 개요](#압축-기술-개요)
3. [양자화 (Quantization)](#양자화-quantization)
4. [프루닝 (Pruning)](#프루닝-pruning)
5. [지식 증류 (Knowledge Distillation)](#지식-증류-knowledge-distillation)
6. [추론 배포 및 성능 측정](#추론-배포-및-성능-측정)
7. [파일 구조](#파일-구조)
8. [실습 가이드](#실습-가이드)
9. [성능 비교](#성능-비교)

## 소개

### 동기

최근 생성형 AI의 발전으로 대규모 언어 모델(LLM)은 엄청난 발전을 보였지만, 수십억 개의 매개변수를 가진 LLM의 규모는 다음과 같은 실무 과제를 야기합니다:

- **배포 효율성**: 제한된 하드웨어 리소스에서의 배포
- **추론 지연**: 실시간 응답 시간 단축
- **운영 비용**: GPU 메모리 및 컴퓨팅 비용 절감

### 모델 압축의 목표

모델 압축 기술은 **원래 모델의 기능을 대부분 유지하면서도 프로덕션 설치 공간을 줄이는** 더 작은 버전의 모델을 만드는 것을 목표로 합니다.

```
원본 모델 (Large)
     ↓
압축 기술 적용 (Quantization, Pruning, Distillation)
     ↓
압축된 모델 (Small)
     ↓
성능 평가 (MMLU, 추론 속도)
```

### 학습 내용

이 가이드에서 다음을 배우게 됩니다:

- ✅ 최신 압축 기술 및 응용 분야
- ✅ 양자화, 프루닝, 증류를 오픈 소스 LLM에 적용하는 실습
- ✅ 압축 모델을 프로덕션 환경에 배포하는 모범 사례
- ✅ 압축 기술 평가 및 장단점 이해

### 사용 프레임워크

| 프레임워크 | 용도 | 주요 기능 |
|-----------|------|----------|
| **TensorRT-LLM** | 양자화 | FP8, INT8, INT4 양자화 |
| **TensorRT Model Optimizer** | 모델 최적화 | PTQ, QAT 지원 |
| **NeMo Framework** | 프루닝 & 증류 | 깊이/폭 프루닝, KD |
| **Triton Inference Server** | 추론 서빙 | 고성능 배포 |

### 기준 모델

이 가이드에서는 **Llama-3.2-3B** 모델을 사용합니다:

- 30억 개의 매개변수
- 단일 GPU에서 효율적인 실습 가능
- 산업 벤치마크에서 뛰어난 성능
- Meta의 오픈 소스 LLM

## 압축 기술 개요

### 기술 비교

| 기술 | 압축률 | 정확도 손실 | 학습 필요 | 적용 난이도 |
|------|--------|------------|-----------|------------|
| **양자화** | 중간-높음 | 낮음 | PTQ: 불필요<br>QAT: 필요 | 쉬움 |
| **프루닝** | 중간 | 중간 | 선택적 | 중간 |
| **지식 증류** | 높음 | 낮음 | 필수 | 어려움 |

### 압축 파이프라인

```
1. 기준 모델 (Llama-3.2-3B BF16)
   MMLU: ~75% | 크기: 100%
   ↓
2. 양자화 (FP8)
   MMLU: ~73% | 크기: 50%
   ↓
3. 프루닝 (25% 레이어 제거)
   MMLU: ~65% | 크기: 75%
   ↓
4. 지식 증류 (원본 모델 → 프루닝 모델)
   MMLU: ~72% | 크기: 75%
   ↓
5. 결합 (FP8 + 증류)
   MMLU: ~70% | 크기: 37.5%
```

## 양자화 (Quantization)

### 개념

양자화는 **수치 값의 정밀도를 줄여** 딥러닝 모델을 압축하는 기술입니다.

**변환 예시:**
- FP32 (32비트) → FP8 (8비트): 4배 압축
- FP32 (32비트) → INT8 (8비트): 4배 압축
- FP32 (32비트) → INT4 (4비트): 8배 압축

### 양자화의 이점

1. **모델 크기 대폭 감소**
   - FP32 → FP8: 75% 크기 감소
   - 메모리 사용량 최소화

2. **메모리 대역폭 감소**
   - 데이터 전송량 감소
   - 더 빠른 로딩 시간

3. **추론 속도 향상**
   - Hopper GPU (H100)에서 FP8 가속
   - 엣지 장치에서 효율적 실행

### 양자화 방법

#### 1. 학습 후 양자화 (PTQ)

**장점:**
- 재학습 불필요
- 빠른 적용
- 추가 데이터 불필요

**단점:**
- 정확도 손실 가능
- 활성화 양자화 어려움

#### 2. 양자화 인식 학습 (QAT)

**장점:**
- 정확도 보존
- 활성화 양자화 가능

**단점:**
- 재학습 필요
- 시간과 리소스 소모

### FP8 양자화

FP8은 **Hopper 아키텍처에서 지원**되는 혁신적인 양자화 솔루션입니다.

#### FP8 형식

<img src="./images/quantization_fp8.png" alt="FP8 Format" width="800">

| 형식 | 지수 | 가수 | 용도 | 특징 |
|------|------|------|------|------|
| **E4M3** | 4비트 | 3비트 | 순방향 패스 | 높은 정밀도 |
| **E5M2** | 5비트 | 2비트 | 역방향 패스 | 높은 동적 범위 |

#### FP8의 특성

- ✅ PTQ 및 QAT 모두 지원
- ✅ W8A8 (8비트 가중치, 8비트 활성화)
- ✅ 모델 정확도 보존
- ✅ 메모리 사용량 50% 감소
- ✅ 추론 속도 2배 향상

### 양자화 형식 비교

| 형식 | 비트 | 압축률 | 정확도 | GPU 지원 |
|------|------|--------|--------|----------|
| **FP32** | 32 | 1x | 100% | 모든 GPU |
| **BF16** | 16 | 2x | ~99.9% | Ampere+ |
| **FP16** | 16 | 2x | ~99.8% | 모든 GPU |
| **FP8** | 8 | 4x | ~98% | Hopper+ |
| **INT8** | 8 | 4x | ~97% | Turing+ |
| **INT4** | 4 | 8x | ~95% | Ampere+ |

### TensorRT-LLM 양자화 실습

#### 1. QNeMo 체크포인트 생성

```bash
python3 /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_ptq.py \
    model.restore_from_path=Llama-3.2-3B.nemo \
    trainer.precision=bf16 \
    quantization.algorithm=fp8 \
    export.decoder_type=llama \
    export.inference_tensor_parallel=1 \
    export.save_path=qnemo/llama-3.2-3b/fp8
```

**주요 매개변수:**
- `quantization.algorithm`: 양자화 형식 지정
  - `null`: 양자화 없음 (FP16/FP32)
  - `fp8`: FP8 양자화
  - `int8_sq`: INT8 Smooth Quantization
  - `int4_awq`: INT4 AWQ

#### 2. TensorRT 엔진 빌드

```bash
trtllm-build --checkpoint_dir qnemo/llama-3.2-3b/fp8 \
    --gemm_plugin auto \
    --output_dir engines/llama-3.2-3b/fp8 \
    --max_batch_size 4 \
    --max_input_len 2048 \
    --max_seq_len 2048 \
    --gather_context_logits
```

#### 3. 추론 실행

```bash
python3 /workspace/tensorrtllm_backend/tensorrt_llm/examples/run.py \
    --engine_dir engines/llama-3.2-3b/fp8 \
    --tokenizer_dir Llama-3.2-3B \
    --max_output_len 25 \
    --input_text "What is a graphics processing unit?"
```

### 양자화 선택 가이드

**FP8을 선택하는 경우:**
- Hopper GPU (H100) 사용
- 정확도 중요
- 최신 하드웨어 활용

**INT8을 선택하는 경우:**
- Turing 이상 GPU
- 균형 잡힌 성능 필요
- 광범위한 호환성

**INT4를 선택하는 경우:**
- 최대 압축 필요
- 엣지 장치 배포
- 정확도 손실 허용

## 프루닝 (Pruning)

### 개념

신경망 프루닝은 **네트워크의 불필요한 부분을 제거**하여 모델을 압축하는 고전적인 기술입니다 ([LeCun, 1989](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)).

**프루닝 대상:**
- 노드 (Nodes)
- 엣지 (Edges)
- 어텐션 헤드 (Attention Heads)
- 전체 레이어 (Layers)
- 채널 (Channels)

### 프루닝 방법

#### 1. 깊이 프루닝 (Depth Pruning)

**정의:** 네트워크의 특정 레이어를 제거

**특징:**
- 간단한 구현
- 빠른 적용
- 구조적 압축

**권장사항:**
- ✅ 상위 레이어 프루닝 (끝부분)
- ❌ 하위 레이어 프루닝 (시작부분) - 성능 손실 큼

**예시:**
```
원본 모델: 28개 레이어
   ↓
25% 프루닝: 레이어 21-27 제거
   ↓
프루닝된 모델: 21개 레이어 (75% 유지)
```

#### 2. 폭 프루닝 (Width Pruning)

**정의:** 레이어 내부의 뉴런, 헤드, 채널을 선별적으로 제거

**프로세스:**
1. 활성화 검사 (Multi-Head Attention, MLP, LayerNorm)
2. 중요도 계산 (L2-norm 사용)
3. 신경 아키텍처 검색 (NAS)
4. 최적 서브네트워크 선택

**Minitron 접근법:**
- NAS 없이 단순화
- 일반적인 차원만 고려
- 단일 샷 프루닝

### 깊이 프루닝 실습

#### NeMo를 사용한 레이어 제거

```bash
python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_drop_layers.py \
    --path_to_nemo "Llama-3.2-3B.nemo" \
    --path_to_save "Llama-3.2-3B-pruned.nemo" \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --gpus_per_node 1 \
    --drop_layers 21 22 23 24 25 26 27
```

**주요 매개변수:**
- `--path_to_nemo`: 원본 모델 경로
- `--path_to_save`: 저장 경로
- `--drop_layers`: 제거할 레이어 번호

### 프루닝 효과

| 모델 | 레이어 수 | 크기 | MMLU | 특징 |
|------|----------|------|------|------|
| **원본** | 28 | 100% | 75% | 기준선 |
| **10% 프루닝** | 25 | 89% | 73% | 경미한 손실 |
| **25% 프루닝** | 21 | 75% | 65% | 중간 손실 |
| **50% 프루닝** | 14 | 50% | 50% | 큰 손실 |

### 프루닝 전략

**1. 보수적 프루닝 (10-20%)**
- 용도: 프로덕션 환경
- 장점: 최소 정확도 손실
- 단점: 제한적 압축

**2. 균형 프루닝 (20-30%)**
- 용도: 일반적인 사용
- 장점: 적절한 압축 및 성능
- 단점: 증류 권장

**3. 공격적 프루닝 (30-50%)**
- 용도: 리소스 제약 환경
- 장점: 최대 압축
- 단점: 증류 필수

## 지식 증류 (Knowledge Distillation)

### 개념

지식 증류는 **큰 모델(TEACHER)의 지식을 작은 모델(STUDENT)에게 전달**하는 기술입니다 ([Hinton, 2015](https://arxiv.org/pdf/1503.02531)).

```
TEACHER (큰 모델)
     ↓ 지식 전이
STUDENT (작은 모델)
     ↓
향상된 STUDENT
```

### 증류 방법

#### 1. Hard Distillation

**프로세스:**
```
TEACHER → 출력 생성 (레이블) → STUDENT 학습
```

**특징:**
- 간단한 구현
- TEACHER의 출력만 사용
- 많은 데이터 필요

#### 2. Soft Distillation

**프로세스:**
```
TEACHER → 내부 상태 (로짓) → STUDENT 학습
```

**특징:**
- 풍부한 정보 전달
- 적은 데이터 필요
- 더 나은 성능

### KL 발산 (Kullback-Leibler Divergence)

지식 증류의 핵심 손실 함수:

$$
D_{\text{KL}}(P\parallel Q) = \sum_{x} P(x) \log\left(\frac{P(x)}{Q(x)}\right)
$$

- `P(x)`: TEACHER 분포
- `Q(x)`: STUDENT 분포
- 목표: KL 발산 최소화

### 증류 실습

#### 1. 환경 설정

**TEACHER:** `Llama-3.2-3B.nemo` (원본 모델)
**STUDENT:** `Llama-3.2-3B-pruned.nemo` (프루닝된 모델)
**데이터:** WikiText-103-v1

#### 2. 증류 실행

```bash
torchrun --nproc-per-node=1 \
    /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_distillation.py \
    name=llama_distill \
    exp_manager.exp_dir=./experiments/ \
    trainer.max_steps=100 \
    trainer.precision=bf16 \
    trainer.devices=1 \
    model.restore_from_path=Llama-3.2-3B-pruned.nemo \
    model.kd_teacher_restore_from_path=Llama-3.2-3B.nemo \
    model.nemo_path=Llama-3.2-3B-distilled.nemo \
    model.tokenizer.type=Llama-3.2-3B \
    model.global_batch_size=64 \
    model.optim.lr=7e-5 \
    model.optim.sched.min_lr=1e-5 \
    model.data.data_prefix={train:[1.0,wikidata/train],validation:[wikidata/val]}
```

**주요 하이퍼파라미터:**

| 매개변수 | 권장값 | 설명 |
|---------|--------|------|
| `max_steps` | 50-100 | 학습 반복 횟수 |
| `global_batch_size` | 64 | 전체 배치 크기 |
| `lr` | 5e-5 ~ 1e-4 | 초기 학습률 |
| `min_lr` | 1e-5 ~ 5e-5 | 최소 학습률 |
| `warmup_steps` | 2-5 | 워밍업 단계 |

#### 3. TensorBoard 모니터링

```bash
tensorboard --logdir=./experiments/ --port=8886
```

**확인 사항:**
- ✅ 손실 감소 추세
- ✅ 수렴 곡선
- ⚠️ 스파이크 또는 이상 패턴

### 증류 효과

| 모델 | MMLU | 크기 | 개선 |
|------|------|------|------|
| **원본** | 75% | 100% | - |
| **프루닝 (25%)** | 65% | 75% | -10%p |
| **증류 (프루닝 후)** | 72% | 75% | +7%p |

**관찰:**
- 증류로 프루닝 손실의 70% 복구
- 크기는 75%로 유지
- 최종 정확도 손실: 3%p만

### 증류 모범 사례

**1. 데이터셋 선택**
- 도메인 적합성 확인
- 충분한 다양성
- 품질 > 양

**2. TEACHER 선택**
- 가능한 큰 모델 사용
- 작업에 최적화된 모델
- 안정적인 성능

**3. 학습 전략**
- 적절한 학습률
- 충분한 학습 시간
- 정기적인 검증

## 추론 배포 및 성능 측정

### Triton Inference Server

#### 개요

Triton은 **NVIDIA의 고성능 추론 서버**로, 다양한 모델을 효율적으로 서빙합니다.

**특징:**
- 다중 모델 동시 서빙
- 동적 배치 처리
- 모델 버전 관리
- HTTP/gRPC 지원

#### PyTriton

**PyTriton**은 Python 환경에서 Triton을 쉽게 사용할 수 있는 인터페이스입니다.

```python
from nemo.deploy import DeployPyTriton

# TensorRT-LLM 엔진 로드
nm = DeployPyTriton(
    model=trt_llm_exporter,
    triton_model_name="llama",
    http_port=8000
)

# 서버 시작
nm.deploy()
nm.run()

# 서버 중지
nm.stop()
```

### 추론 성능 메트릭

#### 1. 지연 시간 (Latency)

**정의:** 단일 요청이 완료되는 데 걸리는 시간

**측정:**
```python
import time

start_time = time.time()
output = nq.query_llm(prompts=prompts, max_output_len=100)
latency = time.time() - start_time

print(f"Latency: {latency:.2f} seconds")
```

**목표:**
- 실시간 애플리케이션: < 100ms
- 대화형 애플리케이션: < 1s
- 배치 처리: < 10s

#### 2. 처리량 (Throughput)

**정의:** 초당 처리되는 요청 수

**측정:**
```python
num_requests = 10
batch_size = 4

start_time = time.time()
for _ in range(num_requests):
    output = nq.query_llm(prompts=prompts, max_output_len=100)
total_time = time.time() - start_time

throughput = num_requests * batch_size / total_time
print(f"Throughput: {throughput:.2f} requests/second")
```

**최적화 전략:**
- 배치 크기 증가
- 모델 병렬화
- 양자화 적용

### 벤치마킹 도구

#### GenAI-Perf

NVIDIA의 LLM 성능 벤치마킹 전문 도구:

```bash
genai-perf \
    --model llama-3.2-3b \
    --endpoint localhost:8000 \
    --concurrency 4 \
    --input-length 512 \
    --output-length 128
```

**측정 메트릭:**
- Token throughput (tokens/sec)
- Time to first token (TTFT)
- Inter-token latency (ITL)
- End-to-end latency

## 실습 가이드

### 전체 워크플로우

```
1. 환경 설정 및 기준 모델
   ↓
2. FP8 양자화 적용
   ↓
3. 깊이 프루닝 수행
   ↓
4. 지식 증류 실행
   ↓
5. Triton 서버 배포
   ↓
6. 성능 측정 및 비교
```

### 1. 환경 설정 (10분)

#### 필수 라이브러리

```bash
# TensorRT-LLM (양자화)
pip install tensorrt-llm

# NeMo Framework (프루닝 & 증류)
# Docker 사용 권장
docker pull nvcr.io/nvidia/nemo:24.07

# Triton Inference Server
pip install tritonclient[all]
```

#### 데이터 준비

```bash
# MMLU 벤치마크 다운로드
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf data.tar -C /workspace/data/mmlu

# WikiText-103 데이터셋
# 이미 전처리된 데이터 사용
```

### 2. 기준 모델 설정 (15분)

#### BF16 엔진 빌드

```bash
# QNeMo 체크포인트 생성
python3 /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_ptq.py \
    model.restore_from_path=Llama-3.2-3B.nemo \
    trainer.precision=bf16 \
    quantization.algorithm=null \
    export.decoder_type=llama \
    export.save_path=qnemo/llama-3.2-3b/bf16

# TensorRT 엔진 빌드
trtllm-build --checkpoint_dir qnemo/llama-3.2-3b/bf16 \
    --output_dir engines/llama-3.2-3b/bf16 \
    --max_batch_size 4
```

#### MMLU 평가

```bash
python3 /workspace/tensorrtllm_backend/tensorrt_llm/examples/mmlu.py \
    --engine_dir engines/llama-3.2-3b/bf16 \
    --test_trt_llm
```

**예상 결과:** MMLU ~75%

### 3. 양자화 적용 (20분)

#### FP8 양자화

```bash
# FP8 QNeMo 생성
python3 /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_ptq.py \
    model.restore_from_path=Llama-3.2-3B.nemo \
    trainer.precision=bf16 \
    quantization.algorithm=fp8 \
    export.save_path=qnemo/llama-3.2-3b/fp8

# 엔진 빌드
trtllm-build --checkpoint_dir qnemo/llama-3.2-3b/fp8 \
    --output_dir engines/llama-3.2-3b/fp8

# 평가
python3 mmlu.py --engine_dir engines/llama-3.2-3b/fp8
```

**예상 결과:** MMLU ~73% (2%p 손실)

### 4. 프루닝 수행 (10분)

```bash
# 25% 레이어 제거
python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_drop_layers.py \
    --path_to_nemo Llama-3.2-3B.nemo \
    --path_to_save Llama-3.2-3B-pruned.nemo \
    --drop_layers 21 22 23 24 25 26 27

# 평가
lm_eval --model nemo_lm \
    --model_args path="Llama-3.2-3B-pruned.nemo" \
    --tasks mmlu_subset \
    --batch_size 4
```

**예상 결과:** MMLU ~65% (10%p 손실)

### 5. 지식 증류 (30분)

```bash
# TensorBoard 시작
tensorboard --logdir=./experiments/ --port=8886 &

# 증류 실행
torchrun --nproc-per-node=1 \
    /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_distillation.py \
    model.restore_from_path=Llama-3.2-3B-pruned.nemo \
    model.kd_teacher_restore_from_path=Llama-3.2-3B.nemo \
    model.nemo_path=Llama-3.2-3B-distilled.nemo \
    trainer.max_steps=100

# 평가
lm_eval --model nemo_lm \
    --model_args path="Llama-3.2-3B-distilled.nemo" \
    --tasks mmlu_subset
```

**예상 결과:** MMLU ~72% (7%p 복구)

### 6. 추론 배포 (15분)

```python
# 증류된 모델 엔진 빌드
from nemo.export.tensorrt_llm import TensorRTLLM

trt_llm = TensorRTLLM(model_dir="engines/distilled/bf16")
trt_llm.export(
    nemo_checkpoint_path="qnemo/distilled/bf16",
    model_type="llama"
)

# Triton 서버 시작
from nemo.deploy import DeployPyTriton

nm = DeployPyTriton(model=trt_llm, triton_model_name="llama")
nm.deploy()
nm.run()
```

### 7. 성능 측정 (10분)

```python
from nemo.deploy.nlp import NemoQueryLLM
import time

nq = NemoQueryLLM(url="localhost", model_name="llama")
prompts = ["test"] * 4

# 지연 시간 측정
start = time.time()
nq.query_llm(prompts=prompts, max_output_len=100)
latency = time.time() - start

# 처리량 측정
num_requests = 10
start = time.time()
for _ in range(num_requests):
    nq.query_llm(prompts=prompts)
throughput = num_requests * len(prompts) / (time.time() - start)

print(f"Latency: {latency:.2f}s")
print(f"Throughput: {throughput:.2f} req/s")
```

## 성능 비교

### 최종 결과표

| 모델 | 정밀도 | 레이어 | MMLU | 크기 | 처리량 | 메모리 |
|------|--------|--------|------|------|--------|--------|
| **기준** | BF16 | 28 | 75% | 100% | 10 req/s | 6.4 GB |
| **양자화** | FP8 | 28 | 73% | 50% | 18 req/s | 3.2 GB |
| **프루닝** | BF16 | 21 | 65% | 75% | 13 req/s | 4.8 GB |
| **증류** | BF16 | 21 | 72% | 75% | 13 req/s | 4.8 GB |
| **FP8+증류** | FP8 | 21 | 70% | 37.5% | 24 req/s | 2.4 GB |

### 파레토 곡선 분석

```
정확도 (MMLU %)
 75 |  ● 기준
    |
 73 |     ● FP8
    |
 72 |        ● 증류
    |
 70 |           ● FP8+증류
    |
 65 |              ● 프루닝
    |________________
     10   15   20   25  처리량 (req/s)
```

**최적 선택:**

1. **최고 품질 필요**: 기준 모델 (BF16)
2. **균형**: FP8 양자화 또는 증류
3. **최대 효율**: FP8 + 증류 조합

### 압축 전략 가이드

#### 사용 사례별 권장

| 사용 사례 | 권장 기술 | 이유 |
|----------|----------|------|
| **클라우드 배포** | FP8 양자화 | 비용 절감, 높은 처리량 |
| **엣지 장치** | INT4 + 프루닝 | 메모리 제약, 경량화 |
| **실시간 서비스** | FP8 | 낮은 지연, 좋은 품질 |
| **배치 처리** | BF16 프루닝 | 정확도 우선 |
| **모바일** | INT4 + 증류 | 극도의 경량화 |

#### 압축 조합 전략

**1단계: 프루닝**
```
원본 → 25% 레이어 제거
크기: 100% → 75%
품질: 75% → 65%
```

**2단계: 증류**
```
프루닝 → 지식 증류
크기: 75% (유지)
품질: 65% → 72%
```

**3단계: 양자화**
```
증류 → FP8 양자화
크기: 75% → 37.5%
품질: 72% → 70%
```

**최종 결과:**
- 크기: 62.5% 감소
- 품질: 5%p 손실
- 속도: 2.4배 향상

## 모범 사례

### 양자화

**DO:**
- ✅ 하드웨어 지원 확인 (FP8은 Hopper 필요)
- ✅ 캘리브레이션 데이터 사용
- ✅ 정확도 검증
- ✅ 대표 데이터로 PTQ 수행

**DON'T:**
- ❌ 블라인드 양자화 (검증 없이)
- ❌ 부적절한 형식 선택
- ❌ 캘리브레이션 생략

### 프루닝

**DO:**
- ✅ 상위 레이어부터 프루닝
- ✅ 점진적 프루닝 (10%, 20%, 30%)
- ✅ 각 단계에서 평가
- ✅ 증류와 결합

**DON'T:**
- ❌ 하위 레이어 프루닝
- ❌ 과도한 프루닝 (50% 이상)
- ❌ 평가 없이 배포

### 지식 증류

**DO:**
- ✅ 큰 TEACHER 사용
- ✅ 충분한 학습 시간
- ✅ 적절한 학습률 선택
- ✅ TensorBoard 모니터링

**DON'T:**
- ❌ 작은 TEACHER 사용
- ❌ 조기 종료
- ❌ 과도한 학습률
- ❌ 검증 생략

## 문제 해결

### 일반적인 오류

#### 1. Out of Memory (OOM)

**증상:**
```
CUDA out of memory
```

**해결:**
```python
# 배치 크기 감소
--max_batch_size 2  # 4 대신

# 시퀀스 길이 감소
--max_input_len 1024  # 2048 대신

# Gradient checkpointing
trainer.activations_checkpoint_granularity=selective
```

#### 2. 낮은 정확도

**증상:**
- MMLU 점수가 예상보다 10%p 이상 낮음

**해결:**
1. 양자화 검증
2. 캘리브레이션 데이터 확인
3. 프루닝 비율 감소
4. 증류 학습 시간 증가

#### 3. 느린 추론

**증상:**
- 처리량이 예상보다 낮음

**해결:**
```bash
# 배치 크기 증가
--max_batch_size 8

# 양자화 적용
--quantization fp8

# GPU 병렬화
--tensor_parallelism 2
```

### 디버깅 팁

**1. 엔진 검증**
```python
# 간단한 추론 테스트
trt_llm.forward(["Hello, world!"])
```

**2. 메모리 모니터링**
```bash
nvidia-smi -l 1  # 1초마다 업데이트
```

**3. 로그 분석**
```bash
# TensorRT 로그 레벨 설정
export TRTLLM_LOG_LEVEL=INFO
```

## 추가 리소스

### 공식 문서

- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [NeMo Framework](https://docs.nvidia.com/nemo-framework/)
- [Triton Inference Server](https://github.com/triton-inference-server)

### 논문

- [FP8 Formats for Deep Learning](https://arxiv.org/pdf/2209.05433) (FP8)
- [SmoothQuant](https://arxiv.org/abs/2211.10438) (INT8)
- [AWQ](https://arxiv.org/abs/2306.00978) (INT4)
- [Minitron](https://arxiv.org/abs/2407.14679) (프루닝)
- [Distilling the Knowledge](https://arxiv.org/pdf/1503.02531) (KD)

### 튜토리얼

- [GenAI-Perf 벤치마킹](https://docs.nvidia.com/nim/benchmarking/llm/latest/)
- [DLI 추론 최적화 과정](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-18+V1)
- [Llama 3.1 Minitron 블로그](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/)

## 라이선스

이 프로젝트는 다음 NVIDIA 오픈 소스 프로젝트를 기반으로 합니다:

- [TensorRT-LLM License](https://github.com/NVIDIA/TensorRT-LLM/blob/main/LICENSE)
- [NeMo Framework License](https://github.com/NVIDIA/NeMo/blob/main/LICENSE)
- [Triton License](https://github.com/triton-inference-server/server/blob/main/LICENSE)
