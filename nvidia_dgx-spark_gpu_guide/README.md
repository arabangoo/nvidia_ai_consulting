# NVIDIA DGX Spark 개발자·엔지니어 가이드

이 가이드는 NVIDIA DGX Spark를 처음 다루는 개발자가 박스를 여는 순간부터 로컬 LLM·이미지 생성·AI 에이전트까지 직접 구축할 수 있도록 작성되었습니다.

> **실전 경험 + 공식 스펙 기반 가이드**: 이 문서는 DGX OS 적응, JupyterLab 첫 추론, 정밀도/양자화, 원격 접속(Tailscale + NVIDIA Sync), Docker 기반 Open WebUI, Ollama + gpt-oss, ComfyUI, 그리고 OpenClaw + Nemotron 로컬 에이전트까지 전 과정을 다룹니다.

## 목차

1. [DGX Spark 소개](#dgx-spark-소개)
2. [하드웨어 사양](#하드웨어-사양)
3. [DGX OS 첫 부팅과 환경 이해](#dgx-os-첫-부팅과-환경-이해)
4. [DGX Dashboard와 JupyterLab으로 첫 추론](#dgx-dashboard와-jupyterlab으로-첫-추론)
5. [정밀도와 양자화: FP16/FP8/FP4의 이해](#정밀도와-양자화-fp16fp8fp4의-이해)
6. [개발 환경 구축: Docker vs venv](#개발-환경-구축-docker-vs-venv)
7. [원격 접속: SSH + Tailscale + NVIDIA Sync](#원격-접속-ssh--tailscale--nvidia-sync)
8. [로컬 LLM 챗봇: Open WebUI + Ollama + gpt-oss](#로컬-llm-챗봇-open-webui--ollama--gpt-oss)
9. [이미지 생성: ComfyUI와 LoRA](#이미지-생성-comfyui와-lora)
10. [로컬 vs API, 모델 자산 구조 이해](#로컬-vs-api-모델-자산-구조-이해)
11. [AI 에이전트 구축: OpenClaw + Nemotron](#ai-에이전트-구축-openclaw--nemotron)
12. [실전 디버깅: 1M 컨텍스트의 함정](#실전-디버깅-1m-컨텍스트의-함정)
13. [에이전트 아키텍처: 게이트웨이와 서비스 구조](#에이전트-아키텍처-게이트웨이와-서비스-구조)
14. [모델 메모리 관리와 모델 큐레이션](#모델-메모리-관리와-모델-큐레이션)
15. [성능 특성과 최적화](#성능-특성과-최적화)
16. [문제 해결](#문제-해결)
17. [빠른 시작 체크리스트](#빠른-시작-체크리스트)
18. [자주 묻는 질문](#자주-묻는-질문)
19. [참고 자료](#참고-자료)

---

## DGX Spark 소개

DGX Spark는 NVIDIA가 만든 손바닥만 한 "개인용 AI 슈퍼컴퓨터"입니다. **GB10 Grace Blackwell Superchip**을 탑재하고, **128GB 통합 메모리(unified memory)**를 CPU와 GPU가 함께 사용합니다. 1리터 우유팩 정도 크기의 본체에 5년 전 데이터센터급 AI 성능이 들어 있습니다.

### 누구를 위한 장비인가

- 클라우드 AI는 써봤지만 **내 데이터를 내 손 안에 두고** 추론·실험하고 싶은 개발자
- 70B~120B급 대형 모델을 **한 대의 장비에서 프로토타이핑·파인튜닝**하고 싶은 엔지니어
- CUDA / Blackwell 생태계를 로컬에서 직접 경험하려는 AI 입문자
- 의뢰인·환자·내부 데이터처럼 **외부 전송이 곤란한 데이터**를 다루는 전문직·기업

### 핵심 가치와 한계

DGX Spark의 정체성은 **"대역폭을 희생하고 대용량 통합 메모리를 택한 장비"**입니다.

- **강점**: 128GB 통합 메모리 덕분에 단일 디스크리트 GPU(VRAM 24~96GB)에는 올라가지 않는 대형 모델을 그대로 적재할 수 있습니다. 두 대를 직접 연결하면 약 405B 파라미터급(FP4)까지 다룰 수 있습니다.
- **한계**: 메모리 대역폭이 약 273GB/s로, RTX 5090(약 1,790GB/s) 같은 디스크리트 GPU 대비 낮습니다. 이는 단일 스트림 토큰 생성(decode) 속도를 제한하는 가장 큰 요인입니다.
- **요약**: 빠른 단일 사용자 생성보다는 **대형 모델 프로토타이핑, 파인튜닝, 배치/동시 서빙, 지연에 너그러운 작업**에 적합합니다.

### 판매 형태와 가격

동일한 GB10 하드웨어가 NVIDIA Founders Edition과 여러 OEM(GIGABYTE AI TOP ATOM, Acer, ASUS, Dell, HP, Lenovo, MSI) 버전으로 판매됩니다. 2025년 10월 15일 출하 시작.

| 구성 | 가격(참고) |
|------|-----------|
| NVIDIA Founders Edition | 출시 약 $3,999 → 이후 약 $4,699로 인상 |
| GIGABYTE AI TOP ATOM (1TB Gen4 SSD) | 약 $3,499 |
| GIGABYTE AI TOP ATOM (4TB Gen4 SSD) | 약 $3,899 |
| GIGABYTE AI TOP ATOM (4TB Gen5 SSD) | 약 $3,999 |

원화 기준 대략 500만~700만원 선이며, SKU와 환율에 따라 달라집니다.

---

## 하드웨어 사양

### GB10 Grace Blackwell Superchip

GB10은 MediaTek가 설계한 Arm CPU 다이와 NVIDIA Blackwell GPU 다이를 하나의 패키지(TSMC 3nm)에 통합한 칩입니다.

- **CPU**: 20코어 Arm — 고성능 Cortex-X925 10개 + 고효율 Cortex-A725 10개
- **GPU**: Blackwell 세대, 48 SM / **6,144 CUDA 코어**, 5세대 Tensor Core, 4세대 RT 코어, NVENC/NVDEC 각 1개
- **아키텍처**: ARM64 (aarch64)

### 메모리 (정체성을 결정하는 부분)

- **용량**: 128GB LPDDR5x 통합 메모리, CPU와 GPU가 일관성(coherent) 있게 공유
- **대역폭**: 256비트 인터페이스, 약 **273GB/s** (16채널)
- **NVLink-C2C (CPU↔GPU 다이 간 연결)**: 약 600GB/s 양방향, NVIDIA 표현으로 "PCIe Gen5의 5배"

> 통합 메모리의 의미: 일반 GPU 시스템은 모델을 CPU 메모리에서 GPU 메모리(VRAM)로 복사해야 하지만, GB10은 같은 메모리 풀을 공유하므로 그 복사가 필요 없습니다. 텍스트 모델, 이미지 모델, 텍스트 인코더, VAE, LoRA 여러 개를 동시에 메모리에 띄워두고 교체할 수 있습니다.

### AI 연산 성능

- **최대 1 PFLOP(1,000 TFLOPS) / 1,000 TOPS — FP4 정밀도 + 희소성(sparsity) 기준**
- 스펙시트의 "1 petaFLOP"은 4비트(FP4) 기준 수치입니다. FP8/FP16으로 가면 단계마다 대략 절반씩 낮아집니다.
- 5세대 Tensor Core는 FP4/FP6/FP8을 지원합니다.

### 저장공간 / 네트워크 / 전력

- **저장공간**: 1TB 또는 4TB NVMe M.2 SSD (자체 암호화 지원)
- **네트워크**:
  - ConnectX-7 SmartNIC, QSFP 2포트, 최대 200GbE → **두 대의 DGX Spark를 직접 연결**해 더 큰 모델을 클러스터로 구동 가능
  - 10GbE RJ-45 1포트, Wi-Fi 7, Bluetooth 5.4
- **전력**: GB10 TDP 140W, 외장 어댑터 240W
- **크기/무게**: 약 150 × 150 × 50.5 mm, 약 1.2kg, 약 1리터
- **포트**: USB-C 4개, HDMI 2.1a 1개

### 소프트웨어 스택

- **OS**: NVIDIA DGX OS (Ubuntu 24.04 기반), DGX Dashboard / NVIDIA Sync 관리 도구 내장
- **CUDA 13.0**, Python 3.12
- **컴퓨트 능력(compute capability)**: 12.1 (sm_121) — 데이터센터 Blackwell의 sm_100과는 다른 컨슈머/엣지 Blackwell
- **사전 설치 스택**: CUDA, cuDNN, TensorRT, NCCL, NGC Docker 컨테이너, PyTorch/TensorFlow/JAX 지원, NIM 마이크로서비스, NGC 카탈로그 접근

> **중요 — 소프트웨어 성숙도 주의**: sm_121은 출시 초기 지원이 미성숙했습니다. PyTorch는 CUDA 13 빌드 기준 **2.9 이상**이 필요하고, vLLM은 초기에 aarch64 sm_121 미지원(GitHub 이슈 #36821), flash_attn/FlashInfer 일부 커널은 sm_80 경로만 제공해 sm_121과 호환되지 않았습니다. 항상 **최신 툴체인**을 사용하세요.

---

## DGX OS 첫 부팅과 환경 이해

### 윈도우가 아니라 DGX OS다

전원을 켜면 윈도우가 아니라 리눅스가 뜹니다. 그냥 우분투가 아니라 **NVIDIA DGX OS**입니다.

> DGX OS = 우분투를 기반으로 NVIDIA가 자기네 AI 하드웨어에 맞게 GPU 드라이버, CUDA, AI 소프트웨어 스택을 미리 세팅해서 출고한 버전입니다. 일반 사용자가 며칠 걸려 세팅하는 환경이 켜자마자 준비되어 있습니다.

### 첫 번째 함정: 업데이트 알림을 함부로 누르지 말 것

리눅스를 처음 만지면 업데이트 알림이 잔뜩 뜹니다. 하지만 일반 우분투 업데이트는 커널이나 GPU 드라이버를 함께 업그레이드하는데, 이것이 NVIDIA가 세팅해둔 환경과 충돌할 수 있습니다. 최악의 경우 다음 부팅 때 GPU가 인식되지 않거나 AI 소프트웨어가 동작하지 않습니다.

> **NVIDIA 공식 권장**: 업데이트는 반드시 **DGX Dashboard를 통해서만** 수행하세요. 브라우저에서 `http://localhost:11000`으로 접속하면 NVIDIA가 검증한 패키지만 안전하게 설치됩니다.

### 터미널은 친구가 되어야 한다

AI 작업 튜토리얼의 대부분은 터미널 명령어로 설명됩니다.

- **일상적인 파일 정리**: 파일 탐색기(GUI) 사용
- **AI 작업·프로그램 설치**: 터미널 필요

### 리눅스 기본 감각

- **대소문자 구분**: `Documents`와 `documents`는 다른 폴더입니다. 명령어 옵션도 `-s`와 `-S`가 완전히 다릅니다.
- **여러 줄 붙여넣기 주의**: 터미널은 한 줄이 실패해도 멈추지 않고 다음 줄을 실행합니다. 붙여넣은 직후 빨간 에러 메시지가 흘러갔는지 확인하는 습관을 들이세요.
- **`rm -rf` 경계**: `rm`(삭제) + `-r`(폴더 내용까지) + `-f`(확인 없이 강제)는 휴지통을 거치지 않는 영구 삭제입니다. 특히 `rm -rf /`는 시스템 전체를 지웁니다. 경로를 항상 두 번 확인하세요.

---

## DGX Dashboard와 JupyterLab으로 첫 추론

### 1단계: DGX Dashboard 열기

1. Super 키(윈도우 키)를 눌러 검색창에 `dgx` 입력 후 **DGX Dashboard** 실행, 또는 브라우저에서 `http://localhost:11000` 접속
2. DGX OS 계정으로 로그인
3. 왼쪽에 시스템 메모리/GPU 상태, 오른쪽에 JupyterLab 패널이 보입니다

### 2단계: JupyterLab 시작

오른쪽 JupyterLab 패널에서 "Start"를 클릭하면 `Starting → Preparing → Running` 순으로 상태가 바뀝니다. 첫 실행은 5~10분(파이썬 패키지 100여 개 자동 설치), 두 번째부터는 1~2분 정도 걸립니다. `Running`이 되면 "Open in Browser"를 클릭합니다.

### 3단계: GPU 인식 확인

새 노트북을 만들고 다음 코드를 셀에 붙여넣은 뒤 `Shift + Enter`로 실행합니다.

```python
import torch
print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0))
print("GPU 메모리:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```

기대 출력:

```text
CUDA 사용 가능: True
GPU 이름: NVIDIA GB10
GPU 메모리: 128.5183373376 GB
```

> 분홍색 "CUDA Capability 12.1" 경고가 함께 떠도 무해한 안내입니다. 12.1은 Blackwell 아키텍처의 신호로, RTX 30 시리즈가 8.x, H100이 9.x, Blackwell이 12.x입니다.

### 4단계: Stable Diffusion XL로 첫 이미지 생성

```python
import warnings
warnings.filterwarnings('ignore', message='.*cuda capability.*')

from diffusers import DiffusionPipeline
import torch
from PIL import Image
from IPython.display import display

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    variant="fp16" if dtype == torch.float16 else None,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "a cozy modern reading nook with a big window, soft natural light, photorealistic"
negative_prompt = "low quality, blurry, distorted, text, watermark"

height = 1024
width = 1024
steps = 30
guidance = 7.0

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=guidance,
    height=height,
    width=width,
)

image = result.images[0]
display(image)
image.save("my_first_image.png")
print("저장 완료: my_first_image.png")
```

생성 파라미터의 의미:

- `height`, `width`: 출력 픽셀 수 (1024 × 1024이 SDXL 기본)
- `steps`: 노이즈 제거 반복 횟수 (많을수록 품질↑ 시간↑, 30이 무난)
- `guidance`: 프롬프트 충실도 (7.0이 표준)

**처음 실행 시간**: 모델 약 7GB 다운로드(15~25분, 최초 1회) + 이미지 생성(약 11초). 모델은 한 번 받으면 디스크에 저장되어 다음부터는 다운로드가 생략됩니다.

> 프롬프트는 **영어**로 작성하세요. SDXL은 영어 데이터로 학습되어 한국어 프롬프트는 품질이 떨어집니다.

작업이 끝나면 노트북을 저장(`Ctrl + S`)하고 DGX Dashboard에서 JupyterLab을 "Stop"합니다. 메모리의 모델은 사라지지만 노트북과 이미지 파일은 보존됩니다.

---

## 정밀도와 양자화: FP16/FP8/FP4의 이해

위 코드의 이 한 줄이 DGX Spark 운영의 핵심 개념을 담고 있습니다.

```python
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
```

"GPU가 있으면 float16, 없으면 float32." 직관과 반대 같지만, **GPU가 좋을수록 오히려 작은 정밀도를 쓰는 것이 정답**입니다.

### 정밀도 형식 비교

| 형식 | 메모리 | 정확도 |
|------|--------|--------|
| float32 (FP32) | 4바이트 | 약 7자리 |
| float16 (FP16) | 2바이트 | 약 3자리 |

### 작은 정밀도를 쓰는 세 가지 이유

1. **메모리에 들어가야 한다**: 같은 모델도 정밀도에 따라 용량이 달라집니다.
   - SDXL: FP32 약 14GB → FP16 약 7GB
   - LLaMA 70B: FP32 약 280GB(초과!) → FP16 약 140GB(빠듯) → int4 약 40GB(여유)
   - 큰 모델일수록 작은 정밀도가 선택이 아닌 필수입니다.
2. **Tensor Core가 작은 정밀도에 최적화**: GB10의 Tensor Core는 FP16/FP8/FP4 계산에 압도적으로 빠릅니다. FP32로만 쓰는 것은 "페라리를 시속 50km로 모는 것"과 같습니다. "1 PetaFLOP" 스펙도 FP4 기준입니다.
3. **정확도 손실이 작업에 따라 무의미**: 과학 계산·시뮬레이션은 FP32가 필요하지만, AI 추론·이미지 생성은 FP16/FP8/FP4로 충분합니다(사람 눈으로 차이를 구분하기 어려움).

> `else torch.float32`가 있는 이유: 이 코드는 DGX Spark 전용이 아니기 때문입니다. CPU는 float16을 잘 다루지 못하므로(CPU 회로는 float32 최적화), GPU가 없는 환경에서는 float32로 폴백합니다. AI 코드에서 자주 보게 될 안전망 패턴입니다.

### 업계 정밀도 추세

```text
2010년대 초   FP32  (표준)
2018년경      FP16  (대중화)
2022년경      BF16  (새 표준)
2023년~       int8, int4 양자화
2024~2025년   FP8, FP4
```

정밀도를 낮추면 메모리 절약 + 계산 속도 향상 + 전력 절감이 동시에 일어납니다. DGX OS에 양자화 라이브러리 **bitsandbytes**가 미리 깔려 있는 것도 이 때문입니다.

---

## 개발 환경 구축: Docker vs venv

DGX Spark에서는 두 종류의 격리(isolation)를 상황에 맞게 씁니다.

### Docker = 별채 짓기 (OS 단위 격리)

운영체제 수준에서 통째로 격리합니다. 무겁지만 안전하고, 한 컨테이너가 망가져도 다른 곳에 영향이 없습니다. 24/7 백그라운드로 띄울 서비스(예: Open WebUI + Ollama)에 적합합니다.

### venv = 방 하나만 빌리기 (파이썬 패키지 단위 격리)

파이썬 패키지만 격리합니다. 가볍고 빠르며, **GPU 직접 접근이 더 자연스럽습니다**(Docker는 `--gpus=all` 옵션이 필요). 프로젝트별로 PyTorch 버전이 충돌(A는 2.3, B는 2.5 요구)하는 문제를 폴더 단위로 분리해 해결합니다.

### venv로 GPU용 PyTorch 설치

```bash
mkdir -p ~/ai-projects
cd ~/ai-projects
python3 -m venv comfyui-env
source comfyui-env/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

> **핵심 옵션 `--index-url`**: 기본 PyPI 대신 NVIDIA가 CUDA용으로 빌드한 휠 저장소를 지정합니다. 위 예시의 `cu130`은 CUDA 13.0 휠입니다. 이 옵션을 빠뜨리면 설치는 되지만 GPU를 못 써서 CPU로 동작합니다(이미지 1장에 수 분). 항상 본인 DGX OS의 CUDA 버전에 맞는 휠을 지정하세요.

가상환경에서 나올 때는 `deactivate`. 가상환경이 망가지면 해당 폴더만 지우면 되고(`rm -rf ~/comfyui-env`), 시스템 파이썬은 무사합니다.

### 데이터와 도구를 분리하라

핵심 원칙: **도구는 가볍게, 무거운 데이터(모델)는 분리해 영구 보관**합니다.

- venv 폴더에는 파이썬 패키지만 둡니다.
- 수 GB 단위 모델은 별도 폴더(`ComfyUI/models/` 등)나 Docker 볼륨에 둡니다.
- 그러면 venv를 새로 만들거나 도구를 재설치해도 모델을 다시 받지 않아도 됩니다.

---

## 원격 접속: SSH + Tailscale + NVIDIA Sync

집/사무실에 둔 본체를 외부에서 안전하게 사용하는 구성입니다.

### NVIDIA Sync는 화면 미러링이 아니다

NVIDIA가 DGX Spark에 공식 권장하는 조합은 **NVIDIA Sync + Tailscale**입니다.

- AnyDesk/TeamViewer는 화면 픽셀 전체를 중계해 데이터가 크고 네트워크에 따라 끊깁니다.
- NVIDIA Sync는 본체 안의 **개별 앱(터미널, JupyterLab, DGX Dashboard, IDE)을 원격 PC에 따로 띄웁니다**. 전송 데이터는 명령어와 텍스트뿐이라 가볍고 안정적입니다. 실제 GPU 연산은 본체가 수행합니다.

### Tailscale 설치 (본체)

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh
tailscale ip -4
```

> **`curl -fsSL ... | sh` 해부**: `-f`(에러 시 조용히 실패) `-s`(진행률 숨김) `-S`(진짜 에러는 표시) `-L`(리다이렉트 따라가기). 파이프(`|`)로 받은 스크립트를 셸에 바로 실행합니다. 내용 확인 없이 실행되므로 **반드시 신뢰할 수 있는 공식 도메인에서만** 사용하세요. 더 안전한 분리 방식:
> ```bash
> curl -fsSL https://tailscale.com/install.sh -o install.sh
> cat install.sh    # 내용을 직접 확인한 뒤
> sh install.sh
> ```

### SSH 서비스 활성화

DGX Spark는 SSH가 설치는 되어 있지만 **비활성 상태로 출고**됩니다(보안 정책). 상태를 확인하고 켭니다.

```bash
sudo systemctl status ssh
# Active: inactive (dead) 로 나오면 아래로 활성화
sudo systemctl enable --now ssh
```

> `enable`(부팅 시 자동 시작) + `--now`(지금 즉시 켜기)를 한 번에. 백그라운드에서 외부 요청을 기다리는 모든 서비스(SSH, Ollama, OpenClaw 등)는 `systemctl`로 관리합니다.

### 원격 PC / 폰 설정

- **사무실 윈도우 PC**: Tailscale 클라이언트 + NVIDIA Sync 설치, 같은 계정 로그인 → 본체 Tailscale IP·사용자명·비밀번호로 연결
- **폰**: Tailscale 앱 + Termius(SSH 클라이언트) 설치 → 본체 Tailscale IP로 접속

### 잊지 말아야 할 설정

- **Tailscale 키 만료 비활성화**: `login.tailscale.com/admin/machines`에서 각 기기 메뉴 → "Disable key expiry". NVIDIA Sync는 자체 Tailscale 노드를 따로 만드는데, 이 노드의 인증 키가 며칠~몇 주 주기로 만료되어 `ssh: handshake failed: EOF` 에러가 날 수 있습니다. 특히 본체 노드에 꼭 적용하세요(본체 키가 풀리면 직접 가서 재인증해야 함).
- **BIOS 자동 부팅**: 정전 시 자동 부팅 옵션을 켜두면 헤드리스 서버로 안정적입니다.
- **로그 읽는 습관**: 문제가 생기면 로그가 시간순 일기장입니다. 예) `Broken pipe`(연결 끊김) → `keepalive failed` → `handshake failed: EOF`(재접속 반복 실패).

---

## 로컬 LLM 챗봇: Open WebUI + Ollama + gpt-oss

본체 위에 ChatGPT 같은 채팅 UI를 Docker로 띄웁니다.

### Docker 권한 설정 (최소 권한 원칙)

```bash
docker ps
# permission denied 에러가 나면 아래로 해결
sudo usermod -aG docker $USER
newgrp docker
```

> `sudo usermod -aG docker $USER`는 사용자를 docker 그룹에 영구 등록합니다. 하지만 현재 터미널 세션은 로그인 시 한 번만 그룹 정보를 읽으므로, `newgrp docker`(즉시 적용) 또는 재로그인이 필요합니다. 매번 `sudo`를 붙이는 것보다 전용 그룹 가입이 **최소 권한 원칙**에 맞습니다(피해 반경을 docker 영역으로 한정).

### Open WebUI 컨테이너 실행

```bash
docker run -d -p 8080:8080 --gpus=all \
  --restart always \
  -v open-webui:/app/backend/data \
  -v open-webui-ollama:/root/.ollama \
  --name open-webui ghcr.io/open-webui/open-webui:ollama
```

각 옵션의 의미:

- `-d`: 백그라운드(detached) 실행
- `-p 8080:8080`: 호스트 8080 포트를 컨테이너 8080 포트에 연결
- `--gpus=all`: **핵심**. 컨테이너에 GPU 사용 권한 부여. 빠뜨리면 LLM이 CPU로 돌아 약 100배 느려집니다.
- `--restart always`: 컨테이너가 죽거나 본체가 재부팅돼도 자동 재시작
- `-v open-webui:/app/backend/data`: 대화 기록·사용자 설정 볼륨
- `-v open-webui-ollama:/root/.ollama`: 모델 저장 볼륨

> **이미지 태그 함정**: GPU가 있으니 `:cuda`가 빠를 것 같지만 아닙니다. `:ollama` 태그만 Ollama를 포함해 LLM을 GPU로 돌릴 수 있습니다. `:cuda`/`:main`은 Open WebUI만 들어 있어 LLM을 구동하지 못합니다(부속 기능 모델만 처리).

### 볼륨(Volume) 개념

컨테이너를 지우면 안의 데이터가 사라집니다. 볼륨은 컨테이너 **밖**에 만든 별도 저장 공간(외장하드)으로, 컨테이너가 폭발해도 살아남습니다. 앱 데이터와 모델을 별도 볼륨으로 분리해두면 앱을 업데이트해도 수십 GB 모델이 보존됩니다.

### 첫 접속과 모델 다운로드

브라우저에서 `http://localhost:8080` 접속 → 로컬 계정 회원가입(외부 전송 없음, 비밀번호 분실 시 컨테이너 재생성 필요).

좌상단 "Select a model" → 모델명 입력 → "Pull ... from Ollama.com"으로 다운로드합니다.

```text
gpt-oss:20b   →  약 12GB,  다운로드 약 10분
```

> **gpt-oss**는 GPT + Open Source Software로, 2025년 8월 OpenAI가 일부 모델 가중치를 공개한 오픈 모델입니다. 인터넷 없이 무료로 본체에서 구동됩니다.

### Ollama 이해

Ollama는 OpenAI와 무관한 별개 회사(2023년 설립, Y Combinator 출신, 창업자들이 Docker/Kitematic 출신)입니다. 그래서 명령 형식이 Docker와 동일합니다.

```bash
docker pull ubuntu:22.04      # Docker
ollama pull llama3.2:7b       # Ollama
```

Ollama Library에는 Google Gemma, Alibaba Qwen(한국어 우수), DeepSeek, OpenAI gpt-oss, Mistral, Microsoft Phi, NVIDIA Nemotron 등이 있습니다.

### 모델의 3가지 상태

모델은 SSD(디스크)에 영구 저장되며, 호스트 경로는 다음과 같습니다(볼륨 사용 시).

```text
/var/lib/docker/volumes/open-webui-ollama/_data/
```

1. **디스크에만 있음** (받기만 한 상태)
2. **메모리에 로드됨** (채팅 시작 시. 첫 응답이 5~10초 느린 이유)
3. **자동 언로드** (보통 5분간 미사용 시 메모리에서 내려감, 디스크엔 그대로)

DGX Spark의 통합 메모리 덕분에 CPU↔GPU 메모리 복사가 없어 로드가 효율적입니다. 응답 속도는 gpt-oss:20b 기준 약 40~60 tok/s입니다.

### 폰에서 호출

폰에서 Tailscale ON → 브라우저로 `http://[본체 Tailscale IP]:8080` 접속 → "홈 화면에 추가"로 PWA 설치하면 앱처럼 쓸 수 있습니다. 데이터 통제권은 본인 서버에 있습니다.

### 정리/롤백 — Docker 3층 구조

| 계층 | 예시 | 비유 |
|------|------|------|
| 이미지(Image) | `ghcr.io/open-webui/open-webui:ollama` | 설치 파일 |
| 컨테이너(Container) | `open-webui` | 실행 인스턴스 |
| 볼륨(Volume) | `open-webui-ollama` | 데이터 저장소(외장하드) |

| 원하는 것 | 명령어 | 결과 |
|-----------|--------|------|
| 한 모델만 삭제 | `ollama rm <모델>` | 모델 크기만큼 확보 |
| 컨테이너 잠시 중지 | `docker stop open-webui` | 멈춤만 |
| 컨테이너 삭제 | `docker rm open-webui` | 볼륨은 보존됨 |
| 이미지 삭제 | `docker rmi <이미지>` | 약 5GB 확보 |
| 볼륨 삭제 | `docker volume rm <볼륨>` | 모델 전부 삭제 |

> `docker rm`으로 컨테이너를 지워도 모델은 사라지지 않습니다(Docker가 의도적으로 분리 설계). 모델까지 지우려면 `docker volume rm`을 명시해야 합니다.

---

## 이미지 생성: ComfyUI와 LoRA

ComfyUI는 박스(노드)를 줄로 연결해 파이프라인을 구성하는 노드 기반 이미지 생성 GUI입니다. venv로 격리해 설치합니다(위 [개발 환경 구축](#개발-환경-구축-docker-vs-venv) 참고).

### 실행과 접속

```bash
cd ~/ai-projects/ComfyUI
source ~/ai-projects/comfyui-env/bin/activate
python main.py
```

브라우저에서 `http://localhost:8188` 접속. 폰 등 외부에서 접속하려면 `--listen 0.0.0.0`을 추가합니다.

```bash
python main.py --listen 0.0.0.0
```

> `--listen` 기본값은 본인 컴퓨터 내부 접속만 허용합니다. `0.0.0.0`은 모든 네트워크에서 접속을 허용하므로, Tailscale과 함께 폰 브라우저에서 `http://[본체 Tailscale IP]:8188`로 접속할 수 있습니다.

### 기본 워크플로우 구조 (확산 모델)

기본 템플릿은 3단계 공장 구조입니다.

```text
재료 준비 (회색 4개)              가공 (노란색)      마무리 (청록색)
  Load Checkpoint  ─┐
  긍정 프롬프트     ─┤
  부정 프롬프트     ─┼──→  KSampler  ──→  VAE Decode  ──→  이미지 저장
  빈 캔버스(Latent) ─┘
```

- **Load Checkpoint**: 본 모델을 메모리에 로드
- **긍정/부정 프롬프트**: 그릴 것 / 피할 것. 영어 권장
- **빈 캔버스(Empty Latent Image)**: 크기 결정 (SD 1.5 = 512×512, SDXL = 1024×1024)
- **KSampler**: 노이즈로 가득한 화면에서 노이즈를 걷어내며 그림을 드러냄(확산 모델 원리)
- **VAE Decode**: 압축된 잠재 공간(latent) 데이터를 사람이 보는 RGB 이미지로 변환. 압축 공간에서 작업하는 이유는 메모리·시간을 크게 아끼기 위함
- **이미지 저장**: `ComfyUI/output/`에 PNG로 자동 저장

### KSampler 주요 파라미터

- **Steps**: 노이즈 제거 반복 횟수 (기본 20~30)
- **CFG (Classifier Free Guidance)**: 프롬프트를 얼마나 강하게 따를지 (너무 높으면 부자연스러움)
- **Sampler / Scheduler**: 노이즈 제거 알고리즘 (예: euler, dpm++ / normal). 처음엔 기본값
- **Seed**: 노이즈 시작점. 같은 시드 = 같은 결과 재현

### LoRA로 능력 추가

LoRA(Low-Rank Adaptation)는 거대한 본 모델은 그대로 두고, 작은 부가 파일(수십~수백 MB)로 스타일·능력을 추가하는 기법입니다(인스타 필터/게임 DLC 비유).

| 종류 | 하는 일 | 예시 |
|------|---------|------|
| 스타일 | 그림체 변경 | 픽사, 애니, 픽셀아트, 수채화 |
| 캐릭터 | 특정 인물 생성 | 영화 캐릭터, 본인 얼굴 |
| 개념·능력 | 새 기능 추가 | 멀티앵글, 특정 자세, 손가락 정확도 |
| 속도 | 적은 스텝으로 빠르게 | Lightning, Turbo (4스텝) |
| 품질 | 디테일 향상 | Detail Tweaker |

사용법:

- 파일을 `ComfyUI/models/loras/`에 넣으면 자동 인식
- 워크플로우에 `Load LoRA` 노드를 추가: `Load Checkpoint → Load LoRA → KSampler → ...`
- 강도는 보통 0.7~1.0
- 다운로드처: Civitai.com

> **호환 함정**: 같은 본 모델용 LoRA끼리만 호환됩니다. SD 1.5용 LoRA를 SDXL에 쓰면 동작하지 않습니다. 다운로드 페이지의 "Base Model" 표시를 확인하세요.

### 직접 LoRA 학습

강아지 사진 20~30장으로도 학습이 가능합니다. DGX Spark 기준 학습 시간 30분~2시간, 결과 파일 200~500MB. 사전 설치된 `bitsandbytes`가 이를 뒷받침합니다.

### 모델 디렉터리 구조

```text
ComfyUI/
├── main.py
├── models/
│   ├── checkpoints/       SD / SDXL / FLUX 본체
│   ├── loras/             LoRA 파일
│   ├── vae/               VAE
│   ├── text_encoders/     텍스트 인코더(CLIP)
│   └── diffusion_models/  메인 확산 모델
└── output/                생성 이미지(PNG)
```

> SDXL의 한계(손가락 오류, 한국어 프롬프트 약함)를 넘으려면 **FLUX.1** 같은 최신 모델(약 24GB)을 쓸 수 있습니다. 일반 GPU에선 빠듯하지만 DGX Spark의 128GB 통합 메모리에는 거뜬히 올라갑니다.

---

## 로컬 vs API, 모델 자산 구조 이해

### 로컬과 API의 차이

ComfyUI는 로컬 모델과 클라우드 API 모델을 같은 노드 화면에서 다룰 수 있습니다(예: Google의 Gemini 2.5 Flash Image, 코드네임 "Nano Banana"는 API 키가 필요).

| 구분 | 로컬(Local) | API(클라우드) |
|------|-------------|---------------|
| 실행 위치 | 본체 GPU | 외부 서버 |
| 모델 위치 | 디스크에 다운로드 | 매번 인터넷 호출 |
| 비용 | 전기세만 | 호출당 과금 |
| 프라이버시 | 본인만 봄 | 외부 서버에 데이터 전송 |
| API 키 | 불필요 | 필수 |
| 예시 | SD 1.5, SDXL, FLUX, gpt-oss | Nano Banana, GPT Image, Veo |

DGX Spark의 가치는 **프라이버시가 중요한 작업을 로컬에서 비용 없이** 무제한 반복할 수 있다는 데 있습니다.

### 같은 모델, 다른 파일

같은 "Qwen 2.5 VL 7B"라도 도구마다 재포장한 별개 파일이라 공유되지 않습니다.

```text
"Qwen 2.5 VL 7B"
├─ Qwen 팀 원본 학습      → 원본 가중치
├─ ComfyUI 팀 가공        → qwen_2.5_vl_7b_fp8_scaled.safetensors (이미지 생성용 텍스트 인코더)
└─ llama.cpp 팀 가공      → .gguf (Ollama가 쓰는 채팅용 풀 LLM)
```

차이가 나는 이유:

1. **파일 형식**: ComfyUI는 `.safetensors`, Ollama는 `.gguf` — 서로 못 읽음
2. **정밀도**: ComfyUI용은 FP8, Ollama용은 보통 Q4(4비트)
3. **용도**: ComfyUI는 모델의 텍스트 이해 부분만 떼어 인풋으로, Ollama는 모델 전체를 채팅에 사용

### 이미지 1장에 필요한 4가지 자산

```text
1. 메인 모델 (diffusion_models)        그림을 만드는 본체        수십 GB
2. 텍스트 인코더 (text_encoders/CLIP)  프롬프트를 이해 가능 형태로  수 GB
3. VAE                                 압축 공간 ↔ 실제 이미지     수백 MB
4. LoRA (선택)                         스타일/능력/속도 추가       각 100MB~1GB
```

일반 GPU(예: RTX 4090, VRAM 24GB)는 이 4개를 한 번에 올리기 빠듯하지만, DGX Spark의 128GB 통합 메모리는 모두 띄워두고 교체할 수 있습니다.

---

## AI 에이전트 구축: OpenClaw + Nemotron

채팅(텍스트 생성기)에서 에이전트(상태를 관리하고 파일/명령/API를 다루는 비서)로 넘어가는 단계입니다.

### 챗봇 vs 에이전트

- **챗봇**: 같은 채팅창 안에서만 기억, 외부 접근 없음(격리), 다단계 작업은 매번 사용자가 지시
- **에이전트**: 세션 간 영구 기억, 파일/명령/API 호출 가능, 스스로 계획해 실행. 능력이 큰 만큼 보안 책임도 따름

> **보안 경고**: 에이전트는 본체에서 명령을 실행할 수 있습니다. 의뢰인·내부 데이터가 있는 메인 PC가 아니라, **학습·실험용으로 격리된 DGX Spark**에서 운영하고, 위험 능력은 신뢰할 수 있는 것만 하나씩 켜세요.

### 모델: Nemotron-3-Nano-30B-A3B

NVIDIA의 오픈소스 에이전트용 LLM입니다(2025년 12월 출시).

| 부분 | 의미 |
|------|------|
| Nemotron | NVIDIA 오픈소스 LLM 패밀리 |
| 3 | 3세대 |
| Nano | 패밀리 내 작은 사이즈 |
| 30B | 총 파라미터 300억 |
| A3B | Active 3B — 토큰당 실제 사용 30억 |

- **MoE(Mixture of Experts)**: 전문가 128명 중 토큰당 6명만 활성화 + 공유 전문가 1명. 메모리엔 30B 전부 대기, 계산은 3B만 → 빠름
- **하이브리드 아키텍처**: Mamba 23 + MoE 23 + Transformer 6. 긴 문맥을 효율 처리
- **Reasoning Mode**: 입력 → 내부 사고(분석 → 시도 → 자기 검증) → 출력. 다단계 판단이 필요한 에이전트 작업에 적합

### 컨텍스트 윈도우

1 토큰 ≈ 영어 3~4글자, 한국어 1~2글자.

| 표기 | 토큰 수 | 분량 |
|------|---------|------|
| 4K | 4,096 | 짧은 글 |
| 32K | 32,768 | 30~50페이지 |
| 128K | 131,072 | 짧은 책 한 권 |
| 1M | 1,048,576 | 두꺼운 책 |

> Nemotron은 이론적으로 1M까지 지원한다고 선언하지만, 실제로 1M으로 잡으면 메모리를 잡아먹고 버그가 발생합니다(아래 [1M 컨텍스트의 함정](#실전-디버깅-1m-컨텍스트의-함정) 참고). **32K가 균형 잡힌 선택**입니다.

### 설치

```bash
curl -fsSL https://openclaw.ai/install.sh | bash
```

설치 마법사의 주요 선택:

- **Skills (도구 권한)**: 처음엔 **No**. 파일 접근·명령 실행·외부 API 권한을 주므로, 웹 UI에서 나중에 하나씩 켭니다.
- **Hooks (자동 실행 코드)**: **모두 켜기**. 메모리 읽기, 시작 메시지, 로그 등 위험하지 않은 동작으로, 에이전트가 "살아있는 느낌"을 줍니다.
  - boot-md, bootstrap-extra-files, command-logger, compaction-notifier, session-memory
- **모델 제공자**: "Skip for now" (아래에서 직접 연결)

> OpenClaw는 Node.js로 제작되어 패키지 매니저로 npm을 씁니다. DGX Spark 안에는 Ollama(Go), ComfyUI(Python), OpenClaw(Node.js) 세 생태계가 공존하는데, Docker/venv/시스템 환경 격리 덕분에 충돌하지 않습니다.

### 두뇌(Ollama)를 호스트에 직접 설치

에이전트가 직접 호출하려면 Ollama를 컨테이너가 아니라 호스트에 설치해야 합니다(Docker로 격리된 Ollama는 바깥 OpenClaw가 닿기 어려움).

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nemotron-3-nano:30b      # 약 24GB
```

연결 전 직접 테스트(컨텍스트를 32K로 지정):

```bash
ollama run nemotron-3-nano:30b
>>> /set parameter num_ctx 32768
>>> 안녕, 자기소개 해줄래?
```

### 에이전트에 두뇌 연결

```bash
ollama launch openclaw --model nemotron-3-nano:30b
```

이 한 줄이 OpenClaw 게이트웨이를 띄우고, Nemotron을 두뇌로 연결하며, 설정 파일 `~/.openclaw/openclaw.json`을 자동 생성합니다. 웹 UI는 `http://127.0.0.1:18789`에서 열립니다.

> **함정**: 자동 생성된 설정에 1M 컨텍스트라는 시한폭탄이 섞여 들어갑니다. 다음 절에서 디버깅 과정을 설명합니다.

---

## 실전 디버깅: 1M 컨텍스트의 함정

DGX Spark 운영에서 마주칠 수 있는 전형적인 디버깅 사례입니다.

### 증상

첫 메시지 전송 후 약 30초 뒤 에러:

```text
run error: 500 model failed to load, this may be due to
resource limitations or an internal error
```

128GB 메모리에 24GB 모델인데 "자원 부족"? 에러 메시지를 그대로 믿지 말고 실제 로그를 확인합니다.

### 1단계: 로그 확인

```bash
sudo journalctl -u ollama -e --no-pager | tail -50
```

핵심 줄:

```text
GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX) failed
```

> INT_MAX는 32비트 정수 최대값(약 21억)입니다. 텐서 계산이 그 한계를 넘어 죽은 것이므로, 메모리 부족이 아닙니다.

### 2단계: 알려진 버그 검색

`ollama nemotron-3-nano load failed`로 검색하면 GitHub 이슈를 찾을 수 있습니다(이슈 #14269 계열): "이 모델은 512K 컨텍스트로는 작동하지만 1M으로 올리면 크래시함." Ollama가 1M으로 띄우려다 텐서가 INT_MAX를 초과해 사망한 것입니다.

### 3단계: Modelfile로 컨텍스트 고정 (1차 우회)

```bash
cat > ~/Modelfile-nemotron-32k << EOF
FROM nemotron-3-nano:30b
PARAMETER num_ctx 32768
EOF

ollama create nemotron-32k -f ~/Modelfile-nemotron-32k
```

> Modelfile은 Docker의 Dockerfile과 같은 개념입니다. 베이스 모델을 참조만 하고 메타데이터(컨텍스트 설정)만 추가하므로 디스크에 24GB가 추가되지 않습니다.

### 4단계: 그래도 실패 — 설정 우선순위의 교훈

새 모델로 재연결해도 상태바가 여전히 `tokens ?/1.0m`. 설정 파일을 확인합니다.

```bash
cat ~/.openclaw/openclaw.json | grep contextWindow
```

```text
"contextWindow": 1048576,
"contextWindow": 1048576,
```

근본 원인:

1. Nemotron 메타데이터가 "최대 1M 지원"을 선언(이론적 한계치)
2. `ollama launch openclaw`가 설정 자동 생성 시 그 **최대치를 그대로 contextWindow에 박음**
3. 실행 시 1M으로 로드 시도 → INT_MAX 초과 → 크래시

> **핵심 교훈**: Modelfile은 "모델 기본값"을 바꾸지만, OpenClaw는 매 호출마다 자기 설정을 들고 와 덮어씁니다. **결정권은 `openclaw.json`이 가집니다.** 두 곳에서 같은 값을 다르게 정의하면 애플리케이션 설정이 이깁니다.

### 5단계: JSON 직접 수정 후 재시작

```bash
sed -i 's/"contextWindow": 1048576/"contextWindow": 32768/g' ~/.openclaw/openclaw.json
systemctl --user restart openclaw-gateway
```

> `sed -i 's/A/B/g'`는 파일을 직접(in place) 열어 A를 B로 전부 치환합니다. 재전송하면 상태바가 `tokens ?/32k`로 바뀌고 정상 작동합니다.

### 디버깅 흐름 요약

```text
에러 메시지 → journalctl 로그 → INT_MAX 포착(메모리 아님) → GitHub 이슈 검색
 → Modelfile로 고정 → 여전히 실패 → openclaw.json 발견 → sed 수정 → 재시작 → 정상
```

---

## 에이전트 아키텍처: 게이트웨이와 서비스 구조

### 첫 응답이 느린 이유 (System Prompt의 무게)

OpenClaw를 경유하면 첫 응답에 약 40초가 걸립니다(직접 Ollama 호출은 즉시). 차이는 시스템 프롬프트의 크기입니다.

- **직접 Ollama 호출**: "안녕" → 약 300 토큰
- **OpenClaw 경유**: 도구 정의(약 5,000자) + 응답 형식 규칙 + Skills 설명 + 행동 규칙 + 사용자 메시지 → 약 3,000~5,000 토큰

약 10배 차이가 첫 응답 지연의 원인입니다. **두 번째 메시지부터는 시스템 프롬프트가 캐시(cache hit)되어 빨라집니다.**

### Tool Call의 정체 (Hooks 작동)

웹 UI에 `Tool call (read) → Tool output → ... → Tool call (write)`가 줄줄이 뜨는 것은 설치 때 켠 Hooks가 작동하는 모습입니다.

| Hook | 동작 |
|------|------|
| boot-md | 시작 시 markdown 파일 자동 로드 (read) |
| bootstrap-extra-files | 추가 파일 자동 로드 (read) |
| session-memory | 세션 상태 저장 (write) |
| compaction-notifier | 히스토리 압축 알림 |
| command-logger | 백그라운드 로그 기록 |

### 게이트웨이 + 클라이언트 구조

```text
클라이언트(입구)              두뇌                       엔진
  TUI(터미널)    ─┐
  Web UI(브라우저) ─┼──→  OpenClaw Gateway  ──HTTP──→  Ollama (모델 서버)
  Telegram(폰)   ─┘      포트 18789, systemd          포트 11434, Nemotron
                          사용자 서비스(상태/메모리/결정)
```

- **OpenClaw Gateway**(결정: 도구 선택, 응답 방식)와 **Ollama**(실행: 토큰 생성)는 분리되어 있고, HTTP로 명시적으로 통신합니다.
- 여러 클라이언트(터미널/브라우저/텔레그램)가 같은 게이트웨이에 붙어 동일한 대화 기록·메모리·능력을 공유합니다.

### systemd Lingering — 로그아웃해도 생존

설치 마법사가 `Enabled systemd lingering`을 자동 처리합니다.

| 상황 | 일반 사용자 서비스 | lingering 활성화 |
|------|--------------------|------------------|
| 로그인 중 | 동작 | 동작 |
| 로그아웃 | 종료 | 계속 동작 |
| 재부팅 후 | 수동 시작 | 자동 시작 |

명령 한 줄 치지 않아도 본체를 켜두기만 하면 24/7 에이전트가 유지됩니다(Docker `--restart always`와 같은 의지).

### 보안 — 게이트웨이 토큰

OpenClaw 접속 URL의 `#token=...`은 비밀번호입니다. 캡처 공유나 GitHub 업로드를 금지하고, 노출 시 재발급합니다.

```bash
openclaw doctor --generate-gateway-token
```

---

## 모델 메모리 관리와 모델 큐레이션

### Ollama 자동 언로드

에이전트가 24시간 살아 있어도 모델이 24시간 메모리를 점유하지는 않습니다. Ollama는 기본적으로 **마지막 사용 후 5분간 요청이 없으면 메모리에서 자동 언로드**합니다.

| 상황 | 메모리 사용 |
|------|------------|
| 채팅 중 | 약 30GB |
| 5분간 미사용 | 자동 언로드, 약 300MB |
| 새 요청 | 재로드 5~15초 |

언로드 대기 시간을 조정하려면 환경 변수를 사용합니다.

```bash
# 예: 30분 유지 (systemd 서비스에 설정)
sudo systemctl edit ollama
# [Service]
# Environment="OLLAMA_KEEP_ALIVE=30m"
sudo systemctl restart ollama
```

### 모델 저장 경로 관리

대형 모델은 수십 GB를 차지합니다. 4TB SSD 모델이라도 경로 관리가 필요합니다.

```bash
# 디스크 사용량 확인
df -h
du -sh ~/.ollama
du -sh /var/lib/docker/volumes

# 모델 목록과 상태
ollama list
ollama ps        # 현재 메모리에 로드된 모델
```

호스트 직접 설치 시 모델 경로를 변경하려면 `OLLAMA_MODELS` 환경 변수를 사용합니다.

```bash
echo 'export OLLAMA_MODELS="/data/.ollama/models"' >> ~/.bashrc
source ~/.bashrc
```

### 모델 큐레이션 가이드

128GB 통합 메모리에 어떤 모델을 올릴지 결정하는 기준입니다(가중치 기준, KV 캐시·오버헤드 별도).

| 모델 규모 | FP16 | 권장 양자화 | 비고 |
|-----------|------|-------------|------|
| 8B~20B | 16~40GB | FP16/MXFP4 | 쾌적, 동시 서빙 우수 |
| 70B | 약 140GB(초과) | FP8(약 70GB) / 4bit(약 35~40GB) | decode 느림, 실험용 |
| 120B (gpt-oss) | — | MXFP4(약 60~65GB) | 단일 장비에서 무난 |
| 200B급 | — | 적극적 4bit | 단일 장비 실질 한계 |
| 405B | — | FP4 | **두 대 연결 필요** |

---

## 성능 특성과 최적화

### prefill은 강하고 decode는 약하다

DGX Spark의 성능 특성을 한 문장으로 요약하면 **"연산은 빠르고 메모리 대역폭은 느리다"**입니다.

- **prefill(프롬프트 처리, 연산 바운드)**: 강력
- **단일 스트림 decode(토큰 생성, 대역폭 바운드)**: 약점

LMSYS 벤치마크(2025년 10월) 참고 수치:

| 모델 / 설정 | prefill (tok/s) | decode (tok/s) |
|-------------|-----------------|----------------|
| gpt-oss 20B (MXFP4, Ollama) | 2,053 | 약 49.7 |
| Llama 3.1 8B (SGLang, batch-1) | 7,991 | 20.5 |
| Llama 3.1 8B (SGLang, batch-32) | 7,949 | 368 (거의 선형 확장) |
| Llama 3.1 70B (FP8) | 803 | 약 2.7 (대역폭 한계) |

> 핵심: **배치/동시 서빙에서 decode가 거의 선형으로 확장**됩니다. 단일 사용자 70B 대화형 생성보다는, 중소형 모델의 동시 처리나 대형 모델 프로토타이핑·파인튜닝에 적합합니다.

### 최적화 팁

- **양자화 우선**: 메모리·속도·전력을 동시에 잡습니다(FP8/MXFP4/int4).
- **배치 처리**: 동시 요청을 묶으면 처리량이 크게 늘어납니다(SGLang, vLLM 등).
- **Speculative Decoding**: EAGLE3 등으로 decode를 최대 약 2배까지 가속해 대역폭 한계를 부분적으로 보완할 수 있습니다.
- **PyTorch 2.0+ 컴파일**:
  ```python
  model = torch.compile(model)
  ```
- **최신 툴체인 고수**: sm_121 지원이 빠르게 개선 중입니다. PyTorch 2.9+, CUDA 13 빌드, 최신 vLLM/SGLang을 사용하세요.

### GPU 모니터링

```bash
# GPU 상태
nvidia-smi

# 실시간 모니터링 (1초 간격)
watch -n 1 nvidia-smi

# 현재 로드된 모델
ollama ps
```

---

## 문제 해결

### 일반적인 실수와 해결책

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 업데이트 후 GPU 인식 안 됨 | 일반 apt 업데이트로 커널/드라이버 충돌 | DGX Dashboard(`localhost:11000`)로만 업데이트 |
| `docker: permission denied` | docker 그룹 미가입 | `sudo usermod -aG docker $USER` 후 `newgrp docker` |
| LLM이 너무 느림 | `--gpus=all` 누락으로 CPU 구동 | 컨테이너 재생성 시 `--gpus=all` 추가 |
| PyTorch가 GPU 못 씀 | CUDA 휠 미지정 | `--index-url`로 CUDA 버전 휠 설치(PyTorch 2.9+) |
| 모델 로드 실패(500) | 컨텍스트 1M 설정 → INT_MAX 초과 | `openclaw.json`의 contextWindow를 32768로 |
| SSH 접속 안 됨 | SSH 비활성 출고 | `sudo systemctl enable --now ssh` |
| NVIDIA Sync 끊김 | Tailscale 노드 키 만료 | 콘솔에서 "Disable key expiry" |
| 컨테이너가 재부팅 후 안 켜짐 | `--restart` 미설정 | `--restart always` 추가 |
| LoRA 동작 안 함 | 본 모델 불일치 | Base Model(SD 1.5/SDXL) 확인 |

### CUDA Out of Memory

```python
import torch
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())
```

- 더 작은 정밀도(FP8/int4)로 모델 로드
- 컨텍스트 윈도우 축소(예: 32K)
- 배치 크기 축소 또는 Gradient Accumulation

### 로그 확인 명령

```bash
# Ollama 로그
sudo journalctl -u ollama -e --no-pager | tail -50

# OpenClaw 사용자 서비스 로그
journalctl --user -u openclaw-gateway -e --no-pager | tail -50

# SSH 상태
sudo systemctl status ssh
```

---

## 빠른 시작 체크리스트

### 1단계: 환경 적응 (첫날)

- [ ] DGX OS 부팅, 계정 생성
- [ ] 업데이트는 DGX Dashboard(`localhost:11000`)로만 수행하는 원칙 숙지
- [ ] 터미널에서 `nvidia-smi`로 GPU 인식 확인

### 2단계: 첫 추론 (JupyterLab)

- [ ] DGX Dashboard에서 JupyterLab Start → Open in Browser
- [ ] `torch.cuda.is_available()` True 확인
- [ ] SDXL로 첫 이미지 생성

### 3단계: 원격 접속

- [ ] 본체에 Tailscale 설치 + `sudo tailscale up --ssh`
- [ ] `sudo systemctl enable --now ssh`
- [ ] Tailscale 콘솔에서 본체 노드 "Disable key expiry"
- [ ] 사무실 PC에 NVIDIA Sync, 폰에 Tailscale + Termius

### 4단계: 로컬 LLM 챗봇

- [ ] docker 그룹 가입(`usermod -aG docker $USER` + `newgrp docker`)
- [ ] Open WebUI 컨테이너 실행(`--gpus=all --restart always`, `:ollama` 태그)
- [ ] `localhost:8080` 접속, gpt-oss:20b 다운로드
- [ ] 폰에서 PWA로 접속 확인

### 5단계: 이미지 생성

- [ ] venv 생성 + CUDA 휠 PyTorch 설치
- [ ] ComfyUI 실행(`localhost:8188`), 기본 워크플로우로 이미지 생성
- [ ] LoRA 추가 실험(Civitai), Base Model 호환 확인

### 6단계: AI 에이전트 (고급)

- [ ] OpenClaw 설치(Skills는 No, Hooks는 모두 켜기)
- [ ] Ollama 호스트 설치 + `nemotron-3-nano:30b` pull
- [ ] `ollama launch openclaw`로 연결
- [ ] `openclaw.json`의 contextWindow를 32768로 조정
- [ ] 게이트웨이 토큰 보안 관리

---

## 자주 묻는 질문

**Q1. DGX Spark는 디스크리트 GPU보다 느린가요?**

단일 스트림 토큰 생성(decode)은 메모리 대역폭(약 273GB/s) 때문에 RTX 5090 등보다 느립니다. 하지만 prefill(연산)은 강하고, 배치/동시 서빙에서는 처리량이 선형에 가깝게 확장됩니다. 무엇보다 128GB 통합 메모리로 디스크리트 GPU에는 올라가지 않는 대형 모델을 구동할 수 있습니다.

**Q2. 128GB에 어떤 모델까지 올라가나요?**

가중치 기준 대략 70B(FP8 약 70GB / 4bit 약 35~40GB), 120B gpt-oss(MXFP4 약 60~65GB)까지 단일 장비에서 가능합니다. 200B급은 적극적 4bit로 실질 한계, 405B는 두 대 연결이 필요합니다.

**Q3. 왜 GPU가 좋은데 float16(작은 정밀도)을 쓰나요?**

메모리 절약 + Tensor Core 가속 + 정확도 손실 미미 때문입니다. AI 추론·이미지 생성은 작은 정밀도로 충분하며, "1 PetaFLOP" 스펙 자체가 FP4 기준입니다.

**Q4. 업데이트는 그냥 `apt upgrade` 하면 안 되나요?**

권장하지 않습니다. 일반 업데이트가 커널/드라이버를 NVIDIA가 세팅한 환경과 충돌시킬 수 있습니다. 반드시 DGX Dashboard를 통해 검증된 패키지만 설치하세요.

**Q5. Docker와 venv 중 무엇을 써야 하나요?**

24/7 백그라운드 서비스(Open WebUI 등)는 Docker, GPU를 직접 쓰는 파이썬 작업(ComfyUI, 파인튜닝)은 venv가 자연스럽습니다. 둘을 병행하며 모델 데이터는 도구와 분리해 보관하세요.

**Q6. PyTorch 설치 시 왜 GPU를 못 쓰나요?**

`--index-url`로 CUDA 버전에 맞는 휠을 지정하지 않으면 CPU 버전이 설치됩니다. sm_121은 PyTorch 2.9 이상(CUDA 13 빌드)이 필요합니다.

**Q7. 에이전트(OpenClaw)는 안전한가요?**

에이전트는 파일·명령·API에 접근할 수 있어 위험합니다. 메인 PC가 아니라 격리된 DGX Spark에서, Skills는 처음에 모두 끄고 신뢰할 수 있는 것만 켜며, 게이트웨이 토큰을 비밀번호처럼 다루세요.

**Q8. 첫 응답이 느린데 고장인가요?**

아닙니다. 모델을 디스크에서 메모리로 로드하는 시간(5~15초)이거나, 에이전트의 큰 시스템 프롬프트 처리 시간(약 40초)입니다. 두 번째 메시지부터는 캐시되어 빨라집니다.

---

## 참고 자료

### 공식 문서

- [NVIDIA DGX Spark 제품 페이지](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [NVIDIA DGX Spark 하드웨어 문서](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA DGX Spark 출시 뉴스룸](https://nvidianews.nvidia.com/news/nvidia-dgx-spark-arrives-for-worlds-ai-developers)
- [GIGABYTE AI TOP ATOM](https://www.gigabyte.com/Consumer/ai-top/AI-TOP-ATOM/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

### 벤치마크·심층 분석

- [LMSYS DGX Spark 심층 벤치마크](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- [ServeTheHome DGX Spark 리뷰](https://www.servethehome.com/nvidia-dgx-spark-review-the-gb10-machine-is-so-freaking-cool/)
- [ServeTheHome GB10 ConnectX-7 200GbE 분석](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/)

### 도구·소프트웨어

- [Ollama](https://github.com/ollama/ollama)
- [Open WebUI](https://github.com/open-webui/open-webui)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Tailscale](https://tailscale.com/)
- [Hugging Face Hub](https://huggingface.co/models)
- [Civitai (LoRA)](https://civitai.com/)
- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/)

---

## 라이선스 및 주의사항

- 이 가이드는 교육 목적으로 작성되었습니다.
- 모델 사용 시 각 모델의 라이선스를 확인하세요(gpt-oss, Nemotron, Qwen 등 라이선스 상이).
- 에이전트(명령 실행 권한)는 격리된 환경에서만 운영하고, 의뢰인·내부 데이터가 있는 메인 PC에서는 사용하지 마세요.
- 원격 접속 자격 증명(SSH 키, Tailscale, 게이트웨이 토큰)을 외부에 노출하지 마세요.

