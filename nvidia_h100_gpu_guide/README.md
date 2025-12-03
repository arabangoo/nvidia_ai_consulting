# NVIDIA H100 GPU 서버 완벽 가이드

이 가이드는 NVIDIA H100 GPU를 처음 다루는 사용자도 바로 학습 및 서비스 개발을 시작할 수 있도록 작성되었습니다.

> **실전 경험 기반 가이드**: 이 문서는 실제 H100 서버 운영 경험을 바탕으로 작성되었습니다. GPT-5 OSS 120B, DeepSeek-R1 70B 등 대규모 모델 실행부터 일상적인 운영까지 모든 과정을 다룹니다.

## 목차
1. [H100 GPU 소개](#h100-gpu-소개)
2. [서버 접속 및 환경 설정](#서버-접속-및-환경-설정)
3. [GPU 상태 확인](#gpu-상태-확인)
4. [개발 환경 구축](#개발-환경-구축)
5. [대규모 언어 모델(LLM) 실행](#대규모-언어-모델llm-실행)
6. [딥러닝 프레임워크 활용](#딥러닝-프레임워크-활용)
7. [모델 학습 및 추론](#모델-학습-및-추론)
8. [성능 최적화 팁](#성능-최적화-팁)
9. [스토리지 관리 및 디스크 최적화](#스토리지-관리-및-디스크-최적화)
10. [실전 프로젝트: 대규모 LLM 운영](#실전-프로젝트-대규모-llm-운영)
11. [문제 해결](#문제-해결)

---

## H100 GPU 소개

### 주요 사양
- **메모리**: 80GB HBM3
- **메모리 대역폭**: 3TB/s
- **CUDA 코어**: 16,896개
- **Tensor 코어**: 528개 (4세대)
- **FP64 성능**: 34 TFLOPS
- **FP32 성능**: 67 TFLOPS
- **TF32 Tensor 성능**: 989 TFLOPS
- **FP16 Tensor 성능**: 1,979 TFLOPS

### 가격 정보
- H100 1대: 약 4,800만원
- 클라우드 임대 (H100): 시간당 약 1만 1천원 (월 약 470만원)
- H200 (후속 모델): 시간당 약 9만원
- H100 3대 임대: 월 약 1,500만원

### H100의 실제 활용 사례
- **GPT-5 OSS 120B (65GB 모델)**: H100 80GB 메모리에서 원활하게 실행 가능
- **DeepSeek-R1 70B (42GB 모델)**: 추론 시 GPU 메모리 약 41GB 사용
- **다중 모델 동시 서빙**: Ollama를 통해 여러 모델 동시 관리 가능
- **추론 성능**: GPT-OSS 120B 기준, 모델 로딩 후 GPU 메모리 사용률 98%, 온도 약 41°C

---

## 서버 접속 및 환경 설정

### 1. SSH 접속 (MobaXterm 권장)

**MobaXterm 사용 이유:**
- 파일 다운로드/업로드 기능 내장 (유료 SecureCRT 대체 가능)
- 세션 관리 및 저장 기능
- X11 포워딩 지원
- 무료 버전으로도 충분한 기능 제공

#### MobaXterm 다운로드 및 설치

```bash
# MobaXterm 공식 사이트에서 다운로드
# https://mobaxterm.mobatek.net/download.html
```

#### SSH 접속 방법

```bash
# 기본 SSH 접속
ssh username@your-h100-server.com

# 특정 포트 지정
ssh -p 22 username@your-h100-server.com

# SSH 키 기반 접속 (권장)
ssh -i ~/.ssh/id_rsa username@your-h100-server.com
```

**MobaXterm에서 새 세션 생성:**
1. 'New session' 버튼 클릭
2. 'SSH' 선택
3. Remote host, Username 입력
4. Port 설정 (기본 22)
5. 'OK' 클릭하여 접속

### 2. 기본 시스템 업데이트

```bash
# 시스템 패키지 업데이트
sudo apt update
sudo apt upgrade -y

# 필수 도구 설치
sudo apt install -y build-essential git wget curl vim
```

---

## GPU 상태 확인

### nvidia-smi 명령어

```bash
# GPU 상태 확인
nvidia-smi

# 실시간 모니터링 (1초마다 갱신)
watch -n 1 nvidia-smi

# GPU 상세 정보
nvidia-smi -L  # GPU 목록
nvidia-smi -q  # 상세 정보
```

### gpustat 설치 및 사용

```bash
# gpustat 설치 (주의: gpustat이 아닌 gptstat로 오타 주의)
sudo apt install -y gpustat

# 또는 pip로 설치
pip install gpustat

# GPU 상태 확인 (더 보기 좋은 형식)
gpustat

# 실시간 모니터링
gpustat -i 1
```

**실제 H100 출력 예시:**
```
tflite-h2-vm                       Wed Aug 27 00:25:36 2025  535.183.06
[0] NVIDIA H100 80GB HBM3 | 34°C,  0 % | 14736 / 81559 MB | ollama(14176M)
[1] NVIDIA H100 80GB HBM3 | 40°C,  0 % | 1076 / 81559 MB | ollama(516M)
```

**GPU 메모리 사용 패턴:**
- **유휴 상태**: 0-2% GPU 메모리 사용
- **GPT-OSS 120B 로딩 중**: 순간적으로 64GB+ 사용
- **GPT-OSS 120B 추론 중**: 98% GPU 메모리 사용 (약 64GB)
- **DeepSeek-R1 70B 추론 중**: 약 41GB GPU 메모리 사용
- **온도**: 일반적으로 32-41°C 범위 유지

---

## 개발 환경 구축

### 1. CUDA 및 cuDNN 확인

```bash
# CUDA 버전 확인
nvcc --version

# 또는
cat /usr/local/cuda/version.txt

# CUDA 경로 확인
which nvcc
```

**H100은 CUDA 12.0 이상을 권장합니다.**

### 2. Python 가상환경 설정

```bash
# Python 3.10 설치 (권장)
sudo apt install -y python3.10 python3.10-venv python3-pip

# 가상환경 생성
python3.10 -m venv h100-env

# 가상환경 활성화
source h100-env/bin/activate

# pip 업그레이드
pip install --upgrade pip
```

### 3. PyTorch 설치 (CUDA 12.x)

```bash
# PyTorch 2.x with CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

### 4. TensorFlow 설치

```bash
# TensorFlow 2.x with GPU support
pip install tensorflow[and-cuda]

# 설치 확인
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')"
```

---

## 대규모 언어 모델(LLM) 실행

### Ollama를 이용한 LLM 실행

#### 1. Ollama 설치

```bash
# Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 설치 확인
ollama --version
```

#### 2. 모델 저장 경로 설정 (선택사항)

```bash
# 기본 경로: ~/.ollama/models
# 커스텀 경로 설정 (디스크 공간이 큰 곳으로)
export OLLAMA_MODELS="/data/.ollama/models"

# 영구 설정 (bashrc에 추가)
echo 'export OLLAMA_MODELS="/data/.ollama/models"' >> ~/.bashrc
source ~/.bashrc
```

#### 3. Ollama 서비스 설정

```bash
# systemd 서비스 파일 편집
sudo systemctl edit ollama

# 다음 내용 추가:
# [Service]
# Environment="OLLAMA_MODELS=/data/.ollama/models"

# 서비스 재시작
sudo systemctl stop ollama
sudo systemctl start ollama

# 서비스 상태 확인
sudo systemctl status ollama
```

#### 4. 모델 다운로드 및 실행

```bash
# GPT-5 OSS 20B 모델 다운로드 (약 30분 소요)
ollama pull gpt-oss:20b

# GPT-5 OSS 120B 모델 다운로드 (약 65GB, 100분 소요)
ollama pull gpt-oss:120b

# DeepSeek-R1 70B 모델
ollama pull deepseek-r1:70b

# Microsoft Phi-4 모델
ollama pull phi4:latest

# 설치된 모델 목록 확인
ollama list
```

**출력 예시:**
```
NAME                ID          SIZE    MODIFIED
deepseek-r1:70b     d37b54d01a76 42 GB   2 minutes ago
gpt-oss:120b        f7f8e2f8f4e0 65 GB   19 minutes ago
phi4:latest         ac896e5b8b34 9.1 GB  34 minutes ago
```

#### 5. 모델 실행

```bash
# 대화형 모드로 실행
ollama run gpt-oss:120b

# 또는 특정 경로 지정
OLLAMA_MODELS="/data/.ollama/models" ollama run gpt-oss:120b
```

#### 6. Python에서 Ollama 사용

```python
from openai import OpenAI

# Ollama는 OpenAI API와 호환됩니다
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # 필수이지만 사용되지 않음
)

response = client.chat.completions.create(
    model="gpt-oss:120b",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)

print(response.choices[0].message.content)
```

#### 7. DeepSeek-R1 사용 (Thinking 기능)

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

response = client.chat.completions.create(
    model="deepseek-r1:70b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "화웨이 Ascend 910 NPU를 이용해서 이미지 vision하는 python 코드 제작해"}
    ]
)

# DeepSeek-R1은 <think> 태그로 사고 과정을 보여줍니다
print(response.choices[0].message.content)
```

### GPU 메모리 사용량 확인

```bash
# H100 80GB에서 GPT-OSS 120B 실행 시
# - 모델 로딩: 약 64GB
# - 추론 시: 약 98% GPU 메모리 사용
# - 온도: 약 41°C

# 실시간 모니터링
gpustat -i 1
```

---

## 딥러닝 프레임워크 활용

### PyTorch 기본 사용법

#### 1. GPU 확인 및 설정

```python
import torch

# GPU 사용 가능 여부 확인
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

#### 2. 간단한 모델 학습 예제

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 디바이스 설정
device = torch.device("cuda")

# 간단한 신경망 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델을 GPU로 이동
model = SimpleNet().to(device)

# 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 더미 데이터 생성
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 학습 루프
model.train()
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        # 데이터를 GPU로 이동
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

#### 3. 사전 학습된 모델 사용 (ResNet-50)

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 디바이스 설정
device = torch.device("cuda")

# ResNet-50 모델 로드
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 이미지 로드 및 추론
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Top-5 예측
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    return top5_idx.cpu().numpy(), top5_prob.cpu().numpy()

# 사용 예시
# indices, probs = predict_image("example.jpg")
# for idx, prob in zip(indices, probs):
#     print(f"Class {idx}: {prob*100:.2f}%")
```

### TensorFlow/Keras 사용법

#### 1. GPU 확인 및 설정

```python
import tensorflow as tf

# GPU 확인
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# GPU 메모리 증가 허용 (필요시)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)
```

#### 2. 간단한 모델 학습

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 모델 정의
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 더미 데이터 생성
import numpy as np
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

# 학습
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=1
)
```

#### 3. 사전 학습된 모델 사용

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# ResNet50 모델 로드
model = ResNet50(weights='imagenet')

def predict_image(img_path):
    # 이미지 로드 및 전처리
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # 예측
    preds = model.predict(x)
    
    # 결과 디코딩
    decoded = decode_predictions(preds, top=5)[0]
    
    print("Top-5 Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded):
        print(f"{i+1}. {label}: {score*100:.2f}%")

# 사용 예시
# predict_image("example.jpg")
```

---

## 모델 학습 및 추론

### 1. 대규모 모델 학습 (Multi-GPU)

#### PyTorch DataParallel

```python
import torch
import torch.nn as nn

# 모델 정의
model = YourLargeModel()

# 여러 GPU 사용
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to('cuda')

# 학습 코드는 동일
```

#### PyTorch DistributedDataParallel (권장)

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # 모델을 특정 GPU에 할당
    model = YourModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # 학습 코드
    # ...
    
    cleanup()

# 실행
# python -m torch.distributed.launch --nproc_per_node=2 train.py
```

### 2. 혼합 정밀도 학습 (Mixed Precision)

H100은 FP16, BF16, TF32를 지원하여 학습 속도를 크게 향상시킬 수 있습니다.

#### PyTorch AMP (Automatic Mixed Precision)

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# 모델, 옵티마이저 설정
model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

# 학습 루프
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

#### TensorFlow Mixed Precision

```python
from tensorflow.keras import mixed_precision

# Mixed precision 정책 설정
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 모델 정의 (자동으로 mixed precision 적용)
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax', dtype='float32')  # 마지막 레이어는 float32
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 3. 모델 저장 및 로드

#### PyTorch

```python
# 모델 저장
torch.save(model.state_dict(), 'model_weights.pth')
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# 모델 로드
model = YourModel()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 체크포인트 로드
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

#### TensorFlow

```python
# 모델 저장
model.save('my_model.h5')  # 전체 모델
model.save_weights('model_weights.h5')  # 가중치만

# 모델 로드
model = keras.models.load_model('my_model.h5')
# 또는
model = YourModel()
model.load_weights('model_weights.h5')
```

---

## 성능 최적화 팁

### 1. 배치 크기 최적화

H100의 80GB 메모리를 최대한 활용하려면 배치 크기를 늘리세요.

```python
# 메모리 사용량 모니터링하며 배치 크기 조정
batch_sizes = [32, 64, 128, 256, 512]

for batch_size in batch_sizes:
    try:
        train_loader = DataLoader(dataset, batch_size=batch_size)
        # 학습 시도
        print(f"Batch size {batch_size} works!")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"Batch size {batch_size} - OOM")
            break
```

### 2. 데이터 로딩 최적화

```python
# PyTorch DataLoader 최적화
train_loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=8,  # CPU 코어 수에 맞게 조정
    pin_memory=True,  # GPU 전송 속도 향상
    persistent_workers=True  # 워커 재사용
)
```

### 3. Gradient Accumulation

메모리가 부족할 때 큰 배치 크기 효과를 얻는 방법:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (batch_x, batch_y) in enumerate(train_loader):
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss = loss / accumulation_steps  # 손실 정규화
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. TensorFloat-32 (TF32) 활성화

H100에서 자동으로 활성화되지만, 명시적으로 설정할 수 있습니다:

```python
# PyTorch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# TensorFlow는 자동으로 TF32 사용
```

### 5. cuDNN 벤치마크 모드

```python
# PyTorch
torch.backends.cudnn.benchmark = True  # 입력 크기가 일정할 때 사용
```

### 6. 모델 컴파일 (PyTorch 2.0+)

```python
import torch

model = YourModel()
model = torch.compile(model)  # 모델 컴파일로 속도 향상
```

---

## 추가 도구 및 라이브러리

### 1. Hugging Face Transformers

대규모 언어 모델 및 트랜스포머 모델 사용:

```bash
pip install transformers accelerate bitsandbytes
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델 로드
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # 자동으로 GPU에 분산
)

# 텍스트 생성
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### 2. vLLM (고속 LLM 추론)

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams

# 모델 로드
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 샘플링 파라미터
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# 추론
prompts = ["Hello, my name is", "The future of AI is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### 3. ONNX Runtime

모델 최적화 및 배포:

```bash
pip install onnx onnxruntime-gpu
```

```python
import torch
import onnx
import onnxruntime as ort

# PyTorch 모델을 ONNX로 변환
dummy_input = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# ONNX 모델 로드 및 추론
session = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: dummy_input.cpu().numpy()})
```

### 4. DeepSpeed (대규모 모델 학습)

```bash
pip install deepspeed
```

```python
import deepspeed

# DeepSpeed 설정
ds_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2
    }
}

# 모델 초기화
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# 학습 루프
for batch in train_loader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### 5. Ray (분산 학습 및 하이퍼파라미터 튜닝)

```bash
pip install ray[tune]
```

```python
import ray
from ray import tune

def train_model(config):
    # 모델 학습 코드
    model = YourModel(config["lr"], config["batch_size"])
    # ...
    return {"loss": final_loss}

# 하이퍼파라미터 탐색
analysis = tune.run(
    train_model,
    config={
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256])
    },
    num_samples=10
)

print("Best config:", analysis.best_config)
```

---

## 실전 예제: 이미지 분류 프로젝트

### 전체 워크플로우

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler
import time

# 1. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. 데이터 전처리
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 데이터셋 로드 (예: ImageNet 또는 커스텀 데이터)
train_dataset = datasets.ImageFolder('path/to/train', transform=transform_train)
val_dataset = datasets.ImageFolder('path/to/val', transform=transform_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,  # H100의 메모리를 활용
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

# 4. 모델 정의
model = models.resnet50(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 5. 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 6. Mixed Precision 설정
scaler = GradScaler()

# 7. 학습 함수
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}, "
                  f"Acc: {100.*correct/total:.2f}%")
    
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, epoch_time

# 8. 검증 함수
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

# 9. 학습 루프
num_epochs = 20
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 50)
    
    # 학습
    train_loss, train_acc, train_time = train_epoch(
        model, train_loader, criterion, optimizer, scaler, device
    )
    
    # 검증
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # 학습률 조정
    scheduler.step()
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Epoch Time: {train_time:.2f}s")
    
    # 최고 성능 모델 저장
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, 'best_model.pth')
        print(f"Best model saved! Acc: {best_acc:.2f}%")

print(f"\nTraining completed! Best Val Acc: {best_acc:.2f}%")
```

---

## 문제 해결

### 1. CUDA Out of Memory 에러

```python
# 해결 방법 1: 배치 크기 줄이기
batch_size = 128  # 256에서 128로

# 해결 방법 2: Gradient Accumulation
accumulation_steps = 4

# 해결 방법 3: 메모리 정리
torch.cuda.empty_cache()

# 해결 방법 4: 메모리 사용량 확인
print(torch.cuda.memory_summary())
```

### 2. GPU 인식 안 됨

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# CUDA 설치 확인
nvcc --version

# PyTorch CUDA 버전 확인
python -c "import torch; print(torch.version.cuda)"

# 드라이버 재설치 (필요시)
sudo apt purge nvidia-*
sudo apt install nvidia-driver-535
sudo reboot
```

### 3. 학습 속도가 느림

```python
# 1. 데이터 로딩 병목 확인
# num_workers 증가
train_loader = DataLoader(dataset, num_workers=16, pin_memory=True)

# 2. Mixed Precision 사용
from torch.cuda.amp import autocast, GradScaler

# 3. 모델 컴파일 (PyTorch 2.0+)
model = torch.compile(model)

# 4. cuDNN 벤치마크
torch.backends.cudnn.benchmark = True

# 5. 프로파일링
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # 학습 코드
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 4. Ollama 모델이 로드되지 않음

```bash
# 서비스 상태 확인
sudo systemctl status ollama

# 서비스 재시작
sudo systemctl restart ollama

# 로그 확인
sudo journalctl -u ollama -f

# 모델 경로 확인
echo $OLLAMA_MODELS
ls -lh $OLLAMA_MODELS

# 디스크 공간 확인
df -h
```

### 5. 멀티 GPU 사용 시 문제

```python
# GPU 간 통신 확인
import torch.distributed as dist

# NCCL 백엔드 사용
dist.init_process_group(backend='nccl')

# GPU 간 대역폭 테스트
# nvidia-smi topo -m
```

---

## 모니터링 및 로깅

### 1. TensorBoard 사용

```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    # 학습 코드
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)
    
    # 로그 기록
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    # 학습률 기록
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

writer.close()
```

```bash
# TensorBoard 실행
tensorboard --logdir=runs --port=6006

# 브라우저에서 접속
# http://localhost:6006
```

### 2. Weights & Biases (wandb)

```bash
pip install wandb
```

```python
import wandb

# 프로젝트 초기화
wandb.init(project="h100-training", config={
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 20
})

# 학습 중 로깅
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)
    
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

wandb.finish()
```

### 3. GPU 사용률 모니터링 스크립트

```bash
# monitor_gpu.sh
#!/bin/bash

while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
    echo ""
    echo "=== Process Info ==="
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    sleep 2
done
```

```bash
chmod +x monitor_gpu.sh
./monitor_gpu.sh
```

---

## 스토리지 관리 및 디스크 최적화

### 1. 디스크 공간 확인

대규모 LLM 모델은 수십 GB의 디스크 공간을 필요로 합니다. 시스템 설정 전에 디스크 공간을 반드시 확인하세요.

```bash
# 전체 디스크 사용량 확인
df -h

# 특정 디렉토리 사용량 확인
du -sh /home/vmuser
du -sh ~/.ollama
du -sh /data

# 디렉토리별 상세 사용량
du -h --max-depth=1 /home/vmuser | sort -hr
```

**실제 예시 출력:**
```
Filesystem              Size  Used Avail Use% Mounted on
tmpfs                   1.0K  512B  512B   1% /run
/dev/mapper/ubuntu--vg-ubuntu-lv  99735008 23612880  71532492  25% /run
/dev/sda1              197628  133800   47568   74% /boot/efi
tmpfs                    5120     0    5120    0% /run/lock
/dev/sda2              24756208  24756208      0 100% /
/dev/vg01-lv01       104805908 16181312 103187596   2% /data
```

### 2. Ollama 모델 저장 경로 변경

**문제 상황:**
- 기본 경로 `~/.ollama/models`는 루트 파티션(/)에 위치
- 루트 파티션이 100GB로 작을 경우, GPT-OSS 120B (65GB) + DeepSeek-R1 70B (42GB) 등 대형 모델 설치 불가
- PyTorch, TensorFlow 등 설치 시 디스크 풀 발생

**해결 방법:**

#### 방법 1: 환경 변수로 경로 변경

```bash
# 임시 변경 (현재 세션만)
export OLLAMA_MODELS="/data/.ollama/models"

# 영구 변경 (.bashrc에 추가)
echo 'export OLLAMA_MODELS="/data/.ollama/models"' >> ~/.bashrc
source ~/.bashrc

# 디렉토리 생성
mkdir -p /data/.ollama/models
```

#### 방법 2: systemd 서비스 파일 수정

```bash
# Ollama 서비스 파일 편집
sudo systemctl edit ollama

# 다음 내용 추가:
[Service]
Environment="OLLAMA_MODELS=/data/.ollama/models"

# 서비스 재시작
sudo systemctl stop ollama
sudo systemctl start ollama

# 서비스 상태 확인
sudo systemctl status ollama
```

#### 방법 3: 심볼릭 링크 사용 (가장 확실한 방법)

```bash
# 기존 모델 디렉토리가 있다면 백업
mv ~/.ollama/models ~/.ollama/models.bak

# 블록 스토리지에 실제 디렉토리 생성
mkdir -p /data/.ollama/models

# 심볼릭 링크 생성
ln -s /data/.ollama/models ~/.ollama/models

# 확인
ls -la ~/.ollama/
```

### 3. 모델 파일 다운로드 시간 참고

| 모델 | 파일 크기 | 다운로드 시간 (빠른 네트워크) |
|------|----------|------------------------------|
| GPT-OSS 20B | 13 GB | 약 30초 |
| Microsoft Phi-4 | 9.1 GB | 약 20초 |
| DeepSeek-R1 70B | 42 GB | 약 2분 |
| GPT-OSS 120B | 65 GB | 약 4분 |

### 4. 디스크 공간 확보

```bash
# Docker 미사용 이미지 제거
docker system prune -a

# apt 캐시 정리
sudo apt clean
sudo apt autoremove

# pip 캐시 정리
pip cache purge

# 로그 파일 정리
sudo journalctl --vacuum-time=7d

# 오래된 커널 제거
sudo apt autoremove --purge
```

---

## 실전 프로젝트: 대규모 LLM 운영

### 1. GPT-OSS 120B 모델 실행 (실전)

#### 모델 다운로드 및 실행

```bash
# 모델 다운로드 (약 65GB, 4-5분 소요)
ollama pull gpt-oss:120b

# 실행
ollama run gpt-oss:120b
```

**실행 중 GPU 모니터링:**
```bash
# 별도 터미널에서 실시간 모니터링
watch -n 1 gpustat

# 또는
nvidia-smi -l 1
```

**실제 리소스 사용 현황:**
```
NAME          ID            SIZE    MODIFIED
gpt-oss:120b  f7f8e2f8f4e0  65 GB   19 minutes ago
deepseek-r1:70b  d37b54d01a76  42 GB   2 minutes ago
phi4:latest   ac896e5b8b34  9.1 GB  34 minutes ago
```

#### Python API 사용

```python
from openai import OpenAI

# Ollama는 OpenAI API와 호환
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # 필수이지만 실제로는 사용되지 않음
)

# GPT-OSS 120B로 추론
response = client.chat.completions.create(
    model="gpt-oss:120b",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)

print(response.choices[0].message.content)
```

### 2. DeepSeek-R1 70B Thinking 기능 활용

DeepSeek-R1은 중간 사고 과정을 `<think>` 태그로 출력하는 독특한 기능이 있습니다.

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

response = client.chat.completions.create(
    model="deepseek-r1:70b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "화웨이 Ascend 910 NPU를 이용해서 이미지 vision하는 python 코드 제작해"}
    ]
)

# 응답에는 <think> 태그로 사고 과정이 포함됨
print(response.choices[0].message.content)
```

**실제 출력 예시:**
```
<think>
Okay, the user wants to create Python code using Huawei's Ascend 910 NPU for image vision tasks.
I need to outline the necessary steps: Ascend Toolkit installation, MindSpore setup,
loading ResNet-50 model, preprocessing images, making predictions...
</think>

화웨이 Ascend 910 NPU를 사용한 이미지 비전 처리를 위한 Python 코드...
[코드 계속]
```

### 3. 여러 모델 동시 관리

```bash
# 설치된 모델 목록 확인
ollama list

# Ollama 서비스 상태 확인
ps -ef | grep ollama

# 특정 모델 실행
OLLAMA_MODELS="/data/.ollama/models" ollama run gpt-oss:120b

# 모델 삭제 (공간 확보)
ollama rm phi4:latest
```

### 4. 파일 다운로드 및 업로드 (MobaXterm)

**MobaXterm을 통한 파일 전송:**

1. 왼쪽 패널에서 파일 선택
2. 우클릭 → "Download" 선택
3. 로컬 저장 위치 지정

**Python 코드 실행 결과를 파일로 저장:**
```bash
# 출력을 파일로 리다이렉트
python3 gpt5_oss_sample1.py > huawei_npu.txt

# 실행 결과와 에러를 모두 저장
python3 deepseek_r1_sample1.py > deepseek_NPU1.txt 2>&1
```

### 5. 실제 코딩 예제: Huawei NPU 이미지 처리 코드 생성

이 폴더에 포함된 `deepseek_r1_sample1.py` 파일은 DeepSeek-R1 모델을 사용하여 Huawei Ascend 910 NPU용 이미지 비전 코드를 생성한 예시입니다.

```python
# deepseek_r1_sample1.py
from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

response = client.chat.completions.create(
  model="deepseek-r1:70b",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "화웨이ascend 910 NPU를이용해서 이미지vision하는  python  코드제작해    "}
  ]
)
print(response.choices[0].message.content)
```

생성된 코드는 `huawei_npu-1.txt`, `huawei_npu-2.txt`, `huawei_NPU3.txt`, `deepseek_NPU1.txt` 파일에서 확인할 수 있습니다.

---

## 보안 및 접근 관리

### 1. SSH 키 기반 인증 설정

```bash
# 로컬에서 SSH 키 생성
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 공개 키를 서버에 복사
ssh-copy-id username@h100-server.com

# 또는 수동으로 복사
cat ~/.ssh/id_rsa.pub | ssh username@h100-server.com "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### 2. Jupyter Notebook 원격 접속

```bash
# Jupyter 설치
pip install jupyter

# 설정 파일 생성
jupyter notebook --generate-config

# 비밀번호 설정
python -c "from jupyter_server.auth import passwd; print(passwd())"

# 설정 파일 편집 (~/.jupyter/jupyter_notebook_config.py)
# c.NotebookApp.password = 'sha1:...'  # 위에서 생성한 해시
# c.NotebookApp.ip = '0.0.0.0'
# c.NotebookApp.port = 8888
# c.NotebookApp.open_browser = False

# Jupyter 실행
jupyter notebook

# 로컬에서 SSH 터널링
ssh -L 8888:localhost:8888 username@h100-server.com
```

### 3. 사용자별 리소스 제한

```bash
# GPU 특정 사용자에게만 할당
export CUDA_VISIBLE_DEVICES=0  # GPU 0번만 사용

# Python 스크립트에서
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # GPU 0, 1번 사용
```

---

## 참고 자료

### 공식 문서
- [NVIDIA H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Ollama Documentation](https://github.com/ollama/ollama)

### 유용한 도구
- [Hugging Face Hub](https://huggingface.co/models)
- [Papers with Code](https://paperswithcode.com/)
- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/)

### 커뮤니티
- [PyTorch Forums](https://discuss.pytorch.org/)
- [TensorFlow Forums](https://discuss.tensorflow.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

---

## 빠른 시작 체크리스트

### 초급 사용자 (첫 H100 사용자)

- [ ] **1단계: 서버 접속**
  - [ ] MobaXterm 설치 및 SSH 접속 설정
  - [ ] 서버 접속 테스트

- [ ] **2단계: GPU 상태 확인**
  - [ ] `nvidia-smi` 명령으로 H100 인식 확인
  - [ ] `gpustat` 설치 및 실행
  - [ ] GPU 드라이버 버전 확인 (CUDA 12.2+ 권장)

- [ ] **3단계: 디스크 공간 확인**
  - [ ] `df -h` 로 디스크 공간 확인
  - [ ] 루트 파티션이 작다면 Ollama 모델 경로 변경 설정

- [ ] **4단계: Ollama 설치 및 첫 모델 실행**
  - [ ] Ollama 설치: `curl -fsSL https://ollama.com/install.sh | sh`
  - [ ] 작은 모델부터 시작: `ollama pull phi4:latest` (9.1GB)
  - [ ] 모델 실행 테스트: `ollama run phi4:latest`

- [ ] **5단계: Python 환경 구축**
  - [ ] Python 가상환경 생성
  - [ ] OpenAI Python 패키지 설치: `pip install openai`
  - [ ] 간단한 Python 코드로 Ollama API 테스트

### 중급 사용자 (딥러닝 프레임워크 사용)

- [ ] **PyTorch 설치 및 테스트**
  - [ ] PyTorch CUDA 12.x 버전 설치
  - [ ] GPU 사용 가능 여부 확인 코드 실행
  - [ ] 간단한 모델 학습 테스트

- [ ] **대규모 모델 실행**
  - [ ] GPT-OSS 120B 다운로드 (65GB)
  - [ ] DeepSeek-R1 70B 다운로드 (42GB)
  - [ ] GPU 메모리 사용량 모니터링

- [ ] **모니터링 도구 설정**
  - [ ] TensorBoard 또는 wandb 설정
  - [ ] GPU 모니터링 스크립트 작성

### 고급 사용자 (프로덕션 환경)

- [ ] **vLLM 설정** (고속 추론)
- [ ] **멀티 GPU 활용** (DataParallel / DistributedDataParallel)
- [ ] **모델 최적화** (Mixed Precision, Gradient Accumulation)
- [ ] **서비스 배포** (FastAPI + vLLM)
- [ ] **보안 설정** (SSH 키, 방화벽, 사용자 권한)

---

## 실전 팁 & 트러블슈팅

### 일반적인 실수와 해결책

| 문제 | 원인 | 해결책 |
|------|------|--------|
| **디스크 풀 에러 발생** | 루트 파티션(/) 공간 부족 | Ollama 모델 경로를 `/data`로 변경. 심볼릭 링크 사용 권장 |
| **모델 다운로드가 안됨** | Ollama 서비스 미실행 | `sudo systemctl start ollama` 실행 |
| **GPU 메모리 부족** | 모델 크기가 GPU 메모리 초과 | 작은 모델 사용 또는 양자화 모델 사용 (8-bit, 4-bit) |
| **환경변수 적용 안됨** | 세션 재시작 필요 | `source ~/.bashrc` 실행 또는 재로그인 |
| **CUDA 버전 불일치** | PyTorch/TensorFlow와 CUDA 버전 차이 | 호환되는 버전 재설치 (`nvidia-smi`로 CUDA 버전 확인) |

### 성능 최적화 체크리스트

1. **모델 로딩 속도 향상**
   - SSD 또는 블록 스토리지 사용
   - 모델 파일을 로컬에 미리 다운로드

2. **추론 속도 향상**
   - Mixed Precision (FP16, BF16) 사용
   - TensorFloat-32 (TF32) 활성화 (H100은 기본 지원)
   - vLLM 사용 (Ollama 대신 프로덕션 환경에서)

3. **메모리 최적화**
   - Gradient Accumulation으로 큰 배치 효과
   - 양자화 모델 사용 (GPTQ, AWQ)
   - Flash Attention 활용

4. **비용 절감**
   - 사용하지 않는 모델 삭제: `ollama rm <model>`
   - GPU 유휴 시간 최소화
   - 스팟 인스턴스 활용 (클라우드)

### 자주 묻는 질문 (FAQ)

**Q1: H100과 A100의 차이는 무엇인가요?**

A: H100은 A100 대비 다음과 같은 개선사항이 있습니다:
- 메모리: 80GB HBM3 (A100은 40/80GB HBM2e)
- 성능: FP16 Tensor 성능 약 2배 향상
- 대역폭: 3TB/s (A100은 2TB/s)
- TF32 연산 성능 대폭 향상

**Q2: 여러 사용자가 동시에 H100을 사용할 수 있나요?**

A: 네, MPS (Multi-Process Service)를 활용하거나 각 사용자에게 특정 GPU를 할당할 수 있습니다.
```bash
# 특정 GPU만 사용하도록 설정
export CUDA_VISIBLE_DEVICES=0  # GPU 0번만 사용
```

**Q3: GPT-OSS 120B와 GPT-4의 차이는?**

A: GPT-OSS는 오픈소스 대안으로, 상업적 제약 없이 로컬에서 실행 가능합니다. 성능은 GPT-4보다 낮을 수 있지만, 데이터 프라이버시와 커스터마이징 측면에서 장점이 있습니다.

**Q4: Ollama와 vLLM의 차이는?**

A:
- **Ollama**: 개발/연구 환경에 적합. 설치 간편, 모델 관리 쉬움
- **vLLM**: 프로덕션 환경에 적합. 더 빠른 추론 속도, 배치 처리 최적화

**Q5: H100에서 실행 가능한 최대 모델 크기는?**

A:
- **단일 H100 (80GB)**: 최대 약 70B 파라미터 (FP16 기준)
- **양자화 (8-bit)**: 최대 약 120-140B 파라미터
- **멀티 GPU**: 수백 B 파라미터 모델도 가능

**Q6: DeepSeek-R1의 Thinking 기능은 무엇인가요?**

A: DeepSeek-R1은 추론 과정에서 중간 사고 과정을 `<think>` 태그로 출력합니다. 이를 통해 모델이 어떻게 결론에 도달했는지 확인할 수 있어 디버깅과 신뢰성 향상에 유용합니다.

**Q7: 모델 다운로드가 중간에 끊기면 어떻게 하나요?**

A: Ollama는 중단된 지점부터 자동으로 재개합니다. 다시 같은 `ollama pull` 명령을 실행하면 됩니다.

**Q8: H100 서버를 처음 사용하는데 어디서부터 시작해야 하나요?**

A:
1. 빠른 시작 체크리스트의 "초급 사용자" 단계를 따라가세요
2. 작은 모델(Phi-4 9GB)부터 시작하여 시스템에 익숙해지세요
3. GPU 모니터링 도구를 항상 켜두고 리소스 사용을 관찰하세요
4. 디스크 공간을 먼저 확인하고 필요시 모델 경로를 변경하세요

### 참고 파일

이 저장소에는 실제 H100 사용 경험을 바탕으로 한 다음 파일들이 포함되어 있습니다:

- `deepseek_r1_sample1.py`: DeepSeek-R1 70B를 사용한 Python 코드 예제
- `deepseek_NPU1.txt`: DeepSeek-R1이 생성한 Huawei NPU 코드 (Thinking 과정 포함)
- `huawei_npu-1.txt`, `huawei_npu-2.txt`, `huawei_NPU3.txt`: GPT-OSS가 생성한 다양한 NPU 관련 코드 및 가이드
- PDF 문서들: 실제 H100 서버 접속 및 운영 경험 기록

---

## 라이선스 및 주의사항

- 이 가이드는 교육 목적으로 작성되었습니다.
- 상용 모델 사용 시 각 모델의 라이선스를 확인하세요.
- H100 서버 사용 시 클라우드 제공업체의 이용 약관을 준수하세요.
- 대규모 모델 학습 시 전력 소비와 비용을 고려하세요.

### 비용 예상 (참고)

| 구성 | 월 비용 (참고) |
|------|---------------|
| H100 1대 클라우드 임대 | 약 470만원 |
| H100 3대 클라우드 임대 | 약 1,500만원 |
| H100 1대 구매 | 약 4,800만원 (초기 투자) |

---

**작성일**: 2025년 12월 3일
**버전**: 2.0 (실전 경험 대폭 보강)
**기여**: 실제 H100 GPU 서버 운영 경험을 바탕으로 작성
**문의**: 추가 질문이나 문제가 있으면 이슈를 등록해주세요.

---

## 추가 학습 자료

### 추천 실습 순서

1. **Week 1**: 서버 접속 → GPU 확인 → Ollama 설치 → Phi-4 실행
2. **Week 2**: Python 환경 구축 → PyTorch 설치 → 간단한 모델 학습
3. **Week 3**: GPT-OSS 120B 실행 → API 활용 → 모니터링 도구 설정
4. **Week 4**: DeepSeek-R1 실험 → 프로젝트 적용 → 최적화

### 유용한 명령어 모음

```bash
# GPU 모니터링 원라이너
watch -n 1 'nvidia-smi && echo "---" && ollama list'

# 디스크 사용량 정렬
du -h --max-depth=1 | sort -hr | head -10

# Ollama 프로세스 확인
ps aux | grep ollama

# 모델 파일 위치 확인
find ~ -name "*.bin" -o -name "*.gguf" 2>/dev/null

# GPU 온도만 추출
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

# 전체 시스템 리소스 한눈에
echo "=== GPU ===" && nvidia-smi && echo "=== DISK ===" && df -h && echo "=== MEMORY ===" && free -h
```

### 다음 단계

H100 기본 사용이 익숙해지면 다음을 시도해보세요:

1. **Fine-tuning**: 자신의 데이터로 모델 파인튜닝
2. **RAG 구축**: Vector DB + LLM으로 지식 기반 시스템 구축
3. **Multi-Modal**: 이미지-텍스트 모델 (LLaVA, CLIP) 실험
4. **Deployment**: FastAPI로 REST API 서비스 구축
5. **Monitoring**: Prometheus + Grafana로 모니터링 대시보드 구축
