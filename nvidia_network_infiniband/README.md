# NVIDIA InfiniBand 완벽 가이드

> 데이터센터 및 HPC 환경을 위한 초고속 인터커넥트 기술의 구축, 운영, 모니터링 가이드

## 목차

1. [개요](#개요)
2. [InfiniBand vs Ethernet](#infiniband-vs-ethernet)
3. [하드웨어 구성 요소](#하드웨어-구성-요소)
4. [아키텍처 및 프로토콜](#아키텍처-및-프로토콜)
5. [구축 가이드](#구축-가이드)
6. [운영 가이드](#운영-가이드)
7. [모니터링 및 옵저버빌리티](#모니터링-및-옵저버빌리티)
8. [트러블슈팅](#트러블슈팅)
9. [참고 자료](#참고-자료)

---

## 개요

### InfiniBand란?

**InfiniBand(IB)**는 데이터센터와 HPC(High-Performance Computing) 환경에서 사용되는 초고속 네트워크 인터커넥트 기술입니다. 서버, 스토리지, GPU 서버들을 **낮은 지연시간(Low Latency)**과 **높은 대역폭(High Bandwidth)**으로 연결하여 대규모 연산(특히 AI 학습, 분산 처리)을 빠르게 수행할 수 있게 해주는 "전용 고속도로"입니다.

### 왜 InfiniBand인가?

| 특성 | 설명 |
|------|------|
| **초저지연** | 마이크로초(μs) 단위의 지연시간 |
| **고대역폭** | NDR 기준 400Gbps, XDR 기준 800Gbps |
| **RDMA 지원** | CPU 개입 없이 메모리 간 직접 데이터 전송 |
| **GPU Direct** | GPU 메모리 간 직접 통신으로 AI 학습 가속 |

### 주요 사용 사례

- **AI/ML 클러스터**: 대규모 딥러닝 모델 분산 학습
- **HPC 클러스터**: 과학 시뮬레이션, 기상 예측
- **고성능 스토리지**: NVMe-oF, 병렬 파일 시스템
- **금융 거래 시스템**: 초저지연이 필요한 트레이딩 시스템

---

## InfiniBand vs Ethernet

### 비교 표

| 특성 | InfiniBand | Ethernet |
|------|------------|----------|
| **주요 용도** | HPC, AI/ML, 저지연 워크로드 | 범용 네트워크, 웹 트래픽 |
| **지연시간** | ~1μs (RDMA) | ~10-50μs (TCP/IP) |
| **대역폭** | 최대 800Gbps (XDR) | 최대 400Gbps |
| **프로토콜** | Native RDMA | TCP/IP, RoCE |
| **CPU 오버헤드** | 매우 낮음 | 상대적으로 높음 |
| **생태계** | 전용 스위치/NIC/케이블 | 범용 장비 |
| **비용** | 높음 | 상대적으로 낮음 |
| **관리 복잡도** | 높음 | 낮음 |

### 언제 InfiniBand를 선택해야 하는가?

```
✅ InfiniBand 선택 조건:
   - 노드 간 통신이 병목인 분산 학습/연산
   - 마이크로초 단위 지연시간이 필요한 경우
   - GPU 클러스터에서 NCCL/GPUDirect 활용 시
   - 대규모 병렬 파일 시스템 연결 시

❌ Ethernet이 더 적합한 경우:
   - 범용 서버/웹 서비스
   - 비용 효율이 중요한 경우
   - 기존 네트워크 인프라 활용 시
```

---

## 하드웨어 구성 요소

### 1. InfiniBand 스위치

데이터센터 랙에 장착되는 네트워크 스위치로, 전면에 QSFP/OSFP 포트가 배열되어 있습니다.

**주요 제품군:**
- NVIDIA Quantum-2 (NDR 400G)
- NVIDIA Quantum (HDR 200G)
- NVIDIA Switch-IB 2 (EDR 100G)

**스위치 구성 예시:**

```
NVIDIA Quantum-2 NDR InfiniBand Switch
======================================
포트 구성: [OSFP] [OSFP] [OSFP] [OSFP] [OSFP] [OSFP] ...
           Port1  Port2  Port3  Port4  Port5  Port6

스펙:
  - 64 ports x 400Gbps NDR
  - 51.2 Tb/s aggregate bandwidth
  - Sub-microsecond latency
```

### 2. InfiniBand NIC (Host Channel Adapter)

서버 PCIe 슬롯에 장착하는 네트워크 어댑터 카드입니다.

**주요 제품군 (NVIDIA/Mellanox ConnectX 시리즈):**

| 모델 | 속도 | PCIe | 포트 |
|------|------|------|------|
| ConnectX-7 | NDR 400Gbps | PCIe 5.0 x16 | 1-2 OSFP |
| ConnectX-6 | HDR 200Gbps | PCIe 4.0 x16 | 1-2 QSFP56 |
| ConnectX-5 | EDR 100Gbps | PCIe 3.0 x16 | 1-2 QSFP28 |

**NIC 구성도:**

```
ConnectX-7 NDR Adapter
======================
인터페이스: PCIe 5.0 x16 Edge Connector
포트 구성:  [OSFP Port 1] [OSFP Port 2]
             400Gbps       400Gbps
```

### 3. InfiniBand 케이블

**케이블 유형:**

| 유형 | 설명 | 거리 | 용도 |
|------|------|------|------|
| DAC (Direct Attach Copper) | 구리 케이블 | ~3m | 랙 내 연결 |
| AOC (Active Optical Cable) | 능동 광 케이블 | ~100m | 랙 간 연결 |
| Transceiver + Fiber | 분리형 광 모듈 | 수 km | 장거리 연결 |

**커넥터 규격:**
- QSFP28: EDR 100Gbps
- QSFP56: HDR 200Gbps
- OSFP/QSFP112: NDR 400Gbps

---

## 아키텍처 및 프로토콜

### InfiniBand 프로토콜 스택

```
Application Layer        --> MPI, NCCL, Storage Protocols
        |
        v
Upper Layer Protocol     --> IPoIB, SRP, iSER, NVMe-oF
        |
        v
Transport Layer          --> RC, UC, UD, XRC Queue Pairs
        |
        v
Network Layer            --> GRH, Routing, Subnet Management
        |
        v
Link Layer               --> LRH, Flow Control, VL
        |
        v
Physical Layer           --> Signaling, Encoding, Cables
```

### RDMA (Remote Direct Memory Access)

RDMA는 InfiniBand의 핵심 기능으로, CPU 개입 없이 원격 메모리에 직접 접근합니다.

**RDMA 동작 방식:**

```
Node A                                      Node B
------                                      ------
Memory  <-------- RDMA Write --------- Memory
   ^                                      |
   +--------- RDMA Read ------------------+

HCA  <=========== CPU Bypass ===========> HCA
```

### Queue Pair (QP) 개념

QP는 InfiniBand 통신의 기본 단위입니다.

**QP 유형:**

| 유형 | 설명 | 연결 모드 | 신뢰성 |
|------|------|-----------|--------|
| RC (Reliable Connected) | 1:1 연결, 재전송 지원 | Connected | 신뢰성 보장 |
| UC (Unreliable Connected) | 1:1 연결, 재전송 없음 | Connected | 비신뢰성 |
| UD (Unreliable Datagram) | 1:N 통신 | Connectionless | 비신뢰성 |
| XRC (Extended RC) | 다중 프로세스 공유 | Connected | 신뢰성 보장 |

**QP 상태 전이:**

```
RESET
  | ibv_modify_qp(INIT)
  v
INIT
  | ibv_modify_qp(RTR)
  v
RTR (Ready to Receive)
  | ibv_modify_qp(RTS)
  v
RTS (Ready to Send)
  | Error Event
  v
ERROR ---> ibv_modify_qp(RESET) ---> RESET (재시작)
```

### 연결 복구 특성 (TCP와의 차이)

**중요**: InfiniBand/RDMA는 TCP와 다르게 자동 복구가 되지 않습니다.

**TCP 연결 끊김 시:**
- 커널이 자동으로 재전송/복구 시도
- 애플리케이션은 대부분 인지하지 못함

**RDMA(InfiniBand) 연결 끊김 시:**
1. QP가 ERROR 상태로 전이
2. Posted Work Request들이 flush됨
3. 애플리케이션이 직접 감지해야 함
4. QP를 Reset → Init → RTR → RTS로 재설정
5. 또는 QP destroy 후 재생성

**관련 이벤트:**
- `IBV_EVENT_QP_FATAL`: QP 치명적 오류 → 수동 복구 필요
- `RDMA_CM_EVENT_DISCONNECTED`: 연결 끊김 → QP가 error state로 전이
- `IBV_EVENT_PORT_ERR`: 포트 다운 → 링크 복구 후에도 retry 초과 가능

---

## 구축 가이드

### 1. 하드웨어 설치

#### 1.1 NIC 설치

```bash
# 1. 서버 전원 OFF 후 PCIe 슬롯에 NIC 장착
# 2. 서버 부팅 후 장치 인식 확인

# NIC 인식 확인
lspci | grep Mellanox

# 출력 예시:
# 3b:00.0 Infiniband controller: Mellanox Technologies MT28908 Family [ConnectX-6]
```

#### 1.2 케이블 연결

**권장 케이블 선택 기준:**

| 거리 | 케이블 유형 | 비고 |
|------|-------------|------|
| ~3m | DAC | 랙 내 연결 |
| 3m~30m | AOC | 랙 간 연결 |
| 30m~2km | Transceiver | 빌딩 간 연결 |

### 2. 소프트웨어 설치

#### 2.1 MLNX_OFED 드라이버 설치 (RHEL/CentOS)

```bash
# 1. 드라이버 다운로드
wget https://content.mellanox.com/ofed/MLNX_OFED-24.10-1.1.4.0/MLNX_OFED_LINUX-24.10-1.1.4.0-rhel9.4-x86_64.tgz

# 2. 압축 해제
tar -xvf MLNX_OFED_LINUX-24.10-1.1.4.0-rhel9.4-x86_64.tgz
cd MLNX_OFED_LINUX-24.10-1.1.4.0-rhel9.4-x86_64

# 3. 설치 (전체 기능)
./mlnxofedinstall --all

# 4. 드라이버 로드
/etc/init.d/openibd restart

# 5. 설치 확인
ofed_info -s
# 출력 예시: MLNX_OFED_LINUX-24.10-1.1.4.0
```

#### 2.2 Ubuntu/Debian 설치

```bash
# 1. 드라이버 다운로드 및 압축 해제
wget https://content.mellanox.com/ofed/MLNX_OFED-24.10-1.1.4.0/MLNX_OFED_LINUX-24.10-1.1.4.0-ubuntu24.04-x86_64.tgz
tar -xvf MLNX_OFED_LINUX-24.10-1.1.4.0-ubuntu24.04-x86_64.tgz
cd MLNX_OFED_LINUX-24.10-1.1.4.0-ubuntu24.04-x86_64

# 2. 설치
./mlnxofedinstall --all

# 3. 드라이버 로드
/etc/init.d/openibd restart
```

### 3. 서브넷 매니저(SM) 구성

InfiniBand 네트워크는 반드시 하나 이상의 Subnet Manager가 필요합니다.

#### 3.1 OpenSM 설치 및 구성

```bash
# OpenSM 서비스 활성화
systemctl enable opensm
systemctl start opensm

# OpenSM 상태 확인
systemctl status opensm

# 로그 확인
tail -f /var/log/opensm.log
```

#### 3.2 OpenSM 설정 파일 (`/etc/opensm/opensm.conf`)

```conf
# 기본 설정
subnet_prefix 0xfe80000000000000
sm_priority 14
log_file /var/log/opensm.log
log_flags 0x03

# QoS 활성화
qos TRUE
qos_policy_file /etc/opensm/qos-policy.conf

# Routing 알고리즘 (대규모 클러스터용)
routing_engine ftree
```

### 4. IPoIB (IP over InfiniBand) 구성

#### 4.1 네트워크 인터페이스 설정

```bash
# IPoIB 인터페이스 확인
ip link show | grep ib

# IP 주소 설정 (수동)
ip addr add 10.0.0.1/24 dev ib0
ip link set ib0 up

# 또는 NetworkManager 사용 (RHEL/CentOS)
nmcli connection add type infiniband con-name ib0 ifname ib0 ip4 10.0.0.1/24
nmcli connection up ib0
```

#### 4.2 /etc/sysconfig/network-scripts/ifcfg-ib0 (RHEL 계열)

```conf
DEVICE=ib0
TYPE=InfiniBand
BOOTPROTO=static
IPADDR=10.0.0.1
NETMASK=255.255.255.0
ONBOOT=yes
CONNECTED_MODE=yes
MTU=65520
```

### 5. GPUDirect RDMA 구성 (AI/ML 클러스터)

#### 5.1 요구사항

- NVIDIA GPU (Volta 이상 권장)
- NVIDIA Driver 450.x 이상
- CUDA 11.0 이상
- MLNX_OFED with GPUDirect 지원

#### 5.2 GPUDirect 활성화

```bash
# nvidia-peermem 모듈 로드
modprobe nvidia-peermem

# 부팅 시 자동 로드
echo "nvidia-peermem" >> /etc/modules-load.d/nvidia-peermem.conf

# 확인
lsmod | grep nvidia_peermem
```

#### 5.3 NCCL 환경 변수 설정

```bash
# ~/.bashrc 또는 작업 스크립트에 추가
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0:1
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=INFO
```

---

## 운영 가이드

### 1. 기본 상태 확인 명령어

#### 1.1 HCA(NIC) 상태 확인

```bash
# HCA 목록 확인
ibstat
# 출력 예시:
# CA 'mlx5_0'
#     CA type: MT4123
#     Number of ports: 1
#     Port 1:
#         State: Active
#         Physical state: LinkUp
#         Rate: 200 Gb/sec (4X HDR)
#         Base lid: 1
#         LMC: 0
#         SM lid: 1
#         Capability mask: 0x2651e84a
#         Port GUID: 0x1070fd0300123456

# 상세 정보
ibv_devinfo

# 펌웨어 버전 확인
ibv_devinfo -v | grep fw_ver
```

#### 1.2 네트워크 토폴로지 확인

```bash
# 서브넷의 모든 노드 확인
ibhosts

# 스위치 목록
ibswitches

# 전체 토폴로지
iblinkinfo

# 특정 노드 경로 확인
ibtracert <src_lid> <dst_lid>
```

#### 1.3 성능 테스트

```bash
# 대역폭 테스트 (Server)
ib_write_bw -d mlx5_0

# 대역폭 테스트 (Client)
ib_write_bw -d mlx5_0 <server_ip>

# 지연시간 테스트 (Server)
ib_write_lat -d mlx5_0

# 지연시간 테스트 (Client)
ib_write_lat -d mlx5_0 <server_ip>

# RDMA 읽기 성능
ib_read_bw -d mlx5_0 <server_ip>

# 예상 결과 (HDR 200Gbps):
# Bandwidth: ~24 GB/sec (단방향)
# Latency: ~1.5 μs
```

### 2. 포트 및 링크 관리

#### 2.1 포트 상태 변경

```bash
# 포트 비활성화
ibportstate -D 0 <port_num> disable

# 포트 활성화
ibportstate -D 0 <port_num> enable

# 포트 리셋
ibportstate -D 0 <port_num> reset
```

#### 2.2 링크 속도 확인/변경

```bash
# 현재 링크 속도 확인
ibstat mlx5_0 | grep Rate

# 지원 속도 확인
ibv_devinfo -d mlx5_0 | grep active_speed

# 속도 제한 설정 (필요시)
mlxconfig -d mlx5_0 set LINK_TYPE_P1=IB
```

### 3. 서브넷 매니저 운영

#### 3.1 SM 상태 확인

```bash
# SM 상태
sminfo

# SM 우선순위 확인
smpquery portinfo -D 0 | grep -i sm

# 여러 SM 있을 때 마스터 확인
ibdiagnet --sm
```

#### 3.2 SM Failover 구성

```bash
# 주 SM (Priority 높음)
# /etc/opensm/opensm.conf
sm_priority 14

# 백업 SM (Priority 낮음)
# /etc/opensm/opensm.conf
sm_priority 8

# SM handover 강제
opensm --force_heavy_sweep
```

### 4. QoS (Quality of Service) 설정

#### 4.1 Virtual Lane 구성

```bash
# /etc/opensm/qos-policy.conf
qos-levels
    qos-level default
        sl 0
        mtu 2048
        rate 25
    end-qos-level

    qos-level high-priority
        sl 1
        mtu 4096
        rate 100
    end-qos-level
end-qos-levels

qos-match-rules
    qos-match-rule gpu-traffic
        source gpu-nodes
        destination gpu-nodes
        qos-level high-priority
    end-qos-match-rule
end-qos-match-rules
```

### 5. 펌웨어 관리

#### 5.1 펌웨어 업데이트

```bash
# 현재 펌웨어 버전 확인
mlxfwmanager --query

# 펌웨어 업데이트
mlxfwmanager --update --yes

# 또는 특정 펌웨어 파일로
flint -d mlx5_0 -i <firmware_file.bin> burn

# 업데이트 후 재부팅 또는 드라이버 리로드
mlxfwreset -d mlx5_0 reset
```

---

## 모니터링 및 옵저버빌리티

### 1. 기본 모니터링 도구

#### 1.1 perfquery - 성능 카운터

```bash
# 포트 카운터 조회
perfquery -x

# 특정 포트 카운터
perfquery -x <lid> <port>

# 주요 카운터:
# - PortXmitData: 전송 데이터량
# - PortRcvData: 수신 데이터량
# - PortXmitPkts: 전송 패킷 수
# - PortRcvPkts: 수신 패킷 수

# 카운터 리셋
perfquery -R
```

#### 1.2 ibdiagnet - 네트워크 진단

```bash
# 전체 네트워크 진단
ibdiagnet

# 출력 파일 위치 지정
ibdiagnet -o /var/log/ibdiagnet/

# 결과 확인
cat /var/log/ibdiagnet/ibdiagnet2.log

# 주요 검사 항목:
# - 토폴로지 일관성
# - 라우팅 테이블
# - 에러 카운터
# - 성능 문제
```

### 2. Prometheus + Grafana 모니터링

#### 2.1 InfiniBand Exporter 설치

```bash
# NVIDIA DCGM (GPU + InfiniBand 통합)
docker run -d --gpus all -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04

# 또는 ib_exporter (InfiniBand 전용)
git clone https://github.com/treydock/infiniband_exporter.git
cd infiniband_exporter
make build
./infiniband_exporter --web.listen-address=":9315"
```

#### 2.2 Prometheus 설정

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'infiniband'
    static_configs:
      - targets:
        - 'ib-node-01:9315'
        - 'ib-node-02:9315'
        - 'ib-node-03:9315'

  - job_name: 'dcgm'
    static_configs:
      - targets:
        - 'gpu-node-01:9400'
        - 'gpu-node-02:9400'
```

#### 2.3 주요 메트릭

```
# 포트 상태
infiniband_port_state{port="1",device="mlx5_0"} 4  # 4=Active

# 데이터 전송량 (bytes)
infiniband_port_transmit_data_bytes_total

# 수신량 (bytes)
infiniband_port_receive_data_bytes_total

# 에러 카운터
infiniband_port_symbol_error_total
infiniband_port_link_error_recovery_total
infiniband_port_link_down_total

# GPU Direct 메트릭 (DCGM)
DCGM_FI_PROF_NVLINK_RX_BYTES
DCGM_FI_PROF_NVLINK_TX_BYTES
```

#### 2.4 Grafana 대시보드

```json
{
  "dashboard": {
    "title": "InfiniBand Cluster Monitoring",
    "panels": [
      {
        "title": "Port Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(infiniband_port_transmit_data_bytes_total[5m])",
            "legendFormat": "TX {{device}}:{{port}}"
          },
          {
            "expr": "rate(infiniband_port_receive_data_bytes_total[5m])",
            "legendFormat": "RX {{device}}:{{port}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(infiniband_port_symbol_error_total[5m])",
            "legendFormat": "Symbol Errors"
          }
        ]
      }
    ]
  }
}
```

### 3. 실시간 모니터링 스크립트

#### 3.1 대역폭 모니터링

```bash
#!/bin/bash
# ib_bandwidth_monitor.sh

DEVICE="mlx5_0"
INTERVAL=1

while true; do
    # 현재 카운터 값
    TX=$(perfquery -x -d $DEVICE 1 | grep PortXmitData | awk '{print $2}')
    RX=$(perfquery -x -d $DEVICE 1 | grep PortRcvData | awk '{print $2}')

    sleep $INTERVAL

    # 새 카운터 값
    TX_NEW=$(perfquery -x -d $DEVICE 1 | grep PortXmitData | awk '{print $2}')
    RX_NEW=$(perfquery -x -d $DEVICE 1 | grep PortRcvData | awk '{print $2}')

    # 대역폭 계산 (bytes -> Gbps)
    TX_BW=$(echo "scale=2; ($TX_NEW - $TX) * 8 / 1000000000 / $INTERVAL" | bc)
    RX_BW=$(echo "scale=2; ($RX_NEW - $RX) * 8 / 1000000000 / $INTERVAL" | bc)

    echo "$(date '+%H:%M:%S') TX: ${TX_BW} Gbps, RX: ${RX_BW} Gbps"
done
```

#### 3.2 에러 모니터링 및 알림

```bash
#!/bin/bash
# ib_error_monitor.sh

THRESHOLD=100
SLACK_WEBHOOK="https://hooks.slack.com/services/xxx"

check_errors() {
    local device=$1
    local errors=$(perfquery -x -d $device 1 | grep -E "SymbolError|LinkRecover|RcvErr" | awk '{sum+=$2} END {print sum}')
    echo $errors
}

while true; do
    for device in $(ibstat -l); do
        errors=$(check_errors $device)

        if [ "$errors" -gt "$THRESHOLD" ]; then
            message="⚠️ InfiniBand Alert: $device has $errors errors"
            curl -X POST -H 'Content-type: application/json' \
                --data "{\"text\":\"$message\"}" \
                $SLACK_WEBHOOK
        fi
    done

    sleep 60
done
```

### 4. UFM (Unified Fabric Manager) 엔터프라이즈 모니터링

NVIDIA UFM은 대규모 InfiniBand 클러스터의 엔터프라이즈급 모니터링 솔루션입니다.

**UFM 주요 기능:**
- 실시간 토폴로지 시각화
- 자동 이상 탐지 및 알림
- 성능 분석 및 트렌드
- 구성 관리 및 자동화
- RESTful API 제공
- 멀티 서브넷 관리

#### UFM REST API 예시

```bash
# 패브릭 상태 조회
curl -X GET "https://ufm-server/ufmRest/resources/systems" \
    -H "Authorization: Basic <credentials>"

# 포트 카운터 조회
curl -X GET "https://ufm-server/ufmRest/monitoring/ports/counters" \
    -H "Authorization: Basic <credentials>"

# 이벤트 조회
curl -X GET "https://ufm-server/ufmRest/app/events?limit=100" \
    -H "Authorization: Basic <credentials>"
```

---

## 트러블슈팅

### 1. 일반적인 문제 및 해결

#### 1.1 포트가 Active 상태가 아닌 경우

```bash
# 증상
ibstat
# Port 1:
#     State: Down
#     Physical state: Polling

# 진단 단계
# 1. 케이블 연결 확인
# 2. 스위치 포트 상태 확인
# 3. 드라이버 로그 확인
dmesg | grep mlx5

# 해결책
# - 케이블 재연결
# - 포트 리셋
ibportstate -D 0 1 reset

# - 드라이버 재로드
modprobe -r mlx5_ib mlx5_core
modprobe mlx5_core
```

#### 1.2 성능 저하

```bash
# 증상: 예상 대역폭의 50% 미만

# 진단
# 1. 링크 속도 확인
ibstat | grep Rate

# 2. 에러 카운터 확인
perfquery -x | grep -i error

# 3. PCIe 대역폭 확인
lspci -vvv -s <device_id> | grep -i width

# 해결책
# - 케이블 교체 (물리적 문제)
# - PCIe 슬롯 확인 (x16 필요)
# - MTU 설정 확인
ip link show ib0 | grep mtu
```

#### 1.3 RDMA 연결 실패

```bash
# 증상: QP 생성 또는 연결 실패

# 진단
# 1. RDMA CM 로그 확인
dmesg | grep rdma

# 2. 방화벽 확인 (RDMA CM 포트)
iptables -L | grep 4791

# 3. IPoIB 연결 테스트
ping -I ib0 <remote_ip>

# 해결책
# - 방화벽에서 RDMA 포트 허용
iptables -A INPUT -p udp --dport 4791 -j ACCEPT

# - SM 상태 확인
sminfo
```

### 2. 에러 카운터 해석

| 카운터 | 의미 |
|--------|------|
| SymbolErrorCounter | 물리적 신호 문제 (케이블/커넥터) |
| LinkErrorRecovery | 링크 복구 횟수 (일시적 문제) |
| LinkDownedCounter | 링크 다운 횟수 (심각한 문제) |
| PortRcvErrors | 수신 에러 (CRC 등) |
| PortRcvRemotePhysErr | 원격 물리 에러 |
| PortXmitDiscards | 전송 폐기 (버퍼 오버플로우) |
| VL15Dropped | 관리 패킷 손실 |

### 3. 로그 파일 위치

```bash
# 시스템 로그
/var/log/messages          # RHEL/CentOS
/var/log/syslog            # Ubuntu/Debian

# OpenSM 로그
/var/log/opensm.log

# MLNX_OFED 로그
/var/log/mlnx_ofed_install.log

# 진단 출력
/var/log/ibdiagnet/

# 커널 메시지
dmesg | grep -E "mlx5|ib_|rdma"
```

### 4. 클러스터 장애 대응

#### 4.1 노드 장애 시 (AI/ML 클러스터)

```bash
# NCCL/MPI 작업 실패 시 일반적인 흐름
1. 작업(Job) 실패로 종료
2. 스케줄러(Slurm/K8s)가 감지
3. 체크포인트에서 작업 재시작

# Slurm에서 노드 상태 확인
sinfo -N -l | grep <node_name>

# 노드 드레인 (유지보수)
scontrol update nodename=<node> state=drain reason="IB maintenance"

# 노드 복구 후
scontrol update nodename=<node> state=resume
```

#### 4.2 스위치 장애 시

```bash
# 토폴로지 변경 확인
iblinkinfo > topology_new.txt
diff topology_old.txt topology_new.txt

# 대체 경로 확인
ibtracert <src_lid> <dst_lid>

# SM이 경로 재계산하도록 강제
opensm --force_heavy_sweep
```

---

## 참고 자료

### 공식 문서

- [NVIDIA MLNX_OFED Documentation](https://docs.nvidia.com/networking/display/MLNXOFEDv24101140)
- [NVIDIA InfiniBand Documentation](https://docs.nvidia.com/networking/category/infinibandproducts)
- [RDMA Core Userspace Library](https://github.com/linux-rdma/rdma-core)
- [OpenSM Documentation](https://docs.nvidia.com/networking/display/opensm)

### 유용한 명령어 요약

```bash
# 상태 확인
ibstat                    # HCA 상태
ibstatus                  # 포트 상태 요약
ibv_devinfo              # 장치 상세 정보

# 네트워크 토폴로지
ibhosts                   # 호스트 목록
ibswitches               # 스위치 목록
iblinkinfo               # 링크 정보

# 성능 테스트
ib_write_bw              # 쓰기 대역폭
ib_read_bw               # 읽기 대역폭
ib_write_lat             # 쓰기 지연시간
ib_read_lat              # 읽기 지연시간

# 진단
ibdiagnet                # 네트워크 진단
perfquery                # 성능 카운터

# 관리
ibportstate              # 포트 상태 변경
sminfo                   # SM 정보
```

### 버전 정보

| 구성 요소 | 권장 버전 |
|-----------|-----------|
| MLNX_OFED | 24.10-1.1.4.0 이상 |
| Kernel | 5.15 이상 |
| NVIDIA Driver | 550.x 이상 (GPUDirect용) |
| CUDA | 12.x 이상 |

