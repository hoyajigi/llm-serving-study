# 동시 요청을 5개 보냈는데 처리량은 정확히 4배였다

*Hands-On LLM Serving and Optimization 스터디 6주차 정리*

---

6주차는 책이 아니라 AWS 워크숍이다. Scaling LLM Inference with vLLM and AWS Trainium. EKS 위에 vLLM을 올리는데 가속기가 NVIDIA가 아니라 Trainium이다.

앞의 다섯 주는 전부 NVIDIA GPU를 전제로 산술 강도와 배치 크기를 따졌다. 그 계산이 다른 칩에서도 그대로 서는지 확인할 기회다.

## llmperf 숫자가 배치 상한을 그대로 뱉었다

Lab5에서 llmperf를 `--num-concurrent-requests 5`로 돌렸다. 하나 끝나면 즉시 다음 요청을 채우는 고정 동시성 워커 풀이라, 서버 입장에서는 항상 5개가 대기 중이다.

결과 중 두 줄:

```
request_output_throughput_token_per_s  mean = 84.9487
Overall Output Throughput: 340.6703
```

나눠보면 4.0103이다.

동시에 5개를 밀어 넣었는데 집계 처리량은 요청 하나 처리량의 4배다. 5배였다면 424.7이 나왔어야 하고 그건 실측보다 24.7% 높다.

4가 어디서 왔는지는 ConfigMap에 적혀 있다.

```
MAX_NUM_SEQS: "4"
```

3주차에 decode 산술 강도를 계산하면서 처리량이 배치 크기에 선형으로 붙는다는 결론을 냈고 5주차에는 양자화가 처리량을 2.7배 올린 게 배치를 2.65배 키운 것과 같은 숫자라는 걸 확인했다. 여기서는 그 관계가 나눗셈 한 번으로 보인다. 배치 상한이 4면 다섯 번째 요청은 자리가 빌 때까지 큐에서 기다린다. 그 대기가 TTFT 분포에도 남아서 p50은 0.219초인데 p95는 0.532초로 2.44배다.

## 그런데 이 4는 런타임 값이 아니다

여기서부터가 NVIDIA와 다르다. init 컨테이너가 모델을 컴파일하는 코드를 보면,

```python
LLM(model=os.environ['MODEL_NAME'],
    max_num_seqs=int(os.environ['MAX_NUM_SEQS']),
    max_model_len=int(os.environ['MAX_MODEL_LEN']),
    tensor_parallel_size=int(os.environ['TENSOR_PARALLEL_SIZE']),
    device='neuron',
    override_neuron_config={'enable_bucketing': False})
```

`MAX_NUM_SEQS`가 컴파일 입력이다. 서버 실행 인자로도 한 번 더 들어가지만 그전에 Neuron 컴파일러가 이 값으로 그래프를 굳혀서 S3에 캐시한다. `enable_bucketing: false`라 여러 shape을 미리 만들어두는 완충 장치도 꺼져 있다.

배치를 8로 올리고 싶으면 값을 바꾸는 걸로 끝나지 않는다. 캐시를 비우고 다시 컴파일해야 한다. 워크숍이 적어둔 소요 시간이 이렇다.

- 첫 배포: 이미지 pull 약 4분 + 모델 컴파일 약 4분 + 서버 기동 약 20초
- 캐시가 있을 때: 파드 시작 약 20초

3주차부터 5주차까지 배치 크기를 "당길 레버"라고 불렀는데, 이 하드웨어에서는 레버가 용접돼 있다. 튜닝 한 번이 배포 파이프라인을 한 바퀴 도는 일이 된다.

컴파일하는 init 컨테이너의 리소스 요청도 눈에 띈다.

```yaml
resources:
  limits:
    aws.amazon.com/neuron: 1
```

빌드 성격의 작업이 추론용 디바이스를 한 장 잡는다. 컴파일 중에는 그 자리에 서빙 파드가 못 뜬다. 캐시를 S3에 두는 게 시작 시간 단축만이 아니라 디바이스 점유를 줄이는 일이기도 하다.

## llmperf의 ITL은 토큰 사이 간격이 아니었다

2주차에 4장을 읽으면서 `E2E = TTFT + ITL × (N-1)`로 정리해뒀다. 이번 결과에 그대로 대입해봤다.

```
TTFT mean  0.2559 s
ITL  mean  0.011974 s
출력 토큰   99.72
→ 0.2559 + 0.011974 × 98.72 = 1.4379 s
```

실측 E2E 평균은 1.1932초다. 계산값이 20.5% 크다.

원인은 ITL의 정의다. 곱해보면 바로 나온다.

```
0.011974 × 99.72 = 1.1940 s   (실측 E2E 1.1932, 오차 0.07%)
```

llmperf의 `inter_token_latency_s`는 토큰 사이 간격이 아니라 E2E를 출력 토큰 수로 나눈 값이다. TTFT가 이미 그 안에 녹아 있다. 그러니 공식에 넣으면 TTFT를 두 번 세게 된다. 개별 요청 레코드로도 확인된다. `e2e 1.078680 / 출력 97 = 0.011120`이고 보고된 ITL이 `0.011119`다.

2주차에는 TPS가 입력 길이를 자르거나 배치 길이를 맞추는 식으로 부풀려질 수 있다고 적었다. 조작이 아니어도 이렇게 어긋난다. 도구마다 같은 이름의 지표를 다르게 정의하고 벤치마크 두 개를 나란히 놓는 순간 그 차이가 20%짜리 결론이 된다.

참고로 ITL에서 역산한 토큰 속도는 1/0.011974 = 83.5 tok/s이고 보고된 요청당 처리량은 84.95 tok/s다. 이 1.7% 차이가 TTFT 몫이다.

## 디바이스를 컨테이너에 넣는 방식이 다르다

워크숍의 심화 절이 NVIDIA와 Neuron의 차이를 정리해뒀는데, 요지는 개입 지점이다.

NVIDIA는 컨테이너 런타임 레벨에서 개입한다. device plugin이 `Allocate()`에 응답하면 `nvidia-container-runtime`이 디바이스 노드를 만들고 호스트의 `libcuda.so`를 컨테이너 안으로 bind-mount한다. 유저스페이스 라이브러리가 커널 드라이버 버전과 정확히 맞아야 해서 주입이 필요하다.

Neuron은 device plugin 레벨에서 끝난다. 커널이 이미 `/dev/neuron0`, `/dev/ngXnY`로 칩과 코어를 나눠 노출하고 SDK는 pip 패키지로 이미지 안에 들어 있다. 그래서 device plugin이 "어떤 노드를 마운트할지"만 정하면 표준 runc가 처리한다. containerd 설정에 전용 런타임 항목이 없는 이유다.

이 대목에서 지난달 홈랩 k8s에서 겪은 일이 떠올랐다. `nvidia.com/gpu`를 전혀 요청하지 않은 파드가 노드의 GPU를 전부 보고 있었다. `nvidia-smi -L`에 다 나오고 `/dev/nvidia0`도 열렸는데, 스케줄러 장부에는 GPU 0개 사용으로 적혀 있었다.

원인은 세 겹이었다. `nvidia/cuda` 베이스 이미지가 `NVIDIA_VISIBLE_DEVICES=all`을 구워서 배포하고, 호스트 설정의 `accept-nvidia-visible-devices-envvar-when-unprivileged`가 기본값 true이고, 런타임이 이미지의 env를 그대로 믿어서 kubelet의 할당 응답을 우회한다.

그때는 NVIDIA 설정 실수로 봤는데, 이번 비교를 보고 나니 위치가 달라 보인다. 런타임이 env var를 읽어 디바이스를 결정하는 경로가 있어야만 생기는 구멍이다. Neuron 경로에는 그 경로 자체가 없다. 마운트할 디바이스 노드가 곧 할당 결과라서 이미지가 무슨 환경변수를 들고 있든 바뀔 게 없다.

가벼운 통합이 보안 면에서 덜 미끄럽다는 뜻이기도 하다. 대신 어느 노드에 코어가 몇 개 남았는지 같은 세밀한 스케줄링은 별도 확장으로 메워야 하고 워크숍도 그래서 커스텀 스케줄러를 따로 붙인다.

## 아직 안 해본 것

`MAX_NUM_SEQS`를 8이나 16으로 올려 재컴파일하면 340 tok/s가 얼마나 오르는지 못 재봤다. TinyLlama 1.1B에 `MAX_MODEL_LEN`이 1024라 메모리는 남아 보이는데, Neuron은 shape이 정적이라 KV 캐시도 컴파일 시점에 자리를 잡는다. 선형으로 따라 오르다가 어디서 컴파일이 실패하는지, 그 지점이 3주차에 계산한 산술 강도 곡선의 어디쯤인지가 궁금하다.

Lab6의 HPA는 계정 vCPU 한도 8에 걸려 못 돌렸다. 파드를 늘리는 확장은 배치 상한이 컴파일에 묶인 이 구조에서 특히 의미가 다를 것 같다. 한 파드 안에서 배치를 못 키우니 수평 확장이 사실상 유일한 처리량 레버가 된다.

---

*Study: Hands-On LLM Serving and Optimization*
*6주차 — [AWS Workshop] Scaling LLM Inference with vLLM and AWS Trainium*
