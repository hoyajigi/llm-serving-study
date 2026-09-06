# Hands-On LLM Serving and Optimization — 스터디 정리

LLM 서빙·최적화 스터디에 참여하며 주차별로 정리한 글입니다.

## 목차

| 주차 | 글 |
|---|---|
| 1주차 | [k_proj는 왜 [128, 896]인가](week1/README.md) |
| 2주차 | [vLLM을 붙였는데 동시 요청이 순차로 처리된다](week2/README.md) |
| 3주차 | [배치를 안 키우면 어떤 GPU를 사도 decode는 빨라지지 않는다](week3/README.md) |
| 4주차 | [추측적 디코딩은 배칭과 같은 레버를 당긴다](week4/README.md) |
| 5주차 | [양자화로 처리량이 2.7배 올랐는데 GPU는 더 빨라지지 않았다](week5/README.md) |
| 6주차 | [동시 요청을 5개 보냈는데 처리량은 정확히 4배였다](week6/README.md) |

1주차: 모델 서빙 개요, 디코더 온리 트랜스포머, KV 캐시, GQA, prefill/decode
2주차: 단일/멀티 모델 서빙 시스템 설계, 프로세스 격리, 성능 지표(E2E/TTFT/ITL, RPS/TPS)
3주차: 산술 강도와 루프라인 모델, GPU 병목 분석, 배칭·어텐션 최적화·모델 압축
4주차: 추측적 디코딩, TP/PP 멀티 GPU, prefill-decode 분리, 고급 KV 캐싱
5주차: Qwen3-14B + vLLM 실전 튜닝, 양자화·prefix caching 벤치마크 해석, 멀티 GPU 확장
6주차: AWS Trainium + vLLM on EKS 워크숍, Neuron 컴파일과 배치 상한, llmperf 지표 해석
