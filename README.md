# Hands-On LLM Serving and Optimization — 스터디 정리

LLM 서빙·최적화 스터디에 참여하며 주차별로 정리한 글입니다.

## 목차

| 주차 | 글 |
|---|---|
| 1주차 | [k_proj는 왜 [128, 896]인가](week1/README.md) |
| 2주차 | [vLLM을 붙였는데 동시 요청이 순차로 처리된다](week2/README.md) |

1주차: 모델 서빙 개요, 디코더 온리 트랜스포머, KV 캐시, GQA, prefill/decode, PagedAttention, batching
2주차: 단일/멀티 모델 서빙 시스템 설계, 프로세스 격리, 에이전틱 서빙, 성능 지표(E2E/TTFT/ITL, RPS/TPS)
