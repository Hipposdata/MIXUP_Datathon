# 2025 MixUP AI Datathon Track 2 대회 정리
2025년 5월 참여했던 '2025 MixUP AI Datathon Track 2' 대회 코드 파일 정리 및 기록입니다.

## 1. 대회 개요

- **대회명:** 2025 MixUP AI Datathon Track 2
- **주최/주관:** Prometheus X BITAmin X TOBIG’s
- **후원/협찬:** upstage, WIZCORE, kpc, 한국생산성본부, 모두의연구소, 유가네닭갈비, MONSTER ENERGY
- **참여 기간:** 5월 12일 (월) 10:00 ~ 5월 18일 (일) 7:00 
- **대회 링크:** [공식 대회 페이지](https://www.kaggle.com/competitions/2025-mix-up-ai-datathon-track2/overview)
- **문제 유형:** 과제 1: 철강 공정 생산품의 OK/NG 예측 모델 개발 / 과제 2: 데이터 기반 생산공정 최적화 및 운용 전략 수립

---

## 2. 대회 문제 및 데이터 설명

### 문제 배경 및 목적
- 공정 파라미터에 따른 생산품의 품질 여부(OK/NG)를 예측하는 모델 개발 및 생산공정의 최적화 또는 운용 전략을 제시

### 데이터 구성
- **학습 데이터:**
track2_train_participant.csv
학습용 데이터 (1,840행, 46개 컬럼)

- **테스트 데이터:**
track2_test_participant.csv
평가용 데이터 (461행, 44개 컬럼)
  
- **추가 데이터:** 
track2_metadata.xlsx
생산품 size별 Ampere 표준화에 대한 메타데이터

### 평가 지표
- F1 Score: 40%
- ROC-AUC Score: 60%

## 3. 데이터 분석 및 전처리

### EDA(탐색적 데이터 분석)
- Pandas Profiling Report 활용
- 
-> 아무런 전처리 없이도 성능 높게 나옴 (0.95이상) -> 피쳐엔지니어링을 통한 파생변수 선정

    
### 결측치 및 이상치 처리
- 

### 특성 엔지니어링
- 

---

## 4. 모델링 및 실험

### 모델 선정 및 하이퍼파라미터 튜닝
- 

### 교차 검증 전략
- 

### 평가지표 및 손실함수
- 

---

## 5. 결과 및 리뷰

### 최종 모델 및 성능
- 

### 성능 개선을 위한 시도
- 

### 한계점 및 개선 아이디어
- 

---
### 참고 자료
- 

---

## 8. 기타

### 느낀 점/후기
- 

### 활용 기술 스택
- 

---
