# 2025 MixUP AI Datathon Track 2 대회 정리

2025년 5월 참여한 '2025 MixUP AI Datathon Track 2' 대회의 코드 파일 정리 및 기록 문서입니다.

## 📋 대회 개요

- **대회명**: 2025 MixUP AI Datathon Track 2
- **주최/주관**: Prometheus X BITAmin X TOBIG's
- **후원/협찬**: upstage, WIZCORE, kpc, 한국생산성본부, 모두의연구소, 유가네닭갈비, MONSTER ENERGY
- **참여 기간**: 2025년 5월 12일 (월) 10:00 ~ 5월 18일 (일) 7:00
- **대회 링크**: [공식 대회 페이지](https://www.kaggle.com/competitions/2025-mix-up-ai-datathon-track2/overview)
- **문제 유형**: 
  - **과제 1**: 철강 공정 생산품의 OK/NG 예측 모델 개발
  - **과제 2**: 데이터 기반 생산공정 최적화 및 운용 전략 수립

## 🎯 문제 정의 및 데이터 설명

### 문제 배경
공정 파라미터에 따른 생산품의 품질 여부(OK/NG)를 예측하는 모델 개발 및 생산공정의 최적화 또는 운용 전략을 제시하는 것이 목표입니다.

### 데이터 구성

#### 학습 데이터
- **파일명**: `track2_train_participant.csv`
- **규모**: 1,840행, 46개 컬럼
![image](https://github.com/user-attachments/assets/5b9a0821-9ca6-4a87-90c1-e22fc5b8b012)


#### 주요 변수 설명
| 변수 그룹 | 변수명 | 설명 |
|-----------|--------|------|
| 물리적 규격 | 두께, 소재폭, 제품폭 | 코일/강판의 실제 물리적 규격 (품질 및 생산공정에 직접적 영향) |
| 공정 여유 | 마진 | 소재폭-제품폭, 공정상 절단·가공 여유 |
| 그룹화 변수 | 두께그룹, 폭그룹, 마진그룹 | 품질관리, 공정제어, 표준화 목적의 구간화된 그룹 |
| 제품 특성 | 강종, 품명그룹, 품명 | 제품의 등급, 표면처리, 용도 등 품질 특성 |
| 공정 조건 | 장력, 스피드 | 라인 운전 조건 (생산성, 품질에 영향) |
| 표면 처리 | 도유 | 오일링 방식/여부, 표면 품질에 영향 |
| 생산성 지표 | 생산중량, 수율 | 생산성과 효율성 지표 |
| 타겟 변수 | OK, NG | 최종 품질 판정 (train 데이터에만 존재) |

#### 테스트 데이터
- **파일명**: `track2_test_participant.csv`
- **규모**: 461행, 44개 컬럼

#### 메타데이터
- **파일명**: `track2_metadata.xlsx`
- **내용**: 생산품 size별 Ampere 표준화에 대한 메타데이터

### 평가 지표
- **F1 Score**: 40%
- **ROC-AUC Score**: 60%

## 🔍 데이터 분석 및 전처리

### 탐색적 데이터 분석 (EDA)
- **Pandas Profiling Report** 활용하여 데이터 전체적인 특성 파악
<img src="https://github.com/user-attachments/assets/7c2dd911-2ff7-4339-b981-94a0efd9e3ff" width="50%" alt="Pandas Profiling Report">


- **핵심 발견사항**:
  - 특정 변수들의 Feature Importance가 매우 높음을 확인
  - 기본 데이터만으로도 높은 성능(0.95 이상) 달성 가능(전처리X, 파생변수X)  
  - 소수의 핵심 변수를 중심으로 한 피처 엔지니어링 전략 수립
<img src="https://github.com/user-attachments/assets/3f839ce0-12fd-4305-919b-1a5bcaf588ad" width="50%" alt="Feature Importance">


### 데이터 전처리

#### 결측치 처리
- **방법**: KNN 알고리즘 활용
- **근거**: 동일 공정의 제품들은 유사한 특성 조건을 보일 것이라는 가정

#### 불필요한 컬럼 제거
- **Unique 값**: No., 제품번호 등
- **상수 값**: 라인, 작업장, 제품구분 등

#### 인코딩 방법 비교
다양한 인코딩 방법의 성능을 비교 분석:
- Label Encoding
- Target Encoding  
- One-hot Encoding

### 피처 엔지니어링

#### 도메인 지식 활용
철강 공정단계에서 불량을 줄이기 위한 장력, 속도 제어를 통한 제어기술 존재 -> 장력/스피드가 청강 공정에 매우 중요한 변수임을 파악 

**참고 논문**: "A Study on Development of Advanced Tension Control Method Using Speed Controller in Wire Rod Mill" (https://oasis.postech.ac.kr/handle/2014.oak/111947)  
<img src="https://github.com/user-attachments/assets/1a20bef2-7780-460b-9cc6-c140863a45c9" width="50%" alt="피쳐 엔지니어링 과정 1">  


#### 다양한 가설기반 파생변수 생성
<img src="https://github.com/user-attachments/assets/aff69ff9-c269-4015-a16e-d641d940e754" width="50%" alt="피쳐 엔지니어링 과정 2">  

## 🤖 모델링 및 실험

### 모델링 파이프라인
1. **AutoML 탐색**: PyCaret을 활용한 초기 모델 성능 파악
2. **데이터 전처리**: 피처 엔지니어링 적용
3. **기본 모델 선정**: 초기 파라미터 튜닝
4. **피처 선택**: RFECV를 통한 최적 피처 선택
5. **모델 최적화**: 선택된 피처로 최종 튜닝
6. **앙상블 구축**: 스태킹, 보팅 등 다양한 앙상블 기법 적용
7. **최종 예측**: 최적 모델로 예측 생성

#### 다양한 모델 실험 결과     
<img src="https://github.com/user-attachments/assets/ee05e90e-14cb-4055-bc9a-a2147a5bc426" width="50%" alt="모델 실험 결과">  


### 실험 설정

#### 하이퍼파라미터 튜닝
- **도구**: Optuna 라이브러리 활용

#### 교차 검증 전략
- **방법**: Stratified K-fold (k=5)
- **근거**: 타겟 불균형 데이터에 적합한 전략

### 최종 모델 선정

#### 전략: "Most Simple + Best Performance"
- **사용 변수**: 2개 (`스피드1`, `초기장력차이(기준장력-장력1)`)
- **결측치 처리**: KNN (k=5)
- **최종 모델**: **Decision Tree** (depth=2)
- **성능**: **Perfect Score (1.0)**
  
<img src="https://github.com/user-attachments/assets/29b57660-c3f0-46ad-abe1-794bc1abfb3e" width="50%" alt="군집화 분석 결과">  
<img src="https://github.com/user-attachments/assets/98254ae2-1664-453f-a38e-a15d1764a05d" width="50%" alt="Decision Tree 모델">  


#### 모델 선정 근거
- 여러 모델 실험 결과 모두 완벽한 예측 성능(score = 1.0) 달성
- 가장 간단하면서도 해석이 용이한 Decision Tree 선택
- 실무 적용 시 직관적 이해와 설명 가능성 확보

## 📊 결과 및 인사이트

### 최종 성능
공정 과정 중 2개 변수(`초기장력차이`, `스피드1`)의 임계값을 기준으로 **완벽한 공정불량 예측** 가능
<img src="https://github.com/user-attachments/assets/e988cc86-6e71-43ab-911b-e3c40ce176cb" width="50%" alt="최종 모델 성능">


### 핵심 인사이트

#### 변수 중요도 분석
- **Variable Importance**, **SHAP 값**, **PDP Plot** 분석을 통해 초기 장력 및 초기 스피드가 불량률 절감의 핵심
<img src="https://github.com/user-attachments/assets/64194c03-0ed1-4d94-b1eb-c211a2607a16" width="50%" alt="변수 중요도 분석"> <img src="https://github.com/user-attachments/assets/e73e40df-f483-44fa-a5cf-98a5dbf4f6b8" width="50%" alt="SHAP 값 분석"> <img src="https://github.com/user-attachments/assets/52c17be0-e10c-4c2c-b74c-ddbdc3b40e5a" width="50%" alt="PDP Plot 분석">

#### 시점별 영향도 분석
장력, 스피드의 **초기값**이 중기/말기 값에 비해 불량률에 가장 큰 영향을 미침  
<img src="https://github.com/user-attachments/assets/2ca223c1-49f2-46ee-8ce4-8a14c03e048e" width="50%" alt="시점별 영향도 분석">  

#### 주요 시사점
1. **기준장력** 대비 **장력1** / **스피드1** 이 품질 결정의 핵심 요소
2. **공정 초기단계**에서 철강 불량여부 선제적 파악 가능
3. 초기 장력 및 스피드 설정 시스템을 통한 품질 안정화 필수

## 🏭 실무 적용 방안
<img src="https://github.com/user-attachments/assets/19e36d7c-05df-4488-923c-2f9d8eef04cc" width="15%" alt="실무 적용 방안">

### 대상 기업: 위즈코어 (WIZCORE)
- **업종**: 제조업 분야 데이터 분석, AI 기술 기반 스마트 팩토리 운영
- **적용 전략**: 장력 및 속도 실시간 이상 감지 및 제어 시스템 구축

### 실시간 대시보드 구현
- **목적**: 공정 초기단계에서 불량여부를 판단할 수 있는 실시간 제어 시스템
- **기술**: Streamlit 활용
- **링크**: [Steel Process Dashboard](https://mixupsteel.streamlit.app/)
<img src="https://github.com/user-attachments/assets/025fc462-720e-4b01-9764-d39af7c2db25" width="50%" alt="실시간 대시보드">

## 💡 프로젝트 회고

### 느낀 점/후기
- 문제가 공개되자 마자 스코어 1.0인 팀이 등장했기에 해당 문제는 뭔가 잘못됐다고 생각했기에 추후 대회중에 문제가 바뀌거나 데이터가 추가되겠거니 싶었다. 하지만 그대로 진행된 터라 스코어 1.0을 쉽게 달성할 수 있을거라 생각하였고 이에 최대한 간단한 모델로 최대한 좋은 성능 달성을 목표로 하였다(most simple + best performance!)
- 최종적으로 스코어 1.0 달성한 팀은 여럿 있었지만, 활용성 / 해석력 측면에서 당락이 갈린 대회였던 것 같다.
- 스코어 1.0 달성은 제일 빠르진 않았지만(3번째) 운이 좋게도 분석시 세웠던 목표(most simple + best performance!)를 위해 최대한 적은 변수(2개) 사용, 해석이 매우 용이하며 모델이 매우 직관적이고 간단한 깊이(depth = 2)의 DT(Decision Tree) 모델로 스코어 1.0을 달성했던 것과 최종적으로 streamlit을 이용해 실시간으로 데이터 입력시 철강공정 불량 여부를 판단할 수 있도록 대쉬보드를 구성한 것이 좋은 성과를 낼 수 있었던 것 같다.



### 핵심 성공 요인
1. **도메인 지식** 기반 피처 엔지니어링
2. **단순함과 성능의 균형** 추구
3. **실무 적용 가능성** 고려한 솔루션 설계
4. **시각화 및 대시보드**를 통한 결과 전달력 강화

## 🛠️ 사용 도구 및 기술 스택

### 개발 환경
- **언어**: Python
- **시각화**: Streamlit
- **협업**: Notion, VSCode

### 주요 라이브러리
- **AutoML**: PyCaret
- **하이퍼파라미터 튜닝**: Optuna
- **데이터 처리**: Pandas, NumPy
- **모델링**: Scikit-learn
- **시각화**: Matplotlib, Seaborn, Plotly

*이 프로젝트는 철강 제조 공정의 품질 예측 및 최적화를 위한 AI 솔루션 개발 사례로, 단순하면서도 효과적인 접근 방식을 통해 실무 적용 가능한 결과를 도출했습니다.*
