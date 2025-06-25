# web-crawling

# 리뷰 감정 분석기 & 워드클라우드 생성기

본 프로젝트는 웹에서 수집한 음식점 리뷰 데이터를 기반으로, 감정을 자동으로 분류하고 키워드를 시각화하는 자연어처리 기반 웹 애플리케이션입니다.  
`딥러닝 기반 자연어처리` 모델을 활용하여 문장의 감정을 부정 / 중립 / 긍정으로 분류하며, 리뷰 전체에 대한 워드클라우드 시각화 및 통계 분석 기능도 함께 제공합니다.

---

## 📌 프로젝트 목적

- 실사용 가능한 감정 분석 웹앱 구현
- 다양한 모델을 실험하고 비교하여 최적 모델 선정 (KoELECTRA)
- 웹에서 수집한 리뷰 데이터로 자연어처리(NLP) 파이프라인 실습
- 감정 분석 결과를 직관적으로 확인할 수 있는 워드클라우드 및 감정 분포 시각화 제공

---


## 시스템 구성도

`plots/system_chart.png` 

---

## 📁 프로젝트 구조

web-crawling/
├── data/ # 크롤링 및 전처리된 리뷰 데이터
│ ├── cleaned_reviews.csv # 2차 전처리 데이터
│ ├── cleaned_reviews_strong.csv # 3차 전처리 데이터
│ └── labeled_reviews_lmstudio_cleaned.csv # 1차 전처리 데이터
│
├── models/ # 학습된 감정 분석 모델
│ ├── koelectra_sentiment_model_tf_updated/ # 2차 데이터 커리큘럼 구성 후 모델
│ └── koelectra_sentiment_model_tf/ # 1차 모델
│
├── plots/ # 학습 시각화 이미지
│ ├── confusion_matrix.png
│ ├── accuracy_plot.png
│ ├── loss_plot.png
│ └── model_progress.png
│
├── train_sentiment_model.py # BiLSTM 모델 학습 코드
├── train_koelectra_sentiment.py # KoELECTRA 모델 학습 코드
├── review.py # 전처리 및 리뷰 처리 모듈
├── save.py # 모델 저장 유틸리티
├── requirements.txt # 실행 환경 패키지 목록
└── README.md
---

## 🧾 사용한 기술 스택

| 항목         | 도구 |
|--------------|------|
| 모델 훈련     | TensorFlow, HuggingFace Transformers (KoELECTRA) |
| NLP 분석     | KoNLPy (Okt), Tokenizer |
| 시각화       | matplotlib, seaborn |
| 전처리       | pandas, numpy |
| 환경         | Python 3.10, Git LFS |
| 실행 플랫폼  | Streamlit (별도 서비스 리포지토리 참고) |

---

## 🧪 데이터 분석 및 재정비 과정

- 400,000+개 가량의 원시 데이터 사용
- 라벨, 정제 후 약 10,890개의 리뷰 수로 단축 / Train Set : 80% (8712개) / Validation Set : 20% (2178개)
  `plots/data_chart.png` 
- 원본 리뷰 데이터에서 감정 표현이 약한 문장을 필터링
- 감정 강도 높은 리뷰만 별도 데이터셋(`cleaned_reviews_strong.csv`) 구성
- 수작업 라벨링된 결과를 기준으로 `train_test_split` 수행

---

## 🧠 감정 분류 모델

- **모델**: TFElectraForSequenceClassification (KoELECTRA 기반)
- **분류 대상**: 부정(0), 중립(1), 긍정(2)
- **학습 데이터**: 웹 크롤링으로 수집한 음식점 리뷰 (라벨링 포함)
- **학습 환경**: CPU 환경에서 작동 가능 (Anaconda 가상환경 사용)

---

## 📈 모델 개선 및 성능 변화 과정

본 프로젝트는 반복적인 실험을 통해 모델 성능을 개선하고 최종적으로 KoELECTRA 기반 모델을 선택하게 되었습니다.

### 1. 실험 이력 및 성능 비교

> `plots/model_progress.png`  

|             단계            |                     설명                     | Accuracy | F1-score 
|-----------------------------|----------------------------------------------|----------|-----------
| **1.    BiLSTM (초기)**     | 기본 구조/전처리 최소                          |   81.0%  | 78.0
| **2. BiLSTM (전처리 강화)** | 텍스트 정제, 중복 제거, 감정 강도 필터링        |   85.0%  | 84.0
| **3.    BiLSTM (튜닝)**     | Dropout, Layer 수 조정 등 하이퍼파라미터 개선  |   87.6%   | 86.5
| **4.     KoELECTRA**        | 사전학습 기반 모델로 문맥 이해 강화            | **89.2%** | **88.3**

> ✔ 최종 선택: KoELECTRA – 일반화 성능이 뛰어나고 실제 사용자 리뷰에 강건하게 작동함

### 2. 시각화 자료

- 📊 혼동 행렬: `plots/confusion_matrix.png`  
- 📈 정확도 변화: `plots/accuracy_plot.png`  
- 📉 손실 변화: `plots/loss_plot.png`  

---

## 🔁 Git LFS 및 모델 관리

- 대용량 모델(`.h5`)은 Git LFS로 업로드
- `.gitattributes` 설정 및 `git lfs track` 사용
- 모델 버전 관리를 위해 커밋 히스토리 클린업 진행

---

## 🧾 관련 프로젝트 및 시연

- 👉 **웹앱 시각화 서비스 리포지토리**: [web-crawling-service](https://github.com/Xox00xoX/web-crawling-service)
