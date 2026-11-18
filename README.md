# 다단계판매업자 금융위험도 예측 모델

공정거래위원회 **「공공데이터 기반 다단계판매사업자 금융위험도 예측 모델」** 과제 참가 코드입니다. 공정위가 공개한 다단계판매업자 정보(2010~2025년, 연도별 xls)를 하나로 합치고, 업체의 종합 위험도(`risk_level`)와 다음 연도 퇴출 여부(`churned`)를 예측합니다.

위험도는 후원수당 비율, 재무 건전성, 사업 지속가능성 세 신호를 규칙으로 합쳐 만든 라벨입니다. 퇴출 여부는 업체가 이후 연도 자료에서 사라졌는지로 정했습니다. 분류 모델은 GradientBoosting, XGBoost, LightGBM이고, 세 모델의 예측 확률을 평균합니다.

> English summary: code for a Korean Fair Trade Commission public-data task. It merges 16 yearly disclosure files of multi-level marketing companies (2010-2025), builds a rule-based 3-class risk label and a churn label, and predicts both with a soft-voting ensemble of GradientBoosting, XGBoost and LightGBM. Models are trained on 2010-2022, validated on 2023, tested on 2024 and used to score 2025.

## 방법

1. **통합** (`1_data_filtering.py`): `data/xls/`의 연도별 xls를 열 위치 기준으로 표준 컬럼명에 맞춰 읽고, 숫자와 회사명을 정리해 `data/csv/mlm_unified_dataset.csv`로 저장합니다.
2. **라벨링** (`2_data_labeling.py`): 아래 규칙으로 라벨을 만들고 `data/csv/mlm_labeled_dataset.csv`로 저장합니다.
3. **학습** (`3_model_training.py`): 5개 모델(LogisticRegression, RandomForest, GradientBoosting, XGBoost, LightGBM)을 비교합니다.
4. **앙상블** (`4_model_ensenble.py`): 상위 3개 부스팅 모델의 확률을 평균합니다.
5. **재학습과 2025 예측** (`5_retrain_and_predict.py`): 2010~2022년으로 학습하고 2023년으로 검증, 2024년으로 테스트한 뒤 2025년을 예측합니다.
6. **위험 기업 선별** (`6_risk_analysis.py`): 2025년 예측 결과에서 고위험 기업과 폐업 위험 기업을 뽑습니다.

`validate_accuracy.py`는 혼동 행렬과 예측 신뢰도, 학습-테스트 정확도 차이를 확인하고, `preview_risk.py`는 위험 기업 CSV를 미리 봅니다.

### 라벨

| 라벨 | 규칙 |
|---|---|
| `commission_flag` | 후원수당 비율 35% 이하 0, 50% 이하 1, 50% 초과 2 |
| `financial_health` | 자본잠식 또는 부채비율 500% 초과 2, 부채비율 200% 초과 또는 당기순손실 1, 그 외 0 |
| `sustainability` | 전년 대비 매출과 판매원 수가 모두 30% 이상 감소 2, 하나만 감소 1, 그 외 0 (첫 연도는 0) |
| `risk_level` | 심각 신호(값 2)가 1개 이상이거나 경미 신호(값 1)가 3개면 2, 경미 신호가 1~2개면 1, 없으면 0 |
| `churned` | 업체의 마지막 관측 연도가 데이터 마지막 해(2025)보다 앞서면 그 마지막 행에 1 |

전체 1,722행(540개 업체)에서 `risk_level`은 0이 621행, 1이 679행, 2가 422행이고, `churned=1`은 435행입니다.

### 모델 입력

`risk_level`을 만드는 데 쓴 `commission_flag`, `financial_health`, `sustainability`는 특징에서 뺐습니다. 남은 특징은 매출액, 후원수당, 판매원 수, 자산, 부채, 자본, 부채비율, 매출 증감률 등 18개입니다. 결측은 0으로 채우고 무한대 값은 0으로 바꾼 뒤 StandardScaler로 표준화합니다.

- `risk_level` 모델: n_estimators 200~300, max_depth 5~6, learning_rate 0.05, subsample 0.8
- `churned` 모델: 같은 설정이고, XGBoost와 LightGBM에는 클래스 불균형 보정(`scale_pos_weight`, `class_weight`)을 넣었습니다.
- 위험 점수: `예측 위험도 × 0.4 + 폐업 확률(%) × 0.6`
- 폐업 판정 기준: 앙상블 확률 0.5 초과

## 결과

아래 값은 저장소의 스크립트와 `models/*.csv`에 기록된 값입니다.

| 대상 | 평가 | Accuracy | F1 |
|---|---|---:|---:|
| `risk_level` (3-class) | 2024년 테스트 | 1.0000 | 1.0000 |
| `risk_level` | 2025년 예측 | 0.9905 | 0.9877 |
| `churned` (binary) | 2024년 테스트 | 0.7500 | 0.3636 |
| `churned` | 2025년 예측 | 0.8571 | 0.0000 |

2025년 예측에서 `churned`의 AUC는 계산되지 않았습니다. 2025년이 데이터의 마지막 해라 `churned` 라벨이 모두 0이기 때문입니다.

2025년 105개 업체의 예측 결과는 고위험 19개, 중간위험 45개, 저위험 41개이고, 폐업 위험 업체는 15개입니다.

**해석할 때 주의할 점**
- `risk_level`은 규칙으로 만든 라벨입니다. 그 규칙에 쓰인 부채비율, 자본잠식 여부, 순손실 여부, 매출·판매원 증감률은 특징에 남아 있습니다. 그래서 정확도가 높은 것은 규칙을 얼마나 복원하는지를 보여 주는 값이고, 별도 기준에 대한 일반화 성능과는 다릅니다.
- `churned` 모델은 학습 정확도와 테스트 정확도 차이가 큽니다. `validate_accuracy.py`로 확인할 수 있습니다.

## 폴더 구조

```
1_data_filtering.py ... 6_risk_analysis.py   파이프라인 단계별 스크립트
validate_accuracy.py                         정확도 검증
preview_risk.py                              위험 기업 CSV 미리보기
data/
  xls/                 연도별 원본 (2010_MLMinfo.xls ~ 2025_MLMinfo.xls)
  csv/                 통합, 라벨링 결과 CSV 4개 (lite 버전 포함)
models/
  *.pkl                학습된 모델, 스케일러, 특징 목록
  results_*_individual.csv     모델별 검증, 테스트 성능
  prediction_2025_comparison.csv
  risk_companies_*_2025.csv    2025년 위험 기업 목록
```

## 설치와 실행

`requirements.txt`는 없습니다. 스크립트가 쓰는 패키지는 다음과 같습니다.

```bash
pip install pandas numpy scikit-learn xgboost lightgbm joblib xlrd
```

저장소 루트에서 순서대로 실행합니다. 경로는 모두 루트 기준 상대경로입니다.

```bash
python 1_data_filtering.py      # xls 통합
python 2_data_labeling.py       # 라벨 생성
python 3_model_training.py      # 모델 비교
python 4_model_ensenble.py      # 앙상블
python 5_retrain_and_predict.py # 재학습과 2025 예측
python 6_risk_analysis.py       # 위험 기업 선별
python validate_accuracy.py     # 선택
```

`models/`에 학습된 파일이 들어 있어 5번부터 다시 실행해도 됩니다. `6_risk_analysis.py`는 미래 연도 분석 함수(`analyze_future_year`)를 `models/analyze_future_year_func.pkl`로 저장합니다. 새 연도 데이터에 적용하려면 이 함수에 데이터프레임, 스케일러, 모델 딕셔너리, 특징 목록을 넘깁니다.

`.pkl` 파일은 joblib으로 저장했습니다. 출처를 알 수 없는 pkl은 열지 마세요. 이 저장소의 파일도 만든 때와 같은 버전의 scikit-learn, XGBoost, LightGBM에서 열어야 합니다.

## 데이터

- **출처:** 공정거래위원회가 공개한 다단계판매업자 정보 공시 자료(공공데이터)입니다. 연도별 파일이 `data/xls/`에 있습니다.
- **포함한 것:** 원본 xls 16개, 통합과 라벨링 CSV, 학습된 모델 파일, 2025년 예측 결과입니다.
- 원본과 그것에서 만든 CSV, 예측 결과에는 업체명이 들어 있습니다. 예측 결과는 모델이 낸 값이며 업체에 대한 사실 판단이 아닙니다.
- 데이터의 이용 조건은 원 제공처(공정거래위원회)의 조건을 따릅니다.

## 라이선스

코드는 [MIT License](LICENSE)를 따릅니다. 저장소에 포함된 공공데이터는 원 제공처의 이용 조건을 따릅니다.
