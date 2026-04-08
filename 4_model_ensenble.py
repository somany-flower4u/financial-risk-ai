"""
상위 3개 부스팅 모델 앙상블 (확률 평균)
- Model A (risk_level): GradientBoosting, XGBoost, LightGBM
- Model B (churned): GradientBoosting, XGBoost, LightGBM
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, roc_auc_score, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 70)
print(" 상위 3개 부스팅 모델 앙상블 (확률 평균)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
#  1. 데이터 및 모델 준비
# ══════════════════════════════════════════════════════════════════

# 데이터 로드
df = pd.read_csv("data/csv/mlm_labeled_lite.csv")

# 누출 피처 제거
leakage_cols = ['financial_health', 'sustainability', 'commission_flag']
if 'risk_score' in df.columns:
    leakage_cols.append('risk_score')

target_cols = ['risk_level', 'churned']
meta_cols = ['company_name', 'company_name_normalized', 'year']
exclude_cols = target_cols + meta_cols + leakage_cols

feature_cols = [c for c in df.columns if c not in exclude_cols]

# 동적 분할
churn_by_year = df.groupby('year')['churned'].sum()
if churn_by_year.get(2024, 0) == 0:
    train_df = df[df['year'] <= 2021].copy()
    valid_df = df[df['year'] == 2022].copy()
    test_df  = df[df['year'] == 2023].copy()
else:
    train_df = df[df['year'] <= 2022].copy()
    valid_df = df[df['year'] == 2023].copy()
    test_df  = df[df['year'] == 2024].copy()

# 피처 준비
X_train = train_df[feature_cols].fillna(0)
X_valid = valid_df[feature_cols].fillna(0)
X_test  = test_df[feature_cols].fillna(0)

# 무한대/NaN 처리
X_train = X_train.replace([np.inf, -np.inf], 0)
X_valid = X_valid.replace([np.inf, -np.inf], 0)
X_test  = X_test.replace([np.inf, -np.inf], 0)

# 타겟
y_train_risk  = train_df['risk_level'].to_numpy().astype(int)
y_valid_risk  = valid_df['risk_level'].to_numpy().astype(int)
y_test_risk   = test_df['risk_level'].to_numpy().astype(int)

y_train_churn = train_df['churned'].to_numpy().astype(int)
y_valid_churn = valid_df['churned'].to_numpy().astype(int)
y_test_churn  = test_df['churned'].to_numpy().astype(int)

# 스케일링
scaler = joblib.load('models/scaler.pkl')
X_train_s = scaler.fit_transform(X_train)
X_valid_s = scaler.transform(X_valid)
X_test_s  = scaler.transform(X_test)

print(f"\n✅ 데이터 준비:")
print(f"   Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")
print(f"   Feature: {len(feature_cols)}")

# 모델 로드
gb_risk   = joblib.load('models/risk_GradientBoosting.pkl')
xgb_risk  = joblib.load('models/risk_XGBoost.pkl')
lgbm_risk = joblib.load('models/risk_LightGBM.pkl')

gb_churn   = joblib.load('models/churn_GradientBoosting.pkl')
xgb_churn  = joblib.load('models/churn_XGBoost.pkl')
lgbm_churn = joblib.load('models/churn_LightGBM.pkl')

print(f"\n✅ 상위 3개 부스팅 모델 로드:")
print(f"   - GradientBoosting (risk_level & churned)")
print(f"   - XGBoost (risk_level & churned)")
print(f"   - LightGBM (risk_level & churned)")

# ══════════════════════════════════════════════════════════════════
#  2. Model A: risk_level 앙상블 (확률 평균)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" [Ensemble A] risk_level 앙상블 (상위 3개 부스팅 모델 확률 평균)")
print("=" * 70)

# Validation
y_prob_gb_v = gb_risk.predict_proba(X_valid_s)
y_prob_xgb_v = xgb_risk.predict_proba(X_valid_s)
y_prob_lgbm_v = lgbm_risk.predict_proba(X_valid_s)

# 확률 평균
y_prob_ensemble_v = (y_prob_gb_v + y_prob_xgb_v + y_prob_lgbm_v) / 3
y_pred_ensemble_v = np.argmax(y_prob_ensemble_v, axis=1)

f1_v = float(f1_score(y_valid_risk, y_pred_ensemble_v, average='macro'))
acc_v = float(accuracy_score(y_valid_risk, y_pred_ensemble_v))

# Test
y_prob_gb_t = gb_risk.predict_proba(X_test_s)
y_prob_xgb_t = xgb_risk.predict_proba(X_test_s)
y_prob_lgbm_t = lgbm_risk.predict_proba(X_test_s)

y_prob_ensemble_t = (y_prob_gb_t + y_prob_xgb_t + y_prob_lgbm_t) / 3
y_pred_ensemble_t = np.argmax(y_prob_ensemble_t, axis=1)

f1_t = float(f1_score(y_test_risk, y_pred_ensemble_t, average='macro'))
acc_t = float(accuracy_score(y_test_risk, y_pred_ensemble_t))

print(f"\n▶ Ensemble (GB + XGB + LGBM 확률 평균)")
print(f"  Valid | Acc: {acc_v:.4f} | F1(macro): {f1_v:.4f}")
print(f"  Test  | Acc: {acc_t:.4f} | F1(macro): {f1_t:.4f}")
print(f"\n  [Valid Report]")
print(classification_report(y_valid_risk, y_pred_ensemble_v,
      target_names=['Low(0)', 'Mid(1)', 'High(2)']))

# ══════════════════════════════════════════════════════════════════
#  3. Model B: churned 앙상블 (확률 평균)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" [Ensemble B] churned 앙상블 (상위 3개 부스팅 모델 확률 평균)")
print("=" * 70)

# Validation
y_prob_gb_churn_v = gb_churn.predict_proba(X_valid_s)[:, 1]
y_prob_xgb_churn_v = xgb_churn.predict_proba(X_valid_s)[:, 1]
y_prob_lgbm_churn_v = lgbm_churn.predict_proba(X_valid_s)[:, 1]

# 확률 평균
y_prob_churn_ensemble_v = (y_prob_gb_churn_v + y_prob_xgb_churn_v + y_prob_lgbm_churn_v) / 3
y_pred_churn_ensemble_v = (y_prob_churn_ensemble_v > 0.5).astype(int)

f1_churn_v = float(f1_score(y_valid_churn, y_pred_churn_ensemble_v, zero_division=0))
acc_churn_v = float(accuracy_score(y_valid_churn, y_pred_churn_ensemble_v))
if len(set(y_valid_churn)) > 1:
    auc_churn_v = float(roc_auc_score(y_valid_churn, y_prob_churn_ensemble_v))
else:
    auc_churn_v = 0.0

# Test
y_prob_gb_churn_t = gb_churn.predict_proba(X_test_s)[:, 1]
y_prob_xgb_churn_t = xgb_churn.predict_proba(X_test_s)[:, 1]
y_prob_lgbm_churn_t = lgbm_churn.predict_proba(X_test_s)[:, 1]

y_prob_churn_ensemble_t = (y_prob_gb_churn_t + y_prob_xgb_churn_t + y_prob_lgbm_churn_t) / 3
y_pred_churn_ensemble_t = (y_prob_churn_ensemble_t > 0.5).astype(int)

f1_churn_t = float(f1_score(y_test_churn, y_pred_churn_ensemble_t, zero_division=0))
acc_churn_t = float(accuracy_score(y_test_churn, y_pred_churn_ensemble_t))
if len(set(y_test_churn)) > 1:
    auc_churn_t = float(roc_auc_score(y_test_churn, y_prob_churn_ensemble_t))
else:
    auc_churn_t = 0.0

print(f"\n▶ Ensemble (GB + XGB + LGBM 확률 평균)")
print(f"  Valid | Acc: {acc_churn_v:.4f} | F1: {f1_churn_v:.4f} | AUC: {auc_churn_v:.4f}")
print(f"  Test  | Acc: {acc_churn_t:.4f} | F1: {f1_churn_t:.4f} | AUC: {auc_churn_t:.4f}")
print(f"\n  [Valid Report]")
print(classification_report(y_valid_churn, y_pred_churn_ensemble_v,
      target_names=['Active(0)', 'Churned(1)'], zero_division=0))

# ══════════════════════════════════════════════════════════════════
#  4. 결과 비교 및 저장
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" 📊 최종 성능 비교 (개별 모델 vs 앙상블)")
print("=" * 70)

# Model A
print("\n[Model A] risk_level")
results_a = [
    ("GradientBoosting", gb_risk.predict(X_test_s)),
    ("XGBoost", xgb_risk.predict(X_test_s)),
    ("LightGBM", lgbm_risk.predict(X_test_s)),
    ("Ensemble", y_pred_ensemble_t)
]

for name, y_pred in results_a:
    f1 = float(f1_score(y_test_risk, y_pred, average='macro'))
    acc = float(accuracy_score(y_test_risk, y_pred))
    marker = "⭐" if name == "Ensemble" else "  "
    print(f"  {marker} {name:20s} | Acc: {acc:.4f} | F1(macro): {f1:.4f}")

# Model B
print("\n[Model B] churned")
results_b = [
    ("GradientBoosting", gb_churn.predict(X_test_s)),
    ("XGBoost", xgb_churn.predict(X_test_s)),
    ("LightGBM", lgbm_churn.predict(X_test_s)),
    ("Ensemble", y_pred_churn_ensemble_t)
]

for name, y_pred in results_b:
    f1 = float(f1_score(y_test_churn, y_pred, zero_division=0))
    acc = float(accuracy_score(y_test_churn, y_pred))
    marker = "⭐" if name == "Ensemble" else "  "
    print(f"  {marker} {name:20s} | Acc: {acc:.4f} | F1: {f1:.4f}")

# ══════════════════════════════════════════════════════════════════
#  5. 앙상블 모델 저장
# ══════════════════════════════════════════════════════════════════

import os
os.makedirs("models", exist_ok=True)

# 앙상블 결과 저장 (메타데이터)
ensemble_info = {
    'type': 'soft_voting_ensemble',
    'models': ['GradientBoosting', 'XGBoost', 'LightGBM'],
    'weights': [1/3, 1/3, 1/3],
    'task_a': {
        'name': 'risk_level',
        'val_acc': round(acc_v, 4),
        'val_f1_macro': round(f1_v, 4),
        'test_acc': round(acc_t, 4),
        'test_f1_macro': round(f1_t, 4)
    },
    'task_b': {
        'name': 'churned',
        'val_acc': round(acc_churn_v, 4),
        'val_f1': round(f1_churn_v, 4),
        'val_auc': round(auc_churn_v, 4),
        'test_acc': round(acc_churn_t, 4),
        'test_f1': round(f1_churn_t, 4),
        'test_auc': round(auc_churn_t, 4)
    }
}

joblib.dump(ensemble_info, 'models/ensemble_info.pkl')
joblib.dump(y_prob_ensemble_t, 'models/ensemble_risk_prob.pkl')
joblib.dump(y_prob_churn_ensemble_t, 'models/ensemble_churn_prob.pkl')

print("\n✅ 앙상블 모델 정보 저장 완료:")
print(f"   - ensemble_info.pkl")
print(f"   - ensemble_risk_prob.pkl")
print(f"   - ensemble_churn_prob.pkl")

# 최종 결과 요약
print("\n" + "=" * 70)
print(" 📋 앙상블 최종 요약")
print("=" * 70)
print(f"""
┌──────────────────────────────────────────────────────────┐
│ [Model A] risk_level 앙상블                             │
│   Val | Acc: {acc_v:.4f} | F1(macro): {f1_v:.4f}                        │
│   Test| Acc: {acc_t:.4f} | F1(macro): {f1_t:.4f}                        │
├──────────────────────────────────────────────────────────┤
│ [Model B] churned 앙상블                                │
│   Val | Acc: {acc_churn_v:.4f} | F1: {f1_churn_v:.4f} | AUC: {auc_churn_v:.4f}    │
│   Test| Acc: {acc_churn_t:.4f} | F1: {f1_churn_t:.4f} | AUC: {auc_churn_t:.4f}    │
├──────────────────────────────────────────────────────────┤
│ 앙상블 구성: GradientBoosting + XGBoost + LightGBM      │
│ 방식: Soft Voting (확률 평균, 균등 가중치)              │
└──────────────────────────────────────────────────────────┘
""")


# ✅ COMPLETED (2026.04.08)
# 과거 데이터(2010-2025) 전부 사용해서 모델 재학습 후 2025년 데이터로 예측 및 평가
# → 5_retrain_and_predict.py 참고
# 
# 결과:
#   [Model A] risk_level: 2024년 Acc 1.0 / 2025년 Acc 0.99 (104/105 정확)
#   [Model B] churned:    2024년 Acc 0.75 / 2025년 Acc 0.86 (90/105 정확)
#   출력: models/prediction_2025_comparison.csv (예측 결과 상세)