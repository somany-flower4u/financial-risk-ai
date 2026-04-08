"""
전체 데이터(2010-2025)로 모델 재학습 및 2025년 예측
- Train: 2010-2022 | Valid: 2023 | Test: 2024 | Predict: 2025
- 상위 3개 부스팅 모델 앙상블 (확률 평균)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, roc_auc_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print(" 📊 전체 데이터(2010-2025) 재학습 및 2025년 예측")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════
#  1. 데이터 준비
# ══════════════════════════════════════════════════════════════════

print("\n[1단계] 데이터 로드 및 분할")
print("-" * 80)

df = pd.read_csv("data/csv/mlm_labeled_lite.csv")
print(f"\n✅ 원본 데이터: {df.shape[0]}행, {df.shape[1]}컬럼")
print(f"   연도 범위: {df['year'].min()}-{df['year'].max()}")

# 누출 피처 제거
leakage_cols = ['financial_health', 'sustainability', 'commission_flag']
if 'risk_score' in df.columns:
    leakage_cols.append('risk_score')

target_cols = ['risk_level', 'churned']
meta_cols = ['company_name', 'company_name_normalized', 'year']
exclude_cols = target_cols + meta_cols + leakage_cols

feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"\n✅ 누출 피처 제거: {len(feature_cols)}개 피처 사용")

# 시간 기반 분할: 2010-2022 / 2023 / 2024 / 2025
train_df = df[df['year'] <= 2022].copy()
valid_df = df[df['year'] == 2023].copy()
test_df  = df[df['year'] == 2024].copy()
pred_df  = df[df['year'] == 2025].copy()

print(f"\n✅ 데이터 분할:")
print(f"   Train (2010-2022): {len(train_df):4d}행")
print(f"   Valid (2023):      {len(valid_df):4d}행")
print(f"   Test  (2024):      {len(test_df):4d}행")
print(f"   Pred  (2025):      {len(pred_df):4d}행")

# 피처 준비
X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
X_valid = valid_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
X_test  = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
X_pred  = pred_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

# 타겟
y_train_risk  = train_df['risk_level'].to_numpy().astype(int)
y_valid_risk  = valid_df['risk_level'].to_numpy().astype(int)
y_test_risk   = test_df['risk_level'].to_numpy().astype(int)

y_train_churn = train_df['churned'].to_numpy().astype(int)
y_valid_churn = valid_df['churned'].to_numpy().astype(int)
y_test_churn  = test_df['churned'].to_numpy().astype(int)

# 스케일링
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_valid_s = scaler.transform(X_valid)
X_test_s  = scaler.transform(X_test)
X_pred_s  = scaler.transform(X_pred)

print(f"\n✅ 스케일링 완료")

# ══════════════════════════════════════════════════════════════════
#  2. 상위 3개 부스팅 모델 재학습 (risk_level)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("[2단계] Model A: risk_level 모델 재학습")
print("=" * 80)

gb_risk = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42
)
xgb_risk = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, use_label_encoder=False, eval_metric='mlogloss'
)
lgbm_risk = LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    class_weight='balanced', random_state=42, verbose=-1
)

# 학습
gb_risk.fit(X_train_s, y_train_risk)
xgb_risk.fit(X_train_s, y_train_risk)
lgbm_risk.fit(X_train_s, y_train_risk)

print("\n✅ 3개 모델 학습 완료")

# Validation 평가
prob_gb_v = gb_risk.predict_proba(X_valid_s)
prob_xgb_v = xgb_risk.predict_proba(X_valid_s)
prob_lgbm_v = lgbm_risk.predict_proba(X_valid_s)
prob_ensemble_v = (prob_gb_v + prob_xgb_v + prob_lgbm_v) / 3
pred_ensemble_v = np.argmax(prob_ensemble_v, axis=1)

f1_v = float(f1_score(y_valid_risk, pred_ensemble_v, average='macro'))
acc_v = float(accuracy_score(y_valid_risk, pred_ensemble_v))

print(f"\n▶ Validation Set (2023)")
print(f"  Ensemble | Acc: {acc_v:.4f} | F1(macro): {f1_v:.4f}")

# Test 평가
prob_gb_t = gb_risk.predict_proba(X_test_s)
prob_xgb_t = xgb_risk.predict_proba(X_test_s)
prob_lgbm_t = lgbm_risk.predict_proba(X_test_s)
prob_ensemble_t = (prob_gb_t + prob_xgb_t + prob_lgbm_t) / 3
pred_ensemble_t = np.argmax(prob_ensemble_t, axis=1)

f1_t = float(f1_score(y_test_risk, pred_ensemble_t, average='macro'))
acc_t = float(accuracy_score(y_test_risk, pred_ensemble_t))

print(f"\n▶ Test Set (2024)")
print(f"  Ensemble | Acc: {acc_t:.4f} | F1(macro): {f1_t:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test_risk, pred_ensemble_t,
    target_names=['Low(0)', 'Mid(1)', 'High(2)']))

# ══════════════════════════════════════════════════════════════════
#  3. 상위 3개 부스팅 모델 재학습 (churned)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("[3단계] Model B: churned 모델 재학습")
print("=" * 80)

scale_pos = float(np.sum(y_train_churn == 0)) / max(float(np.sum(y_train_churn == 1)), 1)

gb_churn = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42
)
xgb_churn = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=scale_pos, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, use_label_encoder=False, eval_metric='logloss'
)
lgbm_churn = LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    class_weight='balanced', random_state=42, verbose=-1
)

# 학습
gb_churn.fit(X_train_s, y_train_churn)
xgb_churn.fit(X_train_s, y_train_churn)
lgbm_churn.fit(X_train_s, y_train_churn)

print("\n✅ 3개 모델 학습 완료")

# Validation 평가
prob_gb_churn_v = np.asarray(gb_churn.predict_proba(X_valid_s))[:, 1]
prob_xgb_churn_v = np.asarray(xgb_churn.predict_proba(X_valid_s))[:, 1]
prob_lgbm_churn_v = np.asarray(lgbm_churn.predict_proba(X_valid_s))[:, 1]
prob_ensemble_churn_v = (prob_gb_churn_v + prob_xgb_churn_v + prob_lgbm_churn_v) / 3
pred_ensemble_churn_v = (prob_ensemble_churn_v > 0.5).astype(int)

f1_churn_v = float(f1_score(y_valid_churn, pred_ensemble_churn_v, zero_division=0))
acc_churn_v = float(accuracy_score(y_valid_churn, pred_ensemble_churn_v))
auc_churn_v = float(roc_auc_score(y_valid_churn, prob_ensemble_churn_v)) if len(set(y_valid_churn)) > 1 else 0.0

print(f"\n▶ Validation Set (2023)")
print(f"  Ensemble | Acc: {acc_churn_v:.4f} | F1: {f1_churn_v:.4f} | AUC: {auc_churn_v:.4f}")

# Test 평가
prob_gb_churn_t = np.asarray(gb_churn.predict_proba(X_test_s))[:, 1]
prob_xgb_churn_t = np.asarray(xgb_churn.predict_proba(X_test_s))[:, 1]
prob_lgbm_churn_t = np.asarray(lgbm_churn.predict_proba(X_test_s))[:, 1]
prob_ensemble_churn_t = (prob_gb_churn_t + prob_xgb_churn_t + prob_lgbm_churn_t) / 3
pred_ensemble_churn_t = (prob_ensemble_churn_t > 0.5).astype(int)

f1_churn_t = float(f1_score(y_test_churn, pred_ensemble_churn_t, zero_division=0))
acc_churn_t = float(accuracy_score(y_test_churn, pred_ensemble_churn_t))
auc_churn_t = float(roc_auc_score(y_test_churn, prob_ensemble_churn_t)) if len(set(y_test_churn)) > 1 else 0.0

print(f"\n▶ Test Set (2024)")
print(f"  Ensemble | Acc: {acc_churn_t:.4f} | F1: {f1_churn_t:.4f} | AUC: {auc_churn_t:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test_churn, pred_ensemble_churn_t,
    target_names=['Active(0)', 'Churned(1)'], zero_division=0))

# ══════════════════════════════════════════════════════════════════
#  4. 2025년 데이터 예측
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("[4단계] 2025년 예측")
print("=" * 80)

# risk_level 예측
prob_gb_pred = gb_risk.predict_proba(X_pred_s)
prob_xgb_pred = xgb_risk.predict_proba(X_pred_s)
prob_lgbm_pred = lgbm_risk.predict_proba(X_pred_s)
prob_ensemble_pred = (prob_gb_pred + prob_xgb_pred + prob_lgbm_pred) / 3
pred_risk_2025 = np.argmax(prob_ensemble_pred, axis=1)

# churned 예측
prob_gb_churn_pred = np.asarray(gb_churn.predict_proba(X_pred_s))[:, 1]
prob_xgb_churn_pred = np.asarray(xgb_churn.predict_proba(X_pred_s))[:, 1]
prob_lgbm_churn_pred = np.asarray(lgbm_churn.predict_proba(X_pred_s))[:, 1]
prob_ensemble_churn_pred = (prob_gb_churn_pred + prob_xgb_churn_pred + prob_lgbm_churn_pred) / 3
pred_churn_2025 = (prob_ensemble_churn_pred > 0.5).astype(int)

print(f"\n✅ 2025년 {len(pred_df)}개 기업의 예측 완료")
print(f"\n▶ risk_level 예측 분포:")
unique, counts = np.unique(pred_risk_2025, return_counts=True)
for u, c in zip(unique, counts):
    print(f"   Level {u}: {c:3d}개 ({100*c/len(pred_risk_2025):5.1f}%)")

print(f"\n▶ churned 예측 분포:")
unique_churn, counts_churn = np.unique(pred_churn_2025, return_counts=True)
for u, c in zip(unique_churn, counts_churn):
    label = "Churned" if u == 1 else "Active"
    print(f"   {label:8s}: {c:3d}개 ({100*c/len(pred_churn_2025):5.1f}%)")

# ══════════════════════════════════════════════════════════════════
#  5. 예측 결과와 실제값 비교
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("[5단계] 예측 vs 실제값 비교 (2025)")
print("=" * 80)

y_actual_risk = pred_df['risk_level'].to_numpy().astype(int)
y_actual_churn = pred_df['churned'].to_numpy().astype(int)

# risk_level 비교
acc_risk_pred = float(accuracy_score(y_actual_risk, pred_risk_2025))
f1_risk_pred = float(f1_score(y_actual_risk, pred_risk_2025, average='macro'))

print(f"\n▶ risk_level 성능 (2025 실제데이터)")
print(f"  Accuracy: {acc_risk_pred:.4f} | F1(macro): {f1_risk_pred:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_actual_risk, pred_risk_2025,
    target_names=['Low(0)', 'Mid(1)', 'High(2)']))

# churned 비교
acc_churn_pred = float(accuracy_score(y_actual_churn, pred_churn_2025))
f1_churn_pred = float(f1_score(y_actual_churn, pred_churn_2025, zero_division=0))
try:
    auc_churn_pred = float(roc_auc_score(y_actual_churn, prob_ensemble_churn_pred))
except:
    auc_churn_pred = 0.0

print(f"\n▶ churned 성능 (2025 실제데이터)")
print(f"  Accuracy: {acc_churn_pred:.4f} | F1: {f1_churn_pred:.4f} | AUC: {auc_churn_pred:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_actual_churn, pred_churn_2025,
    target_names=['Active(0)', 'Churned(1)'], zero_division=0))

# ══════════════════════════════════════════════════════════════════
#  6. 결과 저장 및 요약
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("[6단계] 결과 저장")
print("=" * 80)

# 예측 결과 CSV로 저장
result_df = pd.DataFrame({
    'company_name': pred_df['company_name'].values,
    'year': pred_df['year'].values,
    'actual_risk_level': y_actual_risk,
    'predicted_risk_level': pred_risk_2025,
    'risk_prediction_prob_0': prob_ensemble_pred[:, 0],
    'risk_prediction_prob_1': prob_ensemble_pred[:, 1],
    'risk_prediction_prob_2': prob_ensemble_pred[:, 2],
    'actual_churned': y_actual_churn,
    'predicted_churned': pred_churn_2025,
    'churned_probability': prob_ensemble_churn_pred
})

result_df.to_csv('models/prediction_2025_comparison.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ 예측 결과 저장: models/prediction_2025_comparison.csv")

# 성능 요약 저장
summary = {
    'data_split': {
        'train_years': '2010-2022',
        'valid_years': '2023',
        'test_years': '2024',
        'predict_years': '2025',
        'train_count': len(train_df),
        'valid_count': len(valid_df),
        'test_count': len(test_df),
        'predict_count': len(pred_df)
    },
    'risk_level': {
        'test_2024': {
            'accuracy': round(acc_t, 4),
            'f1_macro': round(f1_t, 4)
        },
        'predict_2025': {
            'accuracy': round(acc_risk_pred, 4),
            'f1_macro': round(f1_risk_pred, 4)
        }
    },
    'churned': {
        'test_2024': {
            'accuracy': round(acc_churn_t, 4),
            'f1': round(f1_churn_t, 4),
            'auc': round(auc_churn_t, 4)
        },
        'predict_2025': {
            'accuracy': round(acc_churn_pred, 4),
            'f1': round(f1_churn_pred, 4),
            'auc': round(auc_churn_pred, 4)
        }
    }
}

joblib.dump(summary, 'models/retrain_summary.pkl')
print(f"✅ 성능 요약 저장: models/retrain_summary.pkl")

# 모델 저장
joblib.dump(gb_risk, 'models/retrain_risk_GradientBoosting.pkl')
joblib.dump(xgb_risk, 'models/retrain_risk_XGBoost.pkl')
joblib.dump(lgbm_risk, 'models/retrain_risk_LightGBM.pkl')
joblib.dump(gb_churn, 'models/retrain_churn_GradientBoosting.pkl')
joblib.dump(xgb_churn, 'models/retrain_churn_XGBoost.pkl')
joblib.dump(lgbm_churn, 'models/retrain_churn_LightGBM.pkl')
joblib.dump(scaler, 'models/retrain_scaler.pkl')
joblib.dump(feature_cols, 'models/feature_cols.pkl')
print(f"✅ 재학습된 모델들 저장 완료")

# 최종 요약
print("\n" + "=" * 80)
print(" 📋 재학습 및 예측 최종 요약")
print("=" * 80)
print(f"""
┌──────────────────────────────────────────────────────────────┐
│ 데이터 분할                                                 │
│  Train: 2010-2022 ({len(train_df)} 행)                            │
│  Valid: 2023 ({len(valid_df)} 행)                               │
│  Test:  2024 ({len(test_df)} 행)  → 2024 성능 평가               │
│  Pred:  2025 ({len(pred_df)} 행)  → 2025 예측 + 비교             │
├──────────────────────────────────────────────────────────────┤
│ [Model A] risk_level (3-class)                              │
│  Test (2024)  │ Acc: {acc_t:.4f} | F1: {f1_t:.4f}                  │
│  Pred (2025)  │ Acc: {acc_risk_pred:.4f} | F1: {f1_risk_pred:.4f}                  │
├──────────────────────────────────────────────────────────────┤
│ [Model B] churned (binary)                                  │
│  Test (2024)  │ Acc: {acc_churn_t:.4f} | F1: {f1_churn_t:.4f} | AUC: {auc_churn_t:.4f} │
│  Pred (2025)  │ Acc: {acc_churn_pred:.4f} | F1: {f1_churn_pred:.4f} | AUC: {auc_churn_pred:.4f} │
├──────────────────────────────────────────────────────────────┤
│ 저장 파일                                                    │
│  ✓ models/prediction_2025_comparison.csv (예측 결과)       │
│  ✓ models/retrain_summary.pkl (성능 요약)                  │
│  ✓ models/retrain_*.pkl (재학습 모델들)                    │
└──────────────────────────────────────────────────────────────┘
""")

print("\n✅ 모든 작업 완료!")
