import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
                             roc_auc_score, average_precision_score)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib, os, warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════
#  1. 데이터 로드 및 누출 피처 제거
# ══════════════════════════════════════════
df = pd.read_csv("data/csv/mlm_labeled_lite.csv")
print(f"원본 shape: {df.shape}")

# ── 타겟 누출 피처 식별 ──
# risk_level을 만들 때 직접 사용된 파생 피처들
leakage_cols = ['financial_health', 'sustainability', 'commission_flag']
# risk_score도 타겟의 원재료이므로 제거 (컬럼 존재 시)
if 'risk_score' in df.columns:
    leakage_cols.append('risk_score')

target_cols = ['risk_level', 'churned']
meta_cols = ['company_name', 'company_name_normalized', 'year']
exclude_cols = target_cols + meta_cols + leakage_cols

feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"\n✅ 누출 피처 제거: {leakage_cols}")
print(f"✅ 최종 피처 수: {len(feature_cols)}")
print(f"   {feature_cols}")

# ══════════════════════════════════════════
#  2. churned 타겟 재점검 및 분할 전략 수정
# ══════════════════════════════════════════
print(f"\n연도별 churned 분포:")
print(df.groupby('year')['churned'].value_counts().unstack(fill_value=0))

# 2024년 churned=1이 없으면 Test set으로 부적합
# → 시간 분할 조정: Train ≤2021, Valid 2022, Test 2023
# (2023년에는 churned=1이 존재할 가능성 높음)

churn_by_year = df.groupby('year')['churned'].sum()
print(f"\n연도별 churned=1 건수:\n{churn_by_year}")

# 동적으로 가장 적합한 분할 결정
if churn_by_year.get(2024, 0) == 0:
    print("\n⚠️ 2024년 churned=1 없음 → 분할 전략 변경")
    print("   Train: ≤2021 | Valid: 2022 | Test: 2023")
    train_df = df[df['year'] <= 2021].copy()
    valid_df = df[df['year'] == 2022].copy()
    test_df  = df[df['year'] == 2023].copy()
else:
    train_df = df[df['year'] <= 2022].copy()
    valid_df = df[df['year'] == 2023].copy()
    test_df  = df[df['year'] == 2024].copy()

print(f"\nTrain: {len(train_df)}건 | Valid: {len(valid_df)}건 | Test: {len(test_df)}건")
print(f"Train churned=1: {train_df['churned'].sum()} | Valid: {valid_df['churned'].sum()} | Test: {test_df['churned'].sum()}")
print(f"Train risk 분포: {train_df['risk_level'].value_counts().sort_index().to_dict()}")
print(f"Valid risk 분포: {valid_df['risk_level'].value_counts().sort_index().to_dict()}")
print(f"Test  risk 분포: {test_df['risk_level'].value_counts().sort_index().to_dict()}")

# ══════════════════════════════════════════
#  3. 피처 준비
# ══════════════════════════════════════════
X_train = train_df[feature_cols].fillna(0)
X_valid = valid_df[feature_cols].fillna(0)
X_test  = test_df[feature_cols].fillna(0)

# 무한대 값 처리 (debt_ratio에서 발생)
X_train = X_train.replace([np.inf, -np.inf], 0)
X_valid = X_valid.replace([np.inf, -np.inf], 0)
X_test  = X_test.replace([np.inf, -np.inf], 0)

y_train_risk  = train_df['risk_level'].to_numpy().astype(int)
y_valid_risk  = valid_df['risk_level'].to_numpy().astype(int)
y_test_risk   = test_df['risk_level'].to_numpy().astype(int)

y_train_churn = train_df['churned'].to_numpy().astype(int)
y_valid_churn = valid_df['churned'].to_numpy().astype(int)
y_test_churn  = test_df['churned'].to_numpy().astype(int)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_valid_s = scaler.transform(X_valid)
X_test_s  = scaler.transform(X_test)

print(f"\n최종 피처 shape: {X_train_s.shape}")

# ══════════════════════════════════════════
#  4. Model A: risk_level 3-class (누출 제거 버전)
# ══════════════════════════════════════════
models_risk = {
    'LogisticRegression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42, multi_class='multinomial'),
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        class_weight='balanced', random_state=42, verbose=-1)
}

print("\n" + "=" * 5)
print("  [Model A] risk_level 3-class (누출 피처 제거)")
print("=" * 5)

results_risk = []
for name, model in models_risk.items():
    model.fit(X_train_s, y_train_risk)

    # Validation
    y_pred_v = model.predict(X_valid_s)
    f1_v = float(f1_score(y_valid_risk, y_pred_v, average='macro'))
    acc_v = float(accuracy_score(y_valid_risk, y_pred_v))

    # Test
    y_pred_t = model.predict(X_test_s)
    f1_t = float(f1_score(y_test_risk, y_pred_t, average='macro'))
    acc_t = float(accuracy_score(y_test_risk, y_pred_t))

    results_risk.append({
        'Model': name,
        'Val_Acc': round(acc_v, 4), 'Val_F1_macro': round(f1_v, 4),
        'Test_Acc': round(acc_t, 4), 'Test_F1_macro': round(f1_t, 4)
    })
    print(f"\n▶ {name}")
    print(f"  Valid | Acc: {acc_v:.4f} | F1(macro): {f1_v:.4f}")
    print(f"  Test  | Acc: {acc_t:.4f} | F1(macro): {f1_t:.4f}")
    print(f"  [Valid Report]")
    print(classification_report(y_valid_risk, y_pred_v,
          target_names=['Low(0)', 'Mid(1)', 'High(2)']))

results_risk_df = pd.DataFrame(results_risk).sort_values('Val_F1_macro', ascending=False)
print("\n📊 risk_level 모델 비교:")
print(results_risk_df.to_string(index=False))

# ══════════════════════════════════════════
#  5. Model B: churned 이진 분류
# ══════════════════════════════════════════
scale_pos = float(np.sum(y_train_churn == 0)) / max(float(np.sum(y_train_churn == 1)), 1)

models_churn = {
    'LogisticRegression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        class_weight='balanced', random_state=42, verbose=-1)
}

print("\n" + "=" * 5)
print("  [Model B] churned 이진 분류")
print("=" * 5)

results_churn = []
for name, model in models_churn.items():
    model.fit(X_train_s, y_train_churn)

    # Validation
    y_pred_v = model.predict(X_valid_s)
    f1_v = float(f1_score(y_valid_churn, y_pred_v, zero_division=0))
    acc_v = float(accuracy_score(y_valid_churn, y_pred_v))
    if hasattr(model, 'predict_proba'):
        y_prob_v = model.predict_proba(X_valid_s)[:, 1]
        auc_v = float(roc_auc_score(y_valid_churn, y_prob_v)) if len(set(y_valid_churn)) > 1 else 0.0
    else:
        auc_v = 0.0

    # Test
    y_pred_t = model.predict(X_test_s)
    f1_t = float(f1_score(y_test_churn, y_pred_t, zero_division=0))
    acc_t = float(accuracy_score(y_test_churn, y_pred_t))
    if hasattr(model, 'predict_proba') and len(set(y_test_churn)) > 1:
        y_prob_t = model.predict_proba(X_test_s)[:, 1]
        auc_t = float(roc_auc_score(y_test_churn, y_prob_t))
    else:
        auc_t = 0.0

    results_churn.append({
        'Model': name,
        'Val_Acc': round(acc_v, 4), 'Val_F1': round(f1_v, 4), 'Val_AUC': round(auc_v, 4),
        'Test_Acc': round(acc_t, 4), 'Test_F1': round(f1_t, 4), 'Test_AUC': round(auc_t, 4)
    })
    print(f"\n▶ {name}")
    print(f"  Valid | Acc: {acc_v:.4f} | F1: {f1_v:.4f} | AUC: {auc_v:.4f}")
    print(f"  Test  | Acc: {acc_t:.4f} | F1: {f1_t:.4f} | AUC: {auc_t:.4f}")
    print(f"  [Valid Report]")
    print(classification_report(y_valid_churn, y_pred_v,
          target_names=['Active(0)', 'Churned(1)'], zero_division=0))

results_churn_df = pd.DataFrame(results_churn).sort_values('Val_F1', ascending=False)
print("\n📊 churned 모델 비교:")
print(results_churn_df.to_string(index=False))

# ══════════════════════════════════════════
#  6. 모델 저장
# ══════════════════════════════════════════
os.makedirs("models", exist_ok=True)
for name, model in models_risk.items():
    joblib.dump(model, f"models/risk_{name}.pkl")
for name, model in models_churn.items():
    joblib.dump(model, f"models/churn_{name}.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

# 결과 저장
results_risk_df.to_csv("models/results_risk_individual.csv", index=False)
results_churn_df.to_csv("models/results_churn_individual.csv", index=False)

print("\n✅ 수정된 모델 저장 완료")
print("   - 누출 피처 제거됨")
print("   - churned 분할 전략 수정됨")
print("   - 다음 단계: 앙상블")
