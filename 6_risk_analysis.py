"""
위험 기업 식별 및 분석 (2025년 현재 + 미래 2026년 준비)
- 2025년 현재 위험도가 높은 기업 식별
- churned 위험이 있는 기업 리스트
- 2026년 데이터 자동 적용 가능하도록 구조화
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print(" ⚠️ 위험 기업 분석 (2025년 현재 + 2026년 예측 준비)")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════
#  1. 데이터 및 모델 로드
# ══════════════════════════════════════════════════════════════════

print("\n[1단계] 데이터 및 모델 로드")
print("-" * 80)

df = pd.read_csv("data/csv/mlm_labeled_lite.csv")

# 재학습된 모델과 scaler 로드
scaler = joblib.load('models/retrain_scaler.pkl')
gb_risk = joblib.load('models/retrain_risk_GradientBoosting.pkl')
xgb_risk = joblib.load('models/retrain_risk_XGBoost.pkl')
lgbm_risk = joblib.load('models/retrain_risk_LightGBM.pkl')
gb_churn = joblib.load('models/retrain_churn_GradientBoosting.pkl')
xgb_churn = joblib.load('models/retrain_churn_XGBoost.pkl')
lgbm_churn = joblib.load('models/retrain_churn_LightGBM.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

print(f"✅ 모델 로드 완료: {len(feature_cols)}개 피처")

# 누출 피처 제거
leakage_cols = ['financial_health', 'sustainability', 'commission_flag']
if 'risk_score' in df.columns:
    leakage_cols.append('risk_score')

target_cols = ['risk_level', 'churned']
meta_cols = ['company_name', 'company_name_normalized', 'year']
exclude_cols = target_cols + meta_cols + leakage_cols

# feature_cols 재계산 (혹시 변경되었을 경우)
if 'feature_cols' not in dir() or not feature_cols:
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"⚠️ feature_cols 재계산: {len(feature_cols)}개")

# ══════════════════════════════════════════════════════════════════
#  2. 2025년 현재 위험 분석
# ══════════════════════════════════════════════════════════════════

print("\n[2단계] 2025년 현재 위험 기업 분석")
print("-" * 80)

# 2025년 데이터
current_year = df[df['year'] == 2025].copy()
print(f"\n✅ 2025년 데이터: {len(current_year)}개 기업")

# 피처 준비
X_current = current_year[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
X_current_s = scaler.transform(X_current)

# 예측
prob_gb_risk = gb_risk.predict_proba(X_current_s)
prob_xgb_risk = xgb_risk.predict_proba(X_current_s)
prob_lgbm_risk = lgbm_risk.predict_proba(X_current_s)
prob_ensemble_risk = (prob_gb_risk + prob_xgb_risk + prob_lgbm_risk) / 3
pred_risk = np.argmax(prob_ensemble_risk, axis=1)

prob_gb_churn = np.asarray(gb_churn.predict_proba(X_current_s))[:, 1]
prob_xgb_churn = np.asarray(xgb_churn.predict_proba(X_current_s))[:, 1]
prob_lgbm_churn = np.asarray(lgbm_churn.predict_proba(X_current_s))[:, 1]
prob_ensemble_churn = (prob_gb_churn + prob_xgb_churn + prob_lgbm_churn) / 3
pred_churn = (prob_ensemble_churn > 0.5).astype(int)

# 결과 데이터프레임 생성
result = pd.DataFrame({
    'company_name': current_year['company_name'].values,
    'company_name_normalized': current_year['company_name_normalized'].values,
    'year': current_year['year'].values,
    'actual_risk_level': current_year['risk_level'].values,
    'predicted_risk_level': pred_risk,
    'risk_prob_low': prob_ensemble_risk[:, 0],
    'risk_prob_mid': prob_ensemble_risk[:, 1],
    'risk_prob_high': prob_ensemble_risk[:, 2],
    'actual_churned': current_year['churned'].values,
    'predicted_churned': pred_churn,
    'churned_probability': prob_ensemble_churn
})

print(f"\n✅ 예측 완료")
print(f"   risk_level 분포: {(pred_risk == 0).sum()} Low / {(pred_risk == 1).sum()} Mid / {(pred_risk == 2).sum()} High")
print(f"   churned 분포: {(pred_churn == 0).sum()} Active / {(pred_churn == 1).sum()} Churned")

# ══════════════════════════════════════════════════════════════════
#  3. 위험 기업 필터링 및 sorting
# ══════════════════════════════════════════════════════════════════

print("\n[3단계] 위험 기업 식별")
print("-" * 80)

# 고위험 기업 (risk_level = High 또는 churned 확률 높음)
high_risk = result[result['predicted_risk_level'] == 2].copy()
churn_risk = result[result['predicted_churned'] == 1].copy()
elevated_risk = result[(result['predicted_risk_level'] == 1) & (result['churned_probability'] > 0.3)].copy()

print(f"\n▶ 고위험 기업 (risk_level = High)")
print(f"   총 {len(high_risk)}개 기업")
if len(high_risk) > 0:
    high_risk_sorted = high_risk.sort_values('risk_prob_high', ascending=False)
    print(f"\n   상위 10개:")
    for idx, (i, row) in enumerate(high_risk_sorted.head(10).iterrows(), 1):
        print(f"   {idx:2d}. {row['company_name']:30s} | Risk High Prob: {row['risk_prob_high']:.4f}")

print(f"\n▶ 폐업 위험 기업 (churned 위험)")
print(f"   총 {len(churn_risk)}개 기업")
if len(churn_risk) > 0:
    churn_risk_sorted = churn_risk.sort_values('churned_probability', ascending=False)
    print(f"\n   상위 10개:")
    for idx, (i, row) in enumerate(churn_risk_sorted.head(10).iterrows(), 1):
        print(f"   {idx:2d}. {row['company_name']:30s} | Churn Prob: {row['churned_probability']:.4f}")

print(f"\n▶ 중간 위험 기업 (risk_level = Mid + high churn prob)")
print(f"   총 {len(elevated_risk)}개 기업")
if len(elevated_risk) > 0:
    elevated_risk_sorted = elevated_risk.sort_values('churned_probability', ascending=False)
    print(f"\n   상위 10개:")
    for idx, (i, row) in enumerate(elevated_risk_sorted.head(10).iterrows(), 1):
        print(f"   {idx:2d}. {row['company_name']:30s} | Risk: {row['predicted_risk_level']} | Churn Prob: {row['churned_probability']:.4f}")

# ══════════════════════════════════════════════════════════════════
#  4. 위험 기업 리스트 저장
# ══════════════════════════════════════════════════════════════════

print("\n[4단계] 위험 기업 리스트 저장")
print("-" * 80)

# 통합 위험 점수 계산
result['risk_score'] = (
    result['predicted_risk_level'] * 0.4 +  # 위험 레벨 (0-2)
    result['churned_probability'] * 100 * 0.6  # 폐업 확률 (0-100)
)
result_sorted = result.sort_values('risk_score', ascending=False)

# 고위험 기업
high_risk_sorted = result[result['predicted_risk_level'] == 2].sort_values('risk_prob_high', ascending=False)
high_risk_sorted.to_csv('models/risk_companies_high_risk_2025.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ 고위험 기업 저장: models/risk_companies_high_risk_2025.csv ({len(high_risk_sorted)}개)")

# 폐업 위험 기업
churn_risk_sorted = result[result['predicted_churned'] == 1].sort_values('churned_probability', ascending=False)
churn_risk_sorted.to_csv('models/risk_companies_churn_2025.csv', index=False, encoding='utf-8-sig')
print(f"✅ 폐업 위험 기업 저장: models/risk_companies_churn_2025.csv ({len(churn_risk_sorted)}개)")

# 전체 위험 순위
result_sorted.to_csv('models/risk_companies_all_2025.csv', index=False, encoding='utf-8-sig')
print(f"✅ 전체 위험 순위 저장: models/risk_companies_all_2025.csv")

# ══════════════════════════════════════════════════════════════════
#  5. 미래 2026년 데이터 처리 함수
# ══════════════════════════════════════════════════════════════════

print("\n[5단계] 2026년 자동 분석 함수 저장")
print("-" * 80)

def analyze_future_year(year, df, scaler, models, feature_cols):
    """
    미래 연도 데이터 분석 함수
    
    Parameters:
    -----------
    year : int
        분석할 연도
    df : pd.DataFrame
        전체 데이터
    scaler : StandardScaler
        학습된 scaler
    models : dict
        학습된 모델들
    feature_cols : list
        피처 컬럼명
    
    Returns:
    --------
    pd.DataFrame : 위험도 분석 결과
    """
    
    # 해당 연도 데이터 추출
    year_data = df[df['year'] == year].copy()
    
    if len(year_data) == 0:
        print(f"경고: {year}년 데이터가 없습니다.")
        return None
    
    # 피처 준비
    X = year_data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_s = scaler.transform(X)
    
    # 모델 예측
    gb_risk, xgb_risk, lgbm_risk = models['risk']
    gb_churn, xgb_churn, lgbm_churn = models['churn']
    
    prob_risk = (gb_risk.predict_proba(X_s) + xgb_risk.predict_proba(X_s) + lgbm_risk.predict_proba(X_s)) / 3
    pred_risk = np.argmax(prob_risk, axis=1)
    
    prob_churn = (np.asarray(gb_churn.predict_proba(X_s))[:, 1] + 
                  np.asarray(xgb_churn.predict_proba(X_s))[:, 1] + 
                  np.asarray(lgbm_churn.predict_proba(X_s))[:, 1]) / 3
    pred_churn = (prob_churn > 0.5).astype(int)
    
    # 결과 데이터프레임
    result = pd.DataFrame({
        'company_name': year_data['company_name'].values,
        'year': year,
        'predicted_risk_level': pred_risk,
        'risk_prob_high': prob_risk[:, 2],
        'predicted_churned': pred_churn,
        'churned_probability': prob_churn
    })
    
    # 위험 점수
    result['risk_score'] = pred_risk * 0.4 + prob_churn * 100 * 0.6
    result = result.sort_values('risk_score', ascending=False)
    
    return result

# 함수 저장
joblib.dump(analyze_future_year, 'models/analyze_future_year_func.pkl')
print(f"\n✅ 미래 연도 분석 함수 저장: models/analyze_future_year_func.pkl")
print(f"   사용법: analyze_future_year(2026, df, scaler, models, feature_cols)")

# 모델 dict 저장 (함수 사용용)
models_dict = {
    'risk': (gb_risk, xgb_risk, lgbm_risk),
    'churn': (gb_churn, xgb_churn, lgbm_churn)
}
joblib.dump(models_dict, 'models/models_dict.pkl')
print(f"✅ 모델 dict 저장: models/models_dict.pkl")

# ══════════════════════════════════════════════════════════════════
#  6. 요약 및 안내
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(" 📊 위험 기업 분석 요약")
print("=" * 80)

total_companies = len(result)
high_risk_count = (result['predicted_risk_level'] == 2).sum()
mid_risk_count = (result['predicted_risk_level'] == 1).sum()
churn_count = (result['predicted_churned'] == 1).sum()

print(f"""
┌──────────────────────────────────────────────────────────────┐
│ 2025년 현황                                                 │
│  전체 기업: {total_companies:3d}개                                    │
├──────────────────────────────────────────────────────────────┤
│ 위험도 분류                                                 │
│  고위험 (Risk High):    {high_risk_count:3d}개 ({100*high_risk_count/total_companies:5.1f}%)               │
│  중간위험 (Risk Mid):   {mid_risk_count:3d}개 ({100*mid_risk_count/total_companies:5.1f}%)               │
│  폐업위험 (Churned):    {churn_count:3d}개 ({100*churn_count/total_companies:5.1f}%)               │
├──────────────────────────────────────────────────────────────┤
│ 생성된 파일                                                 │
│  ✓ risk_companies_high_risk_2025.csv                        │
│  ✓ risk_companies_churn_2025.csv                            │
│  ✓ risk_companies_all_2025.csv                              │
├──────────────────────────────────────────────────────────────┤
│ 2026년 자동 분석 준비 완료                                   │
│  24개월 뒤: 새 데이터 들어오면 자동 적용 가능               │
│  함수: analyze_future_year(year, df, scaler, models, cols)  │
└──────────────────────────────────────────────────────────────┘
""")

print("\n✅ 모든 분석 완료!")
