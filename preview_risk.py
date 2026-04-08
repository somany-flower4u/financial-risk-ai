import pandas as pd

print("="*80)
print("위험 기업 파일 미리보기")
print("="*80)

# 고위험 기업
print("\n[1] 고위험 기업 (risk_level = High)")
df_high = pd.read_csv('models/risk_companies_high_risk_2025.csv')
print(f"\n총 {len(df_high)}개 기업")
print("\n상위 5개:")
print(df_high[['company_name', 'year', 'predicted_risk_level', 'risk_prob_high', 'churned_probability']].head().to_string(index=False))

# 폐업 위험 기업
print("\n" + "="*80)
print("\n[2] 폐업 위험 기업 (churned 위험)")
df_churn = pd.read_csv('models/risk_companies_churn_2025.csv')
print(f"\n총 {len(df_churn)}개 기업")
print("\n상위 5개:")
print(df_churn[['company_name', 'year', 'predicted_churned', 'churned_probability']].head().to_string(index=False))

# 전체 위험 순위
print("\n" + "="*80)
print("\n[3] 전체 위험 순위 (통합 위험 점수)")
df_all = pd.read_csv('models/risk_companies_all_2025.csv')
print(f"\n총 {len(df_all)}개 기업")
print("\n상위 5개 (가장 위험한 기업):")
print(df_all[['company_name', 'year', 'predicted_risk_level', 'predicted_churned', 'risk_score']].head().to_string(index=False))

print("\n" + "="*80)
print("✅ 위험 기업 분석 완료!")
print("="*80)
