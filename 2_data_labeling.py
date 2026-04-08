"""
MLM 데이터 라벨링 스크립트
법적 기준 + 재무 건전성 + 사업 지속가능성 기반 다차원 라벨링
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 로드
# ============================================================
df = pd.read_csv('data/csv/mlm_unified_dataset.csv')
print(f"📂 데이터 로드: {df.shape[0]}행 * {df.shape[1]}열")
print(f"   연도 범위: {df['year'].min()} ~ {df['year'].max()}")
print(f"   고유 업체 수: {df['company_name_normalized'].nunique()}")

# ============================================================
# 2. 파생 피처 생성 (라벨링에 필요한 지표)
# ============================================================
print("\n" + "=" * 10)
print("📐 파생 피처 생성")
print("=" * 10)

# 2-1. 부채비율 = 부채총계 / 자본총 * 00
# 자본이 0 이하인 경우 자본잠식으로 별도 처리
df['equity'] = df['total_assets'] - df['total_liabilities']  # 자본총계 추정
df['debt_ratio'] = np.where(
    df['equity'] > 0,
    (df['total_liabilities'] / df['equity']) * 100,
    np.inf  # 자본잠식 시 무한대
)

# 2-2. 자본잠식 여부
df['is_capital_impaired'] = (df['equity'] <= 0).astype(int)

# 2-3. 당기순손실 여부
df['is_net_loss'] = (df['net_income_prev_year'] < 0).astype(int)

# 2-4. 연도별 매출 변화율 (동일 업체 기준)
df = df.sort_values(['company_name_normalized', 'year'])
df['revenue_yoy_change'] = df.groupby('company_name_normalized')['revenue_vat_included'].pct_change() * 100

# 2-5. 연도별 판매원 수 변화율 (동일 업체 기준)
df['agents_yoy_change'] = df.groupby('company_name_normalized')['total_agents_prev_year'].pct_change() * 100

# 2-6. 1인당 후원수당 (후원수당 수령자 기준)
df['commission_per_recipient'] = np.where(
    df['commission_recipients'] > 0,
    df['commission_total'] / df['commission_recipients'],
    0
)

# 2-7. 판매원 대비 수당 수령 비율
df['commission_recipient_ratio'] = np.where(
    df['total_agents_prev_year'] > 0,
    (df['commission_recipients'] / df['total_agents_prev_year']) * 100,
    0
)

print(f"  ✅ 부채비율 계산 완료")
print(f"  ✅ 자본잠식 업체: {df['is_capital_impaired'].sum()}건")
print(f"  ✅ 당기순손실 업체: {df['is_net_loss'].sum()}건")
print(f"  ✅ 매출 YoY 변화율 계산 완료 (첫 연도 NaN: {df['revenue_yoy_change'].isna().sum()}건)")

# ============================================================
# 3. Label 1: commission_flag (후원수당 비율 위반)
# ============================================================
print("\n" + "=" * 10)
print("🏷️  Label 1: commission_flag (후원수당 비율 위반)")
print("=" * 10)

def label_commission(ratio):
    """
    방문판매법 제20조 제3항 기준:
    - 35% 이하: 정상 (0)
    - 35% 초과 ~ 50% 이하: 법률 위반 (1)
    - 50% 초과: 중대 위반 (2)
    """
    if ratio <= 35:
        return 0
    elif ratio <= 50:
        return 1
    else:
        return 2

df['commission_flag'] = df['commission_ratio_pct'].apply(label_commission)

print(f"  정상 (≤35%):    {(df['commission_flag'] == 0).sum()}건")
print(f"  위반 (35~50%):  {(df['commission_flag'] == 1).sum()}건")
print(f"  중대위반 (>50%): {(df['commission_flag'] == 2).sum()}건")

# ============================================================
# 4. Label 2: financial_health (재무 건전성)
# ============================================================
print("\n" + "=" * 10)
print("🏷️  Label 2: financial_health (재무 건전성)")
print("=" * 10)

def label_financial_health(row):
    """
    재무 건전성 등급:
    - 위험 (2): 자본잠식 또는 부채비율 500% 초과
    - 주의 (1): 부채비율 200% 초과 또는 당기순손실
    - 건전 (0): 그 외
    """
    # 위험: 자본잠식 상태
    if row['is_capital_impaired'] == 1:
        return 2
    # 위험: 부채비율 500% 초과
    if row['debt_ratio'] > 500:
        return 2
    # 주의: 부채비율 200% 초과
    if row['debt_ratio'] > 200:
        return 1
    # 주의: 당기순손실
    if row['is_net_loss'] == 1:
        return 1
    # 건전
    return 0

df['financial_health'] = df.apply(label_financial_health, axis=1)

print(f"  건전 (0): {(df['financial_health'] == 0).sum()}건")
print(f"  주의 (1): {(df['financial_health'] == 1).sum()}건")
print(f"  위험 (2): {(df['financial_health'] == 2).sum()}건")

# ============================================================
# 5. Label 3: sustainability (사업 지속가능성)
# ============================================================
print("\n" + "=" * 10)
print("🏷️  Label 3: sustainability (사업 지속가능성)")
print("=" * 10)

def label_sustainability(row):
    """
    사업 지속가능성 등급:
    - 위험 (2): 매출 30% 이상 감소 AND 판매원 30% 이상 감소
    - 주의 (1): 매출 30% 이상 감소 OR 판매원 30% 이상 감소
    - 양호 (0): 그 외 (첫 연도 NaN 포함)
    """
    rev_change = row['revenue_yoy_change']
    agent_change = row['agents_yoy_change']

    # 첫 연도 데이터는 변화율이 NaN이므로 기본값 0 (양호)
    if pd.isna(rev_change) or pd.isna(agent_change):
        return 0

    rev_decline = rev_change <= -30
    agent_decline = agent_change <= -30

    if rev_decline and agent_decline:
        return 2
    elif rev_decline or agent_decline:
        return 1
    else:
        return 0

df['sustainability'] = df.apply(label_sustainability, axis=1)

print(f"  양호 (0): {(df['sustainability'] == 0).sum()}건")
print(f"  주의 (1): {(df['sustainability'] == 1).sum()}건")
print(f"  위험 (2): {(df['sustainability'] == 2).sum()}건")

# ============================================================
# 6. Label 4: risk_level (종합 위험도, 최종 타겟 라벨)
# ============================================================
print("\n" + "=" * 10)
print("🏷️  Label 4: risk_level (종합 위험도)")
print("=" * 10)

def label_risk_level(row):
    """
    종합 위험도 산정:
    - 개별 라벨을 위험 신호(signal)로 집계
    - 심각 신호: commission_flag==2, financial_health==2, sustainability==2
    - 경미 신호: commission_flag==1, financial_health==1, sustainability==1

    위험(2): 심각 신호 1개 이상 OR 경미 신호 3개
    주의(1): 경미 신호 1~2개
    정상(0): 신호 없음
    """
    severe_signals = sum([
        row['commission_flag'] == 2,
        row['financial_health'] == 2,
        row['sustainability'] == 2,
    ])
    mild_signals = sum([
        row['commission_flag'] == 1,
        row['financial_health'] == 1,
        row['sustainability'] == 1,
    ])

    if severe_signals >= 1 or mild_signals >= 3:
        return 2  # 위험
    elif mild_signals >= 1:
        return 1  # 주의
    else:
        return 0  # 정상

df['risk_level'] = df.apply(label_risk_level, axis=1)

print(f"  정상 (0): {(df['risk_level'] == 0).sum()}건")
print(f"  주의 (1): {(df['risk_level'] == 1).sum()}건")
print(f"  위험 (2): {(df['risk_level'] == 2).sum()}건")

# ============================================================
# 7. 라벨 분포 검증 및 교차 분석
# ============================================================
print("\n" + "=" * 10)
print("📊 라벨 분포 검증")
print("=" * 10)

# 연도별 위험도 분포
print("\n📈 연도별 종합 위험도 분포:")
risk_by_year = pd.crosstab(df['year'], df['risk_level'], margins=True)
risk_by_year.columns = ['정상(0)', '주의(1)', '위험(2)', '전체']
print(risk_by_year)

# 라벨 간 상관관계
print("\n🔗 라벨 간 상관 매트릭스:")
label_cols = ['commission_flag', 'financial_health', 'sustainability', 'risk_level']
print(df[label_cols].corr().round(3))

# 위험 업체 상세 조회
print("\n⚠️  위험(2) 등급 업체 상세:")
danger = df[df['risk_level'] == 2][
    ['year', 'company_name', 'commission_ratio_pct', 'revenue_vat_included',
     'is_capital_impaired', 'debt_ratio', 'commission_flag',
     'financial_health', 'sustainability', 'risk_level']
].sort_values(['company_name', 'year'])
print(danger.to_string(index=False))

# ============================================================
# 8. 퇴출 업체 라벨 (다음 연도에 사라진 업체)
# ============================================================
print("\n" + "=" * 10)
print("🏷️  보너스 Label: churned (퇴출/폐업 여부)")
print("=" * 10)

# 각 업체의 마지막 등장 연도 확인
last_year = df.groupby('company_name_normalized')['year'].max().reset_index()
last_year.columns = ['company_name_normalized', 'last_seen_year']
df = df.merge(last_year, on='company_name_normalized', how='left')

# 2024년(최종 연도) 이전에 마지막으로 관측된 업체 = 퇴출
# 단, 해당 업체의 마지막 관측 행에만 churned=1 부여
df['churned'] = 0
mask = (df['year'] == df['last_seen_year']) & (df['last_seen_year'] < df['year'].max())
df.loc[mask, 'churned'] = 1

print(f"  퇴출 라벨 부여: {df['churned'].sum()}건")
print(f"  (최종연도 {df['year'].max()} 이전 마지막 관측 행 기준)")

# 퇴출 업체의 직전 위험도 분포
churned_risk = df[df['churned'] == 1]['risk_level'].value_counts().sort_index()
print(f"\n  퇴출 업체의 위험도 분포:")
for level, count in churned_risk.items():
    if not isinstance(level, (int, np.integer, str)):
        continue
    try:
        level_int = int(level)
    except (TypeError, ValueError):
        continue
    label_name = {0: '정상', 1: '주의', 2: '위험'}.get(level_int, '미정')
    print(f"    {label_name}({level_int}): {count}건")

# ============================================================
# 9. 저장
# ============================================================
print("\n" + "=" * 10)
print("💾 라벨링 완료 데이터 저장")
print("=" * 10)

# 라벨 컬럼 목록
label_columns = [
    'commission_flag', 'financial_health', 'sustainability',
    'risk_level', 'churned'
]
derived_columns = [
    'equity', 'debt_ratio', 'is_capital_impaired', 'is_net_loss',
    'revenue_yoy_change', 'agents_yoy_change',
    'commission_per_recipient', 'commission_recipient_ratio'
]

# 전체 저장
output_path = 'data/csv/mlm_labeled_dataset.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  ✅ 전체 데이터: {output_path}")
print(f"     크기: {df.shape[0]}행 * {df.shape[1]}열")

# 라벨 + 주요 피처만 저장
key_cols = [
    'year', 'company_name', 'company_name_normalized',
    'revenue_vat_included', 'revenue_financial',
    'commission_total', 'commission_ratio_pct',
    'total_agents_prev_year', 'commission_recipients',
    'net_income_prev_year', 'total_assets', 'total_liabilities', 'capital',
] + derived_columns + label_columns

lite_path = 'data/csv/mlm_labeled_lite.csv'
df[key_cols].to_csv(lite_path, index=False, encoding='utf-8-sig')
print(f"  ✅ 경량 버전: {lite_path}")
print(f"     크기: {df[key_cols].shape[0]}행 * {df[key_cols].shape[1]}열")

# ============================================================
# 10. 최종 요약
# ============================================================
print("\n" + "=" * 10)
print("📋 라벨링 최종 요약")
print("=" * 10)

print(f"""
┌─────────────────────────────────────────────────┐
│  총 데이터: {df.shape[0]}행                              │
│  라벨 컬럼: {len(label_columns)}개                                │
│  파생 피처: {len(derived_columns)}개                                │
├─────────────────────────────────────────────────┤
│  [commission_flag] 후원수당 위반                 │
│    0(정상): {(df['commission_flag']==0).sum():>4}건  │
│    1(위반): {(df['commission_flag']==1).sum():>4}건  │
│    2(중대): {(df['commission_flag']==2).sum():>4}건  │
├─────────────────────────────────────────────────┤
│  [financial_health] 재무 건전성                  │
│    0(건전): {(df['financial_health']==0).sum():>4}건  │
│    1(주의): {(df['financial_health']==1).sum():>4}건  │
│    2(위험): {(df['financial_health']==2).sum():>4}건  │
├─────────────────────────────────────────────────┤
│  [sustainability] 사업 지속가능성                │
│    0(양호): {(df['sustainability']==0).sum():>4}건  │
│    1(주의): {(df['sustainability']==1).sum():>4}건  │
│    2(위험): {(df['sustainability']==2).sum():>4}건  │
├─────────────────────────────────────────────────┤
│  [risk_level] 종합 위험도 (최종 타겟)            │
│    0(정상): {(df['risk_level']==0).sum():>4}건  │
│    1(주의): {(df['risk_level']==1).sum():>4}건  │
│    2(위험): {(df['risk_level']==2).sum():>4}건  │
├─────────────────────────────────────────────────┤
│  [churned] 퇴출 업체: {df['churned'].sum():>4}건                     │
└─────────────────────────────────────────────────┘
""")
