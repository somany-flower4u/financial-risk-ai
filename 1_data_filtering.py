import pandas as pd
import numpy as np
import re
import os
import glob

# ============================================================
# 1. 설정
# ============================================================
DATA_DIR = "data/xls"  # <-- 실제 경로로 변경
OUTPUT_PATH = "data/csv/mlm_unified_dataset.csv"

# ============================================================
# 2. 컬럼 인덱스 → 표준 컬럼명 매핑 (71열 기준)
# ============================================================
COLUMN_INDEX_MAP = {
    # 회사 기본 정보
    0: 'company_name',           # 회사명
    1: 'ceo_name',               # 대표이사
    2: 'registration_no',        # 다단계판매업등록번호
    3: 'registration_date',      # 다단계판매업등록일
    4: 'hq_address',             # 본사 소재지
    5: 'hq_phone',               # 본사 전화

    # 최근 3년간 변경내역
    6: 'change_ceo',             # 대표이사 변경
    7: 'change_company_name',    # 상호 변경
    8: 'change_directors',       # 등기이사 변경
    9: 'change_hq_address',      # 본사소재지 변경

    # 기타
    10: 'business_transfer',     # 영업양도,양수,합병

    # 매출액
    11: 'revenue_vat_included',  # 총매출액(부가가치세 포함, 원)
    12: 'revenue_financial',     # 재무제표상 매출액(원)
    13: 'top1_revenue',          # 상위1위 매출액
    14: 'top1_product',          # 상위1위 품목
    15: 'top2_revenue',          # 상위2위 매출액
    16: 'top2_product',          # 상위2위 품목
    17: 'top3_revenue',          # 상위3위 매출액
    18: 'top3_product',          # 상위3위 품목
    19: 'top4_revenue',          # 상위4위 매출액
    20: 'top4_product',          # 상위4위 품목
    21: 'top5_revenue',          # 상위5위 매출액
    22: 'top5_product',          # 상위5위 품목

    # 기본사항
    23: 'commission_total',          # 후원수당총액(원)
    24: 'commission_ratio_pct',      # 총매출액 대비 후원수당비율(%)
    25: 'refund_request_count',      # 반품·환불요청건수
    26: 'refund_processed_count',    # 반품·환불처리건수
    27: 'refund_request_amount',     # 반품·환불요청액수(원)
    28: 'refund_processed_amount',   # 반품·환불처리액수(원)
    29: 'cs_department',             # 담당부서명
    30: 'cs_phone',                  # 담당부서연락처
    31: 'total_agents_prev_year',    # 전년도 등록 총판매원수
    32: 'commission_recipients',     # 후원수당을 지급받고 있는 판매원수

    # 재무
    33: 'net_income_prev_year',      # 전년도 당기순이익(원)
    34: 'allowance_bad_debt',        # 전년도 대손충당금(원)
    35: 'total_assets',              # 자산(원)
    36: 'total_liabilities',         # 부채(원)
    37: 'capital',                   # 자본금(원)
    38: 'consumer_protection_org',   # 소비자피해보상가입기관
    39: 'consumer_protection_fund',  # 소비자피해보상준비금(원)

    # 신용평가
    40: 'credit_rating',             # 등급
    41: 'credit_agency',             # 기관명칭

    # 시정조치
    42: 'corrective_action_no',      # 의결번호(문서번호)
    43: 'corrective_action_date',    # 조치기관 조치일자
    44: 'violation_details',         # 위반내용(위반법조문)
    45: 'corrective_action_type',    # 조치내용
    46: 'corrective_action_note',    # 비고(이의신청이 있는 경우)

    # 후원수당 지급분포도 - 후원수당 수령 판매원 기준
    47: 'dist_r_top1pct_total',       # 상위1% 후원수당총지급액(원)
    48: 'dist_r_top1pct_avg',         # 상위1% 1인당평균지급액(원)
    49: 'dist_r_1to6pct_total',       # 1%~6% 후원수당총지급액(원)
    50: 'dist_r_1to6pct_avg',         # 1%~6% 1인당평균지급액(원)
    51: 'dist_r_6to30pct_total',      # 6%~30% 후원수당총지급액(원)
    52: 'dist_r_6to30pct_avg',        # 6%~30% 1인당평균지급액(원)
    53: 'dist_r_30to60pct_total',     # 30%~60% 후원수당총지급액(원)
    54: 'dist_r_30to60pct_avg',       # 30%~60% 1인당평균지급액(원)
    55: 'dist_r_60to100pct_total',    # 60%~100% 후원수당총지급액(원)
    56: 'dist_r_60to100pct_avg',      # 60%~100% 1인당평균지급액(원)
    57: 'dist_r_annual_total',        # 연간 후원수당총지급액 합계(원)
    58: 'dist_r_annual_avg',          # 연간 1인당후원수당평균지급액(원)

    # 후원수당 지급분포도 - 전체 판매원 기준
    59: 'dist_a_top1pct_total',       # 상위1% 후원수당총지급액(원)
    60: 'dist_a_top1pct_avg',         # 상위1% 1인당평균지급액(원)
    61: 'dist_a_1to6pct_total',       # 1%~6% 후원수당총지급액(원)
    62: 'dist_a_1to6pct_avg',         # 1%~6% 1인당평균지급액(원)
    63: 'dist_a_6to30pct_total',      # 6%~30% 후원수당총지급액(원)
    64: 'dist_a_6to30pct_avg',        # 6%~30% 1인당평균지급액(원)
    65: 'dist_a_30to60pct_total',     # 30%~60% 후원수당총지급액(원)
    66: 'dist_a_30to60pct_avg',       # 30%~60% 1인당평균지급액(원)
    67: 'dist_a_60to100pct_total',    # 60%~100% 후원수당총지급액(원)
    68: 'dist_a_60to100pct_avg',      # 60%~100% 1인당평균지급액(원)
    69: 'dist_a_annual_total',        # 연간 후원수당총지급액 합계(원)
    70: 'dist_a_annual_avg',          # 연간 1인당후원수당평균지급액(원)
}

# ============================================================
# 3. 숫자 정제 함수
# ============================================================
def clean_numeric(value):
    """다양한 형태의 숫자 데이터를 float으로 변환"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()

    # 빈 값, 대시, 해당없음 등 처리
    if s in ['', '-', 'N/A', 'n/a', '해당없음', '없음', '비공개', '*', 
             '미공개', '변경없음', '해당 없음', '변경내역 없음', '해당사항없음',
             '변경내역없음', '해당사항 없음', '해당없슴']:
        return np.nan

    # 쉼표 제거
    s = s.replace(',', '')

    # 괄호 안 숫자 (음수 표기): (123) → -123
    match = re.match(r'^\((\d+\.?\d*)\)$', s)
    if match:
        return -float(match.group(1))

    # 숫자만 추출 (앞에 - 포함 가능)
    match = re.match(r'^-?\d+\.?\d*', s)
    if match:
        return float(match.group())

    return np.nan

def clean_text(value):
    """텍스트 필드 정제"""
    if pd.isna(value):
        return None
    s = str(value).strip()
    # \t, \n, \r 제거 및 정규화
    s = re.sub(r'[\t\r]', '', s)
    s = re.sub(r'\n+', ' ', s)
    s = re.sub(r'\s+', ' ', s)

    if s in ['', '-', 'NaN', 'nan', 'None']:
        return None
    return s

def clean_company_name(name):
    """회사명 정규화"""
    if pd.isna(name) or name is None:
        return None
    s = str(name).strip()
    # 일반적인 변환
    s = s.replace('㈜', '(주)')
    s = s.replace('(유)', '(유)')
    s = re.sub(r'\s+', '', s)  # 회사명 내 공백 제거 (매칭 용이)
    return s

def clean_phone(value):
    """전화번호 정제"""
    if pd.isna(value):
        return None
    s = str(value).strip()
    s = s.replace('\t', '').replace('\\t', '')
    s = re.sub(r'[^\d\-\*]', '', s)  # 숫자, 하이픈 *  남김
    if not s or s == '-':
        return None
    return s

def clean_date(value):
    """등록일 정제 → YYYY-MM-DD 형태"""
    if pd.isna(value):
        return None
    s = str(value).strip()

    # 8자리 숫자 (20120806)
    match = re.match(r'^(\d{4})(\d{2})(\d{2})$', s)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    # 이미 형식이 맞는 경우

    match = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', s)
    if match:
        return s
    # 숫자만 추출 시도
    digits = re.sub(r'[^\d]', '', s)
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return s
    
def clean_ratio(value):
    """비율(%) 정제 → float"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace('%', '').replace(',', '')
    try:
        return float(s)
    except ValueError:
        return np.nan
    
def process_single_file(filepath, year):
    """
    단일 XLS 파일을 읽어서 정제된 DataFrame 반환
    """
    print(f"\n{'='*60}")
    print(f"📄 처리 중: {os.path.basename(filepath)} (연도: {year})")
    print(f"{'='*60}")
    
    # 1) 파일 읽기 (header=None, 원본 그대로)
    df_raw = pd.read_excel(filepath, header=None, sheet_name=0)
    print(f"  원본 크기: {df_raw.shape[0]}행 * {df_raw.shape[1]}열")
    
    # 2) 데이터 행만 추출 (행 5부터, 0-indexed)
    df_data = df_raw.iloc[5:].reset_index(drop=True)
    print(f"  데이터 행: {len(df_data)}개")
    
    # 3) 인덱스 기반 컬럼명 부여
    col_names = []
    for i in range(df_data.shape[1]):
        if i in COLUMN_INDEX_MAP:
            col_names.append(COLUMN_INDEX_MAP[i])
        else:
            col_names.append(f'unknown_{i}')
    df_data.columns = col_names
    
    # 4) 완전히 빈 행 제거
    df_data = df_data.dropna(how='all').reset_index(drop=True)
    
    # 5) 회사명이 비어있는 행 제거 (합계 행 등)
    df_data = df_data[df_data['company_name'].notna()].reset_index(drop=True)
    
    # 6) 요약/합계 행 제거
    summary_keywords = ['합계', '총계', '평균', '소계', 'Total', 'total', '전체', '계']
    mask = ~df_data['company_name'].astype(str).apply(
        lambda x: any(kw == x.strip() for kw in summary_keywords)
    )
    removed = (~mask).sum()
    if removed > 0:
        print(f"  🗑️ 요약행 {removed}개 제거")
    df_data = df_data[mask].reset_index(drop=True)
    
    # 7) 텍스트 필드 정제
    text_cols = [
        'company_name', 'ceo_name', 'registration_no', 'hq_address', 'hq_phone',
        'change_ceo', 'change_company_name', 'change_directors', 'change_hq_address',
        'business_transfer', 'cs_department', 'cs_phone',
        'consumer_protection_org', 'credit_rating', 'credit_agency',
        'corrective_action_no', 'corrective_action_date',
        'violation_details', 'corrective_action_type', 'corrective_action_note',
        'top1_product', 'top2_product', 'top3_product', 'top4_product', 'top5_product'
    ]
    for col in text_cols:
        if col in df_data.columns:
            df_data[col] = df_data[col].apply(clean_text)
            
    # 8) 회사명 정규화
    df_data['company_name_normalized'] = df_data['company_name'].apply(clean_company_name)
    
    # 9) 전화번호 정제
    for col in ['hq_phone', 'cs_phone']:
        if col in df_data.columns:
            df_data[col] = df_data[col].apply(clean_phone)
            
    # 10) 등록일 정제
    df_data['registration_date'] = df_data['registration_date'].apply(clean_date)
    
    # 11) 숫자 필드 정제
    numeric_cols = [
        'revenue_vat_included', 'revenue_financial',
        'top1_revenue', 'top2_revenue', 'top3_revenue', 'top4_revenue', 'top5_revenue',
        'commission_total',
        'refund_request_count', 'refund_processed_count',
        'refund_request_amount', 'refund_processed_amount',
        'total_agents_prev_year', 'commission_recipients',
        'net_income_prev_year', 'allowance_bad_debt',
        'total_assets', 'total_liabilities', 'capital',
        'consumer_protection_fund',

        # 후원수당 분포 (수령 판매원 기준)
        'dist_r_top1pct_total', 'dist_r_top1pct_avg',
        'dist_r_1to6pct_total', 'dist_r_1to6pct_avg',
        'dist_r_6to30pct_total', 'dist_r_6to30pct_avg',
        'dist_r_30to60pct_total', 'dist_r_30to60pct_avg',
        'dist_r_60to100pct_total', 'dist_r_60to100pct_avg',
        'dist_r_annual_total', 'dist_r_annual_avg',

        # 후원수당 분포 (전체 판매원 기준)
        'dist_a_top1pct_total', 'dist_a_top1pct_avg',
        'dist_a_1to6pct_total', 'dist_a_1to6pct_avg',
        'dist_a_6to30pct_total', 'dist_a_6to30pct_avg',
        'dist_a_30to60pct_total', 'dist_a_30to60pct_avg',
        'dist_a_60to100pct_total', 'dist_a_60to100pct_avg',
        'dist_a_annual_total', 'dist_a_annual_avg',
    ]
    for col in numeric_cols:
        if col in df_data.columns:
            df_data[col] = df_data[col].apply(clean_numeric)
            
    # 12) 비율 필드 정제
    df_data['commission_ratio_pct'] = df_data['commission_ratio_pct'].apply(clean_ratio)
    
    # 13) 연도 컬럼 추가
    df_data['year'] = year
    print(f"  ✅ 정제 완료: {len(df_data)}개 업체")
    return df_data

def build_unified_dataset():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.xls")) + 
               glob.glob(os.path.join(DATA_DIR, "*.xlsx")))

    print(f"📁 발견된 파일 수: {len(files)}")
    all_dfs = []
    for f in files:
        fname = os.path.basename(f)
        
        # 파일명에서 연도 추출
        year_match = re.search(r'(20\d{2})', fname)
        if year_match is None:
            print(f"⚠️ {fname}: 연도 추출 불가, 건너뜀")
            continue
        year = int(year_match.group(1))
        try:
            df = process_single_file(f, year)
            all_dfs.append(df)
        except Exception as e:
            print(f"❌ {fname} 처리 실패: {e}")
            import traceback
            traceback.print_exc()
    if not all_dfs:
        print("❌ 통합할 데이터가 없습니다.")
        return pd.DataFrame()
    
    # 통합
    df_all = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    # ================================================================
    # 컬럼 순서 정리 (year를 맨 앞으로)
    # ================================================================
    cols = list(df_all.columns)
    cols.remove('year')
    cols = ['year'] + cols
    df_all = df_all[cols]
    return df_all

#7. 데이터 검증
df_unified = build_unified_dataset()
print(f"\n{'='*60}")
print(f"📊 통합 데이터 검증")
print(f"{'='*60}")
print(f"\n총 행 수: {len(df_unified)}")
print(f"총 컬럼 수: {len(df_unified.columns)}")
print(f"연도 범위: {df_unified['year'].min()} ~ {df_unified['year'].max()}")
print(f"\n📈 연도별 업체 수:")
print(df_unified['year'].value_counts().sort_index())
print(f"\n📌 고유 업체 수 (정규화 기준): {df_unified['company_name_normalized'].nunique()}")

key_numeric = [
'revenue_vat_included', 'revenue_financial', 'commission_total',
'commission_ratio_pct', 'total_agents_prev_year', 'commission_recipients',
'net_income_prev_year', 'total_assets', 'total_liabilities', 'capital'
]

# 주요 숫자 컬럼 결측치 현황
print(f"\n📋 주요 숫자 컬럼 결측치:")
for col in key_numeric:
    if col in df_unified.columns:
        null_count = df_unified[col].isna().sum()
        null_pct = null_count / len(df_unified) * 100
        print(f"  {col:35s}: {null_count:4d} ({null_pct:.1f}%)")

# 후원수당 비율 검증
print(f"\n💰 후원수당 비율(%) 통계:")
print(df_unified['commission_ratio_pct'].describe())

# 비율 50% 초과 업체 (이상값 점검)
suspicious = df_unified[df_unified['commission_ratio_pct'] > 50]
if len(suspicious) > 0:
    print(f"\n⚠️ 후원수당 비율 50% 초과 업체 {len(suspicious)}건:")
    print(suspicious[['year', 'company_name', 'commission_ratio_pct', 
                    'revenue_vat_included', 'commission_total']].to_string())
    
# 매출액 교차 검증: distrannualtotal ≈ commissiontotal 여부
df_check = df_unified[df_unified['dist_r_annual_total'].notna() & df_unified['commission_total'].notna()].copy()
if len(df_check) > 0:
    df_check['diff'] = abs(df_check['dist_r_annual_total'] - df_check['commission_total'])
    df_check['diff_pct'] = df_check['diff'] / df_check['commission_total'] * 100
    mismatch = df_check[df_check['diff_pct'] > 1]  # 1% 이상 차이
    print(f"\n🔍 후원수당 교차검증: dist_r_annual_total vs commission_total")
    print(f"   전체 비교 가능: {len(df_check)}건")
    print(f"   1% 이상 차이: {len(mismatch)}건")
    
df_unified.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"\n💾 저장 완료: {OUTPUT_PATH}")
print(f"   크기: {len(df_unified)}행 * {len(df_unified.columns)}열")

# 주요 분석용 컬럼만 추출한 경량 버전도 저장
key_cols = [
'year', 'company_name', 'company_name_normalized', 'ceo_name',
'registration_no', 'registration_date', 'hq_address',
'revenue_vat_included', 'revenue_financial', 'commission_total',
'commission_ratio_pct', 'total_agents_prev_year', 'commission_recipients',
'net_income_prev_year', 'total_assets', 'total_liabilities', 'capital',
'consumer_protection_org', 'consumer_protection_fund',
'refund_request_count', 'refund_processed_count',
'refund_request_amount', 'refund_processed_amount',

# 후원수당 분포 (수령 판매원)
'dist_r_top1pct_total', 'dist_r_top1pct_avg',
'dist_r_1to6pct_total', 'dist_r_1to6pct_avg',
'dist_r_6to30pct_total', 'dist_r_6to30pct_avg',
'dist_r_30to60pct_total', 'dist_r_30to60pct_avg',
'dist_r_60to100pct_total', 'dist_r_60to100pct_avg',
'dist_r_annual_total', 'dist_r_annual_avg',

# 후원수당 분포 (전체 판매원)
'dist_a_top1pct_total', 'dist_a_top1pct_avg',
'dist_a_1to6pct_total', 'dist_a_1to6pct_avg',
'dist_a_6to30pct_total', 'dist_a_6to30pct_avg',
'dist_a_30to60pct_total', 'dist_a_30to60pct_avg',
'dist_a_60to100pct_total', 'dist_a_60to100pct_avg',
'dist_a_annual_total', 'dist_a_annual_avg',

# 상위 판매 품목
'top1_product', 'top1_revenue',
'top2_product', 'top2_revenue',
'top3_product', 'top3_revenue',
]

df_lite = df_unified[[c for c in key_cols if c in df_unified.columns]].copy()
litepath = OUTPUT_PATH.replace('.csv', 'lite.csv')
df_lite.to_csv(litepath, index=False, encoding='utf-8-sig')
print(f"💾 경량 버전 저장: {litepath}")
print(f"   크기: {len(df_lite)}행 * {len(df_lite.columns)}열")