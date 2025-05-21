import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from PIL import Image

# 페이지 설정은 앱의 첫 번째 Streamlit 명령이어야 합니다
st.set_page_config(page_title="장기재공/재고 부진화 경험율 계산App", layout="wide")

# ——————————————————————————————
# 헤더 이미지 로드 및 출력
# (프로젝트 폴더에 header.png 를 두거나 URL을 지정하세요)
header_img = Image.open("header.png")
st.image(header_img, use_container_width=True)
# ——————————————————————————————

###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

# 데이터 캐싱을 위한 데코레이터 : 소재성 재공
@st.cache_data
def load_소재성_재공():
    # 데이터프레임을 불러옵니다.
    df = pd.read_parquet('dataset.parquet')
    # 소재성 재공
    check_data = df.loc[df['GBN_NAME'] == '소재성 재공'].copy()
    # 특정 컬럼의 null 데이터를 "일품"으로 대체
    check_data['LOT_NO'] = check_data['LOT_NO'].fillna('일품')
    # 조건에 따라 'LOT_NO' 컬럼의 값을 '일품'으로 대체
    check_data.loc[check_data['LOT_NO'].str[:6] == check_data['HEAT_NO'], 'LOT_NO'] = '일품'
    # 해당 컬럼의 음수 또는 null 데이터를 0으로 대체
    check_data['PASS_MONTH'] = check_data['PASS_MONTH'].apply(lambda x: 0 if pd.isnull(x) or x < 0 else x)
    # 모든 컬럼의 null 데이터를 '정보없음'으로 대체
    check_data.fillna('정보없음', inplace=True)
    # A부터 I 컬럼까지의 컬럼명을 리스트로 추출합니다.
    group_columns = check_data.columns[check_data.columns.get_loc('GBN_NAME') : check_data.columns.get_loc('PASS_MONTH') + 1].tolist()
    # 그룹핑하여 J 컬럼의 합계를 계산하고 새로운 데이터프레임 생성
    result_df = check_data.groupby(group_columns, as_index=False)['WGT'].sum()
    # 중복을 확인할 컬럼명 지정
    duplicate_columns = ['YM', 'LOT_NO', 'HEAT_NO']
    # 중복된 행을 확인하고 결과를 데이터프레임으로 추출
    duplicates = result_df[result_df.duplicated(subset=duplicate_columns, keep=False)]
    # 'YM' 컬럼을 기준으로 그룹화
    grouped = result_df.groupby('YM')
    # 그룹별 데이터프레임 저장할 딕셔너리 초기화
    grouped_dataframes = {}
    # 각 그룹을 딕셔너리에 저장
    for ym, group_df in grouped:
        grouped_dataframes[ym] = group_df.reset_index(drop=True)  # 데이터프레임 저장
    # 엑셀 파일 경로 지정
    input_file = grouped_dataframes
    # 엑셀 파일의 모든 시트 이름 및 시트 수 로드
    sheet_count = len(input_file)
    # 마지막 시트의 이름 확인
    last_sheet_name = sorted(input_file.keys())[-1]
    # initial_import_index 설정
    initial_import_index = 0
    # initial index sheet 데이터 import
    df = list(input_file.values())[0]
    # 두 개의 컬럼을 문자열로 변환한 후 합쳐 새로운 파생변수 생성
    df['HEAT_LOT'] = df['HEAT_NO'].astype(str) + '_' + df['LOT_NO'].astype(str)
    # 연령계산 컬럼 생성: 조건에 따라 라벨링
    def label_c(row):
        if pd.isnull(row['PASS_MONTH']) or row['PASS_MONTH'] in range(0,4):
            return 1
        elif row['PASS_MONTH'] in range(4,7):
            return 2
        elif row['PASS_MONTH'] in range(7,10):
            return 3
        elif row['PASS_MONTH'] in range(10,13):
            return 4
        elif row['PASS_MONTH'] in range(13,16):
            return 5
        elif row['PASS_MONTH'] in range(16,19):
            return 6
        elif row['PASS_MONTH'] in range(19,22):
            return 7
        elif row['PASS_MONTH'] in range(22,25):
            return 8
        elif row['PASS_MONTH'] in range(25,28):
            return 9
        elif row['PASS_MONTH'] in range(28,31):
            return 10
        elif row['PASS_MONTH'] in range(31,34):
            return 11
        elif row['PASS_MONTH'] in range(34,37):
            return 12
        elif row['PASS_MONTH'] >= 37:
            return 13
        # 기본값으로 None 반환
        return None
    # 데이터프레임에 연령계산 컬럼 추가
    df['연령계산'] = df.apply(label_c, axis=1)
    # 'YM' 컬럼을 datetime 형식으로 변환 후 고유값 추출 및 '%Y%m' 형태로 변환
    try:
        unique_ym_values = pd.to_datetime(df['YM'].astype(str), format='%Y%m', errors='coerce').dropna().unique()
    except Exception as e:
        print(f"Error converting 'YM' to datetime: {e}")
    # 3개월 단위로 3~36개월까지의 날짜 계산 및 해당 날짜를 컬럼명으로 생성
    for i in range(3, 37, 3):  # +3개월, +6개월, ..., +36개월
        # 날짜 계산
        new_dates = [date + pd.DateOffset(months=i) for date in unique_ym_values]
        new_date_str = pd.to_datetime(new_dates).strftime('%Y-%m')[0]  # 첫 번째 날짜만 예제로 사용
        df[new_date_str] = new_date_str  # 새 컬럼에 해당 날짜를 값으로 추가 (예시를 위해 날짜값 저장)
    # 12번째부터 마지막 컬럼까지의 데이터를 모두 NaN으로 변경
    df.iloc[:, 12:df.shape[1]] = np.nan
    # import_index의 다음 시트부터 마지막 시트까지 반복
    for sheet_index in range(initial_import_index+1,sheet_count):
        # 현재 시트 데이터 불러오기
        add_df = list(input_file.values())[sheet_index]
        # 두 개의 컬럼을 문자열로 변환한 후 합쳐 새로운 파생변수 생성
        add_df['HEAT_LOT'] = add_df['HEAT_NO'].astype(str) + '_' + add_df['LOT_NO'].astype(str)
        # 'LOT_NO' 컬럼을 기준으로 'WGT'를 df에 매핑
        df = df.merge(add_df[['HEAT_LOT','WGT']], on='HEAT_LOT', how='left')
        # 'YM' 컬럼의 데이터를 문자열로 변환 후, 고유값을 'YYYY-MM' 형식으로 변환하여 첫 번째 값을 새로운 컬럼명으로 사용
        new_column_name = add_df['YM'].astype(str).unique().tolist()[0][:4] + '-' + add_df['YM'].astype(str).unique().tolist()[0][4:]
        # 'WGT_y' 값을 새로운 컬럼에 저장하고, 원본 컬럼은 삭제
        df[new_column_name] = df['WGT_y']
        df = df.drop(columns=['WGT_y'])
        # 'WGT_x' 컬럼명을 'WGT'로 변경
        df = df.rename(columns={'WGT_x':'WGT'})
        # 원본 데이터프레임을 수정하지 않고 중복된 행을 삭제할 경우 inplace=True 사용
        df.drop_duplicates(inplace=True)
    # 'YM' 컬럼의 데이터를 'YYYYMM' 형식으로 변환 및 고유값을 sheet_name으로 반환
    sheet_name = df['YM'].astype(str).str.replace('-', '').str[:6].unique()[0]
    # 그룹별 데이터프레임 저장할 딕셔너리 초기화
    output_file = {}
    # 결과를 output_file 파일에 저장
    output_file[sheet_name] = df
    # add_import_index 설정
    for add_import_index in range(1, sheet_count):
        # initial index sheet 데이터 import
        df = list(input_file.values())[add_import_index]
        # 두 개의 컬럼을 문자열로 변환한 후 합쳐 새로운 파생변수 생성
        df['HEAT_LOT'] = df['HEAT_NO'].astype(str) + '_' + df['LOT_NO'].astype(str)
        # 데이터프레임에 연령계산 컬럼 추가(기존 self함수 적용)
        df['연령계산'] = df.apply(label_c, axis=1)
        # 'YM' 컬럼을 datetime 형식으로 변환 후 고유값 추출 및 '%Y%m' 형태로 변환
        unique_ym_values = pd.to_datetime(df['YM'].astype(str), format='%Y%m').unique()
        # 3개월 단위로 3~36개월까지의 날짜 계산 및 해당 날짜를 컬럼명으로 생성
        for i in range(3, 37, 3):  # +3개월, +6개월, ..., +36개월
            # 날짜 계산
            new_dates = [date + pd.DateOffset(months=i) for date in unique_ym_values]
            # 새로운 날짜를 '%Y-%m' 형식으로 변환하여 컬럼명 생성
            new_date_str = pd.to_datetime(new_dates).strftime('%Y-%m')[0]  # 첫 번째 날짜만 예제로 사용
            df[new_date_str] = new_date_str  # 새 컬럼에 해당 날짜를 값으로 추가 (예시를 위해 날짜값 저장)
        # 12번째부터 마지막 컬럼까지의 데이터를 모두 NaN으로 변경
        df.iloc[:, 12:df.shape[1]] = np.nan
        # import_index의 다음 시트부터 마지막 시트까지 반복
        for sheet_index in range(add_import_index+1,sheet_count):
            # 현재 시트 데이터 불러오기
            add_df = list(input_file.values())[sheet_index]
            # 두 개의 컬럼을 문자열로 변환한 후 합쳐 새로운 파생변수 생성
            add_df['HEAT_LOT'] = add_df['HEAT_NO'].astype(str) + '_' + add_df['LOT_NO'].astype(str)
            # 'LOT_NO' 컬럼을 기준으로 'WGT'를 df에 매핑
            df = df.merge(add_df[['HEAT_LOT','WGT']], on='HEAT_LOT', how='left')
            # 'YM' 컬럼의 데이터를 문자열로 변환 후, 고유값을 'YYYY-MM' 형식으로 변환하여 첫 번째 값을 새로운 컬럼명으로 사용
            new_column_name = add_df['YM'].astype(str).unique().tolist()[0][:4] + '-' + add_df['YM'].astype(str).unique().tolist()[0][4:]
            # 'WGT_y' 값을 새로운 컬럼에 저장하고, 원본 컬럼은 삭제
            df[new_column_name] = df['WGT_y']
            df = df.drop(columns=['WGT_y'])
            # 'WGT_x' 컬럼명을 'WGT'로 변경
            df = df.rename(columns={'WGT_x':'WGT'})
            # 원본 데이터프레임을 수정하지 않고 중복된 행을 삭제할 경우 inplace=True 사용
            df.drop_duplicates(inplace=True)
        # 'YM' 컬럼의 데이터를 'YYYYMM' 형식으로 변환 및 고유값을 sheet_name으로 반환
        sheet_name = df['YM'].astype(str).str.replace('-', '').str[:6].unique()[0]
        # 결과를 output_file 파일에 추가
        output_file[sheet_name] = df
    # 빈 DataFrame 초기화 (모든 반복의 결과를 하나로 모으기 위해)
    final_summary = pd.DataFrame()
    # 0번 sheet부터 마지막 sheet까지 반복수행
    for import_index in range(0, sheet_count):
        # 해당 sheet index 데이터 import
        df = list(output_file.values())[import_index]
        # 'YM' 컬럼을 문자열로 변환 후 고유값 추출 및 첫 번째 값을 sheet_name으로 설정
        sheet_name = df['YM'].astype(str).unique()[0] if df['YM'].astype(str).unique().size > 0 else f'Sheet{import_index}'
        # 요약 데이터 저장용 데이터프레임 초기화
        data = {
            '년월': [sheet_name] * 13, # 해당월 입력
            '구분': ['1~3개월','4~6개월','7~9개월','10~12개월','13~15개월',
                '16~18개월','19~21개월','22~24개월','25~27개월','28~30개월',
                '31~33개월','34~36개월','36개월_초과'],
            '연령구분': [1,2,3,4,5,6,7,8,9,10,11,12,13],
            '당월기준재고': [None] * 13,  # 초기화
            '+3개월_잔여재고': [None] * 13, # 초기화
            '+6개월_잔여재고': [None] * 13, # 초기화
            '+9개월_잔여재고': [None] * 13, # 초기화
            '+12개월_잔여재고': [None] * 13, # 초기화
            '+15개월_잔여재고': [None] * 13, # 초기화
            '+18개월_잔여재고': [None] * 13, # 초기화
            '+21개월_잔여재고': [None] * 13, # 초기화
            '+24개월_잔여재고': [None] * 13, # 초기화
            '+27개월_잔여재고': [None] * 13, # 초기화
            '+30개월_잔여재고': [None] * 13, # 초기화
            '+33개월_잔여재고': [None] * 13, # 초기화
            '+36개월_잔여재고': [None] * 13, # 초기화
            '폐기가능성': [None] * 13 # 초기화
        }
        summary_data = pd.DataFrame(data)
        # 각 연령(1~13)에 대해 데이터 처리
        for age in range(1, 14):
            product_item = '소재'
            # 조건에 맞는 데이터 추출
            filtered_data = df[(df['ITEM_ACCOUNT'] == product_item) & (df['연령계산'] == age)]
            # 현재 '중량' 및 잔여재고(+3개월,+6개월,+9개월,+12개월,+15개월,+18개월,+21개월,+24개월,+27개월,+30개월,+33개월,+36개월) 데이터 합산
            this_month_sum = filtered_data['WGT'].sum() / 1000
            next_3_month_sum = filtered_data.iloc[:, 12].sum() / 1000
            next_6_month_sum = filtered_data.iloc[:, 13].sum() / 1000
            next_9_month_sum = filtered_data.iloc[:, 14].sum() / 1000
            next_12_month_sum = filtered_data.iloc[:, 15].sum() / 1000
            next_15_month_sum = filtered_data.iloc[:, 16].sum() / 1000
            next_18_month_sum = filtered_data.iloc[:, 17].sum() / 1000
            next_21_month_sum = filtered_data.iloc[:, 18].sum() / 1000
            next_24_month_sum = filtered_data.iloc[:, 19].sum() / 1000
            next_27_month_sum = filtered_data.iloc[:, 20].sum() / 1000
            next_30_month_sum = filtered_data.iloc[:, 21].sum() / 1000
            next_33_month_sum = filtered_data.iloc[:, 22].sum() / 1000
            next_36_month_sum = filtered_data.iloc[:, 23].sum() / 1000
            # 분자 선택 (age에 따라 다르게 설정)
            if age == 1:
                numerator = next_36_month_sum
            elif age == 2:
                numerator = next_33_month_sum
            elif age == 3:
                numerator = next_30_month_sum
            elif age == 4:
                numerator = next_27_month_sum
            elif age == 5:
                numerator = next_24_month_sum
            elif age == 6:
                numerator = next_21_month_sum
            elif age == 7:
                numerator = next_18_month_sum
            elif age == 8:
                numerator = next_15_month_sum
            elif age == 9:
                numerator = next_12_month_sum
            elif age == 10:
                numerator = next_9_month_sum
            elif age == 11:
                numerator = next_6_month_sum
            elif age == 12:
                numerator = next_3_month_sum
            elif age == 13:
                numerator = this_month_sum
            # 폐기 가능성 계산 (numerator 또는 this_month_sum이 0일 때 예외 처리)
            if numerator == 0 or this_month_sum == 0:
                scrap_rate = 0  # numerator 또는 this_month_sum이 0이면 scrap_rate는 0
            else:
                scrap_rate = round((numerator / this_month_sum) * 100, 1)
            # 요약 데이터에 값 할당
            summary_data.loc[age-1, '당월기준재고':] = [this_month_sum,next_3_month_sum,next_6_month_sum,
                                                next_9_month_sum,next_12_month_sum,next_15_month_sum,
                                                next_18_month_sum,next_21_month_sum,next_24_month_sum,
                                                next_27_month_sum,next_30_month_sum,next_33_month_sum,
                                                next_36_month_sum,scrap_rate]
        # 현재 sheet_name기준 +36개월 기간 계산
        unique_ym_plus_36 = [(pd.to_datetime(sheet_name, format='%Y%m') + relativedelta(months=36)).strftime('%Y%m')]
        # 문자열을 datetime 형식으로 변환
        date_last_sheet = datetime.strptime(last_sheet_name, '%Y%m') # 마지막 YYYYMM
        date_unique_ym = datetime.strptime(unique_ym_plus_36[0], '%Y%m') # 현재 sheet_name기준 +36개월 YYYYMM
        # 두 날짜의 차이 계산 (개월 수 계산)
        month_difference = (date_unique_ym.year - date_last_sheet.year) * 12 + (date_unique_ym.month - date_last_sheet.month)
        if month_difference <= 0: # month_difference가 음수이거나 0이면 skip
            pass
        elif month_difference == 3: # month_difference가 3일 때 계산
            # '연령구분'이 1,2인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            # 필요한 값 추출
            scrap_index = age_group_1.loc[:, '폐기가능성'].index[0]
            plus_33_month_stock_index = age_group_1.loc[:, '+33개월_잔여재고'].index[0]
            current_month_stock_index = age_group_1.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value = round((
                summary_data.at[plus_33_month_stock_index, '+33개월_잔여재고'] /
                summary_data.at[current_month_stock_index, '당월기준재고']
                ) * summary_data.at[age_group_2.index[0], '폐기가능성'], 1)
            # '연령구분'이 1인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index, '폐기가능성'] = value
        elif month_difference == 6: # month_difference가 6일 때 계산
            # '연령구분'이 1,2,3인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            plus_30_month_stock_index_1 = age_group_1.loc[:, '+30개월_잔여재고'].index[0]
            plus_30_month_stock_index_2 = age_group_2.loc[:, '+30개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_30_month_stock_index_1, '+30개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_3.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_30_month_stock_index_2, '+30개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_3.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
        elif month_difference == 9: # month_difference가 9일 때 계산
            # '연령구분'이 1,2,3,4인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            plus_27_month_stock_index_1 = age_group_1.loc[:, '+27개월_잔여재고'].index[0]
            plus_27_month_stock_index_2 = age_group_2.loc[:, '+27개월_잔여재고'].index[0]
            plus_27_month_stock_index_3 = age_group_3.loc[:, '+27개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_27_month_stock_index_1, '+27개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_27_month_stock_index_2, '+27개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_27_month_stock_index_3, '+27개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
        elif month_difference == 12: # month_difference가 12일 때 계산
            # '연령구분'이 1,2,3,4,5인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            plus_24_month_stock_index_1 = age_group_1.loc[:, '+24개월_잔여재고'].index[0]
            plus_24_month_stock_index_2 = age_group_2.loc[:, '+24개월_잔여재고'].index[0]
            plus_24_month_stock_index_3 = age_group_3.loc[:, '+24개월_잔여재고'].index[0]
            plus_24_month_stock_index_4 = age_group_4.loc[:, '+24개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_24_month_stock_index_1, '+24개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_5.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_24_month_stock_index_2, '+24개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_5.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_24_month_stock_index_3, '+24개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_5.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_24_month_stock_index_4, '+24개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_5.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3,4인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
        elif month_difference == 15: # month_difference가 15일 때 계산
            # '연령구분'이 1,2,3,4,5,6인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            age_group_6 = summary_data[summary_data['연령구분'] == 6]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            scrap_index_5 = age_group_5.loc[:, '폐기가능성'].index[0]
            plus_21_month_stock_index_1 = age_group_1.loc[:, '+21개월_잔여재고'].index[0]
            plus_21_month_stock_index_2 = age_group_2.loc[:, '+21개월_잔여재고'].index[0]
            plus_21_month_stock_index_3 = age_group_3.loc[:, '+21개월_잔여재고'].index[0]
            plus_21_month_stock_index_4 = age_group_4.loc[:, '+21개월_잔여재고'].index[0]
            plus_21_month_stock_index_5 = age_group_5.loc[:, '+21개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_5 = age_group_5.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_21_month_stock_index_1, '+21개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_6.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_21_month_stock_index_2, '+21개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_6.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_21_month_stock_index_3, '+21개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_6.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_21_month_stock_index_4, '+21개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_6.index[0], '폐기가능성'], 1)
            value_5 = round((
                summary_data.at[plus_21_month_stock_index_5, '+21개월_잔여재고'] /
                summary_data.at[current_month_stock_index_5, '당월기준재고']
                ) * summary_data.at[age_group_6.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3,4,5인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
            summary_data.at[scrap_index_5, '폐기가능성'] = value_5
        elif month_difference == 18: # month_difference가 18일 때 계산
            # '연령구분'이 1,2,3,4,5,6,7인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            age_group_6 = summary_data[summary_data['연령구분'] == 6]
            age_group_7 = summary_data[summary_data['연령구분'] == 7]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            scrap_index_5 = age_group_5.loc[:, '폐기가능성'].index[0]
            scrap_index_6 = age_group_6.loc[:, '폐기가능성'].index[0]
            plus_18_month_stock_index_1 = age_group_1.loc[:, '+18개월_잔여재고'].index[0]
            plus_18_month_stock_index_2 = age_group_2.loc[:, '+18개월_잔여재고'].index[0]
            plus_18_month_stock_index_3 = age_group_3.loc[:, '+18개월_잔여재고'].index[0]
            plus_18_month_stock_index_4 = age_group_4.loc[:, '+18개월_잔여재고'].index[0]
            plus_18_month_stock_index_5 = age_group_5.loc[:, '+18개월_잔여재고'].index[0]
            plus_18_month_stock_index_6 = age_group_6.loc[:, '+18개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_5 = age_group_5.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_6 = age_group_6.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_18_month_stock_index_1, '+18개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_18_month_stock_index_2, '+18개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_18_month_stock_index_3, '+18개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_18_month_stock_index_4, '+18개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            value_5 = round((
                summary_data.at[plus_18_month_stock_index_5, '+18개월_잔여재고'] /
                summary_data.at[current_month_stock_index_5, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            value_6 = round((
                summary_data.at[plus_18_month_stock_index_6, '+18개월_잔여재고'] /
                summary_data.at[current_month_stock_index_6, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3,4,5,6인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
            summary_data.at[scrap_index_5, '폐기가능성'] = value_5
            summary_data.at[scrap_index_6, '폐기가능성'] = value_6
        elif month_difference == 21: # month_difference가 21일 때 계산
            # '연령구분'이 1,2,3,4,5,6,7,8인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            age_group_6 = summary_data[summary_data['연령구분'] == 6]
            age_group_7 = summary_data[summary_data['연령구분'] == 7]
            age_group_8 = summary_data[summary_data['연령구분'] == 8]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            scrap_index_5 = age_group_5.loc[:, '폐기가능성'].index[0]
            scrap_index_6 = age_group_6.loc[:, '폐기가능성'].index[0]
            scrap_index_7 = age_group_7.loc[:, '폐기가능성'].index[0]
            plus_15_month_stock_index_1 = age_group_1.loc[:, '+15개월_잔여재고'].index[0]
            plus_15_month_stock_index_2 = age_group_2.loc[:, '+15개월_잔여재고'].index[0]
            plus_15_month_stock_index_3 = age_group_3.loc[:, '+15개월_잔여재고'].index[0]
            plus_15_month_stock_index_4 = age_group_4.loc[:, '+15개월_잔여재고'].index[0]
            plus_15_month_stock_index_5 = age_group_5.loc[:, '+15개월_잔여재고'].index[0]
            plus_15_month_stock_index_6 = age_group_6.loc[:, '+15개월_잔여재고'].index[0]
            plus_15_month_stock_index_7 = age_group_7.loc[:, '+15개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_5 = age_group_5.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_6 = age_group_6.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_7 = age_group_7.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_15_month_stock_index_1, '+15개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_15_month_stock_index_2, '+15개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_15_month_stock_index_3, '+15개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_15_month_stock_index_4, '+15개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_5 = round((
                summary_data.at[plus_15_month_stock_index_5, '+15개월_잔여재고'] /
                summary_data.at[current_month_stock_index_5, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_6 = round((
                summary_data.at[plus_15_month_stock_index_6, '+15개월_잔여재고'] /
                summary_data.at[current_month_stock_index_6, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_7 = round((
                summary_data.at[plus_15_month_stock_index_7, '+15개월_잔여재고'] /
                summary_data.at[current_month_stock_index_7, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3,4,5,6,7인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
            summary_data.at[scrap_index_5, '폐기가능성'] = value_5
            summary_data.at[scrap_index_6, '폐기가능성'] = value_6
            summary_data.at[scrap_index_7, '폐기가능성'] = value_7
        elif month_difference == 24: # month_difference가 24일 때 계산
            # '연령구분'이 1,2,3,4,5,6,7,8,9인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            age_group_6 = summary_data[summary_data['연령구분'] == 6]
            age_group_7 = summary_data[summary_data['연령구분'] == 7]
            age_group_8 = summary_data[summary_data['연령구분'] == 8]
            age_group_9 = summary_data[summary_data['연령구분'] == 9]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            scrap_index_5 = age_group_5.loc[:, '폐기가능성'].index[0]
            scrap_index_6 = age_group_6.loc[:, '폐기가능성'].index[0]
            scrap_index_7 = age_group_7.loc[:, '폐기가능성'].index[0]
            scrap_index_8 = age_group_8.loc[:, '폐기가능성'].index[0]
            plus_12_month_stock_index_1 = age_group_1.loc[:, '+12개월_잔여재고'].index[0]
            plus_12_month_stock_index_2 = age_group_2.loc[:, '+12개월_잔여재고'].index[0]
            plus_12_month_stock_index_3 = age_group_3.loc[:, '+12개월_잔여재고'].index[0]
            plus_12_month_stock_index_4 = age_group_4.loc[:, '+12개월_잔여재고'].index[0]
            plus_12_month_stock_index_5 = age_group_5.loc[:, '+12개월_잔여재고'].index[0]
            plus_12_month_stock_index_6 = age_group_6.loc[:, '+12개월_잔여재고'].index[0]
            plus_12_month_stock_index_7 = age_group_7.loc[:, '+12개월_잔여재고'].index[0]
            plus_12_month_stock_index_8 = age_group_8.loc[:, '+12개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_5 = age_group_5.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_6 = age_group_6.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_7 = age_group_7.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_8 = age_group_8.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_12_month_stock_index_1, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_9.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_12_month_stock_index_2, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_9.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_12_month_stock_index_3, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_9.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_12_month_stock_index_4, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_9.index[0], '폐기가능성'], 1)
            value_5 = round((
                summary_data.at[plus_12_month_stock_index_5, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_5, '당월기준재고']
                ) * summary_data.at[age_group_9.index[0], '폐기가능성'], 1)
            value_6 = round((
                summary_data.at[plus_12_month_stock_index_6, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_6, '당월기준재고']
                ) * summary_data.at[age_group_9.index[0], '폐기가능성'], 1)
            value_7 = round((
                summary_data.at[plus_12_month_stock_index_7, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_7, '당월기준재고']
                ) * summary_data.at[age_group_9.index[0], '폐기가능성'], 1)
            value_8 = round((
                summary_data.at[plus_12_month_stock_index_8, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_8, '당월기준재고']
                ) * summary_data.at[age_group_9.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3,4,5,6,7,8인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
            summary_data.at[scrap_index_5, '폐기가능성'] = value_5
            summary_data.at[scrap_index_6, '폐기가능성'] = value_6
            summary_data.at[scrap_index_7, '폐기가능성'] = value_7
            summary_data.at[scrap_index_8, '폐기가능성'] = value_8
        elif month_difference == 27: # month_difference가 27일 때 계산
            # '연령구분'이 1,2,3,4,5,6,7,8,9,10인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            age_group_6 = summary_data[summary_data['연령구분'] == 6]
            age_group_7 = summary_data[summary_data['연령구분'] == 7]
            age_group_8 = summary_data[summary_data['연령구분'] == 8]
            age_group_9 = summary_data[summary_data['연령구분'] == 9]
            age_group_10 = summary_data[summary_data['연령구분'] == 10]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            scrap_index_5 = age_group_5.loc[:, '폐기가능성'].index[0]
            scrap_index_6 = age_group_6.loc[:, '폐기가능성'].index[0]
            scrap_index_7 = age_group_7.loc[:, '폐기가능성'].index[0]
            scrap_index_8 = age_group_8.loc[:, '폐기가능성'].index[0]
            scrap_index_9 = age_group_9.loc[:, '폐기가능성'].index[0]
            plus_9_month_stock_index_1 = age_group_1.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_2 = age_group_2.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_3 = age_group_3.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_4 = age_group_4.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_5 = age_group_5.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_6 = age_group_6.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_7 = age_group_7.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_8 = age_group_8.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_9 = age_group_9.loc[:, '+9개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_5 = age_group_5.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_6 = age_group_6.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_7 = age_group_7.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_8 = age_group_8.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_9 = age_group_9.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_9_month_stock_index_1, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_10.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_9_month_stock_index_2, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_10.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_9_month_stock_index_3, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_10.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_9_month_stock_index_4, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_10.index[0], '폐기가능성'], 1)
            value_5 = round((
                summary_data.at[plus_9_month_stock_index_5, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_5, '당월기준재고']
                ) * summary_data.at[age_group_10.index[0], '폐기가능성'], 1)
            value_6 = round((
                summary_data.at[plus_9_month_stock_index_6, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_6, '당월기준재고']
                ) * summary_data.at[age_group_10.index[0], '폐기가능성'], 1)
            value_7 = round((
                summary_data.at[plus_9_month_stock_index_7, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_7, '당월기준재고']
                ) * summary_data.at[age_group_10.index[0], '폐기가능성'], 1)
            value_8 = round((
                summary_data.at[plus_9_month_stock_index_8, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_8, '당월기준재고']
                ) * summary_data.at[age_group_10.index[0], '폐기가능성'], 1)
            value_9 = round((
                summary_data.at[plus_9_month_stock_index_9, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_9, '당월기준재고']
                ) * summary_data.at[age_group_10.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3,4,5,6,7,8,9인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
            summary_data.at[scrap_index_5, '폐기가능성'] = value_5
            summary_data.at[scrap_index_6, '폐기가능성'] = value_6
            summary_data.at[scrap_index_7, '폐기가능성'] = value_7
            summary_data.at[scrap_index_8, '폐기가능성'] = value_8
            summary_data.at[scrap_index_9, '폐기가능성'] = value_9
        elif month_difference == 30: # month_difference가 30일 때 계산
            # '연령구분'이 1,2,3,4,5,6,7,8,9,10,11인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            age_group_6 = summary_data[summary_data['연령구분'] == 6]
            age_group_7 = summary_data[summary_data['연령구분'] == 7]
            age_group_8 = summary_data[summary_data['연령구분'] == 8]
            age_group_9 = summary_data[summary_data['연령구분'] == 9]
            age_group_10 = summary_data[summary_data['연령구분'] == 10]
            age_group_11 = summary_data[summary_data['연령구분'] == 11]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            scrap_index_5 = age_group_5.loc[:, '폐기가능성'].index[0]
            scrap_index_6 = age_group_6.loc[:, '폐기가능성'].index[0]
            scrap_index_7 = age_group_7.loc[:, '폐기가능성'].index[0]
            scrap_index_8 = age_group_8.loc[:, '폐기가능성'].index[0]
            scrap_index_9 = age_group_9.loc[:, '폐기가능성'].index[0]
            scrap_index_10 = age_group_10.loc[:, '폐기가능성'].index[0]
            plus_6_month_stock_index_1 = age_group_1.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_2 = age_group_2.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_3 = age_group_3.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_4 = age_group_4.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_5 = age_group_5.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_6 = age_group_6.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_7 = age_group_7.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_8 = age_group_8.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_9 = age_group_9.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_10 = age_group_10.loc[:, '+6개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_5 = age_group_5.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_6 = age_group_6.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_7 = age_group_7.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_8 = age_group_8.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_9 = age_group_9.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_10 = age_group_10.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_6_month_stock_index_1, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_11.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_6_month_stock_index_2, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_11.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_6_month_stock_index_3, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_11.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_6_month_stock_index_4, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_11.index[0], '폐기가능성'], 1)
            value_5 = round((
                summary_data.at[plus_6_month_stock_index_5, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_5, '당월기준재고']
                ) * summary_data.at[age_group_11.index[0], '폐기가능성'], 1)
            value_6 = round((
                summary_data.at[plus_6_month_stock_index_6, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_6, '당월기준재고']
                ) * summary_data.at[age_group_11.index[0], '폐기가능성'], 1)
            value_7 = round((
                summary_data.at[plus_6_month_stock_index_7, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_7, '당월기준재고']
                ) * summary_data.at[age_group_11.index[0], '폐기가능성'], 1)
            value_8 = round((
                summary_data.at[plus_6_month_stock_index_8, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_8, '당월기준재고']
                ) * summary_data.at[age_group_11.index[0], '폐기가능성'], 1)
            value_9 = round((
                summary_data.at[plus_6_month_stock_index_9, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_9, '당월기준재고']
                ) * summary_data.at[age_group_11.index[0], '폐기가능성'], 1)
            value_10 = round((
                summary_data.at[plus_6_month_stock_index_10, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_10, '당월기준재고']
                ) * summary_data.at[age_group_11.index[0], '폐기가능성'], 1)        
            # '연령구분'이 1,2,3,4,5,6,7,8,9,10인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
            summary_data.at[scrap_index_5, '폐기가능성'] = value_5
            summary_data.at[scrap_index_6, '폐기가능성'] = value_6
            summary_data.at[scrap_index_7, '폐기가능성'] = value_7
            summary_data.at[scrap_index_8, '폐기가능성'] = value_8
            summary_data.at[scrap_index_9, '폐기가능성'] = value_9
            summary_data.at[scrap_index_10, '폐기가능성'] = value_10
        elif month_difference == 33: # month_difference가 33일 때 계산
            # '연령구분'이 1,2,3,4,5,6,7,8,9,10,11,12인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            age_group_6 = summary_data[summary_data['연령구분'] == 6]
            age_group_7 = summary_data[summary_data['연령구분'] == 7]
            age_group_8 = summary_data[summary_data['연령구분'] == 8]
            age_group_9 = summary_data[summary_data['연령구분'] == 9]
            age_group_10 = summary_data[summary_data['연령구분'] == 10]
            age_group_11 = summary_data[summary_data['연령구분'] == 11]
            age_group_12 = summary_data[summary_data['연령구분'] == 12]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            scrap_index_5 = age_group_5.loc[:, '폐기가능성'].index[0]
            scrap_index_6 = age_group_6.loc[:, '폐기가능성'].index[0]
            scrap_index_7 = age_group_7.loc[:, '폐기가능성'].index[0]
            scrap_index_8 = age_group_8.loc[:, '폐기가능성'].index[0]
            scrap_index_9 = age_group_9.loc[:, '폐기가능성'].index[0]
            scrap_index_10 = age_group_10.loc[:, '폐기가능성'].index[0]
            scrap_index_11 = age_group_11.loc[:, '폐기가능성'].index[0]
            plus_3_month_stock_index_1 = age_group_1.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_2 = age_group_2.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_3 = age_group_3.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_4 = age_group_4.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_5 = age_group_5.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_6 = age_group_6.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_7 = age_group_7.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_8 = age_group_8.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_9 = age_group_9.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_10 = age_group_10.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_11 = age_group_11.loc[:, '+3개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_5 = age_group_5.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_6 = age_group_6.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_7 = age_group_7.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_8 = age_group_8.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_9 = age_group_9.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_10 = age_group_10.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_11 = age_group_11.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_3_month_stock_index_1, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_12.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_3_month_stock_index_2, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_12.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_3_month_stock_index_3, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_12.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_3_month_stock_index_4, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_12.index[0], '폐기가능성'], 1)
            value_5 = round((
                summary_data.at[plus_3_month_stock_index_5, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_5, '당월기준재고']
                ) * summary_data.at[age_group_12.index[0], '폐기가능성'], 1)
            value_6 = round((
                summary_data.at[plus_3_month_stock_index_6, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_6, '당월기준재고']
                ) * summary_data.at[age_group_12.index[0], '폐기가능성'], 1)
            value_7 = round((
                summary_data.at[plus_3_month_stock_index_7, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_7, '당월기준재고']
                ) * summary_data.at[age_group_12.index[0], '폐기가능성'], 1)
            value_8 = round((
                summary_data.at[plus_3_month_stock_index_8, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_8, '당월기준재고']
                ) * summary_data.at[age_group_12.index[0], '폐기가능성'], 1)
            value_9 = round((
                summary_data.at[plus_3_month_stock_index_9, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_9, '당월기준재고']
                ) * summary_data.at[age_group_12.index[0], '폐기가능성'], 1)
            value_10 = round((
                summary_data.at[plus_3_month_stock_index_10, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_10, '당월기준재고']
                ) * summary_data.at[age_group_12.index[0], '폐기가능성'], 1)
            value_11 = round((
                summary_data.at[plus_3_month_stock_index_11, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_11, '당월기준재고']
                ) * summary_data.at[age_group_12.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3,4,5,6,7,8,9,10,11인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
            summary_data.at[scrap_index_5, '폐기가능성'] = value_5
            summary_data.at[scrap_index_6, '폐기가능성'] = value_6
            summary_data.at[scrap_index_7, '폐기가능성'] = value_7
            summary_data.at[scrap_index_8, '폐기가능성'] = value_8
            summary_data.at[scrap_index_9, '폐기가능성'] = value_9
            summary_data.at[scrap_index_10, '폐기가능성'] = value_10
            summary_data.at[scrap_index_11, '폐기가능성'] = value_11
        # 폐기가능성 컬럼에 '%' 추가
        summary_data['폐기가능성'] = summary_data['폐기가능성'].astype(str) + '%'
        # 합계 행 추가
        sum_row = summary_data.iloc[:, 3:16].sum()
        new_row = pd.Series([sheet_name, '합계', '-', *sum_row, '-'], index=summary_data.columns)
        summary_data = pd.concat([summary_data, new_row.to_frame().T], ignore_index=True)
        # 모든 반복의 요약 데이터를 모으기 위해 최종 데이터프레임에 추가
        final_summary = pd.concat([final_summary, summary_data], ignore_index=True)
    # 결과를 output_file 파일에 저장
    sheet_name = '소재성_재공'
    output_file[sheet_name] = final_summary
    # 특정컬럼의 값이 '합계'인 행을 삭제
    final_summary = final_summary.loc[final_summary['구분'] != '합계'].copy()
    # '폐기가능성' 컬럼에서 '%' 및 '-' 문구를 삭제하고 float64로 변환
    final_summary.loc[:, '폐기가능성'] = (
        final_summary['폐기가능성']
        .str.replace('%', '', regex=False)
        .str.replace('-', '', regex=False)
        .replace('', '0')
        .astype('float64')
    )
    final_summary.loc[:, '연령구분'] = final_summary['연령구분'].astype('float64')
    # 처리할 컬럼 리스트
    columns_to_process = ['당월기준재고','+3개월_잔여재고','+6개월_잔여재고','+9개월_잔여재고',
                        '+12개월_잔여재고','+15개월_잔여재고','+18개월_잔여재고','+21개월_잔여재고',
                        '+24개월_잔여재고','+27개월_잔여재고','+30개월_잔여재고',
                        '+33개월_잔여재고','+36개월_잔여재고','폐기가능성']
    # 결과 저장 리스트
    average_dfs = []
    # 각 컬럼에 대해 처리
    for column in columns_to_process:
        # 0 값 제외한 데이터 필터링 및 평균 계산
        filtered_df = final_summary[final_summary[column] != 0]
        average_df = filtered_df.groupby('연령구분')[column].mean().round(2).reset_index()
        # 컬럼 이름 추가
        average_df.rename(columns={column: f'{column}'}, inplace=True)
        average_dfs.append(average_df)
    # 데이터프레임 합치기
    result_df = average_dfs[0]
    for df in average_dfs[1:]:
        result_df = pd.merge(result_df, df, on='연령구분', how='outer')
    # 폐기가능성 컬럼에 '%' 추가
    result_df['폐기가능성'] = result_df['폐기가능성'].astype(str) + '%'
    # 새로운 컬럼 'NewColumn'을 추가하여 a, b, c, d, e의 데이터를 저장
    result_df['구분'] = ['1~3개월','4~6개월','7~9개월','10~12개월','13~15개월',
                    '16~18개월','19~21개월','22~24개월','25~27개월','28~30개월',
                    '31~33개월','34~36개월','36개월_초과']
    # 컬럼 순서를 재배치할 리스트 정의 (예: ['D', 'C', 'B', 'A'])
    new_order = ['구분','연령구분','당월기준재고',
                '+3개월_잔여재고','+6개월_잔여재고','+9개월_잔여재고',
                '+12개월_잔여재고','+15개월_잔여재고','+18개월_잔여재고',
                '+21개월_잔여재고','+24개월_잔여재고','+27개월_잔여재고',
                '+30개월_잔여재고','+33개월_잔여재고','+36개월_잔여재고','폐기가능성']
    # 새로운 순서로 데이터프레임 재구성
    result_df = result_df[new_order]
    # 합계 계산
    sum_values = result_df[['당월기준재고','+3개월_잔여재고','+6개월_잔여재고',
                            '+9개월_잔여재고','+12개월_잔여재고','+15개월_잔여재고',
                            '+18개월_잔여재고','+21개월_잔여재고','+24개월_잔여재고',
                            '+27개월_잔여재고','+30개월_잔여재고','+33개월_잔여재고',
                            '+36개월_잔여재고']].sum()
    # 새로운 행 추가
    sum_row = {
        '구분': '합계',
        '연령구분': None,  # 연령구분은 Null로 설정
        '당월기준재고': sum_values['당월기준재고'],
        '+3개월_잔여재고': sum_values['+3개월_잔여재고'],
        '+6개월_잔여재고': sum_values['+6개월_잔여재고'],
        '+9개월_잔여재고': sum_values['+9개월_잔여재고'],
        '+12개월_잔여재고': sum_values['+12개월_잔여재고'],
        '+15개월_잔여재고': sum_values['+15개월_잔여재고'],
        '+18개월_잔여재고': sum_values['+18개월_잔여재고'],
        '+21개월_잔여재고': sum_values['+21개월_잔여재고'],
        '+24개월_잔여재고': sum_values['+24개월_잔여재고'],
        '+27개월_잔여재고': sum_values['+27개월_잔여재고'],
        '+30개월_잔여재고': sum_values['+30개월_잔여재고'],
        '+33개월_잔여재고': sum_values['+33개월_잔여재고'],
        '+36개월_잔여재고': sum_values['+36개월_잔여재고'],
        '폐기가능성': None  # 폐기가능성은 Null로 설정
    }
    # 합계 행 DataFrame 생성
    sum_row_df = pd.DataFrame([sum_row])
    # 모든 값이 NA인 컬럼만 뽑아서
    all_na = sum_row_df.columns[sum_row_df.isna().all()]
    # 그 컬럼들만 제거한 뒤 concat
    sum_row_df = sum_row_df.drop(columns=all_na)
    result_df = pd.concat([result_df, sum_row_df], ignore_index=True)
    # 결과를 output_file 파일에 저장
    sheet_name = '소재성_재공_summary'
    output_file[sheet_name] = result_df
    return output_file

###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

# 데이터 캐싱을 위한 데코레이터 : 제품성 재공
@st.cache_data
def load_제품성_재공():
    # 데이터프레임을 불러옵니다.
    df = pd.read_parquet('dataset.parquet')
    # 제품성 재공
    check_data = df[df['GBN_NAME'] == '제품성 재공'].copy()
    # 특정 컬럼의 null 데이터를 "일품"으로 대체
    check_data['LOT_NO'] = check_data['LOT_NO'].fillna('일품')
    # 모든 컬럼의 null 데이터를 '정보없음'으로 대체
    check_data.fillna('정보없음', inplace=True)
    # A부터 I 컬럼까지의 컬럼명을 리스트로 추출합니다.
    group_columns = check_data.columns[check_data.columns.get_loc('GBN_NAME') : check_data.columns.get_loc('PASS_MONTH') + 1].tolist()
    # 그룹핑하여 J 컬럼의 합계를 계산하고 새로운 데이터프레임 생성
    result_df = check_data.groupby(group_columns, as_index=False)['WGT'].sum()
    # 중복을 확인할 컬럼명 지정
    duplicate_columns = ['YM', 'LOT_NO', 'HEAT_NO']
    # 중복된 행을 확인하고 결과를 데이터프레임으로 추출
    duplicates = result_df[result_df.duplicated(subset=duplicate_columns, keep=False)]
    # 'YM' 컬럼을 기준으로 그룹화
    grouped = result_df.groupby('YM')
    # 그룹별 데이터프레임 저장할 딕셔너리 초기화
    grouped_dataframes = {}
    # 각 그룹을 딕셔너리에 저장
    for ym, group_df in grouped:
        grouped_dataframes[ym] = group_df.reset_index(drop=True)  # 데이터프레임 저장
    # 엑셀 파일 경로 지정
    input_file = grouped_dataframes
    # 엑셀 파일의 모든 시트 이름 및 시트 수 로드
    sheet_count = len(input_file)
    # 마지막 시트의 이름 확인
    last_sheet_name = sorted(input_file.keys())[-1]
    # initial_import_index 설정
    initial_import_index = 0
    # initial index sheet 데이터 import
    df = list(input_file.values())[0]
    # 두 개의 컬럼을 문자열로 변환한 후 합쳐 새로운 파생변수 생성
    df['HEAT_LOT'] = df['HEAT_NO'].astype(str) + '_' + df['LOT_NO'].astype(str)
    # 연령계산 컬럼 생성: 조건에 따라 라벨링
    def label_c(row):
        if pd.isnull(row['PASS_MONTH']) or row['PASS_MONTH'] in range(0,4):
            return 1
        elif row['PASS_MONTH'] in range(4,7):
            return 2
        elif row['PASS_MONTH'] in range(7,10):
            return 3
        elif row['PASS_MONTH'] in range(10,13):
            return 4
        elif row['PASS_MONTH'] >= 13:
            return 5
        # 기본값으로 None 반환
        return None
    # 데이터프레임에 연령계산 컬럼 추가
    df['연령계산'] = df.apply(label_c, axis=1)
    # 'YM' 컬럼을 datetime 형식으로 변환 후 고유값 추출 및 '%Y%m' 형태로 변환
    try:
        unique_ym_values = pd.to_datetime(df['YM'].astype(str), format='%Y%m', errors='coerce').dropna().unique()
    except Exception as e:
        print(f"Error converting 'YM' to datetime: {e}")
    # 3개월 단위로 3~36개월까지의 날짜 계산 및 해당 날짜를 컬럼명으로 생성
    for i in range(3, 37, 3):  # +3개월, +6개월, ..., +36개월
        # 날짜 계산
        new_dates = [date + pd.DateOffset(months=i) for date in unique_ym_values]
        new_date_str = pd.to_datetime(new_dates).strftime('%Y-%m')[0]  # 첫 번째 날짜만 예제로 사용
        df[new_date_str] = new_date_str  # 새 컬럼에 해당 날짜를 값으로 추가 (예시를 위해 날짜값 저장)
    # 12번째부터 마지막 컬럼까지의 데이터를 모두 NaN으로 변경
    df.iloc[:, 12:df.shape[1]] = np.nan
    # import_index의 다음 시트부터 마지막 시트까지 반복
    for sheet_index in range(initial_import_index+1,sheet_count):
        # 현재 시트 데이터 불러오기
        add_df = list(input_file.values())[sheet_index]
        # 두 개의 컬럼을 문자열로 변환한 후 합쳐 새로운 파생변수 생성
        add_df['HEAT_LOT'] = add_df['HEAT_NO'].astype(str) + '_' + add_df['LOT_NO'].astype(str)
        # 'LOT_NO' 컬럼을 기준으로 'WGT'를 df에 매핑
        df = df.merge(add_df[['HEAT_LOT','WGT']], on='HEAT_LOT', how='left')
        # 'YM' 컬럼의 데이터를 문자열로 변환 후, 고유값을 'YYYY-MM' 형식으로 변환하여 첫 번째 값을 새로운 컬럼명으로 사용
        new_column_name = add_df['YM'].astype(str).unique().tolist()[0][:4] + '-' + add_df['YM'].astype(str).unique().tolist()[0][4:]
        # 'WGT_y' 값을 새로운 컬럼에 저장하고, 원본 컬럼은 삭제
        df[new_column_name] = df['WGT_y']
        df = df.drop(columns=['WGT_y'])
        # 'WGT_x' 컬럼명을 'WGT'로 변경
        df = df.rename(columns={'WGT_x':'WGT'})
        # 원본 데이터프레임을 수정하지 않고 중복된 행을 삭제할 경우 inplace=True 사용
        df.drop_duplicates(inplace=True)

    # 'YM' 컬럼의 데이터를 'YYYYMM' 형식으로 변환 및 고유값을 sheet_name으로 반환
    sheet_name = df['YM'].astype(str).str.replace('-', '').str[:6].unique()[0]
    # 그룹별 데이터프레임 저장할 딕셔너리 초기화
    output_file = {}
    output_file[sheet_name] = df
    # add_import_index 설정
    for add_import_index in range(1, sheet_count):
        # initial index sheet 데이터 import
        df = list(input_file.values())[add_import_index]
        # 두 개의 컬럼을 문자열로 변환한 후 합쳐 새로운 파생변수 생성
        df['HEAT_LOT'] = df['HEAT_NO'].astype(str) + '_' + df['LOT_NO'].astype(str)
        # 데이터프레임에 연령계산 컬럼 추가(기존 self함수 적용)
        df['연령계산'] = df.apply(label_c, axis=1)
        # 'YM' 컬럼을 datetime 형식으로 변환 후 고유값 추출 및 '%Y%m' 형태로 변환
        unique_ym_values = pd.to_datetime(df['YM'].astype(str), format='%Y%m').unique()
        # 3개월 단위로 3~36개월까지의 날짜 계산 및 해당 날짜를 컬럼명으로 생성
        for i in range(3, 37, 3):  # +3개월, +6개월, ..., +36개월
            # 날짜 계산
            new_dates = [date + pd.DateOffset(months=i) for date in unique_ym_values]
            # 새로운 날짜를 '%Y-%m' 형식으로 변환하여 컬럼명 생성
            new_date_str = pd.to_datetime(new_dates).strftime('%Y-%m')[0]  # 첫 번째 날짜만 예제로 사용
            df[new_date_str] = new_date_str  # 새 컬럼에 해당 날짜를 값으로 추가 (예시를 위해 날짜값 저장)
        # 12번째부터 마지막 컬럼까지의 데이터를 모두 NaN으로 변경
        df.iloc[:, 12:df.shape[1]] = np.nan
        # import_index의 다음 시트부터 마지막 시트까지 반복
        for sheet_index in range(add_import_index+1,sheet_count):
            # 현재 시트 데이터 불러오기
            add_df = list(input_file.values())[sheet_index]
            # 두 개의 컬럼을 문자열로 변환한 후 합쳐 새로운 파생변수 생성
            add_df['HEAT_LOT'] = add_df['HEAT_NO'].astype(str) + '_' + add_df['LOT_NO'].astype(str)
            # 'LOT_NO' 컬럼을 기준으로 'WGT'를 df에 매핑
            df = df.merge(add_df[['HEAT_LOT','WGT']], on='HEAT_LOT', how='left')
            # 'YM' 컬럼의 데이터를 문자열로 변환 후, 고유값을 'YYYY-MM' 형식으로 변환하여 첫 번째 값을 새로운 컬럼명으로 사용
            new_column_name = add_df['YM'].astype(str).unique().tolist()[0][:4] + '-' + add_df['YM'].astype(str).unique().tolist()[0][4:]
            # 'WGT_y' 값을 새로운 컬럼에 저장하고, 원본 컬럼은 삭제
            df[new_column_name] = df['WGT_y']
            df = df.drop(columns=['WGT_y'])
            # 'WGT_x' 컬럼명을 'WGT'로 변경
            df = df.rename(columns={'WGT_x':'WGT'})
            # 원본 데이터프레임을 수정하지 않고 중복된 행을 삭제할 경우 inplace=True 사용
            df.drop_duplicates(inplace=True)
        # 'YM' 컬럼의 데이터를 'YYYYMM' 형식으로 변환 및 고유값을 sheet_name으로 반환
        sheet_name = df['YM'].astype(str).str.replace('-', '').str[:6].unique()[0]
        output_file[sheet_name] = df
    # 빈 DataFrame 초기화 (모든 반복의 결과를 하나로 모으기 위해)
    final_summary = pd.DataFrame()
    # 0번 sheet부터 마지막 sheet까지 반복수행
    for import_index in range(0, sheet_count):
        # 첫번재 sheet index 데이터 import
        df = list(output_file.values())[import_index]
        # 'YM' 컬럼을 문자열로 변환 후 고유값 추출 및 첫 번째 값을 sheet_name으로 설정
        sheet_name = df['YM'].astype(str).unique()[0] if df['YM'].astype(str).unique().size > 0 else f'Sheet{import_index}'
        # 요약 데이터 저장용 데이터프레임 초기화
        data = {
            '년월': [sheet_name] * 5, # 해당월 입력
            '구분': ['1~3개월', '4~6개월', '7~9개월', '10~12개월', '12개월_초과'],
            '연령구분': [1, 2, 3, 4, 5],
            '당월기준재고': [None] * 5,  # 초기화
            '+3개월_잔여재고': [None] * 5, # 초기화
            '+6개월_잔여재고': [None] * 5, # 초기화
            '+9개월_잔여재고': [None] * 5, # 초기화
            '+12개월_잔여재고': [None] * 5, # 초기화
            '폐기가능성': [None] * 5 # 초기화
        }
        summary_data = pd.DataFrame(data)
        # 각 연령(1~5)에 대해 데이터 처리
        for age in range(1, 6):
            product_item = '제품'
            # 조건에 맞는 데이터 추출
            filtered_data = df[(df['ITEM_ACCOUNT'] == product_item) & (df['연령계산'] == age)]
            # 현재 '중량' 및 잔여재고(+3개월, +6개월, +9개월, +12개월) 데이터 합산
            this_month_sum = filtered_data['WGT'].sum() / 1000
            next_3_month_sum = filtered_data.iloc[:, 12].sum() / 1000
            next_6_month_sum = filtered_data.iloc[:, 13].sum() / 1000
            next_9_month_sum = filtered_data.iloc[:, 14].sum() / 1000
            next_12_month_sum = filtered_data.iloc[:, 15].sum() / 1000
            # 분자 선택 (age에 따라 다르게 설정)
            if age == 1:
                numerator = next_12_month_sum
            elif age == 2:
                numerator = next_9_month_sum
            elif age == 3:
                numerator = next_6_month_sum
            elif age == 4:
                numerator = next_3_month_sum
            elif age == 5:
                numerator = this_month_sum
            # 폐기 가능성 계산 (numerator 또는 this_month_sum이 0일 때 예외 처리)
            if numerator == 0 or this_month_sum == 0:
                scrap_rate = 0  # numerator 또는 this_month_sum이 0이면 scrap_rate는 0
            else:
                scrap_rate = round((numerator / this_month_sum) * 100, 1)
            # 요약 데이터에 값 할당
            summary_data.loc[age-1, '당월기준재고':] = [this_month_sum,next_3_month_sum,next_6_month_sum,
                                                next_9_month_sum,next_12_month_sum,scrap_rate]
        # 현재 sheet_name기준 +12개월 기간 계산
        unique_ym_plus_12 = [(pd.to_datetime(sheet_name, format='%Y%m') + relativedelta(months=12)).strftime('%Y%m')]
        # 문자열을 datetime 형식으로 변환
        date_last_sheet = datetime.strptime(last_sheet_name, '%Y%m') # 마지막 YYYYMM
        date_unique_ym = datetime.strptime(unique_ym_plus_12[0], '%Y%m') # 현재 sheet_name기준 +12개월 YYYYMM
        # 두 날짜의 차이 계산 (개월 수 계산)
        month_difference = (date_unique_ym.year - date_last_sheet.year) * 12 + (date_unique_ym.month - date_last_sheet.month)
        if month_difference <= 0: # month_difference가 음수이거나 0이면 skip
            pass
        elif month_difference == 3: # month_difference가 3일 때 계산
            # '연령구분'이 1,2인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            # 필요한 값 추출
            scrap_index = age_group_1.loc[:, '폐기가능성'].index[0]
            plus_9_month_stock_index = age_group_1.loc[:, '+9개월_잔여재고'].index[0]
            current_month_stock_index = age_group_1.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value = round((
                summary_data.at[plus_9_month_stock_index, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index, '당월기준재고']
                ) * summary_data.at[age_group_2.index[0], '폐기가능성'], 1)
            # '연령구분'이 1인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index, '폐기가능성'] = value
        elif month_difference == 6: # month_difference가 6일 때 계산
            # '연령구분'이 1,2,3인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            plus_6_month_stock_index_1 = age_group_1.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_2 = age_group_2.loc[:, '+6개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_6_month_stock_index_1, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_3.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_6_month_stock_index_2, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_3.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
        elif month_difference == 9: # month_difference가 9일 때 계산
            # '연령구분'이 1,2,3,4인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            plus_3_month_stock_index_1 = age_group_1.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_2 = age_group_2.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_3 = age_group_3.loc[:, '+3개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_3_month_stock_index_1, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_3_month_stock_index_2, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_3_month_stock_index_3, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
        # 폐기가능성 컬럼에 '%' 추가
        summary_data['폐기가능성'] = summary_data['폐기가능성'].astype(str) + '%'
        # 합계 행 추가
        sum_row = summary_data.iloc[:, 3:8].sum()
        new_row = pd.Series([sheet_name, '합계', '-', *sum_row, '-'], index=summary_data.columns)
        summary_data = pd.concat([summary_data, new_row.to_frame().T], ignore_index=True)
        # 모든 반복의 요약 데이터를 모으기 위해 최종 데이터프레임에 추가
        final_summary = pd.concat([final_summary, summary_data], ignore_index=True)
    sheet_name = '제품성_재공'
    output_file[sheet_name] = final_summary
    # 특정컬럼의 값이 '합계'인 행을 삭제
    final_summary = final_summary.loc[final_summary['구분'] != '합계'].copy()
    # '폐기가능성' 컬럼에서 '%' 및 '-' 문구를 삭제하고 float64로 변환
    final_summary.loc[:, '폐기가능성'] = (
        final_summary['폐기가능성']
        .str.replace('%', '', regex=False)
        .str.replace('-', '', regex=False)
        .replace('', '0')
        .astype('float64')
    )
    final_summary.loc[:, '연령구분'] = final_summary['연령구분'].astype('float64')
    # 처리할 컬럼 리스트
    columns_to_process = ['당월기준재고', '+3개월_잔여재고', '+6개월_잔여재고', '+9개월_잔여재고', '+12개월_잔여재고', '폐기가능성']
    # 결과 저장 리스트
    average_dfs = []
    # 각 컬럼에 대해 처리
    for column in columns_to_process:
        # 0 값 제외한 데이터 필터링 및 평균 계산
        filtered_df = final_summary[final_summary[column] != 0]
        average_df = filtered_df.groupby('연령구분')[column].mean().round(2).reset_index()
        # 컬럼 이름 추가
        average_df.rename(columns={column: f'{column}'}, inplace=True)
        average_dfs.append(average_df)
    # 데이터프레임 합치기
    result_df = average_dfs[0]
    for df in average_dfs[1:]:
        result_df = pd.merge(result_df, df, on='연령구분', how='outer')
    # 폐기가능성 컬럼에 '%' 추가
    result_df['폐기가능성'] = result_df['폐기가능성'].astype(str) + '%'
    # 새로운 컬럼 'NewColumn'을 추가하여 a, b, c, d, e의 데이터를 저장
    result_df['구분'] = ['1~3개월', '4~6개월', '7~9개월', '10~12개월', '12개월_초과']
    # 컬럼 순서를 재배치할 리스트 정의 (예: ['D', 'C', 'B', 'A'])
    new_order = ['구분','연령구분','당월기준재고',
                '+3개월_잔여재고','+6개월_잔여재고','+9개월_잔여재고',
                '+12개월_잔여재고','폐기가능성']
    # 새로운 순서로 데이터프레임 재구성
    result_df = result_df[new_order]
    # 합계 계산
    sum_values = result_df[['당월기준재고', '+3개월_잔여재고', '+6개월_잔여재고', '+9개월_잔여재고', '+12개월_잔여재고']].sum()
    # 새로운 행 추가
    sum_row = {
        '구분': '합계',
        '연령구분': None,  # 연령구분은 Null로 설정
        '당월기준재고': sum_values['당월기준재고'],
        '+3개월_잔여재고': sum_values['+3개월_잔여재고'],
        '+6개월_잔여재고': sum_values['+6개월_잔여재고'],
        '+9개월_잔여재고': sum_values['+9개월_잔여재고'],
        '+12개월_잔여재고': sum_values['+12개월_잔여재고'],
        '폐기가능성': None  # 폐기가능성은 Null로 설정
    }
    # 합계 행 DataFrame 생성
    sum_row_df = pd.DataFrame([sum_row])
    # 모든 값이 NA인 컬럼만 뽑아서
    all_na = sum_row_df.columns[sum_row_df.isna().all()]
    # 그 컬럼들만 제거한 뒤 concat
    sum_row_df = sum_row_df.drop(columns=all_na)
    result_df = pd.concat([result_df, sum_row_df], ignore_index=True)
    # 결과를 output_file 파일에 저장
    sheet_name = '제품성_재공_summary'
    output_file[sheet_name] = result_df
    return output_file

###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

# 데이터 캐싱을 위한 데코레이터 : 완제품 재고
@st.cache_data
def load_완제품_재고():
    # 데이터프레임을 불러옵니다.
    df = pd.read_parquet('dataset.parquet')
    # 완제품 재고
    check_data = df[df['GBN_NAME'] == '완제품 재고'].copy()
    # 모든 컬럼의 null 데이터를 '정보없음'으로 대체
    check_data.fillna('정보없음', inplace=True)
    # A부터 I 컬럼까지의 컬럼명을 리스트로 추출합니다.
    group_columns = check_data.columns[check_data.columns.get_loc('GBN_NAME') : check_data.columns.get_loc('PASS_MONTH') + 1].tolist()
    # 그룹핑하여 J 컬럼의 합계를 계산하고 새로운 데이터프레임 생성
    result_df = check_data.groupby(group_columns, as_index=False)['WGT'].sum()
    # 중복을 확인할 컬럼명 지정
    duplicate_columns = ['YM', 'LOT_NO', 'HEAT_NO']
    # 중복된 행을 확인하고 결과를 데이터프레임으로 추출
    duplicates = result_df[result_df.duplicated(subset=duplicate_columns, keep=False)]
    # 'YM' 컬럼을 기준으로 그룹화
    grouped = result_df.groupby('YM')
    # 그룹별 데이터프레임 저장할 딕셔너리 초기화
    grouped_dataframes = {}
    # 각 그룹을 딕셔너리에 저장
    for ym, group_df in grouped:
        grouped_dataframes[ym] = group_df.reset_index(drop=True)  # 데이터프레임 저장
    # 엑셀 파일 경로 지정
    input_file = grouped_dataframes
    # 엑셀 파일의 모든 시트 이름 및 시트 수 로드
    sheet_count = len(input_file)
    # 마지막 시트의 이름 확인
    last_sheet_name = sorted(input_file.keys())[-1]
    # initial_import_index 설정
    initial_import_index = 0
    # initial index sheet 데이터 import
    df = list(input_file.values())[0]
    # 연령계산 컬럼 생성: 조건에 따라 라벨링
    def label_c(row):
        if row['SHAPE'] == 'WR':
            # SHAPE가 'WR'일 때의 조건
            if pd.isnull(row['PASS_MONTH']) or row['PASS_MONTH'] in range(0,4):
                return 1
            elif row['PASS_MONTH'] in range(4,7):
                return 2
            elif row['PASS_MONTH'] in range(7,10):
                return 3
            elif row['PASS_MONTH'] in range(10,13):
                return 4
            elif row['PASS_MONTH'] >= 13:
                return 5
        else:
            # SHAPE가 'WR'이 아닐 때의 조건
            if pd.isnull(row['PASS_MONTH']) or row['PASS_MONTH'] in range(0,4):
                return 1
            elif row['PASS_MONTH'] in range(4,7):
                return 2
            elif row['PASS_MONTH'] in range(7,10):
                return 3
            elif row['PASS_MONTH'] in range(10,13):
                return 4
            elif row['PASS_MONTH'] in range(13,16):
                return 5
            elif row['PASS_MONTH'] in range(16,19):
                return 6
            elif row['PASS_MONTH'] in range(19,22):
                return 7
            elif row['PASS_MONTH'] in range(22,25):
                return 8
            elif row['PASS_MONTH'] >= 25:
                return 9
        # 기본값으로 None 반환
        return None
    # 데이터프레임에 연령계산 컬럼 추가
    df['연령계산'] = df.apply(label_c, axis=1)
    # 'YM' 컬럼을 datetime 형식으로 변환 후 고유값 추출 및 '%Y%m' 형태로 변환
    try:
        unique_ym_values = pd.to_datetime(df['YM'].astype(str), format='%Y%m', errors='coerce').dropna().unique()
    except Exception as e:
        print(f"Error converting 'YM' to datetime: {e}")
    # 3개월 단위로 3~45개월까지의 날짜 계산 및 해당 날짜를 컬럼명으로 생성
    for i in range(3, 46, 3):  # 3개월, 6개월, ..., 45개월
        # 날짜 계산
        new_dates = [date + pd.DateOffset(months=i) for date in unique_ym_values]
        new_date_str = pd.to_datetime(new_dates).strftime('%Y-%m')[0]  # 첫 번째 날짜만 예제로 사용
        df[new_date_str] = new_date_str  # 새 컬럼에 해당 날짜를 값으로 추가 (예시를 위해 날짜값 저장)
    # 11번째부터 25번째 컬럼까지의 데이터를 모두 NaN으로 변경
    df.iloc[:, 11:26] = np.nan
    # import_index의 다음 시트부터 마지막 시트까지 반복
    for sheet_index in range(initial_import_index+1,sheet_count):
        # 현재 시트 데이터 불러오기
        add_df = list(input_file.values())[sheet_index]
        # 'LOT_NO' 컬럼을 기준으로 'WGT'를 df에 매핑
        df = df.merge(add_df[['LOT_NO','WGT']], on='LOT_NO', how='left')
        # 'YM' 컬럼의 데이터를 문자열로 변환 후, 고유값을 'YYYY-MM' 형식으로 변환하여 첫 번째 값을 새로운 컬럼명으로 사용
        new_column_name = add_df['YM'].astype(str).unique().tolist()[0][:4] + '-' + add_df['YM'].astype(str).unique().tolist()[0][4:]
        # 'WGT_y' 값을 새로운 컬럼에 저장하고, 원본 컬럼은 삭제
        df[new_column_name] = df['WGT_y']
        df = df.drop(columns=['WGT_y'])
        # 'WGT_x' 컬럼명을 'WGT'로 변경
        df = df.rename(columns={'WGT_x':'WGT'})
        # 원본 데이터프레임을 수정하지 않고 중복된 행을 삭제할 경우 inplace=True 사용
        df.drop_duplicates(inplace=True)
    # 'YM' 컬럼의 데이터를 'YYYYMM' 형식으로 변환 및 고유값을 sheet_name으로 반환
    sheet_name = df['YM'].astype(str).str.replace('-', '').str[:6].unique()[0]
    # 그룹별 데이터프레임 저장할 딕셔너리 초기화
    output_file = {}
    # 결과를 output_file 파일에 저장
    output_file[sheet_name] = df
    # add_import_index 설정
    for add_import_index in range(1, sheet_count):
        # initial index sheet 데이터 import
        df = list(input_file.values())[add_import_index]
        # 데이터프레임에 연령계산 컬럼 추가(기존 self함수 적용)
        df['연령계산'] = df.apply(label_c, axis=1)
        # 'YM' 컬럼을 datetime 형식으로 변환 후 고유값 추출 및 '%Y%m' 형태로 변환
        unique_ym_values = pd.to_datetime(df['YM'].astype(str), format='%Y%m').unique()
        # 3개월 단위로 3~45개월까지의 날짜 계산 및 해당 날짜를 컬럼명으로 생성
        for i in range(3, 46, 3):  # 3개월, 6개월, ..., 45개월
            # 날짜 계산
            new_dates = [date + pd.DateOffset(months=i) for date in unique_ym_values]
            # 새로운 날짜를 '%Y-%m' 형식으로 변환하여 컬럼명 생성
            new_date_str = pd.to_datetime(new_dates).strftime('%Y-%m')[0]  # 첫 번째 날짜만 예제로 사용
            df[new_date_str] = new_date_str  # 새 컬럼에 해당 날짜를 값으로 추가 (예시를 위해 날짜값 저장)
        # 11번째부터 25번째 컬럼까지의 데이터를 모두 NaN으로 변경
        df.iloc[:, 11:26] = np.nan
        # import_index의 다음 시트부터 마지막 시트까지 반복
        for sheet_index in range(add_import_index+1,sheet_count):
            # 현재 시트 데이터 불러오기
            add_df = list(input_file.values())[sheet_index]
            # 'LOT_NO' 컬럼을 기준으로 'WGT'를 df에 매핑
            df = df.merge(add_df[['LOT_NO','WGT']], on='LOT_NO', how='left')
            # 'YM' 컬럼의 데이터를 문자열로 변환 후, 고유값을 'YYYY-MM' 형식으로 변환하여 첫 번째 값을 새로운 컬럼명으로 사용
            new_column_name = add_df['YM'].astype(str).unique().tolist()[0][:4] + '-' + add_df['YM'].astype(str).unique().tolist()[0][4:]
            # 'WGT_y' 값을 새로운 컬럼에 저장하고, 원본 컬럼은 삭제
            df[new_column_name] = df['WGT_y']
            df = df.drop(columns=['WGT_y'])
            # 'WGT_x' 컬럼명을 'WGT'로 변경
            df = df.rename(columns={'WGT_x':'WGT'})
            # 원본 데이터프레임을 수정하지 않고 중복된 행을 삭제할 경우 inplace=True 사용
            df.drop_duplicates(inplace=True)
        # 'YM' 컬럼의 데이터를 'YYYYMM' 형식으로 변환 및 고유값을 sheet_name으로 반환
        sheet_name = df['YM'].astype(str).str.replace('-', '').str[:6].unique()[0]
        # 결과를 output_file 파일에 저장
        output_file[sheet_name] = df
    ###############################################################################################################
    ##선재#########################################################################################################
    ###############################################################################################################
    # 빈 DataFrame 초기화 (모든 반복의 결과를 하나로 모으기 위해)
    final_summary = pd.DataFrame()
    # 0번 sheet부터 마지막 sheet까지 반복수행
    for import_index in range(0, sheet_count):
        # 첫번재 sheet index 데이터 import
        df = list(output_file.values())[import_index]
        # 'YM' 컬럼을 문자열로 변환 후 고유값 추출 및 첫 번째 값을 sheet_name으로 설정
        sheet_name = df['YM'].astype(str).unique()[0] if df['YM'].astype(str).unique().size > 0 else f'Sheet{import_index}'
        # 요약 데이터 저장용 데이터프레임 초기화
        data = {
            '년월': [sheet_name] * 5, # 해당월 입력
            '구분': ['1~3개월', '4~6개월', '7~9개월', '10~12개월', '12개월_초과'],
            '연령구분': [1, 2, 3, 4, 5],
            '당월기준재고': [None] * 5,  # 초기화
            '+3개월_잔여재고': [None] * 5, # 초기화
            '+6개월_잔여재고': [None] * 5, # 초기화
            '+9개월_잔여재고': [None] * 5, # 초기화
            '+12개월_잔여재고': [None] * 5, # 초기화
            '폐기가능성': [None] * 5 # 초기화
        }
        summary_data = pd.DataFrame(data)
        # 각 연령(1~5)에 대해 데이터 처리
        for age in range(1, 6):
            product_item = 'WR'
            # 조건에 맞는 데이터 추출
            filtered_data = df[(df['SHAPE'] == product_item) & (df['연령계산'] == age)]
            # 현재 '중량' 및 잔여재고(+3개월, +6개월, +9개월, +12개월) 데이터 합산
            this_month_sum = filtered_data['WGT'].sum() / 1000
            next_3_month_sum = filtered_data.iloc[:, 11].sum() / 1000
            next_6_month_sum = filtered_data.iloc[:, 12].sum() / 1000
            next_9_month_sum = filtered_data.iloc[:, 13].sum() / 1000
            next_12_month_sum = filtered_data.iloc[:, 14].sum() / 1000
            # 분자 선택 (age에 따라 다르게 설정)
            if age == 1:
                numerator = next_12_month_sum
            elif age == 2:
                numerator = next_9_month_sum
            elif age == 3:
                numerator = next_6_month_sum
            elif age == 4:
                numerator = next_3_month_sum
            elif age == 5:
                numerator = this_month_sum
            # 폐기 가능성 계산 (numerator 또는 this_month_sum이 0일 때 예외 처리)
            if numerator == 0 or this_month_sum == 0:
                scrap_rate = 0  # numerator 또는 this_month_sum이 0이면 scrap_rate는 0
            else:
                scrap_rate = round((numerator / this_month_sum) * 100, 1)
            # 요약 데이터에 값 할당
            summary_data.loc[age-1, '당월기준재고':] = [this_month_sum,next_3_month_sum,next_6_month_sum,
                                                next_9_month_sum,next_12_month_sum,scrap_rate]
        # 현재 sheet_name기준 +12개월 기간 계산
        unique_ym_plus_12 = [(pd.to_datetime(sheet_name, format='%Y%m') + relativedelta(months=12)).strftime('%Y%m')]
        # 문자열을 datetime 형식으로 변환
        date_last_sheet = datetime.strptime(last_sheet_name, '%Y%m') # 마지막 YYYYMM
        date_unique_ym = datetime.strptime(unique_ym_plus_12[0], '%Y%m') # 현재 sheet_name기준 +12개월 YYYYMM
        # 두 날짜의 차이 계산 (개월 수 계산)
        month_difference = (date_unique_ym.year - date_last_sheet.year) * 12 + (date_unique_ym.month - date_last_sheet.month)
        if month_difference <= 0: # month_difference가 음수이거나 0이면 skip
            pass
        elif month_difference == 3: # month_difference가 3일 때 계산
            # '연령구분'이 1,2인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            # 필요한 값 추출
            scrap_index = age_group_1.loc[:, '폐기가능성'].index[0]
            plus_9_month_stock_index = age_group_1.loc[:, '+9개월_잔여재고'].index[0]
            current_month_stock_index = age_group_1.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value = round((
                summary_data.at[plus_9_month_stock_index, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index, '당월기준재고']
                ) * summary_data.at[age_group_2.index[0], '폐기가능성'], 1)
            # '연령구분'이 1인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index, '폐기가능성'] = value
        elif month_difference == 6: # month_difference가 6일 때 계산
            # '연령구분'이 1,2,3인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            plus_6_month_stock_index_1 = age_group_1.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_2 = age_group_2.loc[:, '+6개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_6_month_stock_index_1, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_3.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_6_month_stock_index_2, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_3.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
        elif month_difference == 9: # month_difference가 9일 때 계산
            # '연령구분'이 1,2,3,4인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            plus_3_month_stock_index_1 = age_group_1.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_2 = age_group_2.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_3 = age_group_3.loc[:, '+3개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_3_month_stock_index_1, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_3_month_stock_index_2, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_3_month_stock_index_3, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
        # 폐기가능성 컬럼에 '%' 추가
        summary_data['폐기가능성'] = summary_data['폐기가능성'].astype(str) + '%'
        # 합계 행 추가
        sum_row = summary_data.iloc[:, 3:8].sum()
        new_row = pd.Series([sheet_name, '합계', '-', *sum_row, '-'], index=summary_data.columns)
        summary_data = pd.concat([summary_data, new_row.to_frame().T], ignore_index=True)
        # 모든 반복의 요약 데이터를 모으기 위해 최종 데이터프레임에 추가
        final_summary = pd.concat([final_summary, summary_data], ignore_index=True)
    # 결과를 output_file 파일에 저장
    sheet_name = '제품재고_선재'
    output_file[sheet_name] = final_summary
    # 특정컬럼의 값이 '합계'인 행을 삭제
    final_summary = final_summary.loc[final_summary['구분'] != '합계'].copy()
    # '폐기가능성' 컬럼에서 '%' 및 '-' 문구를 삭제하고 float64로 변환
    final_summary.loc[:, '폐기가능성'] = (
        final_summary['폐기가능성']
        .str.replace('%', '', regex=False)
        .str.replace('-', '', regex=False)
        .replace('', '0')
        .astype('float64')
    )
    final_summary.loc[:, '연령구분'] = final_summary['연령구분'].astype('float64')
    # 처리할 컬럼 리스트
    columns_to_process = ['당월기준재고','+3개월_잔여재고','+6개월_잔여재고',
                        '+9개월_잔여재고','+12개월_잔여재고','폐기가능성']
    # 결과 저장 리스트
    average_dfs = []
    # 각 컬럼에 대해 처리
    for column in columns_to_process:
        # 0 값 제외한 데이터 필터링 및 평균 계산
        filtered_df = final_summary[final_summary[column] != 0]
        average_df = filtered_df.groupby('연령구분')[column].mean().round(2).reset_index()
        # 컬럼 이름 추가
        average_df.rename(columns={column: f'{column}'}, inplace=True)
        average_dfs.append(average_df)
    # 데이터프레임 합치기
    result_df = average_dfs[0]
    for df in average_dfs[1:]:
        result_df = pd.merge(result_df, df, on='연령구분', how='outer')
    # 폐기가능성 컬럼에 '%' 추가
    result_df['폐기가능성'] = result_df['폐기가능성'].astype(str) + '%'
    # 새로운 컬럼 'NewColumn'을 추가하여 a, b, c, d, e의 데이터를 저장
    result_df['구분'] = ['1~3개월','4~6개월','7~9개월','10~12개월','12개월_초과']
    # 컬럼 순서를 재배치할 리스트 정의 (예: ['D', 'C', 'B', 'A'])
    new_order = ['구분','연령구분','당월기준재고',
                '+3개월_잔여재고','+6개월_잔여재고','+9개월_잔여재고',
                '+12개월_잔여재고','폐기가능성']
    # 새로운 순서로 데이터프레임 재구성
    result_df = result_df[new_order]
    # 합계 계산
    sum_values = result_df[['당월기준재고','+3개월_잔여재고','+6개월_잔여재고',
                            '+9개월_잔여재고','+12개월_잔여재고']].sum()
    # 새로운 행 추가
    sum_row = {
        '구분': '합계',
        '연령구분': None,  # 연령구분은 Null로 설정
        '당월기준재고': sum_values['당월기준재고'],
        '+3개월_잔여재고': sum_values['+3개월_잔여재고'],
        '+6개월_잔여재고': sum_values['+6개월_잔여재고'],
        '+9개월_잔여재고': sum_values['+9개월_잔여재고'],
        '+12개월_잔여재고': sum_values['+12개월_잔여재고'],
        '폐기가능성': None  # 폐기가능성은 Null로 설정
    }
    # 합계 행 DataFrame 생성
    sum_row_df = pd.DataFrame([sum_row])
    # 모든 값이 NA인 컬럼만 뽑아서
    all_na = sum_row_df.columns[sum_row_df.isna().all()]
    # 그 컬럼들만 제거한 뒤 concat
    sum_row_df = sum_row_df.drop(columns=all_na)
    result_df = pd.concat([result_df, sum_row_df], ignore_index=True)
    # 결과를 output_file 파일에 저장
    sheet_name = '제품재고_선재_summary'
    output_file[sheet_name] = result_df
    ###############################################################################################################
    ##선재外#######################################################################################################
    ###############################################################################################################
    # 빈 DataFrame 초기화 (모든 반복의 결과를 하나로 모으기 위해)
    final_summary = pd.DataFrame()
    # 0번 sheet부터 마지막 sheet까지 반복수행
    for import_index in range(0, sheet_count):
        # 해당 sheet index 데이터 import
        df = list(output_file.values())[import_index]
        # 'YM' 컬럼을 문자열로 변환 후 고유값 추출 및 첫 번째 값을 sheet_name으로 설정
        sheet_name = df['YM'].astype(str).unique()[0] if df['YM'].astype(str).unique().size > 0 else f'Sheet{import_index}'
        # 요약 데이터 저장용 데이터프레임 초기화
        data = {
            '년월': [sheet_name] * 9, # 해당월 입력
            '구분': ['1~3개월','4~6개월','7~9개월','10~12개월','13~15개월','16~18개월','19~21개월','22~24개월','24개월_초과'],
            '연령구분': [1,2,3,4,5,6,7,8,9],
            '당월기준재고': [None] * 9,  # 초기화
            '+3개월_잔여재고': [None] * 9, # 초기화
            '+6개월_잔여재고': [None] * 9, # 초기화
            '+9개월_잔여재고': [None] * 9, # 초기화
            '+12개월_잔여재고': [None] * 9, # 초기화
            '+15개월_잔여재고': [None] * 9, # 초기화
            '+18개월_잔여재고': [None] * 9, # 초기화
            '+21개월_잔여재고': [None] * 9, # 초기화
            '+24개월_잔여재고': [None] * 9, # 초기화
            '폐기가능성': [None] * 9 # 초기화
        }
        summary_data = pd.DataFrame(data)
        # 각 연령(1~9)에 대해 데이터 처리
        for age in range(1, 10):
            product_item = 'WR'
            # 조건에 맞는 데이터 추출
            filtered_data = df[(df['SHAPE'] != product_item) & (df['연령계산'] == age)]
            # 현재 '중량' 및 잔여재고(+3개월,+6개월,+9개월,+12개월,+15개월,+18개월,+21개월,+24개월) 데이터 합산
            this_month_sum = filtered_data['WGT'].sum() / 1000
            next_3_month_sum = filtered_data.iloc[:, 11].sum() / 1000
            next_6_month_sum = filtered_data.iloc[:, 12].sum() / 1000
            next_9_month_sum = filtered_data.iloc[:, 13].sum() / 1000
            next_12_month_sum = filtered_data.iloc[:, 14].sum() / 1000
            next_15_month_sum = filtered_data.iloc[:, 15].sum() / 1000
            next_18_month_sum = filtered_data.iloc[:, 16].sum() / 1000
            next_21_month_sum = filtered_data.iloc[:, 17].sum() / 1000
            next_24_month_sum = filtered_data.iloc[:, 18].sum() / 1000
            # 분자 선택 (age에 따라 다르게 설정)
            if age == 1:
                numerator = next_24_month_sum
            elif age == 2:
                numerator = next_21_month_sum
            elif age == 3:
                numerator = next_18_month_sum
            elif age == 4:
                numerator = next_15_month_sum
            elif age == 5:
                numerator = next_12_month_sum
            elif age == 6:
                numerator = next_9_month_sum
            elif age == 7:
                numerator = next_6_month_sum
            elif age == 8:
                numerator = next_3_month_sum
            elif age == 9:
                numerator = this_month_sum
            # 폐기 가능성 계산 (numerator 또는 this_month_sum이 0일 때 예외 처리)
            if numerator == 0 or this_month_sum == 0:
                scrap_rate = 0  # numerator 또는 this_month_sum이 0이면 scrap_rate는 0
            else:
                scrap_rate = round((numerator / this_month_sum) * 100, 1)
            # 요약 데이터에 값 할당
            summary_data.loc[age-1, '당월기준재고':] = [this_month_sum,next_3_month_sum,next_6_month_sum,
                                                next_9_month_sum,next_12_month_sum,next_15_month_sum,
                                                next_18_month_sum,next_21_month_sum,next_24_month_sum,scrap_rate]
        # 현재 sheet_name기준 +24개월 기간 계산
        unique_ym_plus_24 = [(pd.to_datetime(sheet_name, format='%Y%m') + relativedelta(months=24)).strftime('%Y%m')]
        # 문자열을 datetime 형식으로 변환
        date_last_sheet = datetime.strptime(last_sheet_name, '%Y%m') # 마지막 YYYYMM
        date_unique_ym = datetime.strptime(unique_ym_plus_24[0], '%Y%m') # 현재 sheet_name기준 +24개월 YYYYMM
        # 두 날짜의 차이 계산 (개월 수 계산)
        month_difference = (date_unique_ym.year - date_last_sheet.year) * 12 + (date_unique_ym.month - date_last_sheet.month)
        if month_difference <= 0: # month_difference가 음수이거나 0이면 skip
            pass
        elif month_difference == 3: # month_difference가 3일 때 계산
            # '연령구분'이 1,2인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            # 필요한 값 추출
            scrap_index = age_group_1.loc[:, '폐기가능성'].index[0]
            plus_21_month_stock_index = age_group_1.loc[:, '+21개월_잔여재고'].index[0]
            current_month_stock_index = age_group_1.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value = round((
                summary_data.at[plus_21_month_stock_index, '+21개월_잔여재고'] /
                summary_data.at[current_month_stock_index, '당월기준재고']
                ) * summary_data.at[age_group_2.index[0], '폐기가능성'], 1)
            # '연령구분'이 1인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index, '폐기가능성'] = value
        elif month_difference == 6: # month_difference가 6일 때 계산
            # '연령구분'이 1,2,3인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            plus_18_month_stock_index_1 = age_group_1.loc[:, '+18개월_잔여재고'].index[0]
            plus_18_month_stock_index_2 = age_group_2.loc[:, '+18개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_18_month_stock_index_1, '+18개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_3.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_18_month_stock_index_2, '+18개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_3.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
        elif month_difference == 9: # month_difference가 9일 때 계산
            # '연령구분'이 1,2,3,4인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            plus_15_month_stock_index_1 = age_group_1.loc[:, '+15개월_잔여재고'].index[0]
            plus_15_month_stock_index_2 = age_group_2.loc[:, '+15개월_잔여재고'].index[0]
            plus_15_month_stock_index_3 = age_group_3.loc[:, '+15개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_15_month_stock_index_1, '+15개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_15_month_stock_index_2, '+15개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_15_month_stock_index_3, '+15개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_4.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
        elif month_difference == 12: # month_difference가 12일 때 계산
            # '연령구분'이 1,2,3,4,5인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            plus_12_month_stock_index_1 = age_group_1.loc[:, '+12개월_잔여재고'].index[0]
            plus_12_month_stock_index_2 = age_group_2.loc[:, '+12개월_잔여재고'].index[0]
            plus_12_month_stock_index_3 = age_group_3.loc[:, '+12개월_잔여재고'].index[0]
            plus_12_month_stock_index_4 = age_group_4.loc[:, '+12개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_12_month_stock_index_1, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_5.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_12_month_stock_index_2, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_5.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_12_month_stock_index_3, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_5.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_12_month_stock_index_4, '+12개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_5.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3,4인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
        elif month_difference == 15: # month_difference가 15일 때 계산
            # '연령구분'이 1,2,3,4,5,6인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            age_group_6 = summary_data[summary_data['연령구분'] == 6]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            scrap_index_5 = age_group_5.loc[:, '폐기가능성'].index[0]
            plus_9_month_stock_index_1 = age_group_1.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_2 = age_group_2.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_3 = age_group_3.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_4 = age_group_4.loc[:, '+9개월_잔여재고'].index[0]
            plus_9_month_stock_index_5 = age_group_5.loc[:, '+9개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_5 = age_group_5.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_9_month_stock_index_1, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_6.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_9_month_stock_index_2, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_6.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_9_month_stock_index_3, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_6.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_9_month_stock_index_4, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_6.index[0], '폐기가능성'], 1)
            value_5 = round((
                summary_data.at[plus_9_month_stock_index_5, '+9개월_잔여재고'] /
                summary_data.at[current_month_stock_index_5, '당월기준재고']
                ) * summary_data.at[age_group_6.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3,4,5인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
            summary_data.at[scrap_index_5, '폐기가능성'] = value_5
        elif month_difference == 18: # month_difference가 18일 때 계산
            # '연령구분'이 1,2,3,4,5,6,7인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            age_group_6 = summary_data[summary_data['연령구분'] == 6]
            age_group_7 = summary_data[summary_data['연령구분'] == 7]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            scrap_index_5 = age_group_5.loc[:, '폐기가능성'].index[0]
            scrap_index_6 = age_group_6.loc[:, '폐기가능성'].index[0]
            plus_6_month_stock_index_1 = age_group_1.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_2 = age_group_2.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_3 = age_group_3.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_4 = age_group_4.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_5 = age_group_5.loc[:, '+6개월_잔여재고'].index[0]
            plus_6_month_stock_index_6 = age_group_6.loc[:, '+6개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_5 = age_group_5.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_6 = age_group_6.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_6_month_stock_index_1, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_6_month_stock_index_2, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_6_month_stock_index_3, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_6_month_stock_index_4, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            value_5 = round((
                summary_data.at[plus_6_month_stock_index_5, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_5, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            value_6 = round((
                summary_data.at[plus_6_month_stock_index_6, '+6개월_잔여재고'] /
                summary_data.at[current_month_stock_index_6, '당월기준재고']
                ) * summary_data.at[age_group_7.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3,4,5,6인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
            summary_data.at[scrap_index_5, '폐기가능성'] = value_5
            summary_data.at[scrap_index_6, '폐기가능성'] = value_6
        elif month_difference == 21: # month_difference가 21일 때 계산
            # '연령구분'이 1,2,3,4,5,6,7,8인 데이터를 필터링
            age_group_1 = summary_data[summary_data['연령구분'] == 1]
            age_group_2 = summary_data[summary_data['연령구분'] == 2]
            age_group_3 = summary_data[summary_data['연령구분'] == 3]
            age_group_4 = summary_data[summary_data['연령구분'] == 4]
            age_group_5 = summary_data[summary_data['연령구분'] == 5]
            age_group_6 = summary_data[summary_data['연령구분'] == 6]
            age_group_7 = summary_data[summary_data['연령구분'] == 7]
            age_group_8 = summary_data[summary_data['연령구분'] == 8]
            # 필요한 값 추출
            scrap_index_1 = age_group_1.loc[:, '폐기가능성'].index[0]
            scrap_index_2 = age_group_2.loc[:, '폐기가능성'].index[0]
            scrap_index_3 = age_group_3.loc[:, '폐기가능성'].index[0]
            scrap_index_4 = age_group_4.loc[:, '폐기가능성'].index[0]
            scrap_index_5 = age_group_5.loc[:, '폐기가능성'].index[0]
            scrap_index_6 = age_group_6.loc[:, '폐기가능성'].index[0]
            scrap_index_7 = age_group_7.loc[:, '폐기가능성'].index[0]
            plus_3_month_stock_index_1 = age_group_1.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_2 = age_group_2.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_3 = age_group_3.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_4 = age_group_4.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_5 = age_group_5.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_6 = age_group_6.loc[:, '+3개월_잔여재고'].index[0]
            plus_3_month_stock_index_7 = age_group_7.loc[:, '+3개월_잔여재고'].index[0]
            current_month_stock_index_1 = age_group_1.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_2 = age_group_2.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_3 = age_group_3.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_4 = age_group_4.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_5 = age_group_5.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_6 = age_group_6.loc[:, '당월기준재고'].index[0]
            current_month_stock_index_7 = age_group_7.loc[:, '당월기준재고'].index[0]
            # 값 계산 및 업데이트
            value_1 = round((
                summary_data.at[plus_3_month_stock_index_1, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_1, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_2 = round((
                summary_data.at[plus_3_month_stock_index_2, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_2, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_3 = round((
                summary_data.at[plus_3_month_stock_index_3, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_3, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_4 = round((
                summary_data.at[plus_3_month_stock_index_4, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_4, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_5 = round((
                summary_data.at[plus_3_month_stock_index_5, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_5, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_6 = round((
                summary_data.at[plus_3_month_stock_index_6, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_6, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            value_7 = round((
                summary_data.at[plus_3_month_stock_index_7, '+3개월_잔여재고'] /
                summary_data.at[current_month_stock_index_7, '당월기준재고']
                ) * summary_data.at[age_group_8.index[0], '폐기가능성'], 1)
            # '연령구분'이 1,2,3,4,5,6,7인 '폐기가능성'의 index 데이터 업데이트
            summary_data.at[scrap_index_1, '폐기가능성'] = value_1
            summary_data.at[scrap_index_2, '폐기가능성'] = value_2
            summary_data.at[scrap_index_3, '폐기가능성'] = value_3
            summary_data.at[scrap_index_4, '폐기가능성'] = value_4
            summary_data.at[scrap_index_5, '폐기가능성'] = value_5
            summary_data.at[scrap_index_6, '폐기가능성'] = value_6
            summary_data.at[scrap_index_7, '폐기가능성'] = value_7
        # 폐기가능성 컬럼에 '%' 추가
        summary_data['폐기가능성'] = summary_data['폐기가능성'].astype(str) + '%'
        # 합계 행 추가
        sum_row = summary_data.iloc[:, 3:12].sum()
        new_row = pd.Series([sheet_name, '합계', '-', *sum_row, '-'], index=summary_data.columns)
        summary_data = pd.concat([summary_data, new_row.to_frame().T], ignore_index=True)
        # 모든 반복의 요약 데이터를 모으기 위해 최종 데이터프레임에 추가
        final_summary = pd.concat([final_summary, summary_data], ignore_index=True)
    # 결과를 output_file 파일에 저장
    sheet_name = '제품재고_선재外'
    output_file[sheet_name] = final_summary
    # 특정컬럼의 값이 '합계'인 행을 삭제
    final_summary = final_summary.loc[final_summary['구분'] != '합계'].copy()
    # '폐기가능성' 컬럼에서 '%' 및 '-' 문구를 삭제하고 float64로 변환
    final_summary.loc[:, '폐기가능성'] = (
        final_summary['폐기가능성']
        .str.replace('%', '', regex=False)
        .str.replace('-', '', regex=False)
        .replace('', '0')
        .astype('float64')
    )
    final_summary.loc[:, '연령구분'] = final_summary['연령구분'].astype('float64')
    # 처리할 컬럼 리스트
    columns_to_process = ['당월기준재고','+3개월_잔여재고','+6개월_잔여재고','+9개월_잔여재고',
                        '+12개월_잔여재고','+15개월_잔여재고','+18개월_잔여재고','+21개월_잔여재고',
                        '+24개월_잔여재고','폐기가능성']
    # 결과 저장 리스트
    average_dfs = []
    # 각 컬럼에 대해 처리
    for column in columns_to_process:
        # 0 값 제외한 데이터 필터링 및 평균 계산
        filtered_df = final_summary[final_summary[column] != 0]
        average_df = filtered_df.groupby('연령구분')[column].mean().round(2).reset_index()
        # 컬럼 이름 추가
        average_df.rename(columns={column: f'{column}'}, inplace=True)
        average_dfs.append(average_df)
    # 데이터프레임 합치기
    result_df = average_dfs[0]
    for df in average_dfs[1:]:
        result_df = pd.merge(result_df, df, on='연령구분', how='outer')
    # 폐기가능성 컬럼에 '%' 추가
    result_df['폐기가능성'] = result_df['폐기가능성'].astype(str) + '%'
    # 새로운 컬럼 'NewColumn'을 추가하여 a, b, c, d, e의 데이터를 저장
    result_df['구분'] = ['1~3개월','4~6개월','7~9개월','10~12개월',
                    '13~15개월','16~18개월','19~21개월','22~24개월',
                    '24개월_초과']
    # 컬럼 순서를 재배치할 리스트 정의 (예: ['D', 'C', 'B', 'A'])
    new_order = ['구분','연령구분','당월기준재고',
                '+3개월_잔여재고','+6개월_잔여재고','+9개월_잔여재고',
                '+12개월_잔여재고','+15개월_잔여재고','+18개월_잔여재고',
                '+21개월_잔여재고','+24개월_잔여재고','폐기가능성']
    # 새로운 순서로 데이터프레임 재구성
    result_df = result_df[new_order]
    # 합계 계산
    sum_values = result_df[['당월기준재고','+3개월_잔여재고','+6개월_잔여재고',
                            '+9개월_잔여재고','+12개월_잔여재고','+15개월_잔여재고',
                            '+18개월_잔여재고','+21개월_잔여재고','+24개월_잔여재고']].sum()
    # 새로운 행 추가
    sum_row = {
        '구분': '합계',
        '연령구분': None,  # 연령구분은 Null로 설정
        '당월기준재고': sum_values['당월기준재고'],
        '+3개월_잔여재고': sum_values['+3개월_잔여재고'],
        '+6개월_잔여재고': sum_values['+6개월_잔여재고'],
        '+9개월_잔여재고': sum_values['+9개월_잔여재고'],
        '+12개월_잔여재고': sum_values['+12개월_잔여재고'],
        '+15개월_잔여재고': sum_values['+15개월_잔여재고'],
        '+18개월_잔여재고': sum_values['+18개월_잔여재고'],
        '+21개월_잔여재고': sum_values['+21개월_잔여재고'],
        '+24개월_잔여재고': sum_values['+24개월_잔여재고'],
        '폐기가능성': None  # 폐기가능성은 Null로 설정
    }
    # 합계 행 DataFrame 생성
    sum_row_df = pd.DataFrame([sum_row])
    # 모든 값이 NA인 컬럼만 뽑아서
    all_na = sum_row_df.columns[sum_row_df.isna().all()]
    # 그 컬럼들만 제거한 뒤 concat
    sum_row_df = sum_row_df.drop(columns=all_na)
    result_df = pd.concat([result_df, sum_row_df], ignore_index=True)
    # 결과를 output_file 파일에 저장
    sheet_name = '제품재고_선재_外_summary'
    output_file[sheet_name] = result_df
    return output_file

###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

# 캐시된 데이터 불러오기
output_소재재공  = load_소재성_재공()
output_제품재공  = load_제품성_재공()
output_제품재고  = load_완제품_재고()

st.title("장기재공/재고 부진화 경험율 계산결과")

# 1) 카테고리 선택용 라디오
category = st.sidebar.radio(
    "☞ 분석 대상 선택",
    ("소재성 재공", "제품성 재공", "완제품 재고")
)

st.subheader(f"♣ 선택 분석 대상: {category}")

# 2) 선택한 카테고리에 맞는 딕셔너리 할당
if category == "소재성 재공":
    output_file = output_소재재공
elif category == "제품성 재공":
    output_file = output_제품재공
else:  # "제품 재고"
    output_file = output_제품재고

# 3) 그 안의 시트(년월 or summary) 선택
sheet = st.sidebar.selectbox(
    "☞ 출력할 시트를 선택하세요",
    list(output_file.keys())
)

# 선택된 시트명도 부제목으로 보여주고 싶다면 아래 한 줄을 추가
st.markdown(f"**▶ 선택시트:** {sheet}")

# 4) 데이터프레임 출력
df = output_file[sheet]
if '연령구분' in df.columns:
    df['연령구분'] = df['연령구분'].astype('string')

st.dataframe(df)