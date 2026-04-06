import pandas as pd
import os

input_file = "t_drive_1_to_10357_sorted.txt"
output_folder = "taxi_log_2008_by_date"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

li = []
print(f"{input_file}로부터 데이터를 읽어옵니다.")

try:
    # 헤더가 없으므로 컬럼명을 수동으로 지정
    df = pd.read_csv(input_file, header=None, names=['id', 'datetime', 'longitude', 'latitude'])

    print("날짜 데이터 변환 중...")
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 날짜별 분할 저장
    print("날짜별로 파일 분할을 시작합니다.")

    # 시간 정보에서 날짜(연-월-일)만 추출하여 임시 컬럼 생성
    df['date_only'] = df['datetime'].dt.date

    # 날짜별로 그룹화하여 반복문 실행
    for date, group in df.groupby('date_only'):
        daily_filename = os.path.join(output_folder, f"{date}.txt")
        # 'date_only' 컬럼은 저장할 때 제외하고 저장
        # index=False, header=False로 원본과 동일한 형식을 유지
        group.drop(columns=['date_only']).to_csv(daily_filename, index=False, header=False)
        print(f"저장 완료: {daily_filename} (데이터: {len(group)}행)")

    print("-" * 30)
    print(f"모든 작업이 완료되었습니다! '{output_folder}' 폴더를 확인하세요.")

except FileNotFoundError:
    print(f"오류: '{input_file}' 파일을 찾을 수 없습니다. 파일명이 정확한지 확인해주세요.")
except Exception as e:
    print(f"오류 발생: {e}")
