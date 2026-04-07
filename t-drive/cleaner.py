import pandas as pd
import os
import glob
from datetime import datetime

# ==========================================
# 1. 경로 및 설정 변수
# ==========================================
INPUT_FOLDER = "taxi_log_2008_by_date"  # 날짜별 파일이 들어있는 폴더명
OUTPUT_PARQUET = "t_drive_total_cleaned.parquet"

# Beijing Bounding Box 설정 (Beijing 관할 구역 기준)
LON_MIN, LON_MAX = 115.5, 117.6
LAT_MIN, LAT_MAX = 39.4, 41.1


def merge_and_clean_t_drive():
    try:
        # 폴더 내 모든 파일 리스트 확보 및 정렬 (날짜순 처리를 보장)
        file_list = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*")))
        if not file_list:
            print(f"오류: '{INPUT_FOLDER}' 폴더에 파일이 없습니다.")
            return

        all_dfs = []
        total_initial_count = 0

        print(f"[{datetime.now()}] 총 {len(file_list)}개 파일 통합 및 정제 시작...")

        for file_path in file_list:
            file_name = os.path.basename(file_path)
            # 2. 개별 파일 로드
            df = pd.read_csv(file_path, names=['taxi_id', 'timestamp', 'longitude', 'latitude'], header=None)
            total_initial_count += len(df)

            # 3. 데이터 타입 및 시간 변환
            df['taxi_id'] = pd.to_numeric(df['taxi_id'], errors='coerce').fillna(0).astype(int)
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # 4. 품질 정제 (중복 제거 및 좌표 필터링)
            # NaT 제거
            df = df.dropna(subset=['timestamp'])
            # 완전 중복 행 제거 (ID, 시간, 위치 모두 일치)
            df = df.drop_duplicates()
            # Bounding Box 필터링
            mask = (
                    (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX) &
                    (df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) &
                    (df['longitude'] != 0) & (df['latitude'] != 0)
            )
            df = df[mask]

            all_dfs.append(df)
            print(f"- {file_name} 처리 완료 (현재 누적 행 수: {sum(len(x) for x in all_dfs):,})")

        # 5. 전체 데이터 통합
        print(f"[{datetime.now()}] 데이터 병합 중...")
        df_total = pd.concat(all_dfs, ignore_index=True)

        # 6. 전체 기준 재정렬 및 인덱스 초기화
        # 이미 날짜별로 정렬되어 있으나, 파일 간 경계 지점의 무결성을 위해 재정렬 수행
        df_total = df_total.sort_values(by='timestamp').reset_index(drop=True)

        # 7. 최종 통계 출력
        final_count = len(df_total)
        print("\n" + "=" * 50)
        print("T-Drive 통합 정제 리포트")
        print("-" * 50)
        print(f"- 통합 전 총 행 수:     {total_initial_count:,}")
        print(f"- 최종 정제 후 행 수:   {final_count:,}")
        print(f"- 제거된 노이즈 수:     {total_initial_count - final_count:,}")
        print(f"- 데이터 시작 지점:     {df_total['timestamp'].min()}")
        print(f"- 데이터 종료 지점:     {df_total['timestamp'].max()}")
        print("=" * 50 + "\n")

        # 8. Parquet 저장 (압축률과 읽기 속도가 가장 우수함)
        print(f"[{datetime.now()}] 최종 Parquet 파일 저장 중: {OUTPUT_PARQUET}")
        df_total.to_parquet(OUTPUT_PARQUET, index=False, engine='pyarrow')
        print(f"[{datetime.now()}] 모든 통합 작업이 완료되었습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    merge_and_clean_t_drive()