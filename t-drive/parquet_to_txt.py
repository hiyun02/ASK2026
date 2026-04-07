import pandas as pd
from datetime import datetime

# ==========================================
# 1. 경로 및 설정 변수
# ==========================================
INPUT_PARQUET = "t_drive_total_cleaned.parquet"
OUTPUT_TXT = "t_drive_total_final.txt"


def convert_parquet_to_txt():
    try:
        print(f"[{datetime.now()}] Parquet 파일 로드 중: {INPUT_PARQUET}")

        # 2. Parquet 파일 읽기
        # Parquet은 헤더 정보를 이미 알고 있으므로 별도 옵션 없이 로드 가능
        df = pd.read_parquet(INPUT_PARQUET)

        print(f"[{datetime.now()}] TXT 변환 및 저장 시작... (총 {len(df):,} 행)")

        # 3. TXT 저장
        # - sep=',': 쉼표로 구분
        # - index=False: 행 번호 제외
        # - header=False: TXT에는 헤더를 넣지 않음 (기존 원본 포맷 유지)
        df.to_csv(OUTPUT_TXT, sep=',', index=False, header=False)

        print(f"[{datetime.now()}] 저장 완료: {OUTPUT_TXT}")

        # 간단한 검증 출력
        print("\n" + "=" * 50)
        print(f"최종 생성된 TXT 샘플 (상위 5행):")
        print("-" * 50)
        with open(OUTPUT_TXT, 'r') as f:
            for _ in range(5):
                print(f.readline().strip())
        print("=" * 50)

    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    convert_parquet_to_txt()