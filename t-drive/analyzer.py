import os
from collections import Counter


def analyze_folder_taxi_logs(folder_path):
    """
    지정한 폴더 내의 모든 .txt 파일을 읽어 초당 빈도 및 통합 평균치를 계산합니다.
    """
    total_counter = Counter()
    file_count = 0

    # 1. 폴더 존재 여부 확인
    if not os.path.exists(folder_path):
        print(f"오류: 폴더 '{folder_path}'를 찾을 수 없습니다.")
        return

    print(f"[{folder_path}] 내 파일 분석 시작...")
    print("-" * 50)

    # 2. 폴더 내 파일 순회
    for filename in os.listdir(folder_path):
        # .txt 확장자 파일만 필터링
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            file_count += 1

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            timestamp = parts[1].strip()
                            # 전체 카운터에 해당 초(second) 빈도 누적
                            total_counter[timestamp] += 1
                print(f"성공: {filename}")
            except Exception as e:
                print(f"실패: {filename} (오류: {e})")

    if not total_counter:
        print("분석할 데이터가 폴더 내에 없습니다.")
        return

    # 3. 결과 요약 출력
    print("-" * 50)
    print(f"총 처리 파일 수: {file_count}개")

    # 빈도순 상위 5개 출력 (데이터가 너무 많을 수 있으므로 예시로 상위 출력)
    sorted_freq = sorted(total_counter.items())
    print(f"\n[초당 수집 빈도 상세 (일부)]")
    for ts, count in sorted_freq[:10]:  # 처음 10개 초만 출력
        print(f"{ts} : {count}개")
    print("...")

    # 4. 전체 평균치 계산
    total_logs = sum(total_counter.values())
    total_seconds = len(total_counter)
    average_freq = total_logs / total_seconds

    print("-" * 50)
    print(f"전체 통합 로그 수: {total_logs}개")
    print(f"데이터가 존재하는 총 시간: {total_seconds}초")
    print(f"폴더 전체 데이터의 초당 평균 빈도: {average_freq:.2f} 개/초")

analyze_folder_taxi_logs('./taxi_log_2008_by_date_cleaned')