import pandas as pd
import os

# 1. 설정 부분
path = './taxi_log_2008_by_id/'
start_num = 1  # 시작 파일 번호
end_num = 10357  # 끝 파일 번호 (예: 10357)
output_filename = f"t_drive_{start_num}_to_{end_num}_sorted.txt"

li = []

print(f"{start_num}번부터 {end_num}번까지 파일을 읽기 시작합니다.")

# 2. 지정한 범위만큼 반복문 실행
for i in range(start_num, end_num + 1):
    filename = os.path.join(path, f"{i}.txt")

    # 파일이 존재하는지 확인
    if os.path.exists(filename):
        try:
            # 100단위로 진행 상황 출력 (너무 자주 찍히면 느려질 수 있음)
            if i % 100 == 0 or i == start_num or i == end_num:
                print(f"현재 {i}번째 파일 읽는 중... (파일명: {i}.txt)")

            # 데이터 읽기 (헤더 없음)
            df = pd.read_csv(filename, header=None, names=['id', 'datetime', 'longitude', 'latitude'])
            li.append(df)
        except Exception as e:
            print(f"\n[오류] {i}.txt 파일 읽기 실패: {e}")
    else:
        # 파일이 번호대로 다 있지 않을 경우를 대비
        continue

# 3. 데이터 통합 및 정렬
if li:
    print("\n데이터 통합 중...")
    frame = pd.concat(li, axis=0, ignore_index=True)

    print("날짜 형식 변환 중...")
    frame['datetime'] = pd.to_datetime(frame['datetime'])

    print("시간 및 ID 순으로 정렬 중...")
    # 시간순으로 정렬하되, 시간이 같으면 ID순으로 정렬하여 객관성 유지
    sorted_frame = frame.sort_values(by=['datetime', 'id'])

    # 4. 파일 저장
    print(f"파일 저장 중: {output_filename}")
    sorted_frame.to_csv(output_filename, index=False, header=False)

    print("-" * 30)
    print(f"작업 완료!")
    print(f"총 처리된 행 수: {len(sorted_frame)}개")
else:
    print("읽어온 데이터가 없습니다. 경로와 파일 번호를 확인해주세요.")