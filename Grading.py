import pandas as pd

# 파일 읽기
key_df = pd.read_excel("답지.xlsx")       
sub_df = pd.read_csv("제출물.csv", encoding="utf-8-sig")  

# 기본 정리
key_df = key_df[['ID', 'Answer']].drop_duplicates(subset=['ID'])
sub_df = sub_df[['ID', 'Answer']].drop_duplicates(subset=['ID'])

# 전체 문제 수(=ID 고유값 수)
total = key_df['ID'].nunique()

# ID 기준으로 병합(답지 기준)
df = key_df.merge(sub_df, on='ID', how='left', suffixes=('_key', '_sub'))

def parse_ans(s):
    """
    숫자 하나('3') 또는 쉼표로 구분된 숫자들('1,3,5')이면 파싱해 정규화.
    그 외(문장, 혼합, 공란 등)는 ('str', None) 반환.
    반환 예:
      ('single', {3})
      ('multi', {1,3,5})
      ('str', None)
    """
    if pd.isna(s):
        return ('str', None)
    s = str(s).strip()
    if s == '':
        return ('str', None)

    # 쉼표 기준 분해 후 모두 정수인지 확인
    parts = [p.strip() for p in s.split(',')]
    if any(p == '' for p in parts):
        return ('str', None)

    ints = []
    for p in parts:
        if not p.isdigit():  # 음수나 공백/문자 포함 시 STR 처리
            return ('str', None)
        ints.append(int(p))

    vals = set(ints)
    if len(vals) == 1:
        return ('single', vals)
    else:
        return ('multi', vals)

correct = 0
str_count = 0  # 제출물에서 STR 형태인 건수

for _, row in df.iterrows():
    t_key, v_key = parse_ans(row['Answer_key'])
    t_sub, v_sub = parse_ans(row['Answer_sub'])

    # 제출물이 STR 형식이면 카운트
    if t_sub == 'str':
        str_count += 1

    # 채점 규칙
    # 1) STR은 무조건 오답
    if t_key == 'str' or t_sub == 'str':
        continue  # 오답으로 둠

    # 2) 숫자/집합 비교: 집합 완전일치만 정답
    if v_key == v_sub:
        correct += 1

print(f"{correct}/{total}")
print(f"STR 형식 제출 수: {str_count}")
