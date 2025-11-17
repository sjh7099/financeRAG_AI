import pandas as pd

# 데이터 불러오기
input_path = "/root/workspace/data/laws_by_article_portion.jsonl"  # jsonl 파일 경로

corpus = {}
with open(input_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):  # doc_id 1부터 시작
        if not line.strip():
            continue
        obj = json.loads(line)
        # ... run() 내부 corpus 구성 부분 ...
        corpus[f"doc{idx}"] = {
            "title": obj.get("law_name", "") + ' ' + obj.get("title", ""),
            "text": obj.get("text", ""),
            "law_name": obj.get("law_name", ""),
            "law_no": obj.get("law_no", None),   # ← 추가
        }

def extract_law_from_question(q: str) -> Tuple[Optional[str], Optional[int]]:
    """
    질문에서 '~~법 ~~조' 패턴을 찾아 (law_name, law_no) 반환.
    law_no가 없으면 (law_name, None).
    ex) "개인정보보호법 제28조에 따르면..." -> ("개인정보보호법", 28)
    """
    if not q:
        return None, None
    q = q.strip()

    law_name = None
    m_name = LAW_NAME_RE.search(q)
    if m_name:
        law_name = m_name.group(1).strip()

    law_no = None
    m_art = ARTICLE_RE.search(q)
    if m_art:
        try:
            law_no = int(m_art.group(1))
        except Exception:
            law_no = None

    return law_name, law_no



print(f"불러온 문서 수: {len(corpus)}")


# ===== 테스트용 질문 =====
questions = [
    "개인정보보호법 제28조에 따르면 무엇을 해야 합니까?",
    "정보통신망법 제48조에 따르면 어떤 의무가 있나요?",
    "신용정보법 제32조는 무엇을 규정하나요?",
    "보험업법 제100조는 어떤 내용을 담고 있나요?",
    "전자서명법 제3조에 따른 요건은 무엇입니까?",
    "전자금융거래법 제6조는 무엇을 규정하나요?",
]

# ===== 테스트 실행 =====
for q in questions:
    law_name, law_no = extract_law_from_question(q)
    print(f"\nQ: {q}")
    print(f" → 추출 결과: (법령명: {law_name}, 조문번호: {law_no})")

    # corpus에서 "doc.law_name이 추출된 law_name을 포함" + law_no 일치하는 경우
    matched = [
        (doc_id, doc)
        for doc_id, doc in corpus.items()
        if (law_name and law_name in doc["law_name"])
        and (law_no is None or doc["law_no"] == law_no)
    ]

    if matched:
        print(f" → corpus 매칭된 문서 수: {len(matched)}")
        for doc_id, doc in matched[:5]:  # 최대 5개만 보여줌
            print(f"   - {doc_id}: {doc['title']}")
    else:
        print(" → corpus 매칭 없음")