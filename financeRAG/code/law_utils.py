# law_utils.py
import re
from typing import Optional, Tuple, List, Dict

LAW_NAME_RE = re.compile(r'([가-힣A-Za-z0-9·\-\s]+법)')
ARTICLE_RE  = re.compile(r'제?\s*(\d+)\s*조')

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

def prioritize_doc_ids(
    sorted_doc_items: List[Tuple[str, float]],
    corpus: Dict[str, Dict],
    q_text: str
) -> List[Tuple[str, float]]:
    """
    기존 점수순(sorted_doc_items)을 유지하면서,
    질문에서 추출된 (law_name, law_no)와 일치하는 문서를 그룹 우선 배치.
      1) (law_name AND law_no) 모두 일치
      2) law_name만 일치
      3) 나머지
    각 그룹 내부 순서는 원래 점수순 유지.
    """
    law_name, law_no = extract_law_from_question(q_text)
    if not law_name and not law_no:
        return sorted_doc_items  # 질문에 법/조가 없으면 그대로 반환

    exact_both = []
    only_name  = []
    others     = []

    # corpus[doc_id]에 'law_name', 'law_no' 필드가 들어있다고 가정
    for doc_id, score in sorted_doc_items:
        meta = corpus.get(doc_id, {})
        dn = (meta.get("law_name") or "").strip()
        dno = meta.get("law_no")
        # dno를 int로 정규화
        try:
            dno = int(dno) if dno is not None and dno != "" else None
        except Exception:
            dno = None

        if law_name and dn == law_name and (law_no is not None) and (dno == law_no):
            exact_both.append((doc_id, score))
        elif law_name and dn == law_name:
            only_name.append((doc_id, score))
        else:
            others.append((doc_id, score))

    return exact_both + only_name + others