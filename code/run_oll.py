from sentence_transformers import SentenceTransformer
import re
import os
import pandas as pd
from tqdm import tqdm
import json
import torch
from retrieval import HybridSearcher, DenseRetrieval, CrossEncoderReranker
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ollama import Client
from post_retrieval import SelectionAgent
from law_utils import prioritize_doc_ids
import argparse
import time


# 객관식 여부 판단 함수
def is_multiple_choice(question_text):
    """
    객관식 여부를 판단: 2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2


def extract_question_and_choices(full_text):
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())

    question = " ".join(q_lines)
    return question, options


# 프롬프트 생성기
def make_prompt_auto(text, rag):
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
            f"질문: {question}\n"
            f"참고 문서: {rag}"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n"
            "콜론, 번호, 불릿으로 시작하는 것은 금지이며, 반드시 줄글으로 작성하세요.\n\n"
            f"질문: {text}\n\n"
            f"참고 문서: {rag}"
            "답변:"
        )
    return prompt


# 후처리 함수
def extract_answer_only(generated_text: str, original_question: str) -> str:
    """
    - "답변:" 이후 텍스트만 추출
    - 객관식 문제면: 정답 숫자만 추출 (실패 시 전체 텍스트 또는 기본값 반환)
    - 주관식 문제면: 전체 텍스트 그대로 반환
    - 공백 또는 빈 응답 방지: 최소 "미응답" 반환
    """
    # "답변:" 기준으로 텍스트 분리
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()

    # 공백 또는 빈 문자열일 경우 기본값 지정
    if not text:
        return "미응답"

    # 객관식 여부 판단
    is_mc = is_multiple_choice(original_question)

    if is_mc:
        # 숫자만 추출
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            return match.group(1)
        else:
            # 숫자 추출 실패 시 "0" 반환
            return "0"
    else:
        return text


# 실행
def run(arg):

    #  검색기 선언
    searcher = HybridSearcher()

    # 데이터 불러오기
    input_path = arg.rag_data  # jsonl 파일 경로

    corpus = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):  # doc_id 1부터 시작
            if not line.strip():
                continue
            obj = json.loads(line)
            corpus[f"doc{idx}"] = {
                "title": obj.get("law_name", "") + " " + obj.get("title", ""),
                "text": obj.get("text", ""),
            }

    df = pd.read_csv(arg.query_data)
    queries = {row["ID"]: row["Question"] for i, row in df.iterrows()}

    # 문서 코퍼스, 검색 쿼리 설정
    searcher.set_corpus(corpus)
    searcher.set_queries(queries)

    # Dense 모델 선언
    # 파인튜닝 모델이 있는 경우 해당 경로 설정
    embedder_name_or_path = arg.embedding_path
    embedder = SentenceTransformer(embedder_name_or_path).to("cuda")
    dense = DenseRetrieval(model=embedder, batch_size=128)  # 배치 키움

    print("Dense 모델 불러오기 완료")

    # 리랭킹 모델 선언
    reranker = CrossEncoderReranker(arg.reranker, device="cuda")

    print("리랭킹 모델 로드 완료")

    # 검색 실행
    # searcher의 메인 search 메서드를 호출하여 전체 검색 파이프라인을 실행
    results = searcher.search(
        dense=dense,  # Dense 검색을 수행할 객체 전달
        alpha=arg.serch_alpha,  # 하이브리드 점수 가중치
        dense_topk=arg.dense_topk,  # Dense 검색에서 가져올 후보 수
        sparse_topk=arg.sparse_topk,  # BM25 검색에서 가져올 후보 수
        final_topk=arg.final_topk,  # 최종적으로 반환할 결과 수
        reranker=reranker,  # 재정렬 단계에서 사용할 재랭커 객체 전달
        reranker_batch_size=32,  # 재랭커의 추론 배치 크기 설정
    )

    query_to_docs = {}

    for qid, cand in results.items():
        sorted_docs = sorted(cand.items(), key=lambda x: x[1], reverse=True)
        # 문서 본문만 추출

        # 법령 추출 기반 우선순위 재정렬 적용
        sorted_docs = prioritize_doc_ids(sorted_docs, corpus, queries[qid])

        doc_texts = [corpus[doc_id]["text"] for doc_id, _ in sorted_docs]
        query_to_docs[queries[qid]] = doc_texts

    print("검색 완료")

    # OLLAMA_MODEL = "exaone"

    OLLAMA_MODEL = arg.model_tag  # Ollama에 등록한 모델 이름
    client = Client()

    print("모델 불러오기 완료")

    # Transformers의 generation 파라미터 매핑:
    GEN_KW = {
        "num_predict": arg.num_predict,
        "temperature": arg.temperature,
        "top_p": arg.top_p,
        "num_ctx": arg.num_ctx,
        "gpu_layers": arg.gpu_layers,
    }

    # agent = SelectionAgent(OLLAMA_MODEL, client, GEN_KW)

    preds = []

    for q in tqdm(df["Question"], desc="Inference"):
        t0 = time.perf_counter()

        # 주관식 pass 하는 코드
        # 주관식이면 LLM 호출 건너뛰고 바로 0 출력
        if not is_multiple_choice(q):
            preds.append("0")
            continue

        cand = query_to_docs[q]  # 검색 문서 반환

        # LLM 문서 선택
        # selected = agent.select(q, cand, arg.final_topk) # 최종 문서 수 부여
        selected_documents = cand
        t_sel = time.perf_counter()
        print(f"[prof] select: {t_sel - t0:.3f}s", flush=True)

        prompt = make_prompt_auto(q, selected_documents)  # 프롬프트 제작

        # Ollama 응답 받기
        resp = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options=GEN_KW,
        )
        t_gen = time.perf_counter()
        print(f"[prof] generate: {t_gen - t_sel:.3f}s", flush=True)

        # 토큰/초 확인(ollama가 메트릭을 주는 경우)
        ec = resp.get("eval_count")
        ed = resp.get("eval_duration")
        if ec and ed:
            tokps = ec / max(ed, 1e-9)
            print(f"[prof] {ec} tokens in {ed:.3f}s → {tokps:.1f} tok/s", flush=True)

        text = resp["response"]  # 생성된 전체 텍스트
        pred_answer = extract_answer_only(text, original_question=q)
        preds.append(pred_answer)

    sample_submission = pd.read_csv("/root/workspace/data/sample_submission.csv")
    sample_submission["Answer"] = preds
    sample_submission.to_csv(arg.output, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama 기반 파이프라인 실행")

    # RAG 검색 관련 입력
    parser.add_argument(
        "--rag_data", type=str, required=True, help="jsonl 형식의 RAG 문서 파일 경로"
    )
    parser.add_argument(
        "--query_data", type=str, required=True, help="질문 CSV 파일 경로"
    )
    parser.add_argument(
        "--output", type=str, default="output.csv", help="예측 결과 저장 경로"
    )

    # 모델 경로들
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="intfloat/multilingual-e5-base",
        help="SentenceTransformer 임베딩 모델 경로",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default="BAAI/bge-reranker-base",
        help="CrossEncoder 재랭커 모델 경로",
    )

    # 검색 하이퍼파라미터
    parser.add_argument(
        "--serch_alpha", type=float, default=0.7, help="dense 가중치 (alpha)"
    )
    parser.add_argument("--dense_topk", type=int, default=10, help="dense top-k")
    parser.add_argument("--sparse_topk", type=int, default=10, help="sparse top-k")
    parser.add_argument(
        "--final_topk", type=int, default=5, help="최종 top-k (재랭커 이후)"
    )

    # Ollama 설정
    parser.add_argument(
        "--model_tag",
        type=str,
        required=True,
        help="Ollama에서 사용할 모델 태그 이름 (예: bllossom-8b)",
    )
    parser.add_argument("--num_predict", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_ctx", type=int, default=3054)
    parser.add_argument("--gpu_layers", type=int, default=-1)

    args = parser.parse_args()
    run(args)
