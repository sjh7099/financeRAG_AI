# 학습용 데이터 제작

# 쿼리 / 코퍼스(정답문서) / 쿼리:정답코퍼스(청킹하면 모두 1로) / all data 반환

from datasets import load_dataset
import pandas as pd
import ast
import re
from embedders import SimpleEmbedderFinetuner
from sentence_transformers import SentenceTransformer


# 질문 문자열 제작
def create_final_question(row):
    """질문과 보기를 합쳐 하나의 완성된 질문 문자열 제작"""
    question_text = row['question']
    options = [
        f"A. {row['A']}",
        f"B. {row['B']}",
        f"C. {row['C']}",
        f"D. {row['D']}",
        f"E. {row['E']}"
    ]
    # 질문과 각 보기를 개행 문자로 병합
    return f"{question_text}\n" + "\n".join(options)


# 질문-정답 문서 사이 관계 생성
def extract_doc_ids(rag_str):
    """rag_data 컬럼의 문자열에서 doc_id 리스트 추출"""
    try:

        # 딕셔너리 경로를 따라 'results' 리스트에 접근
        results = rag_str['question_only']['retrieved_docs']['results']
        # 리스트 컴프리헨션을 사용해 doc_id만 추출
        return [item['doc_id'] for item in results if item['doc_id'].startswith('docid')]
    except (ValueError, KeyError, SyntaxError):
        # 데이터에 오류가 있을 경우 빈 리스트를 반환하여 에러 방지
        return []


## -------------
##     실행
## -------------

# 법령 코퍼스 할당
# 판례도 있으나 용량 문제로 생략
statutes = load_dataset(
    "lbox/kbl-rag",
    data_files={"train": "corpus/statutes.jsonl"},
    split="train"
)

# 문제와 rag 정답문서 할당
bar_civil = load_dataset(
    "json",
    data_files=[
      "https://huggingface.co/datasets/lbox/kbl-rag/resolve/main/bar_exam/civil/rag_s/*.json",
      "https://huggingface.co/datasets/lbox/kbl-rag/resolve/main/bar_exam/civil/rag_ps/*.json",
      "https://huggingface.co/datasets/lbox/kbl-rag/resolve/main/bar_exam/civil/rag_p/*.json"
    ],
    split="train",
)

# df 변환
statutes_df = statutes.to_pandas()
bar_civil_df = bar_civil.to_pandas()


# final_question 컬럼 추가 : 최종 질문 확보
bar_civil_df['final_question'] = bar_civil_df.apply(create_final_question, axis=1)


# 각 질문에 해당하는 doc_id 리스트를 추출하여 'doc_ids' 컬럼으로 추가
bar_civil_df['doc_ids'] = bar_civil_df['rag_data'].apply(extract_doc_ids)

# 'doc_ids' 리스트의 각 아이템을 별도의 행으로 분리
df_exploded = bar_civil_df.explode('doc_ids')
df_exploded = df_exploded.dropna().reset_index(drop=True)

# 필요한 정보만 추출
df_s = df_exploded[['no', 'final_question', 'doc_ids']]


# doc_id 기준으로 병합 >> 이 df는 연관성 있는 조합만 존재
question_document_df = pd.merge(
    df_s,
    statutes_df,
    left_on='doc_ids',
    right_on='id',
    how='inner'
)

question_document_df = question_document_df[['no', 'doc_ids']] # 필요한 정보만 정리
question_document_df['rel'] = 1 # 관계성 부여

# 모든 조합 생성
all_questions = pd.DataFrame({'no': df_s['no'].unique()})
all_docs = pd.DataFrame({'doc_ids': statutes_df['id'].unique()})

# 'cross join'으로 모든 문제-문서 조합 생성
all_combinations = pd.merge(all_questions, all_docs, how='cross')

# all_combinations, question_document_df 조인
final_df = pd.merge(
    all_combinations,
    question_document_df,
    on=['no', 'doc_ids'],
    how='left'
)

# 관계성이 없는 부분은 0으로 지정
final_df['rel'].fillna(0, inplace=True)
final_df['rel'] = final_df['rel'].astype(int)


## 정리
queries = df_exploded[['no', 'final_question']].drop_duplicates()
corpus = statutes_df.rename(columns={'id': 'doc_ids'})
relations = final_df.copy()

print(len(queries), len(corpus), len(relations))


import os

# Wandb 비활성화 : login 필요
os.environ["WANDB_DISABLED"] = "False"

# SimpleEmbedderFinetuner 생
ft = SimpleEmbedderFinetuner()

# df를 표준 포맷으로 변
corpus_std, queries_std, qrels_std, all_data_std = ft.load_from_dataframes(
    queries_df=queries,          # 원본 쿼리 데이터프레임
    corpus_df=corpus,            # 원본 문서(코퍼스) 데이터프레임
    relations_df=relations,      # 쿼리-문서 관계 데이터프레임
    query_id_col="no",           # 쿼리 DF에서 ID 역할을 하는 컬럼명
    query_text_col="final_question", # 쿼리 DF에서 텍스트 역할을 하는 컬럼명
    corpus_id_col="doc_ids",     # 문서 DF에서 ID 역할을 하는 컬럼명
    corpus_text_col="contents",  # 문서 DF에서 본문 텍스트 역할을 하는 컬럼명
    corpus_title_col=None,       # 문서 DF에 제목 컬럼이 없으므로 None으로 설정
    rel_col="rel",               # 관계 DF에서 관련도 점수 역할을 하는 컬럼명
    positive_threshold=None)     # 관련도 점수가 0, 1

# train/val 분할
train_data, val_data, train_rel, val_rel = ft.split_train_val(all_data_std, qrels_std, test_size=0.2, random_state=42)

# 학습 샘플 생성
train_samples = ft.create_train_samples(train_data)

# IR 평가 준비

# 평가에 사용할 코퍼스와 쿼리 생
corpus_dict, queries_dict = ft.prepare_corpus_queries(corpus_std, queries_std, val_rel, random_sample=30)

# 검증 쿼리에 대한 정답 문서 목록을 생성
relevant_docs = ft.build_relevant_docs(val_rel)

# 검증 객체 생성
evaluator = ft.create_ir_evaluator(queries_dict, corpus_dict, relevant_docs, batch_size=8)

# 파인튜닝의 기반이 될 사전 학습된 임베딩 모델 불러오
model = SentenceTransformer("intfloat/e5-base-v2")


# 원본 모델의 성능을 평가하여 베이스라인 점수를 확인
print("Evaluating before fine-tuning:")
main_score = evaluator(model)
print(f"main_score : {main_score}")

# 모델 학습
ft.train_model(
    model=model,                       # 파인튜닝할 모델 객체
    train_samples=train_samples,       # 생성한 학습 샘플
    evaluator=evaluator,               # 생성한 평가자
    output_path="./out-law-embedder",  # 학습 결과(모델)가 저장될 경로
    epochs=50,                          # 에폭 수
    learning_rate=2e-5,                # 학습률 설정
    batch_size=16                       # 배치 사이즈 설정
)

