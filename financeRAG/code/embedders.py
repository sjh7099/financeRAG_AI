
### 임베딩 모델 파인 튜닝
import os
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.datasets import NoDuplicatesDataLoader
import torch, gc

class SimpleEmbedderFinetuner:
    """
    Sentence Transformer 모델의 파인튜닝 과정을 간소화 한 클래스

    """
    def __init__(self):
      """클래스 초기화. 파인튜닝된 모델 객체를 저장할 공간 지."""
      self.model = None

    @staticmethod
    def format_query(query: str) -> str:
        """
        질의 텍스트에 "Query: " 접두사를 붙여 포맷팅
        """
        return f"Query: {query}"

    @staticmethod
    def format_text(title: str, text: str) -> str:
        """
        문서의 제목과 본문을 정해진 형식으로 결합
        "Title: [제목]\nText: [본문]" 형식으로 포맷팅
        """
        return f"Title: {title}\nText: {text}"

    def load_from_dataframes(
        self,
        queries_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        relations_df: pd.DataFrame,
        *,
        query_id_col: str = "no",
        query_text_col: str = "final_question",
        corpus_id_col: str = "doc_ids",
        corpus_text_col: str = "text",
        corpus_title_col: str | None = None, # 없으면 None (id를 title로 씀)
        rel_col: str = "rel",
        positive_threshold: float | None = 0.5,  # 0/1이면 0.5 그대로
    ):
        """
        데이터을 입력받아 파인튜닝에 필요한 표준 형식으로 변환

        Args:
            queries_df (pd.DataFrame): 질의 데이터프레임
            corpus_df (pd.DataFrame): 문서 데이터프레임
            relations_df (pd.DataFrame): 질의-문서 관계 데이터프레임
            query_id_col (str): 질의 DF의 ID 컬럼명
            query_text_col (str): 질의 DF의 텍스트 컬럼명
            corpus_id_col (str): 문서 DF의 ID 컬럼명
            corpus_text_col (str): 문서 DF의 텍스트 컬럼명
            corpus_title_col (str | None): 문서 DF의 제목 컬럼명. 없으면 ID를 제목으로 사용.
            rel_col (str): 관계 DF의 관련도 점수 컬럼명
            positive_threshold (float | None): 점수가 이 값 이상일 때 'Positive' 관계로 간주.

        Returns:
            tuple: 표준화된 (corpus, queries, qrels, all_data) 데이터프레임 튜플
        """
        # 쿼리 표준화
        q_std = pd.DataFrame({
            "_id": queries_df[query_id_col].astype(str),
            "title": queries_df[query_id_col].astype(str),  # 제목 없으니 id를 임시 title로
            "text": queries_df[query_text_col].astype(str),
        })

        # 코퍼스 표준화
        if corpus_title_col is None:
            title_series = corpus_df[corpus_id_col].astype(str)
        else:
            title_series = corpus_df[corpus_title_col].astype(str)

        c_std = pd.DataFrame({
            "_id": corpus_df[corpus_id_col].astype(str),
            "title": title_series,
            "text": corpus_df[corpus_text_col].astype(str),
        })

        # qrels 표준화
        r = relations_df[[query_id_col, corpus_id_col, rel_col]].copy()
        r.rename(columns={
            query_id_col: "query_id",
            corpus_id_col: "corpus_id",
            rel_col: "score"
        }, inplace=True)
        r["query_id"] = r["query_id"].astype(str)
        r["corpus_id"] = r["corpus_id"].astype(str)

        # 양성 관계만 필터링 (학습에는 정답 쌍만 사용)
        if positive_threshold is None:
            r_pos = r[r["score"] > 0]
        else:
            r_pos = r[r["score"] >= positive_threshold]

        # all_data 생성
        q_tmp = q_std.rename(columns={"_id": "query_id", "title": "title_queries", "text": "text_queries"})
        c_tmp = c_std.rename(columns={"_id": "corpus_id", "title": "title_corpus", "text": "text_corpus"})
        all_data_std = r_pos.merge(q_tmp, on="query_id").merge(c_tmp, on="corpus_id")

        return c_std, q_std, r_pos, all_data_std

    def split_train_val(self, all_data: pd.DataFrame, qrels: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        데이터를 학습 및 검증 세트로 분할
        """
        train_rel, val_rel = train_test_split(qrels, test_size=test_size, random_state=random_state)
        train_data = all_data[all_data['query_id'].isin(train_rel['query_id'])]
        val_data = all_data[all_data['query_id'].isin(val_rel['query_id'])]
        return train_data, val_data, train_rel, val_rel

    def create_train_samples(self, train_df: pd.DataFrame) -> list:
        """
        학습 데이터프레임을 sentence-transformers의 InputExample 형식으로 변환
        InputExample은 [질의, 관련문서] 쌍으로 구성
        """
        samples = []
        for _, row in train_df.iterrows():
            samples.append(InputExample(
                texts=[
                    self.format_query(row["text_queries"]),
                    self.format_text(row["title_corpus"], row["text_corpus"])
                ]
            ))
        return samples

    def prepare_corpus_queries(self, corpus_df: pd.DataFrame, queries_df: pd.DataFrame,
                               val_rel: pd.DataFrame, random_sample: int = 3000):
        """ 검증에 사용할 코퍼스, 쿼리 세팅 """
        # corpus 텍스트 포맷팅
        corpus_df = corpus_df.copy()
        corpus_df['text'] = corpus_df.apply(lambda r: self.format_text(r['title'], r['text']), axis=1)

        # queries 텍스트 포맷팅
        queries_df = queries_df.copy()
        queries_df['text'] = queries_df.apply(lambda r: self.format_query(self.format_text(r['title'], r['text'])), axis=1)

        # 평가 시 사용할 코퍼스 선택: 검증 세트의 정답 문서 + 전체 코퍼스에서 랜덤 샘플링
        #정답만 있는 것이 아니라 무관한 문서도 포함
        required_ids = set(map(str, val_rel["corpus_id"]))
        all_ids = corpus_df["_id"].astype(str).tolist()
        if len(all_ids) > 0:
            extra = set(random.sample(all_ids, k=min(random_sample, len(all_ids))))
            required_ids |= extra

        corpus_df = corpus_df.loc[corpus_df["_id"].astype(str).isin(required_ids)]
        corpus_dict = dict(zip(corpus_df["_id"].astype(str), corpus_df["text"]))
        queries_dict = dict(zip(queries_df["_id"].astype(str), queries_df["text"]))
        return corpus_dict, queries_dict

    def build_relevant_docs(self, val_rel: pd.DataFrame) -> dict:
        """
        qrels를 기반으로 정답 딕셔너리를 생성
        형식: {query_id: {relevant_corpus_id_1, relevant_corpus_id_2, ...}}
        """
        relevant_docs = {}
        for qid, cid in zip(val_rel["query_id"], val_rel["corpus_id"]):
            qid, cid = str(qid), str(cid)
            relevant_docs.setdefault(qid, set()).add(cid)
        return relevant_docs

    def create_ir_evaluator(self, queries: dict, corpus: dict, relevant_docs: dict,
                            batch_size: int = 32, name: str = "Evaluate") -> InformationRetrievalEvaluator:
        """
        정보 검색 성능 평가를 위한 Evaluator 객체 생성
        """
        return InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=name,
            batch_size=batch_size
        )


    def train_model(self, model: SentenceTransformer, train_samples: list,
                    evaluator: InformationRetrievalEvaluator, output_path: str,
                    epochs: int = 2, learning_rate: float = 2e-5,
                    warmup_ratio: float = 0.1, batch_size: int = 32):
        """
        실제 모델 파인튜닝을 수행

        Args:
            model (SentenceTransformer): 파인튜닝할 모델 객체
            train_samples (list): create_train_samples로 생성된 학습 샘플
            evaluator (InformationRetrievalEvaluator): create_ir_evaluator로 생성된 평가자
            output_path (str): 학습된 모델과 결과가 저장될 경로
            epochs (int): 총 학습 에폭 수
            learning_rate (float): 학습률
            warmup_ratio (float): 전체 학습 스텝 중 Warmup에 사용할 비율
            batch_size (int): 학습 배치 사이즈
        """
        self.model = model
         #MultipleNegativesRankingLoss에 맞는 데이터로더
         # 한 배치 안에 중복 문장이 없도록 필터링
        loader = NoDuplicatesDataLoader(train_samples, batch_size=batch_size)
        # 한 배치 안에서 현재 쿼리에 해당하지 않는 다른 모든 문장을
        #  자동으로 부정 예시로 판단하여 loss 측정
        loss = losses.MultipleNegativesRankingLoss(model)
        warmup_steps = int(len(loader) * epochs * warmup_ratio)

        model.fit(
            train_objectives=[(loader, loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            optimizer_params={'lr': learning_rate},
            show_progress_bar=True,
            use_amp=True,
            evaluation_steps=len(loader),  # 에폭 마지막에 평가
            save_best_model=True,
        )