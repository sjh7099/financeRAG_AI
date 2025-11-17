import os
import re
import math
import logging
from typing import Dict, List, Tuple, Any, Union
import abc
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Literal, Optional, Tuple, Union
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import heapq

# 전역 함수
@torch.no_grad()
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    두 텐서 간의 코사인 유사도 계산

    Args:
        a (torch.Tensor): 첫 번째 텐서 (N, D) - N개의 D차원 벡터
        b (torch.Tensor): 두 번째 텐서 (M, D) - M개의 D차원 벡터

    Returns:
        torch.Tensor: 코사인 유사도 행렬 (N, M)
    """
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    return torch.mm(
        torch.nn.functional.normalize(a, p=2, dim=1),
        torch.nn.functional.normalize(b, p=2, dim=1).transpose(0, 1),
    )


@torch.no_grad()
def dot_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    두 텐서 간의 내적 점수를 계산

    Args:
        a (torch.Tensor): 첫 번째 텐서 (N, D)
        b (torch.Tensor): 두 번째 텐서 (M, D)

    Returns:
        torch.Tensor: 내적 점수 행렬 (N, M)
    """

    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    return torch.mm(a, b.transpose(0, 1))


def _ensure_tensor(x: Any) -> torch.Tensor:
    """ 입력값를 torch.Tensor로 변환 """

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    return x


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
def simple_tokenize(text: str) -> List[str]:
    """주어진 텍스트를 단순한 토큰 리스트로 분리"""
    return _WORD_RE.findall((text or "").lower())

# Retrieval 클래스
# 추상 클래스로, 상속하여 구현

class Retrieval(abc.ABC):
    @abc.abstractmethod
    def retrieve(
            self,
            corpus: Dict[str, Dict[Literal["title", "text"], str]],
            queries: Dict[str, str],
            top_k: Optional[int] = None,
            score_function: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:

        raise NotImplementedError


class BM25Retriever(Retrieval):
    """
    BM25 알고리즘을 사용하여 문서를 검색하는 클래스
    """
    def __init__(self, corpus: Dict[str, Dict[str, str]]):
        """
        BM25 모델을 초기화하고, 주어진 코퍼스로 학습

        Args:
            corpus (Dict): {'doc_id': {'title': '...', 'text': '...'}, ...} 형태의 문서 딕셔너리
        """
        # 문서 ID들을 리스트로 저장
        self.doc_ids = list(corpus.keys())

        # 각 문서의 제목과 본문을 합친 후 토큰화
        tokenized = [
            simple_tokenize(corpus[i].get("title", "") + " " + corpus[i].get("text", ""))
            for i in self.doc_ids
        ]

        # 토큰화된 문서들로 BM25 모델을 초기화
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, queries: Dict[str, str], top_k: int = 1000, **kwargs) -> Dict[str, Dict[str, float]]:
        """
        주어진 쿼리들에 대해 관련된 문서를 찾아 순위를 매겨 반환

        Args:
            queries (Dict): {'query_id': 'query_text', ...} 형태의 쿼리 딕셔너리
            top_k (int): 각 쿼리당 반환할 최대 문서 개수

        Returns:
            Dict: {'query_id': {'doc_id': score, ...}, ...} 형태의 검색 결과
        """
        results: Dict[str, Dict[str, float]] = {}
        for qid, q in queries.items():
            # 토큰화
            toks = simple_tokenize(q)
            # 모든 문서에 대한 BM25 점수를 계산
            scores = self.bm25.get_scores(toks)

            # 점수가 높은 순으로 정렬하여 상위 top_k개 바놘
            idx = np.argsort(-scores)[:top_k]

            # 결과로 상위 문서 ID와 점수를 저장
            results[qid] = {self.doc_ids[i]: float(scores[i]) for i in idx}

        return results


class DenseRetrieval(Retrieval):
    """
    Sentence Transformer와 같은 임베딩 모델을 사용하여
    Dense Vector 기반의 검색을 수행하는 클래스
    대용량 코퍼스를 처리하기 위해 문서를 청크 단위로 나누어 계산
    """

    def __init__(
        self,
        model, # 임베딩을 생성할 모델 (예: SentenceTransformer)
        batch_size: int = 64, # 인코딩 시 사용할 배치 크기
        score_functions=None, # 유사도 계산 함수 딕셔너리
        corpus_chunk_size: int = 50000, # 코퍼스를 나누어 처리할 청크 크기
    ):

        self.model = model
        self.batch_size = batch_size
        #  코사인 유사도와 내적을 설
        if score_functions is None:
            score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_functions = score_functions
        self.corpus_chunk_size = corpus_chunk_size
        self.results: Dict = {}

    def retrieve(
        self,
        corpus: Dict[str, Dict[Literal["title", "text"], str]], # 전체 문서 코퍼스
        queries: Dict[str, str], # 검색할 쿼리들
        top_k: Optional[int] = None, # 각 쿼리당 반환할 상위 문서 개수
        score_function: Literal["cos_sim", "dot"] | None = "cos_sim", # 사용할 유사도 함수
        return_sorted: bool = False, # 결과를 점수순으로 정렬하여 반환할지 여부
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        주어진 쿼리에 대해 코퍼스에서 가장 관련성 높은 문서를 찾아 반환
        """
        # 입력값 유효성 검사
        if score_function not in self.score_functions:
            raise ValueError(
                f"유사도 함수는 'cos_sim' 또는 'dot' 중 하나여야 합니다."
            )

        # 데이터 저장
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}

        # 모든 쿼리를 한 번에 임베딩으로 변환
        query_texts = list(queries.values())
        query_embeddings = self.model.encode(
            query_texts, batch_size=self.batch_size, **kwargs
        )

        # 코퍼스를 텍스트 길이에 따라 내림차순으로 정렬
        sorted_corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )

        # 각 쿼리별로 상위 K개의 결과를 효율적으로 관리하기 위한 최소 힙
        result_heaps = {qid: [] for qid in query_ids}

        # 정렬된 코퍼스 리스트
        corpus_list = [corpus[cid] for cid in sorted_corpus_ids]

        # 코퍼스를 청크 단위로 나누어 순회
        for start_idx in range(0, len(corpus_list), self.corpus_chunk_size):
            end_idx = min(start_idx + self.corpus_chunk_size, len(corpus_list))

            # 현재 청크에 해당하는 문서들의 텍스트 리스트 생성
            sub_corpus_texts = [
                doc.get("title", "") + " " + doc.get("text", "")
                for doc in corpus_list[start_idx:end_idx]
            ]

            # 현재 청크의 문서들을 임베딩으로 변환
            sub_corpus_embeddings = self.model.encode(
                sub_corpus_texts, batch_size=self.batch_size, **kwargs
            )

            # 쿼리 임베딩과 현재 문서 청크 임베딩 간의 유사도 점수 계산
            cos_scores = self.score_functions[score_function](
                query_embeddings, sub_corpus_embeddings
            )
            cos_scores[torch.isnan(cos_scores)] = -1 # NaN 값을 -1로 처리

            # 현재 청크 내에서 각 쿼리별 상위 K개의 점수와 인덱스를 추출
            if top_k is None:
                top_k = len(cos_scores[0])

            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k + 1, len(cos_scores[0])), # 힙과 비교를 위해 k+1개 추출
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            # 힙을 사용하여 전체 코퍼스에 대한 Top-K 결과를 유지
            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    # 현재 청크 내의 인덱스를 전체 코퍼스의 실제 ID로 변환
                    corpus_id = sorted_corpus_ids[start_idx + sub_corpus_id]

                    # 쿼리와 문서가 동일한 경우 제외 (자기 자신과의 비교 방지)
                    if corpus_id != query_id:
                        # 힙이 아직 K개 미만이면 그냥 추가
                        if len(result_heaps[query_id]) < top_k:
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        # 힙이 K개 찼으면, 현재 점수가 힙의 최소값보다 클 때만 교체
                        else:
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        # 최종 결과 정리
        # 힙에 저장된 (점수, 문서ID) 쌍들을 최종 결과 딕셔너리로 변환
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results


# CrossEncoder Reranker

class CrossEncoderReranker:
    """
    Cross-Encoder 모델을 사용하여 검색 결과를 재정렬하는 클래스
    1차 검색의 결과를 입력받아, 계산 비용이 높은 Cross-Encoder로 점수를 재계산
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-e3",device: str | None =None):
        """
        클래스를 초기화하고, 지정된 Cross-Encoder 모델을 로드

        Args:
            model_name (str): Hugging Face Hub에 있는 Cross-Encoder 모델의 이름
        """
        # sentence-transformers 라이브러리의 CrossEncoder 클래스를 사용해 모델을 로드
        self.model = CrossEncoder(model_name, device = device)

    @torch.no_grad() # 추론 시에는 그래디언트 계산을 비활성화
    def rerank(
        self,
        queries: Dict[str, str], # 전체 쿼리 딕셔너리
        corpus: Dict[str, Dict[str, str]], # 전체 문서 딕셔너리
        candidates: Dict[str, Dict[str, float]], # 1차 검색 결과(재정렬할 후보)
        top_k: int = 10, # 최종적으로 반환할 상위 문서 개수
        batch_size: int = 32 # 모델 추론 시 사용할 배치 크기
    ) -> Dict[str, Dict[str, float]]:
        """
        주어진 후보 문서들의 순위를 Cross-Encoder로 재계산하여 반환

        Returns:
            Dict[str, Dict[str, float]]: 재정렬된 상위 K개의 결과 {qid: {doc_id: score, ...}}
        """
        out: Dict[str, Dict[str, float]] = {}
        # 각 쿼리와 그에 해당하는 후보 문서들에 대해 반복
        for qid, cand in candidates.items():
            # 1차 검색 결과가 없는 경우, 빈 결과를 반환
            if not cand:
                out[qid] = {}
                continue

            # 재정렬할 후보 수를 적절히 제한
            # 1차 점수 기준으로 상위 N개의 후보만 선별
            top_items = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:max(top_k * 10, 100)]

            # Cross-Encoder에 입력으로 넣을 [쿼리, 문서] 쌍(pair)을 생성
            pairs, doc_ids = [], []
            qtext = queries.get(qid, "")
            for doc_id, _ in top_items:
                d = corpus[doc_id]
                pairs.append((qtext, f"Title: {d.get('title', '')}\nText: {d.get('text', '')}"))
                doc_ids.append(doc_id) # 점수와 매칭시키기 위해 문서 ID 순서를 저장

            # 모델의 predict 메서드를 사용해 각 쌍의 관련도 점수를 계산
            scores = self.model.predict(pairs, batch_size=batch_size)

            # 계산된 점수를 내림차순으로 정렬하여 상위 top_k개의 인덱스 획득
            order = np.argsort(-np.array(scores))[:top_k]

            # 재정렬된 순서에 따라 최종 결과를 딕셔너리 형태로 저장
            out[qid] = {doc_ids[i]: float(scores[i]) for i in order}

        return out


## 하이브리드 검색
class HybridSearcher:
    """
    BM25Retrieval 과 Dense Retrieval을 결합하여 하이브리드 검색을 수행하는 클래스
    Cross-Encoder를 이용한 재정렬 기능 포함
    """
    def __init__(self):
        """클래스 초기화. 코퍼스, 쿼리, 최종 검색 결과를 저장할 변수 선언"""
        self.corpus: Dict[str, Dict[str, str]] = {}
        self.queries: Dict[str, str] = {}
        self.retrieval_results: Dict[str, Dict[str, float]] = {}

    def set_corpus(self, corpus: Dict[str, Dict[str, str]]):
        """
        corpus: {doc_id: {"title": str, "text": str}}
        """
        self.corpus = corpus

    def set_queries(self, queries: Dict[str, str]):
        """
        queries: {query_id: "질문 텍스트"}
        """
        self.queries = queries

    @staticmethod
    def _softmax_normalize(results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        검색 결과의 점수(score)를 Softmax를 이용해 0과 1 사이의 확률 값으로 정규화
        서로 다른 스케일의 점수를 결합하기 전 사용
        """
        out = {}
        for qid, cand in results.items():
            if not cand:
                out[qid] = {}
                continue
            doc_ids = list(cand.keys())
            scores = torch.tensor([cand[d] for d in doc_ids], dtype=torch.float32)
            # 수치 안정성을 위해 최대값을 뺀 후 softmax를 적용
            probs = torch.softmax(scores - scores.max(), dim=0).tolist()
            out[qid] = {d: p for d, p in zip(doc_ids, probs)}
        return out

    def build_sparse(self, top_k) -> Dict[str, Dict[str, float]]:
        """BM25를 이용한 검색 수행"""
        bm25 = BM25Retriever(self.corpus)
        return bm25.retrieve(self.queries, top_k=top_k)

    def build_dense(self, dense: DenseRetrieval, top_k) -> Dict[str, Dict[str, float]]:
        """DenseRetrieval 객체로 검색 수행"""
        return dense.retrieve(self.corpus, self.queries, top_k=top_k)

    def build_hybrid(
        self,
        dense: DenseRetrieval,
        alpha: float = 0.7, # Dense 검색 결과에 대한 가중치
        dense_topk: int = 200,
        sparse_topk: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """
        Dense와 Sparse 검색 결과를 정규화하고 가중합하여 하이브리드 점수를 생성
        점수 = alpha * dense_prob + (1 - alpha) * sparse_prob
        Args:
          dense (DenseRetrieval):
            미리 초기화된 DenseRetrieval 객체

          alpha (float, optional):
            하이브리드 점수에서 Dense 검색 결과가 차지할 가중치
            Sparse 검색의 가중치는 자동으로 1 - alpha로 계산

          dense_topk (int, optional):
            Dense 검색 단계에서 검색할 상위 문서의 수

          sparse_topk (int, optional):
            BM25 단계에서 검색할 상위 문서의 수

       Returns:
        Dict[str, Dict[str, float]]:
            하이브리드 점수가 계산된 결과 딕셔너리
            형식: {쿼리ID: {문서ID: 하이브리드_점수, ...}}
        """

        # 각 검색 결과를 가져와 점수를 정규화
        dense_res = self._softmax_normalize(self.build_dense(dense, top_k=dense_topk))
        sparse_res = self._softmax_normalize(self.build_sparse(top_k=sparse_topk))

        hybrid: Dict[str, Dict[str, float]] = {}

        # Dense 결과에 가중치를 적용하여 하이브리드 점수에 추가
        for qid, cand in dense_res.items():
            dst = hybrid.setdefault(qid, {})
            for doc_id, p in cand.items():
                dst[doc_id] = alpha * p

        # BM25 결과에 가중치를 적용하여 하이브리드 점수에 더함
        #  dense 결과에 없던 문서도 이 과정에서 추가
        beta = 1.0 - alpha
        for qid, cand in sparse_res.items():
            dst = hybrid.setdefault(qid, {})
            for doc_id, p in cand.items():
                dst[doc_id] = dst.get(doc_id, 0.0) + beta * p

        return hybrid

    def search(
        self,
        dense: DenseRetrieval,
        alpha: float = 0.7,
        dense_topk: int = 200,
        sparse_topk: int = 1000,
        final_topk: int = 10, # 최종 반환할 문서 수
        reranker: Optional[CrossEncoderReranker] = None,
        reranker_batch_size: int = 32
    ) -> Dict[str, Dict[str, float]]:
        """
        전체 검색 파이프라인을 실행
        Args:
         dense (DenseRetrieval):
            미리 초기화된 DenseRetrieval 객체
        alpha (float, optional):
            하이브리드 점수 계산 시 Dense 검색 결과에 적용할 가중치

        dense_topk (int, optional):
             Dense 검색 단계에서 가져올 상위 후보 문서의 수

        sparse_topk (int, optional):
             BM25 단계에서 가져올 상위 후보 문서의 수

        final_topk (int, optional):
            각 쿼리당 최종적으로 반환할 문서의 수

        reranker (Optional[CrossEncoderReranker], optional):
            CrossEncoderReranker 객체
            이 값이 제공되면, 하이브리드 검색 결과 상위권을 대상으로 점수를 재계산

        reranker_batch_size (int, optional):
            리랭커를 사용할 경우, 모델 추론에 적용할 배치 크기

    Returns:
        Dict[str, Dict[str, float]]:
            최종 검색 결과가 담긴 딕셔너리
            형식: {쿼리ID: {문서ID: 점수, ...}}
    """
        # 하이브리드 검색을 수행하여 1차 후보군을 생성
        hybrid = self.build_hybrid(dense, alpha=alpha, dense_topk=dense_topk, sparse_topk=sparse_topk)

        # reranker가 제공되지 않은 경우
        if reranker is None:
            final_results: Dict[str, Dict[str, float]] = {}
            # 하이브리드 점수가 높은 순으로 정렬하여 최종 top-k개만 반환
            for qid, cand in hybrid.items():
                top = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:final_topk]
                final_results[qid] = dict(top)
            self.retrieval_results = final_results
            return final_results

        # 재정렬기(reranker)가 제공된 경우
        # 하이브리드 검색 결과를 후보로 하여 재정렬을 수행 후 결과 반환
        reranked = reranker.rerank(self.queries, self.corpus, hybrid, top_k=final_topk, batch_size=reranker_batch_size)
        self.retrieval_results = reranked
        return reranked
