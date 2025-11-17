import os
import re
import glob
import json
import math
import heapq
from typing import Dict, List, Tuple, Any, Optional, Literal

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------------------- Utilities preserved --------------------

@torch.no_grad()
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity for (N,D) x (M,D) -> (N,M)."""
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if a.ndim == 1: a = a.unsqueeze(0)
    if b.ndim == 1: b = b.unsqueeze(0)
    a = torch.nn.functional.normalize(a, p=2, dim=1)
    b = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a, b.transpose(0, 1))

@torch.no_grad()
def dot_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dot-product score for (N,D) x (M,D) -> (N,M)."""
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if a.ndim == 1: a = a.unsqueeze(0)
    if b.ndim == 1: b = b.unsqueeze(0)
    return torch.mm(a, b.transpose(0, 1))

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
def simple_tokenize(text: str) -> List[str]:
    """Very simple tokenization compatible with your original code."""
    return _WORD_RE.findall((text or "").lower())

# -------------------- Original retrievers preserved --------------------

class BM25Retriever:
    """BM25 over {doc_id: {'title','text'}} corpus."""
    def __init__(self, corpus: Dict[str, Dict[str, str]]):
        self.doc_ids = list(corpus.keys())
        tokenized = [
            simple_tokenize(corpus[i].get("title","") + " " + corpus[i].get("text",""))
            for i in self.doc_ids
        ]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, queries: Dict[str, str], top_k: int = 1000, **kwargs) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for qid, q in queries.items():
            toks = simple_tokenize(q)
            scores = self.bm25.get_scores(toks)
            idx = np.argsort(-scores)[:top_k]
            results[qid] = {self.doc_ids[i]: float(scores[i]) for i in idx}
        return results

class DenseRetrieval:
    """SentenceTransformer-based dense retrieval (unchanged)."""
    def __init__(self, model, batch_size: int = 64, score_functions=None, corpus_chunk_size: int = 50000):
        self.model = model
        self.batch_size = batch_size
        if score_functions is None:
            score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_functions = score_functions
        self.corpus_chunk_size = corpus_chunk_size
        self.results: Dict[str, Dict[str, float]] = {}

    def retrieve(
        self,
        corpus: Dict[str, Dict[Literal["title", "text"], str]],
        queries: Dict[str, str],
        top_k: Optional[int] = None,
        score_function: Literal["cos_sim", "dot"] | None = "cos_sim",
        return_sorted: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if score_function not in self.score_functions:
            raise ValueError("score_function must be 'cos_sim' or 'dot'")

        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}

        query_texts = list(queries.values())
        query_embeddings = self.model.encode(query_texts, batch_size=self.batch_size, **kwargs)

        # sort by doc length to pack batches well
        sorted_corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title","") + corpus[k].get("text","")), reverse=True)
        result_heaps = {qid: [] for qid in query_ids}
        corpus_list = [corpus[cid] for cid in sorted_corpus_ids]

        for start_idx in range(0, len(corpus_list), self.corpus_chunk_size):
            end_idx = min(start_idx + self.corpus_chunk_size, len(corpus_list))
            sub_corpus_texts = [d.get("title","") + " " + d.get("text","") for d in corpus_list[start_idx:end_idx]]
            sub_corpus_embeddings = self.model.encode(sub_corpus_texts, batch_size=self.batch_size, **kwargs)

            sim = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            sim[torch.isnan(sim)] = -1

            if top_k is None:
                top_k = len(sim[0])

            vals, idxs = torch.topk(sim, k=min(top_k + 1, len(sim[0])), dim=1, largest=True, sorted=return_sorted)
            vals = vals.cpu().tolist()
            idxs = idxs.cpu().tolist()

            for qi in range(len(query_embeddings)):
                qid = query_ids[qi]
                for sub_id, score in zip(idxs[qi], vals[qi]):
                    corpus_id = sorted_corpus_ids[start_idx + sub_id]
                    if corpus_id != qid:
                        if len(result_heaps[qid]) < top_k:
                            heapq.heappush(result_heaps[qid], (float(score), corpus_id))
                        else:
                            heapq.heappushpop(result_heaps[qid], (float(score), corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results

class CrossEncoderReranker:
    """Vanilla Cross-Encoder reranker (unchanged)."""
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-e3", device: Optional[str] = None):
        self.model = CrossEncoder(model_name, device=device)

    @torch.no_grad()
    def rerank(self, queries: Dict[str, str], corpus: Dict[str, Dict[str, str]], candidates: Dict[str, Dict[str, float]],
               top_k: int = 10, batch_size: int = 32) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for qid, cand in candidates.items():
            if not cand:
                out[qid] = {}
                continue
            top_items = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:max(top_k * 10, 100)]
            pairs, doc_ids = [], []
            qtext = queries.get(qid, "")
            for doc_id, _ in top_items:
                d = corpus[doc_id]
                pairs.append((qtext, f"Title: {d.get('title','')}\nText: {d.get('text','')}"))
                doc_ids.append(doc_id)
            scores = self.model.predict(pairs, batch_size=batch_size)
            order = np.argsort(-np.array(scores))[:top_k]
            out[qid] = {doc_ids[i]: float(scores[i]) for i in order}
        return out

class HybridSearcher:
    """Hybrid of dense + sparse + optional rerank (unchanged)."""
    def __init__(self):
        self.corpus: Dict[str, Dict[str, str]] = {}
        self.queries: Dict[str, str] = {}
        self.retrieval_results: Dict[str, Dict[str, float]] = {}

    def set_corpus(self, corpus: Dict[str, Dict[str, str]]):
        self.corpus = corpus

    def set_queries(self, queries: Dict[str, str]):
        self.queries = queries

    @staticmethod
    def _softmax_normalize(results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        out = {}
        for qid, cand in results.items():
            if not cand:
                out[qid] = {}
                continue
            doc_ids = list(cand.keys())
            scores = torch.tensor([cand[d] for d in doc_ids], dtype=torch.float32)
            probs = torch.softmax(scores - scores.max(), dim=0).tolist()
            out[qid] = {d: p for d, p in zip(doc_ids, probs)}
        return out

    def build_sparse(self, top_k) -> Dict[str, Dict[str, float]]:
        bm25 = BM25Retriever(self.corpus)
        return bm25.retrieve(self.queries, top_k=top_k)

    def build_dense(self, dense: DenseRetrieval, top_k) -> Dict[str, Dict[str, float]]:
        return dense.retrieve(self.corpus, self.queries, top_k=top_k)

    def build_hybrid(self, dense: DenseRetrieval, alpha: float = 0.7, dense_topk: int = 200, sparse_topk: int = 1000) -> Dict[str, Dict[str, float]]:
        dense_res = self._softmax_normalize(self.build_dense(dense, top_k=dense_topk))
        sparse_res = self._softmax_normalize(self.build_sparse(top_k=sparse_topk))

        hybrid: Dict[str, Dict[str, float]] = {}
        for qid, cand in dense_res.items():
            dst = hybrid.setdefault(qid, {})
            for doc_id, p in cand.items():
                dst[doc_id] = 0.0 + (alpha * p)

        beta = 1.0 - alpha
        for qid, cand in sparse_res.items():
            dst = hybrid.setdefault(qid, {})
            for doc_id, p in cand.items():
                dst[doc_id] = dst.get(doc_id, 0.0) + (beta * p)
        return hybrid

    def search(self, dense: DenseRetrieval, alpha: float = 0.7, dense_topk: int = 200, sparse_topk: int = 1000,
               final_topk: int = 10, reranker: Optional[CrossEncoderReranker] = None, reranker_batch_size: int = 32) -> Dict[str, Dict[str, float]]:
        hybrid = self.build_hybrid(dense, alpha=alpha, dense_topk=dense_topk, sparse_topk=sparse_topk)

        if reranker is None:
            final_results: Dict[str, Dict[str, float]] = {}
            for qid, cand in hybrid.items():
                top = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:final_topk]
                final_results[qid] = dict(top)
            self.retrieval_results = final_results
            return final_results

        reranked = reranker.rerank(self.queries, self.corpus, hybrid, top_k=final_topk, batch_size=reranker_batch_size)
        self.retrieval_results = reranked
        return reranked

# -------------------- Parent–Child additions --------------------

def _pick_text(row: dict, fields: Tuple[str, ...] = ("prefixed_text","text")) -> str:
    for f in fields:
        v = row.get(f)
        if v: return str(v)
    return ""

def _child_id(law_key: str, start: int, end: int) -> str:
    return f"{law_key}:{int(start)}:{int(end)}"

def build_child_corpus(child_jsonl_path: str, text_field_priority: Tuple[str, ...] = ("prefixed_text","text")) -> Dict[str, Dict[str, str]]:
    """
    Build a corpus from chunks_child_*.jsonl
    Returns {doc_id: {'title','text','law_key','law_name','law_no','title_meta','start','end'}}
    where doc_id = "{law_key}:{start}:{end}".
    Accepts a single path or a glob pattern.
    """
    rows: List[dict] = []
    paths = glob.glob(child_jsonl_path) if any(ch in child_jsonl_path for ch in "*?[]") else [child_jsonl_path]
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

    corpus: Dict[str, Dict[str, str]] = {}
    for r in rows:
        lk = r.get("law_key")
        st = int(r.get("start") or -1)
        en = int(r.get("end") or -1)
        doc_id = _child_id(lk, st, en)
        corpus[doc_id] = {
            "title": r.get("title",""),
            "text": _pick_text(r, text_field_priority),
            "law_key": lk,
            "law_name": r.get("law_name",""),
            "law_no": r.get("law_no",""),
            "title_meta": r.get("title",""),
            "start": st,
            "end": en,
        }
    return corpus

class ExpansionIndex:
    """
    Holds:
      - child (law_key,start,end) -> expanded chunk (prefers 'prefixed_text' else 'text')
      - law_key -> article/parent record (to fetch 'head')
    """
    def __init__(self, expanded_jsonl_path: Optional[str], articles_jsonl_path: Optional[str],
                 text_field_priority: Tuple[str, ...] = ("prefixed_text","text")):
        self._exp: Dict[Tuple[str,int,int], dict] = {}
        self._art: Dict[str, dict] = {}
        self._fields = text_field_priority

        if expanded_jsonl_path:
            rows: List[dict] = []
            paths = glob.glob(expanded_jsonl_path) if any(ch in expanded_jsonl_path for ch in "*?[]") else [expanded_jsonl_path]
            for p in paths:
                if not os.path.exists(p): 
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        if not line: continue
                        try: rows.append(json.loads(line))
                        except: pass
            for r in rows:
                lk = r.get("law_key"); st = int(r.get("start") or -1); en = int(r.get("end") or -1)
                self._exp[(lk,st,en)] = r

        if articles_jsonl_path:
            rows: List[dict] = []
            paths = glob.glob(articles_jsonl_path) if any(ch in articles_jsonl_path for ch in "*?[]") else [articles_jsonl_path]
            for p in paths:
                if not os.path.exists(p): 
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        if not line: continue
                        try: rows.append(json.loads(line))
                        except: pass
            for r in rows:
                self._art[r.get("law_key")] = r

    def _expanded_text_of(self, row: Optional[dict]) -> str:
        if not row: return ""
        for f in self._fields:
            if row.get(f): return str(row.get(f))
        return ""

    def get_expanded_text(self, doc_id: str, corpus: Dict[str, Dict[str, str]]) -> str:
        law_key, s, e = doc_id.split(":")
        row = self._exp.get((law_key, int(s), int(e)))
        return self._expanded_text_of(row) or corpus[doc_id].get("text","")

    def get_parent_head(self, doc_id: str) -> str:
        law_key, *_ = doc_id.split(":")
        return (self._art.get(law_key) or {}).get("head","")

class ParentAwareReranker(CrossEncoderReranker):
    """Cross-Encoder reranker that feeds expanded child text + parent head."""
    def __init__(self, expansion: ExpansionIndex, model_name: str = "BAAI/bge-reranker-v2-e3", device: Optional[str] = None):
        super().__init__(model_name=model_name, device=device)
        self.expansion = expansion

    @torch.no_grad()
    def rerank(self, queries: Dict[str, str], corpus: Dict[str, Dict[str, str]], candidates: Dict[str, Dict[str, float]],
               top_k: int = 10, batch_size: int = 32) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for qid, cand in candidates.items():
            if not cand:
                out[qid] = {}
                continue
            top_items = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:max(top_k * 10, 100)]
            pairs, doc_ids = [], []
            qtext = queries.get(qid, "")
            for doc_id, _ in top_items:
                d = corpus[doc_id]
                expanded = self.expansion.get_expanded_text(doc_id, corpus)
                head = self.expansion.get_parent_head(doc_id)
                right = f"Head: {head}\nTitle: {d.get('title','')}\nText: {expanded}"
                pairs.append((qtext, right))
                doc_ids.append(doc_id)
            scores = self.model.predict(pairs, batch_size=batch_size)
            order = np.argsort(-np.array(scores))[:top_k]
            out[qid] = {doc_ids[i]: float(scores[i]) for i in order}
        return out

def pack_results_with_context(results: Dict[str, Dict[str, float]], corpus: Dict[str, Dict[str, str]], xp: ExpansionIndex,
                              top_n_per_query: int = 10, preview_chars: int = 900) -> Dict[str, List[Dict[str, Any]]]:
    """Return child + expanded + parent-head for each hit, per query."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for qid, cand in results.items():
        hits = []
        for doc_id, score in sorted(cand.items(), key=lambda x: x[1], reverse=True)[:top_n_per_query]:
            d = corpus[doc_id]
            child_text = (d.get("text","") or "").replace("\n"," ")[:preview_chars]
            expanded_text = (xp.get_expanded_text(doc_id, corpus) or child_text).replace("\n"," ")[:preview_chars]
            parent_head = xp.get_parent_head(doc_id)
            hits.append({
                "doc_id": doc_id,
                "score": float(score),
                "law_key": d.get("law_key"),
                "law_name": d.get("law_name"),
                "law_no": d.get("law_no"),
                "title": d.get("title_meta"),
                "child_text": child_text,
                "expanded_text": expanded_text,
                "parent_head": parent_head
            })
        out[qid] = hits
    return out