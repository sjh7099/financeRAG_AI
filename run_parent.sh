#!/bin/bash

python /root/workspace/code/run_oll_parent.py \
  --rag_data /root/workspace/data/chunks_child_20250825_103819.jsonl \
  --expanded_jsonl /root/workspace/data/chunks_child_expanded_20250825_103819.jsonl \
  --articles_jsonl /root/workspace/data/articles_20250825_103819.jsonl \
  --query_data /root/workspace/data/test.csv \
  --output /root/workspace/data/ollama_result.csv \
  --embedding_path intfloat/multilingual-e5-base \
  --reranker BAAI/bge-reranker-base \
  --search_alpha 0.7 \
  --dense_topk 10 \
  --sparse_topk 10 \
  --final_topk 5 \
  --model_tag exaone \
  --num_predict 128 \
  --temperature 0.3 \
  --top_p 0.9 \
  --num_ctx 3072 \
  --gpu_layers -1 