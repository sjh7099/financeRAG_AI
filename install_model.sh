#!/bin/bash

# 설치할 모델 정보
REPO_ID="LGAI-EXAONE/EXAONE-4.0-32B-GGUF"
FILENAME="EXAONE-4.0-32B-Q5_K_M.gguf"
LOCAL_DIR="../root/workspace/models"
MODEL_DIR="../root/workspace/exaone"
MODEL_TAG="exaone"

# 시스템 프롬프트
PROMPT="당신은 한국어를 잘 이해하고 대답하는 유능한 AI 어시스턴트입니다.
답변은 가능한 한 간결하고 명확하게 작성하세요."

# 실행
python ../workspace/code/install_model_v2.py \
  --repo_id "$REPO_ID" \
  --filename "$FILENAME" \
  --local_dir "$LOCAL_DIR" \
  --model_dir "$MODEL_DIR" \
  --temperature 0.65 \
  --top_p 0.85 \
  --top_k 45 \
  --repeat_penalty 1.15 \
  --prompt "$PROMPT"\
  --model_tag "$MODEL_TAG"\
  --create

# ollama create "$MODEL_TAG" -f "$MODEL_DIR/Modelfile"