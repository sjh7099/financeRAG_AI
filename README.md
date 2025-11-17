# financeRAG
금융보안원 AI Challenges : RAG 활용 llm 모델 설계

## 구성된 Shell Script

| 파일명              | 설명 |
|--------------------|------|
| `install.sh`       | Ollama 및 Python 라이브러리 설치 + Ollama 서버 실행 |
| `install_model.sh` | GGUF 모델 다운로드 및 `ollama create` 실행 |
| `run.sh`           | 질의응답 실행 (검색 - LLM 응답 - 결과 저장) |

---


### `install.sh` 실행 방법 

- `install.sh`는 **Ollama 환경 및 Python 라이브러리 설치**
- **Ollama 서버 실행**까지 자동으로 처리하는 초기 설치 스크립트

### 실행 방법

```bash
bash install.sh
```

### 내부 동작

1. Ollama 설치(리눅스)
2. python 라이브러리 설치
3. Ollama 서버 백그라운드 실행


---


### `install_model.sh` 실행 방법 및 설명

- `install_model.sh`는 HuggingFace에서 **GGUF 모델 파일을 다운로드**하고 
- 해당 모델을 기반으로 **Ollama에 사용할 `Modelfile`을 자동 생성한 뒤 모델 등록까지 완료**하는 스크립트



### 실행 방법

```bash
bash install_model.sh
```
- 변동하고 싶은 부분은 sh 파일 내부를 수정해야 합니다

`install_model.sh` 내부 구성:
```bash
python /workspace/code/install_model.py \
  --repo_id "$REPO_ID" \
  --filename "$FILENAME" \
  --local_dir "$LOCAL_DIR" \
  --model_dir "$MODEL_DIR" \
  --temperature 0.65 \
  --top_p 0.85 \
  --top_k 45 \
  --repeat_penalty 1.15 \
  --prompt "$PROMPT"
```
1. 허깅페이스에서 gguf 모델 다운로드
2. 지정한 `MODEL_DIR` 에 Modelfile 자동 생성
    - 추론 파라미터, 시스템 프롬프트 포함 가능 

### 옵션 설명

| 옵션 이름              | 설명                                      |
| ------------------ | --------------------------------------- |
| `--repo_id`        | Hugging Face에서 다운로드할 모델 저장소의 ID         |
| `--filename`       | 다운로드할 GGUF 모델 파일 이름                     |
| `--local_dir`      | 모델 파일이 저장될 로컬 경로                        |
| `--model_dir`      | Modelfile이 생성될 경로 (ollama 모델 생성 시 사용)   |
| `--temperature`    | 응답의 창의성 조절 (높을수록 다양하게, 낮을수록 일관성 ↑)      |
| `--top_p`          | nucleus sampling의 확률 컷오프 기준 (응답 다양성 조절) |
| `--top_k`          | 상위 k개 토큰만 고려하여 응답 생성 (선택지 범위 제한)        |
| `--repeat_penalty` | 반복되는 단어 생성 억제를 위한 penalty 계수            |
| `--prompt`         | 시스템 프롬프트 (모델이 응답할 때 따를 기본 지침)           |


---



### `run.sh` 실행 방법

- **질문-문서 검색 기반의 질의응답 시스템**을 실행
- 검색된 문서를 기반으로 Ollama를 통해 답변을 생성 가능 

### 실행 명령어 

```bash
bash run.sh
```
- 변동하고 싶은 부분은 sh 파일 내부를 수정해야 합니다

`run.sh` 내부 구성 :

```bash
python /workspace/code/run_oll.py \
  --rag_data /workspace/data/ALL_laws_chunks.jsonl \
  --query_data /workspace/data/test.csv \
  --output /workspace/data/ollama_result.csv \
  --embedding_path intfloat/multilingual-e5-base \
  --reranker BAAI/bge-reranker-base \
  --serch_alpha 0.7 \
  --dense_topk 10 \
  --sparse_topk 10 \
  --final_topk 5 \
  --model_tag blossom-8b \
  --num_predict 128 \
  --temperature 0.3 \
  --top_p 0.9 \
  --num_ctx 3072 \
  --gpu_layers -1
```

### 각 옵션 설명

| 옵션명  | 설명    |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| `--rag_data`       | 질의응답에 사용할 전체 문서 코퍼스 파일 (jsonl 형식) |
| `--query_data`     | 사용자가 질의한 질문 목록이 담긴 CSV 파일. `ID`, `Question` 컬럼을 포함.                                         |
| `--output`         | Ollama 모델의 응답 결과를 저장할 CSV 파일 경로.                                                       |
| `--embedding_path` | Dense Retriever로 사용할 임베딩 모델의 Hugging Face 경로 또는 로컬 경로. 예시: `intfloat/multilingual-e5-base`    |
| `--reranker`       | CrossEncoder 기반 리랭커 모델 경로. BM25 + Dense 결과를 재정렬할 때 사용됩니다.                                         |
| `--serch_alpha`    | BM25와 Dense 결과를 결합할 때 사용하는 하이브리드 가중치. `1.0`에 가까울수록 Dense 비중이 높음                                 |
| `--dense_topk`     | Dense Retriever에서 상위 몇 개의 문서를 가져올지 설정                                                        |
| `--sparse_topk`    | BM25 기반 Sparse 검색 결과 중 상위 몇 개를 가져올지 설정                                                     |
| `--final_topk`     | Dense + Sparse 통합 후 최종적으로 상위 몇 개 문서를 사용할지 설정                                         |
| `--model_tag`      | Ollama에서 사용할 LLM 모델의 이름(태그) 예: `blossom-8b`                                                      |
| `--num_predict`    | Ollama에서 생성할 최대 토큰 수. 예: 답변 최대 길이.                                                                |
| `--temperature`    | 생성의 무작위성 정도. 높을수록 창의적, 낮을수록 안정적 결과.                                                               |
| `--top_p`          | nucleus sampling의 누적 확률 컷오프. 낮출수록 안정된 답변.                                                         |
| `--num_ctx`        | 모델이 한 번에 볼 수 있는 최대 토큰 길이 (context window size).                                                   |
| `--gpu_layers`     | 몇 개의 레이어를 GPU에 올릴지 설정. `-1`이면 전체 레이어를 GPU에 로드 (RAM이 부족하면 줄여야 함)                               |


### `run.sh` 실행 전 체크리스트

* Ollama 서버가 실행 중이어야 함

* `ollama list` 명령으로 `--model_tag`에 입력한 모델이 등록되어 있는지 확인

