from huggingface_hub import snapshot_download
import argparse
import os

def install_model(args):
    """허깅페이스 모델 다운로드"""
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        allow_patterns=[args.filename],
    )
    gguf_path = os.path.join(path, args.filename)

    # 모델 파일 제작 경로 지정 
    os.makedirs(args.model_dir, exist_ok=True)

    model_file_path = os.path.join(args.model_dir, "Modelfile")
    with open(model_file_path, "w", encoding="utf-8") as f:
        f.write(f"""# 사용할 GGUF 파일 지정
FROM {gguf_path}

# 모델 기본 동작 설정
PARAMETER temperature {args.temperature}
PARAMETER top_p {args.top_p}
PARAMETER top_k {args.top_k}
PARAMETER repeat_penalty {args.repeat_penalty}

# 시스템 프롬프트 (option)
SYSTEM \"\"\"
{args.prompt}
\"\"\"
""")

    print("GGUF:", gguf_path, "size:", os.path.getsize(gguf_path))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGUF 모델 다운로드")

    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo id (ex: QuantFactory/llama-3-Korean-Bllossom-8B-GGUF)")
    parser.add_argument("--filename", type=str, required=True, help=".gguf 파일 이름 (ex: llama-3-Korean-Bllossom-8B.Q5_K_M.gguf)")
    parser.add_argument("--local_dir", type=str, default="/workspace/models", help="Hugging Face 파일 다운로드 디렉토리")
    parser.add_argument("--model_dir", type=str, required=True, help="Ollama 빌드를 위한 디렉토리 (Modelfile 포함)")

    # ollama 매개변수
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature 설정")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p 설정")
    parser.add_argument("--top_k", type=int, default=40, help="top_k 설정")
    parser.add_argument("--repeat_penalty", type=float, default=1.1, help="repeat_penalty 설정")
    parser.add_argument("--prompt", type=str, default=None, help="시스템 프롬프트 설정")

    args = parser.parse_args()

    install_model(args)
