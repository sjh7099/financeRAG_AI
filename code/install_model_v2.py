#!/usr/bin/env python3
from huggingface_hub import snapshot_download
import argparse
import os
import sys
import hashlib
import shutil
import subprocess
from pathlib import Path

# Ollama blob 저장소 경로
OLLAMA_HOME = Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models")))
BLOBS_DIR = OLLAMA_HOME / "blobs"

def sha256_file(p: Path, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path) -> str:
    """하드링크 시도, 실패하면 복사"""
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"

def install_model(args):
    # 1) 허깅페이스에서 모델 다운로드
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        allow_patterns=[args.filename],
        local_dir_use_symlinks=False,
    )
    gguf_path = Path(path) / args.filename
    if not gguf_path.exists():
        print(f"[ERROR] GGUF 파일 없음: {gguf_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] 다운로드 완료: {gguf_path} ({gguf_path.stat().st_size:,} bytes)")

    # 2) Ollama blob 경로 준비 (중복 방지)
    ensure_dir(BLOBS_DIR)
    sha = sha256_file(gguf_path)
    blob_path = BLOBS_DIR / f"sha256-{sha}"
    action = link_or_copy(gguf_path, blob_path)
    print(f"[INFO] blob 경로: {blob_path} ({action})")

    # 3) Modelfile 작성
    model_dir = Path(args.model_dir)
    ensure_dir(model_dir)
    modelfile = model_dir / "Modelfile"
    with modelfile.open("w", encoding="utf-8") as f:
        f.write(f"""FROM {blob_path}

PARAMETER temperature {args.temperature}
PARAMETER top_p {args.top_p}
PARAMETER top_k {args.top_k}
PARAMETER repeat_penalty {args.repeat_penalty}

SYSTEM \"\"\"
{args.prompt if args.prompt else ""}
\"\"\"
""")
    print(f"[INFO] Modelfile 생성: {modelfile}")

    # 4) (옵션) ollama create 실행
    if args.create:
        # 이미 있으면 제거 (옵션)
        r = subprocess.run(["ollama", "show", args.model_tag],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode == 0 and args.force_recreate:
            subprocess.run(["ollama", "rm", args.model_tag], check=False)
        elif r.returncode == 0:
            print(f"[INFO] 모델 {args.model_tag} 이미 존재 (건너뜀, --force_recreate로 덮어쓰기 가능)")
            return
        subprocess.run(["ollama", "create", args.model_tag, "-f", str(modelfile)], check=True)
        print(f"[SUCCESS] Ollama 모델 생성 완료: {args.model_tag}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HF 모델 다운로드 후 Ollama 등록 (model_tag 전용)")
    parser.add_argument("--repo_id", required=True, help="HuggingFace repo id")
    parser.add_argument("--filename", required=True, help=".gguf 파일명")
    parser.add_argument("--local_dir", default="/workspace/models", help="다운로드 위치")
    parser.add_argument("--model_dir", required=True, help="Modelfile 저장 위치")
    parser.add_argument("--model_tag", required=True, help="Ollama에서 사용할 모델 이름(tag)")

    # 파라미터
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repeat_penalty", type=float, default=1.1)
    parser.add_argument("--prompt", type=str, default=None)

    # 옵션
    parser.add_argument("--create", action="store_true", help="바로 ollama create 실행")
    parser.add_argument("--force_recreate", action="store_true", help="기존 모델 덮어쓰기")

    args = parser.parse_args()
    install_model(args)