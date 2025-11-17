#!/bin/bash

# ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 라이브러리 설치
pip install -r install_lib.txt

# ollama 서버 실행 - 로그
nohup ollama serve > ollama.log 2>&1 &



