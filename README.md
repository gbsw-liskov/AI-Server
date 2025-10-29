# 환경 (Runpod Pytorch 기준)
```
pip install vllm
pip install hf_transformer
pip install uvicorn
# 모델은 알아서 허깅페이스에서 다운로드...
```

# 서버 실행 (run pod)
```
vllm serve 모델명(ex. openai/gpt-oss-20b)
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```