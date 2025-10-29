from fastapi import FastAPI, Request, Query
import requests

app = FastAPI()

VLLM_URL = "http://localhost:8000/v1/chat/completions"

@app.post("/generate")
async def generate(request: Request, type: str = Query("default")):
    data = await request.json()
    prompt = data.get("prompt", "")

    if type == "checklist":
        system_prompt = (
            "당신은 부동산 위험 분석 전문가입니다. "
            "다음 매물 설명을 기반으로 위험 사항 체크리스트를 생성하세요."
            "체크리스트 형식으로 여러개의 항목으로 답변, 구분자로 ;를 사용, 줄바꿈(\n)이나 특수문자 사용금지"
        )
    else:
        return {"contents": "Error: Unsupported type"}

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    res = requests.post(VLLM_URL, json=payload)
    if res.status_code != 200:
        return {"contents": f"Error: {res.text}"}

    result = res.json()
    output = result["choices"][0]["message"]["content"]
    return {"contents": list(output.split(";"))}
