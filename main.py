from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

VLLM_URL = "http://localhost:8000/v1/chat/completions"

# 요청 본문 스키마 정의
class PropertyRequest(BaseModel):
    propertyId: int
    name: str
    address: str
    propertyType: str
    floor: int
    buildYear: int
    area: str
    availableDate: str

@app.post("/checklist")
async def generate(data: PropertyRequest):
    """
    Request 형식:
    {
        "propertyId": int,
        "name": str,
        "address": str,
        "propertyType": str,
        "floor": int,
        "buildYear": int,
        "area": str,
        "availableDate": str
    }

    Response 형식:
    {
        "contents": [ ... ]
    }
    """

    prompt = (
        f"매물명: {data.name}, 주소: {data.address}, 유형: {data.propertyType}, "
        f"층수: {data.floor}, 준공연도: {data.buildYear}, 면적: {data.area}, "
        f"입주가능일: {data.availableDate}"
    )

    system_prompt = (
        "당신은 부동산 위험 분석 전문가입니다. "
        "다음 매물 설명을 기반으로 위험 사항 체크리스트를 생성하세요. "
        "체크리스트 형식으로 여러 항목으로 답변하고, 각 항목은 ';'로 구분하세요. "
        "줄바꿈이나 특수문자는 사용하지 마세요."
    )

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        res = requests.post(VLLM_URL, json=payload, timeout=10)
        res.raise_for_status()
    except Exception as e:
        return {"contents": f"Error: Failed to connect to model server ({e})"}

    result = res.json()
    output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    checklist_items = [item.strip() for item in output.split(";") if item.strip()]

    return {"contents": checklist_items}
