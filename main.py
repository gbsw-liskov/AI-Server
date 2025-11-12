from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import requests
from typing import List, Dict, Any, Optional

app = FastAPI()

VLLM_URL = "http://localhost:8000/v1/chat/completions"


# 기존 checklist API
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
        res = requests.post(VLLM_URL, json=payload, timeout=60)
        res.raise_for_status()
    except Exception as e:
        return {"contents": f"Error: Failed to connect to model server ({e})"}

    result = res.json()
    output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    checklist_items = [item.strip() for item in output.split(";") if item.strip()]

    return {"contents": checklist_items}


# 새로운 analyze API
@app.post("/analyze")
async def analyze_property(
    propertyId: int = Form(...),
    name: str = Form(...),
    address: str = Form(...),
    propertyType: str = Form(...),
    floor: int = Form(...),
    buildYear: int = Form(...),
    area: str = Form(...),
    availableDate: str = Form(...),
    marketPrice: float = Form(...),
    deposit: float = Form(...),
    monthlyRent: float = Form(...),
    files: Optional[List[UploadFile]] = None
):
    """
    Request 형식:
    {
        propertyId,
        name,
        address,
        propertyType,
        floor,
        buildYear,
        area,
        availableDate,
        marketPrice,
        deposit,
        monthlyRent
    }

    [files]: multipart/form-data

    Response 형식:
    {
        "totalRisk": number,
        "summary": string,
        "details": [
            {"title": string, "content": string}
        ]
    }
    """

    # 업로드된 파일 내용 읽기
    file_texts = []
    if files:
        for file in files:
            try:
                content = (await file.read()).decode("utf-8", errors="ignore")
                file_texts.append(f"[{file.filename}] 내용:\n{content}")
            except Exception:
                file_texts.append(f"[{file.filename}] 파일을 읽을 수 없습니다.")

    combined_files = "\n\n".join(file_texts) if file_texts else "첨부 문서 없음"

    # 프롬프트 구성
    prompt = (
        f"매물명: {name}, 주소: {address}, 유형: {propertyType}, "
        f"층수: {floor}, 준공연도: {buildYear}, 면적: {area}, "
        f"입주가능일: {availableDate}, 시세: {marketPrice}원, "
        f"보증금: {deposit}원, 월세: {monthlyRent}원.\n\n"
        f"첨부된 부동산 관련 문서 내용:\n{combined_files}"
    )

    system_prompt = (
        "당신은 부동산 리스크 분석 전문가입니다. "
        "주어진 부동산 정보와 첨부 문서를 종합하여 위험도를 평가하세요. "
        "결과는 JSON 형식으로 다음 구조로 작성하세요:\n"
        "{"
        "\"totalRisk\": 0~100 사이의 숫자, "
        "\"summary\": '요약 설명', "
        "\"details\": ["
        "{\"title\": '위험 항목 제목', \"content\": '상세 설명'}"
        "]"
        "}"
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
        res = requests.post(VLLM_URL, json=payload, timeout=120)
        res.raise_for_status()
    except Exception as e:
        return {"error": f"Error: Failed to connect to model server ({e})"}

    result = res.json()
    output = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    # LLM이 JSON을 그대로 반환할 경우 파싱 시도
    try:
        import json
        parsed_output = json.loads(output)
        return parsed_output
    except Exception:
        return {"raw_output": output}
