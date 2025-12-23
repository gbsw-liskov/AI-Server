from typing import List, Optional

from fastapi import APIRouter, Form, UploadFile

from llm_client import llm_client, format_property_info, read_files

router = APIRouter()


@router.post("/analyze")
async def analyze_property(
    propertyId: int = Form(...),
    name: str = Form(...),
    address: str = Form(...),
    propertyType: str = Form(...),
    floor: int = Form(...),
    builtYear: int = Form(...),
    area: int = Form(...),
    marketPrice: float = Form(...),
    deposit: float = Form(...),
    monthlyRent: float = Form(...),
    files: Optional[List[UploadFile]] = None,
):
    """Analyze real estate risk with contextual documents and return structured risk insights."""

    property_block = format_property_info(
        property_id=propertyId,
        name=name,
        address=address,
        property_type=propertyType,
        floor=floor,
        built_year=builtYear,
        area=area,
        market_price=marketPrice,
        deposit=deposit,
        monthly_rent=monthlyRent,
    )
    attachments = await read_files(files)

    system_prompt = (
        "너는 부동산 리스크 분석 전문가다. "
        "입지, 건물 물리적 상태, 법적 리스크(권리, 인허가, 임대차), "
        "가격 및 수익성, 계약/운영 리스크를 종합 평가한다. "
        "응답은 아래 JSON 스키마만 사용하고, 마크다운·코드블록·백틱·주석 등 JSON 외 텍스트를 절대 포함하지 마라.\n"
        "{"
        "\"totalRisk\": 0~100 사이 정수,"
        "\"summary\": \"핵심 위험 요약(2문장 이내)\","
        "\"details\": ["
        "{"
        "\"title\": \"위험 항목 제목\","
        "\"content\": \"근거와 영향, 확인/완화 필요 조치\","
        "\"severity\": \"low|medium|high\""
        "}"
        "]"
        "}"
        "severity는 영향도와 시급성을 반영하여 high/medium/low 중 하나로만 표기한다. "
        "문장 앞에 불릿/번호를 붙이지 말고 JSON 외 텍스트를 추가하지 마라."
    )

    user_prompt = (
        "다음 매물 정보를 검토하고 위험도를 산출해라. "
        "첨부 문서 내용도 근거로 활용하라.\n\n"
        f"{property_block}\n\n"
        f"첨부 문서:\n{attachments}"
    )

    try:
        output = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
    except Exception as exc:
        return {"error": f"모델 서버 호출 실패: {exc}"}

    parsed = llm_client.parse_json(output)
    if parsed:
        return parsed
    return {"raw_output": output}
