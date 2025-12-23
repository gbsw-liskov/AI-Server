from typing import Any, List, Optional

from fastapi import APIRouter, Form, UploadFile

from llm_client import llm_client, format_property_info, read_files

router = APIRouter()


@router.post("/solution")
async def propose_solution(
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
    totalRisk: float = Form(...),
    summary: str = Form(...),
    details: str = Form(...),
    files: Optional[List[UploadFile]] = None,
):
    """
    Generate tailored mitigation and follow-up checklists based on prior risk analysis.
    `details` expects a JSON string from the analyze API.
    """

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

    parsed_details: Any = llm_client.parse_json(details) or details

    system_prompt = (
        "너는 부동산 컨설팅 전문가다. "
        "주어진 위험 요약을 바탕으로 실행 가능한 대처 방안과 확인 체크리스트를 제안한다. "
        "응답은 JSON만 반환하고 다른 텍스트, 마크다운, 코드블록, 백틱을 절대 포함하지 마라.\n"
        "{"
        "\"coping\": ["
        "{"
        "\"title\": \"대처 전략 제목\","
        "\"actions\": [\"구체적 실행 단계\"]"
        "}"
        "],"
        "\"checklist\": [\"후속 확인 항목\"]"
        "}"
        "actions는 2~5개의 짧은 단계로 작성하며, 바로 실행할 수 있게 작성한다."
    )

    user_prompt = (
        "다음 매물의 위험 요약과 세부 내용을 바탕으로 맞춤형 대처 방안을 제안해라. "
        "첨부 문서에서 근거가 보이면 반영하라.\n\n"
        f"{property_block}\n\n"
        f"총 위험도: {totalRisk}\n"
        f"요약: {summary}\n"
        f"세부 위험: {parsed_details}\n\n"
        f"첨부 문서:\n{attachments}"
    )

    try:
        output = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
    except Exception as exc:
        return {"error": f"모델 서버 호출 실패: {exc}"}

    parsed = llm_client.parse_json(output)
    if parsed:
        return parsed
    return {"raw_output": output}
