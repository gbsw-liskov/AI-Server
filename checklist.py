from fastapi import APIRouter
from pydantic import BaseModel

from llm_client import llm_client, format_property_info

router = APIRouter()


class ChecklistRequest(BaseModel):
    propertyId: int
    name: str
    address: str
    propertyType: str
    floor: int
    buildYear: int
    area: str
    availableDate: str


@router.post("/checklist")
async def generate_checklist(data: ChecklistRequest):
    property_block = format_property_info(
        property_id=data.propertyId,
        name=data.name,
        address=data.address,
        property_type=data.propertyType,
        floor=data.floor,
        build_year=data.buildYear,
        area=data.area,
        available_date=data.availableDate,
    )

    system_prompt = (
        "너는 부동산 위험 사전점검 전문가다. "
        "입지/건물 상태/법적 리스크/가격 및 수익성/계약/관리 관점에서 "
        "구체적인 확인 질문을 만든다. "
        "응답은 JSON 형태로만 반환한다. "
        "스키마: {\"contents\": [\"질문 또는 체크포인트\", ...]}. "
        "불필요한 설명, 번호 매기기, 여는/닫는 텍스트를 넣지 말 것."
    )

    user_prompt = (
        "다음 매물 정보를 기반으로 반드시 필요한 체크리스트를 8~12개 작성해라.\n\n"
        f"{property_block}"
    )

    try:
        output = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.25,
        )
    except Exception as exc:
        return {"error": f"모델 서버 호출 실패: {exc}"}

    parsed = llm_client.parse_json(output)
    if isinstance(parsed, dict) and isinstance(parsed.get("contents"), list):
        return {"contents": parsed["contents"]}

    # Fallback when JSON parsing fails.
    fallback_items = [line.strip("- ").strip() for line in output.splitlines() if line.strip()]
    return {"contents": fallback_items}
