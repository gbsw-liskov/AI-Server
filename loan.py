from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from llm_client import llm_client

router = APIRouter()


class LoanGuideRequest(BaseModel):
    age: int
    isHouseholder: bool
    familyType: str
    annualSalary: float
    monthlySalary: float
    incomeType: str
    incomeCategory: str
    rentalArea: str
    houseType: str
    rentalType: str
    deposit: float
    managementFee: float
    availableLoan: bool
    creditRating: str
    loanType: str
    overdueRecord: bool
    hasLeaseAgreement: bool
    confirmed: bool
    guideKeyword: Optional[str] = None
    guideUrls: Optional[List[str]] = None


def format_loan_profile(
    age: int,
    is_householder: bool,
    family_type: str,
    annual_salary: float,
    monthly_salary: float,
    income_type: str,
    income_category: str,
    rental_area: str,
    house_type: str,
    rental_type: str,
    deposit: float,
    management_fee: float,
    available_loan: bool,
    credit_rating: str,
    loan_type: str,
    overdue_record: bool,
    has_lease_agreement: bool,
    confirmed: bool,
) -> str:
    lines = [
        f"나이: {age}",
        f"세대주 여부: {'예' if is_householder else '아니오'}",
        f"가족 형태: {family_type}",
        f"연소득: {annual_salary}",
        f"월소득: {monthly_salary}",
        f"소득 유형: {income_type}",
        f"소득 구분: {income_category}",
        f"임차 지역: {rental_area}",
        f"주택 유형: {house_type}",
        f"임대 유형: {rental_type}",
        f"보증금: {deposit}",
        f"관리비: {management_fee}",
        f"대출 가능 여부: {'가능' if available_loan else '제한'}",
        f"신용 등급: {credit_rating}",
        f"희망 대출 유형: {loan_type}",
        f"연체 기록: {'있음' if overdue_record else '없음'}",
        f"임대차계약서 보유: {'예' if has_lease_agreement else '아니오'}",
        f"확정일자/전입신고 완료: {'예' if confirmed else '아니오'}",
    ]
    return "\n".join(lines)


@router.post("/loan")
async def recommend_loan(data: LoanGuideRequest):
    profile_block = format_loan_profile(
        age=data.age,
        is_householder=data.isHouseholder,
        family_type=data.familyType,
        annual_salary=data.annualSalary,
        monthly_salary=data.monthlySalary,
        income_type=data.incomeType,
        income_category=data.incomeCategory,
        rental_area=data.rentalArea,
        house_type=data.houseType,
        rental_type=data.rentalType,
        deposit=data.deposit,
        management_fee=data.managementFee,
        available_loan=data.availableLoan,
        credit_rating=data.creditRating,
        loan_type=data.loanType,
        overdue_record=data.overdueRecord,
        has_lease_agreement=data.hasLeaseAgreement,
        confirmed=data.confirmed,
    )

    guide_keyword = data.guideKeyword or "전세자금 대출 가이드"
    guide_urls = data.guideUrls or []
    guide_context = "\n".join(guide_urls) if guide_urls else "참고 문서 없음"

    system_prompt = (
        "너는 부동산 임차인을 위한 대출 가이드 전문 컨설턴트다. "
        "전세/월세, 신용/주택/전세 대출 가능성을 검토해 최적 조합과 비용을 제시한다. "
        "응답은 JSON만 반환하고 마크다운·불릿·백틱·설명 텍스트를 절대 포함하지 마라. "
        '{'
        "\"loanAmount\": number, "
        "\"interestRate\": number, "
        "\"ownCapital\": number, "
        "\"monthlyInterest\": number, "
        "\"managementFee\": number, "
        "\"totalMonthlyCost\": number, "
        "\"loans\": [{\"title\": \"string\", \"content\": \"string\"}], "
        "\"procedures\": [{\"title\": \"string\", \"content\": \"string\"}], "
        "\"channels\": [{\"title\": \"string\", \"content\": \"string\"}], "
        "\"advance\": [{\"title\": \"string\", \"content\": \"string\"}]"
        '} '
        "규칙: totalMonthlyCost = monthlyInterest + managementFee. "
        "loans/procedures/channels/advance는 각각 2~4개를 제공하고, 실행 단계를 명확히 적는다. "
        "수치는 원 단위 금액과 % 금리로 숫자만 기입한다."
    )

    user_prompt = (
        "다음 입주자/임대 조건에 맞춰 대출 가이드와 실행 계획을 작성해라. "
        "availableLoan이 false이거나 연체 기록이 있으면 보수적으로 한도를 낮추거나 대안(보증금 축소, 보증보험 활용 등)을 제시하라. "
        "보증금에서 loanAmount를 뺀 값을 ownCapital로 설정하고, monthlyInterest는 loanAmount*interestRate/12/100으로 근사하라. "
        "주거 형태/임대 유형에 맞는 상품명과 절차를 제시하라. "
        "가능하면 참고 문서 내용을 우선 반영하고, 없으면 일반 가이드를 제공해라.\n\n"
        f"{profile_block}\n\n"
        f"요청 키워드: {guide_keyword}\n"
        f"참고 링크:\n{guide_context}"
    )

    try:
        output = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.28,
        )
    except Exception as exc:
        return {"error": f"LLM 호출 실패: {exc}"}

    parsed = llm_client.parse_json(output)
    response_payload: Dict[str, Any] = parsed if parsed else {"raw_output": output}
    if guide_urls:
        response_payload["sources"] = guide_urls
    return response_payload
