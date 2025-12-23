import json
import os
from typing import Any, Dict, List, Optional

import torch
from fastapi import UploadFile
from transformers import AutoModelForCausalLM, AutoTokenizer

# 환경변수 MODEL_NAME, MAX_NEW_TOKENS 등으로 조정 가능
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))


class LLMClient:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    def _load(self):
        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.35,
        top_p: float = 0.9,
    ) -> str:
        self._load()
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @staticmethod
    def parse_json(text: str) -> Optional[Any]:
        try:
            return json.loads(text)
        except Exception:
            return None


async def read_files(files: Optional[List[UploadFile]]) -> str:
    """Read uploaded files as text and label them; tolerate encoding issues."""
    if not files:
        return "첨부 문서 없음"

    chunks: List[str] = []
    for file in files:
        try:
            content = (await file.read()).decode("utf-8", errors="ignore")
            chunks.append(f"[{file.filename}] 내용:\n{content}")
        except Exception:
            chunks.append(f"[{file.filename}] 파일을 읽을 수 없습니다.")
    return "\n\n".join(chunks)


def format_property_info(
    property_id: int,
    name: str,
    address: str,
    property_type: str,
    floor: int,
    built_year: int,
    area: int,
    market_price: Optional[float] = None,
    deposit: Optional[float] = None,
    monthly_rent: Optional[float] = None,
) -> str:
    """Create a compact property summary for prompts."""
    lines = [
        f"매물 ID: {property_id}",
        f"이름: {name}",
        f"주소: {address}",
        f"유형: {property_type}",
        f"층수: {floor}",
        f"준공연도: {built_year}",
        f"면적: {area}",
    ]
    if market_price is not None:
        lines.append(f"시세: {market_price}")
    if deposit is not None:
        lines.append(f"보증금: {deposit}")
    if monthly_rent is not None:
        lines.append(f"월세: {monthly_rent}")
    return "\n".join(lines)


llm_client = LLMClient()
