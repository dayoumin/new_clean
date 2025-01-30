from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str  # 필수 필드
    user_id: str   # 필수 필드
    response: str | None = None  # 선택적 필드 (기본값: None) 