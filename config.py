import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import time
import shutil
from langchain.vectorstores import FAISS

# .env 파일 로드
load_dotenv()

# API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 모델 설정
LLM_MODEL = "gpt-4o-mini"  
LLM_TEMPERATURE = 0.0

# gpt-4o-mini
EMBEDDING_MODEL = "text-embedding-3-large"
# text-embedding-3-large

EMBEDDING_DIMENSION = 3072
# 3072


# 벡터스토어 설정
VECTORSTORE_PATH = "faiss_vectorstore"
VECTORSTORE_KEY_PATH = "vectorstore_key.key"

# 문서 처리 설정
CHUNK_SIZE = 1000  # 청크 크기 조정 (기존 2000에서 1000으로 줄임)
CHUNK_OVERLAP = 200  # 청크 간 중복 크기
SEPARATOR = "\n"  # 청크 구분자

# 문서 로더 설정
LOADER_KWARGS = {
    "pdf": {
        "extract_images": False,
    },
    "docx": {
        "mode": "elements",
    },
    "txt": {
        "encoding": "utf-8",
    }
}

# 텍스트 분할 설정
TEXT_SPLITTER_KWARGS = {
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "separators": [SEPARATOR],
    "length_function": len,
    "add_start_index": True,
}

# 임베딩 설정
EMBEDDING_KWARGS = {
    "model": EMBEDDING_MODEL,
    "dimensions": EMBEDDING_DIMENSION,
}

# 벡터 검색 설정
VECTOR_SEARCH_KWARGS = {
    "k": 5,  # 검색할 유사 문서 수
    "fetch_k": 10,  # 초기 검색 문서 수
    "score_threshold": 0.5,  # 유사도 임계값
}

# 대화 기록 설정
CONVERSATION_MEMORY_KWARGS = {
    "max_history": 10,  # 유지할 최대 대화 기록 수
    "return_messages": True,  # 메시지 형태로 반환
}

# 프롬프트 설정
SYSTEM_PROMPT = """당신은 법률 문서를 기반으로 답변하는 전문가입니다. 
이전 대화 맥락을 이해하고 문서 내용을 바탕으로 정확하고 객관적인 답변을 제공합니다.
답변은 다음과 같은 원칙을 따릅니다:

1. 문서의 내용에 충실하게 답변합니다.
2. 불확실한 내용은 명시적으로 언급합니다.
3. 필요한 경우 추가 설명이나 예시를 제공합니다.
4. 전문 용어는 가능한 쉽게 설명합니다.
"""

USER_PROMPT_TEMPLATE = """이전 대화 기록과 문서 내용을 참고하여 답변해주세요.

이전 대화:
{conversation_history}

현재 질문: {question}

참고 문서 내용:
{context}

답변은 이전 대화 맥락과 문서 내용을 모두 고려하여 작성해주세요."""

# 벡터스토어 디렉토리가 없으면 생성
if not os.path.exists(VECTORSTORE_PATH):
    os.makedirs(VECTORSTORE_PATH)

# 기타 설정
LOG_LEVEL = "INFO"  # 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
