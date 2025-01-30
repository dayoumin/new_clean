from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph.conversation_graph import ConversationGraph
from utils.config_ import load_config
from utils.error_handler import error_handler
from utils.monitoring import monitoring
from supabase import create_client, Client
from utils.vectorstore import initialize_vectorstore, decrypt_vectorstore, load_vectorstore
import os, traceback
from dotenv import load_dotenv
from datetime import datetime
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from models.chat_models import ChatRequest
from fastapi.staticfiles import StaticFiles
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # 수정된 임포트
import json
import faiss
import numpy as np
from openai import OpenAI
from utils.vectorstore import decrypt_vectorstore  # 복호화 함수 임포트
from config import (
    OPENAI_API_KEY,
    SUPABASE_URL,
    SUPABASE_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    VECTORSTORE_PATH,
    VECTORSTORE_KEY_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATOR,
    LOADER_KWARGS,
    TEXT_SPLITTER_KWARGS,
    EMBEDDING_KWARGS,
    VECTOR_SEARCH_KWARGS,
    CONVERSATION_MEMORY_KWARGS,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    LLM_TEMPERATURE
)
from cryptography.fernet import Fernet
import asyncio
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (실제 배포 시 특정 도메인으로 제한)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모니터링 서버 시작
monitoring.start_metrics_server(port=8001)

# 설정 로드
config = load_config()

try:
    # 벡터스토어 복호화 시도
    if os.path.exists("faiss_vectorstore/index.enc"):
        print("암호화된 벡터스토어 파일을 복호화합니다.")
        decrypt_vectorstore("faiss_vectorstore", "vectorstore_key.key")
    
    # 벡터스토어 초기화
    vectorstore, _ = initialize_vectorstore(EMBEDDING_MODEL)
    print("벡터스토어 초기화 완료")
    
    # 대화 그래프 초기화 (벡터스토어 전달)
    conversation_graph = ConversationGraph(config, vectorstore)
except Exception as e:
    print(f"벡터스토어 초기화 실패: {str(e)}")
    raise

# Supabase 초기화
load_dotenv()

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL 또는 API 키가 설정되지 않았습니다.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 정적 파일 서빙 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# OpenAI 임베딩 초기화
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
)

# FAISS 인덱스 초기화
dimension = 1536  # OpenAI 임베딩 차원
index = faiss.IndexFlatL2(dimension)

@app.post("/api/chat")
@monitoring.track_request("chat")
async def chat(chat_request: ChatRequest):
    try:
        # 벡터스토어 초기화 확인
        if not hasattr(conversation_graph, 'process_query'):
            logger.error("Conversation graph not properly initialized")
            raise HTTPException(status_code=500, detail="Conversation graph not properly initialized")
        
        # 대화 그래프를 통해 처리
        response = conversation_graph.process_query(chat_request.question, chat_request.user_id)
        
        # 응답 데이터 유효성 검사
        if not response:
            logger.error("No response generated")
            raise HTTPException(status_code=500, detail="No response generated")
        
        return response
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/api/save-conversation")
async def save_conversation(chat_request: ChatRequest):
    try:
        # Supabase에 대화 기록 저장
        data, count = supabase.table('conversations').insert({
            'user_id': chat_request.user_id,
            'question': chat_request.question,
            'response': chat_request.response,
            'created_at': datetime.now().isoformat()
        }).execute()
        
        # 삽입된 데이터 확인
        if not data:
            raise HTTPException(status_code=400, detail="Failed to save conversation")
            
        return {"status": "success", "data": data[1]}
    except Exception as e:
        error_handler.log_error({
            "timestamp": datetime.now().isoformat(),
            "error": {
                "message": str(e),
                "stack": traceback.format_exc()
            }
        })
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/api/get-conversations/{user_id}")
async def get_conversations(user_id: str):
    try:
        # Supabase에서 대화 기록 조회 (최신순 정렬)
        response = supabase.table('conversations')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('created_at', desc=True)\
            .execute()
            
        if not response.data:
            return {"conversations": []}
            
        return {"conversations": response.data}
    except Exception as e:
        error_handler.log_error({
            "timestamp": datetime.now().isoformat(),
            "error": {
                "message": str(e),
                "stack": traceback.format_exc()
            }
        })
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/")
async def get():
    return FileResponse("static/index.html")

# API 상태 확인용 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Swagger UI는 자동으로 /docs에서 확인 가능

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 대화 기록을 저장할 리스트 초기화
    conversation_history = []
    
    try:
        # 벡터스토어 로드 (복호화/암호화 자동 처리)
        vector_store = load_vectorstore(embeddings)
        if not vector_store:
            raise Exception("벡터스토어 로드 실패")
        print("벡터스토어 로드 완료")
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"벡터스토어 로드 실패: {str(e)}"
        }))
        return
    
    while True:
        data = await websocket.receive_text()
        json_data = json.loads(data)
        question = json_data.get("question")
        
        # 유저 질문 전송
        await websocket.send_text(json.dumps({
            "type": "user_message",
            "content": question
        }))
        
        try:
            # 유사 문서 검색
            docs = vector_store.similarity_search(
                question, 
                k=VECTOR_SEARCH_KWARGS["k"],
                fetch_k=VECTOR_SEARCH_KWARGS["fetch_k"],
                score_threshold=VECTOR_SEARCH_KWARGS["score_threshold"]
            )
            
            if docs:
                context = "\n\n".join([doc.page_content for doc in docs])
                sources = [
                    f"📚 출처: {doc.metadata.get('source', '출처 정보 없음')}, "
                    f"📄 페이지: {doc.metadata.get('page', '페이지 정보 없음')}"
                    for doc in docs
                ]
                
                # 프롬프트 구성
                prompt = USER_PROMPT_TEMPLATE.format(
                    conversation_history="\n".join([f"{'사용자' if msg['role'] == 'user' else 'AI'}: {msg['content']}" for msg in conversation_history[-4:]]),
                    question=question,
                    context=context
                )

                # OpenAI API 호출
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *conversation_history[-4:],
                        {"role": "user", "content": prompt}
                    ],
                    temperature=LLM_TEMPERATURE,  # LLM_TEMPERATURE 사용
                    stream=True
                )
                
                # 스트리밍 방식으로 답변 전송
                ai_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        ai_response += content
                        await websocket.send_text(json.dumps({
                            "type": "ai_message",
                            "content": content
                        }))
                        await asyncio.sleep(0.01)
                
                # 완성된 AI 응답을 대화 기록에 추가
                conversation_history.append({"role": "assistant", "content": ai_response})
                
                # 출처 정보 전송
                await websocket.send_text(json.dumps({
                    "type": "ai_message",
                    "content": f"<br><br>참고한 문서:<br>" + "<br>".join(sources)
                }))
                
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "관련 문서를 찾을 수 없습니다."
                }))
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"답변 생성 중 오류 발생: {str(e)}"
            }))

def generate_key(key_path):
    """암호화 키 생성 및 저장"""
    if not os.path.exists(key_path):
        key = Fernet.generate_key()
        with open(key_path, 'wb') as key_file:
            key_file.write(key)
    with open(key_path, 'rb') as key_file:
        return key_file.read()

def encrypt_vectorstore(vectorstore_path, key_path):
    """벡터스토어 암호화"""
    key = generate_key(key_path)
    f = Fernet(key)
    
    # index.faiss 파일 암호화
    faiss_path = os.path.join(vectorstore_path, "index.faiss")
    if os.path.exists(faiss_path):
        with open(faiss_path, 'rb') as file:
            encrypted_data = f.encrypt(file.read())
        with open(faiss_path + '.enc', 'wb') as file:
            file.write(encrypted_data)
        os.remove(faiss_path)
    
    # index.pkl 파일 암호화
    pkl_path = os.path.join(vectorstore_path, "index.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as file:
            encrypted_data = f.encrypt(file.read())
        with open(pkl_path + '.enc', 'wb') as file:
            file.write(encrypted_data)
        os.remove(pkl_path)

def decrypt_vectorstore(vectorstore_path, key_path):
    """벡터스토어 복호화"""
    key = generate_key(key_path)
    f = Fernet(key)
    
    # index.faiss.enc 파일 복호화
    faiss_path = os.path.join(vectorstore_path, "index.faiss.enc")
    if os.path.exists(faiss_path):
        with open(faiss_path, 'rb') as file:
            decrypted_data = f.decrypt(file.read())
        with open(faiss_path[:-4], 'wb') as file:
            file.write(decrypted_data)
        os.remove(faiss_path)
    
    # index.pkl.enc 파일 복호화
    pkl_path = os.path.join(vectorstore_path, "index.pkl.enc")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as file:
            decrypted_data = f.decrypt(file.read())
        with open(pkl_path[:-4], 'wb') as file:
            file.write(decrypted_data)
        os.remove(pkl_path)

def initialize_vectorstore(embedding_model):
    """서버에서 벡터스토어 초기화"""
    try:
        embeddings = OpenAIEmbeddings(
            model=embedding_model, 
            openai_api_key=OPENAI_API_KEY
        )
        
        if os.path.exists(VECTORSTORE_PATH):
            vectorstore = load_vectorstore(embeddings)
        else:
            vectorstore = FAISS.from_texts([""], embeddings)
        
        return vectorstore, "hash_file_path"
    except Exception as e:
        logger.error(f"벡터스토어 초기화 실패: {str(e)}")
        return None, None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 