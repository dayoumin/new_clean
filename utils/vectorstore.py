import os
import shutil
from typing import List, Optional, Tuple
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from cryptography.fernet import Fernet
from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    VECTORSTORE_KEY_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATOR,
    TEXT_SPLITTER_KWARGS
)
from pathlib import Path
from utils.encryption import FileEncryptor
import logging
import time

# 로거 초기화
logger = logging.getLogger(__name__)

def generate_key(key_path: str = VECTORSTORE_KEY_PATH) -> bytes:
    """암호화 키 생성 및 저장"""
    if not os.path.exists(key_path):
        key = Fernet.generate_key()
        with open(key_path, "wb") as key_file:
            key_file.write(key)
        print(f"새로운 암호화 키 생성 및 저장: {key_path}")
    else:
        with open(key_path, "rb") as key_file:
            key = key_file.read()
        print(f"기존 암호화 키 로드: {key_path}")
    return key

def encrypt_file(file_path: str, key: bytes) -> None:
    """파일 암호화"""
    fernet = Fernet(key)
    with open(file_path, "rb") as file:
        original = file.read()
    encrypted = fernet.encrypt(original)
    with open(file_path, "wb") as encrypted_file:
        encrypted_file.write(encrypted)
    print(f"파일 암호화 완료: {file_path}")

def decrypt_file(file_path: str, key: bytes) -> None:
    """파일 복호화"""
    fernet = Fernet(key)
    with open(file_path, "rb") as encrypted_file:
        encrypted = encrypted_file.read()
    decrypted = fernet.decrypt(encrypted)
    with open(file_path, "wb") as decrypted_file:
        decrypted_file.write(decrypted)
    print(f"파일 복호화 완료: {file_path}")

def create_vectorstore(
    documents: List[Document],
    embedding_model: Optional[OpenAIEmbeddings] = None,
    batch_size: int = 100
) -> Optional[FAISS]:
    """문서를 배치 단위로 처리하여 벡터스토어 생성"""
    try:
        if embedding_model is None:
            embedding_model = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY,
                chunk_size=500
            )
        
        vectorstore = None
        total_docs = len(documents)
        logger.info(f"총 {total_docs}개의 문서를 처리합니다.")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"배치 처리 중: {i+1}-{min(i + batch_size, total_docs)} / {total_docs}")
            
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embedding_model)
                else:
                    temp_store = FAISS.from_documents(batch, embedding_model)
                    vectorstore.merge_from(temp_store)
                
                if i + batch_size < total_docs:
                    time.sleep(1)
                
            except Exception as batch_error:
                logger.error(f"배치 처리 중 오류 발생: {str(batch_error)}")
                continue
        
        if vectorstore:
            save_and_encrypt_vectorstore(vectorstore)
        
        return vectorstore
    except Exception as e:
        logger.error(f"벡터스토어 생성 중 오류 발생: {str(e)}")
        return None

def save_and_encrypt_vectorstore(vectorstore: FAISS) -> bool:
    """벡터스토어 저장 및 암호화"""
    try:
        logger.info("벡터스토어 저장 시작")
        if not os.path.exists(VECTORSTORE_PATH):
            os.makedirs(VECTORSTORE_PATH)
        vectorstore.save_local(VECTORSTORE_PATH)
        
        logger.info("벡터스토어 암호화 시작")
        key = generate_key()
        encrypt_vectorstore_files(key)
        
        logger.info("벡터스토어 저장 및 암호화 완료")
        return True
    except Exception as e:
        logger.error(f"벡터스토어 저장/암호화 중 오류: {str(e)}")
        return False

def load_vectorstore(embedding_model: OpenAIEmbeddings) -> Optional[FAISS]:
    """벡터스토어 로드"""
    try:
        if os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss.enc")):
            decrypt_vectorstore(VECTORSTORE_PATH, VECTORSTORE_KEY_PATH)
            
        faiss_path = os.path.join(VECTORSTORE_PATH, "index.faiss")
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"벡터스토어 파일이 없습니다: {faiss_path}")
        
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        
        logger.info("벡터스토어 로드 완료")
        return vectorstore
    except Exception as e:
        logger.error(f"벡터스토어 로드 실패: {str(e)}")
        return None

def decrypt_vectorstore_files(key: bytes) -> None:
    """벡터스토어 파일들 복호화"""
    for root, dirs, files in os.walk(VECTORSTORE_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            decrypt_file(file_path, key)

def encrypt_vectorstore_files(key: bytes) -> None:
    """벡터스토어 파일들 암호화"""
    for root, dirs, files in os.walk(VECTORSTORE_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            encrypt_file(file_path, key)

def reset_vectorstore() -> bool:
    """기존 벡터스토어 삭제"""
    try:
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
            logger.info("기존 벡터스토어 삭제 완료")
        return True
    except Exception as e:
        logger.error(f"벡터스토어 삭제 중 오류: {str(e)}")
        return False

def get_similar_documents(
    query: str,
    vectorstore: FAISS,
    k: int = 3
) -> List[Document]:
    """유사 문서 검색"""
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"문서 검색 중 오류 발생: {str(e)}")
        return [] 

def encrypt_vectorstore(vectorstore_path: str, key_path: str) -> bool:
    """벡터스토어 암호화"""
    try:
        # 키 파일이 없으면 생성
        if not os.path.exists(key_path):
            key = Fernet.generate_key()
            with open(key_path, 'wb') as key_file:
                key_file.write(key)
            print(f"새로운 암호화 키 생성: {key_path}")
        
        # 키 로드
        with open(key_path, 'rb') as key_file:
            key = key_file.read()
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
        
        print("벡터스토어 암호화 완료")
        return True
    except Exception as e:
        print(f"벡터스토어 암호화 중 오류 발생: {str(e)}")
        return False

def decrypt_vectorstore(vectorstore_path: str, key_path: str) -> bool:
    """벡터스토어 복호화"""
    try:
        # 키 로드
        with open(key_path, 'rb') as key_file:
            key = key_file.read()
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
        
        print("벡터스토어 복호화 완료")
        return True
    except Exception as e:
        print(f"벡터스토어 복호화 중 오류 발생: {str(e)}")
        return False

def initialize_vectorstore(
    embedding_model: Optional[OpenAIEmbeddings] = None
) -> Tuple[Optional[FAISS], str]:
    """벡터스토어 초기화 함수"""
    try:
        if embedding_model is None:
            embedding_model = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
        
        # 벡터스토어 로드 또는 생성
        if os.path.exists(VECTORSTORE_PATH):
            logger.info("기존 벡터스토어를 로드합니다.")
            vectorstore = load_vectorstore(embedding_model)
        else:
            logger.info("벡터스토어가 없습니다. 새로 생성합니다.")
            # 기본 문서로 벡터스토어 초기화
            documents = process_document("documents/default.txt")
            if not documents:
                raise Exception("Error loading documents\\default.txt")
            logger.info(f"로드된 문서 수: {len(documents)}")
            vectorstore = create_vectorstore(documents, embedding_model)
        
        return vectorstore, "hash_file_path"  # 해시 파일 경로는 임시로 반환
        
    except Exception as e:
        logger.error(f"벡터스토어 초기화 실패: {str(e)}")
        return None, None