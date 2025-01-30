import os
from pathlib import Path
import fitz  # PyMuPDF
from cryptography.fernet import Fernet
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# 루트의 config.py를 임포트
from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    VECTORSTORE_KEY_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATOR,
    LOADER_KWARGS,
    TEXT_SPLITTER_KWARGS
)
from utils.vectorstore import create_vectorstore, load_vectorstore, reset_vectorstore
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def generate_key(key_path):
    """암호화 키 생성 및 저장"""
    try:
        # 키 파일이 저장될 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        
        if not os.path.exists(key_path):
            print(f"새로운 암호화 키를 생성합니다: {key_path}")
            key = Fernet.generate_key()
            with open(key_path, 'wb') as key_file:
                key_file.write(key)
            print("암호화 키가 생성되었습니다.")
        
        with open(key_path, 'rb') as key_file:
            return key_file.read()
    except Exception as e:
        print(f"키 생성/로드 중 오류 발생: {str(e)}")
        raise

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
    try:
        print(f"벡터스토어 복호화 시작: {vectorstore_path}")
        
        key = generate_key(key_path)
        f = Fernet(key)
        
        # index.faiss.enc 파일 복호화
        faiss_path = os.path.join(vectorstore_path, "index.faiss.enc")
        if os.path.exists(faiss_path):
            print(f"복호화 중: {faiss_path}")
            with open(faiss_path, 'rb') as file:
                decrypted_data = f.decrypt(file.read())
            output_path = faiss_path[:-4]  # .enc 확장자 제거
            with open(output_path, 'wb') as file:
                file.write(decrypted_data)
            os.remove(faiss_path)
            print(f"복호화 완료: {output_path}")
        else:
            print(f"파일 없음: {faiss_path}")
        
        # index.pkl.enc 파일 복호화
        pkl_path = os.path.join(vectorstore_path, "index.pkl.enc")
        if os.path.exists(pkl_path):
            print(f"복호화 중: {pkl_path}")
            with open(pkl_path, 'rb') as file:
                decrypted_data = f.decrypt(file.read())
            output_path = pkl_path[:-4]  # .enc 확장자 제거
            with open(output_path, 'wb') as file:
                file.write(decrypted_data)
            os.remove(pkl_path)
            print(f"복호화 완료: {output_path}")
        else:
            print(f"파일 없음: {pkl_path}")
        
        print("벡터스토어 복호화 완료")
    except Exception as e:
        print(f"벡터스토어 복호화 중 오류 발생: {str(e)}")

def create_vector_store():
    """로컬 환경에서 벡터스토어 생성"""
    try:
        documents = load_documents()
        if not documents:
            print("처리할 문서가 없습니다.")
            return None
            
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        
        vectorstore = create_vectorstore(documents, embeddings)
        if vectorstore:
            print("벡터스토어 생성, 저장 및 암호화 완료")
            return vectorstore
        return None
    except Exception as e:
        print(f"벡터스토어 생성 중 오류 발생: {str(e)}")
        return None

def load_vector_store():
    """로컬 환경에서 벡터스토어 로드"""
    try:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        
        vectorstore = load_vectorstore(embeddings)
        if vectorstore:
            print("벡터스토어 로드 완료")
            return vectorstore
        return None
    except Exception as e:
        print(f"벡터스토어 로드 중 오류 발생: {str(e)}")
        return None

def remove_document_from_vectorstore(document_name):
    """벡터스토어에서 특정 문서를 제거하는 함수"""
    try:
        print(f"\n'{document_name}' 문서를 벡터스토어에서 제거합니다...")
        
        # 임베딩 모델 초기화
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        
        # 기존 벡터스토어 로드 (암호화된 경우 복호화)
        if os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss.enc")):
            decrypt_vectorstore(VECTORSTORE_PATH, VECTORSTORE_KEY_PATH)
            
        vector_store = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # 문서 이름으로 필터링
        filtered_docs = []
        filtered_metadatas = []
        removed_pages = 0
        
        # 기존 문서들을 순회하면서 제거할 문서를 제외
        docstore = vector_store.docstore._dict
        for doc_id, doc in docstore.items():
            if doc.metadata["source"] != document_name:
                filtered_docs.append(doc.page_content)
                filtered_metadatas.append(doc.metadata)
            else:
                removed_pages += 1
        
        if removed_pages == 0:
            print(f"'{document_name}' 문서를 찾을 수 없습니다.")
            # 다시 암호화
            encrypt_vectorstore(VECTORSTORE_PATH, VECTORSTORE_KEY_PATH)
            return vector_store
            
        # 새로운 벡터스토어 생성
        new_vector_store = FAISS.from_texts(
            texts=filtered_docs,
            embedding=embeddings,
            metadatas=filtered_metadatas
        )
        
        # 새로운 벡터스토어 저장 및 암호화
        new_vector_store.save_local(VECTORSTORE_PATH)
        encrypt_vectorstore(VECTORSTORE_PATH, VECTORSTORE_KEY_PATH)
        
        print(f"문서 '{document_name}'가 벡터스토어에서 제거되었습니다.")
        print(f"제거된 페이지 수: {removed_pages}")
        return new_vector_store
        
    except Exception as e:
        print(f"문서 제거 중 오류 발생: {str(e)}")
        return None

def add_document_to_vectorstore(document_name):
    """특정 문서만 벡터스토어에 추가하는 함수"""
    try:
        print(f"\n'{document_name}' 문서를 벡터스토어에 추가합니다...")
        
        # 임베딩 모델 초기화
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        
        # 기존 벡터스토어 로드 (암호화된 경우 복호화)
        existing_vectorstore = None
        if os.path.exists(VECTORSTORE_PATH):
            if os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss.enc")):
                decrypt_vectorstore(VECTORSTORE_PATH, VECTORSTORE_KEY_PATH)
            existing_vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        
        # 새로운 문서 처리
        documents = []
        metadatas = []
        file_path = os.path.join(os.getcwd(), "documents", document_name)
        
        if not os.path.exists(file_path):
            print(f"오류: '{document_name}' 파일을 찾을 수 없습니다.")
            return None
            
        try:
            # PyMuPDF로 PDF 파일 열기
            with fitz.open(file_path) as pdf_doc:
                # 각 페이지 처리
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    text = page.get_text()
                    
                    if text.strip():  # 빈 페이지가 아닌 경우만 추가
                        documents.append(text)
                        metadatas.append({
                            "source": document_name,
                            "page": page_num + 1,
                            "total_pages": len(pdf_doc)
                        })
                
        except Exception as e:
            print(f"PDF 파일 읽기 오류: {document_name}, {str(e)}")
            return None
        
        if not documents:
            print("처리할 내용이 없습니다.")
            return None
            
        print(f"\n총 {len(documents)}개의 페이지를 처리했습니다.")
        
        # 새로운 벡터스토어 생성
        new_vectorstore = FAISS.from_texts(
            texts=documents,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        # 기존 벡터스토어와 병합
        if existing_vectorstore:
            existing_vectorstore.merge_from(new_vectorstore)
            final_vectorstore = existing_vectorstore
        else:
            final_vectorstore = new_vectorstore
        
        # 벡터스토어 저장 및 암호화
        final_vectorstore.save_local(VECTORSTORE_PATH)
        encrypt_vectorstore(VECTORSTORE_PATH, VECTORSTORE_KEY_PATH)
        print(f"문서 '{document_name}'가 벡터스토어에 추가되고 암호화되었습니다.")
        
        return final_vectorstore
        
    except Exception as e:
        print(f"문서 추가 중 오류 발생: {str(e)}")
        return None

def extract_document_info(filename):
    """
    파일명에서 문서 정보를 추출하는 함수
    예시 파일명: "민법.pdf" -> 문서명: "민법", 페이지: None
    """
    # 파일명에서 확장자 제거
    filename = filename.split(".")[0]
    
    # 문서명 추출 (페이지 번호는 파일명에 없으므로 None으로 설정)
    doc_name = filename  # 문서명 (예: "민법")
    return doc_name, None  # 페이지 번호는 None

def list_documents():
    """벡터스토어에 저장된 문서 리스트를 출력하는 함수"""
    try:
        print("\n벡터스토어에 저장된 문서 리스트를 출력합니다...")
        
        # 벡터스토어 파일 존재 여부 확인
        if not os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss.enc")):
            print("벡터스토어 파일이 없습니다. 먼저 벡터스토어를 생성해주세요.")
            return
        
        # 암호화 키 파일 확인
        if not os.path.exists(VECTORSTORE_KEY_PATH):
            print("암호화 키 파일이 없습니다. 새로 생성합니다.")
            generate_key(VECTORSTORE_KEY_PATH)
        
        # 임베딩 모델 초기화
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        
        # 기존 벡터스토어 로드 (암호화된 경우 복호화)
        decrypt_vectorstore(VECTORSTORE_PATH, VECTORSTORE_KEY_PATH)
            
        vector_store = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # 문서 리스트 수집
        document_list = {}
        docstore = vector_store.docstore._dict
        for doc_id, doc in docstore.items():
            source = doc.metadata["source"]
            if source not in document_list:
                document_list[source] = {
                    "total_pages": doc.metadata["total_pages"],
                    "pages": []
                }
            document_list[source]["pages"].append(doc.metadata["page"])
        
        # 문서 리스트 출력
        if not document_list:
            print("벡터스토어에 저장된 문서가 없습니다.")
            return
            
        print("\n[저장된 문서 리스트]")
        for doc_name, info in document_list.items():
            print(f"\n문서명: {doc_name}")
            print(f"- 총 페이지 수: {info['total_pages']}")
            print(f"- 처리된 페이지: {sorted(info['pages'])}")
        
        # 다시 암호화
        encrypt_vectorstore(VECTORSTORE_PATH, VECTORSTORE_KEY_PATH)
        
    except Exception as e:
        print(f"문서 리스트 출력 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()  # 상세한 오류 정보 출력

def process_document(file_path):
    from langchain.document_loaders import PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader = PyMuPDFLoader(file_path, **LOADER_KWARGS["pdf"])
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[SEPARATOR],
        **TEXT_SPLITTER_KWARGS
    )
    return text_splitter.split_documents(documents)

def process_documents() -> bool:
    """문서를 처리하여 벡터스토어 생성"""
    try:
        # 기존 벡터스토어 삭제
        reset_vectorstore()
        
        # 문서 로드
        documents = load_documents()
        if not documents:
            print("로드할 문서가 없습니다.")
            return False
        
        # 텍스트 분할기 초기화
        text_splitter = RecursiveCharacterTextSplitter(
            **TEXT_SPLITTER_KWARGS  # chunk_size, chunk_overlap, separators는 TEXT_SPLITTER_KWARGS에 포함됨
        )
        
        # 문서를 청크로 분할
        chunks = text_splitter.split_documents(documents)
        print(f"총 {len(chunks)}개의 청크로 분할되었습니다.")
        
        # 벡터스토어 생성
        vectorstore = create_vectorstore(chunks)
        if vectorstore:
            print("벡터스토어 생성 완료")
            return True
        else:
            print("벡터스토어 생성 실패")
            return False
            
    except Exception as e:
        print(f"문서 처리 중 오류 발생: {str(e)}")
        return False

def load_documents() -> List[Document]:
    """문서 디렉토리에서 모든 문서를 로드"""
    try:
        documents_dir = Path("documents")
        documents = []
        
        # PDF 파일 처리
        for file_name in os.listdir(documents_dir):
            if file_name.lower().endswith(".pdf"):
                file_path = os.path.join(documents_dir, file_name)
                print(f"\n파일 처리 중: {file_name}")
                
                try:
                    # PyMuPDF로 PDF 파일 열기
                    with fitz.open(file_path) as pdf_doc:
                        # 각 페이지 처리
                        for page_num in range(len(pdf_doc)):
                            page = pdf_doc[page_num]
                            text = page.get_text()
                            
                            if text.strip():  # 빈 페이지가 아닌 경우만 추가
                                documents.append(Document(
                                    page_content=text,
                                    metadata={
                                        "source": file_name,
                                        "page": page_num + 1,
                                        "total_pages": len(pdf_doc)
                                    }
                                ))
                except Exception as e:
                    print(f"파일 처리 중 오류 발생: {file_name} - {str(e)}")
        
        print(f"총 {len(documents)}개의 문서를 로드했습니다.")
        return documents
    
    except Exception as e:
        print(f"문서 로드 중 오류 발생: {str(e)}")
        return []

def main():
    # documents 디렉토리 확인 및 생성
    documents_dir = Path("documents")
    if not documents_dir.exists():
        documents_dir.mkdir(parents=True)
        print("documents 디렉토리를 생성했습니다.")
    
    # 명령행 인자로 작업 선택
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "remove" and len(sys.argv) > 2:
            # 문서 제거 명령어: python app_local.py remove "document_name.pdf"
            remove_document_from_vectorstore(sys.argv[2])
        elif sys.argv[1] == "add" and len(sys.argv) > 2:
            # 문서 추가 명령어: python app_local.py add "document_name.pdf"
            add_document_to_vectorstore(sys.argv[2])
        elif sys.argv[1] == "list":
            # 문서 리스트 출력: python app_local.py list
            list_documents()
        else:
            print("잘못된 명령어입니다. 사용법:")
            print("  - 문서 추가: python app_local.py add <문서명.pdf>")
            print("  - 문서 제거: python app_local.py remove <문서명.pdf>")
            print("  - 문서 리스트: python app_local.py list")
    else:
        # 기본: 벡터스토어 생성
        process_documents()

if __name__ == "__main__":
    main() 