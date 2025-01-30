from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from agents.legal_agent import LegalAgent
from agents.context_manager import ContextManager
from langchain.vectorstores import VectorStore
from supabase import create_client, Client
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
import traceback

# 대화 상태 정의
class ConversationState(TypedDict):
    input: str
    context: List
    response: Optional[str]
    conversation_history: List[dict]  # 대화 기록 저장

class ConversationGraph:
    def __init__(self, config, vectorstore):
        self.legal_agent = LegalAgent(config)
        self.context_manager = ContextManager()
        self.vectorstore = vectorstore  # 벡터스토어 저장
        self.graph = self._build_graph()
        
        # Supabase 초기화
        load_dotenv()
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase URL 또는 API 키가 설정되지 않았습니다.")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
    def _build_graph(self):
        workflow = StateGraph(ConversationState)
        
        # 노드 정의
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("format_response", self._format_response)
        
        # 엣지 정의
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "format_response")
        
        # 진입점 및 종료점 설정
        workflow.set_entry_point("retrieve_context")
        workflow.set_finish_point("format_response")
        
        return workflow.compile()
        
    def _retrieve_context(self, state: ConversationState):
        # 벡터스토어에서 관련 문서 검색
        query = state["input"]
        docs = self.vectorstore.similarity_search(query, k=3)
        return {"input": query, "context": docs}
        
    def _generate_response(self, state: ConversationState):
        try:
            # OpenAI를 사용하여 응답 생성
            context = "\n".join(doc.page_content for doc in state["context"])
            response = self.legal_agent.get_executor().invoke({
                "input": f"질문: {state['input']}\n\n참고 문서:\n{context}"
            })
            
            print(f"Debug - Original response type: {type(response)}")  # 디버깅용
            print(f"Debug - Original response: {response}")  # 디버깅용
            
            # 응답 텍스트 추출
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            elif isinstance(response, dict):
                # 응답이 딕셔너리인 경우 content 필드 확인
                if 'content' in response:
                    response_text = response['content']
                elif 'text' in response:
                    response_text = response['text']
                else:
                    # 전체 응답을 문자열로 변환
                    response_text = str(response)
            else:
                response_text = str(response)
            
            # 응답 텍스트가 비어있는 경우 처리
            if not response_text:
                response_text = "죄송합니다. 응답을 생성하지 못했습니다."
            
            print(f"Debug - Final response text: {response_text}")  # 디버깅용
            
            # 문서 메타데이터 추출
            sources = []
            for doc in state["context"]:
                # 파일 경로에서 파일명만 추출
                source = doc.metadata.get('source', 'Unknown')
                if isinstance(source, str) and '/' in source:
                    source = source.split('/')[-1]
                
                sources.append({
                    "source": source,
                    "page": doc.metadata.get('page', '1')
                })
            
            # 대화 기록 업데이트
            conversation_history = state.get("conversation_history", [])
            conversation_history.append({
                "question": state["input"],
                "response": response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "response": response_text,
                "context": state["context"],
                "conversation_history": conversation_history,
                "sources": sources  # 출처 정보 추가
            }
        except Exception as e:
            print(f"Error in _generate_response: {str(e)}")
            print(f"Error type: {type(e)}")  # 디버깅용
            print(f"Error traceback: {traceback.format_exc()}")  # 디버깅용
            return {
                "response": "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.",
                "context": state["context"],
                "conversation_history": state.get("conversation_history", []),
                "sources": []  # 빈 출처 정보 추가
            }
        
    def _format_response(self, state: ConversationState):
        # 응답 포맷팅 로직
        return state
        
    def process_query(self, question: str, user_id: str):
        try:
            # 초기 상태 설정
            state = {
                "input": question,
                "context": [],
                "response": None,
                "conversation_history": []
            }
            
            # 그래프 실행
            result = self.graph.invoke(state)
            
            return {
                "response": result["response"],
                "context": result["context"],
                "conversation_history": result["conversation_history"]
            }
        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            return {
                "response": "Error occurred while processing the query.",
                "context": [],
                "conversation_history": []
            }
    
    def save_conversation(self, user_id: str, question: str, response: str):
        """대화 기록을 Supabase에 저장"""
        try:
            data, count = self.supabase.table('conversations').insert({
                'user_id': user_id,
                'question': question,
                'response': response,
                'created_at': datetime.now().isoformat()
            }).execute()
            
            if not data:
                print("Failed to save conversation")
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")
    
    def get_conversations(self, user_id: str):
        """특정 사용자의 대화 기록을 조회"""
        try:
            response = self.supabase.table('conversations')\
                .select('*')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .execute()
                
            if not response.data:
                return []
                
            return response.data
        except Exception as e:
            print(f"Error retrieving conversations: {str(e)}")
            return [] 