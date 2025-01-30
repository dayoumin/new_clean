from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class LegalAgent:
    def __init__(self, config):
        self.llm = ChatOpenAI(
            model_name=config["model_name"],
            temperature=0,
            streaming=True,
            openai_api_key=config["api_keys"]["openai"]
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("user", "{input}")
        ])
        
    def _get_system_prompt(self):
        return """당신은 한국의 최고 법률 전문가입니다. 다음 지침을 반드시 준수하여 답변하세요:
        1. 법률적인 질문에 대해 정확하고 명확하게 답변
        2. 관련 법적 개념과 구체적인 법적 근거 제공
        3. 「법률명」 제X조 제X항 형식으로 정확한 출처 표시
        4. 결론과 참고 법령을 명시"""
        
    def get_executor(self):
        return self.prompt | self.llm 