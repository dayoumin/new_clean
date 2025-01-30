from langchain_community.chat_message_histories import ChatMessageHistory

class ContextManager:
    def __init__(self):
        self.memory = ChatMessageHistory()
        
    def get_memory(self):
        return self.memory 