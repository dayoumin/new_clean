class ChatApp {
  constructor() {
    this.userId = this.generateUserId();
    this.chatWindow = document.getElementById('chat-window');
    this.chatInput = document.getElementById('chat-input');
    this.sendBtn = document.getElementById('send-btn');
    
    this.initEventListeners();
    this.initErrorHandling();
  }

  initEventListeners() {
    this.sendBtn.addEventListener('click', () => this.handleSendMessage());
    this.chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') this.handleSendMessage();
    });
  }

  initErrorHandling() {
    window.onerror = (message, source, lineno, colno, error) => {
      this.logError({
        message,
        source,
        lineno,
        colno,
        stack: error?.stack
      });
      this.displayErrorMessage('시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
      return true;
    };
  }

  generateUserId() {
    return 'user_' + Math.random().toString(36).substr(2, 9);
  }

  async handleSendMessage() {
    const question = this.chatInput.value.trim();
    if (!question) return;

    this.addMessage('user', question);
    this.chatInput.value = '';

    try {
      const response = await this.sendToBackend({
        question,
        user_id: this.userId
      });
      this.addMessage('assistant', response.response, response.sources || []);
      await this.saveConversation(this.userId, question, response.response);
    } catch (error) {
      this.logError(error);
      this.displayErrorMessage('답변 생성 중 오류가 발생했습니다.');
    }
  }

  addMessage(role, content, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    messageDiv.textContent = content;

    if (sources.length > 0) {
      const sourcesDiv = document.createElement('div');
      sourcesDiv.className = 'sources';
      sourcesDiv.innerHTML = `<strong>참고 문서:</strong><br>` +
        sources.map(source => `- ${source.source} (페이지 ${source.page})`).join('<br>');
      messageDiv.appendChild(sourcesDiv);
    }

    this.chatWindow.appendChild(messageDiv);
    this.chatWindow.scrollTop = this.chatWindow.scrollHeight;
  }

  async sendToBackend(data) {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  displayErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message error-message';
    errorDiv.textContent = message;
    this.chatWindow.appendChild(errorDiv);
  }

  logError(error) {
    const errorData = {
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? {
        message: error.message,
        stack: error.stack
      } : error
    };

    fetch('/api/log-error', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(errorData)
    });
  }

  async loadConversationHistory() {
    try {
      const conversations = await this.getConversations(this.userId);
      
      conversations.reverse().forEach(conv => {
        this.addMessage(conv.role, conv.content);
      });
    } catch (error) {
      this.logError(error);
    }
  }

  async saveConversation(userId, question, response) {
    await fetch("/api/save-conversation", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ user_id: userId, question, response }),
    });
  }

  async getConversations(userId) {
    const response = await fetch(`/api/get-conversations/${userId}`);
    return response.json();
  }
}

const chatApp = new ChatApp();
chatApp.loadConversationHistory(); 