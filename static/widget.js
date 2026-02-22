/**
 * Web chat widget za Smart chatbot
 * Može se integrisati na bilo koji sajt
 */

class SmartChatWidget {
    constructor(config) {
        this.config = {
            apiUrl: 'https://chatbot-backend-hcvx.onrender.com/webhook',
            ...config
        };
        
        this.isOpen = false;
        this.messages = [];
        this.userId = this.generateUserId();
        this.conversationId = this.generateConversationId();
        
        this.init();
    }
    
    init() {
        this.createWidget();
        this.setupEventListeners();
        this.loadHistory();
    }
    
    createWidget() {
        // Kreiraj glavni kontejner
        const container = document.createElement('div');
        container.id = 'smart-chat-widget';
        container.style.cssText = `
            position: fixed;
            ${this.config.position === 'bottom-right' ? 'bottom: 20px; right: 20px;' : 'bottom: 20px; left: 20px;'}
            z-index: 9999;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
        `;
        
        // Kreiraj chat dugme
        this.chatButton = document.createElement('div');
        this.chatButton.innerHTML = `
            <div style="
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: ${this.config.primaryColor};
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                transition: transform 0.3s;
            ">
                <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
            </div>
        `;
        
        // Kreiraj chat prozor
        this.chatWindow = document.createElement('div');
        this.chatWindow.style.cssText = `
            position: absolute;
            bottom: 80px;
            ${this.config.position === 'bottom-right' ? 'right: 0;' : 'left: 0;'}
            width: 350px;
            height: 500px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.2);
            display: none;
            flex-direction: column;
            overflow: hidden;
        `;
        
        // Zaglavlje
        const header = document.createElement('div');
        header.style.cssText = `
            background: ${this.config.primaryColor};
            color: white;
            padding: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
        `;
        header.innerHTML = `
            <div>
                <div style="font-weight: 600;">${this.config.title}</div>
                <div style="font-size: 12px; opacity: 0.9;">${this.config.subtitle}</div>
            </div>
            <div style="cursor: pointer;" id="close-chat">✕</div>
        `;
        
        // Oblast za poruke
        this.messagesArea = document.createElement('div');
        this.messagesArea.style.cssText = `
            flex: 1;
            padding: 16px;
            overflow-y: auto;
            background: #f5f8fa;
        `;
        
        // Input oblast
        const inputArea = document.createElement('div');
        inputArea.style.cssText = `
            padding: 16px;
            border-top: 1px solid #e9ecef;
            display: flex;
            gap: 8px;
        `;
        
        this.input = document.createElement('input');
        this.input.type = 'text';
        this.input.placeholder = 'Unesite poruku...';
        this.input.style.cssText = `
            flex: 1;
            padding: 10px;
            border: 1px solid #dee2e6;
            border-radius: 20px;
            outline: none;
            font-size: 14px;
        `;
        
        const sendButton = document.createElement('button');
        sendButton.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="${this.config.primaryColor}" stroke-width="2">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
        `;
        sendButton.style.cssText = `
            background: none;
            border: none;
            cursor: pointer;
            padding: 0 8px;
        `;
        
        inputArea.appendChild(this.input);
        inputArea.appendChild(sendButton);
        
        this.chatWindow.appendChild(header);
        this.chatWindow.appendChild(this.messagesArea);
        this.chatWindow.appendChild(inputArea);
        
        container.appendChild(this.chatButton);
        container.appendChild(this.chatWindow);
        
        document.body.appendChild(container);
    }
    
    setupEventListeners() {
        // Otvori/zatvori chat
        this.chatButton.addEventListener('click', () => this.toggleChat());
        
        document.getElementById('close-chat').addEventListener('click', () => this.closeChat());
        
        // Slanje poruke na Enter
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Slanje poruke na klik
        const sendButton = document.querySelector('#smart-chat-widget button');
        sendButton.addEventListener('click', () => this.sendMessage());
    }
    
    toggleChat() {
        this.isOpen = !this.isOpen;
        this.chatWindow.style.display = this.isOpen ? 'flex' : 'none';
        
        if (this.isOpen) {
            this.input.focus();
            this.scrollToBottom();
        }
    }
    
    closeChat() {
        this.isOpen = false;
        this.chatWindow.style.display = 'none';
    }
    
    async sendMessage() {
        const message = this.input.value.trim();
        if (!message) return;
        
        // Prikaži poruku korisnika
        this.addMessage(message, 'user');
        this.input.value = '';
        this.input.disabled = true;
        
        // Prikaži indikator kucanja
        this.showTypingIndicator();
        
        try {
            // Pošalji na server
            const response = await fetch(this.config.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    user_id: this.userId,
                    conversation_id: this.conversationId,
                    channel: 'web',
                    metadata: {
                        url: window.location.href,
                        timestamp: new Date().toISOString()
                    }
                })
            });
            
            const data = await response.json();
            
            // Ukloni indikator kucanja
            this.hideTypingIndicator();
            
            // Prikaži odgovor
            this.addMessage(data.response, 'bot', data);
            
            // Ako je potrebna eskalacija
            if (data.escalation_needed) {
                this.showEscalationMessage(data.escalation_data);
            }
        } catch (error) {
            console.error('Greška:', error);
            this.hideTypingIndicator();
            this.addMessage('Došlo je do greške. Molim vas pokušajte ponovo.', 'bot');
        } finally {
            this.input.disabled = false;
            this.input.focus();
        }
    }
    
    addMessage(text, sender, metadata = null) {
        // Debug: prikaži šta stiže
        console.log("🔍 Primljen text za prikaz:", text);
        console.log("📏 Dužina teksta:", text.length);
        
        // Konvertuj linkove u HTML
        let htmlText = text;
        
        // Prvo konvertuj obične URL-ove u klikabilne linkove
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        if (urlRegex.test(htmlText)) {
            console.log("🔎 Pronađen običan URL, konvertujem...");
            htmlText = htmlText.replace(urlRegex, '<a href="$1" target="_blank" style="color: #069806; text-decoration: underline;">$1</a>');
        }
        
        // Zatim konvertuj Markdown linkove [tekst](url)
        const markdownLinkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
        if (markdownLinkRegex.test(htmlText)) {
            console.log("🔎 Pronađen Markdown link, konvertujem...");
            htmlText = htmlText.replace(markdownLinkRegex, '<a href="$2" target="_blank" style="color: #069806; text-decoration: underline;">$1</a>');
        }
        
        // Proveri da li ima HTML tagova
        console.log("🔎 Sadrži <a tag?", htmlText.includes('<a'));
        
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            margin-bottom: 12px;
            display: flex;
            ${sender === 'user' ? 'justify-content: flex-end;' : 'justify-content: flex-start;'}
        `;
        
        const bubble = document.createElement('div');
        bubble.style.cssText = `
            max-width: 80%;
            padding: 10px 14px;
            border-radius: 18px;
            font-size: 14px;
            line-height: 1.4;
            word-wrap: break-word;
            ${sender === 'user' 
                ? `background: ${this.config.primaryColor}; color: white; border-bottom-right-radius: 4px;` 
                : 'background: white; color: #1e293b; border-bottom-left-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'}
        `;
        
        // Koristimo konvertovani HTML
        bubble.innerHTML = htmlText;
        
        // Dodaj vreme
        const time = document.createElement('div');
        time.style.cssText = `
            font-size: 10px;
            opacity: 0.6;
            margin-top: 4px;
            text-align: ${sender === 'user' ? 'right' : 'left'};
        `;
        time.textContent = new Date().toLocaleTimeString('sr-RS', { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.appendChild(bubble);
        bubble.appendChild(time);
        this.messagesArea.appendChild(messageDiv);
        
        // Sačuvaj u lokalnu memoriju
        this.messages.push({
            text,
            sender,
            timestamp: new Date().toISOString(),
            metadata
        });
        
        this.saveToLocalStorage();
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'typing-indicator';
        indicator.style.cssText = `
            margin-bottom: 12px;
            display: flex;
            justify-content: flex-start;
        `;
        
        const dots = document.createElement('div');
        dots.style.cssText = `
            background: white;
            color: #1e293b;
            padding: 12px 16px;
            border-radius: 18px;
            border-bottom-left-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            display: flex;
            gap: 4px;
        `;
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            dot.style.cssText = `
                width: 6px;
                height: 6px;
                background: #94a3b8;
                border-radius: 50%;
                display: inline-block;
                animation: typing 1.4s infinite;
                animation-delay: ${i * 0.2}s;
            `;
            dots.appendChild(dot);
        }
        
        indicator.appendChild(dots);
        this.messagesArea.appendChild(indicator);
        
        // Dodaj keyframes za animaciju
        const style = document.createElement('style');
        style.textContent = `
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-10px); }
            }
        `;
        document.head.appendChild(style);
        
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
    }
    
    showEscalationMessage(escalationData) {
        const escalationDiv = document.createElement('div');
        escalationDiv.style.cssText = `
            margin: 10px 0;
            padding: 12px;
            background: #fff3cd;
            border: 1px solid #ffeeba;
            border-radius: 8px;
            color: #856404;
            font-size: 13px;
        `;
        escalationDiv.textContent = 'Agent će uskoro preuzeti razgovor. Molim vas sačekajte.';
        this.messagesArea.appendChild(escalationDiv);
        this.scrollToBottom();
    }
    
    scrollToBottom() {
        this.messagesArea.scrollTop = this.messagesArea.scrollHeight;
    }
    
    generateUserId() {
        return 'user_' + Math.random().toString(36).substr(2, 9);
    }
    
    generateConversationId() {
        return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    saveToLocalStorage() {
        localStorage.setItem('chat_messages', JSON.stringify({
            userId: this.userId,
            messages: this.messages.slice(-50),
            timestamp: new Date().toISOString()
        }));
    }
    
    loadHistory() {
        const saved = localStorage.getItem('chat_messages');
        if (saved) {
            try {
                const data = JSON.parse(saved);
                if (data.userId === this.userId) {
                    this.messages = data.messages || [];
                    this.messages.slice(-5).forEach(msg => {
                        this.addMessage(msg.text, msg.sender);
                    });
                }
            } catch (e) {
                console.error('Greška pri učitavanju istorije:', e);
            }
        }
    }
}

// Robusnija inicijalizacija - radi bez obzira na to kako se stranica učitava
function initChatWidget() {
    // Proveri da li već postoji da ne bismo duplirali
    if (window.smartChatWidgetInstance) {
        console.log("Chatbot već inicijalizovan");
        return;
    }
    
    console.log("Inicijalizujem chatbot...");
    window.smartChatWidgetInstance = new SmartChatWidget({
        primaryColor: '#069806',
        title: 'Vaša ZAPmoto podrška',
        subtitle: 'Pitajte nas bilo šta o električnim skuterima',
        position: 'bottom-right',
        apiUrl: 'https://chatbot-backend-hcvx.onrender.com/webhook'
    });
}

// Pokušaj odmah ako je dokument već učitan
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    console.log("Dokument već spreman, pokrećem odmah");
    setTimeout(initChatWidget, 100);
} else {
    console.log("Čekam DOMContentLoaded...");
    document.addEventListener('DOMContentLoaded', initChatWidget);
}

window.addEventListener('load', function() {
    console.log("Load event - pokrećem ako već nije pokrenuto");
    initChatWidget();
});