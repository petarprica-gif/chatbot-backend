import os
import logging
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from datetime import datetime
import json
from chatbot_core import ContextAwareChatbot

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicijalizacija chatbot-a
chatbot = None

@app.before_first_request
def initialize_chatbot():
    global chatbot
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        chatbot = ContextAwareChatbot(api_key)
        logger.info("Chatbot inicijalizovan")
    else:
        logger.error("OPENAI_API_KEY nije postavljen")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        message = data.get('message', '')
        user_id = data.get('user_id', 'anonymous')
        conversation_id = data.get('conversation_id', '')
        channel = data.get('channel', 'web')
        
        if not chatbot:
            return jsonify({"error": "Chatbot nije inicijalizovan"}), 500
        
        response = chatbot.generate_response(message, user_id, conversation_id, channel)
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Greška u webhook-u: {str(e)}")
        return jsonify({"error": "Došlo je do greške"}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))  # Render koristi PORT, podrazumevano 10000
    app.run(host='0.0.0.0', port=port, debug=False)