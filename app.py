import os
import sys
import logging
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from datetime import datetime
import json
import bleach
from chatbot_core import ContextAwareChatbot

print("=== 1. APP.PY POČINJE SA UČITAVANJEM ===", file=sys.stderr)
sys.stderr.flush()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicijalizacija chatbot-a
chatbot = None

def sanitize_input(user_input):
    """
    Uklanja sve HTML tagove iz korisničkog unosa radi zaštite od XSS napada.
    
    Args:
        user_input: String koji korisnik šalje
        
    Returns:
        Očišćen string bez HTML tagova
    """
    if not user_input:
        return user_input
    
    # clean funkcija sa praznim listama za tags i attributes uklanja SVE tagove
    cleaned = bleach.clean(user_input, tags=[], attributes={}, strip=True)
    
    # Loguj razliku ako je bilo promena (korisno za debug)
    if user_input != cleaned:
        logger.info(f"Sanitizacija: '{user_input}' -> '{cleaned}'")
    
    return cleaned

def initialize_chatbot():
    global chatbot
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        try:
            chatbot = ContextAwareChatbot(api_key)
            logger.info("Chatbot inicijalizovan")
        except Exception as e:
            logger.error(f"Greška pri inicijalizaciji chatbot-a: {str(e)}")
    else:
        logger.error("OPENAI_API_KEY nije postavljen")

# Pozovi funkciju odmah pri pokretanju
initialize_chatbot()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/webhook', methods=['POST', 'OPTIONS'])
def webhook():
    """
    Glavni webhook za sve kanale komunikacije.
    Prima poruke, sanitizuje ih i prosleđuje chatbot-u.
    """
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        if not data:
            logger.error("Primljen prazan JSON zahtev")
            return jsonify({"error": "Prazan zahtev"}), 400
        
        raw_message = data.get('message', '')
        user_id = data.get('user_id', 'anonymous')
        conversation_id = data.get('conversation_id', '')
        channel = data.get('channel', 'web')
        
        # === SANITIZACIJA KORISNIČKOG UNOSA ===
        sanitized_message = sanitize_input(raw_message)
        
        # Ako je poruka postala prazna nakon sanitizacije, to je verovatno bio pokušaj napada
        if not sanitized_message and raw_message:
            logger.warning(f"Potencijalni XSS napad otkriven: '{raw_message}' od korisnika {user_id}")
            return jsonify({
                "response": "Detektovan je potencijalno opasan unos. Molim vas koristite samo običan tekst.",
                "escalation_needed": False
            })
        
        logger.info(f"Primljena poruka od {user_id}: '{raw_message}' -> sanitizovano: '{sanitized_message}'")
        # ======================================
        
        if not chatbot:
            logger.error("Chatbot nije inicijalizovan")
            return jsonify({"error": "Chatbot nije inicijalizovan"}), 500
        
        # Prosledi sanitizovanu poruku chatbot-u
        response = chatbot.generate_response(
            message=sanitized_message,
            user_id=user_id,
            conversation_id=conversation_id,
            channel=channel
        )
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Greška u webhook-u: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Došlo je do greške"}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))  # Render koristi PORT, podrazumevano 10000
    app.run(host='0.0.0.0', port=port, debug=False)