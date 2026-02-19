import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    WHATSAPP = {
        'phone_number_id': os.getenv('WHATSAPP_PHONE_ID'),
        'access_token': os.getenv('WHATSAPP_ACCESS_TOKEN')
    }
    
    CONVERSATION_STORAGE = {
        'enabled': True,
        'retention_days': 30,
        'storage_type': 'database'
    }
    
    ESCALATION = {
        'enabled': True,
        'max_unknown_intents': 3,
        'agent_platform': 'internal'
    }