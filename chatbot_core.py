import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import json
import os
from datetime import datetime, timedelta
import hashlib
import redis
from dataclasses import dataclass, asdict
from enum import Enum

# Inicijalizacija Redis-a za keširanje (opciono)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

class Intent(Enum):
    GREETING = "greeting"
    PRODUCT_QUESTION = "product_question"
    ORDER_STATUS = "order_status"
    RETURN_REQUEST = "return_request"
    PAYMENT_ISSUE = "payment_issue"
    CONTACT_SUPPORT = "contact_support"
    FAREWELL = "farewell"
    UNKNOWN = "unknown"

@dataclass
class ConversationMemory:
    """Čuva kontekst razgovora"""
    user_id: str
    messages: List[Dict]
    last_updated: datetime
    context: Dict[str, Any]
    sentiment: float = 0.0
    escalation_needed: bool = False

class ContextAwareChatbot:
    def __init__(self, api_key: str, knowledge_base_path: str = "knowledge_base.json"):
        """
        Inicijalizacija chatbota sa kontekstualnom svešću
        
        Args:
            api_key: API ključ za OpenAI ili drugi LLM provajder
            knowledge_base_path: Putanja do fajla sa bazom znanja
        """
        self.api_key = api_key
        openai.api_key = api_key
        
        # Model za generisanje embedinga za semantičko pretraživanje [citation:3]
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        
        # Učitavanje baze znanja
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        
        # Keširanje embedinga za brže pretraživanje
        self.embedding_cache = {}
        
        # Aktivne konverzacije
        self.active_conversations: Dict[str, ConversationMemory] = {}
        
    def load_knowledge_base(self, path: str) -> List[Dict]:
        """
        Učitava bazu znanja iz JSON fajla
        
        Baza znanja može sadržati:
        - FAQ parove
        - Informacije o proizvodima
        - Politike (return, shipping, itd.)
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Dodajemo embedinge za svaki unos
            for item in data:
                text_for_embedding = f"{item.get('question', '')} {item.get('answer', '')} {item.get('keywords', '')}"
                item['embedding'] = self.embedding_model.encode(text_for_embedding).tolist()
            
            logger.info(f"Učitano {len(data)} stavki u bazu znanja")
            return data
        except FileNotFoundError:
            logger.warning("Baza znanja nije pronađena, kreiram praznu")
            return []
    
    def retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Pronalazi najrelevantnije informacije iz baze znanja za korisnički upit
        
        Koristi semantičko pretraživanje (RAG) za pronalaženje relevantnih informacija [citation:6]
        
        Args:
            query: Korisnički upit
            top_k: Broj najrelevantnijih rezultata
        
        Returns:
            Lista relevantnih informacija iz baze znanja
        """
        # Generiši embedding za upit
        query_embedding = self.embedding_model.encode(query)
        
        # Izračunaj sličnosti
        similarities = []
        for idx, item in enumerate(self.knowledge_base):
            item_embedding = np.array(item.get('embedding', self.embedding_model.encode(f"{item.get('question', '')} {item.get('answer', '')}")))
            similarity = np.dot(query_embedding, item_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding))
            similarities.append((similarity, idx, item))
        
        # Sortiraj po sličnosti i uzmi top_k
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        relevant_items = []
        for i in range(min(top_k, len(similarities))):
            if similarities[i][0] > 0.5:  # Prag relevantnosti
                relevant_items.append({
                    'content': similarities[i][2],
                    'relevance_score': float(similarities[i][0]),
                    'source': similarities[i][2].get('source', 'knowledge_base')
                })
        
        return relevant_items
    
    def detect_intent(self, message: str, conversation_history: List[Dict]) -> Intent:
        """
        Detektuje nameru korisnika koristeći NLP
        
        Args:
            message: Trenutna poruka
            conversation_history: Istorija razgovora za kontekst
        
        Returns:
            Detektovana namera
        """
        # Pravimo prompt za OpenAI sa kontekstom
        system_prompt = """
        Ti si AI asistent za detekciju namere. Na osnovu korisničke poruke i istorije razgovora,
        odredi koja je od sledećih namera najverovatnija:
        - greeting: Pozdrav, početak razgovora
        - product_question: Pitanje o proizvodu (cena, dostupnost, karakteristike)
        - order_status: Provera statusa porudžbine
        - return_request: Zahtev za povraćaj ili reklamaciju
        - payment_issue: Problem sa plaćanjem
        - contact_support: Zahtev za kontakt sa agentom
        - farewell: Završetak razgovora, pozdrav
        - unknown: Nejasna namera
        
        Vrati samo naziv namere, ništa drugo.
        """
        
        # Pripremi kontekst iz istorije
        recent_history = conversation_history[-5:] if conversation_history else []
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Istorija:\n{history_text}\n\nTrenutna poruka: {message}"}
                ],
                max_tokens=10,
                temperature=0.3
            )
            
            intent_str = response.choices[0].message.content.strip().lower()
            
            # Mapiramo string u enum
            for intent in Intent:
                if intent.value == intent_str:
                    return intent
            
            return Intent.UNKNOWN
        except Exception as e:
            logger.error(f"Greška pri detekciji namere: {str(e)}")
            return Intent.UNKNOWN
    
    def generate_response(self, 
                         message: str, 
                         user_id: str, 
                         conversation_id: str = None,
                         channel: str = "web") -> Dict[str, Any]:
        """
        Generiše odgovor na korisničku poruku sa kontekstualnom svešću
        
        Args:
            message: Korisnička poruka
            user_id: ID korisnika
            conversation_id: ID konverzacije
            channel: Kanal komunikacije
        
        Returns:
            Dict sa odgovorom i metapodacima
        """
        
        # Dohvati ili kreiraj memoriju konverzacije
        memory_key = f"{user_id}:{conversation_id}" if conversation_id else user_id
        conversation = self.get_or_create_conversation(memory_key, user_id, channel)
        
        # Detektuj nameru
        intent = self.detect_intent(message, conversation.messages)
        
        # Proveri da li je potrebna eskalacija
        if self.should_escalate(message, intent, conversation):
            conversation.escalation_needed = True
            return self.prepare_escalation(message, conversation)
        
        # Pronađi relevantne informacije iz baze znanja
        relevant_knowledge = self.retrieve_relevant_knowledge(message)
        
        # Pripremi kontekst za generisanje odgovora
        context = {
            'intent': intent.value,
            'relevant_knowledge': relevant_knowledge,
            'user_id': user_id,
            'channel': channel,
            'conversation_history': conversation.messages[-10:],  # Poslednjih 10 poruka
            'user_context': conversation.context
        }
        
        # Generiši odgovor koristeći LLM sa kontekstom
        response_text = self.generate_llm_response(message, context)
        
        # Sačuvaj poruke u memoriju
        self.add_to_conversation(memory_key, {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        self.add_to_conversation(memory_key, {
            'role': 'assistant',
            'content': response_text,
            'timestamp': datetime.now().isoformat(),
            'intent': intent.value,
            'knowledge_used': [k['source'] for k in relevant_knowledge] if relevant_knowledge else []
        })
        
        # Pripremi odgovor
        response = {
            'response': response_text,
            'intent': intent.value,
            'conversation_id': conversation_id,
            'escalation_needed': False,
            'knowledge_sources': [k['source'] for k in relevant_knowledge] if relevant_knowledge else [],
            'channel_specific': self.get_channel_specific_response(channel, response_text)
        }
        
        return response
    
    def generate_llm_response(self, message: str, context: Dict) -> str:
        """
        Generiše odgovor koristeći LLM sa ugrađenim kontekstom
        
        Ova metoda kombinuje sve relevantne informacije u prompt
        kako bi se osiguralo da je odgovor tačan i kontekstualno svestan [citation:6][citation:9]
        """
        
        # Pripremi kontekst iz baze znanja
        knowledge_text = ""
        if context['relevant_knowledge']:
            knowledge_text = "Relevantne informacije:\n"
            for idx, item in enumerate(context['relevant_knowledge']):
                content = item['content']
                knowledge_text += f"{idx+1}. Pitanje: {content.get('question', 'Informacija')}\n"
                knowledge_text += f"   Odgovor: {content.get('answer', content.get('content', ''))}\n"
                knowledge_text += f"   Izvor: {content.get('source', 'baza znanja')}\n\n"
        
        # Pripremi istoriju razgovora
        history_text = ""
        if context['conversation_history']:
            history_text = "Istorija razgovora:\n"
            for msg in context['conversation_history']:
                role = "Korisnik" if msg['role'] == 'user' else "Asistent"
                history_text += f"{role}: {msg['content']}\n"
        
        # System prompt za asistenta
        system_prompt = """
        Ti si profesionalni AI asistent za korisničku podršku i e-trgovinu.
        
        VAŽNA UPUTSTVA:
        1. Budi koncizan, ali ljubazan - koristi prirodan ton razgovora
        2. Odgovaraj isključivo na osnovu dostupnih informacija - ako ne znaš odgovor, priznaj to
        3. Izbegavaj halucinacije - nemoj izmišljati informacije koje nisu u bazi znanja [citation:6]
        4. Ako korisnik pita nešto što nije u tvojoj bazi znanja, ljubazno ga uputi da ćeš proslediti agentu
        5. Strukturiraj informacije jasno - koristi bullet points gde je prikladno
        6. Prati kontekst razgovora - ako se korisnik vraća na prethodnu temu, seti se toga
        
        TVOJA ULOGA:
        - Razumevanje namere korisnika uprkos greškama u kucanju
        - Pružanje tačnih informacija o proizvodima, porudžbinama, politikama
        - Asistencija pri kupovini i rešavanju problema
        - Prepoznavanje kada je potrebna ljudska intervencija
        
        Detektovana namera korisnika: {intent}
        Kanal komunikacije: {channel}
        """.format(intent=context['intent'], channel=context['channel'])
        
        # Korisnički prompt sa svim informacijama
        user_prompt = f"""
        {history_text}
        
        {knowledge_text}
        
        Korisnik pita: {message}
        
        Molim te da odgovoriš na ovo pitanje na osnovu dostupnih informacija.
        Ako informacija nije dostupna, reci da ćeš proslediti agentu.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Greška pri generisanju odgovora: {str(e)}")
            return "Izvinite, došlo je do tehničke greške. Molim vas pokušajte ponovo ili kontaktirajte našu podršku."
    
    def should_escalate(self, message: str, intent: Intent, conversation: ConversationMemory) -> bool:
        """
        Odlučuje da li je potrebna eskalacija ljudskom agentu
        
        Uslovi za eskalaciju:
        - Korisnik eksplicitno traži agenta
        - Intent je contact_support
        - Kompleksna pitanja (reklamacije, žalbe)
        - Više neuspešnih pokušaja razumevanja
        """
        
        # Eksplicitni zahtev za agentom
        escalation_keywords = ['agent', 'operater', 'čovek', 'govori sa', 'uživo', 'live chat']
        if any(keyword in message.lower() for keyword in escalation_keywords):
            return True
        
        # Intent contact_support
        if intent == Intent.CONTACT_SUPPORT:
            return True
        
        # Reklamacije i povraćaji - često zahtevaju ljudsku intervenciju
        if intent == Intent.RETURN_REQUEST:
            # Proveri da li imamo informacije o povraćaju
            relevant = self.retrieve_relevant_knowledge("povraćaj novca reklamacija", top_k=1)
            if not relevant or relevant[0]['relevance_score'] < 0.7:
                return True
        
        # Problemi sa plaćanjem - mogu biti kompleksni
        if intent == Intent.PAYMENT_ISSUE:
            # Proveri da li imamo dobre informacije
            relevant = self.retrieve_relevant_knowledge(message, top_k=2)
            if not relevant or all(r['relevance_score'] < 0.6 for r in relevant):
                return True
        
        # Ako je bilo više neuspešnih pokušaja (npr. 3 uzastopna unknown intent)
        recent_messages = conversation.messages[-6:]  # Poslednje 3 interakcije (user+assistant)
        unknown_count = sum(1 for msg in recent_messages if msg.get('intent') == 'unknown')
        if unknown_count >= 3:
            return True
        
        return False
    
    def prepare_escalation(self, message: str, conversation: ConversationMemory) -> Dict[str, Any]:
        """
        Priprema eskalaciju ljudskom agentu
        
        Šalje kompletan transkript razgovora i kontekst agentu [citation:2][citation:8]
        """
        
        # Pripremi transkript razgovora
        transcript = []
        for msg in conversation.messages:
            transcript.append({
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg.get('timestamp', '')
            })
        
        # Dodaj i trenutnu poruku
        transcript.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Pripremi kontekst za agenta
        context_vars = {
            'userId': conversation.user_id,
            'channel': conversation.context.get('channel', 'unknown'),
            'intent': conversation.context.get('last_intent', 'unknown'),
            'sentiment': conversation.sentiment,
            'conversationDuration': str(datetime.now() - conversation.last_updated)
        }
        
        # Logika za slanje agentu (integrisati sa platformom za agente)
        # Ovo može biti API poziv ka vašem sistemu za agente
        
        logger.info(f"Eskalacija za korisnika {conversation.user_id} pripremljena")
        
        return {
            'response': "Povezujem vas sa našim agentom za korisničku podršku. Molim vas sačekajte trenutak.",
            'escalation_needed': True,
            'escalation_data': {
                'transcript': transcript,
                'context': context_vars,
                'conversation_id': conversation.user_id
            },
            'channel_specific': self.get_channel_specific_response(conversation.context.get('channel', 'web'), 
                                                                   None, 
                                                                   escalation=True)
        }
    
    def get_or_create_conversation(self, memory_key: str, user_id: str, channel: str) -> ConversationMemory:
        """
        Dohvata postojeću ili kreira novu konverzaciju u memoriji
        """
        if memory_key in self.active_conversations:
            return self.active_conversations[memory_key]
        
        # Proveri Redis keš
        cached = redis_client.get(f"conversation:{memory_key}")
        if cached:
            data = json.loads(cached)
            conversation = ConversationMemory(
                user_id=data['user_id'],
                messages=data['messages'],
                last_updated=datetime.fromisoformat(data['last_updated']),
                context=data.get('context', {}),
                sentiment=data.get('sentiment', 0.0),
                escalation_needed=data.get('escalation_needed', False)
            )
            self.active_conversations[memory_key] = conversation
            return conversation
        
        # Kreiraj novu konverzaciju
        conversation = ConversationMemory(
            user_id=user_id,
            messages=[],
            last_updated=datetime.now(),
            context={'channel': channel, 'start_time': datetime.now().isoformat()}
        )
        self.active_conversations[memory_key] = conversation
        return conversation
    
    def add_to_conversation(self, memory_key: str, message: Dict):
        """
        Dodaje poruku u konverzaciju i ažurira keš
        """
        if memory_key not in self.active_conversations:
            return
        
        conversation = self.active_conversations[memory_key]
        conversation.messages.append(message)
        conversation.last_updated = datetime.now()
        
        # Ažuriraj kontekst (npr. poslednji intent)
        if 'intent' in message:
            conversation.context['last_intent'] = message['intent']
        
        # Sačuvaj u Redis sa expiration (npr. 24h)
        redis_client.setex(
            f"conversation:{memory_key}",
            86400,  # 24 sata
            json.dumps({
                'user_id': conversation.user_id,
                'messages': conversation.messages,
                'last_updated': conversation.last_updated.isoformat(),
                'context': conversation.context,
                'sentiment': conversation.sentiment,
                'escalation_needed': conversation.escalation_needed
            })
        )
    
    def get_channel_specific_response(self, channel: str, text: str = None, escalation: bool = False) -> Dict:
        """
        Prilagođava odgovor specifičnostima kanala [citation:1]
        """
        channel_configs = {
            'web': {
                'type': 'text',
                'options': {
                    'quick_replies': True,
                    'rich_text': True,
                    'buttons': True
                }
            },
            'whatsapp': {
                'type': 'text',
                'options': {
                    'max_length': 4096,
                    'interactive': True,
                    'buttons': True,
                    'lists': True
                }
            },
            'viber': {
                'type': 'text',
                'options': {
                    'rich_media': True,
                    'keyboard': True,
                    'tracking_data': True
                }
            },
            'telegram': {
                'type': 'text',
                'options': {
                    'markdown': True,
                    'inline_keyboards': True
                }
            }
        }
        
        config = channel_configs.get(channel, channel_configs['web'])
        
        response = {
            'channel': channel,
            'type': config['type'],
            'config': config['options']
        }
        
        if text:
            response['text'] = text
            
        if escalation:
            response['escalation'] = {
                'message': "Povezivanje sa agentom...",
                'estimated_wait': "2-3 minuta"
            }
        
        return response