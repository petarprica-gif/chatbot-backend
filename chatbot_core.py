import openai
import numpy as np
from typing import List, Dict, Any
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import traceback

logger = logging.getLogger(__name__)

# Redis je POTPUNO ISKLJUČEN - ne koristimo ga
# Svi pokušaji povezivanja sa Redis-om su uklonjeni

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
            api_key: API ključ za OpenAI
            knowledge_base_path: Putanja do fajla sa bazom znanja
        """
        self.api_key = api_key
        openai.api_key = api_key
        
        # NEMA Redis-a!
        # NEMA sentence-transformers!
        # Samo OpenAI API
        
        # Učitavanje baze znanja
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        
        # Keširanje embedinga za brže pretraživanje (čuvaćemo u memoriji)
        self.embedding_cache = {}
        
        # Aktivne konverzacije (čuvamo u RAM memoriji)
        self.active_conversations: Dict[str, ConversationMemory] = {}
        
        logger.info(f"Chatbot inicijalizovan sa {len(self.knowledge_base)} stavki u bazi znanja")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generiše embedding koristeći OpenAI API
        
        Args:
            text: Tekst za koji se generiše embedding
            
        Returns:
            Lista float vrednosti (embedding vektor)
        """
        # Provera keša
        cache_key = f"emb_{hash(text)}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Poziv OpenAI API-ja za generisanje embeddinga
            response = openai.Embedding.create(
                model="text-embedding-3-small",  # Mali, brz i jeftin model
                input=text,
                encoding_format="float"
            )
            
            embedding = response['data'][0]['embedding']
            
            # Čuvaj u kešu (ograniči veličinu keša)
            if len(self.embedding_cache) < 1000:
                self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Greška pri generisanju embeddinga: {str(e)}")
            logger.error(traceback.format_exc())
            # Vrati prazan vektor kao fallback (dimenzija 1536 za text-embedding-3-small)
            return [0.0] * 1536
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Računa kosinusnu sličnost između dva vektora
        
        Args:
            vec1, vec2: Vektori za poređenje
            
        Returns:
            Sličnost između 0 i 1
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Izračunaj kosinusnu sličnost
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))
    
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
            
            # Za svaki unos, kreiraj tekst za embedding
            for item in data:
                text_for_embedding = f"{item.get('question', '')} {item.get('answer', '')} {item.get('keywords', '')}"
                item['text_for_embedding'] = text_for_embedding
            
            logger.info(f"Učitano {len(data)} stavki u bazu znanja")
            return data
        except FileNotFoundError:
            logger.warning("Baza znanja nije pronađena, kreiram praznu")
            return []
        except Exception as e:
            logger.error(f"Greška pri učitavanju baze znanja: {str(e)}")
            return []
    
    def retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Pronalazi najrelevantnije informacije iz baze znanja za korisnički upit
        
        Koristi OpenAI embeddige za semantičko pretraživanje
        
        Args:
            query: Korisnički upit
            top_k: Broj najrelevantnijih rezultata
        
        Returns:
            Lista relevantnih informacija iz baze znanja
        """
        if not self.knowledge_base:
            return []
        
        try:
            # Generiši embedding za upit
            query_embedding = self.get_embedding(query)
            
            # Izračunaj sličnosti
            similarities = []
            for idx, item in enumerate(self.knowledge_base):
                # Generiši embedding za stavku ako već nije u kešu
                item_text = item.get('text_for_embedding', '')
                item_key = f"kb_{idx}"
                
                if item_key not in self.embedding_cache:
                    self.embedding_cache[item_key] = self.get_embedding(item_text)
                
                item_embedding = self.embedding_cache[item_key]
                
                # Izračunaj sličnost
                similarity = self.cosine_similarity(query_embedding, item_embedding)
                similarities.append((similarity, idx, item))
            
            # Sortiraj po sličnosti (opadajuće) i uzmi top_k
            similarities.sort(reverse=True, key=lambda x: x[0])
            
            relevant_items = []
            for i in range(min(top_k, len(similarities))):
                if similarities[i][0] > 0.5:  # Prag relevantnosti (možete prilagoditi)
                    relevant_items.append({
                        'content': similarities[i][2],
                        'relevance_score': float(similarities[i][0]),
                        'source': similarities[i][2].get('source', 'knowledge_base')
                    })
            
            return relevant_items
        except Exception as e:
            logger.error(f"Greška u retrieve_relevant_knowledge: {str(e)}")
            return []
    
    def detect_intent(self, message: str, conversation_history: List[Dict]) -> Intent:
        """
        Detektuje nameru korisnika koristeći OpenAI
        
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
                model="gpt-3.5-turbo",
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
        
        try:
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
                'conversation_history': conversation.messages[-10:],
                'user_context': conversation.context
            }
            
            # Generiši odgovor koristeći OpenAI sa kontekstom
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
            
        except Exception as e:
            logger.error(f"Greška u generate_response: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'response': "Došlo je do tehničke greške. Molim vas pokušajte ponovo ili kontaktirajte podršku.",
                'intent': 'unknown',
                'conversation_id': conversation_id,
                'escalation_needed': True,
                'knowledge_sources': []
            }
    
    def generate_llm_response(self, message: str, context: Dict) -> str:
        """
        Generiše odgovor koristeći OpenAI sa ugrađenim kontekstom
        """
        
        try:
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
            system_prompt = f"""
            Ti si profesionalni AI asistent za korisničku podršku i e-trgovinu.
            
            VAŽNA UPUTSTVA:
            1. Budi koncizan, ali ljubazan - koristi prirodan ton razgovora
            2. Odgovaraj isključivo na osnovu dostupnih informacija - ako ne znaš odgovor, priznaj to
            3. Izbegavaj halucinacije - nemoj izmišljati informacije koje nisu u bazi znanja
            4. Ako korisnik pita nešto što nije u tvojoj bazi znanja, ljubazno ga uputi da ćeš proslediti agentu
            5. Strukturiraj informacije jasno - koristi bullet points gde je prikladno
            6. Prati kontekst razgovora - ako se korisnik vraća na prethodnu temu, seti se toga
            
            Detektovana namera korisnika: {context['intent']}
            Kanal komunikacije: {context['channel']}
            """
            
            # Korisnički prompt sa svim informacijama
            user_prompt = f"""
            {history_text}
            
            {knowledge_text}
            
            Korisnik pita: {message}
            
            Molim te da odgovoriš na ovo pitanje na osnovu dostupnih informacija.
            Ako informacija nije dostupna, reci da ćeš proslediti agentu.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
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
        """
        # Eksplicitni zahtev za agentom
        escalation_keywords = ['agent', 'operater', 'čovek', 'govori sa', 'uživo', 'live chat']
        if any(keyword in message.lower() for keyword in escalation_keywords):
            return True
        
        # Intent contact_support
        if intent == Intent.CONTACT_SUPPORT:
            return True
        
        # Ako je bilo više neuspešnih pokušaja
        recent_messages = conversation.messages[-6:]
        unknown_count = sum(1 for msg in recent_messages if msg.get('intent') == 'unknown')
        if unknown_count >= 3:
            return True
        
        return False
    
    def prepare_escalation(self, message: str, conversation: ConversationMemory) -> Dict[str, Any]:
        """
        Priprema eskalaciju ljudskom agentu
        """
        # Pripremi transkript razgovora
        transcript = []
        for msg in conversation.messages:
            transcript.append({
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg.get('timestamp', '')
            })
        
        transcript.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Eskalacija za korisnika {conversation.user_id} pripremljena")
        
        return {
            'response': "Povezujem vas sa našim agentom za korisničku podršku. Molim vas sačekajte trenutak.",
            'escalation_needed': True,
            'escalation_data': {
                'transcript': transcript,
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
        
        # Kreiraj novu konverzaciju (NEMA Redis-a!)
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
        
        # Ažuriraj kontekst
        if 'intent' in message:
            conversation.context['last_intent'] = message['intent']
        
        # NEMA Redis čuvanja - samo u RAM memoriji
        # Ovo je OK za manji broj korisnika
    
    def get_channel_specific_response(self, channel: str, text: str = None, escalation: bool = False) -> Dict:
        """
        Prilagođava odgovor specifičnostima kanala
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
                    'buttons': True
                }
            },
            'viber': {
                'type': 'text',
                'options': {
                    'rich_media': True,
                    'keyboard': True
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