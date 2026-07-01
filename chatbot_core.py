import openai
import numpy as np
from typing import List, Dict, Any
import json
import os
import re
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import traceback
from pathlib import Path
import requests
import bleach

logger = logging.getLogger(__name__)

class Intent(Enum):
    GREETING = "greeting"
    PRODUCT_QUESTION = "product_question"
    ORDER_STATUS = "order_status"
    RETURN_REQUEST = "return_request"
    PAYMENT_ISSUE = "payment_issue"
    CONTACT_SUPPORT = "contact_support"
    FAREWELL = "farewell"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    UNKNOWN = "unknown"

@dataclass
class ConversationMemory:
    user_id: str
    messages: List[Dict]
    last_updated: datetime
    context: Dict[str, Any]
    sentiment: float = 0.0
    escalation_needed: bool = False

class ContextAwareChatbot:
    def __init__(self, api_key: str, knowledge_base_path: str = None):
        self.api_key = api_key
        openai.api_key = api_key
        if knowledge_base_path is None:
            current_dir = Path(__file__).parent.absolute()
            knowledge_base_path = str(current_dir / "knowledge_base.json")
            logger.info(f"Koristim apsolutnu putanju: {knowledge_base_path}")
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        self.embedding_cache = {}
        self.active_conversations: Dict[str, ConversationMemory] = {}
        self.enrich_knowledge_base()
        logger.info(f"Chatbot inicijalizovan sa {len(self.knowledge_base)} stavki u bazi znanja")

    # --------------------------- Pomoćne funkcije ---------------------------
    def get_embedding(self, text: str) -> List[float]:
        cache_key = f"emb_{hash(text)}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        try:
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            embedding = response['data'][0]['embedding']
            if len(self.embedding_cache) < 1000:
                self.embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Greška pri generisanju embeddinga: {str(e)}")
            return [0.0] * 1536

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def load_knowledge_base(self, path: str) -> List[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                text_for_embedding = f"{item.get('question', '')} {item.get('answer', '')} {item.get('keywords', '')}"
                item['text_for_embedding'] = text_for_embedding
            logger.info(f"Učitano {len(data)} stavki iz baze znanja (ručno)")
            return data
        except FileNotFoundError:
            logger.error(f"Baza znanja nije pronađena na putanji: {path}")
            return []
        except Exception as e:
            logger.error(f"Greška pri učitavanju baze znanja: {str(e)}")
            return []

    # --------------------------- Automatsko učenje ---------------------------
    def enrich_knowledge_base(self):
        base_url = "https://zapmoto.rs/wp-json/wp/v2"
        # Vodič
        try:
            resp = requests.get(f"{base_url}/pages", params={"slug": "vodic-za-kupovinu"}, timeout=15)
            if resp.status_code == 200 and resp.json():
                page = resp.json()[0]
                content = page.get('content', {}).get('rendered', '')
                clean = bleach.clean(content, tags=[], attributes={}, strip=True)
                entry = {
                    "id": 300,
                    "question": "Vodič za kupovinu električnog skutera",
                    "answer": clean,
                    "keywords": "vodič, kupovina, izbor, skuter, saveti, kako izabrati, domet, snaga",
                    "category": "vodic",
                    "source": "vodic"
                }
                entry['text_for_embedding'] = f"{entry['question']} {entry['answer']} {entry['keywords']}"
                self.knowledge_base.append(entry)
                logger.info("✅ Vodič dodat iz WordPress-a.")
            else:
                logger.warning("Stranica 'vodic-za-kupovinu' nije pronađena.")
        except Exception as e:
            logger.error(f"❌ Greška pri preuzimanju vodiča: {e}")

        # Blog
        try:
            resp = requests.get(f"{base_url}/posts", params={"per_page": 50, "orderby": "date", "order": "desc"}, timeout=15)
            if resp.status_code == 200:
                posts = resp.json()
                added = 0
                for post in posts:
                    if post.get('status') != 'publish':
                        continue
                    title = post.get('title', {}).get('rendered', '')
                    content = post.get('content', {}).get('rendered', '')
                    clean = bleach.clean(content, tags=[], attributes={}, strip=True)
                    entry = {
                        "id": 301 + added,
                        "question": title,
                        "answer": clean,
                        "keywords": "",
                        "category": "blog",
                        "source": "blog"
                    }
                    entry['text_for_embedding'] = f"{title} {clean}"
                    self.knowledge_base.append(entry)
                    added += 1
                logger.info(f"✅ Dodato {added} članaka bloga.")
            else:
                logger.error(f"WordPress API za članke nije dostupan (status {resp.status_code})")
        except Exception as e:
            logger.error(f"❌ Greška pri preuzimanju članaka: {e}")

        # Proizvodi
        self._fetch_products_from_wp(base_url)
        self._fetch_products_from_woocommerce()

    def _fetch_products_from_wp(self, base_url: str):
        try:
            for endpoint in ["/product", "/products"]:
                try:
                    resp = requests.get(f"{base_url}{endpoint}", params={"per_page": 100}, timeout=15)
                    if resp.status_code == 200:
                        products = resp.json()
                        if not products:
                            continue
                        added = 0
                        start_id = 1000
                        for prod in products:
                            if prod.get('status') != 'publish':
                                continue
                            title = prod.get('title', {}).get('rendered', '')
                            content = prod.get('content', {}).get('rendered', '')
                            clean = bleach.clean(content, tags=[], attributes={}, strip=True)
                            entry = {
                                "id": start_id + added,
                                "question": f"Koje su karakteristike {title}?",
                                "answer": clean,
                                "keywords": title.lower(),
                                "category": "skuteri",
                                "source": "proizvodi"
                            }
                            entry['text_for_embedding'] = f"{entry['question']} {clean} {title.lower()}"
                            self.knowledge_base.append(entry)
                            added += 1
                        logger.info(f"✅ Automatski dodato {added} proizvoda sa WP endpointa {endpoint}.")
                        return
                except Exception:
                    continue
            logger.warning("Nijedan javni WP endpoint za proizvode nije radio.")
        except Exception as e:
            logger.error(f"❌ Greška pri automatskom preuzimanju proizvoda (WP): {e}")

    def _fetch_products_from_woocommerce(self):
        consumer_key = os.getenv('WOOCOMMERCE_CONSUMER_KEY')
        consumer_secret = os.getenv('WOOCOMMERCE_CONSUMER_SECRET')
        if not consumer_key or not consumer_secret:
            logger.info("WooCommerce API ključevi nisu podešeni – preskačem.")
            return
        try:
            url = "https://zapmoto.rs/wp-json/wc/v3/products"
            params = {"per_page": 100, "consumer_key": consumer_key, "consumer_secret": consumer_secret}
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                products = resp.json()
                added = 0
                start_id = 2000
                for prod in products:
                    if prod.get('status') != 'publish':
                        continue
                    name = prod.get('name', '')
                    description = prod.get('description', '')
                    clean_desc = bleach.clean(description, tags=[], attributes={}, strip=True)
                    cats = [c['name'].lower() for c in prod.get('categories', [])]
                    category = "skuteri" if any('skuter' in c for c in cats) else "motocikli"
                    entry = {
                        "id": start_id + added,
                        "question": f"Koje su karakteristike {name}?",
                        "answer": clean_desc,
                        "keywords": name.lower(),
                        "category": category,
                        "source": "proizvodi"
                    }
                    entry['text_for_embedding'] = f"{entry['question']} {clean_desc} {name.lower()}"
                    self.knowledge_base.append(entry)
                    added += 1
                logger.info(f"✅ Automatski dodato {added} proizvoda sa WooCommerce API-ja.")
            else:
                logger.error(f"WooCommerce API greška: {resp.status_code}")
        except Exception as e:
            logger.error(f"❌ Greška pri preuzimanju proizvoda preko WooCommerce API-ja: {e}")

    # --------------------------- Pretraga i namera ---------------------------
    def retrieve_relevant_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.knowledge_base:
            return []
        try:
            query_embedding = self.get_embedding(query)
            similarities = []
            for idx, item in enumerate(self.knowledge_base):
                item_text = item.get('text_for_embedding', '')
                item_key = f"kb_{idx}"
                if item_key not in self.embedding_cache:
                    self.embedding_cache[item_key] = self.get_embedding(item_text)
                item_embedding = self.embedding_cache[item_key]
                similarity = self.cosine_similarity(query_embedding, item_embedding)
                similarities.append((similarity, idx, item))
            similarities.sort(reverse=True, key=lambda x: x[0])
            relevant_items = []
            for i in range(min(top_k, len(similarities))):
                if similarities[i][0] > 0.25:
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
        message_lower = message.lower().strip()
        greetings = [
            'dobro jutro', 'dobar dan', 'dobro veče', 'dobro vece',
            'zdravo', 'ćao', 'cao', 'hej', 'pozdrav', 'pozz',
            'halo', 'ej', 'dober dan', 'dober večer',
            'good morning', 'good afternoon', 'good evening', 'hello', 'hi'
        ]
        for greeting in greetings:
            if message_lower == greeting or message_lower.startswith(greeting + ' ') or message_lower.startswith(greeting + ','):
                return Intent.GREETING
        try:
            relevant = self.retrieve_relevant_knowledge(message_lower, top_k=1)
            if relevant and relevant[0].get('relevance_score', 0) > 0.3:
                return Intent.PRODUCT_QUESTION
        except:
            pass
        if conversation_history:
            last_assistant_msg = None
            for msg in reversed(conversation_history):
                if msg['role'] == 'assistant':
                    last_assistant_msg = msg
                    break
            if last_assistant_msg and last_assistant_msg.get('intent') == 'product_recommendation':
                rejection_keywords = ['ne sviđa', 'drugi', 'neki drugi', 'drugačiji', 'neću', 'ne želim', 'nemoj']
                if any(keyword in message_lower for keyword in rejection_keywords):
                    return Intent.PRODUCT_RECOMMENDATION
        # OpenAI fallback
        system_prompt = """
        Ti si AI asistent za detekciju namere. Na osnovu korisničke poruke i istorije razgovora,
        odredi koja je od sledećih namera najverovatnija:
        - greeting, product_question, product_recommendation, order_status, return_request, payment_issue, contact_support, farewell, unknown
        Vrati samo naziv namere, ništa drugo.
        """
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
            for intent in Intent:
                if intent.value == intent_str:
                    return intent
            return Intent.UNKNOWN
        except Exception as e:
            logger.error(f"Greška pri detekciji namere: {str(e)}")
            return Intent.UNKNOWN

    # --------------------------- Generisanje odgovora ---------------------------
    def generate_greeting_response(self, message: str, user_id: str, conversation_id: str = None, channel: str = "web") -> Dict[str, Any]:
        greeting_responses = {
            'dobro jutro': 'Dobro jutro! Kako vam mogu pomoći danas?',
            'dobar dan': 'Dobar dan! Drago mi je da ste tu. Kako vam mogu pomoći?',
            'dobro veče': 'Dobro veče! Kako vam mogu pomoći?',
            'zdravo': 'Zdravo! Dobrodošli. Kako vam mogu pomoći?',
            'ćao': 'Ćao! Kako vam mogu pomoći?',
            'cao': 'Ćao! Kako vam mogu pomoći?',
            'hej': 'Hej! Kako vam mogu pomoći?',
            'pozdrav': 'Pozdrav! Kako vam mogu pomoći?',
        }
        message_lower = message.lower()
        response_text = "Zdravo! Kako vam mogu pomoći?"
        for greeting, response in greeting_responses.items():
            if greeting in message_lower:
                response_text = response
                break
        memory_key = f"{user_id}:{conversation_id}" if conversation_id else user_id
        self.add_to_conversation(memory_key, {'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()})
        self.add_to_conversation(memory_key, {'role': 'assistant', 'content': response_text, 'timestamp': datetime.now().isoformat(), 'intent': 'greeting', 'knowledge_used': []})
        return {
            'response': response_text,
            'intent': 'greeting',
            'conversation_id': conversation_id,
            'escalation_needed': False,
            'knowledge_sources': [],
            'channel_specific': self.get_channel_specific_response(channel, response_text)
        }

    def extract_criteria_from_message(self, message: str) -> Dict[str, any]:
        criteria_prompt = f"""
        Na osnovu korisničkog pitanja: "{message}"
        Izdvoj kriterijume za preporuku električnog skutera/motocikla.
        Vrati SAMO JSON u formatu:
        {{"kategorija": "skuteri" ili "motocikli" ili null, "min_domet": broj ili null, "max_domet": broj ili null, "max_snaga": broj ili null, "kategorija_vozacke": "AM" ili "A1" ili null, "prenosna_baterija": true ili false ili null}}
        Ako neki kriterijum nije pomenut, vrati null. Vrati SAMO JSON, ništa drugo.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": criteria_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=200,
                temperature=0.3
            )
            criteria_text = response.choices[0].message.content.strip()
            if criteria_text.startswith('```json'):
                criteria_text = criteria_text.replace('```json', '').replace('```', '')
            elif criteria_text.startswith('```'):
                criteria_text = criteria_text.replace('```', '')
            criteria = json.loads(criteria_text.strip())
        except:
            criteria = {}
        for key in ['kategorija', 'min_domet', 'max_domet', 'max_snaga', 'kategorija_vozacke', 'prenosna_baterija']:
            if key not in criteria:
                criteria[key] = None
        return criteria

    def filter_models_by_criteria(self, criteria: Dict[str, any], message: str = "") -> List[Dict]:
        filtered = []
        for item in self.knowledge_base:
            if item.get('source') != 'proizvodi':
                continue
            answer = item.get('answer', '').lower()
            if 'kategorija' in criteria and criteria['kategorija']:
                if item.get('category') != criteria['kategorija']:
                    continue
            if 'kategorija_vozacke' in criteria and criteria['kategorija_vozacke'] == 'AM':
                if ('grad' in message.lower() or 'gradsku' in message.lower() or 'gradska' in message.lower()) and item.get('category') == 'skuteri':
                    filtered.append(item)
                    continue
            if 'kategorija_vozacke' in criteria and criteria['kategorija_vozacke']:
                vozacka = criteria['kategorija_vozacke'].lower()
                if vozacka == 'am' and item.get('category') != 'skuteri':
                    continue
                elif vozacka == 'a1' and 'a1 kategorija' not in answer:
                    continue
            domet_match = re.search(r'domet do (\d+) km', answer)
            if domet_match:
                domet = int(domet_match.group(1))
                domet_passed = True
                if 'min_domet' in criteria and criteria['min_domet']:
                    if domet < criteria['min_domet'] * 0.8:
                        domet_passed = False
                if domet_passed and 'max_domet' in criteria and criteria['max_domet']:
                    if domet > criteria['max_domet'] * 1.2:
                        domet_passed = False
                if not domet_passed:
                    continue
            else:
                if 'min_domet' in criteria or 'max_domet' in criteria:
                    continue
            filtered.append(item)
        return filtered

    def extract_brand_from_query(self, query: str) -> str:
        for brand in ['pusa', 'lipo', 'deer', 'e2go', 'puma', 'tiger', 'lion']:
            if brand in query.lower():
                return brand
        return ""

    def format_product_response(self, relevant_items: List[Dict], original_query: str = "") -> str:
        if not relevant_items:
            return ""
        kontakt_items = [item for item in relevant_items if item['content'].get('category') == 'kontakt']
        if kontakt_items and any(word in original_query.lower() for word in ['kontakt', 'telefon', 'email', 'pozov', 'javite', 'obratim', 'obratiti', 'kontaktirati', 'pisati', 'poruka']):
            relevant_items = kontakt_items
        else:
            brand = self.extract_brand_from_query(original_query)
            if brand:
                filtered = [item for item in relevant_items if brand in item['content'].get('question', '').lower()]
                if filtered:
                    relevant_items = filtered
        response_parts = ["Na osnovu vašeg pitanja, evo relevantnih informacija:"]
        for idx, item in enumerate(relevant_items, 1):
            content = item['content']
            question = content.get('question', '')
            answer = content.get('answer', '')
            model_name = question.replace("Koje su karakteristike ", "").replace("?", "").strip()
            response_parts.append(f"<br><br>{idx}. <strong>{model_name}</strong><br>")
            clean_answer = answer.replace('<a href="', '<a target="_blank" href="')
            response_parts.append(f"&nbsp;&nbsp;&nbsp;- {clean_answer}<br>")
        return "".join(response_parts)

    def offer_contact_options(self, message: str, user_id: str, conversation_id: str = None, channel: str = "web") -> Dict[str, Any]:
        phone = "+381603534000"
        contact_message = f"""
Nažalost, nemam odgovor na ovo pitanje.

Za dodatnu pomoć, možete nas kontaktirati putem:

<br><br>
<div style="margin-bottom: 20px;">
    <a href="https://wa.me/{phone}" target="_blank" style="color: #25D366; text-decoration: none; font-size: 18px; display: flex; align-items: center;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 8px;"><path d="M19.077 4.928C17.191 3.041 14.683 2 12.006 2 6.798 2 2.548 6.193 2.54 11.393c-.003 1.747.456 3.457 1.328 4.984L2.25 21.75l5.428-1.573c1.472.839 3.137 1.286 4.857 1.288h.004c5.192 0 9.457-4.193 9.465-9.393.004-2.51-.972-4.872-2.857-6.758l-.07-.069zM12.03 20.026h-.003c-1.5-.001-2.97-.405-4.248-1.166l-.305-.182-3.222.934.86-3.144-.189-.312a7.925 7.925 0 0 1-1.222-4.222c.006-4.385 3.576-7.96 7.976-7.96 2.13 0 4.13.83 5.636 2.34 1.506 1.509 2.334 3.514 2.33 5.636-.005 4.386-3.576 7.961-7.973 7.961l-.04-.005z" fill="#25D366"/><path d="M16.11 13.454c-.266-.133-1.574-.774-1.818-.863-.244-.089-.422-.133-.599.133-.177.267-.688.863-.843 1.04-.155.178-.31.2-.577.067-.886-.333-1.682-.883-2.256-1.596-.178-.2-.322-.417-.454-.642.056-.033.11-.067.16-.106.088-.066.176-.133.26-.207.295-.257.534-.565.698-.911.027-.056.043-.118.048-.18.005-.063-.008-.127-.036-.185l-.424-.994c-.1-.233-.312-.39-.56-.413-.09-.008-.18-.003-.268.012-.15.021-.294.075-.418.156-.021.014-.041.029-.06.046-.359.316-.653.698-.863 1.127-.015.033-.026.067-.033.102-.094.378-.084.776.029 1.148.331 1.072.92 2.053 1.722 2.862.064.064.13.126.197.187.228.207.469.4.721.578.313.22.645.411.991.571.145.068.293.129.444.184.399.144.812.25 1.232.316.122.02.246.03.369.032.175.003.347-.021.512-.07.18-.048.341-.144.466-.277.192-.197.336-.436.422-.699.043-.133.055-.272.034-.408-.018-.12-.064-.234-.132-.334-.082-.117-.425-.716-.544-.878-.076-.1-.166-.132-.245-.132-.06 0-.12.016-.218.068-.275.146-.483.238-.609.289-.106.043-.187.066-.278-.022-.177-.177-.416-.407-.553-.549-.162-.17-.276-.381-.333-.61.09-.062.235-.15.358-.218.168-.093.31-.195.399-.282.181-.178.275-.409.293-.656.008-.092-.007-.184-.042-.27-.028-.07-.1-.222-.136-.298l-.232-.487c-.04-.084-.078-.168-.115-.253-.025-.058-.065-.11-.116-.148z" fill="#25D366"/></svg>
        <span style="font-weight: bold; color: #25D366;">WhatsApp</span>
    </a>
</div>
<div style="margin-bottom: 20px;">
    <a href="viber://chat?number={phone}" target="_blank" style="color: #7360F2; text-decoration: none; font-size: 18px; display: flex; align-items: center;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 8px;"><path d="M11.995 2C7.58 2 4 5.58 4 9.995c0 1.732.58 3.415 1.584 4.812L4.5 19.5l4.774-1.125c1.34.82 2.881 1.267 4.53 1.267 4.415 0 8-3.58 8-7.995C21.804 5.58 18.41 2 13.995 2h-2z" fill="#7360F2"/><path d="M15.5 13.4c-.3.3-.7.4-1.1.2-.9-.3-2.3-1.1-3.3-2.2-.9-.9-1.6-1.9-1.9-2.8-.1-.4 0-.8.2-1.1l.5-.5c.3-.3.3-.8 0-1.1l-1.1-1.1c-.3-.3-.8-.3-1.1 0l-.5.5c-.6.6-.8 1.5-.5 2.3.5 1.3 1.5 2.8 2.9 4.2 1.4 1.4 2.9 2.3 4.2 2.9.8.3 1.7.1 2.3-.5l.5-.5c.3-.3.3-.8 0-1.1l-1.1-1.1c-.3-.3-.8-.3-1.1 0l-.5.5z" fill="#FFFFFF"/></svg>
        <span style="font-weight: bold; color: #7360F2;">Viber</span>
    </a>
</div>
<div style="margin-bottom: 20px;">
    <a href="sms:{phone}" style="color: #34B7F1; text-decoration: none; font-size: 18px; display: flex; align-items: center;">
        <span style="font-size: 24px; vertical-align: middle; margin-right: 8px;">✉️</span>
        <span style="font-weight: bold; color: #34B7F1;">SMS</span>
    </a>
</div>
<br>
Naš tim će vam rado pomoći u najkraćem mogućem roku.

Da li mogu da vam pomognem oko nečeg drugog?
"""
        memory_key = f"{user_id}:{conversation_id}" if conversation_id else user_id
        self.add_to_conversation(memory_key, {'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()})
        self.add_to_conversation(memory_key, {'role': 'assistant', 'content': contact_message, 'timestamp': datetime.now().isoformat(), 'intent': 'contact_offered', 'knowledge_used': []})
        return {
            'response': contact_message,
            'intent': 'contact_offered',
            'conversation_id': conversation_id,
            'escalation_needed': False,
            'knowledge_sources': [],
            'channel_specific': self.get_channel_specific_response(channel, contact_message)
        }

    def generate_recommendation_response(self, message: str, criteria: Dict[str, any], conversation_history: List[Dict]) -> tuple:
        previous_models = []
        if conversation_history:
            for msg in reversed(conversation_history[-10:]):
                if msg.get('intent') == 'product_recommendation' and 'recommended_models' in msg:
                    previous_models.extend(msg.get('recommended_models', []))
        matching = self.filter_models_by_criteria(criteria, message)
        if previous_models:
            matching = [m for m in matching if m.get('id') not in previous_models]
        def get_domet(m): return int(re.search(r'domet do (\d+) km', m.get('answer', '')).group(1)) if re.search(r'domet do (\d+) km', m.get('answer', '')) else 0
        matching.sort(key=get_domet, reverse=True)
        if not matching:
            return "Nažalost, trenutno nemamo modele koji odgovaraju tvojim kriterijumima.", []
        parts = ["Na osnovu tvojih kriterijuma, preporučujem sledeće modele:\n\n"]
        recommended_ids = []
        for i, model in enumerate(matching[:5], 1):
            mid = model.get('id')
            if mid: recommended_ids.append(mid)
            q = model.get('question', '')
            name = q.replace("Koje su karakteristike ", "").replace("?", "").strip()
            ans = model.get('answer', '')
            domet = re.search(r'domet do (\d+) km', ans).group(1) if re.search(r'domet do (\d+) km', ans) else "?"
            snaga = re.search(r'snagu ([\d\.]+) kw', ans.lower()) or re.search(r'(\d+(?:\.\d+)?)\s*kw', ans.lower())
            snaga = snaga.group(1) if snaga else "?"
            vozacka = "AM" if "am kategorija" in ans.lower() else "A1" if "a1 kategorija" in ans.lower() else "?"
            link = re.search(r"https?://[^\s']+", ans)
            link = link.group(0) if link else "#"
            parts.append(f"{i}. <a href='{link}' target='_blank' style='color: #069806; text-decoration: underline; font-weight: bold;'>{name}</a> - Snaga: {snaga} kW, Domet: {domet} km, Kategorija: {vozacka}\n\n")
        parts.append("Ako želiš više informacija o nekom modelu, slobodno pitaj!")
        return "".join(parts), recommended_ids

    # ==================== GLAVNA generate_response ====================
    def generate_response(self, message: str, user_id: str, conversation_id: str = None, channel: str = "web") -> Dict[str, Any]:
        try:
            logger.info(f"===== GENERATE_RESPONSE POZVAN =====")
            memory_key = f"{user_id}:{conversation_id}" if conversation_id else user_id
            conversation = self.get_or_create_conversation(memory_key, user_id, channel)
            intent = self.detect_intent(message, conversation.messages)

            if intent == Intent.GREETING:
                return self.generate_greeting_response(message, user_id, conversation_id, channel)
            if self.should_escalate(message, intent, conversation):
                conversation.escalation_needed = True
                return self.prepare_escalation(message, conversation)
            if intent == Intent.PRODUCT_RECOMMENDATION:
                criteria = self.extract_criteria_from_message(message)
                response_text, recommended_ids = self.generate_recommendation_response(message, criteria, conversation.messages)
                self.add_to_conversation(memory_key, {'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()})
                self.add_to_conversation(memory_key, {'role': 'assistant', 'content': response_text, 'timestamp': datetime.now().isoformat(), 'intent': intent.value, 'recommended_models': recommended_ids, 'knowledge_used': []})
                return {'response': response_text, 'intent': intent.value, 'conversation_id': conversation_id, 'escalation_needed': False, 'knowledge_sources': [], 'channel_specific': self.get_channel_specific_response(channel, response_text)}

            # ---------- NOVO: DIREKTNO IZLISTAVANJE VESTI/BLOGA ----------
            news_keywords = ['vesti', 'novosti', 'članci', 'blog', 'najnovije', 'vest', 'novost', 'članak']
            if any(keyword in message.lower() for keyword in news_keywords):
                blog_entries = [item for item in self.knowledge_base if item.get('source') == 'blog']
                if blog_entries:
                    blog_entries.sort(key=lambda x: x.get('id', 0), reverse=True)
                    top = blog_entries[:5]
                    parts = ["📰 **Najnovije vesti i članci:**\n"]
                    for i, blog in enumerate(top, 1):
                        title = blog.get('question', 'Bez naslova')
                        answer = blog.get('answer', '')
                        link_match = re.search(r"https?://[^\s']+", answer)
                        link = link_match.group(0) if link_match else "#"
                        parts.append(f"{i}. <a href='{link}' target='_blank' style='color: #069806; text-decoration: underline;'>{title}</a>\n")
                    response_text = "".join(parts)
                    self.add_to_conversation(memory_key, {'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()})
                    self.add_to_conversation(memory_key, {'role': 'assistant', 'content': response_text, 'timestamp': datetime.now().isoformat(), 'intent': intent.value, 'knowledge_used': ['blog']})
                    return {'response': response_text, 'intent': intent.value, 'conversation_id': conversation_id, 'escalation_needed': False, 'knowledge_sources': ['blog'], 'channel_specific': self.get_channel_specific_response(channel, response_text)}
                else:
                    response_text = "Trenutno nemamo dostupnih članaka. Pokušajte ponovo kasnije."
                    self.add_to_conversation(memory_key, {'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()})
                    self.add_to_conversation(memory_key, {'role': 'assistant', 'content': response_text, 'timestamp': datetime.now().isoformat(), 'intent': intent.value, 'knowledge_used': []})
                    return {'response': response_text, 'intent': intent.value, 'conversation_id': conversation_id, 'escalation_needed': False, 'knowledge_sources': [], 'channel_specific': self.get_channel_specific_response(channel, response_text)}

            logger.info("Standardna obrada pitanja...")
            relevant_knowledge = self.retrieve_relevant_knowledge(message)
            has_good_answer = False
            best_score = 0
            if relevant_knowledge:
                best_score = relevant_knowledge[0].get('relevance_score', 0)
                if best_score > 0.29:
                    has_good_answer = True
            if not has_good_answer:
                logger.info(f"Nema dovoljno relevantnog odgovora (najbolji score: {best_score:.2f})")
                return self.offer_contact_options(message, user_id, conversation_id, channel)

            top_item = relevant_knowledge[0]['content']
            if top_item.get('source') in ['vodic', 'blog']:
                knowledge_text = ""
                for item in relevant_knowledge[:3]:
                    content = item['content']
                    knowledge_text += f"Naslov: {content.get('question', '')}\nSadržaj: {content.get('answer', '')}\n\n"
                context = {
                    'relevant_knowledge': relevant_knowledge,
                    'conversation_history': conversation.messages[-6:],
                    'intent': intent.value,
                    'channel': channel
                }
                response_text = self.generate_llm_response(message, context)
            elif top_item.get('source') == 'kontakt' or top_item.get('category') == 'kontakt':
                contact_answer = top_item.get('answer', '')
                response_text = f"Naš tim vam stoji na raspolaganju:\n\n{contact_answer}"
            else:
                response_text = self.format_product_response(relevant_knowledge, message)

            self.add_to_conversation(memory_key, {'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()})
            self.add_to_conversation(memory_key, {'role': 'assistant', 'content': response_text, 'timestamp': datetime.now().isoformat(), 'intent': intent.value, 'knowledge_used': [k['source'] for k in relevant_knowledge] if relevant_knowledge else []})
            return {
                'response': response_text,
                'intent': intent.value,
                'conversation_id': conversation_id,
                'escalation_needed': False,
                'knowledge_sources': [k['source'] for k in relevant_knowledge] if relevant_knowledge else [],
                'channel_specific': self.get_channel_specific_response(channel, response_text)
            }
        except Exception as e:
            logger.error(f"❌ Greška u generate_response: {str(e)}")
            return {
                'response': "Došlo je do tehničke greške. Molim vas pokušajte ponovo ili kontaktirajte podršku.",
                'intent': 'unknown',
                'conversation_id': conversation_id,
                'escalation_needed': True,
                'knowledge_sources': []
            }

    def generate_llm_response(self, message: str, context: Dict) -> str:
        knowledge_text = ""
        if context['relevant_knowledge']:
            knowledge_text = "Relevantne informacije:\n"
            for idx, item in enumerate(context['relevant_knowledge']):
                content = item['content']
                knowledge_text += f"{idx+1}. Pitanje: {content.get('question', 'Informacija')}\n"
                knowledge_text += f"   Odgovor: {content.get('answer', content.get('content', ''))}\n"
                knowledge_text += f"   Izvor: {content.get('source', 'baza znanja')}\n\n"
        history_text = ""
        if context['conversation_history']:
            history_text = "Istorija razgovora:\n"
            for msg in context['conversation_history']:
                role = "Korisnik" if msg['role'] == 'user' else "Asistent"
                history_text += f"{role}: {msg['content']}\n"
        system_prompt = f"""
        Ti si profesionalni AI asistent za korisničku podršku i e-trgovinu.
        VAŽNA UPUTSTVA:
        1. Budi koncizan, ali ljubazan.
        2. Odgovaraj isključivo na osnovu dostupnih informacija.
        3. Nemoj izmišljati informacije.
        4. Ako nema odgovora, reci da ćeš proslediti agentu.
        Detektovana namera: {context['intent']}
        Kanal: {context['channel']}
        """
        user_prompt = f"""
        {history_text}
        {knowledge_text}
        Korisnik pita: {message}
        Molim te da odgovoriš na ovo pitanje na osnovu dostupnih informacija.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()
            response_text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" style="color: #069806; text-decoration: underline;">\1</a>', response_text)
            return response_text
        except Exception as e:
            logger.error(f"Greška pri generisanju odgovora: {str(e)}")
            return "Izvinite, došlo je do tehničke greške."

    def should_escalate(self, message: str, intent: Intent, conversation: ConversationMemory) -> bool:
        escalation_keywords = ['agent', 'operater', 'čovek', 'govori sa', 'uživo', 'live chat', 'kontakt']
        if any(keyword in message.lower() for keyword in escalation_keywords):
            return True
        if intent == Intent.CONTACT_SUPPORT:
            return True
        recent = conversation.messages[-6:]
        if sum(1 for msg in recent if msg.get('intent') == 'unknown') >= 3:
            return True
        return False

    def prepare_escalation(self, message: str, conversation: ConversationMemory) -> Dict[str, Any]:
        transcript = [{'role': msg['role'], 'content': msg['content'], 'timestamp': msg.get('timestamp', '')} for msg in conversation.messages]
        transcript.append({'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()})
        return {
            'response': "Povezujem vas sa našim agentom za korisničku podršku. Molim vas sačekajte trenutak.",
            'escalation_needed': True,
            'escalation_data': {'transcript': transcript, 'conversation_id': conversation.user_id},
            'channel_specific': self.get_channel_specific_response(conversation.context.get('channel', 'web'), None, escalation=True)
        }

    def get_or_create_conversation(self, memory_key: str, user_id: str, channel: str) -> ConversationMemory:
        if memory_key in self.active_conversations:
            return self.active_conversations[memory_key]
        conv = ConversationMemory(user_id=user_id, messages=[], last_updated=datetime.now(), context={'channel': channel, 'start_time': datetime.now().isoformat()})
        self.active_conversations[memory_key] = conv
        return conv

    def add_to_conversation(self, memory_key: str, message: Dict):
        if memory_key not in self.active_conversations:
            return
        conv = self.active_conversations[memory_key]
        conv.messages.append(message)
        conv.last_updated = datetime.now()
        if 'intent' in message:
            conv.context['last_intent'] = message['intent']

    def get_channel_specific_response(self, channel: str, text: str = None, escalation: bool = False) -> Dict:
        configs = {
            'web': {'type': 'text', 'options': {'quick_replies': True, 'rich_text': True, 'buttons': True}},
            'whatsapp': {'type': 'text', 'options': {'max_length': 4096, 'interactive': True, 'buttons': True}},
            'viber': {'type': 'text', 'options': {'rich_media': True, 'keyboard': True}},
            'telegram': {'type': 'text', 'options': {'markdown': True, 'inline_keyboards': True}}
        }
        config = configs.get(channel, configs['web'])
        resp = {'channel': channel, 'type': config['type'], 'config': config['options']}
        if text:
            resp['text'] = text
        if escalation:
            resp['escalation'] = {'message': "Povezivanje sa agentom...", 'estimated_wait': "2-3 minuta"}
        return resp