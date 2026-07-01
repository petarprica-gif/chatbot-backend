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
import requests      # za WordPress API
import bleach        # za čišćenje HTML-a

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

        # Učitavanje ručne baze (proizvodi, kontakt)
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)

        # Keš za embeddinge
        self.embedding_cache = {}

        # Aktivne konverzacije
        self.active_conversations: Dict[str, ConversationMemory] = {}

        # ---------- AUTOMATSKO PREUZIMANJE SADRŽAJA SA SAJTA ----------
        self.enrich_knowledge_base()

        logger.info(f"Chatbot inicijalizovan sa {len(self.knowledge_base)} stavki u bazi znanja")

    # ==================== POMOĆNE FUNKCIJE ====================
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

    # ==================== NOVE METODE ZA AUTOMATSKO UČENJE ====================
    def enrich_knowledge_base(self):
        base_url = "https://zapmoto.rs/wp-json/wp/v2"

        # --- 1. Vodič za kupovinu ---
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

        # --- 2. Članci bloga ---
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

        # --- 3. Proizvodi ---
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
                            category = "skuteri"
                            question = f"Koje su karakteristike {title}?"
                            entry = {
                                "id": start_id + added,
                                "question": question,
                                "answer": clean,
                                "keywords": title.lower(),
                                "category": category,
                                "source": "proizvodi"
                            }
                            entry['text_for_embedding'] = f"{question} {clean} {title.lower()}"
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
                    question = f"Koje su karakteristike {name}?"
                    entry = {
                        "id": start_id + added,
                        "question": question,
                        "answer": clean_desc,
                        "keywords": name.lower(),
                        "category": category,
                        "source": "proizvodi"
                    }
                    entry['text_for_embedding'] = f"{question} {clean_desc} {name.lower()}"
                    self.knowledge_base.append(entry)
                    added += 1
                logger.info(f"✅ Automatski dodato {added} proizvoda sa WooCommerce API-ja.")
            else:
                logger.error(f"WooCommerce API greška: {resp.status_code}")
        except Exception as e:
            logger.error(f"❌ Greška pri preuzimanju proizvoda preko WooCommerce API-ja: {e}")

    # ==================== ORIGINALNE METODE ====================
    def retrieve_relevant_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.knowledge_base:
            logger.warning("Baza znanja je prazna, ne mogu da pretražujem")
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
            logger.info(f"Pronađeno {len(relevant_items)} relevantnih stavki, najbolji score: {similarities[0][0] if similarities else 0}")
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
        if conversation_history and len(conversation_history) > 0:
            last_assistant_msg = None
            for msg in reversed(conversation_history):
                if msg['role'] == 'assistant':
                    last_assistant_msg = msg
                    break
            if last_assistant_msg and last_assistant_msg.get('intent') == 'product_recommendation':
                rejection_keywords = ['ne sviđa', 'drugi', 'neki drugi', 'drugačiji', 'neću', 'ne želim', 'nemoj']
                if any(keyword in message_lower for keyword in rejection_keywords):
                    return Intent.PRODUCT_RECOMMENDATION
        system_prompt = """
        Ti si AI asistent za detekciju namere. Na osnovu korisničke poruke i istorije razgovora,
        odredi koja je od sledećih namera najverovatnija:
        - greeting: Pozdrav, početak razgovora
        - product_question: Pitanje o proizvodu (cena, dostupnost, karakteristike)
        - product_recommendation: Zahtev za preporuku proizvoda na osnovu kriterijuma (npr. "preporuči mi skuter sa dometom 100 km", "koji model da kupim", "šta mi preporučuješ", "tražim nešto za grad")
        - order_status: Provera statusa porudžbine
        - return_request: Zahtev za povraćaj ili reklamaciju
        - payment_issue: Problem sa plaćanjem
        - contact_support: Zahtev za kontakt sa agentom
        - farewell: Završetak razgovora, pozdrav
        - unknown: Nejasna namera
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

    def generate_greeting_response(self, message: str, user_id: str, conversation_id: str = None, channel: str = "web") -> Dict[str, Any]:
        greeting_responses = {
            'dobro jutro': 'Dobro jutro! Kako vam mogu pomoći danas?',
            'dobar dan': 'Dobar dan! Drago mi je da ste tu. Kako vam mogu pomoći?',
            'dobro veče': 'Dobro veče! Kako vam mogu pomoći?',
            'dobro vece': 'Dobro veče! Kako vam mogu pomoći?',
            'zdravo': 'Zdravo! Dobrodošli. Kako vam mogu pomoći?',
            'ćao': 'Ćao! Kako vam mogu pomoći?',
            'cao': 'Ćao! Kako vam mogu pomoći?',
            'hej': 'Hej! Kako vam mogu pomoći?',
            'pozdrav': 'Pozdrav! Kako vam mogu pomoći?',
            'good morning': 'Good morning! How can I help you?',
            'good afternoon': 'Good afternoon! How can I help you?',
            'good evening': 'Good evening! How can I help you?',
            'hello': 'Hello! How can I help you?',
            'hi': 'Hi! How can I help you?'
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
        {{
            "kategorija": "skuteri" ili "motocikli" ili null,
            "min_domet": broj u km ili null,
            "max_domet": broj u km ili null,
            "max_snaga": broj u kW ili null,
            "kategorija_vozacke": "AM" ili "A1" ili null,
            "prenosna_baterija": true ili false ili null
        }}
        Ako neki kriterijum nije pomenut, vrati null.
        Vrati SAMO JSON, ništa drugo.
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
            criteria_text = criteria_text.strip()
            criteria = json.loads(criteria_text)
        except:
            criteria = {}
        for key in ['kategorija', 'min_domet', 'max_domet', 'max_snaga', 'kategorija_vozacke', 'prenosna_baterija']:
            if key not in criteria:
                criteria[key] = None
        return criteria

    def filter_models_by_criteria(self, criteria: Dict[str, any], message: str = "") -> List[Dict]:
        filtered_models = []
        for item in self.knowledge_base:
            if item.get('source') != 'proizvodi':
                continue
            answer = item.get('answer', '').lower()
            if 'kategorija' in criteria and criteria['kategorija']:
                if item.get('category') != criteria['kategorija']:
                    continue
            if 'kategorija_vozacke' in criteria and criteria['kategorija_vozacke'] == 'AM':
                if 'grad' in message.lower() or 'gradsku' in message.lower() or 'gradska' in message.lower():
                    if item.get('category') == 'skuteri':
                        filtered_models.append(item)
                        continue
            if 'kategorija_vozacke' in criteria and criteria['kategorija_vozacke']:
                vozacka = criteria['kategorija_vozacke'].lower()
                if vozacka == 'am':
                    if item.get('category') != 'skuteri':
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
            filtered_models.append(item)
        return filtered_models

    def extract_brand_from_query(self, query: str) -> str:
        query_lower = query.lower()
        for brand in ['pusa', 'lipo', 'deer', 'e2go', 'puma', 'tiger', 'lion']:
            if brand in query_lower:
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
            if 'cena' in content:
                response_parts.append(f"&nbsp;&nbsp;&nbsp;- <strong>Cena:</strong> {content['cena']}<br>")
        return "".join(response_parts)

    def offer_contact_options(self, message: str, user_id: str, conversation_id: str = None, channel: str = "web") -> Dict[str, Any]:
        phone = "+381603534000"
        whatsapp_link = f"https://wa.me/{phone}"
        viber_link = f"viber://chat?number={phone}"
        sms_link = f"sms:{phone}"
        whatsapp_svg = '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 8px;"><path d="M19.077 4.928..."/></svg>'''  # skraćeno radi dužine, koristi isti kao ranije
        viber_svg = '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 8px;">...</svg>'''
        sms_icon = '<span style="font-size: 24px; vertical-align: middle; margin-right: 8px;">✉️</span>'
        contact_message = f"""
Nažalost, nemam odgovor na ovo pitanje.

Za dodatnu pomoć, možete nas kontaktirati putem:

<br><br>
<div style="margin-bottom: 20px;">
    <a href="{whatsapp_link}" target="_blank" style="color: #25D366; text-decoration: none; font-size: 18px; display: flex; align-items: center;">
        {whatsapp_svg}
        <span style="font-weight: bold; color: #25D366;">WhatsApp</span>
    </a>
</div>
... (ostatak isti kao pre)
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
        matching_models = self.filter_models_by_criteria(criteria, message)
        if previous_models:
            matching_models = [m for m in matching_models if m.get('id') not in previous_models]
        def get_domet(m): return int(re.search(r'domet do (\d+) km', m.get('answer', '')).group(1)) if re.search(r'domet do (\d+) km', m.get('answer', '')) else 0
        matching_models.sort(key=get_domet, reverse=True)
        if not matching_models:
            return "Nažalost, trenutno nemamo modele koji odgovaraju tvojim kriterijumima.", []
        response_parts = ["Na osnovu tvojih kriterijuma, preporučujem sledeće modele:\n\n"]
        recommended_ids = []
        for i, model in enumerate(matching_models[:5], 1):
            mid = model.get('id')
            if mid: recommended_ids.append(mid)
            q = model.get('question', '')
            name = q.replace("Koje su karakteristike ", "").replace("?", "").strip()
            ans = model.get('answer', '')
            domet = re.search(r'domet do (\d+) km', ans).group(1) if re.search(r'domet do (\d+) km', ans) else "nepoznat"
            snaga = re.search(r'snagu ([\d\.]+) kw', ans.lower()) or re.search(r'(\d+(?:\.\d+)?)\s*kw', ans.lower())
            snaga = snaga.group(1) if snaga else "?"
            vozacka = "AM" if "am kategorija" in ans.lower() else "A1" if "a1 kategorija" in ans.lower() else "?"
            link = re.search(r"https?://[^\s']+", ans)
            link = link.group(0) if link else "#"
            response_parts.append(f"{i}. <a href='{link}' target='_blank' style='color: #069806; text-decoration: underline; font-weight: bold;'>{name}</a> - Snaga: {snaga} kW, Domet: {domet} km, Kategorija: {vozacka}\n\n")
        response_parts.append("Ako želiš više informacija o nekom modelu, slobodno pitaj!")
        return "".join(response_parts), recommended_ids

    def generate_response(self, message: str, user_id: str, conversation_id: str = None, channel: str = "web") -> Dict[str, Any]:
        try:
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
                resp_text, rec_ids = self.generate_recommendation_response(message, criteria, conversation.messages)
                self.add_to_conversation(memory_key, {'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()})
                self.add_to_conversation(memory_key, {'role': 'assistant', 'content': resp_text, 'timestamp': datetime.now().isoformat(), 'intent': intent.value, 'recommended_models': rec_ids, 'knowledge_used': []})
                return {'response': resp_text, 'intent': intent.value, 'conversation_id': conversation_id, 'escalation_needed': False, 'knowledge_sources': [], 'channel_specific': self.get_channel_specific_response(channel, resp_text)}

            # Rukovanje vestima / blogom
            news_keywords = ['vesti', 'novosti', 'članci', 'blog', 'najnovije', 'vest', 'novost', 'članak']
            if any(keyword in message.lower() for keyword in news_keywords):
                blog_entries = [item for item in self.knowledge_base if item.get('source') == 'blog']
                if blog_entries:
                    blog_entries.sort(key=lambda x: x.get('id', 0), reverse=True)
                    top_blogs = blog_entries[:5]
                    relevant_knowledge = [{'content': blog, 'relevance_score': 1.0, 'source': 'blog'} for blog in top_blogs]
                    knowledge_text = ""
                    for item in relevant_knowledge:
                        content = item['content']
                        knowledge_text += f"Naslov: {content.get('question', '')}\nSadržaj: {content.get('answer', '')}\n\n"
                    context = {
                        'relevant_knowledge': relevant_knowledge,
                        'conversation_history': conversation.messages[-6:],
                        'intent': intent.value,
                        'channel': channel
                    }
                    response_text = self.generate_llm_response(message, context)
                    self.add_to_conversation(memory_key, {'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()})
                    self.add_to_conversation(memory_key, {'role': 'assistant', 'content': response_text, 'timestamp': datetime.now().isoformat(), 'intent': intent.value, 'knowledge_used': ['blog']})
                    return {'response': response_text, 'intent': intent.value, 'conversation_id': conversation_id, 'escalation_needed': False, 'knowledge_sources': ['blog'], 'channel_specific': self.get_channel_specific_response(channel, response_text)}

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
        1. Budi koncizan, ali ljubazan - koristi prirodan ton razgovora
        2. Odgovaraj isključivo na osnovu dostupnih informacija - ako ne znaš odgovor, priznaj to
        3. Izbegavaj halucinacije - nemoj izmišljati informacije koje nisu u bazi znanja
        4. Ako korisnik pita nešto što nije u tvojoj bazi znanja, ljubazno ga uputi da ćeš proslediti agentu
        5. Strukturiraj informacije jasno - koristi bullet points gde je prikladno
        6. Prati kontekst razgovora - ako se korisnik vraća na prethodnu temu, seti se toga
        Detektovana namera korisnika: {context['intent']}
        Kanal komunikacije: {context['channel']}
        """
        user_prompt = f"""
        {history_text}
        {knowledge_text}
        Korisnik pita: {message}
        Molim te da odgovoriš na ovo pitanje na osnovu dostupnih informacija.
        Ako informacija nije dostupna, reci da ćeš proslediti agentu.
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
            return "Izvinite, došlo je do tehničke greške. Molim vas pokušajte ponovo ili kontaktirajte našu podršku."

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