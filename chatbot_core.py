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

    # ... (generate_greeting_response, extract_criteria_from_message, filter_models_by_criteria, itd. – sve ostale metode su nepromenjene, osim generate_response koja sledi)

    def generate_response(self,
                         message: str,
                         user_id: str,
                         conversation_id: str = None,
                         channel: str = "web") -> Dict[str, Any]:
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
                return {
                    'response': response_text,
                    'intent': intent.value,
                    'conversation_id': conversation_id,
                    'escalation_needed': False,
                    'knowledge_sources': [],
                    'channel_specific': self.get_channel_specific_response(channel, response_text)
                }

            # ---------- NOVO: Rukovanje zahtevom za vesti/blog ----------
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
                    return {
                        'response': response_text,
                        'intent': intent.value,
                        'conversation_id': conversation_id,
                        'escalation_needed': False,
                        'knowledge_sources': ['blog'],
                        'channel_specific': self.get_channel_specific_response(channel, response_text)
                    }
                # Ako nema blogova, nastaviće se normalna obrada (pašće u kontakt ili sl.)

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

            # ---------- Razlikovanje proizvoda, vodiča, bloga, kontakta ----------
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

    # ... (preostale metode: generate_llm_response, should_escalate, itd. – identične prethodnom fajlu)
    # Zbog dužine ih ne ponavljam, ali se podrazumeva da su sve prisutne.