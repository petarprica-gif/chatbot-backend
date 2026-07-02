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
    SAVINGS_CALCULATOR = "savings_calculator"
    ADVICE_NEEDED = "advice_needed"
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
    # -----------------------------------------------------------------
    # Konstante kalkulatora
    # -----------------------------------------------------------------
    DAYS_PER_YEAR = 365
    KURS_RSD = 117.4
    GAS_MAINT_RATE_RSD = (100/5000 + 325/20000) * KURS_RSD
    ELEC_MAINT_RATE_RSD = (325/20000) * KURS_RSD
    DEFAULT_MODEL_WH = 33          # Wh/km Deer 45
    DEFAULT_MODEL_NAME = "Deer 45"

    # -----------------------------------------------------------------
    # Tablica modela – Wh/km za preporuke
    # -----------------------------------------------------------------
    MODEL_WH = {
        "e2go": 33, "deer 45": 33, "pusa 45": 34, "lipo 45": 43,
        "deer 100": 37, "pusa 90": 43, "lipo 70": 43,
        "tiger": 49, "puma": 80, "lion": 57,
    }

    def __init__(self, api_key: str, knowledge_base_path: str = None):
        self.api_key = api_key
        openai.api_key = api_key
        if knowledge_base_path is None:
            current_dir = Path(__file__).parent.absolute()
            knowledge_base_path = str(current_dir / "knowledge_base.json")
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        self.embedding_cache = {}
        self.active_conversations: Dict[str, ConversationMemory] = {}
        self.enrich_knowledge_base()
        logger.info(f"Chatbot inicijalizovan sa {len(self.knowledge_base)} stavki u bazi znanja")

    # ---------- pomoćne ----------
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
            emb = response['data'][0]['embedding']
            if len(self.embedding_cache) < 1000:
                self.embedding_cache[cache_key] = emb
            return emb
        except Exception as e:
            logger.error(f"Embedding greška: {e}")
            return [0.0] * 1536

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a, b = np.array(a), np.array(b)
        dot = np.dot(a, b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(dot / (na * nb)) if na and nb else 0.0

    def load_knowledge_base(self, path: str) -> List[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                item['text_for_embedding'] = f"{item.get('question','')} {item.get('answer','')} {item.get('keywords','')}"
            logger.info(f"Ručna baza: {len(data)} stavki")
            return data
        except FileNotFoundError:
            logger.error(f"Baza nije nađena: {path}")
            return []
        except Exception as e:
            logger.error(f"Greška u bazi: {e}")
            return []

    # ---------- automatsko učenje ----------
    def enrich_knowledge_base(self):
        base_url = "https://zapmoto.rs/wp-json/wp/v2"
        # Vodič
        try:
            r = requests.get(f"{base_url}/pages", params={"slug": "vodic-za-kupovinu"}, timeout=15)
            if r.status_code == 200 and r.json():
                page = r.json()[0]
                txt = bleach.clean(page['content']['rendered'], tags=[], attributes={}, strip=True)
                entry = {"id":300, "question":"Vodič za kupovinu električnog skutera",
                         "answer":txt, "keywords":"vodič, kupovina, izbor, skuter, saveti, domet, snaga",
                         "category":"vodic", "source":"vodic"}
                entry['text_for_embedding'] = f"{entry['question']} {entry['answer']} {entry['keywords']}"
                self.knowledge_base.append(entry)
                logger.info("✅ Vodič dodat")
        except Exception as e:
            logger.error(f"Vodič greška: {e}")

        # Blog
        try:
            r = requests.get(f"{base_url}/posts", params={"per_page":50, "orderby":"date", "order":"desc"}, timeout=15)
            if r.status_code == 200:
                posts = r.json()
                added = 0
                for post in posts:
                    if post['status'] != 'publish': continue
                    title = post['title']['rendered']
                    content = bleach.clean(post['content']['rendered'], tags=[], attributes={}, strip=True)
                    entry = {"id":301+added, "question":title, "answer":content, "keywords":"",
                             "category":"blog", "source":"blog", "date":post.get('date','')}
                    entry['text_for_embedding'] = f"{title} {content}"
                    self.knowledge_base.append(entry)
                    added += 1
                logger.info(f"✅ {added} članaka bloga")
        except Exception as e:
            logger.error(f"Blog greška: {e}")

        # Proizvodi
        self._fetch_products_from_wp(base_url)
        self._fetch_products_from_woocommerce()

    def _fetch_products_from_wp(self, base_url):
        for ep in ["/product", "/products"]:
            try:
                r = requests.get(f"{base_url}{ep}", params={"per_page":100}, timeout=15)
                if r.status_code == 200:
                    prods = r.json()
                    if not prods: continue
                    added = 0
                    for p in prods:
                        if p['status'] != 'publish': continue
                        title = p['title']['rendered']
                        clean = bleach.clean(p['content']['rendered'], tags=[], attributes={}, strip=True)
                        entry = {"id":1000+added, "question":f"Koje su karakteristike {title}?",
                                 "answer":clean, "keywords":title.lower(),
                                 "category":"skuteri", "source":"proizvodi"}
                        entry['text_for_embedding'] = f"{entry['question']} {clean} {title.lower()}"
                        self.knowledge_base.append(entry)
                        added += 1
                    logger.info(f"✅ {added} proizvoda sa WP")
                    return
            except: continue
        logger.warning("WP proizvodi nisu dostupni")

    def _fetch_products_from_woocommerce(self):
        key = os.getenv('WOOCOMMERCE_CONSUMER_KEY')
        secret = os.getenv('WOOCOMMERCE_CONSUMER_SECRET')
        if not key or not secret:
            logger.info("WooCommerce ključevi nisu podešeni")
            return
        try:
            url = "https://zapmoto.rs/wp-json/wc/v3/products"
            params = {"per_page":100, "consumer_key":key, "consumer_secret":secret}
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                prods = r.json()
                added = 0
                for p in prods:
                    if p['status'] != 'publish': continue
                    name = p['name']
                    desc = bleach.clean(p.get('description',''), tags=[], attributes={}, strip=True)
                    cats = [c['name'].lower() for c in p.get('categories',[])]
                    cat = "skuteri" if any('skuter' in c for c in cats) else "motocikli"
                    entry = {"id":2000+added, "question":f"Koje su karakteristike {name}?",
                             "answer":desc, "keywords":name.lower(),
                             "category":cat, "source":"proizvodi",
                             "cena": p.get('price')}
                    entry['text_for_embedding'] = f"{entry['question']} {desc} {name.lower()}"
                    self.knowledge_base.append(entry)
                    added += 1
                logger.info(f"✅ {added} proizvoda sa WooCommerce")
        except Exception as e:
            logger.error(f"WooCommerce greška: {e}")

    # ---------- namera ----------
    def retrieve_relevant_knowledge(self, query: str, top_k=5) -> List[Dict]:
        if not self.knowledge_base: return []
        q_emb = self.get_embedding(query)
        sims = []
        for i, item in enumerate(self.knowledge_base):
            txt = item.get('text_for_embedding','')
            key = f"kb_{i}"
            if key not in self.embedding_cache:
                self.embedding_cache[key] = self.get_embedding(txt)
            sims.append((self.cosine_similarity(q_emb, self.embedding_cache[key]), i, item))
        sims.sort(reverse=True, key=lambda x: x[0])
        res = []
        for s in sims[:top_k]:
            if s[0] > 0.25:
                res.append({'content':s[2], 'relevance_score':float(s[0]),
                            'source':s[2].get('source','knowledge_base')})
        return res

    def detect_intent(self, message: str, history: List[Dict]) -> Intent:
        msg = message.lower().strip()

        # Savings calculator (samo ako se pominje ušteda)
        if any(w in msg for w in ['ušted','uštedeti','isplativost','ušteda']):
            return Intent.SAVINGS_CALCULATOR

        # Greetings
        greetings = ['dobro jutro','dobar dan','dobro veče','zdravo','ćao','cao','hej','pozdrav',
                     'good morning','good afternoon','good evening','hello','hi']
        for g in greetings:
            if msg == g or msg.startswith(g+' ') or msg.startswith(g+','):
                return Intent.GREETING

        # Advice / recommendation keywords
        rec_kws = ['preporuka','preporuči','savet','koji model','koji skuter','koji motor',
                   'izabrati','kupiti','kupovinu','potreban mi je savet',
                   'domet','kilometara','vozačku','am kategorija','a1 kategorija']
        if any(kw in msg for kw in rec_kws):
            return Intent.PRODUCT_RECOMMENDATION

        # Check knowledge base
        try:
            rel = self.retrieve_relevant_knowledge(msg, top_k=1)
            if rel and rel[0].get('relevance_score',0) > 0.3:
                return Intent.PRODUCT_QUESTION
        except: pass

        # Check conversation context for continuation of recommendation
        if history:
            last = None
            for m in reversed(history):
                if m['role'] == 'assistant':
                    last = m
                    break
            if last and last.get('intent') == 'product_recommendation':
                if any(kw in msg for kw in ['drugi','drugi model','još','drugi brend']):
                    return Intent.PRODUCT_RECOMMENDATION

        # Fallback OpenAI
        prompt = """Detektuj nameru korisnika. Vrati samo jedno od: greeting, product_question, product_recommendation, order_status, return_request, payment_issue, contact_support, farewell, unknown."""
        history_text = ""
        for m in history[-5:]:
            history_text += f"{m['role']}: {m['content']}\n"
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":prompt},
                    {"role":"user","content": f"Istorija:\n{history_text}\n\nPoruka: {message}"}
                ],
                max_tokens=10, temperature=0.3)
            intent_str = resp.choices[0].message.content.strip().lower()
            for i in Intent:
                if i.value == intent_str: return i
        except: pass
        return Intent.UNKNOWN

    # ---------- odgovori ----------
    def generate_greeting_response(self, message, user_id, conv_id=None, channel="web"):
        responses = {'dobro jutro':'Dobro jutro! ☀️ Kako vam mogu pomoći?',
                     'dobar dan':'Dobar dan! 😊 Kako vam mogu pomoći?',
                     'dobro veče':'Dobro veče! 🌙 Kako vam mogu pomoći?',
                     'zdravo':'Zdravo! 👋 Dobrodošli. Kako vam mogu pomoći?',
                     'ćao':'Ćao! 👋 Kako vam mogu pomoći?',
                     'hej':'Hej! 👋 Kako vam mogu pomoći?',
                     'pozdrav':'Pozdrav! 👋 Kako vam mogu pomoći?'}
        msg_low = message.lower()
        text = "Zdravo! 👋 Kako vam mogu pomoći?"
        for g, r in responses.items():
            if g in msg_low: text = r; break
        mem_key = f"{user_id}:{conv_id}" if conv_id else user_id
        self._add_msg(mem_key, 'user', message)
        self._add_msg(mem_key, 'assistant', text, intent='greeting')
        return {'response':text, 'intent':'greeting', 'conversation_id':conv_id,
                'escalation_needed':False, 'knowledge_sources':[],
                'channel_specific':self._channel(channel, text)}

    def _format_contact(self):
        phone = "+381603534000"
        wa_svg = '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 8px;"><path d="M19.077 4.928C17.191 3.041 14.683 2 12.006 2 6.798 2 2.548 6.193 2.54 11.393c-.003 1.747.456 3.457 1.328 4.984L2.25 21.75l5.428-1.573c1.472.839 3.137 1.286 4.857 1.288h.004c5.192 0 9.457-4.193 9.465-9.393.004-2.51-.972-4.872-2.857-6.758l-.07-.069zM12.03 20.026h-.003c-1.5-.001-2.97-.405-4.248-1.166l-.305-.182-3.222.934.86-3.144-.189-.312a7.925 7.925 0 0 1-1.222-4.222c.006-4.385 3.576-7.96 7.976-7.96 2.13 0 4.13.83 5.636 2.34 1.506 1.509 2.334 3.514 2.33 5.636-.005 4.386-3.576 7.961-7.973 7.961l-.04-.005z" fill="#25D366"/><path d="M16.11 13.454c-.266-.133-1.574-.774-1.818-.863-.244-.089-.422-.133-.599.133-.177.267-.688.863-.843 1.04-.155.178-.31.2-.577.067-.886-.333-1.682-.883-2.256-1.596-.178-.2-.322-.417-.454-.642.056-.033.11-.067.16-.106.088-.066.176-.133.26-.207.295-.257.534-.565.698-.911.027-.056.043-.118.048-.18.005-.063-.008-.127-.036-.185l-.424-.994c-.1-.233-.312-.39-.56-.413-.09-.008-.18-.003-.268.012-.15.021-.294.075-.418.156-.021.014-.041.029-.06.046-.359.316-.653.698-.863 1.127-.015.033-.026.067-.033.102-.094.378-.084.776.029 1.148.331 1.072.92 2.053 1.722 2.862.064.064.13.126.197.187.228.207.469.4.721.578.313.22.645.411.991.571.145.068.293.129.444.184.399.144.812.25 1.232.316.122.02.246.03.369.032.175.003.347-.021.512-.07.18-.048.341-.144.466-.277.192-.197.336-.436.422-.699.043-.133.055-.272.034-.408-.018-.12-.064-.234-.132-.334-.082-.117-.425-.716-.544-.878-.076-.1-.166-.132-.245-.132-.06 0-.12.016-.218.068-.275.146-.483.238-.609.289-.106.043-.187.066-.278-.022-.177-.177-.416-.407-.553-.549-.162-.17-.276-.381-.333-.61.09-.062.235-.15.358-.218.168-.093.31-.195.399-.282.181-.178.275-.409.293-.656.008-.092-.007-.184-.042-.27-.028-.07-.1-.222-.136-.298l-.232-.487c-.04-.084-.078-.168-.115-.253-.025-.058-.065-.11-.116-.148z" fill="#25D366"/></svg>'''
        vb_svg = '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 8px;"><path d="M11.995 2C7.58 2 4 5.58 4 9.995c0 1.732.58 3.415 1.584 4.812L4.5 19.5l4.774-1.125c1.34.82 2.881 1.267 4.53 1.267 4.415 0 8-3.58 8-7.995C21.804 5.58 18.41 2 13.995 2h-2z" fill="#7360F2"/><path d="M15.5 13.4c-.3.3-.7.4-1.1.2-.9-.3-2.3-1.1-3.3-2.2-.9-.9-1.6-1.9-1.9-2.8-.1-.4 0-.8.2-1.1l.5-.5c.3-.3.3-.8 0-1.1l-1.1-1.1c-.3-.3-.8-.3-1.1 0l-.5.5c-.6.6-.8 1.5-.5 2.3.5 1.3 1.5 2.8 2.9 4.2 1.4 1.4 2.9 2.3 4.2 2.9.8.3 1.7.1 2.3-.5l.5-.5c.3-.3.3-.8 0-1.1l-1.1-1.1c-.3-.3-.8-.3-1.1 0l-.5.5z" fill="#FFFFFF"/></svg>'''
        sms_icon = '<span style="font-size:24px; vertical-align:middle; margin-right:8px;">✉️</span>'
        return f"""
<div style="margin-bottom:20px;"><a href="https://wa.me/{phone}" target="_blank" style="color:#25D366;text-decoration:none;font-size:18px;display:flex;align-items:center;">{wa_svg}<span style="font-weight:bold;color:#25D366;">WhatsApp</span></a></div>
<div style="margin-bottom:20px;"><a href="viber://chat?number={phone}" target="_blank" style="color:#7360F2;text-decoration:none;font-size:18px;display:flex;align-items:center;">{vb_svg}<span style="font-weight:bold;color:#7360F2;">Viber</span></a></div>
<div style="margin-bottom:20px;"><a href="sms:{phone}" style="color:#34B7F1;text-decoration:none;font-size:18px;display:flex;align-items:center;">{sms_icon}<span style="font-weight:bold;color:#34B7F1;">SMS</span></a></div>
<br>📞 Telefon: {phone}<br>📧 Email: kontakt@zapmoto.rs<br>🕒 Radno vreme: svakoga dana sem nedelje od 10h do 19h"""

    def _offer_contact(self, message, user_id, conv_id, channel):
        msg = f"""<div style="font-family:Lato,sans-serif;"><p>Nažalost, nemam odgovor na ovo pitanje.</p>
        <p>Za dodatnu pomoć, možete nas kontaktirati putem:</p>{self._format_contact()}
        <p>Naš tim će vam rado pomoći u najkraćem mogućem roku.</p><p>Da li mogu da vam pomognem oko nečeg drugog?</p></div>"""
        self._add_msg(f"{user_id}:{conv_id}" if conv_id else user_id, 'user', message)
        self._add_msg(f"{user_id}:{conv_id}" if conv_id else user_id, 'assistant', msg, intent='contact_offered')
        return {'response':msg, 'intent':'contact_offered', 'conversation_id':conv_id,
                'escalation_needed':False, 'knowledge_sources':[], 'channel_specific':self._channel(channel, msg)}

    # ====== POMOĆNE ZA PROIZVODE ======
    def _filter_duplicates(self, items: List[Dict]) -> List[Dict]:
        groups = {}
        for item in items:
            q = item['content']['question']
            ans = item['content']['answer']
            cat = item['content'].get('category','')
            brand_match = re.search(r'Koje su karakteristike (.*?)\?', q)
            brand = brand_match.group(1) if brand_match else q
            key = (brand.lower(), cat)
            if key not in groups or len(ans) < len(groups[key]['content']['answer']):
                groups[key] = item
        return [v for v in groups.values() if len(v['content']['answer']) < 600]

    def _format_products(self, items: List[Dict], original_query=""):
        if not items: return ""
        items = self._filter_duplicates(items)
        parts = ['<div style="font-family:Lato,sans-serif;">']
        parts.append('<h3 style="color:#069806;">🛵 Pronađeni modeli:</h3>')
        for i, item in enumerate(items, 1):
            c = item['content']
            q = c.get('question','')
            a = c.get('answer','')
            name = q.replace("Koje su karakteristike ","").replace("?","").strip()
            link = re.search(r"https?://[^\s']+", a)
            url = link.group(0) if link else "#"
            cena = c.get('cena')
            parts.append(f'<div style="background:#f9fff9; border-left:4px solid #069806; padding:0.8rem; margin:0.8rem 0; border-radius:8px;">')
            parts.append(f'<strong style="font-size:1.1rem;">{i}. <a href="{url}" target="_blank" style="color:#069806;">{name}</a></strong><br>')
            domet = re.search(r'domet do (\d+) km', a)
            snaga = re.search(r'snagu ([\d\.]+) kw', a.lower()) or re.search(r'(\d+(?:\.\d+)?)\s*kw', a.lower())
            snaga = snaga.group(1) if snaga else "?"
            parts.append(f'🔋 Snaga: {snaga} kW | 🛣️ Domet: {domet.group(1) if domet else "?"} km')
            if cena:
                parts.append(f' | 💰 Cena: {cena} €')
            parts.append(f'<br><a href="{url}" target="_blank" style="color:#069806;">Detaljnije informacije</a>')
            parts.append('</div>')
        parts.append('</div>')
        return "".join(parts)

    def extract_criteria(self, message: str) -> Dict:
        prompt = f"""
        Iz korisničke poruke izdvoj kriterijume. Vrati JSON:
        {{"kategorija": "skuteri" ili "motocikli" ili null, "min_domet": broj ili null,
          "max_domet": null, "max_snaga": null, "kategorija_vozacke": "AM" ili "A1" ili null,
          "prenosna_baterija": null}}
        Ako nije pomenuto, vrati null. SAMO JSON."""
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system","content":prompt},
                          {"role":"user","content":message}],
                max_tokens=200, temperature=0.3)
            txt = resp.choices[0].message.content.strip()
            if txt.startswith('```json'): txt = txt.replace('```json','').replace('```','')
            elif txt.startswith('```'): txt = txt.replace('```','')
            crit = json.loads(txt)
        except:
            crit = {}
        for k in ['kategorija','min_domet','max_domet','max_snaga','kategorija_vozacke','prenosna_baterija']:
            if k not in crit: crit[k] = None
        return crit

    def filter_models(self, criteria: Dict, items: List[Dict]) -> List[Dict]:
        res = []
        for item in items:
            if item.get('source') != 'proizvodi': continue
            ans = item.get('answer','').lower()
            cat = item.get('category','')
            if criteria.get('kategorija') and cat != criteria['kategorija']: continue
            if criteria.get('kategorija_vozacke'):
                voz = criteria['kategorija_vozacke'].lower()
                if voz == 'am' and cat != 'skuteri': continue   # AM dozvola -> samo skuteri
                # A1 dozvola dozvoljava i skutere i motocikle – ne filtriramo
            if criteria.get('min_domet') or criteria.get('max_domet'):
                d = re.search(r'domet do (\d+) km', ans)
                if d:
                    d_int = int(d.group(1))
                    # Zahteva se STROGO VEĆI domet od traženog
                    if criteria.get('min_domet') and d_int <= criteria['min_domet']: continue
                    if criteria.get('max_domet') and d_int > criteria['max_domet']*1.2: continue
                else: continue
            res.append(item)
        return res

    # ====== KALKULATOR ======
    def _parse_daily_km(self, msg: str) -> float:
        nums = re.findall(r'[\d\.,]+', msg)
        for n in nums:
            try:
                val = float(n.replace(',','.'))
                if val > 0: return val
            except: continue
        raise ValueError

    def _calc_savings(self, km_year):
        wh = self.DEFAULT_MODEL_WH
        kwh = wh/10
        petrol = (km_year * 4.0/100) * 193
        struja = (km_year * kwh/100) * 34.61
        maint_gas = km_year * self.GAS_MAINT_RATE_RSD
        maint_elec = km_year * self.ELEC_MAINT_RATE_RSD
        total_gas = petrol + maint_gas
        total_elec = struja + maint_elec
        save = total_gas - total_elec
        return f"""<div style="font-family:Lato,sans-serif; background:#fff; border-radius:12px; padding:1rem; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
<h3 style="color:#069806;">⚡ Kalkulator uštede – {self.DEFAULT_MODEL_NAME} ({wh} Wh/km)</h3>
<p>📅 Godišnja kilometraža: <strong>{km_year:,.0f} km</strong></p>
<ul>
<li>⛽ Benzinski: gorivo {petrol:,.0f} RSD + održavanje {maint_gas:,.0f} RSD = <strong>{total_gas:,.0f} RSD</strong></li>
<li>⚡ {self.DEFAULT_MODEL_NAME}: struja {struja:,.0f} RSD + održavanje {maint_elec:,.0f} RSD = <strong>{total_elec:,.0f} RSD</strong></li>
</ul>
<div style="background:#e6ffe6; padding:0.8rem; border-radius:8px; text-align:center; margin:1rem 0;">
<span style="font-size:1.3rem; font-weight:bold;">💡 UKUPNA GODIŠNJA UŠTEDA: <span style="color:#069806;">{save:,.0f} RSD</span></span></div>
<p style="font-size:0.9rem;">⚠️ *Ušteda za druge modele se razlikuje.*<br>🔗 <a href="https://zapmoto.rs/kalkulator-ustede-elektricnim-skuterom/" target="_blank" style="color:#069806; text-decoration:underline;">Detaljan kalkulator za sve modele</a></p>
</div>"""

    # ====== GLAVNA generate_response ======
    def generate_response(self, message: str, user_id: str, conversation_id: str = None, channel: str = "web") -> Dict:
        try:
            mem_key = f"{user_id}:{conversation_id}" if conversation_id else user_id
            conv = self.get_or_create_conversation(mem_key, user_id, channel)
            intent = self.detect_intent(message, conv.messages)
            msg_low = message.lower().strip()

            # ====== SPEČJALNI SLUČAJEVI ======
            # 1. Pitanje o kalkulatoru (bez "uštede")
            if 'kalkulator' in msg_low and not any(w in msg_low for w in ['ušted','uštedeti','izračunaj','proračun']):
                resp = ('Da, imamo <a href="https://zapmoto.rs/kalkulator-ustede-elektricnim-skuterom/" target="_blank" '
                        'style="color:#069806;">kalkulator uštede</a>. Želite li da izračunamo uštedu? Unesite dnevnu kilometražu.')
                conv.context['awaiting_savings_km'] = True   # <-- pamti da očekuje broj
                self._add_msg(mem_key, 'user', message)
                self._add_msg(mem_key, 'assistant', resp, intent='product_question')
                return {'response':resp, 'intent':'product_question', 'conversation_id':conversation_id,
                        'escalation_needed':False, 'knowledge_sources':[], 'channel_specific':self._channel(channel, resp)}

            # 2. Upit o ceni (sadrži "cena" ili "košta")
            if any(w in msg_low for w in ['cena','košta','cenu','cene']):
                # Pronađi brend (sa deklinacijama)
                brand_map = {
                    'pusa': ['pusa','puse','pusu','pusom'],
                    'lipo': ['lipo','lipa','lipu','lipom'],
                    'deer': ['deer','deera','deeru','deerom'],
                    'e2go': ['e2go','e2goa','e2gu'],
                    'puma': ['puma','pume','pumi','pumom'],
                    'tiger': ['tiger','tigra','tigru','tigrom'],
                    'lion': ['lion','liona','lionu','lionom']
                }
                found_brand = None
                for base, variants in brand_map.items():
                    if any(v in msg_low for v in variants):
                        found_brand = base
                        break
                if found_brand:
                    # Traži snagu
                    snaga_match = re.search(r'(\d+(?:[.,]\d+)?)\s*kw', msg_low)
                    snaga = snaga_match.group(1).replace(',','.') if snaga_match else None
                    # Pretraži bazu
                    prods = [item for item in self.knowledge_base if item.get('source')=='proizvodi' and found_brand in item.get('question','').lower()]
                    if snaga:
                        prods = [p for p in prods if re.search(fr'snagu {snaga} kw', p.get('answer','').lower()) or re.search(fr'{snaga}\s*kw', p.get('answer','').lower())]
                    if prods:
                        # Uzmi prvi (najrelevantniji)
                        p = prods[0]
                        cena = p.get('cena')
                        name = p['question'].replace("Koje su karakteristike ","").replace("?","")
                        if cena:
                            resp = f"Cena modela {name} je {cena} €."
                        else:
                            resp = f"Cenu modela {name} možete pogledati na sajtu."
                        link = re.search(r"https?://[^\s']+", p.get('answer',''))
                        if link:
                            resp += f' <a href="{link.group(0)}" target="_blank" style="color:#069806;">Detalji</a>'
                        self._add_msg(mem_key, 'user', message)
                        self._add_msg(mem_key, 'assistant', resp, intent='product_question', knowledge=['proizvodi'])
                        return {'response':resp, 'intent':'product_question', 'conversation_id':conversation_id,
                                'escalation_needed':False, 'knowledge_sources':['proizvodi'], 'channel_specific':self._channel(channel, resp)}
                # Ako brend nije pronađen, nastaviće se normalna obrada

            # ====== SAVINGS CALCULATOR ======
            if intent == Intent.SAVINGS_CALCULATOR or conv.context.get('awaiting_savings_km'):
                if not conv.context.get('awaiting_savings_km'):
                    conv.context['awaiting_savings_km'] = True
                    resp = "Da bih izračunao uštedu, unesite koliko kilometara dnevno prelazite (npr. 80)."
                    self._add_msg(mem_key, 'user', message)
                    self._add_msg(mem_key, 'assistant', resp, intent=intent.value)
                    return {'response':resp, 'intent':intent.value, 'conversation_id':conversation_id,
                            'escalation_needed':False, 'knowledge_sources':[], 'channel_specific':self._channel(channel, resp)}
                try:
                    daily = self._parse_daily_km(message)
                except:
                    resp = "Molim vas unesite broj kilometara (npr. 80)."
                    self._add_msg(mem_key, 'user', message)
                    self._add_msg(mem_key, 'assistant', resp, intent=intent.value)
                    return {'response':resp, 'intent':intent.value, 'conversation_id':conversation_id,
                            'escalation_needed':False, 'knowledge_sources':[], 'channel_specific':self._channel(channel, resp)}
                yearly = daily * self.DAYS_PER_YEAR
                resp = self._calc_savings(yearly)
                conv.context['awaiting_savings_km'] = False
                self._add_msg(mem_key, 'user', message)
                self._add_msg(mem_key, 'assistant', resp, intent=intent.value, knowledge=['kalkulator'])
                return {'response':resp, 'intent':intent.value, 'conversation_id':conversation_id,
                        'escalation_needed':False, 'knowledge_sources':['kalkulator'], 'channel_specific':self._channel(channel, resp)}

            # ====== ADVICE / RECOMMENDATION ======
            if intent == Intent.PRODUCT_RECOMMENDATION or intent == Intent.ADVICE_NEEDED:
                crit = self.extract_criteria(message)
                # Ako nedostaje domet ili dozvola, uvek pitaj
                if not crit['min_domet'] or not crit['kategorija_vozacke']:
                    questions = []
                    if not crit['min_domet']:
                        questions.append("Koliko kilometara dnevno prelazite?")
                    if not crit['kategorija_vozacke']:
                        questions.append("Da li imate AM ili A1 vozačku dozvolu?")
                    resp = "Da bih vam dao najbolju preporuku, potrebno mi je nekoliko informacija.<br>" + "<br>".join(f"• {q}" for q in questions)
                    self._add_msg(mem_key, 'user', message)
                    self._add_msg(mem_key, 'assistant', resp, intent='product_recommendation')
                    return {'response':resp, 'intent':'product_recommendation', 'conversation_id':conversation_id,
                            'escalation_needed':False, 'knowledge_sources':[], 'channel_specific':self._channel(channel, resp)}
                # Sačuvaj kriterijume
                conv.context['criteria'] = crit
                all_prods = [item for item in self.knowledge_base if item.get('source')=='proizvodi']
                matching = self.filter_models(crit, all_prods)
                if not matching:
                    resp = "Nažalost, nijedan model ne odgovara zadatim kriterijumima. Pokušajte sa manjim dometom ili drugom kategorijom."
                    self._add_msg(mem_key, 'user', message)
                    self._add_msg(mem_key, 'assistant', resp, intent=intent.value)
                    return {'response':resp, 'intent':intent.value, 'conversation_id':conversation_id,
                            'escalation_needed':False, 'knowledge_sources':[], 'channel_specific':self._channel(channel, resp)}
                def get_dom(m): 
                    d = re.search(r'domet do (\d+) km', m.get('answer',''))
                    return int(d.group(1)) if d else 0
                matching.sort(key=get_dom, reverse=True)
                top = matching[:5]
                parts = ['<div style="font-family:Lato,sans-serif;"><h3 style="color:#069806;">⚡ Preporučeni modeli:</h3>']
                for i, m in enumerate(top, 1):
                    q = m.get('question','')
                    name = q.replace("Koje su karakteristike ","").replace("?","").strip()
                    a = m.get('answer','')
                    domet = re.search(r'domet do (\d+) km', a).group(1) if re.search(r'domet do (\d+) km', a) else "?"
                    snaga = re.search(r'snagu ([\d\.]+) kw', a.lower()) or re.search(r'(\d+(?:\.\d+)?)\s*kw', a.lower())
                    snaga = snaga.group(1) if snaga else "?"
                    link = re.search(r"https?://[^\s']+", a)
                    url = link.group(0) if link else "#"
                    cena = m.get('cena')
                    parts.append(f'<div style="background:#f0faf0; border-radius:8px; padding:0.7rem; margin:0.5rem 0;">')
                    parts.append(f'<strong>{i}. <a href="{url}" target="_blank" style="color:#069806;">{name}</a></strong><br>')
                    parts.append(f'🔋 {snaga} kW | 🛣️ {domet} km')
                    if cena: parts.append(f' | 💰 {cena} €')
                    parts.append('</div>')
                parts.append('</div>')
                resp = "".join(parts)
                self._add_msg(mem_key, 'user', message)
                self._add_msg(mem_key, 'assistant', resp, intent=intent.value, knowledge=['proizvodi'])
                return {'response':resp, 'intent':intent.value, 'conversation_id':conversation_id,
                        'escalation_needed':False, 'knowledge_sources':['proizvodi'], 'channel_specific':self._channel(channel, resp)}

            # ====== BRAND ONLY (poboljšano prepoznavanje) ======
            brands = ['pusa','lipo','deer','e2go','puma','tiger','lion']
            clean_msg = re.sub(r'[^\w\s]', '', msg_low).strip()
            words = clean_msg.split()
            for brand in brands:
                if brand in words and len(words) <= 2:
                    crit = conv.context.get('criteria')
                    prods = [item for item in self.knowledge_base if item.get('source')=='proizvodi' and brand in item.get('question','').lower()]
                    if crit:
                        prods = self.filter_models(crit, prods)
                    if prods:
                        items = [{'content':p, 'relevance_score':1.0, 'source':'proizvodi'} for p in prods]
                        resp = self._format_products(items)
                    else:
                        resp = f"Nemamo {brand.title()} modele koji odgovaraju vašim kriterijumima."
                    self._add_msg(mem_key, 'user', message)
                    self._add_msg(mem_key, 'assistant', resp, intent=Intent.PRODUCT_QUESTION.value, knowledge=['proizvodi'])
                    return {'response':resp, 'intent':Intent.PRODUCT_QUESTION.value, 'conversation_id':conversation_id,
                            'escalation_needed':False, 'knowledge_sources':['proizvodi'], 'channel_specific':self._channel(channel, resp)}

            # ====== GREETING ======
            if intent == Intent.GREETING:
                return self.generate_greeting_response(message, user_id, conversation_id, channel)

            # ====== ESCALATION ======
            if self.should_escalate(message, intent, conv):
                conv.escalation_needed = True
                return self.prepare_escalation(message, conv)

            # ====== PRODUCT QUESTION ======
            logger.info("Standardna obrada pitanja...")
            relevant = self.retrieve_relevant_knowledge(message)
            good = False
            best = 0
            if relevant:
                best = relevant[0].get('relevance_score',0)
                if best > 0.29: good = True
            if not good:
                return self._offer_contact(message, user_id, conversation_id, channel)

            top = relevant[0]['content']
            if top.get('source') in ['vodic','blog']:
                ctx = {'relevant_knowledge':relevant, 'conversation_history':conv.messages[-6:],
                       'intent':intent.value, 'channel':channel}
                resp = self._llm_response(message, ctx)
            elif top.get('source')=='kontakt' or top.get('category')=='kontakt':
                resp = f"<div style='font-family:Lato,sans-serif;'>Naš tim vam stoji na raspolaganju:<br><br>{self._format_contact()}</div>"
            else:
                resp = self._format_products(relevant, message)

            self._add_msg(mem_key, 'user', message)
            self._add_msg(mem_key, 'assistant', resp, intent=intent.value, knowledge=[k['source'] for k in relevant])
            return {'response':resp, 'intent':intent.value, 'conversation_id':conversation_id,
                    'escalation_needed':False, 'knowledge_sources':[k['source'] for k in relevant],
                    'channel_specific':self._channel(channel, resp)}
        except Exception as e:
            logger.error(f"Greška u generate_response: {e}\n{traceback.format_exc()}")
            return {'response':"Došlo je do tehničke greške. Molim vas pokušajte ponovo ili kontaktirajte podršku.",
                    'intent':'unknown', 'conversation_id':conversation_id, 'escalation_needed':True,
                    'knowledge_sources':[]}

    # ---------- LLM ----------
    def _llm_response(self, message, context):
        know = ""
        if context['relevant_knowledge']:
            know = "Relevantne informacije:\n"
            for i, item in enumerate(context['relevant_knowledge']):
                c = item['content']
                know += f"{i+1}. Pitanje: {c.get('question','')}\n   Odgovor: {c.get('answer','')}\n\n"
        hist = ""
        if context['conversation_history']:
            for m in context['conversation_history']:
                hist += f"{'Korisnik' if m['role']=='user' else 'Asistent'}: {m['content']}\n"
        system = f"""Ti si profesionalni asistent za e-trgovinu.
        Odgovaraj koncizno i samo na osnovu datih informacija.
        Ako nema odgovora, reci da ćeš proslediti agentu.
        Namera: {context['intent']}, Kanal: {context['channel']}"""
        prompt = f"{hist}\n{know}\nKorisnik pita: {message}\nOdgovori korisniku."
        try:
            r = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system","content":system}, {"role":"user","content":prompt}],
                max_tokens=800, temperature=0.7)
            text = r.choices[0].message.content.strip()
            text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" style="color:#069806;">\1</a>', text)
            return f'<div style="font-family:Lato,sans-serif;">{text}</div>'
        except Exception as e:
            logger.error(f"LLM greška: {e}")
            return "Izvinite, došlo je do tehničke greške."

    # ---------- memorija i kanali ----------
    def should_escalate(self, msg, intent, conv):
        if any(w in msg.lower() for w in ['agent','operater','čovek','govori sa','uživo','live chat']):
            return True
        if intent == Intent.CONTACT_SUPPORT: return True
        recent = conv.messages[-6:]
        if sum(1 for m in recent if m.get('intent')=='unknown') >= 3: return True
        return False

    def prepare_escalation(self, msg, conv):
        transcript = [{'role':m['role'],'content':m['content'],'timestamp':m.get('timestamp','')} for m in conv.messages]
        transcript.append({'role':'user','content':msg,'timestamp':datetime.now().isoformat()})
        return {'response':"Povezujem vas sa našim agentom. Molim sačekajte.",
                'escalation_needed':True, 'escalation_data':{'transcript':transcript,'conversation_id':conv.user_id},
                'channel_specific':self._channel(conv.context.get('channel','web'), None, escalation=True)}

    def get_or_create_conversation(self, key, uid, channel):
        if key in self.active_conversations:
            return self.active_conversations[key]
        conv = ConversationMemory(user_id=uid, messages=[], last_updated=datetime.now(),
                                  context={'channel':channel, 'start_time':datetime.now().isoformat()})
        self.active_conversations[key] = conv
        return conv

    def _add_msg(self, key, role, content, intent=None, knowledge=None):
        if key not in self.active_conversations: return
        msg = {'role':role, 'content':content, 'timestamp':datetime.now().isoformat()}
        if intent: msg['intent'] = intent
        if knowledge: msg['knowledge_used'] = knowledge
        self.active_conversations[key].messages.append(msg)
        if intent: self.active_conversations[key].context['last_intent'] = intent

    def _channel(self, channel, text=None, escalation=False):
        cfg = {'web':{'type':'text','options':{'quick_replies':True,'rich_text':True,'buttons':True}},
               'whatsapp':{'type':'text','options':{'max_length':4096,'interactive':True,'buttons':True}},
               'viber':{'type':'text','options':{'rich_media':True,'keyboard':True}},
               'telegram':{'type':'text','options':{'markdown':True,'inline_keyboards':True}}}
        c = cfg.get(channel, cfg['web'])
        r = {'channel':channel, 'type':c['type'], 'config':c['options']}
        if text: r['text'] = text
        if escalation: r['escalation'] = {'message':"Povezivanje sa agentom...",'estimated_wait':"2-3 minuta"}
        return r