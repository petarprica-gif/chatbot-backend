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
    PRODUCT_RECOMMENDATION = "product_recommendation"
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
        # Prvo proveri da li je u pitanju nastavak preporuke (kontekst)
        if conversation_history and len(conversation_history) > 0:
            last_assistant_msg = None
            for msg in reversed(conversation_history):
                if msg['role'] == 'assistant':
                    last_assistant_msg = msg
                    break
            
            # Ako je poslednja poruka bila preporuka, a korisnik odbija
            if last_assistant_msg and last_assistant_msg.get('intent') == 'product_recommendation':
                rejection_keywords = ['ne sviđa', 'drugi', 'neki drugi', 'drugačiji', 'neću', 'ne želim', 'nemoj']
                if any(keyword in message.lower() for keyword in rejection_keywords):
                    logger.info("Prepoznato odbijanje preporuke, ostajem u PRODUCT_RECOMMENDATION modu")
                    return Intent.PRODUCT_RECOMMENDATION
        
        # NOVO: Provera za osnovne pozdrave (pre OpenAI poziva)
        greetings = ['dobro jutro', 'dobro veče', 'dobar dan', 'zdravo', 'ćao', 'hej', 'cao', 'pozdrav']
        if any(greeting in message.lower() for greeting in greetings):
            logger.info("Prepoznat pozdrav pre OpenAI-ja")
            return Intent.GREETING
        
        # Pravimo prompt za OpenAI sa kontekstom
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
    
    def extract_criteria_from_message(self, message: str) -> Dict[str, any]:
        """
        Koristi OpenAI da iz poruke izdvoji kriterijume za preporuku.
        
        Args:
            message: Korisnička poruka
        
        Returns:
            Rečnik sa kriterijumima
        """
        logger.info(f"===== extract_criteria_from_message POZVAN =====")
        logger.info(f"Poruka: {message}")
        
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
            logger.info("Šaljem zahtev ka OpenAI...")
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
            logger.info(f"Odgovor od OpenAI-ja: {criteria_text}")
            
            # Očisti JSON ako ima markdowna
            if criteria_text.startswith('```json'):
                criteria_text = criteria_text.replace('```json', '').replace('```', '')
            elif criteria_text.startswith('```'):
                criteria_text = criteria_text.replace('```', '')
            
            criteria_text = criteria_text.strip()
            logger.info(f"Očišćen JSON string: {criteria_text}")
            
            # Pokušaj da parsiraš JSON
            try:
                criteria = json.loads(criteria_text)
                logger.info(f"Parsirani kriterijumi: {criteria}")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Neispravan JSON od OpenAI-ja: {criteria_text}")
                logger.error(f"Greška pri parsiranju: {str(e)}")
                # Vrati prazan rečnik kao fallback
                criteria = {}
            
            # Osiguraj da svi očekivani ključevi postoje
            expected_keys = ['kategorija', 'min_domet', 'max_domet', 'max_snaga', 'kategorija_vozacke', 'prenosna_baterija']
            for key in expected_keys:
                if key not in criteria:
                    criteria[key] = None
                    logger.info(f"Dodajem nedostajući ključ: {key} = None")
            
            logger.info(f"Konačni kriterijumi: {criteria}")
            return criteria
            
        except Exception as e:
            logger.error(f"❌ Greška u extract_criteria_from_message: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def filter_models_by_criteria(self, criteria: Dict[str, any], message: str = "") -> List[Dict]:
        """
        Filtrira modele iz baze znanja na osnovu zadatih kriterijuma.
        
        Args:
            criteria: Rečnik sa kriterijumima
            message: Originalna poruka (za dodatne provere)
        
        Returns:
            Lista modela koji ispunjavaju kriterijume
        """
        logger.info(f"===== filter_models_by_criteria POZVAN =====")
        logger.info(f"Primljeni kriterijumi: {criteria}")
        logger.info(f"Poruka: {message}")
        logger.info(f"Ukupno modela u bazi: {len(self.knowledge_base)}")
        
        filtered_models = []
        
        for idx, item in enumerate(self.knowledge_base):
            model_name = item.get('question', 'nepoznato')
            logger.info(f"--- Proveravam model: {model_name} ---")
            
            # Proveri da li je unos proizvod (ima polje 'source' == 'proizvodi')
            if item.get('source') != 'proizvodi':
                logger.info(f"❌ Nije proizvod (source={item.get('source')}), preskačem")
                continue
            
            # Provera kategorije (skuteri/motocikli) ako je zadata
            if 'kategorija' in criteria and criteria['kategorija']:
                logger.info(f"Proveravam kategoriju: očekuje se {criteria['kategorija']}, imamo {item.get('category')}")
                if item.get('category') != criteria['kategorija']:
                    logger.info(f"❌ Kategorija se ne poklapa, preskačem")
                    continue
                else:
                    logger.info(f"✅ Kategorija se poklapa")
            
            # Provera kriterijuma iz odgovora
            answer = item.get('answer', '').lower()
            
            # POSEBNO ZA AM KATEGORIJU I GRADSKU VOŽNJU
            if 'kategorija_vozacke' in criteria and criteria['kategorija_vozacke'] == 'AM':
                if 'grad' in message.lower() or 'gradsku' in message.lower() or 'gradska' in message.lower():
                    logger.info(f"Proveravam za gradsku vožnju sa AM")
                    if item.get('category') == 'skuteri':
                        logger.info(f"✅ Skuter prihvaćen za gradsku vožnju (bez dodatnih provera)")
                        filtered_models.append(item)
                        continue
                    else:
                        logger.info(f"❌ Nije skuter, preskačem")
                        continue
            
            # Provera kategorije vozačke dozvole
            if 'kategorija_vozacke' in criteria and criteria['kategorija_vozacke']:
                vozacka = criteria['kategorija_vozacke'].lower()
                logger.info(f"Proveravam kategoriju vozačke: {vozacka}")
                
                if vozacka == 'am':
                    # AM kategorija - svi skuteri su po zakonu AM
                    if item.get('category') != 'skuteri':
                        logger.info(f"❌ Nije skuter, preskačem")
                        continue
                    logger.info(f"✅ Skuter prihvaćen za AM kategoriju")
                elif vozacka == 'a1':
                    if 'a1 kategorija' not in answer:
                        logger.info(f"❌ Nema 'a1 kategorija' u opisu, preskačem")
                        # Dodatna provera - možda je ipak A1
                        if 'a1' in answer:
                            logger.info(f"⚠️ Ali sadrži 'a1' u tekstu, možda bi trebalo prihvatiti?")
                        continue
                    else:
                        logger.info(f"✅ Sadrži 'a1 kategorija' u opisu")
            
            # Provera dometa - DETALJNO LOGOVANJE
            domet_match = re.search(r'domet do (\d+) km', answer)
            if domet_match:
                domet = int(domet_match.group(1))
                logger.info(f"Pronađen domet: {domet} km")
                
                domet_passed = True
                
                # Ako je zadan min_domet, proveri sa fleksibilnošću
                if 'min_domet' in criteria and criteria['min_domet']:
                    min_allowed = criteria['min_domet'] * 0.8
                    logger.info(f"Proveravam min_domet: {domet} >= {min_allowed:.0f} (20% ispod {criteria['min_domet']})")
                    if domet < min_allowed:
                        logger.info(f"❌ Domet {domet} < {min_allowed:.0f}, preskačem")
                        domet_passed = False
                    else:
                        logger.info(f"✅ Domet {domet} >= {min_allowed:.0f}")
                
                # Ako je zadan max_domet, proveri sa fleksibilnošću
                if domet_passed and 'max_domet' in criteria and criteria['max_domet']:
                    max_allowed = criteria['max_domet'] * 1.2
                    logger.info(f"Proveravam max_domet: {domet} <= {max_allowed:.0f} (20% iznad {criteria['max_domet']})")
                    if domet > max_allowed:
                        logger.info(f"❌ Domet {domet} > {max_allowed:.0f}, preskačem")
                        domet_passed = False
                    else:
                        logger.info(f"✅ Domet {domet} <= {max_allowed:.0f}")
                
                if not domet_passed:
                    continue
            else:
                logger.info(f"⚠️ Nije pronađen domet u opisu")
                # Ako nema podatka o dometu, a kriterijum je zadat, možda ipak treba uključiti?
                if 'min_domet' in criteria or 'max_domet' in criteria:
                    logger.info(f"❌ Kriterijum dometa postoji ali nema podatka, preskačem")
                    continue
            
            # Ako je prošao sve filtere, dodaj u listu
            filtered_models.append(item)
            logger.info(f"✅✅✅ MODEL PRIHVAĆEN: {model_name}")
        
        logger.info(f"===== FILTER ZAVRŠEN: PRONAĐENO {len(filtered_models)} MODEL =====")
        logger.info(f"Prihvaćeni modeli: {[m.get('question', 'nepoznato') for m in filtered_models]}")
        return filtered_models
    
    def offer_contact_options(self, message: str, user_id: str, conversation_id: str = None, channel: str = "web") -> Dict[str, Any]:
        """
        Nudi korisniku opcije za kontakt kada chatbot ne može da odgovori.
        """
        # Broj telefona
        phone = "+381603534000"
        
        # Linkovi za direktan chat
        whatsapp_link = f"https://wa.me/{phone}"
        viber_link = f"viber://chat?number={phone}"
        sms_link = f"sms:{phone}"
        
        # Vertikalno poređane opcije sa ikonama i nazivima (sve klikabilno)
        contact_message = f"""
Nažalost, nemam odgovor na ovo pitanje.

Za dodatnu pomoć, možete nas kontaktirati putem:

<br><br>
<div style="margin-bottom: 20px;">
    <a href="{whatsapp_link}" target="_blank" style="color: #25D366; text-decoration: none; font-size: 18px; display: inline-block;">
        <span style="font-size: 24px; vertical-align: middle;">📱</span>
        <span style="vertical-align: middle; font-weight: bold; color: #25D366; margin-left: 8px;">WhatsApp</span>
    </a>
</div>

<div style="margin-bottom: 20px;">
    <a href="{viber_link}" target="_blank" style="color: #7360F2; text-decoration: none; font-size: 18px; display: inline-block;">
        <span style="font-size: 24px; vertical-align: middle;">📱</span>
        <span style="vertical-align: middle; font-weight: bold; color: #7360F2; margin-left: 8px;">Viber</span>
    </a>
</div>

<div style="margin-bottom: 20px;">
    <a href="{sms_link}" style="color: #34B7F1; text-decoration: none; font-size: 18px; display: inline-block;">
        <span style="font-size: 24px; vertical-align: middle;">✉️</span>
        <span style="vertical-align: middle; font-weight: bold; color: #34B7F1; margin-left: 8px;">SMS</span>
    </a>
</div>
<br>
Naš tim će vam rado pomoći u najkraćem mogućem roku.

Da li mogu da vam pomognem oko nečeg drugog?
"""
        
        # Sačuvaj poruke u memoriju
        memory_key = f"{user_id}:{conversation_id}" if conversation_id else user_id
        self.add_to_conversation(memory_key, {
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        self.add_to_conversation(memory_key, {
            'role': 'assistant',
            'content': contact_message,
            'timestamp': datetime.now().isoformat(),
            'intent': 'contact_offered',
            'knowledge_used': []
        })
        
        return {
            'response': contact_message,
            'intent': 'contact_offered',
            'conversation_id': conversation_id,
            'escalation_needed': False,
            'knowledge_sources': [],
            'channel_specific': self.get_channel_specific_response(channel, contact_message)
        }
    
    def generate_recommendation_response(self, message: str, criteria: Dict[str, any], conversation_history: List[Dict]) -> tuple:
        """
        Generiše odgovor sa preporukama na osnovu kriterijuma.
        
        Args:
            message: Originalna poruka korisnika
            criteria: Izdvojeni kriterijumi
            conversation_history: Istorija razgovora za kontekst
        
        Returns:
            Tuple (tekst odgovora, lista preporučenih ID-jeva)
        """
        logger.info(f"===== generate_recommendation_response POZVAN =====")
        logger.info(f"Kriterijumi: {criteria}")
        logger.info(f"Dužina istorije: {len(conversation_history)}")
        logger.info(f"Poruka: {message}")
        
        # Prošireno pamćenje prethodnih preporuka - uzima iz poslednjih 10 poruka
        previous_models = []
        if conversation_history:
            # Uzmi sve prethodne preporuke iz poslednjih 10 poruka
            for msg in reversed(conversation_history[-10:]):
                if msg.get('intent') == 'product_recommendation' and 'recommended_models' in msg:
                    previous_models.extend(msg.get('recommended_models', []))
                    logger.info(f"Dodajem preporuke iz poruke: {msg.get('recommended_models', [])}")
            # Ukloni duplikate
            previous_models = list(set(previous_models))
            logger.info(f"Sve prethodne preporuke: {previous_models}")
        
        logger.info(f"Prethodno preporučeni modeli: {previous_models}")
        
        # Filtriraj modele (prosledi i originalnu poruku)
        matching_models = self.filter_models_by_criteria(criteria, message)
        
        # Ako imamo prethodne modele, izbaci ih iz preporuke
        if previous_models and matching_models:
            before_count = len(matching_models)
            matching_models = [m for m in matching_models if m.get('id') not in previous_models]
            logger.info(f"Nakon izbacivanja prethodnih: {before_count} -> {len(matching_models)} modela")
            if len(matching_models) == 0:
                logger.info("⚠️ Nema novih modela nakon izbacivanja prethodnih")
        
        # Posebna obrada za AM kategoriju - ako nema rezultata, ponudi sve skutere
        if not matching_models and criteria.get('kategorija_vozacke') == 'AM':
            logger.info("Nema modela za AM kriterijume, tražim sve skutere...")
            all_scooters = [item for item in self.knowledge_base 
                          if item.get('source') == 'proizvodi' 
                          and item.get('category') == 'skuteri']
            logger.info(f"Pronađeno ukupno {len(all_scooters)} skutera u bazi")
            if all_scooters:
                # Izbaci prethodno preporučene
                if previous_models:
                    all_scooters = [s for s in all_scooters if s.get('id') not in previous_models]
                    logger.info(f"Nakon izbacivanja prethodnih: {len(all_scooters)} skutera")
                matching_models = all_scooters[:5]
                logger.info(f"Uzimam prvih {len(matching_models)} skutera kao zamenu")
        
        if not matching_models:
            logger.info("❌ Nema modela koji zadovoljavaju kriterijume")
            if previous_models:
                return "Razumem. Nažalost, trenutno nemamo druge modele koji odgovaraju tvojim kriterijumima. Preporučujem ti da pogledaš našu kompletnu ponudu na sajtu: https://zapmoto.rs/product-category/elektricni-skuteri-i-motori/ ili da mi kažeš koji su ti drugi kriterijumi važni (npr. niža cena, manji domet, prenosiva baterija...).", []
            else:
                return "Nažalost, trenutno nemamo modele koji u potpunosti odgovaraju tvojim kriterijumima. Preporučujem ti da pogledaš našu ponudu na sajtu: https://zapmoto.rs/product-category/elektricni-skuteri-i-motori/ ili da nas kontaktiraš za dodatnu pomoć.", []
        
        # Sortiranje modela po dometu (od najvećeg ka najmanjem)
        def get_domet(model):
            answer = model.get('answer', '')
            domet_match = re.search(r'domet do (\d+) km', answer)
            if domet_match:
                return int(domet_match.group(1))
            return 0
        
        matching_models.sort(key=get_domet, reverse=True)
        logger.info(f"Modeli sortirani po dometu: {[get_domet(m) for m in matching_models[:5]]}")
        
        # Pripremi listu modela za prikaz (maksimalno 5)
        models_text = ""
        recommended_ids = []
        for i, model in enumerate(matching_models[:5], 1):
            model_id = model.get('id')
            if model_id:
                recommended_ids.append(model_id)
                logger.info(f"Dodajem model ID {model_id} u preporuke (domet: {get_domet(model)} km)")
            
            question = model.get('question', '')
            model_name = question.replace("Koje su karakteristike ", "").replace("?", "").strip()
            answer = model.get('answer', '')
            
            domet_match = re.search(r'domet do (\d+) km', answer)
            domet = domet_match.group(1) if domet_match else "nepoznat"
            
            snaga_match = re.search(r'snagu ([\d\.]+) kw', answer.lower())
            if not snaga_match:
                snaga_match = re.search(r'(\d+(?:\.\d+)?)\s*kw', answer.lower())
            snaga = snaga_match.group(1) if snaga_match else "?"
            
            vozacka = "AM" if "am kategorija" in answer.lower() else "A1" if "a1 kategorija" in answer.lower() else "?"
            
            link_match = re.search(r"https?://[^\s]+", answer)
            link = link_match.group(0) if link_match else "#"
            
            models_text += f"\n\n{i}. **{model_name}**\n   - Snaga: {snaga} kW\n   - Domet: {domet} km\n   - Kategorija: {vozacka}\n   - Detaljnije: {link}"
        
        response_text = f"Na osnovu tvojih kriterijuma, preporučujem sledeće modele:{models_text}\n\nAko želiš više informacija o nekom modelu, slobodno pitaj! Ako ti se neki od ovih modela ne sviđa, reci mi pa ću probati da nađem drugačije opcije."
        
        logger.info(f"Generisano {len(recommended_ids)} preporuka: {recommended_ids}")
        return response_text, recommended_ids
    
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
            logger.info(f"===== GENERATE_RESPONSE POZVAN =====")
            logger.info(f"Poruka: {message}")
            logger.info(f"User ID: {user_id}")
            logger.info(f"Conversation ID: {conversation_id}")
            logger.info(f"Channel: {channel}")
            
            # Dohvati ili kreiraj memoriju konverzacije
            memory_key = f"{user_id}:{conversation_id}" if conversation_id else user_id
            conversation = self.get_or_create_conversation(memory_key, user_id, channel)
            
            # Detektuj nameru
            intent = self.detect_intent(message, conversation.messages)
            logger.info(f"Detektovana namera: {intent}")
            
            # Proveri da li je potrebna eskalacija
            if self.should_escalate(message, intent, conversation):
                logger.info("Potrebna eskalacija ka agentu")
                conversation.escalation_needed = True
                return self.prepare_escalation(message, conversation)
            
            # Obrada preporuka
            if intent == Intent.PRODUCT_RECOMMENDATION:
                logger.info("Obrada preporuka...")
                criteria = self.extract_criteria_from_message(message)
                response_text, recommended_ids = self.generate_recommendation_response(message, criteria, conversation.messages)
                
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
                    'recommended_models': recommended_ids,
                    'knowledge_used': []
                })
                
                logger.info("Preporuke uspešno generisane")
                return {
                    'response': response_text,
                    'intent': intent.value,
                    'conversation_id': conversation_id,
                    'escalation_needed': False,
                    'knowledge_sources': [],
                    'channel_specific': self.get_channel_specific_response(channel, response_text)
                }
            
            # Za ostale namere, koristi standardnu pretragu
            logger.info("Standardna obrada pitanja...")
            
            # Pronađi relevantne informacije iz baze znanja
            relevant_knowledge = self.retrieve_relevant_knowledge(message)
            
            # Proveri da li imamo dovoljno relevantne informacije
            has_good_answer = False
            best_score = 0
            if relevant_knowledge:
                best_score = relevant_knowledge[0].get('relevance_score', 0)
                if best_score > 0.6:  # Prag relevantnosti
                    has_good_answer = True
            
            if not has_good_answer:
                # Nemamo dobar odgovor - ponudi kontakt opcije
                logger.info(f"Nema dovoljno relevantnog odgovora (najbolji score: {best_score:.2f})")
                return self.offer_contact_options(message, user_id, conversation_id, channel)
            
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
            
            logger.info("Odgovor uspešno generisan")
            return response
            
        except Exception as e:
            logger.error(f"❌ Greška u generate_response: {str(e)}")
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
        escalation_keywords = ['agent', 'operater', 'čovek', 'govori sa', 'uživo', 'live chat', 'kontakt']
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