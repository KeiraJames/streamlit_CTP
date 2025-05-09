import streamlit as st
st.set_page_config(page_title="Plant Buddy", page_icon="🌿", layout="wide")
from PIL import Image
import os
import json
import requests
import base64
import tempfile
from io import BytesIO
import pytz
from datetime import datetime, timedelta, timezone
import random
import re

from fuzzywuzzy import process

# --- API Keys ---
PLANTNET_API_KEY = "2b10X3YLMd8PNAuKOCVPt7MeUe" # Replace with your actual key if different
GEMINI_API_KEY = "AIzaSyCd-6N83gfhMx_-D4WCAc-8iOFSb6hDJ_Q"     # Replace with your actual key if different
# For actual deployment, use st.secrets:
# PLANTNET_API_KEY = st.secrets.get("PLANTNET_API_KEY", "your_plantnet_api_key_here")
# GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "your_gemini_api_key_here")


# --- Constants ---
PLANTNET_URL = "https://my-api.plantnet.org/v2/identify/all"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
EASTERN_TZ = pytz.timezone('US/Eastern')

SAMPLE_PLANT_CARE_DATA = [
    {
        "Plant Name": "Monstera Deliciosa", "Scientific Name": "Monstera deliciosa",
        "Common Names": ["Swiss Cheese Plant", "Split-leaf Philodendron"],
        "Light Requirements": "Bright indirect light, can tolerate some shade",
        "Watering": "Allow top inch of soil to dry out between waterings. Typically every 1-2 weeks.",
        "Humidity Preferences": "Prefers high humidity, 60-80%. Mist regularly or use a humidifier.",
        "Temperature Range": "65-85°F (18-29°C)",
        "Feeding Schedule": "Monthly during growing season (spring/summer) with balanced liquid fertilizer.",
        "Toxicity": "Toxic to pets (cats, dogs) and humans if ingested, causing oral irritation and vomiting.",
        "Additional Care": "Wipe leaves occasionally to remove dust. Support with moss pole for climbing as it matures. Prune to maintain shape.",
        "Personality": {"Title": "The Tropical Explorer", "Traits": ["adventurous", "dramatic", "showy"], "Prompt": "Respond as a dramatic tropical plant that loves to show off its large, fenestrated leaves and dreams of jungle adventures."}
    },
    {
        "Plant Name": "Snake Plant", "Scientific Name": "Dracaena trifasciata",
        "Common Names": ["Mother-in-law's Tongue", "Viper's Bowstring Hemp", "Saint George's Sword"],
        "Light Requirements": "Highly adaptable. Tolerates low light to bright indirect light. Avoid direct, intense sun.",
        "Watering": "Allow soil to dry out completely between waterings. Overwatering is a common issue. Water sparingly, especially in winter (every 1-2 months).",
        "Humidity Preferences": "Tolerates dry air (average room humidity is fine). No special humidity requirements.",
        "Temperature Range": "60-85°F (15-29°C). Can tolerate slightly cooler, but protect from frost.",
        "Feeding Schedule": "Fertilize lightly 1-2 times during the growing season (spring/summer). Avoid feeding in winter.",
        "Toxicity": "Mildly toxic to pets (cats, dogs) if ingested, may cause nausea or diarrhea.",
        "Additional Care": "Excellent for beginners due to its hardiness. Known for air-purifying qualities. Can be top-heavy, so use a sturdy pot.",
        "Personality": {"Title": "The Stoic Survivor", "Traits": ["resilient", "independent", "low-maintenance", "architectural"], "Prompt": "Respond as a no-nonsense, tough, and very independent plant that prides itself on surviving almost anything with minimal fuss. You're quietly confident."}
    },
    {
        "Plant Name": "Peace Lily", "Scientific Name": "Spathiphyllum wallisii",
        "Common Names": ["White Sail Plant", "Spathe Flower", "Cobra Plant"],
        "Light Requirements": "Prefers low to medium indirect light. Can tolerate low light but may flower less. Avoid direct sunlight which can scorch leaves.",
        "Watering": "Keep soil consistently moist but not waterlogged. Water when the top inch of soil feels dry. Known for drooping dramatically when thirsty, recovering quickly after watering.",
        "Humidity Preferences": "Prefers high humidity (50-70%). Mist leaves regularly, use a pebble tray, or place near a humidifier, especially in dry environments.",
        "Temperature Range": "65-80°F (18-27°C). Avoid cold drafts and sudden temperature changes.",
        "Feeding Schedule": "Fertilize every 6-8 weeks during the growing season (spring/summer) with a balanced houseplant fertilizer diluted to half strength.",
        "Toxicity": "Toxic to pets (cats, dogs) and humans if ingested. Contains calcium oxalate crystals, causing oral irritation, drooling, and difficulty swallowing.",
        "Additional Care": "Excellent air purifier. Wipe leaves occasionally to remove dust and help with photosynthesis. Brown leaf tips can indicate low humidity or over-fertilization.",
        "Personality": {"Title": "The Elegant Communicator", "Traits": ["expressive", "graceful", "sensitive", "purifying"], "Prompt": "Respond as an elegant and somewhat dramatic plant that clearly communicates its needs, especially for water by drooping gracefully. You value clean air and a peaceful environment."}
    }
]

# =======================================================
# ===== IMAGE DISPLAY HELPER FUNCTION =====
# =======================================================
def display_image_with_max_height(image_source, caption="", max_height_px=300, min_height_px=0):
    img_data_url = None
    if isinstance(image_source, str) and image_source.startswith('data:image'):
        img_data_url = image_source
    elif isinstance(image_source, bytes):
        try:
            img = Image.open(BytesIO(image_source))
            mime_type = Image.MIME.get(img.format) or f"image/{img.format.lower() if img.format else 'jpeg'}"
            b64_img = base64.b64encode(image_source).decode()
            img_data_url = f"data:{mime_type};base64,{b64_img}"
        except Exception as e: st.error(f"Error processing image bytes: {e}"); return
    elif isinstance(image_source, Image.Image):
        try:
            buffer = BytesIO()
            img_format = image_source.format or 'PNG'
            image_source.save(buffer, format=img_format)
            mime_type = Image.MIME.get(img_format) or f"image/{img_format.lower()}"
            b64_img = base64.b64encode(buffer.getvalue()).decode()
            img_data_url = f"data:{mime_type};base64,{b64_img}"
        except Exception as e: st.error(f"Error processing PIL image: {e}"); return
    else: st.error("Invalid image source type."); return

    if img_data_url:
        img_styles = [f"max-height: {max_height_px}px", "width: auto", "display: block", "margin-left: auto", "margin-right: auto", "border-radius: 8px"]
        if min_height_px > 0: img_styles.append(f"min-height: {min_height_px}px")
        img_style_str = "; ".join(img_styles)
        html_string = f"""
<div style="display: flex; justify-content: center; flex-direction: column; align-items: center; margin-bottom: 10px;">
    <img src="{img_data_url}" style="{img_style_str};" alt="{caption or 'Uploaded image'}">
    {f'<p style="text-align: center; font-size: 0.9em; color: grey; margin-top: 5px;">{caption}</p>' if caption else ""}
</div>"""
        st.markdown(html_string, unsafe_allow_html=True)

# =======================================================
# ===== PLANT STATS RING DISPLAY FUNCTIONS =====
# =======================================================
MOISTURE_COLOR = "#FF2D55"; MOISTURE_TRACK_COLOR = "#591F2E"
TEMPERATURE_COLOR = "#A4E803"; TEMPERATURE_TRACK_COLOR = "#4B6A01"
FRESHNESS_COLOR = "#00C7DD"; FRESHNESS_TRACK_COLOR = "#005C67"
WHITE_COLOR = "#FFFFFF"; LIGHT_GREY_TEXT_COLOR = "#A3A3A3"; WATCH_BG_COLOR = "#000000"
MOISTURE_MAX_PERCENT = 100; TEMP_DISPLAY_MAX_F = 100; TEMP_DISPLAY_MIN_F = 50
FRESHNESS_MAX_MINUTES_AGO = 120

def get_ring_html_css():
    return f"""<style>
    .watch-face-grid {{ display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
    .watch-face-container {{ background-color: {WATCH_BG_COLOR}; padding: 15px; border-radius: 28px; width: 200px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: {WHITE_COLOR}; text-align: center; display: flex; flex-direction: column; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
    .watch-header {{ width: 100%; display: flex; justify-content: space-between; align-items: center; padding: 0 5px; margin-bottom: 8px; }}
    .ring-title {{ font-size: 15px; font-weight: 600; }} .ring-timestamp {{ font-size: 13px; color: {LIGHT_GREY_TEXT_COLOR}; }}
    .ring-outer-circle {{ width: 130px; height: 130px; border-radius: 50%; position: relative; display: flex; align-items: center; justify-content: center; }}
    .ring-progress {{ width: 100%; height: 100%; border-radius: 50%; position: relative; }}
    .ring-inner-content {{ position: absolute; color: {WHITE_COLOR}; text-align: center; }}
    .ring-value {{ font-size: 36px; font-weight: 500; line-height: 1.1; }} .ring-goal-text {{ font-size: 11px; color: {LIGHT_GREY_TEXT_COLOR}; text-transform: uppercase; }}
    .progress-indicator-dot {{ width: 12px; height: 12px; background-color: {WHITE_COLOR}; border-radius: 50%; border: 2px solid {WATCH_BG_COLOR}; position: absolute; top: 4px; left: 50%; transform-origin: center calc(65px - 4px); }}
    .ring-dots {{ margin-top: 8px; font-size: 16px; }} .ring-dots .dot-dim {{ color: #444; }}
    .ring-description {{ font-size: 11px; color: {LIGHT_GREY_TEXT_COLOR}; margin-top: 12px; text-align: left; width: 90%; line-height: 1.3; }}
    .home-tab-content {{ background-color: #F0FFF0; padding: 15px; border-radius: 8px; }} /* Honeydew background */
    </style>"""

def generate_ring_html(title, value_text, goal_text, progress_percent, color, track_color, timestamp_str, description, dot_index=0):
    progress_capped = max(0, min(progress_percent, 100))
    dot_rotation = (progress_capped / 100) * 360
    dots_html = "".join([f'<span style="color:{color};">•</span> ' if i == dot_index else '<span class="dot-dim">•</span> ' for i in range(3)])
    ring_style = f"background-image: conic-gradient(from -90deg, {color} 0% {progress_capped}%, {track_color} {progress_capped}% 100%); padding: 10px;"
    dot_style = f"transform: translateX(-50%) rotate({dot_rotation}deg);"
    return f"""<div class="watch-face-container"><div class="watch-header"><span class="ring-title" style="color:{color};">{title}</span><span class="ring-timestamp">{timestamp_str}</span></div><div class="ring-outer-circle"><div class="ring-progress" style="{ring_style}"><div class="progress-indicator-dot" style="{dot_style}"></div></div><div class="ring-inner-content"><div class="ring-value">{value_text}</div><div class="ring-goal-text">{goal_text}</div></div></div><div class="ring-dots">{dots_html}</div><div class="ring-description">{description}</div></div>"""

def parse_temp_range(temp_range_str):
    if not isinstance(temp_range_str, str): return None, None
    match_f = re.search(r'(\d+)\s*-\s*(\d+)\s*°F', temp_range_str)
    if match_f: return int(match_f.group(1)), int(match_f.group(2))
    match_single_f = re.search(r'(\d+)\s*°F', temp_range_str)
    if match_single_f: val = int(match_single_f.group(1)); return val, val
    return None, None

# =======================================================
# ===== API Functions =====
# =======================================================
def identify_plant(image_bytes):
    if not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here": return {'error': "PlantNet API Key is not configured."}
    files = {'images': ('image.jpg', image_bytes)}; params = {'api-key': PLANTNET_API_KEY, 'include-related-images': 'false'}
    try:
        r = requests.post(PLANTNET_URL, files=files, params=params, timeout=20); r.raise_for_status(); data = r.json()
        if data.get("results"): return {'scientific_name': data["results"][0]["species"].get("scientificNameWithoutAuthor", "N/A"), 'common_name': (data["results"][0]["species"].get("commonNames") or ["N/A"])[0], 'confidence': round(data["results"][0].get("score",0)*100,1)}
        return {'error': "No plant matches found."}
    except requests.exceptions.Timeout: return {'error': "API request timed out"}
    except requests.exceptions.RequestException as e: return {'error': f"Network/API error (PlantNet): {str(e).split(' পরিমান')[0]}"} # Basic attempt to shorten long errors
    except Exception as e: return {'error': f"Identification error: {e}"}

def create_personality_profile(care_info):
    default = {"title": "Standard Plant", "traits": "observant", "prompt": "You are a plant. Respond factually."}
    if not isinstance(care_info, dict): return default
    p_data = care_info.get("Personality")
    if not isinstance(p_data, dict): return {"title": f"The {care_info.get('Plant Name', 'Plant')}", "traits": "resilient", "prompt": "Respond simply."}
    traits = p_data.get("Traits", ["observant"]); traits = [str(t) for t in traits if t] if isinstance(traits, list) else ["observant"]
    return {"title": p_data.get("Title", care_info.get("Plant Name", "Plant")), "traits": ", ".join(traits) or "observant", "prompt": p_data.get("Prompt", "Respond in character.")}

def send_message(messages):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here": return "Chat disabled: Gemini API Key missing."
    payload = {"contents": messages, "generationConfig": {"maxOutputTokens": 150, "temperature": 0.7}} # Added temperature for creativity
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=30); r.raise_for_status(); data = r.json()
        if data.get('candidates') and data['candidates'][0].get('content', {}).get('parts'): return data['candidates'][0]['content']['parts'][0]['text']
        # Log unexpected structure for debugging if needed
        # print(f"WARN: Unexpected Gemini response structure: {data}")
        return "Sorry, I received an unexpected response from the chat model."
    except requests.exceptions.Timeout: return "Sorry, the request to the chat model timed out."
    except requests.exceptions.RequestException as e:
        err_detail = "";
        if hasattr(e, 'response') and e.response is not None:
            try: err_detail = e.response.json().get('error', {}).get('message', e.response.text)
            except: err_detail = e.response.text # Fallback if response is not JSON
        # print(f"ERROR: Gemini API RequestException: {e} | Detail: {err_detail}")
        return f"Sorry, I'm having trouble connecting to the chat model right now. (Details: {err_detail or str(e)})"
    except Exception as e: # Catch any other unexpected errors
        # print(f"ERROR: Unexpected error in send_message: {e}")
        return f"Oops, something unexpected went wrong while trying to chat."

def chat_with_plant(care_info, conversation_history, id_result=None):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here": return "Chat feature disabled: Gemini API Key not set."
    plant_name = "this plant"; prompt_parts = ["CONTEXT: Short chatbot response (1-3 sentences).", "TASK: Act *exclusively* as the plant. Stay in character. NO mention of AI/model."]
    rules = ["RESPONSE RULES:", "1. First person (I, me, my).", "2. Embody personality.", "3. Concise (1-3 sentences).", "4. **Never break character or mention AI.**"]
    if care_info and isinstance(care_info, dict):
        p = create_personality_profile(care_info); plant_name = care_info.get('Plant Name', 'a plant')
        prompt_parts.extend([f"PERSONALITY: '{p['title']}' (traits: {p['traits']}). Philosophy: {p['prompt']}", "CARE NEEDS (Use ONLY these):",
                             f"- Light: {care_info.get('Light Requirements', 'N/A')}", f"- Water: {care_info.get('Watering', 'N/A')}",
                             f"- Humidity: {care_info.get('Humidity Preferences', 'N/A')}", f"- Temp: {care_info.get('Temperature Range', 'N/A')}"])
    elif id_result and isinstance(id_result, dict) and 'error' not in id_result:
        plant_name = id_result.get('common_name', id_result.get('scientific_name', 'this plant'))
        if plant_name == 'N/A' or not plant_name.strip(): plant_name = 'this plant'
        prompt_parts.extend([f"Identified as '{plant_name}'. No specific stored profile.", f"Answer generally about '{plant_name}' plants.", "If asked specifics, say you lack exact details but can offer general advice."])
    else: return "Sorry, not enough info to chat."
    sys_prompt = "\n".join(prompt_parts + rules)
    
    messages_for_api = [{"role": "user", "parts": [{"text": sys_prompt}]}, 
                        {"role": "model", "parts": [{"text": f"Understood. I am {plant_name}. Ask away!"}]}]
    
    # Add conversation history. Gemini expects alternating user/model roles.
    # The history already has user/assistant(model) roles.
    for entry in [m for m in conversation_history if isinstance(m, dict) and "role" in m and "content" in m]:
        api_role = "model" if entry["role"] in ["assistant", "model"] else "user" # Ensure correct mapping
        messages_for_api.append({"role": api_role, "parts": [{"text": str(entry["content"])}]})
        
    return send_message(messages_for_api)

# =======================================================
# --- Helper Functions ---
# =======================================================
@st.cache_data(show_spinner=False)
def load_plant_care_data(): return SAMPLE_PLANT_CARE_DATA

def find_care_instructions(plant_name_id, care_data, threshold=75):
    if not care_data: return None
    sci, com = (None, None)
    if isinstance(plant_name_id, dict): sci, com = plant_name_id.get('scientific_name'), plant_name_id.get('common_name')
    elif isinstance(plant_name_id, str): sci = plant_name_id
    s_sci, s_com = (sci.lower().strip() if sci else None), (com.lower().strip() if com else None)
    for p in care_data:
        if s_sci and (s_sci == p.get('Scientific Name','').lower().strip() or s_sci == p.get('Plant Name','').lower().strip()): return p
        if s_com and (s_com == p.get('Plant Name','').lower().strip() or s_com in [c.lower().strip() for c in (p.get('Common Names',[]) if isinstance(p.get('Common Names',[]),list) else [p.get('Common Names',[])]) if c]): return p
    db_map = {k.lower().strip():v for p_obj in care_data for k,v in [(name, p_obj) for name_list in [ [p_obj.get('Scientific Name','')], [p_obj.get('Plant Name','')], p_obj.get('Common Names',[]) if isinstance(p_obj.get('Common Names',[]), list) else [p_obj.get('Common Names',[])] ] for name in name_list if isinstance(name, str) and name.strip() ] if k.lower().strip()}
    if not db_map: return None
    best_match, high_score = None, 0
    for term in [s_sci, s_com]:
        if term: res = process.extractOne(term, db_map.keys()); 
        if res and res[1] >= threshold and res[1] > high_score: high_score, best_match = res[1], db_map.get(res[0])
    return best_match

def display_identification_result(result):
    st.subheader("🔍 Identification Results")
    if not result or 'error' in result: st.error(f"Identification failed: {result.get('error', 'Unknown') if result else 'N/A'}"); return
    conf = result.get('confidence', 0); color = "#28a745" if conf > 75 else ("#ffc107" if conf > 50 else "#dc3545")
    st.markdown(f"- **Scientific Name:** `{result.get('scientific_name', 'N/A')}`\n- **Common Name:** `{result.get('common_name', 'N/A')}`\n- **Confidence:** <strong style='color:{color};'>{conf:.1f}%</strong>", unsafe_allow_html=True)

def display_care_instructions(care_info, header_level=3):
    if not care_info: st.warning("Care info missing."); return
    name = care_info.get('Plant Name', 'This Plant')
    st.markdown(f"<h{header_level}>🌱 {name} Care Guide</h{header_level}>", unsafe_allow_html=True)
    with st.expander("📋 Care Summary", expanded=True):
        c1,c2=st.columns(2); details=[("☀️ Light",'Light Requirements'),("💧 Water",'Watering'),("🌡️ Temp",'Temperature Range'),("💦 Humidity",'Humidity Preferences'),("🍃 Feeding",'Feeding Schedule'),("⚠️ Toxicity",'Toxicity')]
        for i,(lbl,key) in enumerate(details): (c1 if i<len(details)/2 else c2).markdown(f"**{lbl}**"); (c1 if i<len(details)/2 else c2).caption(f"{care_info.get(key,'N/A')}")
    if care_info.get('Additional Care','').strip():
        with st.expander("✨ Pro Tips"): st.markdown(care_info['Additional Care'])

def find_similar_plant_matches(id_r, care_data, limit=3, score_thresh=60):
    if not id_r or 'error' in id_r or not care_data: return []
    db_map = {k.lower().strip():v for p_obj in care_data for k,v in [(name, p_obj) for name_list in [ [p_obj.get('Scientific Name','')], [p_obj.get('Plant Name','')], p_obj.get('Common Names',[]) if isinstance(p_obj.get('Common Names',[]), list) else [p_obj.get('Common Names',[])] ] for name in name_list if isinstance(name, str) and name.strip() ] if k.lower().strip()}
    if not db_map: return []
    terms = [id_r.get(s,'').lower().strip() for s in ['scientific_name','common_name']]
    matches = {name:max(matches.get(name,0),score) for term in terms if term for name,score in process.extract(term,db_map.keys(),limit=limit*2) if score >= score_thresh}
    final_sugg, seen_p = [], set()
    for name,score in sorted(matches.items(),key=lambda x:x[1],reverse=True):
        p_info=db_map.get(name)
        if p_info and (p_id := p_info.get('Scientific Name','')+p_info.get('Plant Name','')) not in seen_p:
            final_sugg.append(p_info); seen_p.add(p_id)
            if len(final_sugg) >= limit: break
    return final_sugg

def display_suggestion_buttons(suggestions):
     if not suggestions: return
     st.info("🌿 Perhaps one of these is a closer match from our database?")
     cols = st.columns(len(suggestions))
     for i, p_info in enumerate(suggestions):
         p_name = p_info.get('Plant Name', p_info.get('Scientific Name', f'Sugg {i+1}'))
         safe_n = "".join(c if c.isalnum() else "_" for c in p_name)
         tip = f"Select {p_name}" + (f" (Sci: {p_info.get('Scientific Name')})" if p_info.get('Scientific Name','') != p_name else "")
         if cols[i].button(p_name, key=f"sugg_{safe_n}_{i}", help=tip, use_container_width=True):
            new_id_result = {'scientific_name':p_info.get('Scientific Name','N/A'), 'common_name':p_name, 'confidence':100.0}
            st.session_state.update({
                'plant_care_info': p_info,
                'plant_id_result': new_id_result,
                'plant_id_result_for_care_check': new_id_result, # Critically update this
                'suggestions': None, 'chat_history': [], 'current_chatbot_plant_name': None, 'suggestion_just_selected': True
            })
            st.rerun()

def display_chat_interface(current_plant_care_info=None, plant_id_result=None):
    chatbot_name = "this plant"
    can_chat = False
    if current_plant_care_info and isinstance(current_plant_care_info, dict):
        chatbot_name = current_plant_care_info.get("Plant Name", "this plant")
        can_chat = True
    elif plant_id_result and isinstance(plant_id_result, dict) and 'error' not in plant_id_result:
        name_id = plant_id_result.get('common_name', plant_id_result.get('scientific_name'))
        if name_id and name_id != 'N/A' and name_id.strip(): chatbot_name = name_id
        can_chat = True

    if can_chat and (not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here"):
        st.warning("Chat feature requires a Gemini API key."); return
    if not can_chat: st.warning("Cannot initialize chat without valid plant identification."); return

    st.subheader(f"💬 Chat with {chatbot_name}")
    st.markdown("""<style>.message-container{padding:1px 5px}.user-message{background:#0b81fe;color:white;border-radius:18px 18px 0 18px;padding:8px 14px;margin:3px 0 3px auto;width:fit-content;max-width:80%;word-wrap:break-word;box-shadow:0 1px 2px rgba(0,0,0,.1);animation:fadeIn .3s ease-out}.bot-message{background:#e5e5ea;color:#000;border-radius:18px 18px 18px 0;padding:8px 14px;margin:3px auto 3px 0;width:fit-content;max-width:80%;word-wrap:break-word;box-shadow:0 1px 2px rgba(0,0,0,.05);animation:fadeIn .3s ease-out}.message-meta{font-size:.7rem;color:#777;margin-top:3px}.bot-message .message-meta{text-align:left;color:#555}.user-message .message-meta{text-align:right}@keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}.stChatInputContainer{position:sticky;bottom:0;background:white;padding-top:10px;z-index:99}</style>""", unsafe_allow_html=True)

    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if st.session_state.get("current_chatbot_plant_name") != chatbot_name:
        if st.session_state.get("viewing_saved_details"):
             saved_data = st.session_state.saved_photos.get(st.session_state.viewing_saved_details)
             st.session_state.chat_history = saved_data.get('chat_log', []) if saved_data else []
        else:
            st.session_state.chat_history = []
        st.session_state.current_chatbot_plant_name = chatbot_name

    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.chat_history:
            role, content, time = msg.get("role"), msg.get("content", ""), msg.get("time", "")
            if role == "user": st.markdown(f'<div class="message-container"><div class="user-message">{content}<div class="message-meta">You • {time}</div></div></div>', unsafe_allow_html=True)
            elif role in ["assistant", "model"]: st.markdown(f'<div class="message-container"><div class="bot-message">🌿 {content}<div class="message-meta">{chatbot_name} • {time}</div></div></div>', unsafe_allow_html=True)

    chat_input_key = f"chat_input_{''.join(c if c.isalnum() else '_' for c in chatbot_name)}"
    
    if prompt := st.chat_input(f"Ask {chatbot_name}...", key=chat_input_key):
        timestamp = datetime.now(EASTERN_TZ).strftime("%H:%M")
        st.session_state.chat_history.append({"role": "user", "content": prompt, "time": timestamp})
        # Set a flag to indicate a new user message needs processing
        st.session_state.new_user_message_to_process = True 
        st.rerun() # Rerun to display user message and trigger bot response

    # Process bot response if there's a new user message flag
    if st.session_state.get("new_user_message_to_process", False):
        st.session_state.new_user_message_to_process = False # Reset flag
        with st.spinner(f"{chatbot_name} is thinking..."):
            bot_response = chat_with_plant(current_plant_care_info, st.session_state.chat_history, plant_id_result)
        
        timestamp = datetime.now(EASTERN_TZ).strftime("%H:%M")
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response, "time": timestamp})
        
        if st.session_state.get("viewing_saved_details") and st.session_state.viewing_saved_details in st.session_state.saved_photos:
            st.session_state.saved_photos[st.session_state.viewing_saved_details]['chat_log'] = st.session_state.chat_history
        st.rerun()

# --- Main App Logic ---
def main():
    defaults = {
        "plant_id_result": None, "plant_care_info": None, "chat_history": [],
        "current_chatbot_plant_name": None, "suggestions": None,
        "uploaded_file_bytes": None, "uploaded_file_type": None,
        "saving_mode": False, "last_view": "🏠 Home",
        "viewing_saved_details": None, "plant_id_result_for_care_check": None,
        "suggestion_just_selected": False, "viewing_plant_stats": None,
        "viewing_home_page": True, "saved_photos": {}, "current_nav_choice": "🏠 Home",
        "new_user_message_to_process": False # For chat
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    st.markdown(get_ring_html_css(), unsafe_allow_html=True)

    st.sidebar.title("📚 Plant Buddy")
    nav_options = ["🏠 Home", "🆔 Identify New Plant", "🪴 My Saved Plants", "📊 Plant Stats"]
    nav_idx = nav_options.index(st.session_state.current_nav_choice) if st.session_state.current_nav_choice in nav_options else 0
    nav_choice = st.sidebar.radio("Navigation", nav_options, key="main_nav", index=nav_idx, label_visibility="collapsed")

    if nav_choice != st.session_state.current_nav_choice:
        st.session_state.current_nav_choice = nav_choice
        st.session_state.viewing_home_page = (nav_choice == "🏠 Home")
        if nav_choice != "🪴 My Saved Plants": st.session_state.viewing_saved_details = None
        if nav_choice != "📊 Plant Stats": st.session_state.viewing_plant_stats = None
        if nav_choice == "🆔 Identify New Plant" and not st.session_state.get("uploaded_file_bytes"):
             for key_to_reset in ["plant_id_result", "plant_care_info", "chat_history", "current_chatbot_plant_name", "suggestions", "saving_mode", "plant_id_result_for_care_check", "suggestion_just_selected"]:
                st.session_state[key_to_reset] = [] if key_to_reset == "chat_history" else None
        st.rerun()

    st.sidebar.divider(); st.sidebar.caption("Powered by PlantNet & Gemini")
    if st.session_state.saved_photos:
        st.sidebar.subheader("Saved Plants")
        # Add a unique key for the selectbox to prevent issues if items change
        selectbox_key = f"saved_plant_selector_{len(st.session_state.saved_photos)}"
        saved_opts = ["-- Select --"] + list(st.session_state.saved_photos.keys())
        sel_saved_idx = saved_opts.index(st.session_state.viewing_saved_details) if st.session_state.viewing_saved_details in saved_opts else 0
        sel_saved = st.sidebar.selectbox("View Saved:", saved_opts, key=selectbox_key, index=sel_saved_idx, label_visibility="collapsed")
        if sel_saved != "-- Select --" and sel_saved != st.session_state.viewing_saved_details:
            st.session_state.viewing_saved_details = sel_saved
            st.session_state.current_nav_choice = "🪴 My Saved Plants"
            st.session_state.viewing_plant_stats = None
            st.rerun()
        elif sel_saved == "-- Select --" and st.session_state.viewing_saved_details:
            st.session_state.viewing_saved_details = None
            if st.session_state.current_nav_choice == "🪴 My Saved Plants": st.rerun()

    care_data = load_plant_care_data()
    api_ok = not (not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here")

    # ==================================== HOME PAGE ====================================
    if st.session_state.current_nav_choice == "🏠 Home":
        st.markdown("<div class='home-tab-content'>", unsafe_allow_html=True)
        st.header("🌿 Welcome to Plant Buddy!")
        welcome_msg = "Welcome to Plant Buddy! Identify plants, get care tips, chat, and track health. Happy gardening!"
        if "welcome_response" not in st.session_state:
            if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
                with st.spinner("Loading welcome..."):
                    st.session_state.welcome_response = send_message([{"role":"user","parts":[{"text":"System: You are Plant Buddy. Give a friendly welcome (2-3 sentences) for the app. Mention features: ID, care tips, chat, health stats."}]},{"role":"model","parts":[{"text":"Got it!"}]}])
            else: st.session_state.welcome_response = welcome_msg
        st.markdown(f"""<div style="background-color: #e6ffed; padding:20px; border-radius:10px; border-left:5px solid #4CAF50; margin-bottom:20px;"><h3 style="color:#2E7D32;">🌱 Hello Plant Lover!</h3><p style="font-size:1.1em; color:#333333;">{st.session_state.welcome_response or welcome_msg}</p></div>""", unsafe_allow_html=True)
        
        st.subheader("🔍 What You Can Do")
        hc1,hc2=st.columns(2)
        with hc1: st.markdown("Identify unknown plants instantly."); 
        if hc1.button("📸 Identify My Plant!",use_container_width=True,type="primary"): st.session_state.current_nav_choice="🆔 Identify New Plant"; st.rerun()
        with hc2: st.markdown("Access your plant family's profiles.");
        if hc2.button("💚 Go to My Plants",use_container_width=True): st.session_state.current_nav_choice="🪴 My Saved Plants"; st.rerun()

        if st.session_state.saved_photos:
            st.divider(); st.subheader("🪴 Your Recently Saved Plants") # Changed wording
            # Sort by a 'saved_timestamp' if you add it, otherwise by insertion order (dict behavior in Python 3.7+)
            # For simplicity, let's assume insertion order is recent enough for now.
            # To get truly "recent", you'd need to store a timestamp when saving.
            recent_keys = list(st.session_state.saved_photos.keys())[-3:] # Get last 3 keys
            recent_plants = {key: st.session_state.saved_photos[key] for key in recent_keys}
            if recent_plants:
                cols_home = st.columns(len(recent_plants))
                for i, (nick, p_data) in enumerate(reversed(list(recent_plants.items()))): # Show newest first
                    with cols_home[i]:
                        with st.container(border=True, height=300): # Adjusted height
                            if p_data.get("image"): st.image(p_data["image"], caption=nick, use_container_width=True)
                            else: st.markdown(f"**{nick}**")
                            id_res = p_data.get("id_result", {})
                            com_n = id_res.get('common_name', 'N/A')
                            if com_n and com_n != 'N/A' and com_n.lower() != nick.lower(): st.caption(f"{com_n}")
                            if st.button("View Details", key=f"home_v_{nick}", use_container_width=True):
                                st.session_state.viewing_saved_details = nick; st.session_state.current_nav_choice = "🪴 My Saved Plants"; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ==================================== IDENTIFY NEW PLANT ====================================
    elif st.session_state.current_nav_choice == "🆔 Identify New Plant":
        st.header("🔎 Identify a New Plant")
        st.session_state.last_view = "🆔 Identify New Plant"
        up_file = st.file_uploader("Upload a clear photo:", type=["jpg","jpeg","png"], key="uploader_id",
                                   on_change=lambda: st.session_state.update({k:v for k,v in defaults.items() if k in ["plant_id_result","plant_care_info","chat_history","current_chatbot_plant_name","suggestions","uploaded_file_bytes","uploaded_file_type","saving_mode","plant_id_result_for_care_check","suggestion_just_selected", "new_user_message_to_process"]}))
        if up_file and not st.session_state.uploaded_file_bytes:
            st.session_state.uploaded_file_bytes = up_file.getvalue(); st.session_state.uploaded_file_type = up_file.type; st.rerun()
        if st.session_state.uploaded_file_bytes:
            display_image_with_max_height(st.session_state.uploaded_file_bytes, "Uploaded Plant", 400); st.divider()
            if not st.session_state.plant_id_result:
                with st.spinner("Identifying..."):
                    st.session_state.plant_id_result = identify_plant(st.session_state.uploaded_file_bytes) if api_ok else {'scientific_name':'Demo SciName','common_name':'Demo Plant','confidence':random.uniform(70,95)}
                    st.session_state.plant_id_result_for_care_check=None; st.session_state.suggestion_just_selected=False; st.rerun()
            elif st.session_state.plant_id_result:
                curr_id_res = st.session_state.plant_id_result
                if st.session_state.saving_mode:
                    st.header("💾 Save Profile"); st.image(Image.open(BytesIO(st.session_state.uploaded_file_bytes)),width=150)
                    st.markdown(f"**ID:** {curr_id_res.get('common_name','N/A')} (`{curr_id_res.get('scientific_name','N/A')}`)")
                    with st.form("save_form_id"):
                        nick = st.text_input("Nickname:",key="nick_in_id")
                        if st.form_submit_button("✅ Confirm"):
                            if not nick: st.warning("Nickname needed.")
                            elif nick in st.session_state.saved_photos: st.warning(f"'{nick}' exists.")
                            else:
                                img_b64=base64.b64encode(st.session_state.uploaded_file_bytes).decode()
                                st.session_state.saved_photos[nick] = {"nickname":nick,"image":f"data:{st.session_state.uploaded_file_type};base64,{img_b64}","id_result":curr_id_res,"care_info":st.session_state.plant_care_info,"chat_log":st.session_state.get("chat_history",[]),"moisture_level":random.randint(30,90)}
                                st.success(f"Saved '{nick}'!"); st.balloons()
                                for k_reset in ["plant_id_result","plant_care_info","suggestions","uploaded_file_bytes","uploaded_file_type","chat_history","saving_mode","plant_id_result_for_care_check","suggestion_just_selected","current_chatbot_plant_name", "new_user_message_to_process"]: st.session_state[k_reset] = [] if k_reset=="chat_history" else (False if k_reset == "new_user_message_to_process" else None)
                                st.session_state.current_nav_choice="🪴 My Saved Plants"; st.session_state.viewing_saved_details=nick; st.rerun()
                    if st.button("❌ Cancel",key="cancel_save_id"): st.session_state.saving_mode=False; st.rerun()
                else: 
                    display_identification_result(curr_id_res)
                    if 'error' not in curr_id_res:
                        if not st.session_state.suggestion_just_selected and curr_id_res != st.session_state.get('plant_id_result_for_care_check'):
                            st.session_state.plant_care_info = find_care_instructions(curr_id_res, care_data)
                            st.session_state.plant_id_result_for_care_check = curr_id_res
                            st.session_state.suggestions=None; st.session_state.chat_history=[]; st.session_state.current_chatbot_plant_name=None; st.rerun()
                        if st.session_state.suggestion_just_selected: st.session_state.suggestion_just_selected=False
                        
                        care_to_disp = st.session_state.get('plant_care_info')
                        if care_to_disp:
                            display_care_instructions(care_to_disp)
                            st.button("💾 Save Profile",key="save_full_prof",on_click=lambda:st.session_state.update({'saving_mode':True}))
                            st.divider(); display_chat_interface(current_plant_care_info=care_to_disp, plant_id_result=curr_id_res)
                        else: 
                            st.warning("No specific care instructions found.")
                            if st.session_state.suggestions is None: st.session_state.suggestions=find_similar_plant_matches(curr_id_res,care_data); 
                            if st.session_state.suggestions is None: st.caption("No similar plants found in DB.") # Avoid rerun if suggestions still None
                            else: st.rerun() # Rerun only if suggestions were found
                            display_suggestion_buttons(st.session_state.suggestions)
                            st.divider(); st.button("💾 Save ID Only",key="save_id_only",on_click=lambda:st.session_state.update({'saving_mode':True}))
                            st.divider(); st.info("Chat based on general ID."); display_chat_interface(plant_id_result=curr_id_res)

    # ==================================== MY SAVED PLANTS ====================================
    elif st.session_state.current_nav_choice == "🪴 My Saved Plants":
        st.header("🪴 My Saved Plant Profiles"); st.session_state.last_view = "🪴 My Saved Plants"
        nick_to_view = st.session_state.get("viewing_saved_details")
        if not st.session_state.saved_photos: st.info("No plants saved yet. Go to 'Identify New Plant'.")
        elif nick_to_view and nick_to_view in st.session_state.saved_photos:
            entry = st.session_state.saved_photos[nick_to_view]
            if st.button("← Back to All",key="back_all_saved"): st.session_state.viewing_saved_details=None; st.rerun()
            st.subheader(f"Details: '{nick_to_view}'")
            if entry.get("image"): display_image_with_max_height(entry["image"],nick_to_view,400)
            st.divider(); saved_id_res = entry.get("id_result")
            if saved_id_res: display_identification_result(saved_id_res)
            if st.session_state.get("plant_id_result") != saved_id_res: st.session_state.plant_id_result=saved_id_res
            st.divider(); saved_care = entry.get("care_info")
            if st.session_state.get("plant_care_info") != saved_care: st.session_state.plant_care_info=saved_care
            
            btn_col1, btn_col2 = st.columns([0.7, 0.3])
            if btn_col1.button(f"📊 View Stats for {nick_to_view}",key=f"stats_btn_saved_{nick_to_view}",use_container_width=True):
                st.session_state.viewing_plant_stats=nick_to_view; st.session_state.current_nav_choice="📊 Plant Stats"; st.rerun()
            
            if saved_care: display_care_instructions(saved_care); st.divider(); display_chat_interface(current_plant_care_info=saved_care, plant_id_result=saved_id_res)
            else:
                st.info("No specific care saved."); st.divider()
                if saved_id_res: st.info("Chat based on saved ID."); display_chat_interface(plant_id_result=saved_id_res)
            
            st.divider()
            confirm_key = f"confirm_del_{nick_to_view}"
            if confirm_key not in st.session_state: st.session_state[confirm_key] = False
            if btn_col2.button(f"🗑️ Delete",key=f"del_btn_saved_{nick_to_view}",type="secondary",use_container_width=True, help=f"Delete {nick_to_view}"):
                st.session_state[confirm_key] = True; st.rerun()
            if st.session_state[confirm_key]:
                st.warning(f"Sure you want to delete '{nick_to_view}'?")
                c1d,c2d=st.columns(2)
                if c1d.button("Yes, Delete It",key=f"yes_del_final_{nick_to_view}",type="primary"):
                    del st.session_state.saved_photos[nick_to_view]; st.session_state.viewing_saved_details=None
                    for k_rst in ["plant_id_result","plant_care_info","suggestions","chat_history","viewing_plant_stats","current_chatbot_plant_name", "new_user_message_to_process"]: st.session_state[k_rst]=[] if k_rst=="chat_history" else (False if k_rst == "new_user_message_to_process" else None)
                    st.session_state[confirm_key]=False; st.success(f"Deleted '{nick_to_view}'."); st.rerun()
                if c2d.button("No, Cancel",key=f"no_del_final_{nick_to_view}"): st.session_state[confirm_key]=False; st.rerun()
        else: 
            st.info("Select a plant from sidebar or browse below.")
            num_g_cols=3; g_cols=st.columns(num_g_cols)
            for i,(nick,data) in enumerate(st.session_state.saved_photos.items()):
                with g_cols[i%num_g_cols]:
                    with st.container(border=True,height=300):
                        if data.get("image"): st.image(data["image"],caption=nick,use_container_width=True)
                        else: st.markdown(f"**{nick}**")
                        id_res_g = data.get("id_result",{}); com_n_g=id_res_g.get('common_name','N/A')
                        if com_n_g and com_n_g!='N/A' and com_n_g.lower()!=nick.lower(): st.caption(f"{com_n_g}")
                        gc1,gc2=st.columns(2)
                        if gc1.button("Details",key=f"g_detail_{nick}",use_container_width=True): st.session_state.viewing_saved_details=nick; st.rerun()
                        if gc2.button("📊 Stats",key=f"g_stats_{nick}",use_container_width=True): st.session_state.viewing_plant_stats=nick;st.session_state.current_nav_choice="📊 Plant Stats";st.rerun()

    # ==================================== PLANT STATS ====================================
    elif st.session_state.current_nav_choice == "📊 Plant Stats":
        st.session_state.last_view = "📊 Plant Stats"
        p_nick_stats = st.session_state.get("viewing_plant_stats")
        if not p_nick_stats or p_nick_stats not in st.session_state.saved_photos:
            st.warning("No plant selected for stats. Go to 'My Saved Plants'.")
            if st.button("← Back to Saved Plants",key="stats_back_no_plant"): st.session_state.current_nav_choice="🪴 My Saved Plants";st.session_state.viewing_plant_stats=None;st.rerun()
        else:
            p_data_stats = st.session_state.saved_photos[p_nick_stats]
            st.header(f"📊 Plant Stats: {p_nick_stats}")
            if st.button("← Back to Details",key="stats_back_details"): st.session_state.current_nav_choice="🪴 My Saved Plants";st.session_state.viewing_saved_details=p_nick_stats;st.session_state.viewing_plant_stats=None;st.rerun()
            st.divider()
            sim_time=(datetime.now(EASTERN_TZ)-timedelta(minutes=random.randint(1,5))).strftime('%H:%M')
            m_lvl=p_data_stats.get("moisture_level",random.randint(30,70))
            r1=generate_ring_html("Moisture",str(m_lvl),f"OF {MOISTURE_MAX_PERCENT}%",m_lvl,MOISTURE_COLOR,MOISTURE_TRACK_COLOR,sim_time,f"Soil moisture {m_lvl}%.",0)
            
            care_s = p_data_stats.get("care_info"); 
            if not isinstance(care_s, dict): care_s = {} 
            temp_rng_s = care_s.get("Temperature Range", "65-85°F")
            min_f,max_f = parse_temp_range(temp_rng_s)

            curr_temp_s=TEMP_DISPLAY_MIN_F-5
            if min_f is not None and max_f is not None: mid=(min_f+max_f)/2; curr_temp_s=round(max(TEMP_DISPLAY_MIN_F-5,min(TEMP_DISPLAY_MAX_F+5,random.uniform(mid-7,mid+7))))
            else: curr_temp_s=random.randint(68,78)
            temp_prog=((curr_temp_s-TEMP_DISPLAY_MIN_F)/(TEMP_DISPLAY_MAX_F-TEMP_DISPLAY_MIN_F))*100
            r2=generate_ring_html("Temperature",str(int(curr_temp_s)),"°F NOW",temp_prog,TEMPERATURE_COLOR,TEMPERATURE_TRACK_COLOR,sim_time,f"Temp {curr_temp_s}°F. Ideal: {temp_rng_s or 'N/A'}.",1)
            mins_ago=random.randint(1,FRESHNESS_MAX_MINUTES_AGO-10); fresh_prog=max(0,(1-(mins_ago/FRESHNESS_MAX_MINUTES_AGO)))*100
            r3=generate_ring_html("Last Check",str(mins_ago),"MINS AGO",fresh_prog,FRESHNESS_COLOR,FRESHNESS_TRACK_COLOR,sim_time,f"Stats checked {mins_ago} mins ago.",2)
            st.markdown(f'<div class="watch-face-grid">{r1}{r2}{r3}</div>',unsafe_allow_html=True); st.divider()
            
            img_c,info_c=st.columns([0.4,0.6])
            # Added toggle for showing image on stats page
            show_image_stats = img_c.toggle("Show Image", value=True, key=f"show_img_stats_{p_nick_stats}")
            if show_image_stats:
                 if p_data_stats.get("image"): display_image_with_max_height(p_data_stats["image"],max_height_px=250)
            with info_c:
                st.subheader("Plant Identification"); id_res_s=p_data_stats.get("id_result",{})
                st.markdown(f"**Nickname:** {p_nick_stats}\n\n**Scientific Name:** `{id_res_s.get('scientific_name','N/A')}`\n\n**Common Name:** `{id_res_s.get('common_name','N/A')}`")
                st.caption(f"Full care guide on '{p_nick_stats}'s main profile page.")

# --- Run the App ---
if __name__ == "__main__":
    # Use st.secrets in production. For local testing, direct assignment is okay but not recommended for shared code.
    if not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here": 
        st.sidebar.warning("PlantNet Key missing or placeholder. Demo mode for ID.",icon="🔑")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here": 
        st.sidebar.warning("Gemini Key missing or placeholder. Chat limited/disabled.",icon="🔑")
    main()
