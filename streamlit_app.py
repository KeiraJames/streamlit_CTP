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
PLANTNET_API_KEY = st.secrets.get("PLANTNET_API_KEY", "your_plantnet_api_key_here")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "your_gemini_api_key_here")

# --- Constants ---
PLANTNET_URL = "https://my-api.plantnet.org/v2/identify/all"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
EASTERN_TZ = pytz.timezone('US/Eastern')

# --- Sample Plant Care Data ---
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
        img_styles = [f"max-height: {max_height_px}px", "width: auto", "display: block", "margin-left: auto", "margin-right: auto", "border-radius: 8px", "object-fit: contain;"] # Added object-fit
        if min_height_px > 0: img_styles.append(f"min-height: {min_height_px}px")
        img_style_str = "; ".join(img_styles)
        # For card-like display, ensure the div itself can be centered if image is smaller than max_height
        html_string = f"""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: {max_height_px}px; margin-bottom: 5px;">
    <img src="{img_data_url}" style="{img_style_str};" alt="{caption or 'Uploaded image'}">
</div>
{f'<p style="text-align: center; font-size: 0.9em; color: grey; margin-top: 0px; margin-bottom: 5px;">{caption}</p>' if caption else ""}
"""
        st.markdown(html_string, unsafe_allow_html=True)


# =======================================================
# ===== PLANT STATS RING DISPLAY FUNCTIONS =====
# =======================================================
MOISTURE_COLOR, MOISTURE_TRACK_COLOR = "#FF2D55", "#591F2E"
TEMPERATURE_COLOR, TEMPERATURE_TRACK_COLOR = "#A4E803", "#4B6A01"
FRESHNESS_COLOR, FRESHNESS_TRACK_COLOR = "#00C7DD", "#005C67"
WHITE_COLOR, LIGHT_GREY_TEXT_COLOR, WATCH_BG_COLOR = "#FFFFFF", "#A3A3A3", "#000000"
MOISTURE_MAX_PERCENT, TEMP_DISPLAY_MAX_F, TEMP_DISPLAY_MIN_F, FRESHNESS_MAX_MINUTES_AGO = 100, 100, 50, 120

def get_ring_html_css():
    return f"""<style>
    .watch-face-grid {{ display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
    .watch-face-container {{ background-color: {WATCH_BG_COLOR}; padding: 15px; border-radius: 28px; width: 200px; height: auto; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: {WHITE_COLOR}; text-align: center; display: flex; flex-direction: column; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
    .watch-header {{ width: 100%; display: flex; justify-content: space-between; align-items: center; padding: 0 5px; margin-bottom: 8px; }}
    .ring-title {{ font-size: 15px; font-weight: 600; }} .ring-timestamp {{ font-size: 13px; color: {LIGHT_GREY_TEXT_COLOR}; }}
    .ring-outer-circle {{ width: 130px; height: 130px; border-radius: 50%; position: relative; display: flex; align-items: center; justify-content: center; }}
    .ring-progress {{ width: 100%; height: 100%; border-radius: 50%; position: relative; }}
    .ring-inner-content {{ position: absolute; color: {WHITE_COLOR}; text-align: center; }}
    .ring-value {{ font-size: 36px; font-weight: 500; line-height: 1.1; }} .ring-goal-text {{ font-size: 11px; color: {LIGHT_GREY_TEXT_COLOR}; text-transform: uppercase; }}
    .progress-indicator-dot {{ width: 12px; height: 12px; background-color: {WHITE_COLOR}; border-radius: 50%; border: 2px solid {WATCH_BG_COLOR}; position: absolute; top: 4px; left: 50%; transform-origin: center calc(65px - 4px); }}
    .ring-dots {{ margin-top: 8px; font-size: 16px; }} .ring-dots .dot-dim {{ color: #444; }}
    .ring-description {{ font-size: 11px; color: {LIGHT_GREY_TEXT_COLOR}; margin-top: 12px; text-align: left; width: 90%; line-height: 1.3; }}
    .home-page-background {{ padding: 20px; border-radius: 10px; background-image: linear-gradient(to bottom right, #f0fff0, #d0f0c0); }}
    </style>"""

def generate_ring_html(title, value_text, goal_text, progress_percent, color, track_color, timestamp_str, description, dot_index=0):
    progress_capped = max(0, min(progress_percent, 100))
    dot_rotation_deg = (progress_capped / 100) * 360
    dots = "".join([f'<span class="dot-main" style="color:{color};">•</span> ' if i == dot_index else '<span class="dot-dim">•</span> ' for i in range(3)])
    ring_style = f"background-image: conic-gradient(from -90deg, {color} 0% {progress_capped}%, {track_color} {progress_capped}% 100%); padding: 10px;"
    dot_style = f"transform: translateX(-50%) rotate({dot_rotation_deg}deg);"
    return f"""<div class="watch-face-container"><div class="watch-header"><span class="ring-title" style="color:{color};">{title}</span><span class="ring-timestamp">{timestamp_str}</span></div><div class="ring-outer-circle"><div class="ring-progress" style="{ring_style}"><div class="progress-indicator-dot" style="{dot_style}"></div></div><div class="ring-inner-content"><div class="ring-value">{value_text}</div><div class="ring-goal-text">{goal_text}</div></div></div><div class="ring-dots">{dots}</div><div class="ring-description">{description}</div></div>"""

def parse_temp_range(temp_range_str):
    if not temp_range_str or not isinstance(temp_range_str, str): return None, None
    match_f = re.search(r'(\d+)\s*-\s*(\d+)\s*°F', temp_range_str)
    if match_f: return int(match_f.group(1)), int(match_f.group(2))
    match_single_f = re.search(r'(\d+)\s*°F', temp_range_str)
    if match_single_f: val = int(match_single_f.group(1)); return val, val
    return None, None

# =======================================================
# ===== API Functions =====
# =======================================================
def identify_plant(image_bytes):
    if not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here": return {'error': "PlantNet API Key not configured."}
    files, params = {'images': ('image.jpg', image_bytes)}, {'api-key': PLANTNET_API_KEY, 'include-related-images': 'false'}
    try:
        r = requests.post(PLANTNET_URL, files=files, params=params, timeout=20); r.raise_for_status(); data = r.json()
        if "results" in data and data["results"]:
            b = data["results"][0]; s, c, conf = b["species"].get("scientificNameWithoutAuthor","N/A"), (b["species"].get("commonNames")or["N/A"])[0], round(b.get("score",0)*100,1)
            return {'scientific_name': s, 'common_name': c, 'confidence': conf}
        return {'error': "No plant matches found."}
    except requests.exceptions.Timeout: st.error("PlantNet API timeout."); return {'error': "API timeout"}
    except requests.exceptions.RequestException as e: st.error(f"Network error (PlantNet): {str(e).split(':')[0]}"); return {'error': "Network error."}
    except Exception as e: st.error(f"Error during ID: {e}"); return {'error': f"Unexpected Error."}

def create_personality_profile(care_info):
    d_pers = {"title":"Standard Plant", "traits":"observant", "prompt":"Respond factually."}
    if not care_info or not isinstance(care_info, dict): return d_pers
    p_data = care_info.get("Personality")
    if not p_data or not isinstance(p_data, dict): return {"title":f"The {care_info.get('Plant Name','Plant')}", "traits":"resilient", "prompt":"Respond simply."}
    t, tr, pr = p_data.get("Title",care_info.get("Plant Name","Plant")), p_data.get("Traits",["observant"]), p_data.get("Prompt","In character.")
    tr_l = tr if isinstance(tr, list) else ["observant"]; v_tr = [str(x) for x in tr_l if x]; f_tr = ", ".join(v_tr) if v_tr else "observant"
    return {"title":t, "traits":f_tr, "prompt":pr}

def send_message(messages):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here": return "Gemini API Key not configured."
    payload, headers = {"contents":messages, "generationConfig":{"maxOutputTokens":150}}, {"Content-Type":"application/json"}
    try:
        r = requests.post(GEMINI_API_URL,json=payload,headers=headers,timeout=30); r.raise_for_status(); data = r.json()
        cand = data.get('candidates')
        if cand and isinstance(cand,list) and cand[0].get('content',{}).get('parts'): return cand[0]['content']['parts'][0]['text']
        st.warning("Unexpected Gemini response format."); print("WARN: Gemini Structure:",data); return "Unexpected response."
    except requests.exceptions.Timeout: st.error("Gemini API timeout."); return "Request timed out."
    except requests.exceptions.RequestException as e:
        err_msg = f"Gemini API Error: {e}"; rd = ""
        if hasattr(e,'response') and e.response is not None:
            try: rd = e.response.json().get('error',{}).get('message',e.response.text)
            except: rd = e.response.text
        st.error(f"{err_msg.split(':')[0]}: {rd}" if rd else err_msg.split(':')[0]); return "Language model communication error."
    except Exception as e: st.error(f"Gemini Error: {e}"); return "Unexpected chat error."

def chat_with_plant(care_info, conversation_history, id_result=None):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here": return "Chat disabled: Gemini API Key missing."
    p_name, sys_p_parts = "this plant", ["CONTEXT: Short chatbot response (1-3 sentences).","TASK: Act *exclusively* as the plant. Stay in character. NEVER mention being AI/model."]
    resp_rules = ["RESPONSE RULES:","1. First person.","2. Embody personality.","3. Concise (1-3 sentences).","4. **Never break character/reveal AI.**"]
    if care_info and isinstance(care_info, dict):
        pers = create_personality_profile(care_info); p_name = care_info.get('Plant Name','a plant')
        sys_p_parts.extend([f"YOUR PERSONALITY: '{pers['title']}' (traits: {pers['traits']}). Philosophy: {pers['prompt']}",
                            "CARE NEEDS (Refer to these *only*):",f"- Light: {care_info.get('Light Requirements','N/A')}",
                            f"- Water: {care_info.get('Watering','N/A')}",f"- Humidity: {care_info.get('Humidity Preferences','N/A')}",
                            f"- Temp: {care_info.get('Temperature Range','N/A')}",f"- Feed: {care_info.get('Feeding Schedule','N/A')}",
                            f"- Toxicity: {care_info.get('Toxicity','N/A')}","Base answers *only* on these needs."])
    elif id_result and isinstance(id_result, dict) and 'error' not in id_result:
        p_name = id_result.get('common_name',id_result.get('scientific_name','this plant'))
        if p_name=='N/A' or not p_name.strip(): p_name='this plant'
        sys_p_parts.extend([f"You are '{p_name}'. No specific profile.","Answer generally based on '{p_name}' knowledge.",
                            "If asked specifics, say you lack exact details but can offer general advice."])
    else: return "Not enough info to chat."
    sys_prompt = "\n".join(sys_p_parts + resp_rules)
    msgs = [{"role":"user","parts":[{"text":sys_prompt}]},{"role":"model","parts":[{"text":f"Understood. I am {p_name}. Ask away."}]}]
    v_hist = [m for m in conversation_history if isinstance(m,dict) and "role" in m and "content" in m]
    for entry in v_hist: msgs.append({"role":"model" if entry["role"] in ["assistant","model"] else "user","parts":[{"text":str(entry["content"])}]})
    return send_message(msgs)

# =======================================================
# --- Helper Functions ---
# =======================================================
@st.cache_data(show_spinner=False)
def load_plant_care_data(): return SAMPLE_PLANT_CARE_DATA

def find_care_instructions(plant_name_id, care_data, match_threshold=75):
    if not care_data: return None
    s_name, c_name = (None,None)
    if isinstance(plant_name_id,dict): s_name,c_name=plant_name_id.get('scientific_name'),plant_name_id.get('common_name')
    elif isinstance(plant_name_id,str): s_name=plant_name_id
    s_sci,s_com = (s_name.lower().strip() if s_name and isinstance(s_name,str) else None), (c_name.lower().strip() if c_name and isinstance(c_name,str) else None)
    for p in care_data:
        db_s,db_pn_s = p.get('Scientific Name','').lower().strip(), p.get('Plant Name','').lower().strip()
        if s_sci and (s_sci==db_s or s_sci==db_pn_s): return p
        if s_com:
            if s_com==p.get('Plant Name','').lower().strip(): return p
            db_c_raw = p.get('Common Names',[]); db_c_l = [db_c_raw] if isinstance(db_c_raw,str) else (db_c_raw if isinstance(db_c_raw,list) else [])
            if s_com in [c.lower().strip() for c in db_c_l if isinstance(c,str)]: return p
    db_map={}; 
    for p in care_data:
        k_l = [p.get('Scientific Name',''), p.get('Plant Name','')] + ([p.get('Common Names','')] if isinstance(p.get('Common Names',''),str) else p.get('Common Names',[]))
        for k_r in k_l: 
            k=(k_r.lower().strip() if isinstance(k_r,str) else None)
            if k and k not in db_map: db_map[k]=p
    db_n=list(db_map.keys()); 
    if not db_n: return None
    bm_res,hs=(None,0)
    for st_t in [s_sci,s_com]:
        if st_t: 
            res=process.extractOne(st_t,db_n)
            if res and res[1]>=match_threshold and res[1]>hs: hs,bm_res=res[1],db_map.get(res[0])
    return bm_res

def display_identification_result(result):
    st.subheader("🔍 Identification Results")
    if not result: st.error("No ID result."); return
    if 'error' in result: st.error(f"ID failed: {result.get('error','Unknown')}"); return
    conf=result.get('confidence',0); color="#28a745" if conf>75 else ("#ffc107" if conf>50 else "#dc3545")
    st.markdown(f"""- **Scientific Name:** `{result.get('scientific_name','N/A')}`\n- **Common Name:** `{result.get('common_name','N/A')}`\n- **Confidence:** <strong style='color:{color};'>{conf:.1f}%</strong>""", unsafe_allow_html=True)

def display_care_instructions(care_info, header_level=3):
    if not care_info or not isinstance(care_info,dict): st.warning("Care info missing."); return
    name=care_info.get('Plant Name','This Plant'); st.markdown(f"<h{header_level}>🌱 {name} Care Guide</h{header_level}>",unsafe_allow_html=True)
    with st.expander("📋 Care Summary",expanded=True):
        c1,c2=st.columns(2); dets=[("☀️ Light",'Light Requirements'),("💧 Water",'Watering'),("🌡️ Temp",'Temperature Range'),("💦 Humidity",'Humidity Preferences'),("🍃 Feeding",'Feeding Schedule'),("⚠️ Toxicity",'Toxicity')]
        for i,(l,k) in enumerate(dets): (c1 if i<len(dets)/2 else c2).markdown(f"**{l}**"); (c1 if i<len(dets)/2 else c2).caption(f"{care_info.get(k,'N/A')}")
    add_c=care_info.get('Additional Care'); 
    if add_c and isinstance(add_c,str) and add_c.strip(): 
        with st.expander("✨ Pro Tips"): st.markdown(add_c)

def find_similar_plant_matches(id_result, plant_care_data, limit=3, score_threshold=60):
    if not id_result or 'error' in id_result or not plant_care_data: return []
    db_map={}; 
    for p in plant_care_data:
        k_l = [p.get('Scientific Name',''), p.get('Plant Name','')] + ([p.get('Common Names','')] if isinstance(p.get('Common Names',''),str) else p.get('Common Names',[]))
        for k_r in k_l: 
            k=(k_r.lower().strip() if isinstance(k_r,str) else None)
            if k and k not in db_map: db_map[k]=p
    db_n=list(db_map.keys()); 
    if not db_n: return []
    s_terms=[id_result.get('scientific_name','').lower().strip(), id_result.get('common_name','').lower().strip()]; matches={}
    for t in s_terms:
        if t:
            f_res=process.extract(t,db_n,limit=limit*2)
            for n,s in f_res:
                if s>=score_threshold: matches[n]=max(matches.get(n,0),s)
    srt_m=sorted(matches.items(),key=lambda i:i[1],reverse=True); fin_sug,seen_p_act=([],set())
    for n,s in srt_m:
        p_i=db_map.get(n)
        if p_i:
            p_id_d=p_i.get('Scientific Name','')+p_i.get('Plant Name','')
            if p_id_d not in seen_p_act: fin_sug.append(p_i); seen_p_act.add(p_id_d)
            if len(fin_sug)>=limit: break
    return fin_sug

def display_suggestion_buttons(suggestions):
     if not suggestions: return
     st.info("🌿 Perhaps one of these is a closer match from our database?")
     cols=st.columns(len(suggestions))
     for i,p_i in enumerate(suggestions):
         p_n=p_i.get('Plant Name',p_i.get('Scientific Name',f'Suggest {i+1}')); s_p_n="".join(c if c.isalnum() else "_" for c in p_n)
         t_tip=f"Select {p_n}"+(f" (Sci: {p_i.get('Scientific Name')})" if p_i.get('Scientific Name') and p_i.get('Scientific Name')!=p_n else "")
         if cols[i].button(p_n,key=f"suggest_{s_p_n}_{i}",help=t_tip,use_container_width=True):
            st.session_state.plant_care_info=p_i; st.session_state.plant_id_result={'scientific_name':p_i.get('Scientific Name','N/A'),'common_name':p_n,'confidence':100.0}
            st.session_state.plant_id_result_for_care_check=st.session_state.plant_id_result; st.session_state.suggestions=None; st.session_state.chat_history=[]
            st.session_state.current_chatbot_plant_name=None; st.session_state.suggestion_just_selected=True; st.rerun()

def display_chat_interface(current_plant_care_info=None, plant_id_result=None):
    c_disp_n="this plant"; can_c=False
    if current_plant_care_info and isinstance(current_plant_care_info,dict): c_disp_n=current_plant_care_info.get("Plant Name","this plant"); can_c=True
    elif plant_id_result and isinstance(plant_id_result,dict) and 'error' not in plant_id_result:
        n_id=plant_id_result.get('common_name',plant_id_result.get('scientific_name'))
        if n_id and n_id!='N/A' and n_id.strip(): c_disp_n=n_id
        can_c=True
    if can_c and (not GEMINI_API_KEY or GEMINI_API_KEY=="your_gemini_api_key_here"): st.warning("Chat requires Gemini API key."); return
    if not can_c: st.warning("Cannot init chat: no valid plant ID."); return
    st.subheader(f"💬 Chat with {c_disp_n}")
    # Minified CSS for chat bubbles
    st.markdown("""<style>.message-container{padding:1px 5px}.user-message{background:#0b81fe;color:white;border-radius:18px 18px 0 18px;padding:8px 14px;margin:3px 0 3px auto;width:fit-content;max-width:80%;word-wrap:break-word;box-shadow:0 1px 2px rgba(0,0,0,.1);animation:fadeIn .3s ease-out}.bot-message{background:#e5e5ea;color:#000;border-radius:18px 18px 18px 0;padding:8px 14px;margin:3px auto 3px 0;width:fit-content;max-width:80%;word-wrap:break-word;box-shadow:0 1px 2px rgba(0,0,0,.05);animation:fadeIn .3s ease-out}.message-meta{font-size:.7rem;color:#777;margin-top:3px}.bot-message .message-meta{text-align:left;color:#555}.user-message .message-meta{text-align:right}@keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}.stChatInputContainer{position:sticky;bottom:0;background:white;padding-top:10px;z-index:99}</style>""", unsafe_allow_html=True)

    if "chat_history" not in st.session_state or st.session_state.get("current_chatbot_plant_name")!=c_disp_n:
        if st.session_state.get("viewing_saved_details"):
             s_p_d=st.session_state.saved_photos.get(st.session_state.viewing_saved_details)
             st.session_state.chat_history=s_p_d.get('chat_log',[]) if s_p_d else []
        else: st.session_state.chat_history=[]
        st.session_state.current_chatbot_plant_name=c_disp_n
    
    chat_cont=st.container(height=400)
    with chat_cont:
        for msg in st.session_state.get("chat_history",[]):
            r,c,t = msg.get("role"),msg.get("content",""),msg.get("time","")
            if r=="user": st.markdown(f'<div class="message-container"><div class="user-message">{c}<div class="message-meta">You • {t}</div></div></div>',unsafe_allow_html=True)
            elif r in ["assistant","model"]: st.markdown(f'<div class="message-container"><div class="bot-message">🌿 {c}<div class="message-meta">{c_disp_n} • {t}</div></div></div>',unsafe_allow_html=True)
    
    # Robust key for chat_input
    s_disp_n="".join(x if x.isalnum() else "_" for x in c_disp_n)
    chat_key = f"chat_input_{s_disp_n}_{st.session_state.get('current_nav_choice', 'default_view')}"
    if st.session_state.get('viewing_saved_details'):
        chat_key += f"_{st.session_state.viewing_saved_details}"
    elif st.session_state.get('uploaded_file_bytes'): # Add a part of hash if an image is uploaded for ID page
        chat_key += f"_id_{str(hash(st.session_state.uploaded_file_bytes))[:8]}"


    if prompt := st.chat_input(f"Ask {c_disp_n}...", key=chat_key):
        ts=datetime.now(EASTERN_TZ).strftime("%H:%M")
        st.session_state.chat_history.append({"role":"user","content":prompt,"time":ts})
        st.rerun()
    
    if st.session_state.get("chat_history") and st.session_state.chat_history[-1].get("role")=="user":
        with st.spinner(f"{c_disp_n} is thinking..."):
            bot_resp=chat_with_plant(current_plant_care_info,st.session_state.chat_history,plant_id_result)
        ts=datetime.now(EASTERN_TZ).strftime("%H:%M")
        st.session_state.chat_history.append({"role":"assistant","content":bot_resp,"time":ts})
        if st.session_state.get("viewing_saved_details") and st.session_state.viewing_saved_details in st.session_state.saved_photos:
            st.session_state.saved_photos[st.session_state.viewing_saved_details]['chat_log']=st.session_state.chat_history
        st.rerun()

# --- Main App Logic ---
def main():
    defaults={"plant_id_result":None,"plant_care_info":None,"chat_history":[],"current_chatbot_plant_name":None,"suggestions":None,"uploaded_file_bytes":None,"uploaded_file_type":None,"saving_mode":False,"last_view":"🏠 Home","viewing_saved_details":None,"plant_id_result_for_care_check":None,"suggestion_just_selected":False,"viewing_plant_stats":None,"viewing_home_page":True,"saved_photos":{},"current_nav_choice":"🏠 Home"}
    for k,v in defaults.items(): 
        if k not in st.session_state: st.session_state[k]=v
    st.markdown(get_ring_html_css(), unsafe_allow_html=True)

    st.sidebar.title("📚 Plant Buddy")
    nav_opts=["🏠 Home","🆔 Identify New Plant","🪴 My Saved Plants","📊 Plant Stats"]
    try: cur_nav_idx=nav_opts.index(st.session_state.current_nav_choice)
    except ValueError: cur_nav_idx=0
    nav_c=st.sidebar.radio("Navigation",nav_opts,key="main_nav_radio",index=cur_nav_idx,label_visibility="collapsed")
    if nav_c!=st.session_state.current_nav_choice:
        st.session_state.current_nav_choice=nav_c
        if nav_c=="🏠 Home": st.session_state.viewing_home_page,st.session_state.viewing_saved_details,st.session_state.viewing_plant_stats=True,None,None
        elif nav_c=="🆔 Identify New Plant": st.session_state.viewing_home_page,st.session_state.viewing_saved_details,st.session_state.viewing_plant_stats=False,None,None
        elif nav_c=="🪴 My Saved Plants": st.session_state.viewing_home_page,st.session_state.viewing_plant_stats=False,None
        elif nav_c=="📊 Plant Stats": st.session_state.viewing_home_page=False
        st.rerun()
    st.sidebar.divider(); st.sidebar.caption("Powered by PlantNet & Gemini")

    s_p_nicks=list(st.session_state.saved_photos.keys())
    if s_p_nicks:
        st.sidebar.subheader("Saved Plants")
        v_opts=["-- Select to View --"]+s_p_nicks; cur_sel_sb=st.session_state.get("viewing_saved_details")
        sb_idx=v_opts.index(cur_sel_sb) if cur_sel_sb and cur_sel_sb in v_opts else 0
        sel_s_p_sb=st.sidebar.selectbox("View Saved Plant:",v_opts,key="saved_view_selector_sb",index=sb_idx,label_visibility="collapsed")
        if sel_s_p_sb!="-- Select to View --":
            if st.session_state.get("viewing_saved_details")!=sel_s_p_sb:
                st.session_state.viewing_saved_details=sel_s_p_sb; st.session_state.current_nav_choice="🪴 My Saved Plants"; st.session_state.viewing_plant_stats=None; st.rerun()
        elif st.session_state.get("viewing_saved_details") is not None and sel_s_p_sb=="-- Select to View --":
            st.session_state.viewing_saved_details=None; st.session_state.current_nav_choice="🪴 My Saved Plants"; st.rerun()

    p_c_data=load_plant_care_data(); api_ok=not (not PLANTNET_API_KEY or PLANTNET_API_KEY=="your_plantnet_api_key_here")

    # ====================================
    # ===== Home Page View (MODIFIED) =====
    # ====================================
    if st.session_state.current_nav_choice=="🏠 Home":
        st.markdown('<div class="home-page-background">', unsafe_allow_html=True) # Start home page background div
        st.header("🌿 Welcome to Plant Buddy!")
        wel_p="You are Plant Buddy... Mention ID, care, chat, stats." # Abridged for brevity
        if "welcome_response" not in st.session_state:
            with st.spinner("Loading welcome..."):
                 msgs=[{"role":"user","parts":[{"text":"System: Be Plant Buddy..."}]},{"role":"model","parts":[{"text":"Okay!"}]},{"role":"user","parts":[{"text":wel_p}]}]
                 st.session_state.welcome_response=send_message(msgs) if GEMINI_API_KEY and GEMINI_API_KEY!="your_gemini_api_key_here" else "Welcome to Plant Buddy! ID, care, chat & stats. Happy gardening!"
        st.markdown(f"""<div style="background-color:#ffffff; padding:20px; border-radius:10px; border-left:5px solid #4CAF50; margin-bottom:20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h3 style="color:#2E7D32;">🌱 Hello Plant Lover!</h3>
                        <p style="font-size:1.1em; color:#333333;">{st.session_state.welcome_response}</p></div>""", unsafe_allow_html=True)
        st.subheader("🔍 What You Can Do")
        hc1,hc2=st.columns(2)
        with hc1: st.markdown("#### 📸 Identify Plants"); st.caption("Upload photo to ID."); 
        if hc1.button("Identify My Plant!",use_container_width=True,type="primary"): st.session_state.current_nav_choice="🆔 Identify New Plant"; st.rerun()
        with hc2: st.markdown("#### 💚 My Plant Collection"); st.caption("View saved plants, guides, stats."); 
        if hc2.button("Go to My Plants",use_container_width=True): st.session_state.current_nav_choice="🪴 My Saved Plants"; st.rerun()
        if st.session_state.saved_photos:
            st.divider(); st.markdown("<div style='text-align:center;'>",unsafe_allow_html=True); st.subheader("🪴 Your Recent Plants")
            rec_p=list(st.session_state.saved_photos.items())[-3:]; rec_p.reverse()
            if rec_p:
                st.markdown("<div style='display:flex;justify-content:center;gap:15px;flex-wrap:wrap;'>",unsafe_allow_html=True)
                n_d_p=len(rec_p); cols=st.columns(n_d_p)
                for i,(nick,p_d) in enumerate(rec_p):
                    with cols[i]:
                        with st.container(border=True,height=280): # Fixed height for card
                            display_image_with_max_height(p_d.get("image"),nick if p_d.get("image") else "",180) # Max height for image in card
                            if not p_d.get("image"): st.markdown(f"<div style='text-align:center;font-weight:bold;margin-top:10px;'>{nick}</div>", unsafe_allow_html=True) # Center nickname if no image
                            id_r=p_d.get("id_result",{}); com_n=id_r.get('common_name','N/A')
                            if com_n and com_n!='N/A' and com_n.lower()!=nick.lower(): st.caption(f"{com_n}")
                            if st.button("View Details",key=f"home_view_{nick}",use_container_width=True): st.session_state.viewing_saved_details=nick; st.session_state.current_nav_choice="🪴 My Saved Plants"; st.rerun()
                st.markdown("</div>",unsafe_allow_html=True)
            else: st.caption("No plants saved yet.")
            st.markdown("</div>",unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True) # End home page background div

    elif st.session_state.current_nav_choice == "🆔 Identify New Plant":
        st.header("🔎 Identify a New Plant")
        if st.session_state.last_view != "🆔 Identify New Plant" and not st.session_state.get("uploaded_file_bytes"):
            keys_to_clear_for_new_id = ["plant_id_result", "plant_care_info", "chat_history", "current_chatbot_plant_name", "suggestions", "saving_mode", "plant_id_result_for_care_check", "suggestion_just_selected"]
            for key in keys_to_clear_for_new_id: st.session_state[key] = None if key not in ["chat_history"] else []
        st.session_state.last_view = "🆔 Identify New Plant"

        uploaded_file = st.file_uploader(
            "Upload a clear photo of your plant:", type=["jpg", "jpeg", "png"],
            key="plant_uploader_main", help="Upload an image file (JPG, PNG).",
            on_change=lambda: st.session_state.update({
                 "plant_id_result": None, "plant_care_info": None, "chat_history": [],
                 "current_chatbot_plant_name": None, "suggestions": None,
                 "uploaded_file_bytes": None, "uploaded_file_type": None,
                 "saving_mode": False, "plant_id_result_for_care_check": None,
                 "suggestion_just_selected": False
            })
        )

        if uploaded_file is not None and st.session_state.uploaded_file_bytes is None: 
            st.session_state.uploaded_file_bytes = uploaded_file.getvalue()
            st.session_state.uploaded_file_type = uploaded_file.type
            st.rerun() 

        if st.session_state.uploaded_file_bytes is not None:
            display_image_with_max_height(st.session_state.uploaded_file_bytes, "Your Uploaded Plant", 400)
            st.divider()

            if st.session_state.plant_id_result is None: 
                with st.spinner("Identifying plant..."):
                    if not api_ok: 
                        import time; time.sleep(1)
                        random_plant = random.choice(p_c_data)
                        st.session_state.plant_id_result = {'scientific_name': random_plant.get('Scientific Name', 'Demo SciName'), 'common_name': random_plant.get('Plant Name', 'Demo Plant'), 'confidence': random.uniform(70,95)}
                    else:
                        st.session_state.plant_id_result = identify_plant(st.session_state.uploaded_file_bytes)
                    st.session_state.plant_id_result_for_care_check = None 
                    st.session_state.suggestion_just_selected = False
                st.rerun()

            elif st.session_state.plant_id_result is not None: 
                current_id_result = st.session_state.plant_id_result

                if st.session_state.saving_mode:
                    st.header("💾 Save This Plant Profile")
                    if st.session_state.uploaded_file_bytes: st.image(Image.open(BytesIO(st.session_state.uploaded_file_bytes)), width=150)
                    st.markdown(f"**Identified as:** {current_id_result.get('common_name','N/A')} (`{current_id_result.get('scientific_name','N/A')}`)")
                    with st.form("save_form_main"):
                        save_nickname = st.text_input("Enter a nickname:", key="save_nickname_input_main")
                        submitted = st.form_submit_button("✅ Confirm Save")
                        if submitted:
                            if not save_nickname: st.warning("Nickname required.")
                            elif save_nickname in st.session_state.saved_photos: st.warning(f"'{save_nickname}' already exists.")
                            else:
                                encoded_img = base64.b64encode(st.session_state.uploaded_file_bytes).decode()
                                st.session_state.saved_photos[save_nickname] = {
                                    "nickname": save_nickname, "image": f"data:{st.session_state.uploaded_file_type};base64,{encoded_img}",
                                    "id_result": current_id_result, "care_info": st.session_state.plant_care_info,
                                    "chat_log": st.session_state.get("chat_history", []),
                                    "moisture_level": random.randint(30,90) 
                                }
                                st.success(f"Saved '{save_nickname}'!"); st.balloons()
                                for k in ["plant_id_result", "plant_care_info", "current_chatbot_plant_name", "suggestions", "uploaded_file_bytes", "uploaded_file_type", "chat_history", "saving_mode", "plant_id_result_for_care_check", "suggestion_just_selected"]:
                                    st.session_state[k] = None if k not in ["chat_history"] else []
                                st.session_state.current_nav_choice = "🪴 My Saved Plants" 
                                st.session_state.viewing_saved_details = save_nickname
                                st.rerun()
                    if st.button("❌ Cancel Save", key="cancel_save_main"): st.session_state.saving_mode = False; st.rerun()
                else: 
                    display_identification_result(current_id_result)
                    if 'error' not in current_id_result:
                        care_info_current = st.session_state.get('plant_care_info')
                        id_for_check = st.session_state.get('plant_id_result_for_care_check')
                        
                        if not st.session_state.suggestion_just_selected and current_id_result != id_for_check:
                            found_care = find_care_instructions(current_id_result, p_c_data)
                            st.session_state.plant_care_info = found_care
                            st.session_state.plant_id_result_for_care_check = current_id_result
                            st.session_state.suggestions = None 
                            st.session_state.chat_history = []  
                            st.session_state.current_chatbot_plant_name = None
                            st.rerun() 

                        if st.session_state.suggestion_just_selected:
                            st.session_state.suggestion_just_selected = False 

                        care_to_display = st.session_state.get('plant_care_info') 
                        
                        if care_to_display:
                            display_care_instructions(care_to_display)
                            st.button("💾 Save Plant Profile", key="save_id_care_btn", on_click=lambda: st.session_state.update({"saving_mode": True}))
                            st.divider()
                            display_chat_interface(current_plant_care_info=care_to_display, plant_id_result=current_id_result)
                        else: 
                            st.warning("Could not find specific care instructions for this exact plant in our database.")
                            if st.session_state.suggestions is None: 
                                st.session_state.suggestions = find_similar_plant_matches(current_id_result, p_c_data)
                                if st.session_state.suggestions is not None: 
                                     st.rerun() 

                            display_suggestion_buttons(st.session_state.suggestions)
                            st.divider()
                            st.button("💾 Save Identification Only", key="save_id_only_btn", on_click=lambda: st.session_state.update({"saving_mode": True}))
                            st.divider()
                            st.info("You can still chat with the plant based on its general identification.")
                            display_chat_interface(current_plant_care_info=None, plant_id_result=current_id_result)
    
    elif st.session_state.current_nav_choice == "🪴 My Saved Plants":
        st.header("🪴 My Saved Plant Profiles")
        st.session_state.last_view = "🪴 My Saved Plants"
        nickname_to_view = st.session_state.get("viewing_saved_details")

        if not st.session_state.saved_photos:
            st.info("You haven't saved any plants yet. Go to 'Identify New Plant' to add some!")
        elif nickname_to_view and nickname_to_view in st.session_state.saved_photos:
            entry = st.session_state.saved_photos[nickname_to_view]
            if st.button("← Back to All Plants", key="back_to_all_saved"):
                st.session_state.viewing_saved_details = None; st.rerun()
            
            st.subheader(f"Showing Details for: '{nickname_to_view}'")
            if entry.get("image"): display_image_with_max_height(entry["image"], nickname_to_view, 400)
            st.divider()

            saved_id_result = entry.get("id_result")
            if saved_id_result: display_identification_result(saved_id_result)
            if st.session_state.get("plant_id_result") != saved_id_result: st.session_state.plant_id_result = saved_id_result
            st.divider()
            
            saved_care_info = entry.get("care_info")
            if st.session_state.get("plant_care_info") != saved_care_info: st.session_state.plant_care_info = saved_care_info

            col_stats_btn, _ = st.columns([0.7, 0.3]) # Placeholder for delete if needed or adjust ratio
            with col_stats_btn:
                if st.button(f"📊 View Stats for {nickname_to_view}", key=f"stats_btn_{nickname_to_view}", use_container_width=True):
                    st.session_state.viewing_plant_stats = nickname_to_view
                    st.session_state.current_nav_choice = "📊 Plant Stats"
                    st.rerun()
            
            if saved_care_info:
                display_care_instructions(saved_care_info)
                st.divider()
                display_chat_interface(current_plant_care_info=saved_care_info, plant_id_result=saved_id_result)
            else:
                st.info("No specific care instructions were saved for this plant.")
                st.divider()
                if saved_id_result:
                    st.info("You can chat with the plant based on its saved identification.")
                    display_chat_interface(current_plant_care_info=None, plant_id_result=saved_id_result)
            
            st.divider()
            # Improved Delete Confirmation
            delete_placeholder = st.empty() # Placeholder for delete button/confirmation
            confirm_delete_key = f"confirm_delete_{nickname_to_view}"

            if st.session_state.get(confirm_delete_key, False):
                with delete_placeholder.container():
                    st.warning(f"Are you sure you want to delete '{nickname_to_view}'? This cannot be undone.")
                    c1_del, c2_del, _ = st.columns([1,1,2]) # Adjust layout for buttons
                    if c1_del.button("Yes, Delete Permanently", key=f"yes_del_perm_{nickname_to_view}", type="primary", use_container_width=True):
                        del st.session_state.saved_photos[nickname_to_view]
                        st.session_state.viewing_saved_details = None
                        for k in ["plant_id_result", "plant_care_info", "current_chatbot_plant_name", "suggestions", "chat_history", "viewing_plant_stats"]:
                            st.session_state[k] = None if k not in ["chat_history"] else []
                        st.session_state[confirm_delete_key] = False 
                        st.success(f"Deleted '{nickname_to_view}'.")
                        st.rerun()
                    if c2_del.button("Cancel Deletion", key=f"no_del_cancel_{nickname_to_view}", use_container_width=True):
                        st.session_state[confirm_delete_key] = False
                        st.rerun()
            else:
                if delete_placeholder.button(f"🗑️ Delete '{nickname_to_view}' Profile", key=f"del_btn_main_{nickname_to_view}", type="secondary"):
                    st.session_state[confirm_delete_key] = True
                    st.rerun()

        else: 
            st.info("Select a plant from the 'View Saved Plant' dropdown in the sidebar to see its details, or browse below.")
            st.markdown("---")
            num_cols_grid = 3
            grid_cols = st.columns(num_cols_grid)
            for i, (nick, data) in enumerate(st.session_state.saved_photos.items()):
                with grid_cols[i % num_cols_grid]:
                    with st.container(border=True, height=300): 
                        display_image_with_max_height(data.get("image"), nick if data.get("image") else "", 180)
                        if not data.get("image"): st.markdown(f"<div style='text-align:center;font-weight:bold;margin-top:10px;'>{nick}</div>", unsafe_allow_html=True)
                        
                        id_res_grid = data.get("id_result", {}); com_n_grid = id_res_grid.get('common_name', 'N/A')
                        if com_n_grid and com_n_grid != 'N/A' and com_n_grid.lower() != nick.lower(): st.caption(f"{com_n_grid}")
                        
                        btn_c1, btn_c2 = st.columns(2)
                        with btn_c1:
                            if st.button("Details", key=f"grid_detail_{nick}", use_container_width=True):
                                st.session_state.viewing_saved_details = nick; st.rerun()
                        with btn_c2:
                            if st.button("📊 Stats", key=f"grid_stats_{nick}", use_container_width=True):
                                st.session_state.viewing_plant_stats = nick
                                st.session_state.current_nav_choice = "📊 Plant Stats"; st.rerun()

    elif st.session_state.current_nav_choice == "📊 Plant Stats":
        st.session_state.last_view = "📊 Plant Stats"
        plant_nickname_for_stats = st.session_state.get("viewing_plant_stats")

        if not plant_nickname_for_stats or plant_nickname_for_stats not in st.session_state.saved_photos:
            st.warning("No plant selected for stats view. Please select a plant from 'My Saved Plants'.")
            if st.button("← Back to Saved Plants", key="stats_back_to_saved_no_plant_mod"):
                st.session_state.current_nav_choice = "🪴 My Saved Plants"
                st.session_state.viewing_plant_stats = None; st.rerun()
        else:
            plant_data_stats = st.session_state.saved_photos[plant_nickname_for_stats]
            st.header(f"📊 Plant Stats: {plant_nickname_for_stats}")
            if st.button("← Back to Plant Details", key="stats_back_to_details_mod"):
                st.session_state.current_nav_choice = "🪴 My Saved Plants"
                st.session_state.viewing_saved_details = plant_nickname_for_stats
                st.session_state.viewing_plant_stats = None; st.rerun()
            st.divider()

            simulated_time_str = (datetime.now(EASTERN_TZ) - timedelta(minutes=random.randint(1,5))).strftime('%H:%M')
            moisture_level_perc = plant_data_stats.get("moisture_level", random.randint(30,70))
            moisture_desc_ring = f"Current soil moisture is {moisture_level_perc}%. Ideal levels vary."
            ring1_html = generate_ring_html("Moisture", str(moisture_level_perc), f"OF {MOISTURE_MAX_PERCENT}%", moisture_level_perc, MOISTURE_COLOR, MOISTURE_TRACK_COLOR, simulated_time_str, moisture_desc_ring, 0)

            care_info_stats = plant_data_stats.get("care_info", {})
            temp_range_str_stats = care_info_stats.get("Temperature Range", "65-85°F")
            min_f, max_f = parse_temp_range(temp_range_str_stats)
            current_temp_f_sim = TEMP_DISPLAY_MIN_F - 5
            if min_f is not None and max_f is not None:
                ideal_mid = (min_f + max_f) / 2
                current_temp_f_sim = random.uniform(ideal_mid - 7, ideal_mid + 7)
                current_temp_f_sim = round(max(TEMP_DISPLAY_MIN_F - 5, min(TEMP_DISPLAY_MAX_F + 5, current_temp_f_sim)))
            else: current_temp_f_sim = random.randint(68, 78)
            temp_progress = ((current_temp_f_sim - TEMP_DISPLAY_MIN_F) / (TEMP_DISPLAY_MAX_F - TEMP_DISPLAY_MIN_F)) * 100
            temp_desc_ring = f"Ambient temp {current_temp_f_sim}°F. Ideal: {temp_range_str_stats if temp_range_str_stats else 'N/A'}."
            ring2_html = generate_ring_html("Temperature", str(int(current_temp_f_sim)), "°F NOW", temp_progress, TEMPERATURE_COLOR, TEMPERATURE_TRACK_COLOR, simulated_time_str, temp_desc_ring, 1)

            minutes_ago_sim = random.randint(1, FRESHNESS_MAX_MINUTES_AGO - 10)
            freshness_progress_sim = max(0, (1 - (minutes_ago_sim / FRESHNESS_MAX_MINUTES_AGO))) * 100
            freshness_desc_ring = f"Stats checked {minutes_ago_sim} min(s) ago."
            ring3_html = generate_ring_html("Last Check", str(minutes_ago_sim), "MINS AGO", freshness_progress_sim, FRESHNESS_COLOR, FRESHNESS_TRACK_COLOR, simulated_time_str, freshness_desc_ring, 2)

            st.markdown(f'<div class="watch-face-grid">{ring1_html}{ring2_html}{ring3_html}</div>', unsafe_allow_html=True)
            st.divider()
            
            img_col, info_col = st.columns([0.4, 0.6]) 
            with img_col:
                if plant_data_stats.get("image"):
                    display_image_with_max_height(plant_data_stats["image"], max_height_px=250 )
            with info_col:
                st.subheader("Plant Identification") 
                id_res_stats = plant_data_stats.get("id_result", {})
                st.markdown(f"**Nickname:** {plant_nickname_for_stats}")
                st.markdown(f"**Scientific Name:** `{id_res_stats.get('scientific_name', 'N/A')}`")
                st.markdown(f"**Common Name:** `{id_res_stats.get('common_name', 'N/A')}`")
                st.caption(f"Full care guide available on '{plant_nickname_for_stats}'s main profile page.")

if __name__ == "__main__":
    if not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here":
        st.sidebar.warning("PlantNet API Key missing. Demo mode.", icon="🔑")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        st.sidebar.warning("Gemini API Key missing. Chat limited.", icon="🔑")
    main()
