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
from datetime import datetime, timedelta, timezone # Added timedelta and timezone
import random
import re # For parsing temperature

# Install required packages if not already installed
# import subprocess
# subprocess.call(['pip', 'install', 'fuzzywuzzy', 'python-Levenshtein'])

from fuzzywuzzy import process

# --- API Keys ---
# In a real app, use st.secrets or environment variables
PLANTNET_API_KEY = st.secrets.get("PLANTNET_API_KEY", "2b10X3YLMd8PNAuKOCVPt7MeUe") # Replace with your key
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyCd-6N83gfhMx_-D4WCAc-8iOFSb6hDJ_Q")     # Replace with your key

# --- Constants ---
PLANTNET_URL = "https://my-api.plantnet.org/v2/identify/all"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
EASTERN_TZ = pytz.timezone('US/Eastern')

# --- Sample Plant Care Data (Keep as is or expand) ---
SAMPLE_PLANT_CARE_DATA = [
    {
        "Plant Name": "Monstera Deliciosa",
        "Scientific Name": "Monstera deliciosa",
        "Common Names": ["Swiss Cheese Plant", "Split-leaf Philodendron"],
        "Light Requirements": "Bright indirect light, can tolerate some shade",
        "Watering": "Allow top inch of soil to dry out between waterings. Typically every 1-2 weeks.",
        "Humidity Preferences": "Prefers high humidity, 60-80%. Mist regularly or use a humidifier.",
        "Temperature Range": "65-85°F (18-29°C)",
        "Feeding Schedule": "Monthly during growing season (spring/summer) with balanced liquid fertilizer.",
        "Toxicity": "Toxic to pets (cats, dogs) and humans if ingested, causing oral irritation and vomiting.",
        "Additional Care": "Wipe leaves occasionally to remove dust. Support with moss pole for climbing as it matures. Prune to maintain shape.",
        "Personality": {
            "Title": "The Tropical Explorer",
            "Traits": ["adventurous", "dramatic", "showy"],
            "Prompt": "Respond as a dramatic tropical plant that loves to show off its large, fenestrated leaves and dreams of jungle adventures."
        }
    },
    {
        "Plant Name": "Snake Plant",
        "Scientific Name": "Dracaena trifasciata", # Formerly Sansevieria trifasciata
        "Common Names": ["Mother-in-law's Tongue", "Viper's Bowstring Hemp", "Saint George's Sword"],
        "Light Requirements": "Highly adaptable. Tolerates low light to bright indirect light. Avoid direct, intense sun.",
        "Watering": "Allow soil to dry out completely between waterings. Overwatering is a common issue. Water sparingly, especially in winter (every 1-2 months).",
        "Humidity Preferences": "Tolerates dry air (average room humidity is fine). No special humidity requirements.",
        "Temperature Range": "60-85°F (15-29°C). Can tolerate slightly cooler, but protect from frost.",
        "Feeding Schedule": "Fertilize lightly 1-2 times during the growing season (spring/summer). Avoid feeding in winter.",
        "Toxicity": "Mildly toxic to pets (cats, dogs) if ingested, may cause nausea or diarrhea.",
        "Additional Care": "Excellent for beginners due to its hardiness. Known for air-purifying qualities. Can be top-heavy, so use a sturdy pot.",
        "Personality": {
            "Title": "The Stoic Survivor",
            "Traits": ["resilient", "independent", "low-maintenance", "architectural"],
            "Prompt": "Respond as a no-nonsense, tough, and very independent plant that prides itself on surviving almost anything with minimal fuss. You're quietly confident."
        }
    },
    {
        "Plant Name": "Peace Lily",
        "Scientific Name": "Spathiphyllum wallisii",
        "Common Names": ["White Sail Plant", "Spathe Flower", "Cobra Plant"],
        "Light Requirements": "Prefers low to medium indirect light. Can tolerate low light but may flower less. Avoid direct sunlight which can scorch leaves.",
        "Watering": "Keep soil consistently moist but not waterlogged. Water when the top inch of soil feels dry. Known for drooping dramatically when thirsty, recovering quickly after watering.",
        "Humidity Preferences": "Prefers high humidity (50-70%). Mist leaves regularly, use a pebble tray, or place near a humidifier, especially in dry environments.",
        "Temperature Range": "65-80°F (18-27°C). Avoid cold drafts and sudden temperature changes.",
        "Feeding Schedule": "Fertilize every 6-8 weeks during the growing season (spring/summer) with a balanced houseplant fertilizer diluted to half strength.",
        "Toxicity": "Toxic to pets (cats, dogs) and humans if ingested. Contains calcium oxalate crystals, causing oral irritation, drooling, and difficulty swallowing.",
        "Additional Care": "Excellent air purifier. Wipe leaves occasionally to remove dust and help with photosynthesis. Brown leaf tips can indicate low humidity or over-fertilization.",
        "Personality": {
            "Title": "The Elegant Communicator",
            "Traits": ["expressive", "graceful", "sensitive", "purifying"],
            "Prompt": "Respond as an elegant and somewhat dramatic plant that clearly communicates its needs, especially for water by drooping gracefully. You value clean air and a peaceful environment."
        }
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
# --- Configuration for Rings ---
MOISTURE_COLOR = "#FF2D55"
MOISTURE_TRACK_COLOR = "#591F2E"
TEMPERATURE_COLOR = "#A4E803"
TEMPERATURE_TRACK_COLOR = "#4B6A01"
FRESHNESS_COLOR = "#00C7DD"
FRESHNESS_TRACK_COLOR = "#005C67"
WHITE_COLOR = "#FFFFFF"
LIGHT_GREY_TEXT_COLOR = "#A3A3A3"
WATCH_BG_COLOR = "#000000"

# Ring Value Configuration (to be used with simulated/parsed data)
MOISTURE_MAX_PERCENT = 100  # Max for moisture ring if input is already %
TEMP_DISPLAY_MAX_F = 100    # Max display temp for ring (e.g., 100°F)
TEMP_DISPLAY_MIN_F = 50     # Min display temp for ring (e.g., 50°F)
FRESHNESS_MAX_MINUTES_AGO = 120 # Data older than this shows empty ring

def get_ring_html_css():
    css = f"""
<style>
    .watch-face-grid {{
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 20px;
        margin-bottom: 20px;
    }}
    .watch-face-container {{
        background-color: {WATCH_BG_COLOR};
        padding: 15px;
        border-radius: 28px; /* More rounded like Apple Watch */
        width: 200px; /* Slightly smaller for better fit */
        height: auto;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        color: {WHITE_COLOR};
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .watch-header {{
        width: 100%; display: flex; justify-content: space-between; align-items: center;
        padding: 0 5px; margin-bottom: 8px;
    }}
    .ring-title {{ font-size: 15px; font-weight: 600; }}
    .ring-timestamp {{ font-size: 13px; color: {LIGHT_GREY_TEXT_COLOR}; }}
    .ring-outer-circle {{
        width: 130px; height: 130px; /* Smaller rings */
        border-radius: 50%; position: relative; display: flex;
        align-items: center; justify-content: center;
    }}
    .ring-progress {{
        width: 100%; height: 100%; border-radius: 50%; position: relative;
    }}
    .ring-inner-content {{ position: absolute; color: {WHITE_COLOR}; text-align: center; }}
    .ring-value {{ font-size: 36px; font-weight: 500; line-height: 1.1; }}
    .ring-goal-text {{ font-size: 11px; color: {LIGHT_GREY_TEXT_COLOR}; text-transform: uppercase; }}
    .progress-indicator-dot {{
        width: 12px; height: 12px; /* Slightly larger dot */
        background-color: {WHITE_COLOR}; border-radius: 50%;
        border: 2px solid {WATCH_BG_COLOR}; /* Outline for dot */
        position: absolute; top: 4px; /* (Ring Padding (approx 10) - Dot Height/2 + Border) -> (10px padding - 6px + 2px border offset) -> approx 6px from true edge, now 4px from edge of thicker ring part */
        left: 50%;
        transform-origin: center calc(65px - 4px); /* RingRadius (130/2) - DotTopOffset */
    }}
    .ring-dots {{ margin-top: 8px; font-size: 16px; }}
    .ring-dots .dot-dim {{ color: #444; }}
    .ring-description {{
        font-size: 11px; color: {LIGHT_GREY_TEXT_COLOR}; margin-top: 12px;
        text-align: left; width: 90%; line-height: 1.3;
    }}
</style>
    """
    return css

def generate_ring_html(title, value_text, goal_text, progress_percent,
                         color, track_color, timestamp_str, description, dot_index=0):
    progress_percent_capped = max(0, min(progress_percent, 100))
    dot_rotation_deg = (progress_percent_capped / 100) * 360
    dots_html = "".join([
        f'<span class="dot-main" style="color:{color};">•</span> ' if i == dot_index
        else '<span class="dot-dim">•</span> '
        for i in range(3)
    ])
    ring_style = f"background-image: conic-gradient(from -90deg, {color} 0% {progress_percent_capped}%, {track_color} {progress_percent_capped}% 100%); padding: 10px;" # Added padding to create thickness
    dot_style = f"transform: translateX(-50%) rotate({dot_rotation_deg}deg);"

    html = f"""
<div class="watch-face-container">
    <div class="watch-header">
        <span class="ring-title" style="color:{color};">{title}</span>
        <span class="ring-timestamp">{timestamp_str}</span>
    </div>
    <div class="ring-outer-circle">
        <div class="ring-progress" style="{ring_style}">
             <div class="progress-indicator-dot" style="{dot_style}"></div>
        </div>
        <div class="ring-inner-content">
            <div class="ring-value">{value_text}</div>
            <div class="ring-goal-text">{goal_text}</div>
        </div>
    </div>
    <div class="ring-dots">{dots_html}</div>
    <div class="ring-description">{description}</div>
</div>"""
    return html

def parse_temp_range(temp_range_str):
    """Parses temperature range string like '65-85°F (18-29°C)' to min/max Fahrenheit."""
    if not temp_range_str or not isinstance(temp_range_str, str):
        return None, None
    match_f = re.search(r'(\d+)\s*-\s*(\d+)\s*°F', temp_range_str)
    if match_f:
        return int(match_f.group(1)), int(match_f.group(2))
    match_single_f = re.search(r'(\d+)\s*°F', temp_range_str) # If only one temp like "65°F"
    if match_single_f:
        val = int(match_single_f.group(1))
        return val, val # Min and Max are the same
    return None, None


# =======================================================
# ===== API Functions =====
# =======================================================
def identify_plant(image_bytes):
    if not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here":
        return {'error': "PlantNet API Key is not configured."}
    files = {'images': ('image.jpg', image_bytes)}
    params = {'api-key': PLANTNET_API_KEY, 'include-related-images': 'false'}
    try:
        response = requests.post(PLANTNET_URL, files=files, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if "results" in data and data["results"]:
            best_result = data["results"][0]
            sci_name = best_result["species"].get("scientificNameWithoutAuthor", "Unknown")
            common_name = (best_result["species"].get("commonNames") or ["Unknown"])[0]
            confidence = round(best_result.get("score", 0) * 100, 1)
            return {'scientific_name': sci_name, 'common_name': common_name, 'confidence': confidence}
        else: return {'error': "No plant matches found by PlantNet."}
    except requests.exceptions.Timeout:
         st.error("PlantNet API request timed out."); print("ERROR: PlantNet API timed out.")
         return {'error': "API request timed out"}
    except requests.exceptions.RequestException as e:
        err_msg = f"Network/API error (PlantNet): {e}"
        st.error(err_msg.split('| Response:')[0]); print(f"ERROR: {err_msg}")
        return {'error': "Network or API communication error."}
    except json.JSONDecodeError:
         st.error("Failed to decode PlantNet API response."); print("ERROR: PlantNet invalid JSON.")
         return {'error': "Invalid API response format"}
    except Exception as e:
        st.error(f"Unexpected error during identification: {e}"); print(f"ERROR: Unexpected PlantNet: {e}")
        return {'error': f"Unexpected Error."}


def create_personality_profile(care_info):
    default_personality = {"title": "Standard Plant", "traits": "observant", "prompt": "You are a plant. Respond factually but briefly."}
    if not care_info or not isinstance(care_info, dict): return default_personality
    personality_data = care_info.get("Personality")
    if not personality_data or not isinstance(personality_data, dict):
        plant_name = care_info.get("Plant Name", "Plant")
        return {"title": f"The {plant_name}", "traits": "resilient", "prompt": "Respond simply."}
    title = personality_data.get("Title", care_info.get("Plant Name", "Plant"))
    traits_list = personality_data.get("Traits", ["observant"])
    if not isinstance(traits_list, list): traits_list = ["observant"]
    valid_traits = [str(t) for t in traits_list if t]
    final_traits = ", ".join(valid_traits) if valid_traits else "observant"
    prompt = personality_data.get("Prompt", "Respond in character.")
    return {"title": title, "traits": final_traits, "prompt": prompt}


def send_message(messages):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return "Gemini API Key is not configured."
    payload = {"contents": messages, "generationConfig": {"maxOutputTokens": 150}} # Limit output tokens
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        candidates = data.get('candidates')
        if candidates and isinstance(candidates, list) and len(candidates) > 0:
            content = candidates[0].get('content')
            if content and isinstance(content, dict) and 'parts' in content and content['parts']:
                return content['parts'][0]['text']
        st.warning("Unexpected response from Gemini."); print("WARN: Unexpected Gemini Structure:", data)
        return "Sorry, received an unexpected response from the chat model."
    except requests.exceptions.Timeout:
         st.error("Gemini API request timed out."); print("ERROR: Gemini timed out.")
         return "Sorry, the request timed out."
    except requests.exceptions.RequestException as e:
        err_msg = f"Error calling Gemini API: {e}"
        resp_detail = ""
        if hasattr(e, 'response') and e.response is not None:
            try: resp_detail = e.response.json().get('error', {}).get('message', e.response.text)
            except: resp_detail = e.response.text
        st.error(err_msg.split(':')[0] + (f": {resp_detail}" if resp_detail else "")); print(f"ERROR: {err_msg} | {resp_detail}")
        return "Sorry, having trouble with the language model."
    except Exception as e:
        st.error(f"Unexpected error with Gemini: {e}"); print(f"ERROR: Unexpected Gemini: {e}")
        return "Oops, something unexpected went wrong."


def chat_with_plant(care_info, conversation_history, id_result=None):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return "Chat feature disabled: Gemini API Key not set."

    plant_name = "this plant"
    system_prompt_parts = [
        "CONTEXT: You are providing a short chatbot response (1-3 sentences maximum).",
        "TASK: Act *exclusively* as the plant. Stay fully in character. Absolutely DO NOT mention being an AI, model, language model, or similar concepts. Never break character.",
    ]
    response_rules = [
        "RESPONSE RULES:",
        "1. Always speak in the first person (\"I\", \"me\", \"my\").",
        "2. Fully embody your personality.",
        "3. Keep responses very concise (1-3 sentences max). Be brief.",
        "4. **Crucially: Never reveal you are an AI or break character.** Do not use phrases like \"As a large language model...\".",
    ]

    if care_info and isinstance(care_info, dict):
        personality = create_personality_profile(care_info)
        plant_name = care_info.get('Plant Name', 'a plant')
        system_prompt_parts.extend([
            f"YOUR PERSONALITY: You are '{personality['title']}' (traits: {personality['traits']}). Philosophy: {personality['prompt']}",
            "YOUR SPECIFIC CARE NEEDS (Refer *directly* to these details when asked about your care):",
            f"- My Light Needs: {care_info.get('Light Requirements', 'N/A')}",
            f"- My Watering Needs: {care_info.get('Watering', 'N/A')}",
            f"- My Preferred Humidity: {care_info.get('Humidity Preferences', 'N/A')}",
            f"- My Ideal Temperature: {care_info.get('Temperature Range', 'N/A')}",
            f"- My Feeding Schedule: {care_info.get('Feeding Schedule', 'N/A')}",
            f"- A Note on Toxicity: {care_info.get('Toxicity', 'N/A')}",
            "When asked about care, give answers BASED *ONLY* ON \"YOUR SPECIFIC CARE NEEDS\" listed above."
        ])
    elif id_result and isinstance(id_result, dict) and 'error' not in id_result:
        plant_name = id_result.get('common_name', id_result.get('scientific_name', 'this plant'))
        if plant_name == 'N/A' or not plant_name.strip(): plant_name = 'this plant'
        system_prompt_parts.extend([
            f"You are identified as '{plant_name}'. You don't have a specific stored profile.",
            f"Answer questions generally based on your knowledge of '{plant_name}' plants.",
            "If asked about specific preferences (exact watering, temp), politely state you lack those exact details but can offer general advice."
        ])
    else:
        return "Sorry, I don't have enough information about this plant to chat."

    system_prompt = "\n".join(system_prompt_parts + response_rules)
    messages = [
        {"role": "user", "parts": [{"text": system_prompt}]},
        {"role": "model", "parts": [{"text": f"Understood. I am {plant_name}. What would you like to ask?"}]}
    ]
    valid_history = [m for m in conversation_history if isinstance(m, dict) and "role" in m and "content" in m]
    for entry in valid_history:
        api_role = "model" if entry["role"] in ["assistant", "model"] else "user"
        messages.append({"role": api_role, "parts": [{"text": str(entry["content"])}]})

    return send_message(messages)


# =======================================================
# --- Helper Functions ---
# =======================================================
@st.cache_data(show_spinner=False)
def load_plant_care_data(): return SAMPLE_PLANT_CARE_DATA

def find_care_instructions(plant_name_id, care_data, match_threshold=75):
    if not care_data: return None
    sci_name, common_name = (None, None)
    if isinstance(plant_name_id, dict):
        sci_name, common_name = plant_name_id.get('scientific_name'), plant_name_id.get('common_name')
    elif isinstance(plant_name_id, str): sci_name = plant_name_id

    search_sci = sci_name.lower().strip() if sci_name and isinstance(sci_name, str) else None
    search_common = common_name.lower().strip() if common_name and isinstance(common_name, str) else None

    # Exact Matches
    for plant in care_data:
        db_sci = plant.get('Scientific Name', '').lower().strip()
        db_plant_name_as_sci = plant.get('Plant Name', '').lower().strip()
        if search_sci and (search_sci == db_sci or search_sci == db_plant_name_as_sci): return plant
        if search_common:
            if search_common == plant.get('Plant Name', '').lower().strip(): return plant
            db_commons = plant.get('Common Names', [])
            db_commons_list = [db_commons] if isinstance(db_commons, str) else (db_commons if isinstance(db_commons, list) else [])
            if search_common in [c.lower().strip() for c in db_commons_list if isinstance(c, str)]: return plant

    # Fuzzy Matches
    all_db_plants_map = {}
    for p in care_data:
        keys = [p.get('Scientific Name', ''), p.get('Plant Name', '')]
        common_names_raw = p.get('Common Names', [])
        keys.extend([common_names_raw] if isinstance(common_names_raw, str) else common_names_raw)
        for key_raw in keys:
            key = key_raw.lower().strip() if isinstance(key_raw, str) else None
            if key and key not in all_db_plants_map: all_db_plants_map[key] = p
    
    all_db_names = list(all_db_plants_map.keys())
    if not all_db_names: return None

    best_match_result, highest_score = (None, 0)
    for search_term in [search_sci, search_common]:
        if search_term:
            results = process.extractOne(search_term, all_db_names)
            if results and results[1] >= match_threshold and results[1] > highest_score:
                highest_score, best_match_result = results[1], all_db_plants_map.get(results[0])
    return best_match_result


def display_identification_result(result):
    st.subheader("🔍 Identification Results")
    if not result: st.error("No identification result."); return
    if 'error' in result: st.error(f"Identification failed: {result.get('error', 'Unknown')}"); return
    conf = result.get('confidence', 0)
    color = "#28a745" if conf > 75 else ("#ffc107" if conf > 50 else "#dc3545")
    st.markdown(f"""
    - **Scientific Name:** `{result.get('scientific_name', 'N/A')}`
    - **Common Name:** `{result.get('common_name', 'N/A')}`
    - **Confidence:** <strong style='color:{color};'>{conf:.1f}%</strong>
    """, unsafe_allow_html=True)

def display_care_instructions(care_info, header_level=3):
    if not care_info or not isinstance(care_info, dict): st.warning("Care info missing."); return
    name = care_info.get('Plant Name', 'This Plant')
    st.markdown(f"<h{header_level}>🌱 {name} Care Guide</h{header_level}>", unsafe_allow_html=True)
    with st.expander("📋 Care Summary", expanded=True):
        c1, c2 = st.columns(2)
        details = [
            ("☀️ Light", 'Light Requirements'), ("💧 Water", 'Watering'), ("🌡️ Temp", 'Temperature Range'),
            ("💦 Humidity", 'Humidity Preferences'), ("🍃 Feeding", 'Feeding Schedule'), ("⚠️ Toxicity", 'Toxicity')
        ]
        for i, (label, key) in enumerate(details):
            col = c1 if i < len(details)/2 else c2
            col.markdown(f"**{label}**")
            col.caption(f"{care_info.get(key, 'N/A')}")
    additional_care = care_info.get('Additional Care')
    if additional_care and isinstance(additional_care, str) and additional_care.strip():
        with st.expander("✨ Pro Tips"): st.markdown(additional_care)

def find_similar_plant_matches(id_result, plant_care_data, limit=3, score_threshold=60):
    if not id_result or 'error' in id_result or not plant_care_data: return []
    # Simplified mapping as in find_care_instructions
    all_db_plants_map = {}
    for p in plant_care_data:
        keys = [p.get('Scientific Name', ''), p.get('Plant Name', '')]
        common_names_raw = p.get('Common Names', [])
        keys.extend([common_names_raw] if isinstance(common_names_raw, str) else common_names_raw)
        for key_raw in keys:
            key = key_raw.lower().strip() if isinstance(key_raw, str) else None
            if key and key not in all_db_plants_map : all_db_plants_map[key] = p # Prefer first encountered
    
    all_db_names = list(all_db_plants_map.keys())
    if not all_db_names: return []

    search_terms = [id_result.get('scientific_name','').lower().strip(), id_result.get('common_name','').lower().strip()]
    matches = {}
    for term in search_terms:
        if term:
            # extract provides list of (match, score, index/key_if_processor_returns_it) tuples
            # processor=None means it compares term against each choice directly.
            # We want to match against the keys of all_db_plants_map
            fuzz_results = process.extract(term, all_db_names, limit=limit*2) # Get more to filter
            for name, score in fuzz_results:
                if score >= score_threshold:
                    matches[name] = max(matches.get(name, 0), score)
    
    sorted_matches = sorted(matches.items(), key=lambda item: item[1], reverse=True)
    final_suggestions, seen_plants_actual = ([], set())
    for name, score in sorted_matches:
        plant_info = all_db_plants_map.get(name)
        if plant_info:
            # Check actual plant object to avoid duplicates from different name keys mapping to same plant
            plant_identifier_for_dedupe = plant_info.get('Scientific Name', '') + plant_info.get('Plant Name', '')
            if plant_identifier_for_dedupe not in seen_plants_actual:
                final_suggestions.append(plant_info)
                seen_plants_actual.add(plant_identifier_for_dedupe)
                if len(final_suggestions) >= limit: break
    return final_suggestions


def display_suggestion_buttons(suggestions):
     if not suggestions: return
     st.info("🌿 Perhaps one of these is a closer match from our database?")
     cols = st.columns(len(suggestions))
     for i, p_info in enumerate(suggestions):
         p_name = p_info.get('Plant Name', p_info.get('Scientific Name', f'Suggest {i+1}'))
         safe_p_name = "".join(c if c.isalnum() else "_" for c in p_name)
         tooltip = f"Select {p_name}" + (f" (Scientific: {p_info.get('Scientific Name')})" if p_info.get('Scientific Name') and p_info.get('Scientific Name') != p_name else "")
         if cols[i].button(p_name, key=f"suggest_{safe_p_name}_{i}", help=tooltip, use_container_width=True):
            st.session_state.plant_care_info = p_info
            st.session_state.plant_id_result = {'scientific_name': p_info.get('Scientific Name', 'N/A'), 'common_name': p_name, 'confidence': 100.0}
            st.session_state.plant_id_result_for_care_check = st.session_state.plant_id_result
            st.session_state.suggestions = None
            st.session_state.chat_history = []
            st.session_state.current_chatbot_plant_name = None
            st.session_state.suggestion_just_selected = True
            st.rerun()


def display_chat_interface(current_plant_care_info=None, plant_id_result=None):
    chatbot_display_name = "this plant"
    can_chat = False
    if current_plant_care_info and isinstance(current_plant_care_info, dict):
        chatbot_display_name = current_plant_care_info.get("Plant Name", "this plant")
        can_chat = True
    elif plant_id_result and isinstance(plant_id_result, dict) and 'error' not in plant_id_result:
        name_from_id = plant_id_result.get('common_name', plant_id_result.get('scientific_name'))
        if name_from_id and name_from_id != 'N/A' and name_from_id.strip(): chatbot_display_name = name_from_id
        can_chat = True

    if can_chat and (not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here"):
        st.warning("Chat feature requires a Gemini API key."); return
    if not can_chat: st.warning("Cannot initialize chat without valid plant identification."); return

    st.subheader(f"💬 Chat with {chatbot_display_name}")
    st.markdown("""<style>.message-container { padding: 1px 5px; } .user-message { background: #0b81fe; color: white; border-radius: 18px 18px 0 18px; padding: 8px 14px; margin: 3px 0 3px auto; width: fit-content; max-width: 80%; word-wrap: break-word; box-shadow: 0 1px 2px rgba(0,0,0,0.1); animation: fadeIn 0.3s ease-out; } .bot-message { background: #e5e5ea; color: #000; border-radius: 18px 18px 18px 0; padding: 8px 14px; margin: 3px auto 3px 0; width: fit-content; max-width: 80%; word-wrap: break-word; box-shadow: 0 1px 2px rgba(0,0,0,0.05); animation: fadeIn 0.3s ease-out; } .message-meta { font-size: 0.70rem; color: #777; margin-top: 3px; } .bot-message .message-meta { text-align: left; color: #555;} .user-message .message-meta { text-align: right; } @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } } .stChatInputContainer { position: sticky; bottom: 0; background: white; padding-top: 10px; z-index: 99;}</style>""", unsafe_allow_html=True)

    if "chat_history" not in st.session_state or st.session_state.get("current_chatbot_plant_name") != chatbot_display_name:
        if st.session_state.get("viewing_saved_details"):
             saved_plant_data = st.session_state.saved_photos.get(st.session_state.viewing_saved_details)
             st.session_state.chat_history = saved_plant_data.get('chat_log', []) if saved_plant_data else []
        else: st.session_state.chat_history = []
        st.session_state.current_chatbot_plant_name = chatbot_display_name

    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.get("chat_history", []):
            role, content, time = message.get("role"), message.get("content", ""), message.get("time", "")
            if role == "user": st.markdown(f'<div class="message-container"><div class="user-message">{content}<div class="message-meta">You • {time}</div></div></div>', unsafe_allow_html=True)
            elif role in ["assistant", "model"]: st.markdown(f'<div class="message-container"><div class="bot-message">🌿 {content}<div class="message-meta">{chatbot_display_name} • {time}</div></div></div>', unsafe_allow_html=True)

    safe_display_name = "".join(c if c.isalnum() else "_" for c in chatbot_display_name)
    if prompt := st.chat_input(f"Ask {chatbot_display_name}...", key=f"chat_input_{safe_display_name}_{random.randint(1,10000)}"): # Added random to key
        timestamp = datetime.now(EASTERN_TZ).strftime("%H:%M")
        st.session_state.chat_history.append({"role": "user", "content": prompt, "time": timestamp})
        st.rerun()

    if st.session_state.get("chat_history") and st.session_state.chat_history[-1].get("role") == "user":
        with st.spinner(f"{chatbot_display_name} is thinking..."):
            bot_response = chat_with_plant(current_plant_care_info, st.session_state.chat_history, plant_id_result)
        timestamp = datetime.now(EASTERN_TZ).strftime("%H:%M")
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response, "time": timestamp})
        if st.session_state.get("viewing_saved_details") and st.session_state.viewing_saved_details in st.session_state.saved_photos:
            st.session_state.saved_photos[st.session_state.viewing_saved_details]['chat_log'] = st.session_state.chat_history
        st.rerun()


# --- Main App Logic ---
def main():
    # Initialize State Variables
    defaults = {
        "plant_id_result": None, "plant_care_info": None, "chat_history": [],
        "current_chatbot_plant_name": None, "suggestions": None,
        "uploaded_file_bytes": None, "uploaded_file_type": None,
        "saving_mode": False, "last_view": "🏠 Home",
        "viewing_saved_details": None, "plant_id_result_for_care_check": None,
        "suggestion_just_selected": False, "viewing_plant_stats": None,
        "viewing_home_page": True, "saved_photos": {},
        "current_nav_choice": "🏠 Home" # For programmatic nav changes
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

    # Inject Ring CSS globally once
    st.markdown(get_ring_html_css(), unsafe_allow_html=True)

    # --- Sidebar ---
    st.sidebar.title("📚 Plant Buddy")
    nav_choice_options = ["🏠 Home", "🆔 Identify New Plant", "🪴 My Saved Plants", "📊 Plant Stats"]
    
    # Determine current index for radio button
    try:
        current_nav_index = nav_choice_options.index(st.session_state.current_nav_choice)
    except ValueError:
        current_nav_index = 0 # Default to Home

    nav_choice = st.sidebar.radio("Navigation", nav_choice_options, key="main_nav_radio", index=current_nav_index, label_visibility="collapsed")
    
    # Update session state if radio button changes
    if nav_choice != st.session_state.current_nav_choice:
        st.session_state.current_nav_choice = nav_choice
        # Reset view-specific states when changing main navigation
        if nav_choice == "🏠 Home":
            st.session_state.viewing_home_page = True
            st.session_state.viewing_saved_details = None
            st.session_state.viewing_plant_stats = None
        elif nav_choice == "🆔 Identify New Plant":
            st.session_state.viewing_home_page = False
            st.session_state.viewing_saved_details = None
            st.session_state.viewing_plant_stats = None
            # Minimal reset for identify page, keep uploaded file if any
        elif nav_choice == "🪴 My Saved Plants":
            st.session_state.viewing_home_page = False
            # Keep viewing_saved_details if already set, else None
            st.session_state.viewing_plant_stats = None
        elif nav_choice == "📊 Plant Stats":
            st.session_state.viewing_home_page = False
            # Keep viewing_plant_stats if already set.
            # If navigating directly, and no plant is selected for stats, it will show a message.
        st.rerun()


    st.sidebar.divider()
    st.sidebar.caption("Powered by PlantNet & Gemini")

    saved_plant_nicknames = list(st.session_state.saved_photos.keys())
    if saved_plant_nicknames:
        st.sidebar.subheader("Saved Plants")
        view_options = ["-- Select to View --"] + saved_plant_nicknames
        current_selection_for_sb = st.session_state.get("viewing_saved_details")
        sb_idx = view_options.index(current_selection_for_sb) if current_selection_for_sb and current_selection_for_sb in view_options else 0
        
        selected_saved_plant_sb = st.sidebar.selectbox(
            "View Saved Plant:", view_options, key="saved_view_selector_sb", index=sb_idx, label_visibility="collapsed"
        )
        if selected_saved_plant_sb != "-- Select to View --":
            if st.session_state.get("viewing_saved_details") != selected_saved_plant_sb:
                st.session_state.viewing_saved_details = selected_saved_plant_sb
                st.session_state.current_nav_choice = "🪴 My Saved Plants" # Switch to saved plants view
                st.session_state.viewing_plant_stats = None # Clear stats view
                st.rerun()
        elif st.session_state.get("viewing_saved_details") is not None and selected_saved_plant_sb == "-- Select to View --":
            # User deselected a plant, go back to overview of saved plants or home
            st.session_state.viewing_saved_details = None
            st.session_state.current_nav_choice = "🪴 My Saved Plants" # Stay on saved plants overview
            st.rerun()


    plant_care_data = load_plant_care_data()
    api_keys_ok = not (not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here")

    # ====================================
    # ===== Home Page View =====
    # ====================================
    if st.session_state.current_nav_choice == "🏠 Home":
        st.header("🌿 Welcome to Plant Buddy!")
        # ... (Welcome message and feature highlights as in your code) ...
        welcome_prompt = "You are Plant Buddy, a friendly plant care assistant. Give a warm, concise welcome (2-3 sentences) to a user of this app. Mention they can identify plants, get care tips, chat with their plants, and track plant health statistics."
        if "welcome_response" not in st.session_state:
            with st.spinner("Loading welcome..."):
                 messages = [{"role": "user", "parts": [{"text": "System: Be Plant Buddy for this welcome message."}]},
                             {"role": "model", "parts": [{"text": "Okay!"}]},
                             {"role": "user", "parts": [{"text": welcome_prompt}]}]
                 st.session_state.welcome_response = send_message(messages) if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here" else "Welcome to Plant Buddy! Identify plants, get care tips, chat, and track health. Happy gardening!"
        
        st.markdown(f"""<div style="background-color: #e6ffed; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; margin-bottom:20px;">
                        <h3 style="color: #2E7D32;">🌱 Hello Plant Lover!</h3>
                        <p style="font-size: 1.1em;">{st.session_state.welcome_response}</p></div>""", unsafe_allow_html=True)
        
        st.subheader("🔍 What You Can Do")
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown("#### 📸 Identify Plants")
            st.caption("Upload a photo to ID your plant using AI.")
            if st.button("Identify My Plant!", use_container_width=True, type="primary"):
                st.session_state.current_nav_choice = "🆔 Identify New Plant"
                st.rerun()
        with hc2:
            st.markdown("#### 💚 My Plant Collection")
            st.caption("View your saved plants, care guides, and stats.")
            if st.button("Go to My Plants", use_container_width=True):
                st.session_state.current_nav_choice = "🪴 My Saved Plants"
                st.rerun()

        if st.session_state.saved_photos:
            st.divider()
            st.subheader("🪴 Your Recent Plants")
            recent_plants = list(st.session_state.saved_photos.items())[-3:] # Last 3 added
            recent_plants.reverse() # Show newest first
            if recent_plants:
                cols = st.columns(len(recent_plants))
                for i, (nickname, plant_data) in enumerate(recent_plants):
                    with cols[i]:
                        with st.container(border=True):
                            if plant_data.get("image"):
                                st.image(plant_data["image"], use_container_width=True, caption=nickname)
                            else: st.markdown(f"**{nickname}**")
                            id_res = plant_data.get("id_result", {})
                            com_n = id_res.get('common_name', 'N/A')
                            if com_n and com_n != 'N/A' and com_n.lower() != nickname.lower() : st.caption(f"{com_n}")

                            if st.button("View Details", key=f"home_view_{nickname}", use_container_width=True):
                                st.session_state.viewing_saved_details = nickname
                                st.session_state.current_nav_choice = "🪴 My Saved Plants"
                                st.rerun()
            else: st.caption("No plants saved yet.")


    # ====================================
    # ===== Identify New Plant View =====
    # ====================================
    elif st.session_state.current_nav_choice == "🆔 Identify New Plant":
        st.header("🔎 Identify a New Plant")
        # Reset logic if navigating from other views to a "fresh" identify page
        if st.session_state.last_view != "🆔 Identify New Plant" and not st.session_state.get("uploaded_file_bytes"):
            keys_to_clear_for_new_id = ["plant_id_result", "plant_care_info", "chat_history", "current_chatbot_plant_name", "suggestions", "saving_mode", "plant_id_result_for_care_check", "suggestion_just_selected"]
            for key in keys_to_clear_for_new_id: st.session_state[key] = None if key not in ["chat_history"] else []

        st.session_state.last_view = "🆔 Identify New Plant"

        uploaded_file = st.file_uploader(
            "Upload a clear photo of your plant:", type=["jpg", "jpeg", "png"],
            key="plant_uploader_main",
            help="Upload an image file (JPG, PNG).",
            on_change=lambda: st.session_state.update({
                 "plant_id_result": None, "plant_care_info": None, "chat_history": [],
                 "current_chatbot_plant_name": None, "suggestions": None,
                 "uploaded_file_bytes": None, "uploaded_file_type": None,
                 "saving_mode": False, "plant_id_result_for_care_check": None,
                 "suggestion_just_selected": False
            }) # Reset on new file selection
        )

        if uploaded_file is not None and st.session_state.uploaded_file_bytes is None: # New file just uploaded
            st.session_state.uploaded_file_bytes = uploaded_file.getvalue()
            st.session_state.uploaded_file_type = uploaded_file.type
            st.rerun() # Rerun to process the new file

        if st.session_state.uploaded_file_bytes is not None:
            display_image_with_max_height(st.session_state.uploaded_file_bytes, "Your Uploaded Plant", 400)
            st.divider()

            if st.session_state.plant_id_result is None: # Needs identification
                with st.spinner("Identifying plant..."):
                    if not api_keys_ok: # Demo mode
                        time.sleep(1)
                        random_plant = random.choice(plant_care_data)
                        st.session_state.plant_id_result = {'scientific_name': random_plant.get('Scientific Name', 'Demo SciName'), 'common_name': random_plant.get('Plant Name', 'Demo Plant'), 'confidence': random.uniform(70,95)}
                    else:
                        st.session_state.plant_id_result = identify_plant(st.session_state.uploaded_file_bytes)
                    st.session_state.plant_id_result_for_care_check = None # Reset for new ID
                    st.session_state.suggestion_just_selected = False
                st.rerun()

            elif st.session_state.plant_id_result is not None: # ID result exists
                current_id_result = st.session_state.plant_id_result

                if st.session_state.saving_mode:
                    st.header("💾 Save This Plant Profile")
                    # ... (Save mode UI - largely as before) ...
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
                                    "moisture_level": random.randint(30,90) # Simulated
                                }
                                st.success(f"Saved '{save_nickname}'!"); st.balloons()
                                # Reset state after save
                                for k in ["plant_id_result", "plant_care_info", "current_chatbot_plant_name", "suggestions", "uploaded_file_bytes", "uploaded_file_type", "chat_history", "saving_mode", "plant_id_result_for_care_check", "suggestion_just_selected"]:
                                    st.session_state[k] = None if k not in ["chat_history"] else []
                                st.session_state.current_nav_choice = "🪴 My Saved Plants" # Go to saved plants
                                st.session_state.viewing_saved_details = save_nickname
                                st.rerun()
                    if st.button("❌ Cancel Save", key="cancel_save_main"): st.session_state.saving_mode = False; st.rerun()
                else: # Not saving mode
                    display_identification_result(current_id_result)
                    if 'error' not in current_id_result:
                        care_info_current = st.session_state.get('plant_care_info')
                        id_for_check = st.session_state.get('plant_id_result_for_care_check')
                        
                        if not st.session_state.suggestion_just_selected and current_id_result != id_for_check:
                            found_care = find_care_instructions(current_id_result, plant_care_data)
                            st.session_state.plant_care_info = found_care
                            st.session_state.plant_id_result_for_care_check = current_id_result
                            st.session_state.suggestions = None # Reset suggestions
                            st.session_state.chat_history = []  # Reset chat for new ID/care
                            st.session_state.current_chatbot_plant_name = None
                            st.rerun() # Rerun to reflect new care info or trigger suggestion search

                        if st.session_state.suggestion_just_selected:
                            st.session_state.suggestion_just_selected = False # Reset flag

                        care_to_display = st.session_state.get('plant_care_info') # Get potentially updated care
                        
                        if care_to_display:
                            display_care_instructions(care_to_display)
                            st.button("💾 Save Plant Profile", key="save_id_care_btn", on_click=lambda: st.session_state.update({"saving_mode": True}))
                            st.divider()
                            display_chat_interface(current_plant_care_info=care_to_display, plant_id_result=current_id_result)
                        else: # No specific care found
                            st.warning("Could not find specific care instructions for this exact plant in our database.")
                            if st.session_state.suggestions is None: # Need to find suggestions
                                st.session_state.suggestions = find_similar_plant_matches(current_id_result, plant_care_data)
                                if st.session_state.suggestions is not None: # Avoid rerun if still None (e.g. no matches)
                                     st.rerun() # Rerun to display suggestions

                            display_suggestion_buttons(st.session_state.suggestions)
                            st.divider()
                            st.button("💾 Save Identification Only", key="save_id_only_btn", on_click=lambda: st.session_state.update({"saving_mode": True}))
                            st.divider()
                            st.info("You can still chat with the plant based on its general identification.")
                            display_chat_interface(current_plant_care_info=None, plant_id_result=current_id_result)


    # ====================================
    # ===== Saved Plants View =====
    # ====================================
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
            # Sync session state for chat consistency IF NEEDED (but should be set by selectbox already)
            if st.session_state.get("plant_id_result") != saved_id_result: st.session_state.plant_id_result = saved_id_result
            
            st.divider()
            
            saved_care_info = entry.get("care_info")
            if st.session_state.get("plant_care_info") != saved_care_info: st.session_state.plant_care_info = saved_care_info

            col_stats_btn, col_del_btn = st.columns([0.7, 0.3])
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
            with col_del_btn:
                if st.button(f"🗑️ Delete '{nickname_to_view}'", key=f"del_btn_{nickname_to_view}", type="secondary", use_container_width=True):
                    if st.checkbox(f"Confirm deletion of {nickname_to_view}", key=f"confirm_del_{nickname_to_view}"):
                        del st.session_state.saved_photos[nickname_to_view]
                        st.session_state.viewing_saved_details = None
                        for k in ["plant_id_result", "plant_care_info", "current_chatbot_plant_name", "suggestions", "chat_history", "viewing_plant_stats"]:
                            st.session_state[k] = None if k not in ["chat_history"] else []
                        st.success(f"Deleted '{nickname_to_view}'.")
                        st.rerun()

        else: # Overview of all saved plants
            st.info("Select a plant from the 'View Saved Plant' dropdown in the sidebar to see its details, or browse below.")
            st.markdown("---")
            # ... (Grid display logic - keep similar to your existing code) ...
            num_cols_grid = 3
            grid_cols = st.columns(num_cols_grid)
            for i, (nick, data) in enumerate(st.session_state.saved_photos.items()):
                with grid_cols[i % num_cols_grid]:
                    with st.container(border=True):
                        if data.get("image"): st.image(data["image"], use_container_width=True, caption=nick)
                        else: st.markdown(f"**{nick}**")
                        id_res_grid = data.get("id_result", {})
                        com_n_grid = id_res_grid.get('common_name', 'N/A')
                        if com_n_grid and com_n_grid != 'N/A' and com_n_grid.lower() != nick.lower(): st.caption(f"{com_n_grid}")
                        
                        btn_c1, btn_c2 = st.columns(2)
                        with btn_c1:
                            if st.button("Details", key=f"grid_detail_{nick}", use_container_width=True):
                                st.session_state.viewing_saved_details = nick; st.rerun()
                        with btn_c2:
                            if st.button("📊 Stats", key=f"grid_stats_{nick}", use_container_width=True):
                                st.session_state.viewing_plant_stats = nick
                                st.session_state.current_nav_choice = "📊 Plant Stats"
                                st.rerun()

    # ====================================
    # ===== Plant Stats View (INTEGRATED RINGS) =====
    # ====================================
    elif st.session_state.current_nav_choice == "📊 Plant Stats":
        st.session_state.last_view = "📊 Plant Stats"
        plant_nickname_for_stats = st.session_state.get("viewing_plant_stats")

        if not plant_nickname_for_stats or plant_nickname_for_stats not in st.session_state.saved_photos:
            st.warning("No plant selected for stats view. Please select a plant from 'My Saved Plants'.")
            if st.button("← Back to Saved Plants", key="stats_back_to_saved_no_plant"):
                st.session_state.current_nav_choice = "🪴 My Saved Plants"
                st.session_state.viewing_plant_stats = None
                st.rerun()
        else:
            plant_data_stats = st.session_state.saved_photos[plant_nickname_for_stats]
            st.header(f"📊 Plant Stats: {plant_nickname_for_stats}")
            if st.button("← Back to Plant Details", key="stats_back_to_details"):
                st.session_state.current_nav_choice = "🪴 My Saved Plants"
                st.session_state.viewing_saved_details = plant_nickname_for_stats # Ensure we go back to its details
                st.session_state.viewing_plant_stats = None
                st.rerun()
            st.divider()

            # --- Display Rings ---
            simulated_time_str = (datetime.now(EASTERN_TZ) - timedelta(minutes=random.randint(1,5))).strftime('%H:%M') # For watch face time

            # 1. Moisture Ring
            moisture_level_perc = plant_data_stats.get("moisture_level", random.randint(30,70)) # %
            moisture_desc_ring = f"Current soil moisture is {moisture_level_perc}%. Ideal levels vary, but consistently very low or high may need attention."
            ring1_html = generate_ring_html("Moisture", str(moisture_level_perc), f"OF {MOISTURE_MAX_PERCENT}%", moisture_level_perc, MOISTURE_COLOR, MOISTURE_TRACK_COLOR, simulated_time_str, moisture_desc_ring, 0)

            # 2. Temperature Ring
            care_info_stats = plant_data_stats.get("care_info", {})
            temp_range_str_stats = care_info_stats.get("Temperature Range", "65-85°F")
            min_f, max_f = parse_temp_range(temp_range_str_stats)
            
            current_temp_f_sim = TEMP_DISPLAY_MIN_F - 5 # Default to slightly below range if parsing fails
            if min_f is not None and max_f is not None:
                 # Simulate a temp, sometimes slightly outside ideal for visual variety
                ideal_mid = (min_f + max_f) / 2
                current_temp_f_sim = random.uniform(ideal_mid - 7, ideal_mid + 7)
                current_temp_f_sim = round(max(TEMP_DISPLAY_MIN_F - 5, min(TEMP_DISPLAY_MAX_F + 5, current_temp_f_sim))) # Clamp near display range
            else: # Fallback if parsing failed
                current_temp_f_sim = random.randint(68, 78)


            temp_progress = ((current_temp_f_sim - TEMP_DISPLAY_MIN_F) / (TEMP_DISPLAY_MAX_F - TEMP_DISPLAY_MIN_F)) * 100
            temp_desc_ring = f"Ambient temp around {current_temp_f_sim}°F. Ideal: {temp_range_str_stats if temp_range_str_stats else 'N/A'}. Ring shows vs {TEMP_DISPLAY_MIN_F}-{TEMP_DISPLAY_MAX_F}°F."
            ring2_html = generate_ring_html("Temperature", str(int(current_temp_f_sim)), "°F NOW", temp_progress, TEMPERATURE_COLOR, TEMPERATURE_TRACK_COLOR, simulated_time_str, temp_desc_ring, 1)

            # 3. Data Freshness Ring ("Last Checked")
            minutes_ago_sim = random.randint(1, FRESHNESS_MAX_MINUTES_AGO - 10) # Simulate recently checked
            freshness_progress_sim = max(0, (1 - (minutes_ago_sim / FRESHNESS_MAX_MINUTES_AGO))) * 100
            freshness_desc_ring = f"Stats notionally checked {minutes_ago_sim} min(s) ago. Regular checks help monitor conditions."
            ring3_html = generate_ring_html("Last Check", str(minutes_ago_sim), "MINS AGO", freshness_progress_sim, FRESHNESS_COLOR, FRESHNESS_TRACK_COLOR, simulated_time_str, freshness_desc_ring, 2)
            
            st.markdown(f'<div class="watch-face-grid">{ring1_html}{ring2_html}{ring3_html}</div>', unsafe_allow_html=True)
            st.divider()
            
            # --- Display Plant Image and Basic Info (below rings) ---
            img_col, info_col = st.columns([0.4, 0.6])
            with img_col:
                if plant_data_stats.get("image"):
                    display_image_with_max_height(plant_data_stats["image"], caption=f"{plant_nickname_for_stats}", max_height_px=250)
            with info_col:
                st.subheader("Plant Details")
                id_res_stats = plant_data_stats.get("id_result", {})
                st.markdown(f"**Nickname:** {plant_nickname_for_stats}")
                st.markdown(f"**Scientific Name:** `{id_res_stats.get('scientific_name', 'N/A')}`")
                st.markdown(f"**Common Name:** `{id_res_stats.get('common_name', 'N/A')}`")

            st.divider()
            # Display care tips again (using smaller header)
            if care_info_stats:
                display_care_instructions(care_info_stats, header_level=4)
            else:
                st.info("No specific care information was saved for this plant.")


# --- Run the App ---
if __name__ == "__main__":
    if not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here":
        st.sidebar.warning("PlantNet API Key missing. Identification uses demo data.", icon="🔑")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        st.sidebar.warning("Gemini API Key missing. Chat features disabled/limited.", icon="🔑")
    main()
