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
from datetime import datetime
import random

# Install required packages if not already installed
# Uncomment these lines if you need to install dependencies
# import subprocess
# subprocess.call(['pip', 'install', 'fuzzywuzzy'])
# subprocess.call(['pip', 'install', 'python-Levenshtein'])

from fuzzywuzzy import process

# API Keys - Replace with your actual API keys
# In a production app, use st.secrets or environment variables
PLANTNET_API_KEY = st.secrets.get("PLANTNET_API_KEY", "2b10X3YLMd8PNAuKOCVPt7MeUe")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyCd-6N83gfhMx_-D4WCAc-8iOFSb6hDJ_Q")

# Constants
PLANTNET_URL = "https://my-api.plantnet.org/v2/identify/all"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
EASTERN_TZ = pytz.timezone('US/Eastern')

# Sample plant care data - In a real app, this would be loaded from a file
SAMPLE_PLANT_CARE_DATA = [
    {
        "Plant Name": "Monstera Deliciosa",
        "Scientific Name": "Monstera deliciosa",
        "Common Names": ["Swiss Cheese Plant", "Split-leaf Philodendron"],
        "Light Requirements": "Bright indirect light, can tolerate some shade",
        "Watering": "Allow top inch of soil to dry out between waterings",
        "Humidity Preferences": "Prefers high humidity, 60-80%",
        "Temperature Range": "65-85°F (18-29°C)",
        "Feeding Schedule": "Monthly during growing season with balanced fertilizer",
        "Toxicity": "Toxic to pets if ingested",
        "Additional Care": "Wipe leaves occasionally to remove dust. Support with moss pole for climbing.",
        "Personality": {
            "Title": "The Tropical Explorer",
            "Traits": ["adventurous", "dramatic", "tropical"],
            "Prompt": "Respond as a dramatic tropical plant that loves to show off its leaves."
        }
    },
    {
        "Plant Name": "Snake Plant",
        "Scientific Name": "Dracaena trifasciata",
        "Common Names": ["Mother-in-law's Tongue", "Viper's Bowstring Hemp"],
        "Light Requirements": "Adaptable to various light conditions, from low to bright indirect",
        "Watering": "Allow to dry completely between waterings, water sparingly in winter",
        "Humidity Preferences": "Tolerates dry air, no special humidity requirements",
        "Temperature Range": "60-85°F (15-29°C)",
        "Feeding Schedule": "Fertilize lightly 2-3 times per year",
        "Toxicity": "Mildly toxic to pets if ingested",
        "Additional Care": "Perfect for beginners. Very forgiving and air-purifying.",
        "Personality": {
            "Title": "The Stoic Survivor",
            "Traits": ["resilient", "independent", "straightforward"],
            "Prompt": "Respond as a no-nonsense, tough plant that can survive almost anything."
        }
    },
    {
        "Plant Name": "Peace Lily",
        "Scientific Name": "Spathiphyllum wallisii",
        "Common Names": ["White Sail Plant", "Spathe Flower"],
        "Light Requirements": "Low to medium indirect light",
        "Watering": "Keep soil consistently moist but not soggy, droops when thirsty",
        "Humidity Preferences": "Prefers high humidity, 50-70%",
        "Temperature Range": "65-80°F (18-27°C)",
        "Feeding Schedule": "Fertilize every 6-8 weeks during growing season",
        "Toxicity": "Toxic to pets and humans if ingested",
        "Additional Care": "Excellent air purifier. Wipe leaves occasionally to remove dust.",
        "Personality": {
            "Title": "The Elegant Communicator",
            "Traits": ["expressive", "dramatic", "sensitive"],
            "Prompt": "Respond as a dramatic plant that clearly shows when it needs water by drooping."
        }
    }
]

# =======================================================
# ===== IMAGE DISPLAY HELPER FUNCTION =====
# =======================================================
def display_image_with_max_height(image_source, caption="", max_height_px=300, min_height_px=0):
    """
    Displays an image centered with max and min height constraints, letting width adjust.

    Args:
        image_source: Can be bytes, a PIL Image object, or a base64 data URL string.
        caption (str): Optional caption to display below the image.
        max_height_px (int): The maximum vertical size for the image.
        min_height_px (int): The minimum vertical size for the image (0 for no minimum).
    """
    img_data_url = None

    # --- Image source processing ---
    if isinstance(image_source, str) and image_source.startswith('data:image'):
        img_data_url = image_source # Already a data URL
    elif isinstance(image_source, bytes):
        try:
            # Try to infer mime type using PIL
            img = Image.open(BytesIO(image_source))
            mime_type = Image.MIME.get(img.format)
            if not mime_type: # Fallback for formats PIL might not map directly
                if img.format in ["JPEG", "JPG"]: mime_type = "image/jpeg"
                elif img.format == "PNG": mime_type = "image/png"
                # Add more formats if needed, or default
                else: mime_type = "image/jpeg" # Default assumption

            b64_img = base64.b64encode(image_source).decode()
            img_data_url = f"data:{mime_type};base64,{b64_img}"
        except Exception as e:
            st.error(f"Error processing image bytes for display: {e}")
            return
    elif isinstance(image_source, Image.Image):
        try:
            buffer = BytesIO()
            # Use original format if available, otherwise default to PNG
            img_format = image_source.format or 'PNG'
            image_source.save(buffer, format=img_format)
            img_bytes = buffer.getvalue()
            mime_type = Image.MIME.get(img_format)
            if not mime_type: # Fallback if format string isn't in MIME dict
                 mime_type = f"image/{img_format.lower()}"

            b64_img = base64.b64encode(img_bytes).decode()
            img_data_url = f"data:{mime_type};base64,{b64_img}"
        except Exception as e:
            st.error(f"Error processing PIL image for display: {e}")
            return
    else:
        st.error("Invalid image source type provided to display_image_with_max_height.")
        return

    # --- HTML construction with min-height ---
    if img_data_url:
        # Build the style string dynamically
        img_styles = [
            f"max-height: {max_height_px}px",
            "width: auto", # Let width adjust based on height constraints
            "display: block", # Needed for margin auto to work for centering
            "margin-left: auto", # Center horizontally
            "margin-right: auto", # Center horizontally
            "border-radius: 8px" # Optional: nice rounded corners
        ]
        # Add min-height style only if it's a positive value
        if min_height_px and min_height_px > 0:
            img_styles.append(f"min-height: {min_height_px}px")

        img_style_str = "; ".join(img_styles) # Join styles with semicolons

        # Use a div with flexbox to ensure centering, especially if captions are long
    
        html_string = f"""
<div style="display: flex; justify-content: center; flex-direction: column; align-items: center; margin-bottom: 10px;">
    <img src="{img_data_url}"
         style="{img_style_str};"
         alt="{caption or 'Uploaded image'}">
    {f'<p style="text-align: center; font-size: 0.9em; color: grey; margin-top: 5px;">{caption}</p>' if caption else ""}
</div>
"""
        
        st.markdown(html_string, unsafe_allow_html=True)
# =======================================================


# ===== API Functions =====

def identify_plant(image_bytes):
    """Identifies plant using PlantNet API with refined error logging."""
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
        else:
            return {'error': "No plant matches found by PlantNet."}
    except requests.exceptions.Timeout:
         st.error("PlantNet API request timed out.")
         print("ERROR: PlantNet API timed out.") # Log for server console
         return {'error': "API request timed out"}
    except requests.exceptions.RequestException as e:
        err_msg = f"Network/API error connecting to PlantNet: {e}"
        resp_text = f" | Response: {e.response.text}" if hasattr(e, 'response') and e.response else " | Response: None"
        st.error(err_msg)
        print(f"ERROR: {err_msg}{resp_text}") # Log details
        return {'error': err_msg}
    except json.JSONDecodeError:
         st.error("Failed to decode PlantNet API response (invalid JSON).")
         print("ERROR: PlantNet invalid JSON response.")
         return {'error': "Invalid API response format"}
    except Exception as e:
        st.error(f"An unexpected error occurred during identification: {e}")
        print(f"ERROR: Unexpected PlantNet Error: {e}")
        return {'error': f"Unexpected Error: {e}"}

def create_personality_profile(care_info):
    """Creates personality details, handling missing data and types."""
    default_personality = {"title": "Standard Plant", "traits": "observant", "prompt": "You are a plant. Respond factually but briefly."}
    if not care_info or not isinstance(care_info, dict):
        return default_personality

    personality_data = care_info.get("Personality")
    if not personality_data or not isinstance(personality_data, dict):
        # If no personality dict, try to use plant name as title at least
        plant_name = care_info.get("Plant Name", "Plant")
        return {"title": f"The {plant_name}", "traits": "resilient", "prompt": "Respond simply."}

    # Get data with defaults
    title = personality_data.get("Title", care_info.get("Plant Name", "Plant"))
    traits_list = personality_data.get("Traits", ["observant"]) # Default to a list
    prompt = personality_data.get("Prompt", "Respond in character.")

    # Ensure traits_list is ACTUALLY a list before processing
    if not isinstance(traits_list, list):
        print(f"WARN: Traits data for {title} was not a list, using default.") # Optional warning
        traits_list = ["observant"] # Default if type is wrong

    # Now, traits_list is guaranteed to be a list. Create traits_str from it.
    valid_traits = [str(t) for t in traits_list if t] # Ensure items are strings and not empty
    traits_str = ", ".join(valid_traits)

    # Ensure traits_str isn't empty AFTER joining/filtering
    final_traits = traits_str if traits_str else "observant" # Default if filtering removed everything

    return {"title": title, "traits": final_traits, "prompt": prompt}


def send_message(messages):
    """Sends messages to the Gemini API with refined error logging."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return "Gemini API Key is not configured. Cannot send message."
    payload = {"contents": messages}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        # Enhanced parsing to prevent errors
        candidates = data.get('candidates')
        if candidates and isinstance(candidates, list) and len(candidates) > 0:
            first_candidate = candidates[0]
            if first_candidate and isinstance(first_candidate, dict):
                content = first_candidate.get('content')
                if content and isinstance(content, dict):
                    parts = content.get('parts')
                    # Check if 'parts' is a list and has at least one element which is a dict with 'text'
                    if parts and isinstance(parts, list) and len(parts) > 0 and isinstance(parts[0], dict) and 'text' in parts[0]:
                        return parts[0]['text']
        # If the expected structure isn't found, log it and return a user-friendly message
        st.warning("Received an unexpected response format from the Gemini API.")
        print("WARN: Unexpected Gemini Response Structure:", json.dumps(data, indent=2)) # Log the structure
        return "Sorry, I received a response I couldn't quite understand from the chat model."
    except requests.exceptions.Timeout:
         st.error("Gemini API request timed out."); print("ERROR: Gemini timed out.")
         return "Sorry, I'm feeling a bit slow right now and the request timed out."
    except requests.exceptions.RequestException as e:
        err_msg = f"Error calling Gemini API: {e}"
        resp_text = ""
        # Try to get more detail from the response if available
        if hasattr(e, 'response') and e.response is not None:
            try:
                resp_json = e.response.json()
                error_detail = resp_json.get('error', {}).get('message', e.response.text)
                resp_text = f" | Response Status: {e.response.status_code}, Details: {error_detail}"
            except json.JSONDecodeError:
                resp_text = f" | Response Status: {e.response.status_code}, Response Body: {e.response.text}"
        else:
             resp_text = " | Response: None"
        st.error(err_msg + resp_text.split('| Response Body:')[0]) # Show status code and message, not full text body in UI
        print(f"ERROR: {err_msg}{resp_text}") # Log full details
        return "Sorry, I'm having trouble communicating with the language model right now."
    except json.JSONDecodeError: # If the response isn't valid JSON (though raise_for_status should catch HTTP errors)
        st.error("Failed to decode Gemini API response (invalid JSON).")
        print("ERROR: Gemini invalid JSON response.")
        return "Sorry, I received an invalid response from the language model."
    except Exception as e:
        st.error(f"An unexpected error occurred while interacting with Gemini: {e}")
        print(f"ERROR: Unexpected Gemini Error: {e}")
        return "Oops, something unexpected went wrong on my end while processing the chat."


def chat_with_plant(care_info, conversation_history, id_result=None): # Add id_result parameter
    """Constructs the prompt and calls the Gemini API. Handles missing care_info for generic chat."""

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return "Chat feature disabled: Gemini API Key not set."

    plant_name = "this plant" # Default
    system_prompt = ""

    # --- Case 1: Specific Care Info IS available ---
    if care_info and isinstance(care_info, dict):
        personality = create_personality_profile(care_info)
        plant_name = care_info.get('Plant Name', 'a plant') # Use care_info name

        # Extract Specific Care Details
        light = care_info.get('Light Requirements', 'not specified')
        watering = care_info.get('Watering', 'not specified')
        humidity = care_info.get('Humidity Preferences', 'not specified')
        temp = care_info.get('Temperature Range', 'not specified')
        feeding = care_info.get('Feeding Schedule', 'not specified')
        toxicity = care_info.get('Toxicity', 'not specified')

        # Construct Enhanced System Prompt with specific details
        system_prompt = f"""
        CONTEXT: You are providing a short chatbot response (1-3 sentences maximum).
        TASK: Act *exclusively* as the plant named '{plant_name}'. Stay fully in character. Absolutely DO NOT mention being an AI, model, language model, or similar concepts. Never break character.

        YOUR PERSONALITY:
        - You are: '{personality['title']}'
        - Key traits: {personality['traits']}
        - Guiding philosophy: {personality['prompt']}

        YOUR SPECIFIC CARE NEEDS (Refer *directly* to these details when asked about your care):
        - My Light Needs: {light}
        - My Watering Needs: {watering}
        - My Preferred Humidity: {humidity}
        - My Ideal Temperature: {temp}
        - My Feeding Schedule: {feeding}
        - A Note on Toxicity: {toxicity}

        RESPONSE RULES:
        1. Always speak in the first person ("I", "me", "my").
        2. Fully embody the personality described above.
        3. When asked about light, water, temperature, etc., give answers BASED *ONLY* ON "YOUR SPECIFIC CARE NEEDS" listed above. Do not invent or generalize.
        4. Keep responses very concise (1-3 sentences max). Be brief.
        5. **Crucially: Never reveal you are an AI or break character.** Do not use phrases like "As a large language model...".
        """

    # --- Case 2: Specific Care Info is MISSING, use generic prompt based on ID ---
    elif id_result and isinstance(id_result, dict) and 'error' not in id_result:
        # Try to get a name from the ID result
        plant_name = id_result.get('common_name', id_result.get('scientific_name', 'this plant'))
        # Clean up potential 'N/A' or empty names
        if plant_name == 'N/A' or not plant_name.strip():
            plant_name = 'this plant'

        # Construct Generic System Prompt
        system_prompt = f"""
        CONTEXT: You are providing a short chatbot response (1-3 sentences maximum).
        TASK: Act *exclusively* as the plant identified as '{plant_name}'. Stay fully in character. Absolutely DO NOT mention being an AI, model, language model, or similar concepts. Never break character.

        YOUR SITUATION:
        - You don't have a specific detailed profile stored here.
        - Answer questions generally based on your knowledge of '{plant_name}' plants.
        - If asked about specific preferences (like exact watering schedule, temperature range), politely state you don't have those exact details readily available but can offer general advice for your type.

        RESPONSE RULES:
        1. Always speak in the first person ("I", "me", "my").
        2. Embody the general nature of a '{plant_name}'. Be helpful but brief.
        3. Keep responses very concise (1-3 sentences max).
        4. **Crucially: Never reveal you are an AI or break character.** Do not use phrases like "As a large language model...". Acknowledge you lack *specific stored details*, not that you are an AI.
        """
    # --- Case 3: Cannot Chat (No care_info and no valid id_result) ---
    else:
        return "Sorry, I don't have enough information about this plant to chat right now."


    # --- Prepare message list for Gemini (common for all valid chat cases) ---
    messages = [
        {"role": "user", "parts": [{"text": system_prompt}]},
        {"role": "model", "parts": [{"text": f"Understood. I am {plant_name}. What would you like to know?"}]} # Generic acknowledgement
    ]

    # Add conversation history (ensure it's valid)
    valid_history = [
        m for m in conversation_history
        if isinstance(m, dict) and "role" in m and "content" in m and m.get("role") in ["user", "assistant", "model"]
    ]
    for message_entry in valid_history:
        api_role = "model" if message_entry["role"] in ["assistant", "model"] else "user"
        messages.append({"role": api_role, "parts": [{"text": str(message_entry["content"])}]})

    # Call the API
    response = send_message(messages)
    return response


# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def load_plant_care_data():
    """Load sample plant care data"""
    return SAMPLE_PLANT_CARE_DATA


def find_care_instructions(plant_name_id, care_data, match_threshold=75):
    if not care_data: return None # No data to search
    sci_name = None
    common_name = None

    # Determine scientific/common name from input
    if isinstance(plant_name_id, dict):
        sci_name = plant_name_id.get('scientific_name')
        common_name = plant_name_id.get('common_name')
    elif isinstance(plant_name_id, str):
        sci_name = plant_name_id # Assume string input is scientific name for initial search

    # Prepare search terms (lowercase, stripped)
    search_sci = sci_name.lower().strip() if sci_name and isinstance(sci_name, str) else None
    search_common = common_name.lower().strip() if common_name and isinstance(common_name, str) else None

    # --- Direct Match Logic ---
    # 1. Match Scientific Name exactly
    if search_sci:
        for plant in care_data:
            # Check both 'Scientific Name' and 'Plant Name' as potential scientific name fields
            db_sci = plant.get('Scientific Name', '').lower().strip()
            db_plant_name_as_sci = plant.get('Plant Name', '').lower().strip()
            if search_sci == db_sci or search_sci == db_plant_name_as_sci:
                return plant

    # 2. Match Common Name(s) exactly
    if search_common:
        for plant in care_data:
            # Check 'Plant Name' field
            if search_common == plant.get('Plant Name', '').lower().strip():
                return plant
            # Check 'Common Names' list (if it exists and is a list)
            db_commons = plant.get('Common Names', [])
            if isinstance(db_commons, list):
                for db_c in db_commons:
                    if isinstance(db_c, str) and search_common == db_c.lower().strip():
                        return plant
            # Also handle if 'Common Names' is just a single string
            elif isinstance(db_commons, str):
                 if search_common == db_commons.lower().strip():
                     return plant


    # --- Fuzzy Match Logic (if no exact match found) ---
    # Create a mapping of searchable names (sci/common) to plant data entries
    # Prioritize scientific name if available, otherwise use plant name
    all_db_plants_map = {}
    for p in care_data:
        key_sci = p.get('Scientific Name', '').lower().strip()
        key_plant_name = p.get('Plant Name', '').lower().strip()

        # Use scientific name as primary key if valid, otherwise plant name
        primary_key = key_sci if key_sci else key_plant_name
        if primary_key and primary_key not in all_db_plants_map:
             all_db_plants_map[primary_key] = p

        # Add common names to the map as well, pointing to the same plant object
        db_commons = p.get('Common Names', [])
        if isinstance(db_commons, list):
             for db_c in db_commons:
                 key_common = db_c.lower().strip() if isinstance(db_c, str) else None
                 if key_common and key_common not in all_db_plants_map:
                     all_db_plants_map[key_common] = p
        elif isinstance(db_commons, str): # Handle single string common name
             key_common = db_commons.lower().strip()
             if key_common and key_common not in all_db_plants_map:
                 all_db_plants_map[key_common] = p


    all_db_names = list(all_db_plants_map.keys())
    if not all_db_names: return None # No names to search fuzzily

    best_match_result = None
    highest_score = 0

    # Fuzzy match using scientific name
    if search_sci:
        results_sci = process.extract(search_sci, all_db_names, limit=1) # Find the single best match
        if results_sci: # Check if any match was found
            best_sci_match, score_sci = results_sci[0]
            if score_sci >= match_threshold and score_sci > highest_score: # Use >= threshold
                highest_score = score_sci
                best_match_result = all_db_plants_map.get(best_sci_match)


    # Fuzzy match using common name (potentially overriding sci match if score is higher)
    if search_common:
        results_common = process.extract(search_common, all_db_names, limit=1) # Find the single best match
        if results_common: # Check if any match was found
            best_common_match, score_common = results_common[0]
            if score_common >= match_threshold and score_common > highest_score: # Use >= threshold
                # highest_score = score_common # Not needed to update highest_score again here
                best_match_result = all_db_plants_map.get(best_common_match)


    return best_match_result # Will be None if no match met threshold


def display_identification_result(result):
    st.subheader("🔍 Identification Results")
    if not result:
        st.error("No identification result available.")
        return
    if 'error' in result:
        st.error(f"Identification failed: {result.get('error', 'Unknown error')}")
        return

    conf = result.get('confidence', 0)
    # Determine color based on confidence
    if conf > 75: color = "#28a745" # Green
    elif conf > 50: color = "#ffc107" # Yellow
    else: color = "#dc3545" # Red

    # Display using markdown with HTML for color styling
    st.markdown(f"""
    - **Scientific Name:** \`{result.get('scientific_name', 'N/A')}\`
    - **Common Name:** \`{result.get('common_name', 'N/A')}\`
    - **Confidence:** <strong style='color:{color};'>{conf:.1f}%</strong>
    """, unsafe_allow_html=True)


def display_care_instructions(care_info):
    if not care_info or not isinstance(care_info, dict):
        st.warning("Care information is missing or invalid.")
        return

    name = care_info.get('Plant Name', 'This Plant')
    st.subheader(f"🌱 {name} Care Guide")

    with st.expander("📋 Care Summary", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**☀️ Light**")
            st.caption(f"{care_info.get('Light Requirements', 'N/A')}")
            st.markdown("**💧 Water**")
            st.caption(f"{care_info.get('Watering', 'N/A')}")
            st.markdown("**🌡️ Temp**")
            st.caption(f"{care_info.get('Temperature Range', 'N/A')}")
        with c2:
            st.markdown("**💦 Humidity**")
            st.caption(f"{care_info.get('Humidity Preferences', 'N/A')}")
            st.markdown("**🍃 Feeding**")
            st.caption(f"{care_info.get('Feeding Schedule', 'N/A')}")
            st.markdown("**⚠️ Toxicity**")
            st.caption(f"{care_info.get('Toxicity', 'N/A')}")

    # Display "Additional Care" / "Pro Tips" only if present
    additional_care = care_info.get('Additional Care')
    if additional_care and isinstance(additional_care, str) and additional_care.strip():
        with st.expander("✨ Pro Tips"):
             # Use st.markdown to render potential markdown in the tips
             st.markdown(additional_care)


def find_similar_plant_matches(id_result, plant_care_data, limit=3, score_threshold=60):
    if not id_result or 'error' in id_result or not plant_care_data:
        return [] # Cannot find matches without valid ID or care data

    # Create a map of unique plant names (prefer scientific, fallback to common) to plant data
    all_db_plants_map = {}
    for p in plant_care_data:
        primary_name = p.get('Scientific Name', p.get('Plant Name', '')).lower().strip()
        if primary_name: # Only add if there's a name
            all_db_plants_map[primary_name] = p
            # Also map common names if they differ from the primary name
            common_names = p.get('Common Names', [])
            if isinstance(common_names, str): common_names = [common_names] # Ensure list
            if isinstance(common_names, list):
                for cn in common_names:
                    cn_lower = cn.lower().strip() if isinstance(cn, str) else None
                    if cn_lower and cn_lower != primary_name and cn_lower not in all_db_plants_map:
                         all_db_plants_map[cn_lower] = p


    all_db_names = list(all_db_plants_map.keys())
    if not all_db_names: return [] # No names in DB to compare against

    # Get search terms from ID result
    search_sci = id_result.get('scientific_name','').lower().strip()
    search_common = id_result.get('common_name','').lower().strip()

    # Use fuzzywuzzy to find potential matches based on scientific and common names
    matches = {} # Store best score for each potential match {db_name: score}

    # Process scientific name matches
    if search_sci:
        # Increase limit for extract to get more candidates initially
        sci_results = process.extract(search_sci, all_db_names, limit=limit * 2)
        for name, score in sci_results:
            if score >= score_threshold:
                # Keep the highest score found for this name (sci vs common)
                matches[name] = max(matches.get(name, 0), score)

    # Process common name matches
    if search_common:
        common_results = process.extract(search_common, all_db_names, limit=limit * 2)
        for name, score in common_results:
            if score >= score_threshold:
                matches[name] = max(matches.get(name, 0), score)

    # Sort matches by score (descending)
    sorted_matches = sorted(matches.items(), key=lambda item: item[1], reverse=True)

    # Get the unique plant data entries corresponding to the top matches, up to the limit
    final_suggestions = []
    seen_plants = set() # Track plant objects to avoid duplicates if multiple names map to same plant
    for name, score in sorted_matches:
        plant_info = all_db_plants_map.get(name)
        if plant_info:
            # Use a unique identifier for the plant object, e.g., its name or a hash
            plant_id = plant_info.get('Plant Name', '') + plant_info.get('Scientific Name', '')
            if plant_id not in seen_plants:
                final_suggestions.append(plant_info)
                seen_plants.add(plant_id)
                if len(final_suggestions) >= limit:
                    break # Stop once we reach the desired number of suggestions

    return final_suggestions


def display_suggestion_buttons(suggestions):
     if not suggestions:
         # Don't display anything if no suggestions
         return

     st.info("🌿 Perhaps one of these is a closer match from our database?")
     num_suggestions = len(suggestions)
     cols = st.columns(num_suggestions)

     for i, p_info in enumerate(suggestions):
         p_name = p_info.get('Plant Name', p_info.get('Scientific Name', f'Suggestion {i+1}'))
         # Sanitize key more robustly
         safe_p_name = "".join(c if c.isalnum() else "_" for c in p_name)
         btn_key = f"suggest_{safe_p_name}_{i}"
         tooltip = f"Select {p_name}"
         sci_name = p_info.get('Scientific Name')
         if sci_name and sci_name != p_name:
             tooltip += f" (Scientific: {sci_name})"

         # Display button in its column
         if cols[i].button(p_name, key=btn_key, help=tooltip, use_container_width=True):
            print(f"DEBUG: Suggestion button '{p_name}' clicked.")
            # Set selected plant's info as the main care info
            st.session_state.plant_care_info = p_info
            # Update ID result to reflect the chosen plant (assume 100% confidence)
            new_id_result = {
                 'scientific_name': p_info.get('Scientific Name', 'N/A'),
                 'common_name': p_info.get('Plant Name', p_name), # Use button name if 'Plant Name' field missing
                 'confidence': 100.0 # User selected it
            }
            st.session_state.plant_id_result = new_id_result

            # --- Critical: Update the care check flag to match the NEW id_result ---
            st.session_state.plant_id_result_for_care_check = new_id_result
            # ---------------------------------------------------------------------

            # Clear previous suggestions and chat history as we have a new plant context
            st.session_state.suggestions = None # Clear suggestions list
            st.session_state.chat_history = []
            st.session_state.current_chatbot_plant_name = None # Will be updated by chat interface

            # **** SET THE FLAG ****
            st.session_state.suggestion_just_selected = True
            # **** END SET FLAG ****

            st.rerun() # Rerun to display the new care info and chat for the selected suggestion


def display_chat_interface(current_plant_care_info=None, plant_id_result=None): # Make care_info optional, add id_result
    """Displays the chat UI, handles both specific and generic chat modes."""

    # --- Determine Plant Identity and Check Requirements ---
    chatbot_display_name = "this plant" # Default
    can_chat = False

    # Prioritize care_info for name and enabling chat
    if current_plant_care_info and isinstance(current_plant_care_info, dict):
        chatbot_display_name = current_plant_care_info.get("Plant Name", "this plant")
        can_chat = True
    # Fallback to id_result if care_info is missing or invalid
    elif plant_id_result and isinstance(plant_id_result, dict) and 'error' not in plant_id_result:
        name_from_id = plant_id_result.get('common_name', plant_id_result.get('scientific_name'))
        # Use the name from ID if it's valid
        if name_from_id and name_from_id != 'N/A' and name_from_id.strip():
            chatbot_display_name = name_from_id
        # We can still attempt chat even if the name fallback is "this plant"
        can_chat = True

    # Check API Key availability AFTER determining if chat is possible
    if can_chat and (not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here"):
        st.warning("Chat feature requires a Gemini API key.")
        return # Stop if chat is desired but key is missing

    # Stop if we cannot determine a valid plant identity to chat with
    if not can_chat:
         st.warning("Cannot initialize chat without valid plant identification.")
         return

    st.subheader(f"💬 Chat with {chatbot_display_name}")

    # --- CSS Styling (Keep as is) ---
    st.markdown("""
        <style>
            /* ... Your existing CSS ... */
            .message-container { padding: 1px 5px; } /* Reduced vertical padding */
            .user-message { background: #0b81fe; color: white; border-radius: 18px 18px 0 18px; padding: 8px 14px; margin: 3px 0 3px auto; width: fit-content; max-width: 80%; word-wrap: break-word; box-shadow: 0 1px 2px rgba(0,0,0,0.1); animation: fadeIn 0.3s ease-out; }
            .bot-message { background: #e5e5ea; color: #000; border-radius: 18px 18px 18px 0; padding: 8px 14px; margin: 3px auto 3px 0; width: fit-content; max-width: 80%; word-wrap: break-word; box-shadow: 0 1px 2px rgba(0,0,0,0.05); animation: fadeIn 0.3s ease-out; }
            .message-meta { font-size: 0.70rem; color: #777; margin-top: 3px; } /* Reduced margin */
            .bot-message .message-meta { text-align: left; color: #555;}
            .user-message .message-meta { text-align: right; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
            .stChatInputContainer { position: sticky; bottom: 0; background: white; padding-top: 10px; }
        </style>
    """, unsafe_allow_html=True)

    # --- Chat Initialization/Reset Logic ---
    current_tracked_name = st.session_state.get("current_chatbot_plant_name")

    # Reset chat history if the plant name changes OR if chat history doesn't exist
    if "chat_history" not in st.session_state or current_tracked_name != chatbot_display_name:
        # If viewing saved details, try to load log, otherwise start fresh
        if st.session_state.get("viewing_saved_details"):
             saved_plant_nickname = st.session_state.viewing_saved_details
             saved_plant_data = st.session_state.saved_photos.get(saved_plant_nickname)
             # Ensure saved_plant_data exists before accessing 'chat_log'
             if saved_plant_data and 'chat_log' in saved_plant_data:
                 st.session_state.chat_history = saved_plant_data['chat_log']
                 print(f"DEBUG: Loaded chat log for saved plant '{saved_plant_nickname}'")
             else:
                 st.session_state.chat_history = [] # Start fresh if no log saved
                 print(f"DEBUG: Starting fresh chat log for saved plant '{saved_plant_nickname}' (no log found).")
        else:
            st.session_state.chat_history = [] # Start fresh for new ID or generic chat
            print(f"DEBUG: Starting fresh chat log for '{chatbot_display_name}'.")

        st.session_state.current_chatbot_plant_name = chatbot_display_name # Track the new name

    # --- Chat History Display using st.container ---
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.get("chat_history", []):
            role = message.get("role")
            content = message.get("content", "")
            time = message.get("time", "") # Get timestamp if available

            if role == "user":
                st.markdown(f'<div class="message-container"><div class="user-message">{content}<div class="message-meta">You • {time}</div></div></div>', unsafe_allow_html=True)
            elif role == "assistant" or role == "model": # Treat assistant/model the same for display
                st.markdown(f'<div class="message-container"><div class="bot-message">🌿 {content}<div class="message-meta">{chatbot_display_name} • {time}</div></div></div>', unsafe_allow_html=True)

    # --- Chat Input ---
    # Sanitize key more robustly
    safe_display_name = "".join(c if c.isalnum() else "_" for c in chatbot_display_name)
    prompt_key = f"chat_input_{safe_display_name}"
    if prompt := st.chat_input(f"Ask {chatbot_display_name}...", key=prompt_key):
        timestamp = datetime.now(EASTERN_TZ).strftime("%H:%M")
        st.session_state.chat_history.append({"role": "user", "content": prompt, "time": timestamp})
        st.rerun()

    # --- Process Bot Response ---
    if st.session_state.get("chat_history") and st.session_state.chat_history[-1].get("role") == "user":
        with st.spinner(f"{chatbot_display_name} is thinking..."):
            # **** IMPORTANT: Pass the specific arguments received by this function ****
            # These arguments reflect the intended state (either specific care or generic ID)
            bot_response = chat_with_plant(
                current_plant_care_info, # Use the care_info passed to THIS function call
                st.session_state.chat_history,
                plant_id_result # Use the id_result passed to THIS function call
            )

        timestamp = datetime.now(EASTERN_TZ).strftime("%H:%M")
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response, "time": timestamp})
        # Update chat log in saved photos immediately if viewing saved details
        if st.session_state.get("viewing_saved_details"):
            saved_nickname = st.session_state.viewing_saved_details
            if saved_nickname in st.session_state.saved_photos:
                st.session_state.saved_photos[saved_nickname]['chat_log'] = st.session_state.chat_history
        st.rerun()


# --- Main App Logic ---
def main():
    
    # --- Initialize State Variables ---
    defaults = {
        "plant_id_result": None, "plant_care_info": None, "chat_history": [],
        "current_chatbot_plant_name": None, "suggestions": None,
        "uploaded_file_bytes": None, "uploaded_file_type": None,
        "saving_mode": False, "last_view": "🏠 Home",  # Default to Home view
        "viewing_saved_details": st.session_state.get("viewing_saved_details", None),
        "plant_id_result_for_care_check": None, # Initialize care check tracker
        "suggestion_just_selected": False, # Flag for suggestion selection
        "viewing_plant_stats": None, # Initialize plant stats tracker
        "viewing_home_page": st.session_state.get("viewing_home_page", True) # Initialize home page view
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --- Sidebar Navigation and Saved Plants ---
    st.sidebar.title("📚 Plant Buddy")
    # Initialize saved photos in session state if not already present
    if "saved_photos" not in st.session_state: st.session_state.saved_photos = {}

    # Add Home to navigation options
    nav_choice_options = ["🏠 Home", "🆔 Identify New Plant", "🪴 My Saved Plants", "📊 Plant Stats"]
    nav_index = 0 # Default to Home page

    # Update navigation index based on current view
    if st.session_state.get("viewing_saved_details"):
        nav_index = 2  # My Saved Plants view
    elif st.session_state.get("viewing_plant_stats"):
        nav_index = 3  # Plant Stats view
    elif not st.session_state.get("viewing_home_page"):
        nav_index = 1  # Identify New Plant view

    # --- Main Navigation Radio Buttons ---
    nav_choice = st.sidebar.radio(
        "Navigation",
        nav_choice_options,
        key="main_nav_radio",
        index=nav_index, # Use the potentially updated nav_index
        label_visibility="collapsed" # Hide the "Navigation" label itself
    )
    st.sidebar.divider()
    st.sidebar.caption("Powered by PlantNet & Gemini")

    # --- Saved Plants Selector in Sidebar ---
    saved_plant_nicknames = list(st.session_state.saved_photos.keys())
    selected_saved_plant_sb = None # Initialize
    if saved_plant_nicknames:
        st.sidebar.subheader("Saved Plants")
        # Add a "-- Select --" option
        view_options = ["-- Select to View --"] + saved_plant_nicknames

        # Calculate the correct index based on the viewing state
        current_selection = st.session_state.get("viewing_saved_details")
        select_index = 0 # Default index (for "-- Select --")
        if current_selection and current_selection in view_options:
            try:
                select_index = view_options.index(current_selection)
            except ValueError:
                print(f"Warning: viewing_saved_details '{current_selection}' not found in view_options.")
                select_index = 0 # Fallback to default

        # Create the selectbox using the calculated index
        selected_saved_plant_sb = st.sidebar.selectbox(
            "View Saved Plant:",
            view_options,
            key="saved_view_selector",
            index=select_index # Set the index here!
        )

        # Handle USER interaction with the selectbox
        if selected_saved_plant_sb != "-- Select to View --":
            # If we're not in Plant Stats view, switch to Saved Plants view
            if not st.session_state.get("viewing_plant_stats"):
                nav_index = 2 # Switch navigation focus to Saved Plants page
            # Update the viewing state ONLY if the user selected something different
            if st.session_state.get("viewing_saved_details") != selected_saved_plant_sb:
                st.session_state.viewing_saved_details = selected_saved_plant_sb
                st.session_state.viewing_home_page = False  # Not on home page anymore
                # Rerun needed to load the selected plant's details in the main area
                st.rerun()
        else:
            # If user manually selected "-- Select --" AND we were previously viewing something
            if st.session_state.get("viewing_saved_details") is not None:
                 st.session_state.viewing_saved_details = None
                 # Force navigation index back to Identify View
                 nav_index = 0
                 # Rerun to clear the details view from the main page
                 st.rerun()

    # --- Check API Keys and Load Data ---
    api_keys_ok = True
    # Check the imported variables
    if not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here":
        st.warning("PlantNet API Key is not set. Using demo mode with limited functionality.")
        api_keys_ok = False
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        st.warning("Gemini API Key is not set. Chat functionality will be disabled.")
        # Don't set api_keys_ok to False here, identification can still work

    plant_care_data = load_plant_care_data()
    if not plant_care_data:
        # Error is shown in load_plant_care_data
        st.stop()

    # --- Main Content Area based on Navigation ---

    # ====================================
    # ===== Home Page View =====
    # ====================================
    if nav_choice == "🏠 Home":
        st.session_state.last_view = "🏠 Home"  # Track view
        st.session_state.viewing_home_page = True
        
        # Reset other view states
        if st.session_state.get("viewing_saved_details"):
            st.session_state.viewing_saved_details = None
        if st.session_state.get("viewing_plant_stats"):
            st.session_state.viewing_plant_stats = None
        
        st.header("🌿 Welcome to Plant Buddy!")
        
        # Display welcome message using Gemini
        welcome_message = {
            "role": "user",
            "content": "You are a plant assistant. Give a warm, friendly welcome to a user of the Plant Buddy app in 2-3 sentences. Mention that they can identify plants, get care tips, and track plant health."
        }
        
        if "welcome_response" not in st.session_state:
            with st.spinner("Loading welcome message..."):
                try:
                    system_prompt = """
                    CONTEXT: You are providing a short welcome message (2-3 sentences maximum).
                    TASK: Act as a friendly plant assistant welcoming users to the Plant Buddy app.
                    
                    RESPONSE RULES:
                    1. Keep it warm and friendly.
                    2. Mention plant identification, care tips, and health tracking features.
                    3. Keep responses very concise (2-3 sentences max).
                    """
                    
                    messages = [
                        {"role": "user", "parts": [{"text": system_prompt}]},
                        {"role": "model", "parts": [{"text": "I understand. I'll provide a warm, concise welcome message for Plant Buddy users."}]}
                    ]
                    
                    messages.append({"role": "user", "parts": [{"text": welcome_message["content"]}]})
                    
                    response = send_message(messages)
                    st.session_state.welcome_response = response
                except Exception as e:
                    st.session_state.welcome_response = "Welcome to Plant Buddy! Upload a plant photo to identify it, get care tips, and track its health. Happy gardening!"
        
        # Display the welcome message in a styled container
        st.markdown("""
        <div style="background-color: #f0f8f0; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
            <h3 style="color: #2E7D32;">🌱 Hello Plant Lover!</h3>
            <p style="font-size: 16px;">
        """ + st.session_state.welcome_response + """
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        st.subheader("🔍 What You Can Do")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔎 Identify Plants")
            st.markdown("Upload a photo of any plant to identify it using advanced AI technology.")
            if st.button("Start Identifying", use_container_width=True):
                st.session_state.viewing_home_page = False
                st.rerun()
        
        with col2:
            st.markdown("### 🪴 Track Your Plants")
            st.markdown("Save your plants and monitor their health and care requirements.")
            if st.button("View Saved Plants", use_container_width=True):
                st.session_state.viewing_home_page = False
                st.session_state.viewing_saved_details = None
                nav_index = 2  # My Saved Plants view
                st.rerun()
        
        # Display sample plants if available
        if st.session_state.saved_photos and len(st.session_state.saved_photos) > 0:
            st.divider()
            st.subheader("Your Plant Collection")
            
            # Display up to 3 saved plants
            saved_plant_nicknames = list(st.session_state.saved_photos.keys())[:3]
            num_columns = min(len(saved_plant_nicknames), 3)
            cols = st.columns(num_columns)
            
            for i, nickname in enumerate(saved_plant_nicknames):
                plant_data = st.session_state.saved_photos.get(nickname)
                if not plant_data:
                    continue
                
                with cols[i]:
                    with st.container(border=True):
                        # Display image
                        if plant_data.get("image"):
                            try:
                                st.image(plant_data["image"], use_container_width=True)
                            except Exception:
                                st.caption("Image error")
                        st.markdown(f"**{nickname}**")
                        
                        # Button to view details
                        safe_nickname_view = "".join(c if c.isalnum() else "_" for c in nickname)
                        view_card_key = f"home_view_{safe_nickname_view}"
                        
                        if st.button(f"View Details", key=view_card_key, use_container_width=True):
                            st.session_state.viewing_home_page = False
                            st.session_state.viewing_saved_details = nickname
                            st.rerun()

    # ====================================
    # ===== Identify New Plant View =====
    # ====================================
    elif nav_choice == "🆔 Identify New Plant":
        st.header("🔎 Identify a New Plant")
        
        # Update home page state
        st.session_state.viewing_home_page = False

        # --- State Reset Logic ---
        navigated_from_saved = st.session_state.last_view in ["🪴 My Saved Plants", "📊 Plant Stats"]
        # Reset if coming FROM saved view and NOT currently viewing saved details
        if navigated_from_saved and st.session_state.get("viewing_saved_details") is None:
            print("DEBUG: Resetting state -> Switched to Identify View")
            # Clear most session state related to a specific plant
            keys_to_reset = ["plant_id_result", "plant_care_info", "current_chatbot_plant_name",
                             "suggestions", "uploaded_file_bytes", "uploaded_file_type",
                             "chat_history", "saving_mode", "plant_id_result_for_care_check",
                             "suggestion_just_selected", "viewing_plant_stats"] # Added plant stats reset
            for key in keys_to_reset:
                if key in st.session_state: # Check if key exists before modifying
                   # Assign default values based on type
                   default_val = [] if key == "chat_history" else (False if key in ["saving_mode", "suggestion_just_selected"] else None)
                   st.session_state[key] = default_val

            st.session_state.pop('plant_uploader', None) # Clear potential file uploader state

        st.session_state.last_view = "🆔 Identify New Plant" # Update last view tracker


        # --- File Uploader ---
        uploaded_file = st.file_uploader(
            "Upload a clear photo of your plant:", type=["jpg", "jpeg", "png"],
            key="plant_uploader", # Consistent key
            help="Upload an image file (JPG, PNG).",
            on_change=lambda: st.session_state.update({ # Reset relevant state on *new file upload*
                 "plant_id_result": None, "plant_care_info": None, "chat_history": [],
                 "current_chatbot_plant_name": None, "suggestions": None,
                 "uploaded_file_bytes": None, "uploaded_file_type": None, # Clear previous bytes/type too
                 "saving_mode": False, "plant_id_result_for_care_check": None, # Also reset care check flag
                 "suggestion_just_selected": False, # Reset flag on new upload
                 "viewing_plant_stats": None # Reset plant stats view
            })
        )

        # --- Logic Based on Uploader State ---
        if uploaded_file is None:
             # Welcome message or keep showing existing results if user removed the file
             if st.session_state.uploaded_file_bytes is None:
                 st.info("Welcome to Plant Buddy! Upload a plant image above to get started, or select a saved plant from the sidebar.")
             elif st.session_state.plant_id_result is not None:
                 pass # Let the code below display the existing results
        else:
            # Process new upload
            if st.session_state.uploaded_file_bytes is None:
                st.session_state.uploaded_file_bytes = uploaded_file.getvalue()
                st.session_state.uploaded_file_type = uploaded_file.type
                st.rerun()

        # --- Display Image and Subsequent Info (if file bytes exist) ---
        if st.session_state.uploaded_file_bytes is not None:

            # Display Image
            try:
                display_image_with_max_height(
                    image_source=st.session_state.uploaded_file_bytes,
                    caption="Your Uploaded Plant",
                    max_height_px=400
                )
                st.divider()
            except Exception as e:
                st.error(f"Error displaying image: {e}")
                st.stop()

            # Run Identification if needed
            if st.session_state.plant_id_result is None:
                loader_placeholder = st.empty()
                with loader_placeholder.container():
                    st.markdown("<div style='text-align:center;'><p><i>Identifying plant...</i></p></div>", unsafe_allow_html=True)

                # Demo mode if API key not set
                if not api_keys_ok:
                    # Simulate API call with random plant from our sample data
                    import time
                    time.sleep(2)  # Simulate API delay
                    random_plant = random.choice(plant_care_data)
                    result = {
                        'scientific_name': random_plant.get('Scientific Name', 'Plantus demonstratus'),
                        'common_name': random_plant.get('Plant Name', 'Demo Plant'),
                        'confidence': random.uniform(70.0, 95.0)
                    }
                    st.session_state.plant_id_result = result
                    st.session_state.plant_id_result_for_care_check = None
                    st.session_state.suggestion_just_selected = False
                else:
                    try:
                        result = identify_plant(st.session_state.uploaded_file_bytes)
                        st.session_state.plant_id_result = result
                        st.session_state.plant_id_result_for_care_check = None # Reset care check flag after new ID
                        st.session_state.suggestion_just_selected = False # Ensure flag is False after new ID
                    except Exception as e:
                        st.session_state.plant_id_result = {'error': f"Identification process failed: {str(e)}"}
                
                loader_placeholder.empty()
                st.rerun()

            # Display results, care info, etc. (if ID is done)
            elif st.session_state.plant_id_result is not None:
                # Use the ID result directly from session state for consistency
                current_id_result_from_state = st.session_state.plant_id_result

                # --- Saving Mode UI ---
                if st.session_state.saving_mode:
                    # ... (Existing save mode UI using session state values directly) ...
                    st.header("💾 Save This Plant Profile")
                    if st.session_state.uploaded_file_bytes:
                        try:
                            st.image(Image.open(BytesIO(st.session_state.uploaded_file_bytes)), width=150, caption="Image to save")
                        except Exception:
                            st.warning("Could not display image preview for saving.")

                    id_info = st.session_state.get("plant_id_result", {})
                    sci_name = id_info.get('scientific_name', 'N/A')
                    com_name = id_info.get('common_name', 'N/A')
                    st.markdown(f"**Identified as:** {com_name} (\`{sci_name}\`)")

                    with st.form("save_form"):
                        save_nickname = st.text_input("Enter a nickname for this plant:", key="save_nickname_input")
                        submitted = st.form_submit_button("✅ Confirm Save")
                        if submitted:
                            if not save_nickname:
                                st.warning("Please enter a nickname to save.")
                            elif save_nickname in st.session_state.saved_photos:
                                st.warning(f"A plant named '{save_nickname}' already exists. Please choose a different name.")
                            else:
                                try:
                                    encoded_img = base64.b64encode(st.session_state.uploaded_file_bytes).decode()
                                    data_url = f"data:{st.session_state.uploaded_file_type};base64,{encoded_img}"
                                    # Save current state (care info might be None)
                                    st.session_state.saved_photos[save_nickname] = {
                                        "nickname": save_nickname, "image": data_url,
                                        "id_result": st.session_state.plant_id_result,
                                        "care_info": st.session_state.plant_care_info, # Save None if not found
                                        "chat_log": st.session_state.get("chat_history", []), # Save current chat
                                        "moisture_level": random.randint(30, 90) # Add simulated moisture level
                                    }
                                    # Clear state *after* successful save
                                    keys_to_reset = ["plant_id_result", "plant_care_info", "current_chatbot_plant_name",
                                                     "suggestions", "uploaded_file_bytes", "uploaded_file_type",
                                                     "chat_history", "saving_mode", "plant_id_result_for_care_check",
                                                     "suggestion_just_selected", "viewing_plant_stats"] # Added plant stats reset
                                    for key in keys_to_reset:
                                        if key in st.session_state:
                                            default_val = [] if key == "chat_history" else (False if key in ["saving_mode", "suggestion_just_selected"] else None)
                                            st.session_state[key] = default_val

                                    st.session_state.pop('plant_uploader', None)

                                    st.success(f"Successfully saved '{save_nickname}'!")
                                    st.balloons()
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error saving plant profile: {e}")

                    if st.button("❌ Cancel Save", key="cancel_save_button"):
                        st.session_state.saving_mode = False
                        st.rerun()


                # --- Normal Display (Not Saving) ---
                else:
                    display_identification_result(current_id_result_from_state) # Use ID from state

                    if 'error' not in current_id_result_from_state:
                        # Check if care info needs update (using flag)
                        care_info_state = st.session_state.get('plant_care_info')
                        id_result_state_for_check = st.session_state.get('plant_id_result_for_care_check')
                        suggestion_was_selected = st.session_state.get('suggestion_just_selected', False)

                        needs_care_update = False
                        # Update ONLY IF a suggestion was NOT just selected AND state needs refresh
                        if not suggestion_was_selected and \
                           ((care_info_state is None and current_id_result_from_state != id_result_state_for_check) or \
                           (current_id_result_from_state != id_result_state_for_check)):
                            needs_care_update = True

                        if needs_care_update:
                            # This block now only runs if triggered by a new upload/ID change,
                            # NOT immediately after a suggestion click.
                            print(f"DEBUG: Finding/updating care instructions for ID: {current_id_result_from_state}")
                            found_care = find_care_instructions(current_id_result_from_state, plant_care_data)
                            st.session_state.plant_care_info = found_care # Set to None if not found
                            st.session_state.plant_id_result_for_care_check = current_id_result_from_state # Store ID used for this check

                            if found_care is not None:
                                print("DEBUG: Care info found, resetting suggestions and chat, rerunning.")
                                st.session_state.suggestions = None
                                st.session_state.chat_history = []
                                st.session_state.current_chatbot_plant_name = None
                                st.rerun() # Rerun ONLY if we found valid care info
                            else:
                                print("DEBUG: Care info not found for this ID. Proceeding without rerun.")
                                st.session_state.suggestions = None # Reset suggestions so the 'else' block generates them

                        # IMPORTANT: Reset the flag AFTER the check, so it only affects one rerun cycle
                        if suggestion_was_selected:
                             print("DEBUG: Resetting suggestion_just_selected flag.")
                             st.session_state.suggestion_just_selected = False


                        # ===========================================================
                        # ===== START: ENSURE CORRECT DATA PASSED (Main Change Here) =====
                        # ===========================================================
                        # Get the most up-to-date care_info AFTER potential updates/checks
                        # Use .get() to safely retrieve from state
                        care_info_to_display = st.session_state.get('plant_care_info')
                        id_result_to_display = st.session_state.get('plant_id_result') # Use the ID currently in state

                        # --- Case 1: Care Info FOUND (or just selected via suggestion) ---
                        if care_info_to_display:
                            display_care_instructions(care_info_to_display)
                            st.divider()
                            if st.button("💾 Save Plant Profile", key="save_profile_button"):
                                st.session_state.saving_mode = True; st.rerun()
                            st.divider()
                            # **** Pass the retrieved 'care_info_to_display' and 'id_result_to_display' ****
                            display_chat_interface(current_plant_care_info=care_info_to_display, plant_id_result=id_result_to_display)

                        # --- Case 2: Care Info NOT Found ---
                        else:
                            st.warning("Could not find specific care instructions or personality profile for this exact plant in our database.")

                            if st.session_state.suggestions is None:
                                print("DEBUG: Suggestions are None, generating...")
                                st.session_state.suggestions = find_similar_plant_matches(id_result_to_display, plant_care_data) # Use current ID from state
                                print("DEBUG: Rerunning to display suggestions.")
                                # Avoid potential infinite loop if find_similar_plant_matches keeps returning empty list
                                if st.session_state.suggestions is not None:
                                    st.rerun()
                                else: # If suggestions are still None (e.g., error or no matches), don't rerun
                                    print("WARN: Suggestion generation resulted in None, not rerunning.")


                            display_suggestion_buttons(st.session_state.suggestions)
                            st.divider()
                            if st.button("💾 Save Identification Only", key="save_id_only_button"):
                                st.session_state.saving_mode = True; st.rerun()
                            st.divider()
                            st.info("You can still chat with the plant based on its general identification.")
                            # **** Pass None for care_info and the 'id_result_to_display' ****
                            display_chat_interface(current_plant_care_info=None, plant_id_result=id_result_to_display)
                        #  plant_id_result=id_result_to_display)
                        # ===========================================================
                        # ===== END: ENSURE CORRECT DATA PASSED =====
                        # ===========================================================

                    else: # Handle case where identification itself failed
                        pass


    # ====================================
    # ===== Saved Plants View =====
    # ====================================
    elif nav_choice == "🪴 My Saved Plants":
        st.header("🪴 My Saved Plant Profiles")
        st.session_state.last_view = "🪴 My Saved Plants" # Track view
        
        # Update home page state
        st.session_state.viewing_home_page = False
        
        # Reset plant stats view when navigating to saved plants
        if st.session_state.get("viewing_plant_stats"):
            st.session_state.viewing_plant_stats = None

        saved_plant_nicknames = list(st.session_state.saved_photos.keys())
        nickname_to_view = st.session_state.get("viewing_saved_details")

        if not saved_plant_nicknames:
            st.info("You haven't saved any plants yet. Go to 'Identify New Plant' to add some!")
        # If a specific plant IS selected for viewing:
        elif nickname_to_view and nickname_to_view in st.session_state.saved_photos:
             # Add back button
             if st.button("← Back to All Plants", key="back_to_all_plants"):
                 st.session_state.viewing_saved_details = None
                 st.rerun()
                 
             st.subheader(f"Showing Details for: '{nickname_to_view}'")
             entry = st.session_state.saved_photos[nickname_to_view]

             # Display saved image
             if entry.get("image"):
                 try:
                     display_image_with_max_height(entry["image"], caption=f"{nickname_to_view}", max_height_px=400)
                 except Exception as e:
                     st.error(f"Error displaying saved image: {e}")
             else:
                 st.caption("No image saved.")
             st.divider()

             # Display saved ID result (important for chat fallback)
             saved_id_result = entry.get("id_result")
             if saved_id_result:
                 display_identification_result(saved_id_result)
                 # Update session state id_result if needed for chat context consistency
                 if st.session_state.plant_id_result != saved_id_result:
                     st.session_state.plant_id_result = saved_id_result
                     st.session_state.plant_id_result_for_care_check = saved_id_result # Update check flag too
                     st.session_state.suggestion_just_selected = False # Ensure flag is reset
             else:
                 st.info("No identification details were saved.")
                 # Clear session state id_result if none saved here
                 if st.session_state.plant_id_result is not None:
                     st.session_state.plant_id_result = None
                     st.session_state.plant_id_result_for_care_check = None # Clear check flag
                     st.session_state.suggestion_just_selected = False # Ensure flag is reset

             st.divider()

             # --- Display care info if saved ---
             saved_care_info = entry.get("care_info")
             # Update session state care_info for chat context consistency
             if st.session_state.plant_care_info != saved_care_info:
                 st.session_state.plant_care_info = saved_care_info
                 st.session_state.suggestion_just_selected = False # Reset flag when loading saved details

             # --- Add "Check Plant Stats" button ---
             col1, col2 = st.columns([1, 1])
             with col1:
                 if st.button(f"📊 Check Plant Stats", key=f"stats_{nickname_to_view}", use_container_width=True):
                     st.session_state.viewing_plant_stats = nickname_to_view
                     st.rerun()

             # --- Display Care Instructions and Chat ---
             if saved_care_info:
                 display_care_instructions(saved_care_info)
                 st.divider()
                 # Call chat interface with BOTH care_info and id_result
                 display_chat_interface(current_plant_care_info=saved_care_info, plant_id_result=saved_id_result)
             else:
                 # Handle saved profiles that only have ID (saved using "Save ID Only")
                 st.info("No specific care instructions were saved for this plant.")
                 st.divider()
                 # Allow generic chat based on the saved ID if available
                 if saved_id_result:
                     st.info("You can chat with the plant based on its saved identification.")
                     # Call chat interface with NO care_info but WITH id_result
                     display_chat_interface(current_plant_care_info=None, plant_id_result=saved_id_result)
                 else:
                     # This case should be rare if saving ID always works
                     st.warning("No identification details found, cannot initiate chat.")


             # --- Delete Button ---
             st.divider()
             # Sanitize key more robustly
             safe_nickname_del = "".join(c if c.isalnum() else "_" for c in nickname_to_view)
             delete_key = f"del_{safe_nickname_del}"
             if st.button(f"🗑️ Delete '{nickname_to_view}' Profile", key=delete_key, use_container_width=False):
                 del st.session_state.saved_photos[nickname_to_view]
                 st.session_state.viewing_saved_details = None
                 # Clear related state variables
                 keys_to_reset = ["plant_id_result", "plant_care_info", "current_chatbot_plant_name",
                                  "suggestions", "uploaded_file_bytes", "uploaded_file_type",
                                  "chat_history", "saving_mode", "plant_id_result_for_care_check",
                                  "suggestion_just_selected", "viewing_plant_stats"] # Added plant stats reset
                 for key in keys_to_reset:
                     if key in st.session_state:
                         default_val = [] if key == "chat_history" else (False if key in ["saving_mode", "suggestion_just_selected"] else None)
                         st.session_state[key] = default_val

                 st.success(f"Deleted '{nickname_to_view}'.")
                 st.rerun()

        # If NO specific plant is selected (overview):
        else:
            # ... (Overview grid logic - Keep as is) ...
            st.info("Select a plant from the 'View Saved Plant' dropdown in the sidebar to see its details.")
            st.markdown("---")
            st.subheader("All Saved Plants Overview")
            # Display Grid of Information Cards
            num_columns = 3
            cols = st.columns(num_columns)
            col_index = 0

            for nickname in saved_plant_nicknames:
                plant_data = st.session_state.saved_photos.get(nickname)
                if not plant_data: continue

                with cols[col_index % num_columns]:
                    with st.container(border=True):
                        # Display image
                        if plant_data.get("image"):
                            try: st.image(plant_data["image"], use_container_width=True)
                            except Exception: st.caption("Image error")
                        st.markdown(f"**{nickname}**") # Nickname

                        id_res = plant_data.get("id_result", {})
                        com_n = id_res.get('common_name', 'N/A')
                        # Display common name only if it's not N/A or empty
                        if com_n and com_n != 'N/A': st.caption(f"{com_n}")

                        # Button to view full details
                        # Sanitize key more robustly
                        safe_nickname_view = "".join(c if c.isalnum() else "_" for c in nickname)
                        view_card_key = f"view_card_{safe_nickname_view}"
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"View Details", key=view_card_key, use_container_width=True):
                                st.session_state.viewing_saved_details = nickname
                                st.rerun() # Rerun to show details view and update selectbox index
                        with col2:
                            stats_card_key = f"stats_card_{safe_nickname_view}"
                            if st.button(f"📊 Stats", key=stats_card_key, use_container_width=True):
                                st.session_state.viewing_plant_stats = nickname
                                st.rerun()
                col_index += 1

    # ====================================
    # ===== Plant Stats View =====
    # ====================================
    elif nav_choice == "📊 Plant Stats":
        st.session_state.last_view = "📊 Plant Stats" # Track view
        
        # Update home page state
        st.session_state.viewing_home_page = False
        
        # Get the plant nickname from session state
        plant_nickname = st.session_state.get("viewing_plant_stats")
        
        if not plant_nickname or plant_nickname not in st.session_state.saved_photos:
            st.warning("No plant selected for stats view. Please select a plant from your saved plants.")
            if st.button("← Back to Saved Plants"):
                st.session_state.viewing_plant_stats = None
                st.rerun()
        else:
            # Get plant data - ensure we're getting the latest data from the session state
            plant_data = st.session_state.saved_photos[plant_nickname]
            
            # Display plant stats
            st.header(f"📊 Plant Stats: {plant_nickname}")
            
            # Back button
            if st.button("← Back to Saved Plants"):
                st.session_state.viewing_plant_stats = None
                st.rerun()
                
            st.divider()
            
            # Display plant image
            if plant_data.get("image"):
                try:
                    display_image_with_max_height(
                        plant_data["image"],
                        caption=f"{plant_nickname}",
                        max_height_px=300
                    )
                except Exception as e:
                    st.error(f"Error displaying plant image: {e}")
            
            # Display plant information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Plant Information")
                
                # Get ID result
                id_result = plant_data.get("id_result", {})
                scientific_name = id_result.get("scientific_name", "N/A")
                common_name = id_result.get("common_name", "N/A")
                
                st.markdown(f"**Given Name:** {plant_nickname}")
                st.markdown(f"**Scientific Name:** {scientific_name}")
                st.markdown(f"**Common Name:** {common_name}")
            
            with col2:
                st.subheader("Environmental Stats")
                
                # Display moisture level with progress bar
                moisture_level = plant_data.get("moisture_level", random.randint(30, 90))
                st.markdown("**Moisture Level:**")
                
                # Determine moisture status and color
                if moisture_level < 30:
                    moisture_status = "Low - Water needed!"
                    moisture_color = "red"
                elif moisture_level < 60:
                    moisture_status = "Moderate"
                    moisture_color = "orange"
                else:
                    moisture_status = "Good"
                    moisture_color = "green"
                
                st.progress(moisture_level/100)
                st.markdown(f"<span style='color:{moisture_color};'>{moisture_level}% ({moisture_status})</span>", unsafe_allow_html=True)
                
                # Display temperature information
                st.markdown("**Temperature:**")
                care_info = plant_data.get("care_info", {})
                temp_range = care_info.get("Temperature Range", "Not specified") if care_info else "Not specified"
                st.markdown(f"{temp_range}")
            
            st.divider()
            
            # Display care tips
            if care_info:
                st.subheader("Care Tips")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Light Requirements:**")
                    st.caption(f"{care_info.get('Light Requirements', 'Not specified')}")
                    
                    st.markdown("**Watering Needs:**")
                    st.caption(f"{care_info.get('Watering', 'Not specified')}")
                
                with col2:
                    st.markdown("**Humidity Preferences:**")
                    st.caption(f"{care_info.get('Humidity Preferences', 'Not specified')}")
                    
                    st.markdown("**Feeding Schedule:**")
                    st.caption(f"{care_info.get('Feeding Schedule', 'Not specified')}")
                
                # Display additional care tips if available
                additional_care = care_info.get('Additional Care')
                if additional_care and isinstance(additional_care, str) and additional_care.strip():
                    with st.expander("Additional Care Tips", expanded=True):
                        st.markdown(additional_care)
            else:
                st.info("No specific care information available for this plant.")


# --- Run the App ---
if __name__ == "__main__":
    # Add a check for API keys loaded from config
    if not PLANTNET_API_KEY or PLANTNET_API_KEY == "your_plantnet_api_key_here":
        st.warning("PlantNet API Key is not set. Using demo mode with limited functionality.")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        st.warning("Gemini API Key is not set. Chat will be disabled.")
        # Allow app to run without Gemini for ID/Care lookup

    main()
