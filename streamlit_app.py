import streamlit as st
from PIL import Image
import os
import json
import tempfile
from fuzzywuzzy import process
from plant_net import PlantNetAPI
from plant_chatbot import PlantChatbot
from dotenv import load_dotenv
from openai import OpenAI
# Load environment variables
load_dotenv()

def main():
    st.set_page_config(
        page_title="Plant Identifier",
        page_icon="🌿",
        layout="centered"
    )
    
    st.title("🌿 Smart Plant Identifier")
    st.markdown("Upload a plant photo to get care instructions and chat with your plant!")

    # Initialize services
    plant_care_data = load_plant_care_data()
    plantnet = PlantNetAPI(os.getenv("PLANTNET_API_KEY"))

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a plant image (leaves, flowers, or fruits work best)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        process_uploaded_image(uploaded_file, plantnet, plant_care_data)

    # Sidebar information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app helps you:
        - Identify unknown plants
        - Get care instructions
        - Chat with your plant's personality
        """)

def load_plant_care_data():
    """Load plant care instructions from JSON file."""
    try:
        with open('plant_care_instructions.json') as f:
            data = json.load(f)
            if not isinstance(data, list):
                st.error("Invalid data format in care instructions")
                return []
            return data
    except Exception as e:
        st.error(f"Failed to load care instructions: {str(e)}")
        return []

def process_uploaded_image(uploaded_file, plantnet, plant_care_data):
    """Handle the image upload and processing pipeline."""
    try:
        # Display uploaded image
        with st.spinner("Analyzing your plant..."):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True, caption="Your Plant")

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name

            # Identify plant
            result = plantnet.identify_plant(temp_path)
            
            with col2:
                if 'error' in result:
                    st.error(result['error'])
                    return

                display_identification_result(result)
                handle_care_instructions(result, plant_care_data)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

def display_identification_result(result):
    """Display the plant identification results."""
    st.subheader("🔍 Identification Results")
    
    confidence = result.get('confidence', 0)
    confidence_color = "green" if confidence > 75 else "orange" if confidence > 50 else "red"
    
    st.markdown(f"""
    - 🧪 **Scientific Name**: `{result.get('scientific_name', 'Unknown')}`
    - 🌎 **Common Name**: `{result.get('common_name', 'Unknown')}`
    - 🎯 **Confidence**: <span style='color:{confidence_color}'>{confidence}%</span>
    """, unsafe_allow_html=True)

def handle_care_instructions(result, plant_care_data):
    """Find and display care instructions for the identified plant."""
    # Try scientific name first, then common name
    care_info = find_care_instructions(result['scientific_name'], plant_care_data)
    if not care_info and result.get('common_name'):
        care_info = find_care_instructions(result['common_name'], plant_care_data)

    if care_info:
        display_care_instructions(care_info)
        initialize_chatbot(care_info)
    else:
        st.warning("No complete care instructions found for this plant")
        suggest_similar_plants(result, plant_care_data)

def find_care_instructions(plant_name, care_data):
    """Find care instructions using exact, partial, and fuzzy matching."""
    if not plant_name or not care_data:
        return None
    
    plant_name_lower = plant_name.lower().strip()

    # 1. Try exact match
    for plant in care_data:
        if plant_name_lower == plant.get('Plant Name', '').lower().strip():
            return plant

    # 2. Try partial matches
    partial_matches = []
    for plant in care_data:
        db_name = plant.get('Plant Name', '').lower()
        if (plant_name_lower in db_name or 
            db_name in plant_name_lower or
            any(word in db_name for word in plant_name_lower.split())):
            partial_matches.append(plant)
    
    if len(partial_matches) == 1:
        return partial_matches[0]

    # 3. Try fuzzy matching
    all_plant_names = []
    plant_map = {}
    for plant in care_data:
        name = plant.get('Plant Name', '').lower().strip()
        if name:
            all_plant_names.append(name)
            plant_map[name] = plant
    
    if all_plant_names:
        best_match, score = process.extractOne(plant_name_lower, all_plant_names)
        if score > 65:  # Only use good matches
            return plant_map.get(best_match)
    
    # 4. Return first partial match if exists
    return partial_matches[0] if partial_matches else None

def display_care_instructions(care_info):
    """Display the care instructions in an organized layout."""
    st.subheader(f"🌱 {care_info['Plant Name']} Care Guide")
    
    with st.expander("📋 Care Summary", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"""
            **☀️ Light**  
            {care_info.get('Light Requirements', 'Not specified')}
            
            **💧 Water**  
            {care_info.get('Watering', 'Not specified')}
            
            **🌡️ Temperature**  
            {care_info.get('Temperature Range', 'Not specified')}
            """)
        
        with cols[1]:
            st.markdown(f"""
            **💦 Humidity**  
            {care_info.get('Humidity Preferences', 'Not specified')}
            
            **🍃 Feeding**  
            {care_info.get('Feeding Schedule', 'Not specified')}
            
            **⚠️ Toxicity**  
            {care_info.get('Toxicity', 'Not specified')}
            """)
    
    if care_info.get('Additional Care'):
        with st.expander("✨ Pro Tips"):
            st.markdown(care_info['Additional Care'])

def suggest_similar_plants(result, plant_care_data):
    """Suggest similar plants when exact match isn't found."""
    st.info("🌿 Try these similar plants:")
    
    # Create list of all plant names for matching
    all_plant_names = []
    plant_map = {}
    for plant in plant_care_data:
        name = plant.get('Plant Name', '').lower().strip()
        if name:
            all_plant_names.append(name)
            plant_map[name] = plant
    
    # Find matches for scientific name
    if result.get('scientific_name'):
        matches = process.extract(
            result['scientific_name'].lower(), 
            all_plant_names, 
            limit=3
        )
        display_plant_matches(matches, plant_map)
    
    # Find matches for common name if no scientific matches
    if result.get('common_name') and not st.session_state.get('shown_suggestions'):
        matches = process.extract(
            result['common_name'].lower(), 
            all_plant_names, 
            limit=3
        )
        display_plant_matches(matches, plant_map)
        st.session_state.shown_suggestions = True

def display_plant_matches(matches, plant_map):
    """Display matched plants as selectable cards."""
    for match, score in matches:
        if score > 50:  # Only show decent matches
            plant = plant_map.get(match)
            if plant:
                with st.expander(f"🌱 {plant.get('Plant Name', 'Unknown')}"):
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"""
                        **☀️ Light**  
                        {plant.get('Light Requirements', 'Not specified')}
                        
                        **💧 Water**  
                        {plant.get('Watering', 'Not specified')}
                        """)
                    with cols[1]:
                        st.markdown(f"""
                        **🌡️ Temp**  
                        {plant.get('Temperature Range', 'Not specified')}
                        
                        **⚠️ Toxicity**  
                        {plant.get('Toxicity', 'Not specified')}
                        """)
                    
                    if st.button("Select this plant", 
                               key=f"select_{plant.get('Plant Name', '')}"):
                        display_care_instructions(plant)
                        initialize_chatbot(plant)
                        st.rerun()

def initialize_chatbot(care_info):
    """Initialize and manage the chatbot session"""
    st.divider()
    st.subheader(f"💬 Chat with {care_info['Plant Name']}")
    
    # Initialize chatbot if not already in session state
    if "plant_chatbot" not in st.session_state:
        st.session_state.plant_chatbot = PlantChatbot(care_info)
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle new messages
    if prompt := st.chat_input(f"Ask {care_info['Plant Name']}..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        bot_response = st.session_state.plant_chatbot.respond(prompt)
        
        # Add and display bot response
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant", avatar="🌿"):
            st.markdown(bot_response)
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()