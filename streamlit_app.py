import streamlit as st
import os
import time
import uuid
import shutil
from PIL import Image
import io
import base64
import datetime
import google.generativeai as genai
from plantcv import plantcv as pcv
import numpy as np

# --- SETUP ---
DEMO_IMAGE = "demo_plant.jpg"
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-001",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# --- PLANT ID FUNCTION ---
def identify_plant(image_bytes):
    try:
        # Encode the image bytes to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare the prompt with the base64 encoded image
        prompt_parts = [
            """You are an expert botanist. A user will upload an image of a plant, and you will identify the species of the plant.
            You will also provide a brief overview of the plant, including its native region, typical size, and growing season.
            Finally, you will provide detailed care instructions for the plant, including sunlight, watering, soil, and fertilization requirements.
            If the image does not contain a plant, or the plant is not identifiable, respond that you are unable to identify the plant.
            Do not mention that you are using PlantCV.
            """,
            {"mime_type": "image/jpeg", "data": image_base64},
        ]

        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"Error identifying plant: {e}"

# --- PLANT HEALTH FUNCTION ---
def assess_plant_health(image_bytes):
    try:
        # Decode the image bytes to a NumPy array
        image_array = np.array(Image.open(io.BytesIO(image_bytes)))

        # Run the PlantCV health analysis
        health_data = pcv.analyze_health(img=image_array, mask=None, debug="none")

        # Extract the health score from the analysis
        health_score = health_data[0]  # Assuming the first element is the health score

        # Prepare the response
        response_text = f"Plant Health Score: {health_score:.2f}\n"
        response_text += "Based on this score, here's a general assessment:\n"

        if health_score > 75:
            response_text += "Your plant appears to be very healthy! Keep up the good work."
        elif 50 <= health_score <= 75:
            response_text += "Your plant is moderately healthy. Monitor it closely and adjust care as needed."
        else:
            response_text += "Your plant shows signs of stress or disease. Investigate and take action to improve its health."

        return response_text
    except Exception as e:
        return f"Error assessing plant health: {e}"

# --- GEMINI MESSAGE FUNCTION ---
def send_message(messages):
    try:
        response = model.generate_content(messages)
        return response.parts[0].text
    except Exception as e:
        return f"Error: {e}"

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Plant Buddy", page_icon="🌿")

    # --- Initialize State Variables ---
    defaults = {
        "image_bytes": None,
        "plant_details": None,
        "health_details": None,
        "saved_photos": {},
        "viewing_saved_details": None,
        "viewing_plant_stats": None,
        "last_view": "🏠 Home",
        "viewing_home_page": st.session_state.get("viewing_home_page", True),
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- Sidebar ---
    st.sidebar.title("📚 Plant Buddy")

    # Initialize viewing_home_page in session state if not already present
    if "viewing_home_page" not in st.session_state:
        st.session_state.viewing_home_page = True  # Default to home page on first load

    # Add Home to navigation options
    nav_choice_options = ["🏠 Home", "🆔 Identify New Plant", "🪴 My Saved Plants", "📊 Plant Stats"]
    nav_index = 0  # Default to Home page

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
        index=nav_index,
        label_visibility="collapsed"
    )
    st.sidebar.divider()
    st.sidebar.caption("Powered by PlantNet & Gemini")

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

        uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Plant Image", use_column_width=True)

            # Convert the image to bytes
            image_bytes = uploaded_file.read()
            st.session_state.image_bytes = image_bytes  # Store in session state

            # Plant Identification
            if st.button("Identify Plant"):
                with st.spinner("Identifying plant..."):
                    plant_details = identify_plant(image_bytes)
                    st.session_state.plant_details = plant_details  # Store in session state

            # Plant Health Assessment
            if st.button("Assess Plant Health"):
                with st.spinner("Assessing plant health..."):
                    health_details = assess_plant_health(image_bytes)
                    st.session_state.health_details = health_details  # Store in session state

            # Display Plant Details
            if st.session_state.plant_details:
                st.subheader("🌱 Plant Identification")
                st.write(st.session_state.plant_details)

            # Display Health Details
            if st.session_state.health_details:
                st.subheader("💪 Plant Health Assessment")
                st.write(st.session_state.health_details)

            # Save Photo Feature
            if st.session_state.image_bytes and st.session_state.plant_details:
                nickname = st.text_input("Give this plant a nickname:")
                if nickname:
                    if st.button("Save Plant"):
                        # Prepare data to save
                        plant_data = {
                            "image": image,
                            "plant_details": st.session_state.plant_details,
                            "health_details": st.session_state.health_details,
                            "date_saved": datetime.datetime.now().isoformat()
                        }
                        st.session_state.saved_photos[nickname] = plant_data
                        st.success(f"Plant '{nickname}' saved!")

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

             # Display image
             if entry.get("image"):
                 try:
                     st.image(entry["image"], use_column_width=True)
                 except Exception:
                     st.caption("Image error")

             # Display plant details
             if entry.get("plant_details"):
                 st.subheader("🌱 Plant Identification")
                 st.write(entry["plant_details"])

             # Display health details
             if entry.get("health_details"):
                 st.subheader("💪 Plant Health Assessment")
                 st.write(entry["health_details"])

             # Button to view plant stats
             if st.button("📊 View Plant Stats", key=f"stats_{nickname_to_view}"):
                 st.session_state.viewing_plant_stats = nickname_to_view
                 st.rerun()

        # If NO specific plant is selected for viewing:
        else:
            st.subheader("Your Saved Plants:")
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
                        view_card_key = f"view_{safe_nickname_view}"

                        if st.button(f"View Details", key=view_card_key, use_container_width=True):
                            st.session_state.viewing_saved_details = nickname
                            st.rerun()

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

            # Display image
            if plant_data.get("image"):
                try:
                    st.image(plant_data["image"], use_column_width=True)
                except Exception:
                    st.caption("Image error")

            # Display plant details
            if plant_data.get("plant_details"):
                st.subheader("🌱 Plant Identification")
                st.write(plant_data["plant_details"])

            # Display health details
            if plant_data.get("health_details"):
                st.subheader("💪 Plant Health Assessment")
                st.write(plant_data["health_details"])

            # Basic Stats Display (replace with actual stats later)
            st.subheader("💧 Watering Frequency")
            st.write("Recommended: Once a week")

            st.subheader("☀️ Sunlight Exposure")
            st.write("Recommended: 6-8 hours of direct sunlight")

            st.subheader("🌡️ Ideal Temperature")
            st.write("Recommended: 65-80°F (18-27°C)")

if __name__ == "__main__":
    main()
