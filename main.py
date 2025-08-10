import streamlit as st
from model import AIAgent
from gtts import gTTS
import io


# Initialize the AI agent
# This will create an instance of your AIAgent class from model.py
@st.cache_resource
def load_agent():
    return AIAgent(llm_type="auto")


agent = load_agent()


hide_st_style = """
                    <style>
                    MainMenu {visibility: hidden;}
                    headerNoPadding {visibility: hidden;}
                    _terminalButton_rix23_138 {visibility: hidden;}
                    header {visibility: hidden;}
                    </style>
                    """
st.markdown(hide_st_style, unsafe_allow_html=True)
# --- Text-to-Speech Function ---
def text_to_speech(text):
    """Converts text to speech and returns the audio data as bytes."""
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang='en-in',slow=False)

        # Save audio to an in-memory file
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)

        return audio_fp.read()
    except Exception as e:
        st.error(f"Could not generate audio. Error: {e}")
        return None


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Lavi",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)
welcome_msg = """Hello! I'm your Lavi. I can help with:
- 🔍 Web searches
- 🧮 Mathematical calculations
- 📊 Technology explanations
- ⚙️ System automation

Try asking:
- "What is quantum computing?"
- "Calculate 2^8 + sqrt(16)"
- "Search for GTA 6 release date"
- "Open calculator"
"""
# --- Sidebar ---

# --- Session State for Audio ---
if "audio_to_play" not in st.session_state:
    st.session_state.audio_to_play = None

# --- Page Title and Subtitle ---
st.title("Lavi AI Agent ")
st.caption("Your friendly AI assistant with Google Search, Calculator & System Control")
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    st.session_state.audio_to_play = None
    st.rerun()

auto_play_audio = st.toggle("Auto-play Audio", value=False, help="Automatically play the assistant's response.")

# --- Initialize Chat History ---
# We use st.session_state to keep the chat history persistent across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a welcome message to the chat
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# --- Display Chat History ---
# This loop goes through all the messages in the session state and displays them
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if st.button("🔊", key=f"tts_{i}", help="Read this response aloud"):
                audio_bytes = text_to_speech(message["content"][1:])
                if audio_bytes:
                    st.session_state.audio_to_play = audio_bytes

# --- User Input ---
# This creates a text input at the bottom of the page for the user to type their message
if prompt := st.chat_input("What can I help you with?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Clear any previously played audio
    st.session_state.audio_to_play = None
    # Rerun to display the new message immediately
    st.rerun()


# --- Generate and Display Assistant Response if last message was from user ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.chat(user_prompt)
            st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Auto-play audio if toggled
            if auto_play_audio:
                audio_bytes = text_to_speech(response)
                if audio_bytes:
                    st.session_state.audio_to_play = audio_bytes

            # Rerun to clear the input box and update the chat
            st.rerun()

# --- Play Audio ---
# This will display the audio player if there is audio to be played
if st.session_state.audio_to_play:
    st.audio(st.session_state.audio_to_play, autoplay=True)
    # Clear the audio after playing so it doesn't replay on every interaction
    if not auto_play_audio:
        st.session_state.audio_to_play = None
