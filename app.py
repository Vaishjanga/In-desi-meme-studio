import streamlit as st
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import requests
import psycopg2
from psycopg2 import pool
import datetime
import random
import re
from packaging import version
from dotenv import load_dotenv
import tempfile
import warnings
from typing import Optional, Dict, List

# Import utils and config properly
from utils import (
    DesiMemeAPIClient, 
    validate_email, 
    validate_password,
    validate_phone_number,
    show_success_message, 
    show_error_message,
    detect_language_script,
    generate_meme_filename,
    validate_file_upload,
    format_file_size
)
from config import settings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Initialize translation model (keeping the original functionality)
translator_model = None
translator_tokenizer = None

@st.cache_resource
def load_translation_model():
    """Load Hugging Face English to Telugu translation model"""
    global translator_model, translator_tokenizer
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        
        st.info("Loading English-Telugu translation model...")
        
        try:
            translator_pipeline = pipeline("text2text-generation", model="Meher2006/english-to-telugu-model")
            st.success("Translation model loaded successfully using text2text pipeline!")
            return translator_pipeline, None, None
        except Exception as e1:
            st.warning(f"Text2text pipeline failed: {str(e1)}")
            
            try:
                translator_pipeline = pipeline("translation", model="Meher2006/english-to-telugu-model")
                st.success("Translation model loaded successfully using translation pipeline!")
                return translator_pipeline, None, None
            except Exception as e2:
                st.warning(f"Translation pipeline failed: {str(e2)}")
                
                try:
                    st.info("Trying direct model loading...")
                    tokenizer = AutoTokenizer.from_pretrained("Meher2006/english-to-telugu-model")
                    model = AutoModelForSeq2SeqLM.from_pretrained("Meher2006/english-to-telugu-model")
                    st.success("Translation model loaded successfully using direct loading!")
                    return None, tokenizer, model
                except Exception as e3:
                    st.error(f"Direct model loading failed: {str(e3)}")
                    st.error("Could not load Hugging Face model. Using fallback translation.")
                    return None, None, None
                
    except ImportError as e:
        st.error("Transformers library not installed. Install with: pip install transformers torch")
        return None, None, None
    except Exception as e:
        st.error(f"Translation model loading failed: {str(e)}")
        return None, None, None

# Fallback translation function (keeping original)
def fallback_translate_english_to_telugu(text):
    """Enhanced fallback translation using comprehensive word mapping"""
    
    if not text.strip():
        return text
    
    # Expanded English to Telugu word translations
    word_translations = {
        # Basic words
        'hello': 'హలో', 'hi': 'హాయ్', 'hey': 'హే', 'good': 'మంచి', 'bad': 'చెడు',
        'yes': 'అవును', 'no': 'లేదు', 'ok': 'సరే', 'okay': 'సరే',
        
        # Emotions and reactions
        'happy': 'సంతోషం', 'sad': 'దుఃఖం', 'funny': 'సరదా', 'love': 'ప్రేమ',
        'angry': 'కోపం', 'excited': 'ఉత్సాహం', 'surprised': 'ఆశ్చర్యం',
        
        # Common phrases
        'thank you': 'ధన్యవాదాలు', 'sorry': 'క్షమించండి', 'excuse me': 'క్షమించండి',
        'how are you': 'ఎలా ఉన్నావు', 'what happened': 'ఏమి జరిగింది', 'what happen': 'ఏమి జరిగింది',
        
        # Family
        'mom': 'అమ్మ', 'dad': 'నాన్న', 'mother': 'తల్లి', 'father': 'తండ్రి',
        'brother': 'అన్న', 'sister': 'అక్క', 'friend': 'స్నేహితుడు',
        
        # Common objects and concepts
        'work': 'పని', 'home': 'ఇల్లు', 'food': 'తిండి', 'water': 'నీరు',
        'money': 'డబ్బు', 'time': 'సమయం', 'day': 'రోజు', 'night': 'రాత్రి',
        
        # Quantities and modifiers
        'very': 'చాలా', 'much': 'చాలా', 'more': 'ఎక్కువ', 'less': 'తక్కువ',
        'big': 'పెద్ద', 'small': 'చిన్న', 'new': 'కొత్త', 'old': 'పాత',
        
        # Question words
        'what': 'ఏమిటి', 'when': 'ఎప్పుడు', 'where': 'ఎక్కడ', 'who': 'ఎవరు',
        'why': 'ఎందుకు', 'how': 'ఎలా',
        
        # Common verbs
        'go': 'వెళ్లు', 'come': 'రా', 'see': 'చూడు', 'eat': 'తిను',
        'drink': 'త్రాగు', 'sleep': 'నిద్రపో', 'wake': 'లేచు',
        
        # Pronouns
        'i': 'నేను', 'you': 'నువ్వు', 'he': 'అతడు', 'she': 'ఆమె',
        'we': 'మేము', 'they': 'వాళ్ళు', 'this': 'ఇది', 'that': 'అది',
    }
    
    # Simple word replacement with phrase handling
    text_lower = text.lower().strip()
    
    # Check for common phrases first
    for phrase, translation in word_translations.items():
        if phrase in text_lower and len(phrase.split()) > 1:  # Multi-word phrases
            text_lower = text_lower.replace(phrase, translation)
    
    # Then handle individual words
    words = text_lower.split()
    translated_words = []
    
    for word in words:
        clean_word = word.strip('.,!?;:()[]{}"\'-')
        if clean_word in word_translations:
            punctuation = word[len(clean_word):]
            translated_words.append(word_translations[clean_word] + punctuation)
        else:
            # If word contains Telugu characters, keep as is
            if any('\u0C00' <= char <= '\u0C7F' for char in word):
                translated_words.append(word)
            else:
                translated_words.append(word)
    
    result = ' '.join(translated_words).strip()
    
    # If no translation occurred, return original
    return result if result != text_lower else text

# Translation functions (keeping original logic)
def translate_english_to_telugu(text):
    """Translate English to Telugu using Hugging Face model with fallback"""
    
    if not text.strip():
        return text
    
    # Check if text is already Telugu
    telugu_chars = sum(1 for char in text if '\u0C00' <= char <= '\u0C7F')
    if telugu_chars > len(text) * 0.3:  # More than 30% Telugu characters
        return text
    
    # Get the current translator model from session state or global
    current_translator_model = st.session_state.get('current_translator_model')
    current_translator_tokenizer = st.session_state.get('current_translator_tokenizer') 
    current_direct_model = st.session_state.get('current_direct_model')
    
    # Try Hugging Face model first
    if current_translator_model is not None:
        try:
            try:
                result = current_translator_model(text, max_length=200, do_sample=False)
                if result and len(result) > 0:
                    translated_text = result[0]['generated_text'] if 'generated_text' in result[0] else result[0]['translation_text']
                    
                    if translated_text:
                        cleaned_text = clean_translation_output(translated_text, text)
                        if cleaned_text and cleaned_text != text and is_valid_telugu_output(cleaned_text):
                            return cleaned_text
                        
            except Exception as e1:
                st.warning(f"Direct translation failed: {str(e1)}")
            
            try:
                formatted_text = f"translate English to Telugu: {text}"
                result = current_translator_model(formatted_text, max_length=200, do_sample=False)
                if result and len(result) > 0:
                    translated_text = result[0]['generated_text'] if 'generated_text' in result[0] else result[0]['translation_text']
                    
                    if translated_text:
                        if translated_text.startswith(formatted_text):
                            translated_text = translated_text[len(formatted_text):].strip()
                        
                        cleaned_text = clean_translation_output(translated_text, text)
                        if cleaned_text and cleaned_text != text and is_valid_telugu_output(cleaned_text):
                            return cleaned_text
                        
            except Exception as e2:
                st.warning(f"Formatted translation failed: {str(e2)}")
                
        except Exception as e:
            st.warning(f"Model translation failed: {str(e)}. Using fallback.")
    
    elif current_translator_tokenizer is not None and current_direct_model is not None:
        try:
            inputs = current_translator_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
            
            with st.spinner("Translating..."):
                outputs = current_direct_model.generate(
                    inputs, 
                    max_length=200, 
                    num_beams=4, 
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=current_translator_tokenizer.eos_token_id
                )
            
            translated_text = current_translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if translated_text:
                cleaned_text = clean_translation_output(translated_text, text)
                if cleaned_text and cleaned_text != text and is_valid_telugu_output(cleaned_text):
                    return cleaned_text
            
        except Exception as e:
            st.warning(f"Direct model translation failed: {str(e)}. Using fallback.")
    
    # Fallback to word mapping
    return fallback_translate_english_to_telugu(text)


def clean_translation_output(translated_text, original_text):
    """Aggressively clean translation output to remove instruction artifacts"""
    
    if not translated_text or not translated_text.strip():
        return ""
    
    cleaned = translated_text.strip()
    
    # Remove English instruction patterns (case insensitive)
    english_patterns = [
        "translate english to telugu:",
        "translate english to telugu",
        "english to telugu:",
        "english to telugu",
        "translation:",
        "translate:",
        "translate",
    ]
    
    for pattern in english_patterns:
        if cleaned.lower().startswith(pattern):
            cleaned = cleaned[len(pattern):].strip()
    
    # Remove Telugu instruction patterns
    telugu_instruction_patterns = [
        "ఆంగ్లాన్ని టెలుగుకు అనువదించండి:",
        "ఆంగ్లాన్ని టెలుగుకు అనువదించండి",
        "ఆంగ్లం టెలుగుకు అనువదించండి:",
        "ఆంగ్లం టెలుగుకు అనువదించండి",
        "అనువదించండి:",
        "అనువదించండి",
        "అనువాదం:",
        "అనువాదం",
        "పెరగగనువ అనువదించండి:",
        "పెరగగనువ అనువదించండి",
        "అనవదనచడ:",
        "అనవదనచడ",
    ]
    
    for pattern in telugu_instruction_patterns:
        if cleaned.startswith(pattern):
            cleaned = cleaned[len(pattern):].strip()
    
    # Word-level cleanup
    words = cleaned.split()
    english_stop_words = ["translate", "english", "to", "telugu", "translation", "the", "is", "a", "an"]
    telugu_stop_words = ["ఆంగ్లాన్ని", "టెలుగుకు", "అనువదించండి", "అనువాదం", "పెరగగనువ", "అనవదనచడ"]
    
    removed_count = 0
    while words and removed_count < 5:
        first_word = words[0].lower().strip(".,!?:;")
        if first_word in english_stop_words or words[0] in telugu_stop_words:
            words.pop(0)
            removed_count += 1
        else:
            break
    
    cleaned = ' '.join(words).strip()
    cleaned = cleaned.lstrip(".,!?:;-_")
    
    return cleaned

def is_valid_telugu_output(text):
    """Check if the output contains valid Telugu content"""
    
    if not text or len(text.strip()) < 2:
        return False
    
    # Count Telugu characters
    telugu_chars = sum(1 for char in text if '\u0C00' <= char <= '\u0C7F')
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars == 0:
        return False
    
    telugu_ratio = telugu_chars / total_chars
    
    # Additional check - shouldn't contain too many English instruction words
    english_instruction_words = ["translate", "english", "telugu", "translation"]
    word_count = len(text.split())
    instruction_word_count = sum(1 for word in text.lower().split() if word.strip(".,!?:;") in english_instruction_words)
    
    if word_count > 0 and (instruction_word_count / word_count) > 0.3:
        return False
    
    return telugu_ratio >= 0.3

def detect_language(text):
    """Detect if text is primarily English or Telugu using utils"""
    script_type = detect_language_script(text)
    
    # Map script types to display names
    script_mapping = {
        'telugu': 'Telugu',
        'english': 'English', 
        'mixed': 'Mixed',
        'unknown': 'English'  # Default to English
    }
    
    return script_mapping.get(script_type, 'English')

def initialize_app():
    """Initialize application with proper error handling"""
    try:
        # Check configuration first
        if not check_app_configuration():
            st.error("Application not properly configured. Please check settings.")
            return False
        
        # Download fonts (this depends on settings being valid)
        download_fonts()
        
        # Initialize database (this also depends on settings)
        init_db()
        
        return True
    except Exception as e:
        st.error(f"Application initialization failed: {str(e)}")
        return False
def check_app_configuration():
    """Check if app is properly configured"""
    issues = settings.validate_settings()
    if issues:
        st.sidebar.error("Configuration Issues:")
        for issue in issues:
            st.sidebar.warning(f"• {issue}")
        return False
    return True

# Initialize speech recognition
def initialize_speech_recognition():
    try:
        recognizer = sr.Recognizer()
        
        try:
            mic_list = sr.Microphone.list_microphone_names()
            if not mic_list:
                st.warning("No microphones detected")
                return None
            else:
                st.info(f"Found {len(mic_list)} microphone(s)")
                
            with sr.Microphone() as source:
                st.info("Testing microphone access...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                st.success("Microphone access successful!")
                
        except OSError as e:
            if "No Default Input Device Available" in str(e):
                st.error("No microphone found. Please connect a microphone and refresh.")
            else:
                st.error(f"Microphone error: {str(e)}")
            return None
        except Exception as e:
            st.warning(f"Microphone test failed: {str(e)}")
            
        return recognizer
        
    except ImportError:
        st.error("Speech recognition libraries not installed. Run: pip install SpeechRecognition pyaudio")
        return None
    except Exception as e:
        st.warning(f"Speech recognition initialization failed: {str(e)}")
        return None

# Database setup
use_session_storage = False
db_pool = None

def init_db():
    """Initialize database connection using settings"""
    global use_session_storage, db_pool
    
    db_url = settings.get_database_url()
    if not db_url:
        st.warning("Database credentials not configured in settings.")
        st.info("Switching to session-based storage")
        use_session_storage = True
        return
    
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, db_url)
        if db_pool:
            # Create table if it doesn't exist
            conn = db_pool.getconn()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meme_corpus (
                    id SERIAL PRIMARY KEY,
                    language VARCHAR(50),
                    text TEXT,
                    color_theme VARCHAR(20),
                    font_size INTEGER,
                    text_position VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            cursor.close()
            db_pool.putconn(conn)
            st.success("Connected to database!")
    except Exception as e:
        st.warning(f"Database connection failed: {str(e)}")
        st.info("Switching to session-based storage")
        use_session_storage = True
# Page configuration
st.set_page_config(
    page_title="Desi Meme Studio Pro", 
    page_icon="😂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .feature-box {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .stats-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
    .game-card {
        border: 2px solid #e6f3ff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Session state management
def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_data" not in st.session_state:
        st.session_state.user_data = None
    if "access_token" not in st.session_state:
        st.session_state.access_token = None
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'selected_image_name' not in st.session_state:
        st.session_state.selected_image_name = None
    if 'meme_corpus' not in st.session_state:
        st.session_state.meme_corpus = []

def authenticate_user(phone_number: str, password: str) -> Optional[Dict]:
    """Authenticate user with backend API"""
    if not st.session_state.get('api_client'):
        st.session_state.api_client = DesiMemeAPIClient()
    
    try:
        result = st.session_state.api_client.login_for_access_token(phone_number, password)
        if result and "access_token" in result:
            st.session_state.api_client.set_auth_token(result["access_token"])
            return result
        return None
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return None

def register_user(phone_number: str, name: str, email: str, password: str) -> bool:
    """Register new user with OTP verification"""
    if not st.session_state.get('api_client'):
        st.session_state.api_client = DesiMemeAPIClient()
    
    try:
        # Send OTP first
        otp_result = st.session_state.api_client.send_signup_otp(phone_number)
        if otp_result:
            st.session_state.pending_registration = {
                'phone_number': phone_number,
                'name': name,
                'email': email,
                'password': password
            }
            return True
        return False
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return False

def verify_otp_and_complete_registration(otp_code: str) -> bool:
    """Complete registration after OTP verification"""
    if not st.session_state.get('pending_registration'):
        st.error("No pending registration found")
        return False
    
    reg_data = st.session_state.pending_registration
    api_client = st.session_state.get('api_client', DesiMemeAPIClient())
    
    try:
        result = api_client.verify_signup_otp(
            reg_data['phone_number'],
            otp_code,
            reg_data['name'],
            reg_data['email'],
            reg_data['password'],
            True  # has_given_consent
        )
        
        if result:
            del st.session_state.pending_registration
            return True
        return False
    except Exception as e:
        st.error(f"OTP verification failed: {str(e)}")
        return False


def download_fonts():
    """Download fonts if they don't exist"""
    for font_file, url in settings.FONT_URLS.items():
        if not os.path.exists(font_file):
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(font_file), exist_ok=True)
                response = requests.get(url)
                response.raise_for_status()
                with open(font_file, "wb") as f:
                    f.write(response.content)
                st.info(f"Downloaded font: {font_file}")
            except Exception as e:
                st.warning(f"Error downloading font {font_file}: {str(e)}. Using default font.")

if not use_session_storage:
    init_db()

# Meme creation functions
def draw_text_with_outline(draw, text, x, y, font, fill_color="white", outline_color="black", outline_width=2):
    try:
        # Draw outline
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        # Draw main text
        draw.text((x, y), text, font=font, fill=fill_color)
    except Exception as e:
        st.error(f"Error rendering text on image: {str(e)}")
        st.stop()

def create_meme(image_input, top_text, bottom_text, font_size, language_name, color_theme, add_shadow=False, brightness=1.0, contrast=1.0):
    try:
        # Open image
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = Image.open(image_input)
        
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Apply image enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        if add_shadow:
            img = img.filter(ImageFilter.SMOOTH)
        
        draw = ImageDraw.Draw(img)
        
        # Use settings for font mapping with better error handling
        font = None
        font_loaded = False
        
        # Try to load the appropriate font
        font_file = settings.get_font_path(language_name.lower())
        try:
            font = ImageFont.truetype(font_file, font_size)
            font_loaded = True
        except Exception as e1:
            st.warning(f"Failed to load font {font_file}: {str(e1)}")
            
            # Try default font
            try:
                default_font_file = settings.get_font_path('default')
                font = ImageFont.truetype(default_font_file, font_size)
                font_loaded = True
            except Exception as e2:
                st.warning(f"Failed to load default font {default_font_file}: {str(e2)}")
                
                # Last resort: system default
                try:
                    font = ImageFont.load_default()
                    font_loaded = True
                except Exception as e3:
                    st.error(f"Could not load any fonts: {str(e3)}")
                    return None
        
        if not font_loaded:
            st.error("Could not load any fonts for text rendering")
            return None
        
        img_width, img_height = img.size
        
        # Use settings for color themes
        colors = settings.COLOR_THEMES.get(color_theme, settings.COLOR_THEMES["Classic"])
        
        # Draw top text
        if top_text and top_text.strip():
            top_text_upper = top_text.upper()
            bbox = draw.textbbox((0, 0), top_text_upper, font=font)
            text_width = bbox[2] - bbox[0]
            x = (img_width - text_width) // 2
            y = 20
            draw_text_with_outline(
                draw, top_text_upper, x, y, font, 
                colors["fill"], colors["outline"]
            )
        
        # Draw bottom text
        if bottom_text and bottom_text.strip():
            bottom_text_upper = bottom_text.upper()
            bbox = draw.textbbox((0, 0), bottom_text_upper, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (img_width - text_width) // 2
            y = img_height - text_height - 20
            draw_text_with_outline(
                draw, bottom_text_upper, x, y, font, 
                colors["fill"], colors["outline"]
            )
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        return img_byte_arr
        
    except Exception as e:
        st.error(f"Error creating meme: {str(e)}")
        return None

def speech_to_text(recognizer):
    if not recognizer:
        st.error("Speech recognition not available")
        return None
    
    try:
        status_placeholder = st.empty()
        
        try:
            mic_list = sr.Microphone.list_microphone_names()
            if not mic_list:
                status_placeholder.error("No microphones detected. Please connect a microphone.")
                return None
                
            status_placeholder.info(f"Using microphone: {mic_list[0] if mic_list else 'Default'}")
            
        except Exception as mic_error:
            status_placeholder.error(f"Microphone check failed: {str(mic_error)}")
            return None
        
        try:
            with sr.Microphone() as source:
                status_placeholder.info("Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1.5)
                
                status_placeholder.success("RECORDING... Speak clearly now!")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=8)
                
        except sr.WaitTimeoutError:
            status_placeholder.error("No speech detected. Please try again.")
            return None
        except Exception as e:
            status_placeholder.error(f"Recording failed: {str(e)}")
            return None
            
        status_placeholder.info("Processing speech...")
        
        try:
            text = recognizer.recognize_google(audio)
            if text and text.strip():
                status_placeholder.success(f"Recognized: '{text}'")
                return text.strip()
        except sr.UnknownValueError:
            status_placeholder.error("Could not understand audio. Please speak clearer.")
        except sr.RequestError as e:
            status_placeholder.error(f"Recognition service error: {e}")
        except Exception as e:
            status_placeholder.error(f"Recognition failed: {str(e)}")
        
        return None
            
    except Exception as e:
        st.error(f"Speech recognition error: {str(e)}")
        return None

def save_to_db(language, text, color_theme, font_size, text_position):
    """Save meme data using API client"""
    global use_session_storage
    
    if use_session_storage or not st.session_state.get('authenticated'):
        # Fallback to session storage
        entry = {
            'language': language,
            'text': text,
            'color_theme': color_theme,
            'font_size': font_size,
            'text_position': text_position,
            'created_at': datetime.datetime.now()
        }
        st.session_state.meme_corpus.append(entry)
        show_success_message(f"Meme data saved locally! Language: {language}")
        return
    
    # Use API client to save to backend
    try:
        api_client = st.session_state.get('api_client')
        if not api_client:
            raise Exception("API client not initialized")
        
        meme_data = {
            'title': f"Meme - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            'top_text': text.split('\n')[0] if text else '',
            'bottom_text': text.split('\n')[1] if '\n' in text else '',
            'language': language,
            'color_theme': color_theme,
            'font_size': font_size,
            'text_position': text_position,
            'user_id': st.session_state.user_data.get('id')
        }
        
        result = api_client.create_meme(meme_data)
        if result:
            show_success_message(f"Meme saved successfully! Language: {language}")
        else:
            raise Exception("Failed to save meme")
            
    except Exception as e:
        st.error(f"Error saving to backend: {str(e)}")
        # Fallback to session storage
        use_session_storage = True
        save_to_db(language, text, color_theme, font_size, text_position)

def get_corpus_with_stats():
    global use_session_storage
    if use_session_storage:
        corpus_data = st.session_state.meme_corpus
        if not corpus_data:
            return pd.DataFrame(), [], 0
        
        df = pd.DataFrame(corpus_data)
        df.columns = ['Language', 'Text', 'Color Theme', 'Font Size', 'Text Position', 'Created At']
        lang_counts = df['Language'].value_counts()
        lang_stats = [(lang, count) for lang, count in lang_counts.items()]
        total_count = len(corpus_data)
        return df.tail(50), lang_stats, total_count

    try:
        conn = db_pool.getconn()
        cursor = conn.cursor()
        cursor.execute("SELECT language, text, color_theme, font_size, text_position, created_at FROM meme_corpus ORDER BY created_at DESC LIMIT 50")
        df = pd.DataFrame(cursor.fetchall(), columns=['Language', 'Text', 'Color Theme', 'Font Size', 'Text Position', 'Created At'])
        cursor.execute("SELECT language, COUNT(*) as count FROM meme_corpus GROUP BY language ORDER BY count DESC")
        lang_stats = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM meme_corpus")
        total_count = cursor.fetchone()[0]
        cursor.close()
        db_pool.putconn(conn)
        return df, lang_stats, total_count
    except Exception as e:
        st.error(f"Error retrieving corpus: {str(e)}")
        use_session_storage = True
        return get_corpus_with_stats()

def main():
    # Initialize app first
    if not initialize_app():
        st.stop()
    
    init_session_state()
    
    # Initialize API client
    if 'api_client' not in st.session_state:
        st.session_state.api_client = DesiMemeAPIClient()
    
    # Main header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title(settings.APP_NAME)
    st.markdown(f"### *Version {settings.APP_VERSION} - Create Viral Telugu & English Memes with AI Translation*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Authentication check
    if not st.session_state.authenticated:
        show_auth_pages()
    else:
        show_main_app()

def show_auth_pages():
    """Show authentication pages (login/signup)"""
    auth_tab = st.sidebar.selectbox("Choose Action", ["Login", "Sign Up"])
    
    if auth_tab == "Login":
        show_login_page()
    else:
        show_signup_page()

# 3. Update the show_login_page function
def show_login_page():
    """Show login page with proper validation"""
    st.header("Login to Your Account")
    
    with st.form("login_form"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            phone_number = st.text_input("Phone Number", placeholder="Enter your 10-digit mobile number")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        with col2:
            st.markdown("### Welcome Back!")
            st.markdown("Login to continue creating amazing memes with AI-powered translation.")
            st.markdown("**New here?** Switch to Sign Up to create an account.")
        
        submitted = st.form_submit_button("Login", use_container_width=True)
        
    if submitted:
        # Validate inputs
        phone_valid, phone_msg = validate_phone_number(phone_number)
        if not phone_valid:
            show_error_message(phone_msg)
            return
        
        if not password:
            show_error_message("Password is required")
            return
        
        with st.spinner("Logging in..."):
            result = authenticate_user(phone_number, password)
            if result and "access_token" in result:
                st.session_state.access_token = result["access_token"]
                st.session_state.user_data = result.get("user", {})
                st.session_state.authenticated = True
                show_success_message("Login successful! Redirecting...")
                st.rerun()
            else:
                show_error_message("Login failed. Please check your credentials.")

# 4. Update the show_signup_page function
def show_signup_page():
    """Show signup page with OTP verification"""
    st.header("Create New Account")
    
    # Check if OTP verification is pending
    if st.session_state.get('pending_registration'):
        show_otp_verification_form()
        return
    
    st.subheader("Join the Meme Community")
    with st.form("signup_form"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            phone_number = st.text_input("Phone Number", placeholder="Enter your 10-digit mobile number")
            name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email", placeholder="Enter your email address")
            password = st.text_input("Password", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            has_given_consent = st.checkbox("I agree to the terms and conditions")
        
        with col2:
            st.markdown("### Join Our Community!")
            st.markdown("Create and share memes in Telugu and English with our AI-powered platform.")
            st.markdown("**Key Features:**")
            st.markdown("- AI-powered English to Telugu translation")
            st.markdown("- Voice input for hands-free meme creation")
            st.markdown("- Multiple meme templates and themes")
            st.markdown("- Community analytics and sharing")
        
        submitted = st.form_submit_button("Create Account", use_container_width=True)

        if submitted:
            # Validate all inputs
            phone_valid, phone_msg = validate_phone_number(phone_number)
            if not phone_valid:
                show_error_message(phone_msg)
                return
                
            if not validate_email(email):
                show_error_message("Please enter a valid email address.")
                return
                
            password_valid, password_msg = validate_password(password)
            if not password_valid:
                show_error_message(password_msg)
                return
                
            if not all([phone_number, name, email, password, confirm_password]):
                show_error_message("Please fill in all fields.")
                return
                
            if password != confirm_password:
                show_error_message("Passwords don't match.")
                return
                
            if not has_given_consent:
                show_error_message("Please agree to the terms and conditions.")
                return
            
            with st.spinner("Sending OTP..."):
                result = register_user(phone_number, name, email, password)
                if result:
                    show_success_message("OTP sent to your phone! Please verify to complete registration.")
                    st.rerun()
                else:
                    show_error_message("Account creation failed. Please try again.")

def show_otp_verification_form():
    """Show OTP verification form"""
    st.subheader("Verify Your Phone Number")
    
    reg_data = st.session_state.pending_registration
    st.info(f"OTP sent to {reg_data['phone_number']}")
    
    with st.form("otp_form"):
        otp_code = st.text_input("Enter OTP", placeholder="Enter 6-digit OTP", max_chars=6)
        
        col1, col2 = st.columns(2)
        with col1:
            verify_submitted = st.form_submit_button("Verify OTP", use_container_width=True)
        with col2:
            if st.form_submit_button("Resend OTP", use_container_width=True):
                api_client = st.session_state.get('api_client', DesiMemeAPIClient())
                result = api_client.resend_signup_otp(reg_data['phone_number'])
                if result:
                    show_success_message("OTP resent successfully!")
                else:
                    show_error_message("Failed to resend OTP. Please try again.")
    
    if verify_submitted and otp_code:
        if len(otp_code) != 6 or not otp_code.isdigit():
            show_error_message("Please enter a valid 6-digit OTP")
            return
            
        with st.spinner("Verifying OTP..."):
            if verify_otp_and_complete_registration(otp_code):
                show_success_message("Registration completed successfully! Please login with your credentials.")
                st.balloons()
                st.rerun()
            else:
                show_error_message("OTP verification failed. Please try again.")
    
    if st.button("Cancel Registration"):
        if 'pending_registration' in st.session_state:
            del st.session_state.pending_registration
        st.rerun()

def show_main_app():
    """Show main application after authentication"""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown(f"**Welcome, {st.session_state.user_data.get('name', 'User')}!**")
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Meme Creator", "Dashboard", "Profile"]
    )
    
    # Logout button
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_data = None
        st.session_state.access_token = None
        st.rerun()
    
    # Route to appropriate page
    if page == "Home":
        show_home_page()
    elif page == "Meme Creator":
        show_meme_creator_page()
    elif page == "Dashboard":
        show_dashboard()
    elif page == "Profile":
        show_profile_page()

def show_home_page():
    """Show the home page with project overview"""
    st.header("Welcome to Desi Meme Studio Pro!")
    
    st.markdown("""
    Desi Meme Studio Pro is a revolutionary platform that combines traditional meme creation with cutting-edge AI technology 
    to bridge language barriers and cultural expression. Our mission is to make meme creation accessible to everyone, 
    regardless of their preferred language.

    **Our Mission:**
    * **Create:** Generate hilarious memes in both Telugu and English with ease
    * **Translate:** Use advanced AI models to seamlessly convert English text to Telugu
    * **Connect:** Build a community of meme creators sharing cultural humor
    * **Preserve:** Document and celebrate regional linguistic expressions through memes

    **What you can do here:**
    * **Explore:** Browse through various meme templates and themes
    * **Create:** Build custom memes with voice input and AI translation
    * **Share:** Contribute to our growing collection of cultural memes
    * **Analyze:** Track trends and popular content in our analytics dashboard

    Join thousands of creators who are making the internet a funnier, more inclusive place!
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### AI Translation")
        st.markdown("State-of-the-art Hugging Face models for English to Telugu translation")
        st.markdown("Fallback system ensures 100% translation coverage")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### Voice Input")
        st.markdown("Hands-free meme creation with speech recognition")
        st.markdown("Perfect for mobile users and accessibility")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("### Rich Templates")
        st.markdown("10+ professional meme templates")
        st.markdown("Multiple color themes and customization options")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    ### తెలుగులో స్వాగతం! (Welcome in Telugu!)
    
    దేశీ మీమ్ స్టూడియో ప్రో అనేది సాంప్రదాయ మీమ్ సృష్టితో అత్యాధునిక AI సాంకేతికతను కలిపి 
    భాషా అడ్డంకులు మరియు సాంస్కృతిక వ్యక్తీకరణలను తగ్గించే విప్లవాత్మక వేదిక.
    
    మా లక్ష్యం ప్రతి ఒక్కరికీ, వారి ఇష్టపడే భాష ఏదైనప్పటికీ, మీమ్ సృష్టిని అందుబాటులో ఉంచడం.
    """)

def show_meme_creator_page():
    """Show the meme creator page with all original functionality"""
    st.header("Meme Creator Studio")
    
    # Sidebar for translation and voice features
    with st.sidebar:
        st.subheader("Creator Controls")
        st.markdown("---")
        
        # Translation Features
        st.subheader("Translation")
        use_translation = st.checkbox("Enable English to Telugu Translation", 
                                    help="Uses advanced Hugging Face AI model for better translation")
        
        # Load translation model if enabled
        if use_translation:
            translator_model, translator_tokenizer, direct_model = load_translation_model()
            
            # Store in session state for use in translation function
            st.session_state.current_translator_model = translator_model
            st.session_state.current_translator_tokenizer = translator_tokenizer
            st.session_state.current_direct_model = direct_model
            
            if translator_model or translator_tokenizer:
                st.success("AI Translation Model Ready!")
                st.info("Using Hugging Face: Meher2006/english-to-telugu-model")
            else:
                st.warning("AI model failed. Using fallback translation.")
                st.info("Basic word mapping available")
        else:
            # Clear translation models from session state
            st.session_state.current_translator_model = None
            st.session_state.current_translator_tokenizer = None
            st.session_state.current_direct_model = None
        
        st.subheader("Voice Input")
        use_voice = st.checkbox("Enable Voice Input", help="Use voice to add text to memes")
        
        st.markdown("---")
        
        # Initialize speech recognition if enabled
        recognizer = initialize_speech_recognition() if use_voice else None
        
        # Audio diagnostics
        if use_voice:
            if recognizer:
                st.success("Audio system ready!")
            else:
                st.error("Audio system not available")

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Create Meme", "Analytics Dashboard", "Translation Examples"])

    with tab1:
        st.markdown("### Step 1: Choose Your Canvas")
        
        # Image input selection
        image_input_option = st.radio("Select Image Source:", ("Browse Meme Templates", "Upload Custom Image"))
        
        # Predefined images for selection
        image_options = {
            "Viral Template 1": "images/meme1.jpg",
            "Classic Meme 2": "images/meme2.jpg",
            "Trending 3": "images/meme3.jpg",
            "Popular 4": "images/meme4.jpg",
            "Hot 5": "images/meme5.jpg",
            "Electric 6": "images/meme6.jpg",
            "Star 7": "images/meme7.jpg",
            "Fun 8": "images/meme8.jpg",
            "Creative 9": "images/meme9.jpg",
            "Champion 10": "images/meme10.jpg"
        }
        
        if image_input_option == "Browse Meme Templates":
            st.markdown("#### Select from Templates")
            # First row
            cols1 = st.columns(5)
            for idx, (img_name, img_path) in enumerate(list(image_options.items())[:5]):
                try:
                    with cols1[idx]:
                        if os.path.exists(img_path):
                            img = Image.open(img_path)
                            st.image(img, caption=img_name, use_container_width=True)
                            if st.button(f"Choose", key=f"select_{img_name}"):
                                st.session_state.selected_image = img_path
                                st.session_state.selected_image_name = img_name
                        else:
                            st.info(f"Template {img_name} not available")
                except Exception as e:
                    st.warning(f"Error loading {img_name}: {str(e)}")

            # Second row
            cols2 = st.columns(5)
            for idx, (img_name, img_path) in enumerate(list(image_options.items())[5:]):
                try:
                    with cols2[idx]:
                        if os.path.exists(img_path):
                            img = Image.open(img_path)
                            st.image(img, caption=img_name, use_container_width=True)
                            if st.button(f"Choose", key=f"select_{img_name}"):
                                st.session_state.selected_image = img_path
                                st.session_state.selected_image_name = img_name
                        else:
                            st.info(f"Template {img_name} not available")
                except Exception as e:
                    st.warning(f"Error loading {img_name}: {str(e)}")

            if st.session_state.selected_image_name:
                st.success(f"Selected Template: {st.session_state.selected_image_name}")
        else:
            st.markdown("#### Upload Your Own Image")
            uploaded_image = st.file_uploader(
                "Choose image file", 
                type=settings.ALLOWED_IMAGE_EXTENSIONS
            )
            if uploaded_image:
                # Validate the file
                is_valid, message = validate_uploaded_file(uploaded_image)
                if not is_valid:
                    show_error_message(message)
                    return
                
                st.session_state.selected_image = uploaded_image
                st.session_state.selected_image_name = None
                show_success_message("Custom image uploaded successfully!")

        st.markdown("### Step 2: Add Your Text")
        
        # Voice input section
        if use_voice and recognizer:
            st.markdown("#### Voice Input")
            col1, col2, col3 = st.columns(3)
            with col1:
                font_size = st.slider("Font Size", 20, 120, 50)
            with col2:
                # FIX: Use settings instead of hardcoded color_themes
                color_theme = st.selectbox("Color Theme", list(settings.COLOR_THEMES.keys()))
            with col3:
                add_shadow = st.checkbox("Add Shadow Effect")


            with col1:
                if st.button("Record Top Text", type="secondary"):
                    with st.spinner("Recording..."):
                        voice_text = speech_to_text(recognizer)
                        if voice_text:
                            if use_translation:
                                translated_text = translate_english_to_telugu(voice_text)
                                st.session_state.voice_top_text = translated_text
                                if translated_text != voice_text:
                                    st.success(f"Voice → Telugu: {translated_text}")
                                else:
                                    st.success(f"Voice Input: {voice_text}")
                            else:
                                st.session_state.voice_top_text = voice_text
                                st.success(f"Voice Input: {voice_text}")
            
            with col2:
                if st.button("Record Bottom Text", type="secondary"):
                    with st.spinner("Recording..."):
                        voice_text = speech_to_text(recognizer)
                        if voice_text:
                            if use_translation:
                                translated_text = translate_english_to_telugu(voice_text)
                                st.session_state.voice_bottom_text = translated_text
                                if translated_text != voice_text:
                                    st.success(f"Voice → Telugu: {translated_text}")
                                else:
                                    st.success(f"Voice Input: {voice_text}")
                            else:
                                st.session_state.voice_bottom_text = voice_text
                                st.success(f"Voice Input: {voice_text}")
            
            st.markdown("---")
        
        # Text input with translation
        col1, col2 = st.columns(2)
        with col1:
            default_top = st.session_state.get('voice_top_text', '')
            top_text = st.text_area(
                "Top Text", 
                value=default_top,
                placeholder="Enter top text in Telugu or English...",
                help="Enable translation to auto-convert English to Telugu"
            )
            
            # Translation button for top text
            if use_translation and top_text:
                if st.button("Translate Top to Telugu", key="trans_top"):
                    translated = translate_english_to_telugu(top_text)
                    if translated != top_text:
                        st.session_state.translated_top = translated
                        st.success(f"Translated: {translated}")
                    else:
                        st.info("Text is already in Telugu or no translation needed")
        
        with col2:
            default_bottom = st.session_state.get('voice_bottom_text', '')
            bottom_text = st.text_area(
                "Bottom Text",
                value=default_bottom, 
                placeholder="Enter bottom text in Telugu or English...",
                help="Enable translation to auto-convert English to Telugu"
            )
            
            # Translation button for bottom text
            if use_translation and bottom_text:
                if st.button("Translate Bottom to Telugu", key="trans_bottom"):
                    translated = translate_english_to_telugu(bottom_text)
                    if translated != bottom_text:
                        st.session_state.translated_bottom = translated
                        st.success(f"Translated: {translated}")
                    else:
                        st.info("Text is already in Telugu or no translation needed")
        
        # Use translated text if available
        if 'translated_top' in st.session_state:
            top_text = st.session_state.translated_top
        if 'translated_bottom' in st.session_state:
            bottom_text = st.session_state.translated_bottom

        st.markdown("### Step 3: Customize Your Style")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            font_size = st.slider("Font Size", 20, 120, 50)
        with col2:
            color_theme = st.selectbox("Color Theme", list(color_themes.keys()))
        with col3:
            add_shadow = st.checkbox("Add Shadow Effect")

        # Image enhancement controls
        col1, col2 = st.columns(2)
        with col1:
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        with col2:
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)

        # Generate meme button
        if st.button("Generate Meme", type="primary"):
            if st.session_state.selected_image is not None and (top_text or bottom_text):
                with st.spinner("Creating your meme..."):
                    # Auto-translate if enabled
                    final_top_text = top_text
                    final_bottom_text = bottom_text
                    
                    if use_translation:
                        if top_text:
                            final_top_text = translate_english_to_telugu(top_text)
                        if bottom_text:
                            final_bottom_text = translate_english_to_telugu(bottom_text)
                    
                    # Detect language
                    combined_text = f"{final_top_text} {final_bottom_text}".strip()
                    language_name = detect_language(combined_text)
                    
                    meme_image = create_meme(
                        st.session_state.selected_image, 
                        final_top_text, 
                        final_bottom_text, 
                        font_size, 
                        language_name, 
                        color_theme, 
                        add_shadow, 
                        brightness, 
                        contrast
                    )
                    
                    if meme_image:
                        st.markdown("### Your Meme is Ready!")
                        st.image(meme_image, caption="Fresh Meme Created!", use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.download_button(
                                label="Download Meme",
                                data=meme_image,
                                file_name=f"desi_meme_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                        with col2:
                            st.markdown(f"**Detected Language:** {language_name}")
                            if use_translation and (final_top_text != top_text or final_bottom_text != bottom_text):
                                st.markdown("**Translated:** Yes")
                        with col3:
                            st.markdown(f"**Theme:** {color_theme}")
                            if use_voice and (st.session_state.get('voice_top_text') or st.session_state.get('voice_bottom_text')):
                                st.markdown("**Voice Input:** Yes")

                        save_to_db(language_name, combined_text, color_theme, font_size, "top_bottom")
                        
                        # Clear session states after successful generation
                        for key in ['voice_top_text', 'voice_bottom_text', 'translated_top', 'translated_bottom']:
                            if key in st.session_state:
                                del st.session_state[key]
            else:
                st.error("Please select an image and add at least one text to create your meme!")

    with tab2:
        st.markdown("### Meme Studio Analytics")
        
        df, lang_stats, total_count = get_corpus_with_stats()
        
        if total_count > 0:
            # Statistics overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Memes Created", total_count)
            with col2:
                st.metric("Languages Used", len(lang_stats))
            with col3:
                most_used_lang = lang_stats[0][0] if lang_stats else "N/A"
                st.metric("Top Language", most_used_lang)
            with col4:
                avg_font_size = int(df['Font Size'].mean()) if not df.empty else 0
                st.metric("Avg Font Size", avg_font_size)
            
            # Language distribution
            if lang_stats:
                st.markdown("#### Language Distribution")
                lang_df = pd.DataFrame(lang_stats, columns=['Language', 'Count'])
                st.bar_chart(lang_df.set_index('Language'))
            
            # Recent memes table
            if not df.empty:
                st.markdown("#### Recent Meme Creations")
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No memes created yet. Start creating to see analytics!")

    with tab3:
        st.markdown("### AI Translation & Examples")
        
        # Translation test section
        st.markdown("#### Test AI Translation")
        
        test_input = st.text_input("Enter English text to test Hugging Face translation:")
        if test_input:
            if use_translation:
                translated_output = translate_english_to_telugu(test_input)
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**English:** {test_input}")
                with col2:
                    st.write(f"**Telugu:** {translated_output}")
                
                # Show if AI model was used or fallback
                if translator_model or translator_tokenizer:
                    st.success("Translated using AI model")
                else:
                    st.info("Translated using fallback method")
            else:
                st.warning("Enable translation in the sidebar to test")
        
        st.markdown("#### Model Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Primary Model:**
            - **Name:** Meher2006/english-to-telugu-model
            - **Type:** Hugging Face Transformer
            - **Quality:** High accuracy for full sentences
            - **Speed:** Moderate (model loading required)
            """)
        
        with col2:
            st.markdown("""
            **Fallback System:**
            - **Type:** Word mapping dictionary
            - **Quality:** Good for common words
            - **Speed:** Very fast
            - **Coverage:** 100+ common Telugu words
            """)

def show_dashboard():
    """Show dashboard with overview statistics"""
    st.header("Dashboard Overview")
    
    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        user_id = st.session_state.user_data.get('id')
        user_memes = [m for m in st.session_state.meme_corpus if True]  # All memes for current user
        st.metric("Your Memes Created", len(user_memes), help="Total memes you've created")
    
    with col2:
        df, lang_stats, total_count = get_corpus_with_stats()
        st.metric("Total Community Memes", total_count, help="All memes created by the community")
    
    with col3:
        if lang_stats:
            most_used_lang = lang_stats[0][0]
            st.metric("Most Popular Language", most_used_lang)
        else:
            st.metric("Most Popular Language", "N/A")
    
    with col4:
        recent_memes_today = len([m for m in st.session_state.meme_corpus 
                                 if m.get('created_at', datetime.datetime.now()).date() == datetime.datetime.now().date()])
        st.metric("Memes Today", recent_memes_today)
    
    # Recent activity
    st.subheader("Recent Community Activity")
    
    df, lang_stats, total_count = get_corpus_with_stats()
    
    if not df.empty:
        st.success(f"Found {len(df)} recent memes!")
        for _, meme in df.head(10).iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Language:** {meme['Language']}")
                    st.caption(f"Text: {meme['Text'][:100]}..." if len(meme['Text']) > 100 else f"Text: {meme['Text']}")
                    st.caption(f"Theme: {meme['Color Theme']} | Font Size: {meme['Font Size']}")
                with col2:
                    st.markdown(f"*{meme['Created At']}*")
            st.markdown("---")
    else:
        st.info("No recent activity found. Be the first to create a meme!")

def show_profile_page():
    """Show user profile page"""
    st.header("User Profile")
    
    if st.session_state.user_data:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Profile Information")
            st.markdown(f"**Phone:** {st.session_state.user_data.get('phone', 'N/A')}")
            st.markdown(f"**Name:** {st.session_state.user_data.get('name', 'N/A')}")
            st.markdown(f"**Email:** {st.session_state.user_data.get('email', 'N/A')}")
            st.markdown(f"**Member Since:** Recently")
        
        with col2:
            st.markdown("### Your Meme Statistics")
            user_memes = st.session_state.meme_corpus
            if user_memes:
                total_memes = len(user_memes)
                languages_used = list(set([meme['language'] for meme in user_memes]))
                
                st.metric("Total Memes Created", total_memes)
                st.metric("Languages Used", len(languages_used))
                
                if languages_used:
                    st.markdown("**Your Languages:** " + ", ".join(languages_used))
            else:
                st.info("You haven't created any memes yet!")
        
        st.markdown("---")
        
        # Profile settings
        st.markdown("### Account Settings")
        
        with st.expander("Change Password"):
            with st.form("change_password_form"):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_new_password = st.text_input("Confirm New Password", type="password")
                
                if st.form_submit_button("Update Password"):
                    if new_password != confirm_new_password:
                        st.error("New passwords don't match!")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long!")
                    else:
                        # Mock password change - replace with actual API call
                        st.success("Password changed successfully!")
        
        with st.expander("Update Profile"):
            with st.form("update_profile_form"):
                updated_name = st.text_input("Name", value=st.session_state.user_data.get('name', ''))
                updated_email = st.text_input("Email", value=st.session_state.user_data.get('email', ''))
                
                if st.form_submit_button("Update Profile"):
                    if not validate_email(updated_email):
                        st.error("Please enter a valid email address.")
                    else:
                        # Mock profile update - replace with actual API call
                        st.session_state.user_data['name'] = updated_name
                        st.session_state.user_data['email'] = updated_email
                        st.success("Profile updated successfully!")
                        st.rerun()

# Helper functions
def validate_email(email):
    """Simple email validation"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# 9. Update file validation using utils
def validate_uploaded_file(uploaded_file):
    """Validate uploaded file using utils"""
    if not uploaded_file:
        return True, "No file uploaded"
    
    return validate_file_upload(
        uploaded_file, 
        max_size_mb=settings.MAX_IMAGE_SIZE // (1024 * 1024),
        allowed_types=settings.ALLOWED_IMAGE_EXTENSIONS
    )

# 10. Add error handling for settings validation
def check_app_configuration():
    """Check if app is properly configured"""
    issues = settings.validate_settings()
    if issues:
        st.sidebar.error("Configuration Issues:")
        for issue in issues:
            st.sidebar.warning(f"• {issue}")
        return False
    return True

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    import math
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

# Footer with example texts
def show_footer():
    """Show footer with example texts and information"""
    st.markdown("---")
    st.markdown("### Example Meme Texts")

    examples_col1, examples_col2 = st.columns(2)

    with examples_col1:
        st.markdown("""
        **Telugu Examples:**
        - ఇది చాలా సరదాగా ఉంది
        - కాఫీ లేకుండా పని చేయలేను  
        - మీమ్ క్రియేటర్ అద్భుతం
        - నవ్వు రాదు ఆపలేను
        """)

    with examples_col2:
        st.markdown("""
        **English Examples (will be translated):**
        - this is very funny
        - monday morning vibes
        - when you see your salary
        - mom calling during office
        """)

    st.markdown("---")
    st.markdown("*Built with AI-powered English-Telugu translation | Hugging Face Models | Community-driven content*")

def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_data" not in st.session_state:
        st.session_state.user_data = None
    if "access_token" not in st.session_state:
        st.session_state.access_token = None
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'selected_image_name' not in st.session_state:
        st.session_state.selected_image_name = None
    if 'meme_corpus' not in st.session_state:
        st.session_state.meme_corpus = []
    
    # ADD THESE FOR TRANSLATION MODEL STATE
    if 'current_translator_model' not in st.session_state:
        st.session_state.current_translator_model = None
    if 'current_translator_tokenizer' not in st.session_state:
        st.session_state.current_translator_tokenizer = None
    if 'current_direct_model' not in st.session_state:
        st.session_state.current_direct_model = None

# Run the application
if __name__ == "__main__":
    main()
    
    # Show footer only on authenticated pages
    if st.session_state.get('authenticated', False):
        show_footer()
