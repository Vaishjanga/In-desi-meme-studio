import streamlit as st
import os, pandas as pd, io, requests, datetime, random, re, tempfile, warnings
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import psycopg2
from psycopg2 import pool
from packaging import version
from dotenv import load_dotenv
from typing import Optional, Dict, List

# Import utils and config
from utils import (
    DesiMemeAPIClient, validate_email, validate_password, validate_phone_number,
    show_success_message, show_error_message, detect_language_script,
    generate_meme_filename, validate_file_upload, format_file_size
)
from config import settings

warnings.filterwarnings("ignore")
load_dotenv()

# Global variables
translator_model = translator_tokenizer = None
use_session_storage = False
db_pool = None

@st.cache_resource
def load_translation_model():
    """Load Hugging Face English to Telugu translation model"""
    global translator_model, translator_tokenizer
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        st.info("Loading English-Telugu translation model...")
        
        # Try different approaches
        for method in ["text2text-generation", "translation"]:
            try:
                translator_pipeline = pipeline(method, model="Meher2006/english-to-telugu-model")
                st.success(f"Translation model loaded using {method} pipeline!")
                return translator_pipeline, None, None
            except Exception:
                continue
        
        # Direct model loading
        tokenizer = AutoTokenizer.from_pretrained("Meher2006/english-to-telugu-model")
        model = AutoModelForSeq2SeqLM.from_pretrained("Meher2006/english-to-telugu-model")
        st.success("Translation model loaded using direct loading!")
        return None, tokenizer, model
        
    except Exception as e:
        st.error(f"Translation model loading failed: {str(e)}")
        return None, None, None

def fallback_translate_english_to_telugu(text):
    """Enhanced fallback translation using word mapping"""
    if not text.strip():
        return text
    
    word_translations = {
        'hello': 'హలో', 'hi': 'హాయ్', 'good': 'మంచి', 'bad': 'చెడు', 'yes': 'అవును',
        'no': 'లేదు', 'ok': 'సరే', 'happy': 'సంతోషం', 'sad': 'దుఃఖం', 'funny': 'సరదా',
        'love': 'ప్రేమ', 'angry': 'కోపం', 'mom': 'అమ్మ', 'dad': 'నాన్న', 'work': 'పని',
        'home': 'ఇల్లు', 'food': 'తిండి', 'water': 'నీరు', 'money': 'డబ్బు', 'time': 'సమయం',
        'very': 'చాలా', 'much': 'చాలా', 'big': 'పెద్ద', 'small': 'చిన్న', 'what': 'ఏమిటి',
        'when': 'ఎప్పుడు', 'where': 'ఎక్కడ', 'who': 'ఎవరు', 'why': 'ఎందుకు', 'how': 'ఎలా',
        'i': 'నేను', 'you': 'నువ్వు', 'thank you': 'ధన్యవాదాలు'
    }
    
    text_lower = text.lower().strip()
    
    # Handle phrases first
    for phrase, translation in word_translations.items():
        if len(phrase.split()) > 1 and phrase in text_lower:
            text_lower = text_lower.replace(phrase, translation)
    
    # Handle individual words
    words = text_lower.split()
    translated_words = []
    
    for word in words:
        clean_word = word.strip('.,!?;:()[]{}"\'-')
        if clean_word in word_translations:
            punctuation = word[len(clean_word):]
            translated_words.append(word_translations[clean_word] + punctuation)
        else:
            translated_words.append(word)
    
    return ' '.join(translated_words).strip() or text

def translate_english_to_telugu(text):
    """Translate English to Telugu using Hugging Face model with fallback"""
    if not text.strip():
        return text
    
    # Check if already Telugu
    telugu_chars = sum(1 for char in text if '\u0C00' <= char <= '\u0C7F')
    if telugu_chars > len(text) * 0.3:
        return text
    
    current_translator_model = st.session_state.get('current_translator_model')
    current_translator_tokenizer = st.session_state.get('current_translator_tokenizer')
    current_direct_model = st.session_state.get('current_direct_model')
    
    # Try Hugging Face model
    if current_translator_model:
        try:
            for prompt in [text, f"translate English to Telugu: {text}"]:
                result = current_translator_model(prompt, max_length=200, do_sample=False)
                if result and len(result) > 0:
                    translated = result[0].get('generated_text') or result[0].get('translation_text')
                    if translated:
                        cleaned = clean_translation_output(translated, text)
                        if cleaned and is_valid_telugu_output(cleaned):
                            return cleaned
        except Exception as e:
            st.warning(f"Model translation failed: {str(e)}")
    
    elif current_translator_tokenizer and current_direct_model:
        try:
            inputs = current_translator_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
            outputs = current_direct_model.generate(inputs, max_length=200, num_beams=4, early_stopping=True)
            translated = current_translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if translated:
                cleaned = clean_translation_output(translated, text)
                if cleaned and is_valid_telugu_output(cleaned):
                    return cleaned
        except Exception as e:
            st.warning(f"Direct model translation failed: {str(e)}")
    
    return fallback_translate_english_to_telugu(text)

def clean_translation_output(translated_text, original_text):
    """Clean translation output"""
    if not translated_text:
        return ""
    
    cleaned = translated_text.strip()
    
    # Remove instruction patterns
    patterns = ["translate english to telugu:", "translation:", "translate:", "ఆంగ్లాన్ని టెలుగుకు అనువదించండి:"]
    for pattern in patterns:
        if cleaned.lower().startswith(pattern):
            cleaned = cleaned[len(pattern):].strip()
    
    return cleaned.lstrip(".,!?:;-_")

def is_valid_telugu_output(text):
    """Check if output contains valid Telugu content"""
    if not text or len(text.strip()) < 2:
        return False
    
    telugu_chars = sum(1 for char in text if '\u0C00' <= char <= '\u0C7F')
    total_chars = len([char for char in text if char.isalpha()])
    
    return total_chars > 0 and (telugu_chars / total_chars) >= 0.3

def detect_language(text):
    """Detect language using utils"""
    script_mapping = {'telugu': 'Telugu', 'english': 'English', 'mixed': 'Mixed', 'unknown': 'English'}
    return script_mapping.get(detect_language_script(text), 'English')

def initialize_app():
    """Initialize application"""
    try:
        if not settings.validate_settings():
            return False
        download_fonts()
        init_db()
        return True
    except Exception as e:
        st.error(f"Application initialization failed: {str(e)}")
        return False

def init_db():
    """Initialize database connection"""
    global use_session_storage, db_pool
    
    db_url = settings.get_database_url()
    if not db_url:
        st.warning("Database not configured. Using session storage.")
        use_session_storage = True
        return
    
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, db_url)
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
        use_session_storage = True

def download_fonts():
    """Download fonts if needed"""
    for font_file, url in settings.FONT_URLS.items():
        if not os.path.exists(font_file):
            try:
                os.makedirs(os.path.dirname(font_file), exist_ok=True)
                response = requests.get(url)
                response.raise_for_status()
                with open(font_file, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                st.warning(f"Font download failed: {str(e)}")

# Page configuration
st.set_page_config(page_title="Desi Meme Studio Pro", page_icon="😂", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        padding: 1rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;
    }
    .feature-box {
        border: 2px solid #e0e0e0; border-radius: 10px; padding: 1rem; margin: 1rem 0; background-color: #f8f9fa;
    }
    .stats-box {
        background-color: #e3f2fd; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state"""
    defaults = {
        "authenticated": False, "user_data": None, "access_token": None,
        'selected_image': None, 'selected_image_name': None, 'meme_corpus': [],
        'current_translator_model': None, 'current_translator_tokenizer': None, 'current_direct_model': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def authenticate_user(phone_number: str, password: str) -> Optional[Dict]:
    """Authenticate user"""
    if not st.session_state.get('api_client'):
        st.session_state.api_client = DesiMemeAPIClient()
    
    try:
        result = st.session_state.api_client.login_for_access_token(phone_number, password)
        if result and "access_token" in result:
            st.session_state.api_client.set_auth_token(result["access_token"])
            return result
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
    return None

def register_user(phone_number: str, name: str, email: str, password: str) -> bool:
    """Register new user"""
    if not st.session_state.get('api_client'):
        st.session_state.api_client = DesiMemeAPIClient()
    
    try:
        otp_result = st.session_state.api_client.send_signup_otp(phone_number)
        if otp_result:
            st.session_state.pending_registration = {
                'phone_number': phone_number, 'name': name, 'email': email, 'password': password
            }
            return True
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
    return False

def verify_otp_and_complete_registration(otp_code: str) -> bool:
    """Complete registration after OTP verification"""
    if not st.session_state.get('pending_registration'):
        return False
    
    reg_data = st.session_state.pending_registration
    api_client = st.session_state.get('api_client', DesiMemeAPIClient())
    
    try:
        result = api_client.verify_signup_otp(
            reg_data['phone_number'], otp_code, reg_data['name'],
            reg_data['email'], reg_data['password'], True
        )
        if result:
            del st.session_state.pending_registration
            return True
    except Exception as e:
        st.error(f"OTP verification failed: {str(e)}")
    return False

def draw_text_with_outline(draw, text, x, y, font, fill_color="white", outline_color="black", outline_width=2):
    """Draw text with outline"""
    try:
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        draw.text((x, y), text, font=font, fill=fill_color)
    except Exception as e:
        st.error(f"Error rendering text: {str(e)}")

def create_meme(image_input, top_text, bottom_text, font_size, language_name, color_theme, 
                add_shadow=False, brightness=1.0, contrast=1.0):
    """Create meme with text overlay"""
    try:
        img = Image.open(image_input)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Apply enhancements
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        if add_shadow:
            img = img.filter(ImageFilter.SMOOTH)
        
        draw = ImageDraw.Draw(img)
        
        # Load font
        font_file = settings.get_font_path(language_name.lower())
        try:
            font = ImageFont.truetype(font_file, font_size)
        except Exception:
            try:
                font = ImageFont.truetype(settings.get_font_path('default'), font_size)
            except Exception:
                font = ImageFont.load_default()
        
        img_width, img_height = img.size
        colors = settings.COLOR_THEMES.get(color_theme, settings.COLOR_THEMES["Classic"])
        
        # Draw texts
        if top_text:
            top_text_upper = top_text.upper()
            bbox = draw.textbbox((0, 0), top_text_upper, font=font)
            x = (img_width - (bbox[2] - bbox[0])) // 2
            draw_text_with_outline(draw, top_text_upper, x, 20, font, colors["fill"], colors["outline"])
        
        if bottom_text:
            bottom_text_upper = bottom_text.upper()
            bbox = draw.textbbox((0, 0), bottom_text_upper, font=font)
            x = (img_width - (bbox[2] - bbox[0])) // 2
            y = img_height - (bbox[3] - bbox[1]) - 20
            draw_text_with_outline(draw, bottom_text_upper, x, y, font, colors["fill"], colors["outline"])
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        return img_byte_arr
        
    except Exception as e:
        st.error(f"Error creating meme: {str(e)}")
        return None

def save_to_db(language, text, color_theme, font_size, text_position):
    """Save meme data"""
    global use_session_storage
    
    if use_session_storage:
        entry = {
            'language': language, 'text': text, 'color_theme': color_theme,
            'font_size': font_size, 'text_position': text_position,
            'created_at': datetime.datetime.now()
        }
        st.session_state.meme_corpus.append(entry)
        show_success_message(f"Meme saved locally! Language: {language}")
    else:
        try:
            api_client = st.session_state.get('api_client')
            meme_data = {
                'title': f"Meme - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                'top_text': text.split('\n')[0] if text else '',
                'bottom_text': text.split('\n')[1] if '\n' in text else '',
                'language': language, 'color_theme': color_theme, 'font_size': font_size
            }
            api_client.create_meme(meme_data)
            show_success_message(f"Meme saved! Language: {language}")
        except Exception as e:
            st.error(f"Save failed: {str(e)}")

def get_corpus_with_stats():
    """Get corpus data and statistics"""
    global use_session_storage
    
    if use_session_storage:
        corpus_data = st.session_state.meme_corpus
        if not corpus_data:
            return pd.DataFrame(), [], 0
        
        df = pd.DataFrame(corpus_data)
        df.columns = ['Language', 'Text', 'Color Theme', 'Font Size', 'Text Position', 'Created At']
        lang_counts = df['Language'].value_counts()
        return df.tail(50), [(lang, count) for lang, count in lang_counts.items()], len(corpus_data)
    
    try:
        conn = db_pool.getconn()
        cursor = conn.cursor()
        cursor.execute("SELECT language, text, color_theme, font_size, text_position, created_at FROM meme_corpus ORDER BY created_at DESC LIMIT 50")
        df = pd.DataFrame(cursor.fetchall(), columns=['Language', 'Text', 'Color Theme', 'Font Size', 'Text Position', 'Created At'])
        cursor.execute("SELECT language, COUNT(*) FROM meme_corpus GROUP BY language ORDER BY COUNT(*) DESC")
        lang_stats = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM meme_corpus")
        total_count = cursor.fetchone()[0]
        cursor.close()
        db_pool.putconn(conn)
        return df, lang_stats, total_count
    except Exception:
        use_session_storage = True
        return get_corpus_with_stats()

def show_auth_pages():
    """Show authentication pages"""
    auth_tab = st.sidebar.selectbox("Choose Action", ["Login", "Sign Up"])
    
    if auth_tab == "Login":
        show_login_page()
    else:
        show_signup_page()

def show_login_page():
    """Show login page"""
    st.header("Login to Your Account")
    
    with st.form("login_form"):
        phone_number = st.text_input("Phone Number", placeholder="10-digit mobile number")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)
        
    if submitted:
        phone_valid, phone_msg = validate_phone_number(phone_number)
        if not phone_valid:
            show_error_message(phone_msg)
            return
        
        if not password:
            show_error_message("Password required")
            return
        
        with st.spinner("Logging in..."):
            result = authenticate_user(phone_number, password)
            if result:
                st.session_state.access_token = result["access_token"]
                st.session_state.user_data = result.get("user", {})
                st.session_state.authenticated = True
                show_success_message("Login successful!")
                st.rerun()
            else:
                show_error_message("Login failed. Check credentials.")

def show_signup_page():
    """Show signup page"""
    st.header("Create New Account")
    
    if st.session_state.get('pending_registration'):
        show_otp_verification_form()
        return
    
    with st.form("signup_form"):
        phone_number = st.text_input("Phone Number")
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        has_consent = st.checkbox("I agree to terms and conditions")
        submitted = st.form_submit_button("Create Account", use_container_width=True)

    if submitted:
        # Validate inputs
        phone_valid, phone_msg = validate_phone_number(phone_number)
        if not phone_valid:
            show_error_message(phone_msg)
            return
            
        if not validate_email(email) or not all([phone_number, name, email, password]):
            show_error_message("Please fill all fields correctly.")
            return
            
        if password != confirm_password:
            show_error_message("Passwords don't match.")
            return
            
        if not has_consent:
            show_error_message("Please agree to terms.")
            return
        
        with st.spinner("Sending OTP..."):
            if register_user(phone_number, name, email, password):
                show_success_message("OTP sent! Please verify.")
                st.rerun()

def show_otp_verification_form():
    """Show OTP verification"""
    st.subheader("Verify Your Phone Number")
    
    with st.form("otp_form"):
        otp_code = st.text_input("Enter OTP", max_chars=6)
        verify_submitted = st.form_submit_button("Verify OTP")
    
    if verify_submitted and otp_code:
        if len(otp_code) == 6 and otp_code.isdigit():
            with st.spinner("Verifying..."):
                if verify_otp_and_complete_registration(otp_code):
                    show_success_message("Registration completed! Please login.")
                    st.balloons()
                    st.rerun()
                else:
                    show_error_message("OTP verification failed.")

def show_main_app():
    """Show main application"""
    st.sidebar.title("Navigation")
    st.sidebar.markdown(f"**Welcome, {st.session_state.user_data.get('name', 'User')}!**")
    
    page = st.sidebar.selectbox("Choose a page", ["Home", "Meme Creator", "Dashboard", "Profile"])
    
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    
    if page == "Home":
        show_home_page()
    elif page == "Meme Creator":
        show_meme_creator_page()
    elif page == "Dashboard":
        show_dashboard()
    else:
        show_profile_page()

def show_home_page():
    """Show home page"""
    st.header("Welcome to Desi Meme Studio Pro!")
    
    st.markdown("""
    Create hilarious memes in Telugu and English with AI-powered translation!
    
    **Features:**
    - AI translation from English to Telugu
    - Voice input for hands-free creation
    - Multiple templates and themes
    - Community analytics
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### AI Translation")
        st.markdown("Advanced Hugging Face models")
    with col2:
        st.markdown("### Voice Input") 
        st.markdown("Hands-free meme creation")
    with col3:
        st.markdown("### Rich Templates")
        st.markdown("Professional meme templates")

def show_meme_creator_page():
    """Show meme creator"""
    st.header("Meme Creator Studio")
    
    with st.sidebar:
        st.subheader("Creator Controls")
        use_translation = st.checkbox("Enable AI Translation")
        
        if use_translation:
            translator_model, translator_tokenizer, direct_model = load_translation_model()
            st.session_state.current_translator_model = translator_model
            st.session_state.current_translator_tokenizer = translator_tokenizer
            st.session_state.current_direct_model = direct_model

    tab1, tab2 = st.tabs(["Create Meme", "Analytics"])

    with tab1:
        # Image selection
        st.markdown("### Choose Image")
        image_options = {f"Template {i}": f"images/meme{i}.jpg" for i in range(1, 11)}
        
        option = st.radio("Image Source:", ("Templates", "Upload"))
        
        if option == "Templates":
            cols = st.columns(5)
            for idx, (name, path) in enumerate(list(image_options.items())[:5]):
                with cols[idx]:
                    if os.path.exists(path):
                        st.image(Image.open(path), caption=name, use_container_width=True)
                        if st.button("Choose", key=f"select_{name}"):
                            st.session_state.selected_image = path
        else:
            uploaded = st.file_uploader("Upload Image", type=settings.ALLOWED_IMAGE_EXTENSIONS)
            if uploaded:
                st.session_state.selected_image = uploaded

        # Text input
        st.markdown("### Add Text")
        col1, col2 = st.columns(2)
        with col1:
            top_text = st.text_area("Top Text")
        with col2:
            bottom_text = st.text_area("Bottom Text")

        # Customization
        st.markdown("### Customize")
        col1, col2, col3 = st.columns(3)
        with col1:
            font_size = st.slider("Font Size", 20, 120, 50)
        with col2:
            color_theme = st.selectbox("Theme", list(settings.COLOR_THEMES.keys()))
        with col3:
            add_shadow = st.checkbox("Add Shadow")

        # Generate meme
        if st.button("Generate Meme", type="primary"):
            if st.session_state.selected_image and (top_text or bottom_text):
                with st.spinner("Creating meme..."):
                    # Auto-translate if enabled
                    final_top = translate_english_to_telugu(top_text) if use_translation else top_text
                    final_bottom = translate_english_to_telugu(bottom_text) if use_translation else bottom_text
                    
                    combined_text = f"{final_top} {final_bottom}".strip()
                    language_name = detect_language(combined_text)
                    
                    meme_image = create_meme(
                        st.session_state.selected_image, final_top, final_bottom,
                        font_size, language_name, color_theme, add_shadow
                    )
                    
                    if meme_image:
                        st.image(meme_image, caption="Your Meme!", use_container_width=True)
                        st.download_button(
                            "Download Meme", meme_image,
                            file_name=f"meme_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                        save_to_db(language_name, combined_text, color_theme, font_size, "top_bottom")

    with tab2:
        st.markdown("### Analytics")
        df, lang_stats, total_count = get_corpus_with_stats()
        
        if total_count > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Memes", total_count)
            col2.metric("Languages", len(lang_stats))
            col3.metric("Top Language", lang_stats[0][0] if lang_stats else "N/A")
            
            if lang_stats:
                lang_df = pd.DataFrame(lang_stats, columns=['Language', 'Count'])
                st.bar_chart(lang_df.set_index('Language'))

def show_dashboard():
    """Show dashboard"""
    st.header("Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    user_memes = st.session_state.meme_corpus
    df, lang_stats, total_count = get_corpus_with_stats()
    
    col1.metric("Your Memes", len(user_memes))
    col2.metric("Community Memes", total_count)
    col3.metric("Top Language", lang_stats[0][0] if lang_stats else "N/A")
    col4.metric("Today's Memes", len([m for m in user_memes if m.get('created_at', datetime.datetime.now()).date() == datetime.datetime.now().date()]))

def show_profile_page():
    """Show profile page"""
    st.header("User Profile")
    
    if st.session_state.user_data:
        st.markdown(f"**Name:** {st.session_state.user_data.get('name', 'N/A')}")
        st.markdown(f"**Phone:** {st.session_state.user_data.get('phone', 'N/A')}")
        st.markdown(f"**Email:** {st.session_state.user_data.get('email', 'N/A')}")
        
        user_memes = st.session_state.meme_corpus
        st.metric("Your Memes Created", len(user_memes))

def main():
    """Main application"""
    if not initialize_app():
        st.stop()
    
    init_session_state()
    
    if 'api_client' not in st.session_state:
        st.session_state.api_client = DesiMemeAPIClient()
    
    # Main header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title(settings.APP_NAME)
    st.markdown(f"### *Version {settings.APP_VERSION} - Create Viral Telugu & English Memes*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Authentication check
    if not st.session_state.authenticated:
        show_auth_pages()
    else:
        show_main_app()

def initialize_speech_recognition():
    """Initialize speech recognition"""
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        mic_list = sr.Microphone.list_microphone_names()
        if not mic_list:
            st.warning("No microphones detected")
            return None
        
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            st.success("Microphone ready!")
            
        return recognizer
    except ImportError:
        st.error("Speech recognition not installed. Run: pip install SpeechRecognition pyaudio")
    except Exception as e:
        st.warning(f"Speech recognition failed: {str(e)}")
    return None

def speech_to_text(recognizer):
    """Convert speech to text"""
    if not recognizer:
        return None
    
    try:
        import speech_recognition as sr
        with sr.Microphone() as source:
            st.info("Recording... Speak now!")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=8)
            
        st.info("Processing speech...")
        text = recognizer.recognize_google(audio)
        if text:
            st.success(f"Recognized: '{text}'")
            return text.strip()
            
    except sr.WaitTimeoutError:
        st.error("No speech detected")
    except sr.UnknownValueError:
        st.error("Could not understand audio")
    except Exception as e:
        st.error(f"Speech recognition error: {str(e)}")
    return None

# Run application
if __name__ == "__main__":
    main()