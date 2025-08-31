import re
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
import math
from datetime import datetime
import uuid
import base64
import hashlib
import os

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def validate_phone_number(phone: str) -> Tuple[bool, str]:
    """Validate Indian phone number format"""
    # Remove any spaces, dashes, or parentheses
    phone_clean = re.sub(r'[\s\-\(\)]', '', phone)
    
    # Indian phone number patterns
    if re.match(r'^[6-9]\d{9}$', phone_clean):  # 10 digits starting with 6-9
        return True, "Valid Indian mobile number"
    elif re.match(r'^91[6-9]\d{9}$', phone_clean):  # With country code 91
        return True, "Valid Indian mobile number with country code"
    elif re.match(r'^\+91[6-9]\d{9}$', phone_clean):  # With +91
        return True, "Valid Indian mobile number with country code"
    else:
        return False, "Please enter a valid Indian mobile number (10 digits starting with 6-9)"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def sanitize_input(text: str) -> str:
    """Basic input sanitization for meme text"""
    if not text:
        return ""
    # Remove potential harmful characters but keep Telugu characters and common punctuation
    sanitized = re.sub(r'[<>"\']', '', text.strip())
    # Remove excessive whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized)
    return sanitized.strip()

def format_datetime(datetime_str: str) -> str:
    """Format datetime string for display"""
    try:
        if isinstance(datetime_str, str):
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        else:
            dt = datetime_str
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return str(datetime_str)

def validate_file_upload(uploaded_file, max_size_mb: int = 10, allowed_types: List[str] = None) -> Tuple[bool, str]:
    """Validate uploaded file for meme creation"""
    if not uploaded_file:
        return True, "No file uploaded"
    
    if allowed_types is None:
        allowed_types = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']
    
    # Check file size
    if uploaded_file.size > max_size_mb * 1024 * 1024:
        return False, f"File size exceeds {max_size_mb}MB limit"
    
    # Check file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension not in allowed_types:
        return False, f"File type '{file_extension}' not supported. Allowed types: {', '.join(allowed_types)}"
    
    return True, "File is valid"

def create_meme_download_link(data: bytes, filename: str, text: str) -> str:
    """Create download link for meme data"""
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'

def show_success_message(message: str, icon: str = "🎉"):
    """Show styled success message"""
    st.markdown(f"""
    <div style="
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
        margin: 10px 0;
    ">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)

def show_error_message(message: str, icon: str = "❌"):
    """Show styled error message"""
    st.markdown(f"""
    <div style="
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        color: #721c24;
        margin: 10px 0;
    ">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)

def show_info_message(message: str, icon: str = "ℹ️"):
    """Show styled info message"""
    st.markdown(f"""
    <div style="
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 10px;
        color: #0c5460;
        margin: 10px 0;
    ">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)

def detect_language_script(text: str) -> str:
    """Detect if text is primarily English, Telugu, or mixed"""
    if not text or not text.strip():
        return "unknown"
    
    # Count different character types
    telugu_chars = sum(1 for char in text if '\u0C00' <= char <= '\u0C7F')
    english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
    total_alpha_chars = telugu_chars + english_chars
    
    if total_alpha_chars == 0:
        return "unknown"
    
    telugu_ratio = telugu_chars / total_alpha_chars
    
    if telugu_ratio >= 0.7:
        return "telugu"
    elif telugu_ratio <= 0.3:
        return "english"
    else:
        return "mixed"

def clean_translation_text(text: str) -> str:
    """Clean translated text output"""
    if not text:
        return ""
    
    # Remove common translation artifacts
    cleaned = text.strip()
    
    # Remove instruction patterns
    patterns_to_remove = [
        r'^translate\s*:?\s*',
        r'^translation\s*:?\s*',
        r'^english\s+to\s+telugu\s*:?\s*',
        r'^telugu\s*:?\s*',
        r'^output\s*:?\s*'
    ]
    
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def generate_meme_filename(top_text: str = "", bottom_text: str = "", user_id: str = None) -> str:
    """Generate a unique filename for meme"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a short hash from text content
    content = f"{top_text}_{bottom_text}".strip('_')
    if content:
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    else:
        content_hash = str(uuid.uuid4())[:8]
    
    if user_id:
        user_hash = hashlib.md5(user_id.encode()).hexdigest()[:6]
        return f"meme_{user_hash}_{timestamp}_{content_hash}.png"
    else:
        return f"meme_{timestamp}_{content_hash}.png"

def generate_upload_uuid() -> str:
    """Generate a unique UUID for file uploads"""
    return str(uuid.uuid4())

def chunk_file_data(file_data: bytes, chunk_size: int = 1024 * 1024) -> List[bytes]:
    """Split file data into chunks for upload"""
    chunks = []
    for i in range(0, len(file_data), chunk_size):
        chunk = file_data[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def filter_memes(memes: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
    """Filter memes based on criteria"""
    if not memes:
        return []
    
    filtered = memes.copy()
    
    # Filter by language
    if filters.get('language') and filters['language'] != "All":
        filtered = [m for m in filtered 
                   if m.get('language', '').lower() == filters['language'].lower()]
    
    # Filter by theme/color
    if filters.get('theme') and filters['theme'] != "All":
        filtered = [m for m in filtered 
                   if m.get('color_theme') == filters['theme']]
    
    # Filter by search term in text
    if filters.get('search_term'):
        search_lower = filters['search_term'].lower()
        filtered = [m for m in filtered if (
            search_lower in m.get('top_text', '').lower() or
            search_lower in m.get('bottom_text', '').lower() or
            search_lower in m.get('title', '').lower()
        )]
    
    # Filter by date range
    if filters.get('date_from'):
        date_from = filters['date_from']
        filtered = [m for m in filtered 
                   if datetime.fromisoformat(m.get('created_at', '1970-01-01')) >= date_from]
    
    if filters.get('date_to'):
        date_to = filters['date_to']
        filtered = [m for m in filtered 
                   if datetime.fromisoformat(m.get('created_at', '9999-12-31')) <= date_to]
    
    return filtered

def generate_meme_stats(memes: List[Dict]) -> Dict[str, Any]:
    """Generate statistics from meme data"""
    if not memes:
        return {}
    
    stats = {
        'total_memes': len(memes),
        'languages': len(set(m.get('language', 'Unknown') for m in memes)),
        'themes': len(set(m.get('color_theme', 'Unknown') for m in memes)),
        'total_likes': sum(m.get('likes_count', 0) for m in memes),
        'avg_likes': sum(m.get('likes_count', 0) for m in memes) / len(memes) if memes else 0
    }
    
    # Calculate top languages and themes
    from collections import Counter
    
    languages = [m.get('language', 'Unknown') for m in memes]
    themes = [m.get('color_theme', 'Unknown') for m in memes]
    
    stats['top_languages'] = dict(Counter(languages).most_common(5))
    stats['top_themes'] = dict(Counter(themes).most_common(5))
    
    # Recent activity
    now = datetime.now()
    today_memes = [m for m in memes 
                   if datetime.fromisoformat(m.get('created_at', '1970-01-01')).date() == now.date()]
    
    this_week_memes = [m for m in memes 
                       if (now - datetime.fromisoformat(m.get('created_at', '1970-01-01'))).days <= 7]
    
    stats['memes_today'] = len(today_memes)
    stats['memes_this_week'] = len(this_week_memes)
    
    return stats

def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text for display purposes"""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def format_number(number: int) -> str:
    """Format large numbers for display"""
    if number >= 1000000:
        return f"{number / 1000000:.1f}M"
    elif number >= 1000:
        return f"{number / 1000:.1f}K"
    else:
        return str(number)

def is_valid_image_url(url: str) -> bool:
    """Check if URL points to a valid image"""
    if not url:
        return False
    
    # Basic URL validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP address
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)', re.IGNORECASE
    )
    
    if not url_pattern.match(url):
        return False
    
    # Check if URL ends with common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']
    return any(url.lower().endswith(ext) for ext in image_extensions)

def get_color_contrast(bg_color: str, text_color: str) -> float:
    """Calculate color contrast ratio (simplified)"""
    # This is a simplified version - in production you'd use proper color contrast calculations
    bg_luminance = 0.5 if bg_color.lower() in ['white', '#ffffff', '#fff'] else 0.1
    text_luminance = 0.1 if text_color.lower() in ['black', '#000000', '#000'] else 0.9
    
    lighter = max(bg_luminance, text_luminance)
    darker = min(bg_luminance, text_luminance)
    
    return (lighter + 0.05) / (darker + 0.05)

def validate_meme_text(text: str, max_length: int = 100) -> Tuple[bool, str]:
    """Validate meme text input"""
    if not text or not text.strip():
        return True, "Text is optional"
    
    text = text.strip()
    
    if len(text) > max_length:
        return False, f"Text too long. Maximum {max_length} characters allowed."
    
    # Check for potentially problematic content (basic filter)
    inappropriate_patterns = [
        r'\b(hate|kill|die|death)\b',
        r'(fuck|shit|damn)',
        r'[<>{}[\]\\|`~]'  # Potentially problematic characters
    ]
    
    for pattern in inappropriate_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False, "Text contains inappropriate content"
    
    return True, "Text is valid"

def create_pagination_info(current_page: int, items_per_page: int, total_items: int) -> Dict[str, Any]:
    """Create pagination information"""
    total_pages = math.ceil(total_items / items_per_page) if total_items > 0 else 1
    start_item = (current_page - 1) * items_per_page + 1
    end_item = min(current_page * items_per_page, total_items)
    
    return {
        'current_page': current_page,
        'total_pages': total_pages,
        'items_per_page': items_per_page,
        'total_items': total_items,
        'start_item': start_item,
        'end_item': end_item,
        'has_previous': current_page > 1,
        'has_next': current_page < total_pages
    }

def safe_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with nested key support"""
    try:
        keys = key.split('.')
        value = dictionary
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError, IndexError):
        return default

def convert_to_indian_timezone(utc_datetime: datetime) -> datetime:
    """Convert UTC datetime to Indian Standard Time"""
    try:
        from datetime import timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        if utc_datetime.tzinfo is None:
            utc_datetime = utc_datetime.replace(tzinfo=timezone.utc)
        return utc_datetime.astimezone(ist)
    except Exception:
        return utc_datetime

def create_share_link(meme_id: str, base_url: str = None) -> str:
    """Create a shareable link for a meme"""
    if not base_url:
        base_url = "https://desimemes.app"  # Replace with your actual domain
    return f"{base_url}/meme/{meme_id}"

def log_user_action(user_id: str, action: str, details: Dict = None):
    """Log user action for analytics (simplified version)"""
    try:
        log_entry = {
            'user_id': user_id,
            'action': action,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        # In production, this would write to a proper logging system
        st.write(f"[LOG] {log_entry}")  # Debug only
    except Exception as e:
        st.error(f"Logging error: {str(e)}")

def get_device_info() -> Dict[str, str]:
    """Get basic device information from Streamlit"""
    # This is limited in Streamlit, but we can get some basic info
    try:
        import platform
        return {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'user_agent': 'streamlit_app'  # Limited info available
        }
    except Exception:
        return {'platform': 'unknown', 'python_version': 'unknown', 'user_agent': 'unknown'}