import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Configuration
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.desimemes.app")
    API_KEY = os.getenv("DESI_MEME_API_KEY")
    
    # App Configuration
    APP_NAME = os.getenv("APP_NAME", "Desi Meme Studio Pro")
    APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Meme Configuration
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']
    MAX_TEXT_LENGTH = 200
    MIN_TEXT_LENGTH = 1
    
    # File Upload Configuration
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    MAX_CHUNKS = 100
    UPLOAD_TIMEOUT = 300  # 5 minutes
    
    # Pagination
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100
    
    # Translation Configuration
    HUGGINGFACE_MODEL = "Meher2006/english-to-telugu-model"
    TRANSLATION_TIMEOUT = 10  # seconds
    FALLBACK_TRANSLATION = True
    
    # Supported Languages
    SUPPORTED_LANGUAGES = {
        "english": "English",
        "telugu": "తెలుగు",
        "hindi": "हिंदी",
        "tamil": "தமிழ்",
        "kannada": "ಕನ್ನಡ",
        "malayalam": "മലയാളം"
    }
    
    # Meme Templates
    TEMPLATE_CATEGORIES = [
        "Classic Memes", 
        "Bollywood", 
        "South Indian Cinema", 
        "Trending", 
        "Sports", 
        "Politics",
        "Daily Life",
        "Festival Memes",
        "Food Memes",
        "Tech Memes"
    ]
    
    # Color Themes
    COLOR_THEMES = {
        "Classic": {"fill": "white", "outline": "black", "shadow": "#888888"},
        "Fire": {"fill": "#FF4444", "outline": "white", "shadow": "#CC0000"},
        "Cool": {"fill": "#44AAFF", "outline": "white", "shadow": "#0066CC"},
        "Golden": {"fill": "#FFD700", "outline": "#8B4513", "shadow": "#B8860B"},
        "Neon": {"fill": "#00FF00", "outline": "black", "shadow": "#008000"},
        "Royal": {"fill": "#800080", "outline": "gold", "shadow": "#4B0082"},
        "Sunset": {"fill": "#FF6B35", "outline": "white", "shadow": "#CC4400"},
        "Ocean": {"fill": "#20B2AA", "outline": "navy", "shadow": "#008B8B"},
        "Forest": {"fill": "#228B22", "outline": "white", "shadow": "#006400"},
        "Rose": {"fill": "#FF69B4", "outline": "white", "shadow": "#C71585"}
    }
    
    # Font Configuration
    FONT_SIZES = {
        "small": 24,
        "medium": 36,
        "large": 48,
        "xlarge": 60,
        "xxlarge": 72
    }
    
    DEFAULT_FONT_SIZE = 36
    MIN_FONT_SIZE = 12
    MAX_FONT_SIZE = 100
    
    # Font Files
    FONT_MAPPING = {
        'telugu': 'fonts/NotoSansTelugu-Regular.ttf',
        'english': 'fonts/NotoSans-Regular.ttf',
        'hindi': 'fonts/NotoSansDevanagari-Regular.ttf',
        'tamil': 'fonts/NotoSansTamil-Regular.ttf',
        'kannada': 'fonts/NotoSansKannada-Regular.ttf',
        'malayalam': 'fonts/NotoSansMalayalam-Regular.ttf',
        'default': 'fonts/NotoSans-Regular.ttf'
    }
    
    # Font Download URLs
    FONT_URLS = {
        'fonts/NotoSansTelugu-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Regular.ttf',
        'fonts/NotoSans-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf',
        'fonts/NotoSansDevanagari-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf',
        'fonts/NotoSansTamil-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTamil/NotoSansTamil-Regular.ttf',
        'fonts/NotoSansKannada-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansKannada/NotoSansKannada-Regular.ttf',
        'fonts/NotoSansMalayalam-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansMalayalam/NotoSansMalayalam-Regular.ttf'
    }
    
    # Database Configuration (if using local database)
    DB_NAME = os.getenv("DB_NAME", "desi_memes")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    
    # Redis Configuration (for caching)
    REDIS_URL = os.getenv("REDIS_URL")
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    
    # Session Configuration
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_SESSIONS_PER_USER = 5
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = 60
    RATE_LIMIT_PER_HOUR = 1000
    RATE_LIMIT_PER_DAY = 10000
    
    # Analytics
    ENABLE_ANALYTICS = os.getenv("ENABLE_ANALYTICS", "True").lower() == "true"
    ANALYTICS_RETENTION_DAYS = 365
    
    # Social Features
    MAX_LIKES_PER_USER_PER_DAY = 100
    MAX_MEMES_PER_USER_PER_DAY = 50
    ENABLE_COMMENTS = os.getenv("ENABLE_COMMENTS", "False").lower() == "true"
    
    # Content Moderation
    ENABLE_CONTENT_FILTER = True
    PROFANITY_FILTER_LEVEL = "medium"  # low, medium, high
    MANUAL_REVIEW_THRESHOLD = 5  # reports needed for manual review
    
    # Image Processing
    MAX_IMAGE_WIDTH = 1920
    MAX_IMAGE_HEIGHT = 1080
    IMAGE_QUALITY = 85  # JPEG quality
    THUMBNAIL_SIZE = (300, 300)
    
    # Voice Recognition
    SPEECH_RECOGNITION_TIMEOUT = 10
    SPEECH_PHRASE_TIME_LIMIT = 8
    SUPPORTED_AUDIO_FORMATS = ['wav', 'mp3', 'ogg', 'flac']
    
    # AI/ML Configuration
    AI_MODEL_CACHE_SIZE = 1000  # MB
    TRANSLATION_CACHE_TTL = 3600  # seconds
    AI_REQUEST_TIMEOUT = 30  # seconds
    
    # Backup and Recovery
    BACKUP_ENABLED = os.getenv("BACKUP_ENABLED", "False").lower() == "true"
    BACKUP_INTERVAL_HOURS = 24
    BACKUP_RETENTION_DAYS = 30
    
    # CDN Configuration
    CDN_BASE_URL = os.getenv("CDN_BASE_URL")
    STATIC_FILES_CDN = os.getenv("STATIC_FILES_CDN", "False").lower() == "true"
    
    # Email Configuration (for notifications)
    SMTP_SERVER = os.getenv("SMTP_SERVER")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@desimemes.app")
    
    # Notification Settings
    ENABLE_EMAIL_NOTIFICATIONS = os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "False").lower() == "true"
    ENABLE_PUSH_NOTIFICATIONS = os.getenv("ENABLE_PUSH_NOTIFICATIONS", "False").lower() == "true"
    
    # Regional Settings
    DEFAULT_TIMEZONE = "Asia/Kolkata"
    DEFAULT_CURRENCY = "INR"
    DEFAULT_LOCALE = "en_IN"
    
    # Indian States (for regional categorization)
    INDIAN_STATES = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
        "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
        "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
        "Delhi", "Puducherry", "Jammu and Kashmir", "Ladakh", "Lakshadweep",
        "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu"
    ]
    
    # Popular Meme Formats
    MEME_FORMATS = [
        "top_bottom",  # Text at top and bottom
        "top_only",    # Text only at top
        "bottom_only", # Text only at bottom
        "center",      # Text in center
        "left_right",  # Text on left and right sides
        "overlay",     # Text overlay anywhere on image
        "speech_bubble", # Text in speech bubbles
        "caption"      # Caption style at bottom
    ]
    
    # Trending Algorithm
    TRENDING_TIME_WINDOW_HOURS = 24
    TRENDING_MIN_LIKES = 10
    TRENDING_DECAY_FACTOR = 0.8
    
    # Feature Flags
    ENABLE_VOICE_INPUT = os.getenv("ENABLE_VOICE_INPUT", "True").lower() == "true"
    ENABLE_AI_TRANSLATION = os.getenv("ENABLE_AI_TRANSLATION", "True").lower() == "true"
    ENABLE_MEME_SHARING = os.getenv("ENABLE_MEME_SHARING", "True").lower() == "true"
    ENABLE_MEME_DOWNLOAD = os.getenv("ENABLE_MEME_DOWNLOAD", "True").lower() == "true"
    ENABLE_USER_PROFILES = os.getenv("ENABLE_USER_PROFILES", "True").lower() == "true"
    ENABLE_MEME_CONTESTS = os.getenv("ENABLE_MEME_CONTESTS", "False").lower() == "true"
    
    # Security
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this")
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # CORS Settings
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8501",
        "https://desimemes.app",
        "https://www.desimemes.app"
    ]
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/desimemes.log")
    MAX_LOG_SIZE_MB = 100
    LOG_BACKUP_COUNT = 5
    
    # Health Check
    HEALTH_CHECK_ENDPOINT = "/health"
    SERVICE_NAME = "desi-meme-studio"
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database connection URL"""
        if not cls.DB_PASSWORD:
            return None
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
    
    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis connection URL"""
        if cls.REDIS_URL:
            return cls.REDIS_URL
        if cls.REDIS_PASSWORD:
            return f"redis://:{cls.REDIS_PASSWORD}@{cls.REDIS_HOST}:{cls.REDIS_PORT}/0"
        return f"redis://{cls.REDIS_HOST}:{cls.REDIS_PORT}/0"
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    @classmethod
    def get_font_path(cls, language: str) -> str:
        """Get font file path for a given language"""
        return cls.FONT_MAPPING.get(language.lower(), cls.FONT_MAPPING['default'])
    
    @classmethod
    def validate_settings(cls) -> List[str]:
        """Validate required settings and return list of missing/invalid ones"""
        issues = []
        
        if not cls.API_BASE_URL:
            issues.append("API_BASE_URL is not configured")
        
        if not cls.JWT_SECRET_KEY or cls.JWT_SECRET_KEY == "your-secret-key-change-this":
            issues.append("JWT_SECRET_KEY must be set to a secure value")
        
        if cls.ENABLE_EMAIL_NOTIFICATIONS and not all([cls.SMTP_SERVER, cls.SMTP_USERNAME, cls.SMTP_PASSWORD]):
            issues.append("Email notifications enabled but SMTP settings incomplete")
        
        if cls.MAX_IMAGE_SIZE > 50 * 1024 * 1024:  # 50MB
            issues.append("MAX_IMAGE_SIZE too large, consider reducing for better performance")
        
        return issues

# Create settings instance
settings = Settings()

# Validate settings on import (in debug mode)
if settings.DEBUG:
    validation_issues = settings.validate_settings()
    if validation_issues:
        print(f"⚠️  Configuration Issues Detected:")
        for issue in validation_issues:
            print(f"   - {issue}")
    else:
        print("✅ All settings validated successfully")