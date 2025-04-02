import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import hashlib
import numpy as np
import pytz
import requests
from bs4 import BeautifulSoup
from cryptography.fernet import Fernet
from google.cloud import translate_v2 as translate
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
    Dispatcher,
    Filters,
    JobQueue,
    MessageHandler,
    Updater,
)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache
import backoff
from prometheus_client import start_http_server, Counter, Gauge

# ##################################################
# ## ---------- تنظیمات پیشرفته سیستم ---------- ##
# ##################################################

# تنظیمات پایه لاگینگ
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# متریک‌های Prometheus
REQUEST_COUNTER = Counter('bot_requests_total', 'Total bot requests', ['endpoint'])
ERROR_COUNTER = Counter('bot_errors_total', 'Total bot errors', ['endpoint'])
ACTIVE_USERS_GAUGE = Gauge('bot_active_users', 'Currently active users')
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time taken to load models', ['model_name'])

# ##################################################
# ## ---------- ساختارهای داده اصلی ---------- ##
# ##################################################

@dataclass(frozen=True)
class SubscriptionPlan:
    """طرح اشتراک با تمام ویژگی‌ها و محدودیت‌ها"""
    name: str
    monthly_price: int
    annual_price: int
    features: List[str]
    rate_limit: int
    max_content_length: int
    advanced_analytics: bool
    api_access: bool
    priority_support: bool
    team_members: int
    color_code: str = "#4CAF50"  # رنگ پیش‌فرض

@dataclass
class Config:
    """پیکربندی اصلی سیستم با مقادیر پیش‌فرض"""
    BOT_TOKEN: str = os.getenv('BOT_TOKEN', '')
    MODEL_CACHE_DIR: str = "model_cache"
    USER_DATA_DIR: str = "user_data"
    BACKUP_DIR: str = "backups"
    MAX_CONTENT_LENGTH: int = 30000
    KEYWORD_SUGGESTIONS: int = 25
    CONTENT_TYPES: Dict[str, str] = field(default_factory=lambda: {
        "article": "مقاله",
        "product": "محصول",
        "landing": "صفحه فرود",
        "blog": "پست وبلاگ",
        "video": "محتوی ویدئویی",
        "social": "پست شبکه اجتماعی"
    })
    DEFAULT_RATE_LIMIT: int = 15
    ENCRYPTION_KEY: str = os.getenv('ENCRYPTION_KEY', '')
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY')
    TIMEZONE: str = "Asia/Tehran"
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    SENTRY_DSN: Optional[str] = os.getenv('SENTRY_DSN')
    
    SUBSCRIPTION_PLANS: Dict[str, SubscriptionPlan] = field(default_factory=lambda: {
        "free": SubscriptionPlan(
            "رایگان", 0, 0,
            ["پیشنهاد کلمات کلیدی", "تحلیل سئو پایه", "۵ درخواست در روز"],
            5, 2000, False, False, False, 1, "#9E9E9E"
        ),
        "pro": SubscriptionPlan(
            "حرفه‌ای", 99000, 990000,
            ["تمام امکانات پایه", "تولید محتوا", "بهینه‌سازی پیشرفته", "۵۰ درخواست در روز"],
            50, 10000, True, False, False, 3, "#2196F3"
        ),
        "enterprise": SubscriptionPlan(
            "سازمانی", 299000, 2990000,
            ["تمام امکانات", "پشتیبانی اختصاصی", "API دسترسی", "تیم ۵ نفره", "درخواست نامحدود"],
            1000, 30000, True, True, True, 5, "#FF5722"
        )
    })

# ##################################################
# ## ---------- سیستم مدیریت مدل‌ها ---------- ##
# ##################################################

class ModelManager:
    """مدیریت هوشمند مدل‌های یادگیری ماشین با بهینه‌سازی منابع"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_dirs()
        self.models = {}
        self.load_times = {}
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_REQUESTS)
        self.model_cache = TTLCache(maxsize=10, ttl=3600)  # کش مدل‌ها برای 1 ساعت
        
    def _setup_dirs(self):
        """ایجاد دایرکتوری‌های مورد نیاز با بررسی مجوزها"""
        try:
            os.makedirs(self.config.MODEL_CACHE_DIR, exist_ok=True, mode=0o755)
            os.makedirs(self.config.USER_DATA_DIR, exist_ok=True, mode=0o700)
        except Exception as e:
            logger.error(f"خطا در ایجاد دایرکتوری‌ها: {e}")
            raise

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def load_models(self):
        """بارگذاری مدل‌ها با قابلیت تلاش مجدد"""
        try:
            start_time = time.time()
            
            # بارگذاری موازی مدل‌ها
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self._load_keyword_model): "keyword",
                    executor.submit(self._load_content_model): "content",
                    executor.submit(self._load_similarity_model): "similarity",
                    executor.submit(self._load_optimization_model): "optimization",
                    executor.submit(self._load_translation_model): "translation"
                }
                
                for future in futures:
                    model_name = futures[future]
                    try:
                        self.models[model_name] = future.result()
                    except Exception as e:
                        logger.error(f"خطا در بارگذاری مدل {model_name}: {e}")
                        raise

            logger.info(f"تمامی مدل‌ها در {time.time() - start_time:.2f} ثانیه بارگذاری شدند")
            return True
        except Exception as e:
            logger.error(f"خطای بحرانی در بارگذاری مدل‌ها: {e}")
            return False

    @lru_cache(maxsize=1)
    def _load_keyword_model(self):
        """بارگذاری مدل پیشنهاد کلمات کلیدی"""
        logger.info("بارگذاری مدل کلمات کلیدی...")
        start_time = time.time()
        
        try:
            model = pipeline(
                "text2text-generation",
                model="google/flan-t5-large",
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_dir=self.config.MODEL_CACHE_DIR
            )
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.labels(model_name="keyword").set(load_time)
            logger.info(f"مدل کلمات کلیدی در {load_time:.2f} ثانیه بارگذاری شد")
            return model
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل کلمات کلیدی: {e}")
            raise

    @lru_cache(maxsize=1)
    def _load_content_model(self):
        """بارگذاری مدل تولید محتوا"""
        logger.info("بارگذاری مدل تولید محتوا...")
        start_time = time.time()
        
        try:
            model = pipeline(
                "text-generation",
                model="facebook/bart-large-cnn",
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_dir=self.config.MODEL_CACHE_DIR
            )
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.labels(model_name="content").set(load_time)
            logger.info(f"مدل تولید محتوا در {load_time:.2f} ثانیه بارگذاری شد")
            return model
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل تولید محتوا: {e}")
            raise

    # ادامه مدل‌های دیگر...

    def unload_model(self, model_name: str):
        """تخلیه مدل از حافظه با مدیریت صحیح منابع"""
        if model_name in self.models:
            del self.models[model_name]
            if model_name == "keyword":
                self._load_keyword_model.cache_clear()
            elif model_name == "content":
                self._load_content_model.cache_clear()
            logger.info(f"مدل {model_name} با موفقیت تخلیه شد")

    async def async_predict(self, model_name: str, input_data: Any):
        """پیش‌بینی ناهمزمان با مدیریت صف"""
        if model_name not in self.models:
            raise ValueError(f"مدل {model_name} بارگذاری نشده است")
        
        try:
            future = self.executor.submit(self.models[model_name], input_data)
            return await asyncio.wrap_future(future)
        except Exception as e:
            logger.error(f"خطا در پیش‌بینی مدل {model_name}: {e}")
            raise

# ##################################################
# ## ---------- سیستم امنیتی پیشرفته ---------- ##
# ##################################################

class SecurityManager:
    """مدیریت امنیتی پیشرفته با رمزنگاری و احراز هویت"""
    
    def __init__(self, encryption_key: str):
        if not encryption_key:
            raise ValueError("کلید رمزنگاری نمی‌تواند خالی باشد")
        
        if len(encryption_key) < 32:
            raise ValueError("کلید رمزنگاری باید حداقل ۳۲ کاراکتر باشد")
            
        self.cipher = Fernet(Fernet.generate_key())
        self.hmac_key = os.urandom(32)
        self.token_cache = TTLCache(maxsize=1000, ttl=3600)
        
    def encrypt_data(self, data: str) -> str:
        """رمزنگاری داده با احراز هویت پیام (HMAC)"""
        if not data:
            raise ValueError("داده ورودی نمی‌تواند خالی باشد")
            
        encrypted = self.cipher.encrypt(data.encode())
        hmac = hmac.new(self.hmac_key, encrypted, hashlib.sha256).hexdigest()
        return f"{encrypted.decode()}:{hmac}"

    def decrypt_data(self, encrypted_data: str) -> str:
        """رمزگشایی داده با بررسی صحت پیام"""
        if not encrypted_data:
            raise ValueError("داده رمزنگاری شده نمی‌تواند خالی باشد")
            
        try:
            encrypted, hmac_value = encrypted_data.split(":")
            if not encrypted or not hmac_value:
                raise ValueError("فرمت داده رمزنگاری شده نامعتبر است")
                
            calculated_hmac = hmac.new(self.hmac_key, encrypted.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(calculated_hmac, hmac_value):
                raise ValueError("عدم تطابق HMAC - احتمال دستکاری داده")
                
            return self.cipher.decrypt(encrypted.encode()).decode()
        except Exception as e:
            logger.error(f"خطا در رمزگشایی داده: {e}")
            raise

    def generate_token(self, user_id: int, expires_in: int = 3600) -> str:
        """تولید توکن امنیتی با زمان انقضا"""
        token = hashlib.sha256(f"{user_id}{time.time()}{os.urandom(16)}".encode()).hexdigest()
        self.token_cache[token] = {
            "user_id": user_id,
            "expires_at": time.time() + expires_in
        }
        return token

    def validate_token(self, token: str) -> Optional[int]:
        """اعتبارسنجی توکن و بازگرداندن شناسه کاربر"""
        if token in self.token_cache:
            token_data = self.token_cache[token]
            if token_data["expires_at"] > time.time():
                return token_data["user_id"]
        return None

# ##################################################
# ## ---------- سیستم تحلیل سئو پیشرفته ---------- ##
# ##################################################

class SEOAnalytics:
    """تحلیل پیشرفته سئو با الگوریتم‌های بهینه"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.readability_cache = TTLCache(maxsize=1000, ttl=3600)
        self.keyword_cache = TTLCache(maxsize=1000, ttl=1800)
        
    @staticmethod
    def preprocess_text(text: str) -> str:
        """پیش‌پردازش متن برای تحلیل"""
        if not text:
            return ""
            
        # حذف فاصله‌های اضافی
        text = re.sub(r'\s+', ' ', text).strip()
        # نرمال‌سازی نویسه‌های فارسی
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        return text

    def calculate_readability(self, text: str, lang: str = 'fa') -> float:
        """محاسبه سطح خوانایی با کش کردن نتایج"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.readability_cache:
            return self.readability_cache[cache_key]
            
        text = self.preprocess_text(text)
        if not text:
            return 0.0
            
        try:
            if lang == 'fa':
                # الگوریتم بهبودیافته برای فارسی
                words = text.split()
                sentences = [s for s in re.split(r'[.!?؟]+', text) if s.strip()]
                
                if not words or not sentences:
                    return 0.0
                    
                words_count = len(words)
                sentences_count = len(sentences)
                syllables_count = sum(self.count_syllables(word) for word in words)
                
                avg_words = words_count / sentences_count
                avg_syllables = syllables_count / words_count
                
                readability = 206.835 - (1.3 * avg_words) - (60.6 * avg_syllables)
                result = max(0, min(100, readability))
            else:
                # الگوریتم برای انگلیسی
                result = textstat.flesch_reading_ease(text)
                
            self.readability_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"خطا در محاسبه خوانایی: {e}")
            return 0.0

    def keyword_density(self, text: str, keywords: List[str]) -> Dict[str, float]:
        """محاسبه تراکم کلمات کلیدی با کش و پیش‌پردازش"""
        if not text or not keywords:
            return {}
            
        cache_key = hashlib.md5((text + ''.join(sorted(keywords))).encode()).hexdigest()
        if cache_key in self.keyword_cache:
            return self.keyword_cache[cache_key]
            
        text = self.preprocess_text(text).lower()
        words = text.split()
        total_words = len(words)
        result = {}
        
        for keyword in keywords:
            keyword = self.preprocess_text(keyword).lower()
            if not keyword:
                continue
                
            # جستجوی پیشرفته با در نظر گرفتن صورت‌های مختلف
            pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
            count = len(pattern.findall(text))
            density = (count / total_words) * 100 if total_words > 0 else 0
            result[keyword] = round(density, 2)
            
        self.keyword_cache[cache_key] = result
        return result

    def analyze_competition(self, url: str, user_content: str) -> Dict[str, Any]:
        """تحلیل پیشرفته رقابت با وبسایت‌های دیگر"""
        try:
            # دریافت محتوای رقیب
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # استخراج بخش‌های اصلی
            competitor_content = ' '.join([
                tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3'])
            ])
            
            # تحلیل مقایسه‌ای
            return self.compare_contents(user_content, competitor_content)
        except Exception as e:
            logger.error(f"خطا در تحلیل رقابت: {e}")
            return {"error": str(e)}

    def compare_contents(self, content1: str, content2: str) -> Dict[str, Any]:
        """مقایسه پیشرفته دو محتوا"""
        try:
            # محاسبه شباهت متنی
            model = self.model_manager.models.get("similarity")
            if not model:
                raise ValueError("مدل مقایسه متون بارگذاری نشده")
                
            emb1 = model.encode([self.preprocess_text(content1)])
            emb2 = model.encode([self.preprocess_text(content2)])
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            # تحلیل کلمات کلیدی
            keywords1 = self.extract_keywords(content1)
            keywords2 = self.extract_keywords(content2)
            
            # تحلیل خوانایی
            readability1 = self.calculate_readability(content1)
            readability2 = self.calculate_readability(content2)
            
            return {
                "similarity_score": round(similarity * 100, 2),
                "content1_keywords": keywords1[:10],
                "content2_keywords": keywords2[:10],
                "missing_keywords": list(set(keywords2) - set(keywords1))[:10],
                "content1_readability": readability1,
                "content2_readability": readability2,
                "suggestions": self.generate_suggestions(content1, content2)
            }
        except Exception as e:
            logger.error(f"خطا در مقایسه محتوا: {e}")
            return {"error": str(e)}


# ##################################################
# ## ------ سیستم مدیریت کاربران پیشرفته ------ ##
# ##################################################

class UserProfile:
    """پروفایل کاربری پیشرفته با قابلیت‌های مدیریت اشتراک و تنظیمات"""
    
    def __init__(self, user_id: int, config: Config, security_manager: SecurityManager):
        self.user_id = user_id
        self.config = config
        self.security = security_manager
        self.data = {
            "subscription": {
                "plan": "free",
                "start_date": datetime.now(pytz.timezone(config.TIMEZONE)).isoformat(),
                "expiry_date": None,
                "payment_method": None,
                "renewal": False
            },
            "preferences": {
                "language": "fa",
                "content_style": "formal",
                "tone": "professional",
                "notifications": True,
                "dark_mode": False
            },
            "usage": {
                "daily_requests": 0,
                "monthly_requests": 0,
                "total_requests": 0,
                "last_request": None,
                "request_history": []
            },
            "content": {
                "saved_items": [],
                "favorites": [],
                "collections": {}
            },
            "security": {
                "last_login": datetime.now(pytz.timezone(config.TIMEZONE)).isoformat(),
                "login_history": [],
                "two_fa_enabled": False
            }
        }
        self.lock = threading.Lock()
        
    def update_usage(self, request_type: str) -> bool:
        """به‌روزرسانی آمار استفاده کاربر با مدیریت همزمانی"""
        with self.lock:
            now = datetime.now(pytz.timezone(self.config.TIMEZONE))
            today = now.date()
            last_date = datetime.fromisoformat(self.data["usage"]["last_request"]).date() if self.data["usage"]["last_request"] else None
            
            # بازنشانی آمار روزانه اگر روز جدید باشد
            if last_date is None or last_date != today:
                self.data["usage"]["daily_requests"] = 0
                
            # بازنشانی آمار ماهانه اگر ماه جدید باشد
            if last_date is None or last_date.month != today.month:
                self.data["usage"]["monthly_requests"] = 0
                
            # بررسی محدودیت اشتراک
            plan = self.config.SUBSCRIPTION_PLANS.get(self.data["subscription"]["plan"])
            if not plan:
                return False
                
            if self.data["usage"]["daily_requests"] >= plan.rate_limit:
                return False
                
            # به‌روزرسانی آمار
            self.data["usage"]["daily_requests"] += 1
            self.data["usage"]["monthly_requests"] += 1
            self.data["usage"]["total_requests"] += 1
            self.data["usage"]["last_request"] = now.isoformat()
            self.data["usage"]["request_history"].append({
                "type": request_type,
                "timestamp": now.isoformat(),
                "status": "completed"
            })
            
            return True

    def can_make_request(self, request_type: str) -> Tuple[bool, str]:
        """بررسی امکان انجام درخواست با جزئیات خطا"""
        plan = self.config.SUBSCRIPTION_PLANS.get(self.data["subscription"]["plan"])
        if not plan:
            return False, "طرح اشتراک نامعتبر"
            
        if self.data["usage"]["daily_requests"] >= plan.rate_limit:
            reset_time = (datetime.now(pytz.timezone(self.config.TIMEZONE)) + timedelta(days=1)
            reset_str = reset_time.strftime("%H:%M")
            return False, f"محدودیت درخواست روزانه. تا ساعت {reset_str} صبر کنید"
            
        return True, ""

    def save_content(self, content_data: Dict) -> str:
        """ذخیره محتوای کاربر با تولید شناسه یکتا"""
        content_id = self.security.hash_data(f"{content_data['title']}{time.time()}")
        
        content_item = {
            "id": content_id,
            "type": content_data.get("type", "article"),
            "title": content_data.get("title", "بدون عنوان"),
            "content": content_data["content"],
            "tags": content_data.get("tags", []),
            "created_at": datetime.now(pytz.timezone(self.config.TIMEZONE)).isoformat(),
            "modified_at": datetime.now(pytz.timezone(self.config.TIMEZONE)).isoformat(),
            "metadata": content_data.get("metadata", {})
        }
        
        with self.lock:
            self.data["content"]["saved_items"].append(content_item)
            
        return content_id

    def get_content(self, content_id: str) -> Optional[Dict]:
        """دریافت محتوای ذخیره شده با بررسی اعتبار"""
        with self.lock:
            for item in self.data["content"]["saved_items"]:
                if item["id"] == content_id:
                    return item
        return None

    def upgrade_subscription(self, plan_id: str, payment_method: str, duration: str = "monthly") -> bool:
        """ارتقاء اشتراک کاربر با مدیریت پرداخت"""
        if plan_id not in self.config.SUBSCRIPTION_PLANS:
            return False
            
        now = datetime.now(pytz.timezone(self.config.TIMEZONE))
        
        with self.lock:
            self.data["subscription"]["plan"] = plan_id
            self.data["subscription"]["payment_method"] = payment_method
            self.data["subscription"]["start_date"] = now.isoformat()
            
            if duration == "monthly":
                expiry = now + timedelta(days=30)
            else:  # annual
                expiry = now + timedelta(days=365)
                
            self.data["subscription"]["expiry_date"] = expiry.isoformat()
            self.data["subscription"]["renewal"] = True
            
        return True

# ##################################################
# ## ----- سیستم پرداخت و اشتراک پیشرفته ----- ##
# ##################################################

class PaymentManager:
    """مدیریت پرداخت‌ها و اشتراک‌ها با قابلیت اتصال به دروازه‌های پرداخت"""
    
    def __init__(self, config: Config):
        self.config = config
        self.plans = config.SUBSCRIPTION_PLANS
        self.payment_providers = {
            "zarinpal": self._init_zarinpal(),
            "idpay": self._init_idpay()
        }
        self.receipts = TTLCache(maxsize=1000, ttl=86400)  # کش رسیدها برای 24 ساعت
        
    def _init_zarinpal(self):
        """تنظیمات دروازه پرداخت زرین‌پال"""
        return {
            "api_key": os.getenv("ZARINPAL_API_KEY"),
            "sandbox": os.getenv("ZARINPAL_SANDBOX", "false").lower() == "true",
            "callback_url": os.getenv("ZARINPAL_CALLBACK_URL")
        }
        
    def _init_idpay(self):
        """تنظیمات دروازه پرداخت آیدی پی"""
        return {
            "api_key": os.getenv("IDPAY_API_KEY"),
            "sandbox": os.getenv("IDPAY_SANDBOX", "false").lower() == "true",
            "callback_url": os.getenv("IDPAY_CALLBACK_URL")
        }

    def initiate_payment(self, user_id: int, plan_id: str, provider: str = "zarinpal") -> Optional[str]:
        """شروع فرآیند پرداخت و بازگرداندن لینک پرداخت"""
        if provider not in self.payment_providers:
            return None
            
        plan = self.plans.get(plan_id)
        if not plan:
            return None
            
        amount = plan.monthly_price
        description = f"ارتقاء به طرح {plan.name}"
        
        try:
            if provider == "zarinpal":
                payment_url = self._zarinpal_payment(user_id, amount, description)
            elif provider == "idpay":
                payment_url = self._idpay_payment(user_id, amount, description)
            else:
                return None
                
            return payment_url
        except Exception as e:
            logger.error(f"خطا در شروع پرداخت: {e}")
            return None

    def verify_payment(self, payment_id: str, provider: str) -> Tuple[bool, Dict]:
        """اعتبارسنجی پرداخت و بازگرداندن نتیجه"""
        if provider not in self.payment_providers:
            return False, {"error": "پروایدر پرداخت نامعتبر"}
            
        try:
            if provider == "zarinpal":
                return self._verify_zarinpal(payment_id)
            elif provider == "idpay":
                return self._verify_idpay(payment_id)
            else:
                return False, {"error": "پروایدر پرداخت نامعتبر"}
        except Exception as e:
            logger.error(f"خطا در تأیید پرداخت: {e}")
            return False, {"error": str(e)}

    def _zarinpal_payment(self, user_id: int, amount: int, description: str) -> Optional[str]:
        """پیاده‌سازی پرداخت از طریق زرین‌پال"""
        # پیاده‌سازی واقعی نیاز به کلاس زرین‌پال دارد
        payment_url = f"https://zarinpal.com/pg/StartPay/{user_id}_{int(time.time())}"
        self.receipts[payment_url] = {
            "user_id": user_id,
            "amount": amount,
            "description": description,
            "timestamp": time.time()
        }
        return payment_url

    def _verify_zarinpal(self, payment_id: str) -> Tuple[bool, Dict]:
        """اعتبارسنجی پرداخت زرین‌پال"""
        # پیاده‌سازی واقعی نیاز به کلاس زرین‌پال دارد
        return True, {
            "success": True,
            "amount": self.receipts.get(payment_id, {}).get("amount", 0),
            "transaction_id": f"zarinpal_{int(time.time())}"
        }

# ##################################################
# ## ----- سیستم گزارش‌گیری و آنالیتیکس ----- ##
# ##################################################

class AnalyticsManager:
    """مدیریت گزارش‌گیری و آمارهای پیشرفته"""
    
    def __init__(self, config: Config):
        self.config = config
        self.report_cache = TTLCache(maxsize=100, ttl=3600)
        
    def generate_seo_report(self, analysis_data: Dict, user_id: int = None) -> bytes:
        """تولید گزارش PDF حرفه‌ای"""
        cache_key = hashlib.md5(json.dumps(analysis_data).encode()).hexdigest()
        
        if cache_key in self.report_cache:
            return self.report_cache[cache_key]
            
        try:
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            
            # هدر گزارش
            self._draw_header(c, "گزارش تحلیل سئو", user_id)
            
            # محتوای گزارش
            y_position = 650
            for section, content in analysis_data.items():
                y_position = self._draw_section(c, section, content, y_position)
                if y_position < 100:
                    c.showPage()
                    self._draw_header(c, "گزارش تحلیل سئو (ادامه)", user_id)
                    y_position = 650
                    
            # فوتر گزارش
            self._draw_footer(c)
            
            c.save()
            pdf_data = buffer.getvalue()
            self.report_cache[cache_key] = pdf_data
            return pdf_data
        except Exception as e:
            logger.error(f"خطا در تولید گزارش: {e}")
            raise

    def _draw_header(self, c, title: str, user_id: int = None):
        """رسم هدر گزارش"""
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, title)
        
        c.setFont("Helvetica", 10)
        c.drawString(100, 730, f"تاریخ تولید: {datetime.now(pytz.timezone(self.config.TIMEZONE)).strftime('%Y-%m-%d %H:%M')}")
        
        if user_id:
            c.drawString(100, 710, f"شناسه کاربر: {user_id}")
            
        c.line(100, 700, 500, 700)

    def _draw_section(self, c, title: str, content: Any, y_pos: int) -> int:
        """رسم یک بخش از گزارش"""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y_pos, title)
        y_pos -= 25
        
        c.setFont("Helvetica", 12)
        if isinstance(content, dict):
            for key, value in content.items():
                c.drawString(120, y_pos, f"{key}: {value}")
                y_pos -= 20
        elif isinstance(content, list):
            for item in content:
                c.drawString(120, y_pos, f"- {item}")
                y_pos -= 20
        else:
            c.drawString(120, y_pos, str(content))
            y_pos -= 20
            
        return y_pos - 10  # فاصله قبل از بخش بعدی

    def _draw_footer(self, c):
        """رسم فوتر گزارش"""
        c.setFont("Helvetica", 8)
        c.drawString(100, 30, f"تولید شده توسط ربات سئوکار - {datetime.now(pytz.timezone(self.config.TIMEZONE)).strftime('%Y-%m-%d')}")

# ##################################################
# ## ----- سیستم یکپارچه‌سازی گوگل ----- ##
# ##################################################

class GoogleIntegration:
    """مدیریت یکپارچه‌سازی با سرویس‌های گوگل"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.translate_client = translate.Client(api_key) if api_key else None
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        
    def translate_text(self, text: str, target_lang: str = 'fa') -> str:
        """ترجمه متن با استفاده از Google Cloud Translation"""
        if not self.translate_client:
            raise ValueError("کلید API ترجمه تنظیم نشده است")
            
        try:
            result = self.translate_client.translate(
                text,
                target_language=target_lang,
                format_='text'
            )
            return result['translatedText']
        except Exception as e:
            logger.error(f"خطا در ترجمه متن: {e}")
            raise

    def get_search_console_data(self, site_url: str, start_date: str, end_date: str) -> Dict:
        """دریافت داده‌های سرچ کنسول گوگل"""
        try:
            # شبیه‌سازی درخواست واقعی
            params = {
                "siteUrl": site_url,
                "startDate": start_date,
                "endDate": end_date,
                "dimensions": ["query", "page"],
                "rowLimit": 100
            }
            
            # در یک سیستم واقعی، اینجا درخواست به API گوگل ارسال می‌شود
            return {
                "clicks": 1200,
                "impressions": 8500,
                "ctr": 0.14,
                "position": 8.3,
                "top_keywords": [
                    {"keyword": "آموزش سئو", "clicks": 320},
                    {"keyword": "بهینه‌سازی سایت", "clicks": 210},
                    {"keyword": "ربات سئو", "clicks": 150}
                ],
                "top_pages": [
                    {"page": "/blog", "clicks": 450},
                    {"page": "/products", "clicks": 320},
                    {"page": "/contact", "clicks": 210}
                ]
            }
        except Exception as e:
            logger.error(f"خطا در دریافت داده‌های سرچ کنسول: {e}")
            raise

    def get_analytics_data(self, view_id: str, start_date: str, end_date: str) -> Dict:
        """دریافت داده‌های گوگل آنالیتیکس"""
        try:
            # شبیه‌سازی درخواست واقعی
            params = {
                "viewId": view_id,
                "dateRanges": [{"startDate": start_date, "endDate": end_date}],
                "metrics": [
                    {"expression": "ga:sessions"},
                    {"expression": "ga:users"},
                    {"expression": "ga:pageviews"}
                ],
                "dimensions": [{"name": "ga:pagePath"}]
            }
            
            # در یک سیستم واقعی، اینجا درخواست به API گوگل ارسال می‌شود
            return {
                "sessions": 4500,
                "users": 3200,
                "pageviews": 12000,
                "avg_session_duration": "00:02:45",
                "bounce_rate": 0.42,
                "top_pages": [
                    {"page": "/blog/seo-guide", "views": 1200},
                    {"page": "/products/seo-tool", "views": 850},
                    {"page": "/contact", "views": 620}
                ]
            }
        except Exception as e:
            logger.error(f"خطا در دریافت داده‌های آنالیتیکس: {e}")
            raise

# ##################################################
# ## ------ ربات تلگرام با معماری پیشرفته ------ ##
# ##################################################

class SEOAssistantBot:
    """ربات هوشمند سئوکار با معماری مقیاس‌پذیر و پیشرفته"""
    
    def __init__(self, config: Config):
        # تأیید تنظیمات ضروری
        if not config.BOT_TOKEN:
            raise ValueError("توکن ربات باید تنظیم شود")
        if not config.ENCRYPTION_KEY:
            raise ValueError("کلید رمزنگاری باید تنظیم شود")
            
        self.config = config
        self._init_managers()
        self._init_telegram()
        self._setup_metrics()
        self._load_user_data()
        
        logger.info("تنظیمات اولیه ربات با موفقیت انجام شد")

    def _init_managers(self):
        """مقداردهی مدیران سرویس"""
        self.model_manager = ModelManager(self.config)
        self.security_manager = SecurityManager(self.config.ENCRYPTION_KEY)
        self.payment_manager = PaymentManager(self.config)
        self.analytics_manager = AnalyticsManager(self.config)
        self.google_integration = GoogleIntegration(self.config.GOOGLE_API_KEY) if self.config.GOOGLE_API_KEY else None
        
        # بارگذاری مدل‌های ML
        if not self.model_manager.load_models():
            raise RuntimeError("خطا در بارگذاری مدل‌های یادگیری ماشین")

    def _init_telegram(self):
        """تنظیمات ارتباط با تلگرام"""
        self.updater = Updater(
            self.config.BOT_TOKEN,
            use_context=True,
            request_kwargs={
                'read_timeout': self.config.REQUEST_TIMEOUT,
                'connect_timeout': self.config.REQUEST_TIMEOUT
            }
        )
        self.dp = self.updater.dispatcher
        self.job_queue = self.updater.job_queue
        
        # تنظیم هندلرهای خطا
        self.dp.add_error_handler(self._handle_error)
        
        # تنظیم هندلرهای دستورات
        self._setup_command_handlers()
        self._setup_message_handlers()
        self._setup_callback_handlers()

    def _setup_metrics(self):
        """تنظیم سیستم مانیتورینگ"""
        if os.getenv('ENABLE_METRICS', 'false').lower() == 'true':
            start_http_server(8000)
            logger.info("سیستم متریک‌ها روی پورت 8000 راه‌اندازی شد")

    def _load_user_data(self):
        """بارگذاری داده کاربران از ذخیره‌سازی"""
        try:
            data_dir = Path(self.config.USER_DATA_DIR)
            for user_file in data_dir.glob('*.json'):
                try:
                    user_id = int(user_file.stem)
                    with open(user_file, 'r', encoding='utf-8') as f:
                        encrypted_data = json.load(f)
                        decrypted_data = json.loads(self.security_manager.decrypt_data(encrypted_data))
                        self.user_profiles[user_id] = UserProfile(user_id, self.config, self.security_manager)
                        self.user_profiles[user_id].data = decrypted_data
                except Exception as e:
                    logger.error(f"خطا در بارگذاری داده کاربر {user_file.stem}: {e}")
            
            logger.info(f"داده {len(self.user_profiles)} کاربر با موفقیت بارگذاری شد")
            ACTIVE_USERS_GAUGE.set(len(self.user_profiles))
        except Exception as e:
            logger.error(f"خطا در بارگذاری داده کاربران: {e}")
            raise

    def _setup_command_handlers(self):
        """تنظیم هندلرهای دستورات"""
        handlers = [
            CommandHandler('start', self._handle_start),
            CommandHandler('help', self._handle_help),
            CommandHandler('menu', self._handle_menu),
            CommandHandler('keywords', self._handle_keywords),
            CommandHandler('content', self._handle_content),
            CommandHandler('optimize', self._handle_optimize),
            CommandHandler('analyze', self._handle_analyze),
            CommandHandler('compare', self._handle_compare),
            CommandHandler('competitor', self._handle_competitor),
            CommandHandler('save', self._handle_save),
            CommandHandler('list', self._handle_list),
            CommandHandler('get', self._handle_get),
            CommandHandler('profile', self._handle_profile),
            CommandHandler('subscribe', self._handle_subscribe),
            CommandHandler('upgrade', self._handle_upgrade),
            CommandHandler('language', self._handle_language),
            CommandHandler('report', self._handle_report),
            CommandHandler('stats', self._handle_stats),
            CommandHandler('backup', self._handle_backup)
        ]
        
        for handler in handlers:
            self.dp.add_handler(handler)

    def _setup_message_handlers(self):
        """تنظیم هندلرهای پیام‌های متنی"""
        self.dp.add_handler(MessageHandler(Filters.text & ~Filters.command, self._handle_text_message))
        self.dp.add_handler(MessageHandler(Filters.document, self._handle_document))

    def _setup_callback_handlers(self):
        """تنظیم هندلرهای callback دکمه‌ها"""
        self.dp.add_handler(CallbackQueryHandler(self._handle_callback))

    def _handle_error(self, update: Update, context: CallbackContext):
        """مدیریت متمرکز خطاها"""
        error = context.error
        user_id = update.effective_user.id if update.effective_user else None
        
        logger.error(f"خطا در پردازش درخواست کاربر {user_id}: {error}", exc_info=error)
        ERROR_COUNTER.labels(endpoint=update.message.text.split()[0] if update.message else 'unknown').inc()
        
        try:
            if user_id:
                context.bot.send_message(
                    chat_id=user_id,
                    text="⚠️ خطایی در پردازش درخواست شما رخ داد. لطفاً مجدداً تلاش کنید."
                )
        except Exception as e:
            logger.error(f"خطا در ارسال پیام خطا به کاربر: {e}")

    # ##################################################
    # ## -------- دستورات اصلی ربات -------- ##
    # ##################################################

    def _handle_start(self, update: Update, context: CallbackContext):
        """مدیریت دستور /start"""
        REQUEST_COUNTER.labels(endpoint='start').inc()
        user = update.effective_user
        user_profile = self._get_user_profile(user.id)
        
        if not self._check_rate_limit(user.id, 'start'):
            update.message.reply_text("⏳ تعداد درخواست‌های شما امروز به حد مجاز رسیده است.")
            return
        
        welcome_msg = self._generate_welcome_message(user)
        keyboard = self._generate_main_keyboard()
        
        update.message.reply_text(
            welcome_msg,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    def _handle_help(self, update: Update, context: CallbackContext):
        """مدیریت دستور /help"""
        REQUEST_COUNTER.labels(endpoint='help').inc()
        user_profile = self._get_user_profile(update.effective_user.id)
        
        if not self._check_rate_limit(update.effective_user.id, 'help'):
            update.message.reply_text("⏳ تعداد درخواست‌های شما امروز به حد مجاز رسیده است.")
            return
        
        help_text = self._generate_help_text(user_profile)
        update.message.reply_text(help_text, parse_mode="Markdown")

    def _handle_keywords(self, update: Update, context: CallbackContext):
        """مدیریت دستور /keywords"""
        REQUEST_COUNTER.labels(endpoint='keywords').inc()
        user = update.effective_user
        user_profile = self._get_user_profile(user.id)
        
        if not self._check_rate_limit(user.id, 'keywords'):
            update.message.reply_text("⏳ تعداد درخواست‌های شما امروز به حد مجاز رسیده است.")
            return
        
        if not context.args:
            update.message.reply_text("لطفاً موضوع مورد نظر خود را وارد کنید.\nمثال: /keywords آموزش سئو")
            return
        
        query = " ".join(context.args)
        if len(query) > 200:
            update.message.reply_text("⚠️ طول موضوع نباید از 200 کاراکتر بیشتر باشد.")
            return
        
        try:
            update.message.reply_text("🔍 در حال جستجوی بهترین کلمات کلیدی...")
            
            # استفاده از مدل برای پیشنهاد کلمات کلیدی
            keywords = self._generate_keyword_suggestions(query)
            
            response = (
                f"🔎 *کلمات کلیدی پیشنهادی برای '{query}':*\n\n" +
                "\n".join(f"🔹 {kw}" for kw in keywords) +
                "\n\n💡 می‌توانید از این کلمات در تولید محتوا استفاده کنید."
            )
            
            update.message.reply_text(response, parse_mode="Markdown")
            
            # ذخیره در تاریخچه کاربر
            user_profile.save_content({
                "type": "keyword_research",
                "title": f"پژوهش کلمات کلیدی برای {query}",
                "content": "\n".join(keywords),
                "tags": [query]
            })
            
        except Exception as e:
            logger.error(f"خطا در پیشنهاد کلمات کلیدی: {e}")
            update.message.reply_text("⚠️ خطایی در پردازش درخواست شما رخ داد. لطفاً مجدداً تلاش کنید.")

    # ##################################################
    # ## -------- متدهای کمکی پیشرفته -------- ##
    # ##################################################

    def _generate_keyword_suggestions(self, query: str) -> List[str]:
        """تولید پیشنهادات کلمات کلیدی با استفاده از مدل ML"""
        try:
            # استفاده از مدل برای تولید کلمات کلیدی
            result = self.model_manager.models["keyword"](
                f"Generate 15 SEO keywords for: {query}",
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50
            )
            
            keywords = result[0]["generated_text"].split(",")
            cleaned_keywords = [kw.strip() for kw in keywords if kw.strip()]
            
            # حذف کلمات تکراری و مرتب‌سازی
            unique_keywords = list(dict.fromkeys(cleaned_keywords))
            return unique_keywords[:self.config.KEYWORD_SUGGESTIONS]
        except Exception as e:
            logger.error(f"خطا در تولید کلمات کلیدی: {e}")
            # بازگشت به پیشنهادات پیش‌فرض در صورت خطا
            return [
                f"{query} آموزش",
                f"آموزش حرفه‌ای {query}",
                f"بهترین روش‌های {query}",
                f"{query} 2023",
                f"آموزش رایگان {query}",
                f"نکات کلیدی {query}",
                f"{query} پیشرفته",
                f"متدهای جدید {query}",
                f"آموزش تصویری {query}",
                f"{query} برای مبتدیان"
            ]

    def _generate_welcome_message(self, user: User) -> str:
        """تولید پیام خوشامدگویی شخصی‌سازی شده"""
        plan = self.config.SUBSCRIPTION_PLANS.get(
            self._get_user_profile(user.id).data["subscription"]["plan"], 
            self.config.SUBSCRIPTION_PLANS["free"]
        )
        
        return (
            f"✨ سلام {user.first_name} عزیز!\n"
            f"به ربات هوشمند سئوکار خوش آمدید!\n\n"
            f"🔹 شما در حال استفاده از طرح *{plan.name}* هستید\n"
            f"🔸 امکانات فعلی شما:\n"
            + "\n".join(f"• {feature}" for feature in plan.features) +
            "\n\nبرای شروع کار می‌توانید از منوی زیر اقدام کنید:"
        )

    def _generate_main_keyboard(self) -> List[List[InlineKeyboardButton]]:
        """تولید کیبورد اصلی با دکمه‌های شیشه‌ای"""
        return [
            [InlineKeyboardButton("🔍 پیشنهاد کلمات کلیدی", callback_data='keywords')],
            [InlineKeyboardButton("✍️ تولید محتوای سئو شده", callback_data='content')],
            [InlineKeyboardButton("⚡ بهینه‌سازی متن", callback_data='optimize')],
            [InlineKeyboardButton("📊 تحلیل سئو", callback_data='analyze')],
            [InlineKeyboardButton("🔄 مقایسه دو متن", callback_data='compare')],
            [InlineKeyboardButton("🏆 تحلیل رقبا", callback_data='competitor')],
            [InlineKeyboardButton("📚 راهنمای جامع", callback_data='help')],
            [InlineKeyboardButton("👤 پروفایل کاربری", callback_data='profile')]
        ]

    def _generate_help_text(self, user_profile: UserProfile) -> str:
        """تولید متن راهنمای جامع"""
        plan = self.config.SUBSCRIPTION_PLANS.get(
            user_profile.data["subscription"]["plan"], 
            self.config.SUBSCRIPTION_PLANS["free"]
        )
        
        return f"""
📚 *راهنمای جامع ربات سئوکار*

🔍 *پیشنهاد کلمات کلیدی*
/keywords [موضوع]
- دریافت کلمات کلیدی مرتبط
- مثال: `/keywords آموزش سئو`

✍️ *تولید محتوا*
/content [نوع] [موضوع]
- تولید محتوای بهینه شده
- انواع: مقاله، محصول، صفحه فرود، پست وبلاگ
- مثال: `/content article آموزش سئو`

⚡ *بهینه‌سازی متن*
/optimize [متن]
- بهبود ساختار و سئو متن
- مثال: `/optimize این یک نمونه متن است...`

📊 *تحلیل سئو*
/analyze [متن یا URL]
- تحلیل کامل از نظر سئو
- مثال: `/analyze https://example.com`

🔄 *مقایسه دو متن*
/compare [متن1]\n[متن2]
- مقایسه شباهت و کیفیت دو متن
- مثال: `/compare متن اول...\nمتن دوم...`

🏆 *تحلیل رقبا*
/competitor [URL رقیب] [متن شما]
- مقایسه محتوای شما با رقیب
- مثال: `/competitor https://example.com این محتوای من است...`

💎 *طرح اشتراک شما*: {plan.name}
📌 *محدودیت درخواست روزانه*: {plan.rate_limit}
"""

    def _check_rate_limit(self, user_id: int, endpoint: str) -> bool:
        """بررسی محدودیت درخواست کاربر"""
        user_profile = self._get_user_profile(user_id)
        can_request, message = user_profile.can_make_request(endpoint)
        
        if not can_request:
            REQUEST_COUNTER.labels(endpoint=endpoint).inc()
            return False
            
        return True

    def _get_user_profile(self, user_id: int) -> UserProfile:
        """دریافت پروفایل کاربر با ایجاد خودکار در صورت عدم وجود"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id, self.config, self.security_manager)
            ACTIVE_USERS_GAUGE.inc()
            logger.info(f"پروفایل جدید برای کاربر {user_id} ایجاد شد")
            
        return self.user_profiles[user_id]

    # ##################################################
    # ## -------- کارهای زمان‌بندی شده -------- ##
    # ##################################################

    def _schedule_jobs(self):
        """برنامه‌ریزی کارهای زمان‌بندی شده"""
        # پشتیبان‌گیری روزانه
        self.job_queue.run_daily(
            self._daily_backup_task,
            time=datetime.strptime("03:00", "%H:%M").time(),
            days=(0, 1, 2, 3, 4, 5, 6),
            name="daily_backup"
        )
        
        # ارسال گزارش هفتگی
        self.job_queue.run_daily(
            self._weekly_report_task,
            time=datetime.strptime("10:00", "%H:%M").time(),
            days=(6,),  # شنبه
            name="weekly_report"
        )
        
        # به‌روزرسانی مدل‌های ML هر 24 ساعت
        self.job_queue.run_repeating(
            self._refresh_models_task,
            interval=86400,
            first=0,
            name="refresh_models"
        )
        
        logger.info("کارهای زمان‌بندی شده با موفقیت تنظیم شدند")

    def _daily_backup_task(self, context: CallbackContext):
        """وظیفه پشتیبان‌گیری روزانه"""
        try:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d')}"
            all_data = {
                str(user_id): profile.data 
                for user_id, profile in self.user_profiles.items()
            }
            
            if self.backup_manager.create_backup(all_data, backup_name):
                logger.info(f"پشتیبان‌گیری روزانه با نام {backup_name} با موفقیت انجام شد")
            else:
                logger.warning("خطا در انجام پشتیبان‌گیری روزانه")
        except Exception as e:
            logger.error(f"خطا در وظیفه پشتیبان‌گیری روزانه: {e}")

    def _weekly_report_task(self, context: CallbackContext):
        """وظیفه ارسال گزارش هفتگی"""
        try:
            total_users = len(self.user_profiles)
            active_users = sum(1 for profile in self.user_profiles.values() 
                             if profile.data["usage"]["daily_requests"] > 0)
            
            report = (
                "📊 گزارش هفتگی ربات سئوکار\n\n"
                f"👥 کاربران کل: {total_users}\n"
                f"🔄 کاربران فعال این هفته: {active_users}\n"
                f"📌 مجموع درخواست‌ها: {sum(p.data['usage']['total_requests'] for p in self.user_profiles.values())}\n\n"
                "✅ سیستم به درستی کار می‌کند"
            )
            
            # ارسال گزارش به ادمین‌ها (در اینجا فقط لاگ می‌کنیم)
            logger.info(report)
        except Exception as e:
            logger.error(f"خطا در وظیفه گزارش هفتگی: {e}")

    def _refresh_models_task(self, context: CallbackContext):
        """تازه‌سازی مدل‌های ML"""
        try:
            logger.info("شروع تازه‌سازی مدل‌های یادگیری ماشین...")
            self.model_manager.load_models()
            logger.info("مدل‌های یادگیری ماشین با موفقیت تازه‌سازی شدند")
        except Exception as e:
            logger.error(f"خطا در تازه‌سازی مدل‌ها: {e}")

    # ##################################################
    # ## -------- نقطه ورود اصلی ربات -------- ##
    # ##################################################

    def run(self):
        """راه‌اندازی ربات با مدیریت خطا"""
        try:
            # راه‌اندازی متریک‌ها
            if os.getenv('ENABLE_METRICS', 'false').lower() == 'true':
                start_http_server(8000)
                logger.info("سیستم متریک‌ها روی پورت 8000 راه‌اندازی شد")
            
            # برنامه‌ریزی کارهای زمان‌بندی شده
            self._schedule_jobs()
            
            # شروع ربات
            self.updater.start_polling()
            logger.info("✅ ربات سئوکار با موفقیت شروع به کار کرد")
            self.updater.idle()
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی ربات: {e}")
            raise
        finally:
            # ذخیره داده‌های کاربران قبل از خروج
            for user_id in self.user_profiles:
                self._save_user_data(user_id)
            logger.info("داده‌های کاربران ذخیره شدند")

    def _save_user_data(self, user_id: int):
        """ذخیره داده کاربر با مدیریت خطا"""
        try:
            if user_id not in self.user_profiles:
                return
                
            user_dir = Path(self.config.USER_DATA_DIR)
            user_dir.mkdir(exist_ok=True, mode=0o700)
            
            user_file = user_dir / f"{user_id}.json"
            encrypted_data = self.security_manager.encrypt_data(
                json.dumps(self.user_profiles[user_id].data, ensure_ascii=False)
            
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(encrypted_data, f, ensure_ascii=False)
            
            logger.debug(f"داده کاربر {user_id} با موفقیت ذخیره شد")
        except Exception as e:
            logger.error(f"خطا در ذخیره داده کاربر {user_id}: {e}")

if __name__ == '__main__':
    try:
        # تنظیمات اولیه
        config = Config()
        
        # بررسی تنظیمات ضروری
        if not config.BOT_TOKEN:
            raise ValueError("توکن ربات باید تنظیم شود. متغیر محیطی BOT_TOKEN را تنظیم کنید.")
            
        if not config.ENCRYPTION_KEY:
            raise ValueError("کلید رمزنگاری باید تنظیم شود. متغیر محیطی ENCRYPTION_KEY را تنظیم کنید.")
        
        # راه‌اندازی ربات
        bot = SEOAssistantBot(config)
        bot.run()
    except Exception as e:
        logger.critical(f"خطای بحرانی در راه‌اندازی ربات: {e}")
        raise
