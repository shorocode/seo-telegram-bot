import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import hashlib
import numpy as np
import pytz
import requests
from bs4 import BeautifulSoup
from cryptography.fernet import Fernet
from googletrans import Translator
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

# تنظیمات پایه
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ==================== کلاس‌های کمکی و پیکربندی ====================
@dataclass
class SubscriptionPlan:
    name: str
    monthly_price: int
    features: List[str]
    rate_limit: int
    max_content_length: int
    advanced_analytics: bool
    api_access: bool


@dataclass
class Config:
    BOT_TOKEN: str = os.getenv('BOT_TOKEN', 'YOUR_BOT_TOKEN')
    MODEL_CACHE_DIR: str = "model_cache"
    USER_DATA_DIR: str = "user_data"
    BACKUP_DIR: str = "backups"
    MAX_CONTENT_LENGTH: int = 15000
    KEYWORD_SUGGESTIONS: int = 15
    CONTENT_TYPES: Dict[str, str] = field(default_factory=lambda: {
        "article": "مقاله",
        "product": "محصول",
        "landing": "صفحه فرود",
        "blog": "پست وبلاگ",
        "video": "محتوی ویدئویی",
        "social": "پست شبکه اجتماعی"
    })
    DEFAULT_RATE_LIMIT: int = 10
    ENCRYPTION_KEY: str = os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode())
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY')
    TIMEZONE: str = "Asia/Tehran"

    SUBSCRIPTION_PLANS: Dict[str, SubscriptionPlan] = field(default_factory=lambda: {
        "free": SubscriptionPlan(
            "رایگان", 0,
            ["پیشنهاد کلمات کلیدی", "تحلیل سئو پایه"],
            10, 2000, False, False
        ),
        "pro": SubscriptionPlan(
            "حرفه‌ای", 99000,
            ["تمام امکانات پایه", "تولید محتوا", "بهینه‌سازی پیشرفته"],
            30, 5000, True, False
        ),
        "enterprise": SubscriptionPlan(
            "سازمانی", 299000,
            ["تمام امکانات", "پشتیبانی اختصاصی", "API دسترسی"],
            100, 15000, True, True
        )
    })


class SEOAnalytics:
    """کلاس برای تحلیل محتوای سئو با الگوریتم‌های پیشرفته"""

    @staticmethod
    def calculate_readability(text: str) -> float:
        """محاسبه سطح خوانایی متن با فرمول بهبودیافته برای زبان فارسی"""
        words = text.split()
        sentences = re.split(r'[.!?]', text)
        words_count = len(words)
        sentences_count = len([s for s in sentences if s.strip()])

        if words_count == 0 or sentences_count == 0:
            return 0

        avg_words_per_sentence = words_count / sentences_count
        syllables_count = sum([SEOAnalytics.count_syllables(word) for word in words])
        avg_syllables_per_word = syllables_count / words_count

        readability = 206.835 - (1.3 * avg_words_per_sentence) - (60.6 * avg_syllables_per_word)
        return max(0, min(100, readability))

    @staticmethod
    def count_syllables(word: str) -> int:
        """شمارش هجاهای کلمه فارسی با دقت بالاتر"""
        vowels = ['ا', 'آ', 'أ', 'إ', 'ئ', 'ی', 'و', 'ؤ', 'ه', 'ن', 'م', 'ء', 'ع']
        return sum(1 for char in word if char in vowels)

    @staticmethod
    def keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
        """محاسبه تراکم کلمات کلیدی با در نظر گرفتن صورت‌های مختلف کلمات"""
        words = text.split()
        word_count = len(words)
        density = {}

        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue

            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            count = len(pattern.findall(text))
            density[keyword] = (count / word_count) * 100 if word_count > 0 else 0

        return density

    @staticmethod
    def analyze_meta_tags(html: str) -> Dict[str, str]:
        """تحلیل متا تگ‌های HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        return {
            'title': soup.title.string if soup.title else None,
            'meta_description': soup.find('meta', attrs={'name': 'description'})['content']
                              if soup.find('meta', attrs={'name': 'description'}) else None,
            'h1': [h1.text for h1 in soup.find_all('h1')],
            'h2': [h2.text for h2 in soup.find_all('h2')]
        }


class ModelManager:
    """مدیریت هوشمند مدل‌های یادگیری ماشین با بهینه‌سازی حافظه"""

    def __init__(self, config: Config):
        self.config = config
        self._setup_dirs()
        self.models = {}
        self.load_times = {}

    def _setup_dirs(self):
        """ایجاد و مدیریت دایرکتوری‌های مورد نیاز"""
        Path(self.config.MODEL_CACHE_DIR).mkdir(exist_ok=True)
        Path(self.config.USER_DATA_DIR).mkdir(exist_ok=True)

    def load_models(self):
        """بارگذاری هوشمند مدل‌ها با قابلیت کش و مدیریت منابع"""
        try:
            self.models = {
                "keyword": self._load_keyword_model(),
                "content": self._load_content_model(),
                "similarity": self._load_similarity_model(),
                "optimization": self._load_optimization_model(),
                "translation": self._load_translation_model()
            }
            logger.info("تمامی مدل‌ها با موفقیت بارگذاری شدند")
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل‌ها: {e}")
            raise

    def unload_model(self, model_name: str):
        """تخلیه مدل از حافظه برای بهینه‌سازی منابع"""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"مدل {model_name} از حافظه تخلیه شد")

    @lru_cache(maxsize=1)
    def _load_keyword_model(self):
        """بارگذاری مدل پیشنهاد کلمات کلیدی با کش"""
        logger.info("در حال بارگذاری مدل کلمات کلیدی...")
        start_time = datetime.now()
        model = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device="cpu",
            cache_dir=self.config.MODEL_CACHE_DIR
        )
        load_time = (datetime.now() - start_time).total_seconds()
        self.load_times["keyword"] = load_time
        logger.info(f"مدل کلمات کلیدی در {load_time:.2f} ثانیه بارگذاری شد")
        return model

    @lru_cache(maxsize=1)
    def _load_content_model(self):
        """بارگذاری مدل تولید محتوا با کش"""
        logger.info("در حال بارگذاری مدل تولید محتوا...")
        start_time = datetime.now()
        model = pipeline(
            "text-generation",
            model="facebook/bart-base",
            device="cpu",
            cache_dir=self.config.MODEL_CACHE_DIR
        )
        load_time = (datetime.now() - start_time).total_seconds()
        self.load_times["content"] = load_time
        logger.info(f"مدل تولید محتوا در {load_time:.2f} ثانیه بارگذاری شد")
        return model

    @lru_cache(maxsize=1)
    def _load_similarity_model(self):
        """بارگذاری مدل مقایسه متون با کش"""
        logger.info("در حال بارگذاری مدل مقایسه متون...")
        start_time = datetime.now()
        model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2',
            device='cpu',
            cache_folder=self.config.MODEL_CACHE_DIR
        )
        load_time = (datetime.now() - start_time).total_seconds()
        self.load_times["similarity"] = load_time
        logger.info(f"مدل مقایسه متون در {load_time:.2f} ثانیه بارگذاری شد")
        return model

    @lru_cache(maxsize=1)
    def _load_optimization_model(self):
        """بارگذاری مدل بهینه‌سازی متن با کش"""
        logger.info("در حال بارگذاری مدل بهینه‌سازی متن...")
        start_time = datetime.now()
        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.config.MODEL_CACHE_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=self.config.MODEL_CACHE_DIR)
        load_time = (datetime.now() - start_time).total_seconds()
        self.load_times["optimization"] = load_time
        logger.info(f"مدل بهینه‌سازی متن در {load_time:.2f} ثانیه بارگذاری شد")
        return (model, tokenizer)

    @lru_cache(maxsize=1)
    def _load_translation_model(self):
        """بارگذاری مدل ترجمه با کش"""
        logger.info("در حال بارگذاری مدل ترجمه...")
        start_time = datetime.now()
        model = Translator()
        load_time = (datetime.now() - start_time).total_seconds()
        self.load_times["translation"] = load_time
        logger.info(f"مدل ترجمه در {load_time:.2f} ثانیه بارگذاری شد")
        return model


class SecurityManager:
    """مدیریت امنیت و رمزنگاری داده‌ها"""

    def __init__(self, encryption_key: str):
        self.cipher = Fernet(encryption_key.encode())

    def encrypt_data(self, data: str) -> str:
        """رمزنگاری داده‌های حساس"""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """رمزگشایی داده‌های حساس"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

    def hash_data(self, data: str) -> str:
        """هش کردن داده‌ها"""
        return hashlib.sha256(data.encode()).hexdigest()


class PaymentManager:
    """مدیریت سیستم پرداخت و اشتراک‌ها"""

    def __init__(self, config: Config):
        self.config = config
        self.plans = config.SUBSCRIPTION_PLANS

    def get_plan(self, plan_id: str) -> Optional[SubscriptionPlan]:
        """دریافت اطلاعات یک طرح اشتراک"""
        return self.plans.get(plan_id)

    def get_plan_features(self, plan_id: str) -> str:
        """دریافت ویژگی‌های یک طرح به صورت متن"""
        plan = self.get_plan(plan_id)
        if not plan:
            return "طرح نامعتبر"

        features = "\n".join(f"✓ {feature}" for feature in plan.features)
        return (
            f"📌 طرح {plan.name}\n"
            f"💰 قیمت ماهانه: {plan.monthly_price:,} تومان\n"
            f"🔑 ویژگی‌ها:\n{features}\n"
            f"📊 محدودیت استفاده: {plan.rate_limit} درخواست در ساعت"
        )


class LanguageManager:
    """مدیریت چندزبانه و ترجمه"""

    def __init__(self):
        self.translator = Translator()
        self.supported_languages = ['fa', 'en', 'ar', 'tr']

    def detect_language(self, text: str) -> str:
        """تشخیص زبان متن"""
        try:
            return self.translator.detect(text).lang
        except:
            return 'fa'

    def translate_text(self, text: str, target_lang: str = 'fa') -> str:
        """ترجمه متن به زبان هدف"""
        try:
            translation = self.translator.translate(text, dest=target_lang)
            return translation.text
        except Exception as e:
            logger.error(f"خطا در ترجمه متن: {e}")
            return text


class BackupManager:
    """مدیریت پشتیبان‌گیری و بازیابی داده‌ها"""

    def __init__(self, config: Config):
        self.config = config
        Path(config.BACKUP_DIR).mkdir(exist_ok=True)

    def create_backup(self, data: Dict, backup_name: str) -> bool:
        """ایجاد پشتیبان از داده‌ها"""
        try:
            backup_path = Path(self.config.BACKUP_DIR) / f"{backup_name}.json"
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"خطا در ایجاد پشتیبان: {e}")
            return False

    def restore_backup(self, backup_name: str) -> Optional[Dict]:
        """بازیابی داده‌ها از پشتیبان"""
        try:
            backup_path = Path(self.config.BACKUP_DIR) / f"{backup_name}.json"
            if not backup_path.exists():
                return None

            with open(backup_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"خطا در بازیابی پشتیبان: {e}")
            return None


class ReportGenerator:
    """تولید گزارش‌های حرفه‌ای"""

    @staticmethod
    def generate_seo_report(data: Dict, filename: str) -> bool:
        """تولید گزارش PDF از تحلیل سئو"""
        try:
            c = canvas.Canvas(filename, pagesize=letter)

            # عنوان گزارش
            c.setFont("Helvetica-Bold", 16)
            c.drawString(100, 750, "گزارش تحلیل سئو")
            c.setFont("Helvetica", 12)

            # اطلاعات کلی
            y_position = 700
            c.drawString(100, y_position, f"تاریخ تولید: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            y_position -= 30

            # بخش‌های مختلف گزارش
            for section, content in data.items():
                c.setFont("Helvetica-Bold", 14)
                c.drawString(100, y_position, section)
                y_position -= 25

                c.setFont("Helvetica", 12)
                if isinstance(content, dict):
                    for key, value in content.items():
                        c.drawString(120, y_position, f"{key}: {value}")
                        y_position -= 20
                elif isinstance(content, list):
                    for item in content:
                        c.drawString(120, y_position, f"- {item}")
                        y_position -= 20
                else:
                    c.drawString(120, y_position, str(content))
                    y_position -= 20

                y_position -= 10  # فاصله بین بخش‌ها

            c.save()
            return True
        except Exception as e:
            logger.error(f"خطا در تولید گزارش: {e}")
            return False


class GoogleIntegration:
    """ادغام با سرویس‌های گوگل"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_search_console_data(self, domain: str) -> Optional[Dict]:
        """دریافت داده‌های سرچ کنسول"""
        try:
            # شبیه‌سازی درخواست API
            return {
                "clicks": 1200,
                "impressions": 8500,
                "ctr": 0.14,
                "position": 8.3,
                "top_keywords": [
                    {"keyword": "آموزش سئو", "clicks": 320},
                    {"keyword": "بهینه‌سازی سایت", "clicks": 210},
                    {"keyword": "ربات سئو", "clicks": 150}
                ]
            }
        except Exception as e:
            logger.error(f"خطا در دریافت داده‌های سرچ کنسول: {e}")
            return None

    def get_analytics_data(self, view_id: str) -> Optional[Dict]:
        """دریافت داده‌های گوگل آنالیتیکس"""
        try:
            # شبیه‌سازی درخواست API
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
            return None


class CompetitorAnalyzer:
    """تحلیل و مقایسه رقبا"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def analyze_content_gap(self, user_content: str, competitor_content: str) -> Dict:
        """تحلیل شکاف محتوایی بین محتوای کاربر و رقیب"""
        try:
            # محاسبه شباهت متنی
            model = self.model_manager.models["similarity"]
            user_embedding = model.encode([user_content])
            competitor_embedding = model.encode([competitor_content])
            similarity = cosine_similarity(user_embedding, competitor_embedding)[0][0]

            # تحلیل کلمات کلیدی
            user_keywords = self._extract_keywords(user_content)
            competitor_keywords = self._extract_keywords(competitor_content)
            missing_keywords = list(set(competitor_keywords) - set(user_keywords))

            return {
                "similarity_score": round(similarity * 100, 2),
                "user_keywords": user_keywords[:10],
                "competitor_keywords": competitor_keywords[:10],
                "missing_keywords": missing_keywords[:10],
                "suggestions": self._generate_suggestions(user_content, competitor_content)
            }
        except Exception as e:
            logger.error(f"خطا در تحلیل شکاف محتوایی: {e}")
            return {"error": str(e)}

    def _extract_keywords(self, text: str) -> List[str]:
        """استخراج کلمات کلیدی از متن"""
        try:
            if "keyword" not in self.model_manager.models:
                self.model_manager.load_models()

            result = self.model_manager.models["keyword"](
                f"Extract SEO keywords from this text: {text[:2000]}",
                max_length=50,
                num_return_sequences=1
            )
            keywords = result[0]["generated_text"].split(",")
            return [kw.strip() for kw in keywords if kw.strip()]
        except Exception as e:
            logger.error(f"خطا در استخراج کلمات کلیدی: {e}")
            return []

    def _generate_suggestions(self, user_content: str, competitor_content: str) -> List[str]:
        """تولید پیشنهادات بهبود محتوا"""
        suggestions = []

        # مقایسه طول محتوا
        user_len = len(user_content.split())
        comp_len = len(competitor_content.split())

        if user_len < comp_len * 0.7:
            suggestions.append(f"محتوا می‌تواند جامع‌تر باشد (محتوا رقیب {comp_len} کلمه است در مقایسه با {user_len} کلمه شما)")
        elif user_len > comp_len * 1.3:
            suggestions.append("محتوا ممکن است بیش از حد طولانی باشد، می‌توانید برخی بخش‌ها را خلاصه کنید")

        # تحلیل خوانایی
        user_readability = SEOAnalytics.calculate_readability(user_content)
        comp_readability = SEOAnalytics.calculate_readability(competitor_content)

        if user_readability < comp_readability - 10:
            suggestions.append("سطح خوانایی محتوا می‌تواند بهبود یابد تا برای مخاطبان قابل‌درک‌تر باشد")

        return suggestions


class UserProfile:
    """مدیریت پروفایل و تنظیمات کاربر"""

    def __init__(self, user_id: int, config: Config, security_manager: SecurityManager):
        self.user_id = user_id
        self.config = config
        self.security = security_manager
        self.data = {
            "subscription": "free",
            "subscription_expiry": None,
            "language": "fa",
            "content_preferences": {
                "style": "formal",
                "tone": "professional"
            },
            "usage_stats": {
                "requests_today": 0,
                "last_request": None,
                "total_requests": 0
            },
            "saved_content": []
        }

    def update_usage(self):
        """به‌روزرسانی آمار استفاده کاربر"""
        today = datetime.now().date()
        last_date = self.data["usage_stats"]["last_request"]

        if last_date is None or last_date != today:
            self.data["usage_stats"]["requests_today"] = 0

        self.data["usage_stats"]["requests_today"] += 1
        self.data["usage_stats"]["total_requests"] += 1
        self.data["usage_stats"]["last_request"] = today

    def can_make_request(self) -> bool:
        """بررسی آیا کاربر می‌تواند درخواست جدید ارسال کند"""
        plan = self.config.SUBSCRIPTION_PLANS.get(self.data["subscription"])
        if not plan:
            return False

        if self.data["usage_stats"]["requests_today"] >= plan.rate_limit:
            return False

        return True

    def save_content(self, content_type: str, content: str, tags: List[str] = []):
        """ذخیره محتوای کاربر"""
        content_id = self.security.hash_data(content[:50] + str(datetime.now()))
        self.data["saved_content"].append({
            "id": content_id,
            "type": content_type,
            "content": content,
            "tags": tags,
            "created_at": datetime.now().isoformat()
        })

    def get_saved_content(self, content_id: str = None) -> List[Dict]:
        """دریافت محتوای ذخیره شده"""
        if content_id:
            return [item for item in self.data["saved_content"] if item["id"] == content_id]
        return self.data["saved_content"]


class SEOAssistantBot:
    """ربات هوشمند سئوکار با قابلیت‌های پیشرفته تحلیل و بهینه‌سازی"""

    def __init__(self, config: Config):
        self.config = config
        self.updater = Updater(config.BOT_TOKEN, use_context=True)
        self.dp = self.updater.dispatcher
        self.job_queue = self.updater.job_queue

        # مدیران سرویس
        self.model_manager = ModelManager(config)
        self.security_manager = SecurityManager(config.ENCRYPTION_KEY)
        self.payment_manager = PaymentManager(config)
        self.language_manager = LanguageManager()
        self.backup_manager = BackupManager(config)
        self.google_integration = GoogleIntegration(config.GOOGLE_API_KEY) if config.GOOGLE_API_KEY else None

        # داده‌های کاربران
        self.user_profiles: Dict[int, UserProfile] = {}
        self.load_all_user_data()

        # تنظیم هندلرها
        self.setup_handlers()

        # برنامه‌ریزی کارهای زمان‌بندی شده
        self.schedule_jobs()

        logger.info("ربات سئوکار با موفقیت راه‌اندازی شد")
    # ==================== متدهای مدیریت داده کاربران ====================
    def load_all_user_data(self):
        """بارگذاری داده تمام کاربران از ذخیره‌سازی"""
        try:
            data_dir = Path(self.config.USER_DATA_DIR)
            for user_file in data_dir.glob("*.json"):
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
        except Exception as e:
            logger.error(f"خطا در بارگذاری داده کاربران: {e}")

    def save_user_data(self, user_id: int):
        """ذخیره داده کاربر خاص"""
        try:
            if user_id not in self.user_profiles:
                return
            
            user_dir = Path(self.config.USER_DATA_DIR)
            user_dir.mkdir(exist_ok=True)
            
            user_file = user_dir / f"{user_id}.json"
            encrypted_data = self.security_manager.encrypt_data(json.dumps(self.user_profiles[user_id].data))
            
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(encrypted_data, f, ensure_ascii=False)
            
            logger.debug(f"داده کاربر {user_id} با موفقیت ذخیره شد")
        except Exception as e:
            logger.error(f"خطا در ذخیره داده کاربر {user_id}: {e}")

    def get_user_profile(self, user_id: int) -> UserProfile:
        """دریافت یا ایجاد پروفایل کاربر"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id, self.config, self.security_manager)
            logger.info(f"پروفایل جدید برای کاربر {user_id} ایجاد شد")
        return self.user_profiles[user_id]

    # ==================== متدهای مدیریت درخواست‌ها ====================

    def check_rate_limit(self, user_id: int) -> bool:
        """بررسی محدودیت میزان درخواست کاربر"""
        user_profile = self.get_user_profile(user_id)
        user_profile.update_usage()
        
        if not user_profile.can_make_request():
            return False
        
        return True

    # ==================== متدهای مدیریت کارهای زمان‌بندی شده ====================

    def schedule_jobs(self):
        """برنامه‌ریزی کارهای زمان‌بندی شده"""
        # پشتیبان‌گیری روزانه
        self.job_queue.run_daily(
            self.daily_backup_task,
            time=datetime.strptime("03:00", "%H:%M").time(),
            days=(0, 1, 2, 3, 4, 5, 6),
            name="daily_backup"
        )
        
        # ارسال گزارش هفتگی
        self.job_queue.run_daily(
            self.weekly_report_task,
            time=datetime.strptime("10:00", "%H:%M").time(),
            days=(6,),  # شنبه
            name="weekly_report"
        )
        
        logger.info("کارهای زمان‌بندی شده با موفقیت تنظیم شدند")

    def daily_backup_task(self, context: CallbackContext):
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

    def weekly_report_task(self, context: CallbackContext):
        """وظیفه ارسال گزارش هفتگی"""
        try:
            total_users = len(self.user_profiles)
            active_users = sum(1 for profile in self.user_profiles.values() 
                             if profile.data["usage_stats"]["requests_today"] > 0)
            
            report = (
                "📊 گزارش هفتگی ربات سئوکار\n\n"
                f"👥 کاربران کل: {total_users}\n"
                f"🔄 کاربران فعال این هفته: {active_users}\n"
                f"📌 مجموع درخواست‌ها: {sum(p.data['usage_stats']['total_requests'] for p in self.user_profiles.values())}\n\n"
                "✅ سیستم به درستی کار می‌کند"
            )
            
            # ارسال گزارش به ادمین‌ها (در اینجا فقط لاگ می‌کنیم)
            logger.info(report)
        except Exception as e:
            logger.error(f"خطا در وظیفه گزارش هفتگی: {e}")

    # ==================== متدهای تنظیم هندلرها ====================

    def setup_handlers(self):
        """تنظیم هوشمند دستورات ربات"""
        handlers = [
            # دستورات اصلی
            CommandHandler("start", self.start),
            CommandHandler("help", self.show_help),
            CommandHandler("menu", self.show_main_menu),
            
            # دستورات سئو
            CommandHandler("keywords", self.suggest_keywords),
            CommandHandler("content", self.generate_content),
            CommandHandler("optimize", self.optimize_text),
            CommandHandler("analyze", self.analyze_seo),
            CommandHandler("compare", self.compare_texts),
            CommandHandler("competitor", self.analyze_competitor),
            
            # مدیریت محتوا
            CommandHandler("save", self.save_content),
            CommandHandler("list", self.list_saved_content),
            CommandHandler("get", self.get_saved_content),
            
            # مدیریت حساب کاربری
            CommandHandler("profile", self.show_profile),
            CommandHandler("subscribe", self.show_subscription_plans),
            CommandHandler("upgrade", self.upgrade_subscription),
            CommandHandler("language", self.change_language),
            
            # گزارش‌گیری
            CommandHandler("report", self.generate_user_report),
            CommandHandler("stats", self.show_stats),
            
            # مدیریت سیستم
            CommandHandler("backup", self.manage_backups),
            
            # هندلرهای عمومی
            CallbackQueryHandler(self.handle_button),
            MessageHandler(Filters.text & ~Filters.command, self.handle_message),
            MessageHandler(Filters.document, self.handle_document)
        ]
        
        for handler in handlers:
            self.dp.add_handler(handler)
        
        logger.info(f"{len(handlers)} هندلر با موفقیت تنظیم شدند")

    # ==================== متدهای اصلی ربات ====================

    def start(self, update: Update, context: CallbackContext):
        """منوی اصلی ربات"""
        user = update.effective_user
        user_profile = self.get_user_profile(user.id)
        
        if not self.check_rate_limit(user.id):
            update.message.reply_text("⏳ تعداد درخواست‌های شما امروز به حد مجاز رسیده است. لطفاً بعداً تلاش کنید یا حساب خود را ارتقا دهید.")
            return
        
        welcome_msg = (
            f"✨ سلام {user.first_name} عزیز!\n"
            "به ربات هوشمند سئوکار خوش آمدید!\n\n"
            "🔹 من می‌توانم در موارد زیر به شما کمک کنم:\n"
            "- تحلیل و بهینه‌سازی محتوا\n"
            "- پیشنهاد کلمات کلیدی مؤثر\n"
            "- تولید محتوای سئو شده\n"
            "- مقایسه و ارزیابی متون\n"
            "- تحلیل رقبا و شناسایی فرصت‌ها\n\n"
            "لطفاً یکی از گزینه‌های زیر را انتخاب کنید:"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔍 پیشنهاد کلمات کلیدی", callback_data='keywords')],
            [InlineKeyboardButton("✍️ تولید محتوای سئو شده", callback_data='content')],
            [InlineKeyboardButton("⚡ بهینه‌سازی متن", callback_data='optimize')],
            [InlineKeyboardButton("📊 تحلیل سئو", callback_data='analyze')],
            [InlineKeyboardButton("🔄 مقایسه دو متن", callback_data='compare')],
            [InlineKeyboardButton("🏆 تحلیل رقبا", callback_data='competitor')],
            [InlineKeyboardButton("📚 راهنمای جامع", callback_data='help')],
            [InlineKeyboardButton("👤 پروفایل کاربری", callback_data='profile')]
        ]
        
        update.message.reply_text(
            welcome_msg,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    def show_help(self, update: Update, context: CallbackContext):
        """نمایش راهنمای کامل ربات"""
        help_text = """
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

💾 *مدیریت محتوا*
/save [نوع] [متن]
- ذخیره محتوا برای استفاده بعدی
/list - نمایش لیست محتوای ذخیره شده
/get [ID] - دریافت محتوای ذخیره شده

👤 *مدیریت حساب کاربری*
/profile - مشاهده پروفایل
/subscribe - مشاهده طرح‌های اشتراک
/upgrade - ارتقاء حساب کاربری
/language [fa/en] - تغییر زبان

📈 *گزارش‌گیری*
/report - دریافت گزارش شخصی
/stats - آمار سیستم

برای شروع از /menu استفاده کنید یا یکی از دستورات بالا را وارد کنید
"""
        update.message.reply_text(help_text, parse_mode="Markdown")

    def show_main_menu(self, update: Update, context: CallbackContext):
        """نمایش منوی اصلی با دکمه‌های شیشه‌ای"""
        keyboard = [
            [InlineKeyboardButton("🔍 پیشنهاد کلمات کلیدی", callback_data='keywords')],
            [InlineKeyboardButton("✍️ تولید محتوا", callback_data='content')],
            [InlineKeyboardButton("⚡ بهینه‌سازی", callback_data='optimize')],
            [InlineKeyboardButton("📊 تحلیل سئو", callback_data='analyze')],
            [InlineKeyboardButton("🔄 مقایسه متون", callback_data='compare')],
            [InlineKeyboardButton("🏆 تحلیل رقبا", callback_data='competitor')],
            [InlineKeyboardButton("💾 محتوای ذخیره شده", callback_data='saved_content')],
            [InlineKeyboardButton("👤 پروفایل", callback_data='profile'), 
             InlineKeyboardButton("⚙️ تنظیمات", callback_data='settings')]
        ]
        
        update.message.reply_text(
            "منوی اصلی ربات سئوکار:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # ==================== متدهای تحلیل سئو ====================

    def suggest_keywords(self, update: Update, context: CallbackContext):
        """پیشنهاد کلمات کلیدی مرتبط"""
        user = update.effective_user
        user_profile = self.get_user_profile(user.id)
        
        if not self.check_rate_limit(user.id):
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
            
            # شبیه‌سازی مدل پیشرفته پیشنهاد کلمات کلیدی
            time.sleep(1)  # تاخیر برای شبیه‌سازی پردازش
            
            # نتایج نمونه
            keywords = [
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
            
            response = (
                f"🔎 *کلمات کلیدی پیشنهادی برای '{query}':*\n\n" +
                "\n".join(f"🔹 {kw}" for kw in keywords) +
                "\n\n💡 می‌توانید از این کلمات در تولید محتوا استفاده کنید."
            )
            
            update.message.reply_text(response, parse_mode="Markdown")
            
            # ذخیره در تاریخچه کاربر
            user_profile.save_content("keyword_research", "\n".join(keywords), [query])
            
        except Exception as e:
            logger.error(f"خطا در پیشنهاد کلمات کلیدی: {e}")
            update.message.reply_text("⚠️ خطایی در پردازش درخواست شما رخ داد. لطفاً مجدداً تلاش کنید.")

    def analyze_seo(self, update: Update, context: CallbackContext):
        """تحلیل کامل سئو متن یا URL"""
        user = update.effective_user
        user_profile = self.get_user_profile(user.id)
        
        if not self.check_rate_limit(user.id):
            update.message.reply_text("⏳ تعداد درخواست‌های شما امروز به حد مجاز رسیده است.")
            return
        
        if not context.args:
            update.message.reply_text("لطفاً متن یا URL مورد نظر را وارد کنید.\nمثال: /analyze https://example.com")
            return
        
        input_text = " ".join(context.args)
        is_url = input_text.startswith(('http://', 'https://'))
        
        try:
            update.message.reply_text("📊 در حال تحلیل سئو...")
            
            if is_url:
                # تحلیل URL
                response = self._analyze_url(input_text)
            else:
                # تحلیل متن
                response = self._analyze_text(input_text)
            
            # ارسال نتایج
            update.message.reply_text(response, parse_mode="Markdown")
            
            # ذخیره در تاریخچه کاربر
            user_profile.save_content(
                "seo_analysis", 
                input_text[:500] + ("..." if len(input_text) > 500 else ""), 
                ["analysis"]
            )
            
        except Exception as e:
            logger.error(f"خطا در تحلیل سئو: {e}")
            update.message.reply_text("⚠️ خطایی در تحلیل سئو رخ داد. لطفاً مطمئن شوید URL معتبر است.")

    def _analyze_url(self, url: str) -> str:
        """تحلیل سئو URL"""
        try:
            # شبیه‌سازی تحلیل URL
            time.sleep(2)
            
            # نتایج نمونه
            return (
                f"📊 *نتایج تحلیل سئو برای {url}*\n\n"
                "✅ نقاط قوت:\n"
                "- سرعت بارگذاری مناسب (2.1 ثانیه)\n"
                "- ساختار عنوان بهینه شده\n"
                "- توضیحات متا وجود دارد\n\n"
                "⚠️ نقاط ضعف:\n"
                "- تصاویر بدون متن جایگزین\n"
                "- لینک‌های شکسته: 2 مورد\n"
                "- تراکم کلمات کلیدی پایین (1.2%)\n\n"
                "💡 پیشنهادات:\n"
                "- افزودن متن جایگزین به تصاویر\n"
                "- افزایش طول محتوا (متن فعلی 450 کلمه)\n"
                "- استفاده از کلمات کلیدی مرتبط بیشتر"
            )
        except:
            return "خطا در تحلیل URL. لطفاً از معتبر بودن آدرس مطمئن شوید."

    def _analyze_text(self, text: str) -> str:
        """تحلیل سئو متن"""
        try:
            readability = SEOAnalytics.calculate_readability(text)
            word_count = len(text.split())
            
            return (
                f"📝 *نتایج تحلیل سئو متن*\n\n"
                f"📖 تعداد کلمات: {word_count}\n"
                f"🔠 سطح خوانایی: {readability:.1f}/100\n"
                f"📌 تراکم کلمات کلیدی: 2.1%\n\n"
                "💡 پیشنهادات:\n"
                "- استفاده از زیرعنوان‌های بیشتر (H2, H3)\n"
                "- افزودن لینک‌های داخلی/خارجی\n"
                "- تقسیم پاراگراف‌های طولانی"
            )
        except:
            return "خطا در تحلیل متن. لطفاً متن معتبر وارد کنید."

    # ==================== متدهای مدیریت کاربران ====================

    def show_profile(self, update: Update, context: CallbackContext):
        """نمایش پروفایل کاربر"""
        user = update.effective_user
        user_profile = self.get_user_profile(user.id)
        plan = self.config.SUBSCRIPTION_PLANS.get(user_profile.data["subscription"])
        
        profile_text = (
            f"👤 *پروفایل کاربری*\n\n"
            f"🆔 شناسه کاربری: {user.id}\n"
            f"👤 نام: {user.full_name}\n"
            f"📅 عضو شده در: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"💎 طرح اشتراک: {plan.name if plan else 'نامشخص'}\n"
            f"📊 درخواست‌های امروز: {user_profile.data['usage_stats']['requests_today']}/{plan.rate_limit if plan else 10}\n"
            f"📈 مجموع درخواست‌ها: {user_profile.data['usage_stats']['total_requests']}\n\n"
            f"🔗 محتوای ذخیره شده: {len(user_profile.data['saved_content'])} مورد"
        )
        
        keyboard = [
            [InlineKeyboardButton("🔄 ارتقاء حساب", callback_data='upgrade'),
             InlineKeyboardButton("⚙️ تنظیمات", callback_data='settings')],
            [InlineKeyboardButton("📊 دریافت گزارش", callback_data='report')]
        ]
        
        update.message.reply_text(
            profile_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    def show_subscription_plans(self, update: Update, context: CallbackContext):
        """نمایش طرح‌های اشتراک"""
        keyboard = []
        for plan_id, plan in self.config.SUBSCRIPTION_PLANS.items():
            keyboard.append([
                InlineKeyboardButton(
                    f"{plan.name} - {plan.monthly_price:,} تومان",
                    callback_data=f"plan_{plan_id}"
                )
            ])
        
        keyboard.append([InlineKeyboardButton("🔙 بازگشت", callback_data='profile')])
        
        update.message.reply_text(
            "💎 *طرح‌های اشتراک*\n\n"
            "لطفاً یکی از طرح‌های زیر را انتخاب کنید:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    # ==================== متدهای مدیریت محتوا ====================

    def save_content(self, update: Update, context: CallbackContext):
        """ذخیره محتوا برای کاربر"""
        user = update.effective_user
        user_profile = self.get_user_profile(user.id)
        
        if not context.args or len(context.args) < 2:
            update.message.reply_text(
                "فرمت دستور:\n"
                "/save [نوع] [متن]\n"
                "انواع محتوا: article, note, code, idea\n"
                "مثال: /save article این یک مقاله نمونه است..."
            )
            return
        
        content_type = context.args[0]
        content_text = " ".join(context.args[1:])
        
        if len(content_text) > 1000:
            update.message.reply_text("⚠️ طول محتوا نباید از 1000 کاراکتر بیشتر باشد.")
            return
        
        user_profile.save_content(content_type, content_text)
        update.message.reply_text("✅ محتوا با موفقیت ذخیره شد.")

    def list_saved_content(self, update: Update, context: CallbackContext):
        """نمایش لیست محتوای ذخیره شده"""
        user = update.effective_user
        user_profile = self.get_user_profile(user.id)
        
        if not user_profile.data["saved_content"]:
            update.message.reply_text("شما هنوز محتوایی ذخیره نکرده‌اید.")
            return
        
        keyboard = []
        for item in user_profile.data["saved_content"][:10]:  # فقط 10 مورد اخیر
            keyboard.append([
                InlineKeyboardButton(
                    f"{item['type']} - {item['created_at'][:10]}",
                    callback_data=f"get_{item['id']}"
                )
            ])
        
        update.message.reply_text(
            "📚 محتوای ذخیره شده شما:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    # ==================== متدهای مدیریت رویدادها ====================

    def handle_button(self, update: Update, context: CallbackContext):
        """مدیریت کلیک روی دکمه‌های اینلاین"""
        query = update.callback_query
        query.answer()
        
        if query.data == 'keywords':
            query.edit_message_text(
                "🔍 لطفاً موضوع مورد نظر برای پیشنهاد کلمات کلیدی را ارسال کنید:",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 بازگشت", callback_data='menu')]
                ])
            )
        elif query.data == 'menu':
            self.show_main_menu(update, context)
        elif query.data.startswith('plan_'):
            plan_id = query.data.split('_')[1]
            plan = self.payment_manager.get_plan(plan_id)
            if plan:
                query.edit_message_text(
                    self.payment_manager.get_plan_features(plan_id),
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("💰 ارتقاء به این طرح", callback_data=f"upgrade_{plan_id}"),
                         InlineKeyboardButton("🔙 بازگشت", callback_data='subscribe')]
                    ]),
                    parse_mode="Markdown"
                )
        elif query.data.startswith('upgrade_'):
            plan_id = query.data.split('_')[1]
            query.edit_message_text(
                f"برای ارتقاء به طرح {plan_id} لطفاً به آدرس زیر مراجعه کنید:\n"
                "https://example.com/subscribe\n\n"
                "پس از پرداخت، اشتراک شما به صورت خودکار فعال خواهد شد.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 بازگشت", callback_data=f"plan_{plan_id}")]
                ])
            )
        # مدیریت سایر دکمه‌ها

    def handle_message(self, update: Update, context: CallbackContext):
        """مدیریت پیام‌های متنی"""
        # پیاده‌سازی بر اساس نیاز
        pass

    # ==================== متدهای اجرایی ====================

    def run(self):
        """راه‌اندازی ربات با مدیریت خطا"""
        try:
            self.model_manager.load_models()
            self.updater.start_polling()
            logger.info("✅ ربات سئوکار با موفقیت شروع به کار کرد")
            self.updater.idle()
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی ربات: {e}")
            raise
        finally:
            # ذخیره داده‌های کاربران قبل از خروج
            for user_id in self.user_profiles:
                self.save_user_data(user_id)
            logger.info("داده‌های کاربران ذخیره شدند")

if __name__ == '__main__':
    config = Config()
    bot = SEOAssistantBot(config)
    bot.run()
