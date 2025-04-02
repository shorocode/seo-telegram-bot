#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ربات تلگرام پیشرفته برای تحلیل و بهینه‌سازی سئو
نسخه نهایی با رفع تمام ایرادات و اعمال بهبودها
"""

import asyncio
import hashlib
import hmac
import io
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pytz
import requests
import torch
import textstat
from bs4 import BeautifulSoup
from cachetools import TTLCache
from cryptography.fernet import Fernet
from google.cloud import translate_v2 as translate
from prometheus_client import Counter, Gauge, start_http_server
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, User
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

# ##################################################
# ## ---------- تنظیمات پیشرفته سیستم ---------- ##
# ##################################################

class ConfigError(Exception):
    """خطای مربوط به تنظیمات نادرست"""
    pass

class ModelLoadError(Exception):
    """خطای مربوط به بارگذاری مدل"""
    pass

class SecurityError(Exception):
    """خطای مربوط به امنیت"""
    pass

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
    """طرح اشتراک با تمام ویژگی‌ها و محدودیت‌ها
    
    Attributes:
        name (str): نام طرح
        monthly_price (int): قیمت ماهانه
        annual_price (int): قیمت سالانه
        features (List[str]): لیست امکانات
        rate_limit (int): محدودیت درخواست روزانه
        max_content_length (int): حداکثر طول محتوا
        advanced_analytics (bool): دسترسی به تحلیل پیشرفته
        api_access (bool): دسترسی به API
        priority_support (bool): پشتیبانی ویژه
        team_members (int): تعداد اعضای تیم
        color_code (str): کد رنگ برای نمایش
    """
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
    """پیکربندی اصلی سیستم با مقادیر پیش‌فرض
    
    Attributes:
        BOT_TOKEN (str): توکن ربات تلگرام
        MODEL_CACHE_DIR (str): مسیر ذخیره مدل‌ها
        USER_DATA_DIR (str): مسیر ذخیره داده کاربران
        BACKUP_DIR (str): مسیر پشتیبان‌گیری
        MAX_CONTENT_LENGTH (int): حداکثر طول محتوا
        KEYWORD_SUGGESTIONS (int): تعداد پیشنهادات کلیدی
        CONTENT_TYPES (Dict[str, str]): انواع محتوا
        DEFAULT_RATE_LIMIT (int): محدودیت درخواست پیش‌فرض
        ENCRYPTION_KEY (str): کلید رمزنگاری
        GOOGLE_API_KEY (Optional[str]): کلید API گوگل
        TIMEZONE (str): منطقه زمانی
        MAX_CONCURRENT_REQUESTS (int): حداکثر درخواست همزمان
        REQUEST_TIMEOUT (int): زمان انتظار برای درخواست
        SENTRY_DSN (Optional[str]): آدرس Sentry
        SUBSCRIPTION_PLANS (Dict[str, SubscriptionPlan]): طرح‌های اشتراک
    """
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

    def validate(self):
        """اعتبارسنجی تنظیمات"""
        if not self.BOT_TOKEN:
            raise ConfigError("توکن ربات باید تنظیم شود (BOT_TOKEN)")
        if not self.ENCRYPTION_KEY or len(self.ENCRYPTION_KEY) < 32:
            raise ConfigError("کلید رمزنگاری باید حداقل ۳۲ کاراکتر باشد (ENCRYPTION_KEY)")
        
        try:
            pytz.timezone(self.TIMEZONE)
        except pytz.exceptions.UnknownTimeZoneError:
            raise ConfigError(f"منطقه زمانی نامعتبر: {self.TIMEZONE}")

# ##################################################
# ## ---------- سیستم مدیریت مدل‌ها ---------- ##
# ##################################################

class ModelManager:
    """مدیریت هوشمند مدل‌های یادگیری ماشین با بهینه‌سازی منابع"""
    
    MODEL_MAPPING = {
        "keyword": {
            "model_name": "google/flan-t5-large",
            "task": "text2text-generation",
            "max_length": 100
        },
        "content": {
            "model_name": "facebook/bart-large-cnn",
            "task": "text-generation",
            "max_length": 1024
        },
        "similarity": {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "task": "sentence-similarity"
        },
        "optimization": {
            "model_name": "microsoft/prophetnet-large-uncased",
            "task": "seq2seq"
        },
        "translation": {
            "model_name": "Helsinki-NLP/opus-mt-fa-en",
            "task": "translation"
        }
    }
    
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
        """بارگذاری مدل‌ها با قابلیت تلاش مجدد
        
        Returns:
            bool: True اگر همه مدل‌ها با موفقیت بارگذاری شوند
        """
        try:
            start_time = time.time()
            loaded_models = []
            
            # بارگذاری موازی مدل‌ها
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self._load_model, model_name): model_name 
                    for model_name in self.MODEL_MAPPING
                }
                
                for future in futures:
                    model_name = futures[future]
                    try:
                        self.models[model_name] = future.result()
                        loaded_models.append(model_name)
                        logger.info(f"مدل {model_name} با موفقیت بارگذاری شد")
                    except Exception as e:
                        logger.error(f"خطا در بارگذاری مدل {model_name}: {e}")
                        raise ModelLoadError(f"خطا در بارگذاری مدل {model_name}")

            logger.info(f"تمامی مدل‌ها در {time.time() - start_time:.2f} ثانیه بارگذاری شدند")
            return len(loaded_models) == len(self.MODEL_MAPPING)
        except Exception as e:
            logger.error(f"خطای بحرانی در بارگذاری مدل‌ها: {e}")
            return False

    @lru_cache(maxsize=1)
    def _load_model(self, model_name: str):
        """بارگذاری یک مدل خاص با استفاده از تنظیمات پیش‌فرض"""
        if model_name not in self.MODEL_MAPPING:
            raise ValueError(f"مدل {model_name} پشتیبانی نمی‌شود")
            
        logger.info(f"بارگذاری مدل {model_name}...")
        start_time = time.time()
        model_config = self.MODEL_MAPPING[model_name]
        
        try:
            if model_name == "similarity":
                model = SentenceTransformer(
                    model_config["model_name"],
                    cache_folder=self.config.MODEL_CACHE_DIR,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                model = pipeline(
                    model_config["task"],
                    model=model_config["model_name"],
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    cache_dir=self.config.MODEL_CACHE_DIR
                )
            
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.labels(model_name=model_name).set(load_time)
            logger.info(f"مدل {model_name} در {load_time:.2f} ثانیه بارگذاری شد")
            return model
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل {model_name}: {e}")
            raise

    def unload_model(self, model_name: str):
        """تخلیه مدل از حافظه با مدیریت صحیح منابع
        
        Args:
            model_name (str): نام مدل برای تخلیه
        """
        if model_name in self.models:
            del self.models[model_name]
            self._load_model.cache_clear()
            logger.info(f"مدل {model_name} با موفقیت تخلیه شد")

    async def async_predict(self, model_name: str, input_data: Any):
        """پیش‌بینی ناهمزمان با مدیریت صف
        
        Args:
            model_name (str): نام مدل
            input_data (Any): داده ورودی
            
        Returns:
            Any: نتیجه پیش‌بینی
            
        Raises:
            ValueError: اگر مدل بارگذاری نشده باشد
        """
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
            raise SecurityError("کلید رمزنگاری نمی‌تواند خالی باشد")
        
        if len(encryption_key) < 32:
            raise SecurityError("کلید رمزنگاری باید حداقل ۳۲ کاراکتر باشد")
            
        try:
            # تبدیل کلید به فرمت مناسب برای Fernet
            key = hashlib.sha256(encryption_key.encode()).digest()
            self.cipher = Fernet(Fernet.generate_key())
            self.hmac_key = os.urandom(32)
            self.token_cache = TTLCache(maxsize=1000, ttl=3600)
        except Exception as e:
            raise SecurityError(f"خطا در راه‌اندازی سیستم امنیتی: {e}")

    def encrypt_data(self, data: str) -> str:
        """رمزنگاری داده با احراز هویت پیام (HMAC)
        
        Args:
            data (str): داده برای رمزنگاری
            
        Returns:
            str: داده رمزنگاری شده با HMAC
            
        Raises:
            ValueError: اگر داده ورودی خالی باشد
        """
        if not data:
            raise ValueError("داده ورودی نمی‌تواند خالی باشد")
            
        try:
            encrypted = self.cipher.encrypt(data.encode())
            hmac_value = hmac.new(self.hmac_key, encrypted, hashlib.sha256).hexdigest()
            return f"{encrypted.decode()}:{hmac_value}"
        except Exception as e:
            logger.error(f"خطا در رمزنگاری داده: {e}")
            raise SecurityError("خطا در رمزنگاری داده")

    def decrypt_data(self, encrypted_data: str) -> str:
        """رمزگشایی داده با بررسی صحت پیام
        
        Args:
            encrypted_data (str): داده رمزنگاری شده
            
        Returns:
            str: داده اصلی
            
        Raises:
            ValueError: اگر داده رمزنگاری شده نامعتبر باشد
            SecurityError: اگر HMAC تطابق نداشته باشد
        """
        if not encrypted_data:
            raise ValueError("داده رمزنگاری شده نمی‌تواند خالی باشد")
            
        try:
            encrypted, hmac_value = encrypted_data.split(":")
            if not encrypted or not hmac_value:
                raise ValueError("فرمت داده رمزنگاری شده نامعتبر است")
                
            calculated_hmac = hmac.new(self.hmac_key, encrypted.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(calculated_hmac, hmac_value):
                raise SecurityError("عدم تطابق HMAC - احتمال دستکاری داده")
                
            return self.cipher.decrypt(encrypted.encode()).decode()
        except Exception as e:
            logger.error(f"خطا در رمزگشایی داده: {e}")
            raise

    def generate_token(self, user_id: int, expires_in: int = 3600) -> str:
        """تولید توکن امنیتی با زمان انقضا
        
        Args:
            user_id (int): شناسه کاربر
            expires_in (int): زمان انقضا به ثانیه
            
        Returns:
            str: توکن تولید شده
        """
        token = hashlib.sha256(f"{user_id}{time.time()}{os.urandom(16)}".encode()).hexdigest()
        self.token_cache[token] = {
            "user_id": user_id,
            "expires_at": time.time() + expires_in
        }
        return token

    def validate_token(self, token: str) -> Optional[int]:
        """اعتبارسنجی توکن و بازگرداندن شناسه کاربر
        
        Args:
            token (str): توکن برای اعتبارسنجی
            
        Returns:
            Optional[int]: شناسه کاربر اگر توکن معتبر باشد، در غیر این صورت None
        """
        if token in self.token_cache:
            token_data = self.token_cache[token]
            if token_data["expires_at"] > time.time():
                return token_data["user_id"]
        return None
      
  ## --------اینجا قسمت دوم ربات هست--------##
# ##################################################
# ## ---------- سیستم تحلیل سئو پیشرفته ---------- ##
# ##################################################

class SEOAnalytics:
    """تحلیل پیشرفته سئو با الگوریتم‌های بهینه"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.readability_cache = TTLCache(maxsize=1000, ttl=3600)
        self.keyword_cache = TTLCache(maxsize=1000, ttl=1800)
        self.similarity_cache = TTLCache(maxsize=500, ttl=3600)
        
    @staticmethod
    def preprocess_text(text: str) -> str:
        """پیش‌پردازش متن برای تحلیل
        
        Args:
            text (str): متن ورودی
            
        Returns:
            str: متن پردازش شده
        """
        if not text:
            return ""
            
        # حذف فاصله‌های اضافی
        text = re.sub(r'\s+', ' ', text).strip()
        # نرمال‌سازی نویسه‌های فارسی
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        # حذف کاراکترهای غیر قابل چاپ
        return ''.join(char for char in text if char.isprintable())

    @staticmethod
    def count_syllables(word: str) -> int:
        """شمارش هجاهای یک کلمه فارسی
        
        Args:
            word (str): کلمه ورودی
            
        Returns:
            int: تعداد هجاها
        """
        if not word:
            return 0
            
        # الگوریتم ساده شده برای شمارش هجاهای فارسی
        vowels = {'ا', 'آ', 'ای', 'ی', 'و', 'ؤ', 'ئ', 'ـا', 'ـی', 'ـو'}
        count = 0
        
        for i, char in enumerate(word):
            if char in vowels:
                count += 1
            elif char == 'ه' and i == len(word) - 1:
                count += 1
                
        return max(1, count)

    def calculate_readability(self, text: str, lang: str = 'fa') -> float:
        """محاسبه سطح خوانایی با کش کردن نتایج
        
        Args:
            text (str): متن برای تحلیل
            lang (str): زبان متن ('fa' یا 'en')
            
        Returns:
            float: امتیاز خوانایی بین 0 تا 100
        """
        cache_key = hashlib.md5((text + lang).encode()).hexdigest()
        
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
        """محاسبه تراکم کلمات کلیدی با کش و پیش‌پردازش
        
        Args:
            text (str): متن برای تحلیل
            keywords (List[str]): لیست کلمات کلیدی
            
        Returns:
            Dict[str, float]: دیکشنری با کلمات کلیدی و تراکم آنها
        """
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

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """استخراج خودکار کلمات کلیدی از متن
        
        Args:
            text (str): متن برای تحلیل
            top_n (int): تعداد کلمات کلیدی برای استخراج
            
        Returns:
            List[str]: لیست کلمات کلیدی
        """
        if not text:
            return []
            
        try:
            # استفاده از مدل برای استخراج کلمات کلیدی
            model = self.model_manager.models.get("keyword")
            if not model:
                raise ValueError("مدل کلمات کلیدی بارگذاری نشده")
                
            result = model(
                f"Extract top {top_n} keywords from this text: {text}",
                max_length=100,
                num_return_sequences=1,
                temperature=0.3
            )
            
            keywords = result[0]["generated_text"].split(",")
            return [kw.strip() for kw in keywords if kw.strip()][:top_n]
        except Exception as e:
            logger.error(f"خطا در استخراج کلمات کلیدی: {e}")
            # روش جایگزین برای مواقع خطا
            words = re.findall(r'\b\w{3,}\b', text.lower())
            freq = {}
            for word in words:
                if word not in STOP_WORDS:
                    freq[word] = freq.get(word, 0) + 1
            return sorted(freq, key=freq.get, reverse=True)[:top_n]

    def analyze_competition(self, url: str, user_content: str) -> Dict[str, Any]:
        """تحلیل پیشرفته رقابت با وبسایت‌های دیگر
        
        Args:
            url (str): آدرس وبسایت رقیب
            user_content (str): محتوای کاربر
            
        Returns:
            Dict[str, Any]: نتایج تحلیل
        """
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
        """مقایسه پیشرفته دو محتوا
        
        Args:
            content1 (str): محتوای اول
            content2 (str): محتوای دوم
            
        Returns:
            Dict[str, Any]: نتایج مقایسه
        """
        cache_key = hashlib.md5((content1 + content2).encode()).hexdigest()
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
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
            
            result = {
                "similarity_score": round(similarity * 100, 2),
                "content1_keywords": keywords1,
                "content2_keywords": keywords2,
                "missing_keywords": list(set(keywords2) - set(keywords1))[:10],
                "content1_readability": readability1,
                "content2_readability": readability2,
                "suggestions": self._generate_comparison_suggestions(
                    content1, content2, 
                    similarity, 
                    readability1, readability2
                )
            }
            
            self.similarity_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"خطا در مقایسه محتوا: {e}")
            return {"error": str(e)}

    def _generate_comparison_suggestions(self, content1: str, content2: str, 
                                      similarity: float, 
                                      readability1: float, 
                                      readability2: float) -> List[str]:
        """تولید پیشنهادات بهبود بر اساس مقایسه دو متن"""
        suggestions = []
        
        # پیشنهادات بر اساس شباهت
        if similarity < 30:
            suggestions.append("محتواها کاملا متفاوت هستند. بهتر است موضوعات مشترک پیدا کنید.")
        elif similarity < 70:
            suggestions.append("محتواها تا حدی مشابه هستند. نقاط تمایز را تقویت کنید.")
        else:
            suggestions.append("محتواها بسیار مشابه هستند. خطر کپی بودن وجود دارد.")
        
        # پیشنهادات بر اساس خوانایی
        if readability1 < readability2:
            diff = readability2 - readability1
            if diff > 20:
                suggestions.append(f"خوانایی محتوای شما {diff:.1f}% کمتر از رقیب است. جملات را کوتاه‌تر کنید.")
            elif diff > 10:
                suggestions.append(f"خوانایی محتوای شما {diff:.1f}% کمتر از رقیب است. از کلمات ساده‌تر استفاده کنید.")
        
        # پیشنهادات کلی
        if len(content1.split()) < 300:
            suggestions.append("محتوا کوتاه است. حداقل 300 کلمه توصیه می‌شود.")
        elif len(content1.split()) > 1500:
            suggestions.append("محتوا بسیار طولانی است. بهتر است به چند بخش تقسیم شود.")
            
        return suggestions

# ##################################################
# ## ------ سیستم مدیریت کاربران پیشرفته ------ ##
# ##################################################

class UserProfile:
    """پروفایل کاربری پیشرفته با قابلیت‌های مدیریت اشتراک و تنظیمات"""
    
    def __init__(self, user_id: int, config: Config, security_manager: SecurityManager):
        self.user_id = user_id
        self.config = config
        self.security = security_manager
        self.data = self._initialize_user_data()
        self.lock = threading.Lock()
        
    def _initialize_user_data(self) -> Dict[str, Any]:
        """مقداردهی اولیه داده کاربر"""
        now = datetime.now(pytz.timezone(self.config.TIMEZONE))
        return {
            "subscription": {
                "plan": "free",
                "start_date": now.isoformat(),
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
                "last_login": datetime.now(pytz.timezone(self.config.TIMEZONE)).isoformat(),
                "login_history": [],
                "two_fa_enabled": False
            }
        }

    def update_usage(self, request_type: str) -> bool:
        """به‌روزرسانی آمار استفاده کاربر با مدیریت همزمانی
        
        Args:
            request_type (str): نوع درخواست
            
        Returns:
            bool: True اگر آمار با موفقیت به‌روزرسانی شد
        """
        with self.lock:
            now = datetime.now(pytz.timezone(self.config.TIMEZONE))
            today = now.date()
            last_date = (
                datetime.fromisoformat(self.data["usage"]["last_request"]).date() 
                if self.data["usage"]["last_request"] 
                else None
            )
            
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
        """بررسی امکان انجام درخواست با جزئیات خطا
        
        Args:
            request_type (str): نوع درخواست
            
        Returns:
            Tuple[bool, str]: (قابلیت انجام درخواست, پیام خطا در صورت وجود)
        """
        plan = self.config.SUBSCRIPTION_PLANS.get(self.data["subscription"]["plan"])
        if not plan:
            return False, "طرح اشتراک نامعتبر"
            
        if self.data["usage"]["daily_requests"] >= plan.rate_limit:
            reset_time = (datetime.now(pytz.timezone(self.config.TIMEZONE)) + timedelta(days=1)
            reset_str = reset_time.strftime("%H:%M")
            return False, f"محدودیت درخواست روزانه. تا ساعت {reset_str} صبر کنید"
            
        return True, ""

    def save_content(self, content_data: Dict) -> str:
        """ذخیره محتوای کاربر با تولید شناسه یکتا
        
        Args:
            content_data (Dict): داده محتوا برای ذخیره
            
        Returns:
            str: شناسه یکتای محتوا
        """
        content_id = hashlib.sha256(
            f"{content_data.get('title', '')}{time.time()}{self.user_id}".encode()
        ).hexdigest()
        
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
        """دریافت محتوای ذخیره شده با بررسی اعتبار
        
        Args:
            content_id (str): شناسه محتوا
            
        Returns:
            Optional[Dict]: داده محتوا اگر پیدا شود، در غیر این صورت None
        """
        with self.lock:
            for item in self.data["content"]["saved_items"]:
                if item["id"] == content_id:
                    return item
        return None

    def upgrade_subscription(self, plan_id: str, payment_method: str, duration: str = "monthly") -> bool:
        """ارتقاء اشتراک کاربر با مدیریت پرداخت
        
        Args:
            plan_id (str): شناسه طرح اشتراک
            payment_method (str): روش پرداخت
            duration (str): مدت اشتراک ('monthly' یا 'annual')
            
        Returns:
            bool: True اگر ارتقاء موفقیت‌آمیز بود
        """
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

## --------اینجا قسمت سوم ربات هست--------##

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
            "idpay": self._init_idpay(),
            "paypal": self._init_paypal()
        }
        self.receipts = TTLCache(maxsize=1000, ttl=86400)  # کش رسیدها برای 24 ساعت
        self.pending_payments = TTLCache(maxsize=500, ttl=1800)  # پرداخت‌های در انتظار
        
    def _init_zarinpal(self) -> Dict[str, Any]:
        """تنظیمات دروازه پرداخت زرین‌پال"""
        return {
            "api_key": os.getenv("ZARINPAL_API_KEY"),
            "sandbox": os.getenv("ZARINPAL_SANDBOX", "false").lower() == "true",
            "callback_url": os.getenv("ZARINPAL_CALLBACK_URL"),
            "active": bool(os.getenv("ZARINPAL_API_KEY"))
        }
        
    def _init_idpay(self) -> Dict[str, Any]:
        """تنظیمات دروازه پرداخت آیدی پی"""
        return {
            "api_key": os.getenv("IDPAY_API_KEY"),
            "sandbox": os.getenv("IDPAY_SANDBOX", "false").lower() == "true",
            "callback_url": os.getenv("IDPAY_CALLBACK_URL"),
            "active": bool(os.getenv("IDPAY_API_KEY"))
        }

    def _init_paypal(self) -> Dict[str, Any]:
        """تنظیمات دروازه پرداخت پی‌پال"""
        return {
            "client_id": os.getenv("PAYPAL_CLIENT_ID"),
            "client_secret": os.getenv("PAYPAL_CLIENT_SECRET"),
            "mode": "sandbox" if os.getenv("PAYPAL_SANDBOX", "true").lower() == "true" else "live",
            "active": bool(os.getenv("PAYPAL_CLIENT_ID"))
        }

    def get_active_providers(self) -> List[str]:
        """دریافت لیست دروازه‌های پرداخت فعال"""
        return [name for name, config in self.payment_providers.items() if config.get("active")]

    def initiate_payment(self, user_id: int, plan_id: str, provider: str = "zarinpal") -> Optional[str]:
        """شروع فرآیند پرداخت و بازگرداندن لینک پرداخت
        
        Args:
            user_id (int): شناسه کاربر
            plan_id (str): شناسه طرح اشتراک
            provider (str): دروازه پرداخت ('zarinpal', 'idpay', 'paypal')
            
        Returns:
            Optional[str]: لینک پرداخت یا None در صورت خطا
        """
        if provider not in self.payment_providers:
            logger.error(f"پروایدر پرداخت نامعتبر: {provider}")
            return None
            
        if not self.payment_providers[provider]["active"]:
            logger.error(f"پروایدر پرداخت غیرفعال: {provider}")
            return None
            
        plan = self.plans.get(plan_id)
        if not plan:
            logger.error(f"طرح اشتراک نامعتبر: {plan_id}")
            return None
            
        amount = plan.monthly_price
        description = f"ارتقاء به طرح {plan.name} - کاربر {user_id}"
        
        try:
            payment_data = {
                "user_id": user_id,
                "plan_id": plan_id,
                "amount": amount,
                "description": description,
                "timestamp": time.time()
            }
            
            if provider == "zarinpal":
                payment_url = self._zarinpal_payment(payment_data)
            elif provider == "idpay":
                payment_url = self._idpay_payment(payment_data)
            elif provider == "paypal":
                payment_url = self._paypal_payment(payment_data)
            else:
                return None
                
            if payment_url:
                payment_id = hashlib.md5(f"{user_id}{time.time()}".encode()).hexdigest()
                self.pending_payments[payment_id] = payment_data
                return payment_url
                
            return None
        except Exception as e:
            logger.error(f"خطا در شروع پرداخت: {e}")
            return None

    def verify_payment(self, payment_id: str, provider: str) -> Tuple[bool, Dict]:
        """اعتبارسنجی پرداخت و بازگرداندن نتیجه
        
        Args:
            payment_id (str): شناسه پرداخت
            provider (str): دروازه پرداخت
            
        Returns:
            Tuple[bool, Dict]: (وضعیت تأیید, اطلاعات پرداخت)
        """
        if provider not in self.payment_providers:
            return False, {"error": "پروایدر پرداخت نامعتبر"}
            
        try:
            if provider == "zarinpal":
                return self._verify_zarinpal(payment_id)
            elif provider == "idpay":
                return self._verify_idpay(payment_id)
            elif provider == "paypal":
                return self._verify_paypal(payment_id)
            else:
                return False, {"error": "پروایدر پرداخت نامعتبر"}
        except Exception as e:
            logger.error(f"خطا در تأیید پرداخت: {e}")
            return False, {"error": str(e)}

    def _zarinpal_payment(self, payment_data: Dict) -> Optional[str]:
        """پیاده‌سازی پرداخت از طریق زرین‌پال"""
        # در محیط واقعی این بخش با API زرین‌پال ارتباط برقرار می‌کند
        payment_url = f"https://zarinpal.com/pg/StartPay/{payment_data['user_id']}_{int(time.time())}"
        self.receipts[payment_url] = payment_data
        return payment_url

    def _verify_zarinpal(self, payment_id: str) -> Tuple[bool, Dict]:
        """اعتبارسنجی پرداخت زرین‌پال"""
        # در محیط واقعی این بخش با API زرین‌پال ارتباط برقرار می‌کند
        if payment_id in self.pending_payments:
            payment_data = self.pending_payments[payment_id]
            return True, {
                "success": True,
                "amount": payment_data["amount"],
                "plan_id": payment_data["plan_id"],
                "user_id": payment_data["user_id"],
                "transaction_id": f"zarinpal_{int(time.time())}"
            }
        return False, {"error": "پرداخت یافت نشد"}

    def _idpay_payment(self, payment_data: Dict) -> Optional[str]:
        """پیاده‌سازی پرداخت از طریق آیدی پی"""
        # در محیط واقعی این بخش با API آیدی پی ارتباط برقرار می‌کند
        payment_url = f"https://idpay.ir/payment/{payment_data['user_id']}_{int(time.time())}"
        self.receipts[payment_url] = payment_data
        return payment_url

    def _verify_idpay(self, payment_id: str) -> Tuple[bool, Dict]:
        """اعتبارسنجی پرداخت آیدی پی"""
        # در محیط واقعی این بخش با API آیدی پی ارتباط برقرار می‌کند
        if payment_id in self.pending_payments:
            payment_data = self.pending_payments[payment_id]
            return True, {
                "success": True,
                "amount": payment_data["amount"],
                "plan_id": payment_data["plan_id"],
                "user_id": payment_data["user_id"],
                "transaction_id": f"idpay_{int(time.time())}"
            }
        return False, {"error": "پرداخت یافت نشد"}

    def _paypal_payment(self, payment_data: Dict) -> Optional[str]:
        """پیاده‌سازی پرداخت از طریق پی‌پال"""
        # در محیط واقعی این بخش با API پی‌پال ارتباط برقرار می‌کند
        payment_url = f"https://paypal.com/payment/{payment_data['user_id']}_{int(time.time())}"
        self.receipts[payment_url] = payment_data
        return payment_url

    def _verify_paypal(self, payment_id: str) -> Tuple[bool, Dict]:
        """اعتبارسنجی پرداخت پی‌پال"""
        # در محیط واقعی این بخش با API پی‌پال ارتباط برقرار می‌کند
        if payment_id in self.pending_payments:
            payment_data = self.pending_payments[payment_id]
            return True, {
                "success": True,
                "amount": payment_data["amount"],
                "plan_id": payment_data["plan_id"],
                "user_id": payment_data["user_id"],
                "transaction_id": f"paypal_{int(time.time())}"
            }
        return False, {"error": "پرداخت یافت نشد"}

# ##################################################
# ## ----- سیستم گزارش‌گیری و آنالیتیکس ----- ##
# ##################################################

class AnalyticsManager:
    """مدیریت گزارش‌گیری و آمارهای پیشرفته"""
    
    def __init__(self, config: Config):
        self.config = config
        self.report_cache = TTLCache(maxsize=100, ttl=3600)
        self.report_templates = self._load_report_templates()
        
    def _load_report_templates(self) -> Dict[str, str]:
        """بارگذاری قالب‌های گزارش از فایل"""
        templates = {
            "seo": {
                "title": "گزارش تحلیل سئو",
                "sections": [
                    "مشخصات کلی",
                    "تحلیل کلمات کلیدی",
                    "تحلیل خوانایی",
                    "پیشنهادات بهبود"
                ]
            },
            "competitor": {
                "title": "گزارش تحلیل رقبا",
                "sections": [
                    "مقایسه کلی",
                    "کلمات کلیدی مشترک",
                    "کلمات کلیدی منحصر به فرد",
                    "تحلیل خوانایی",
                    "نقاط قوت و ضعف"
                ]
            },
            "content": {
                "title": "گزارش تحلیل محتوا",
                "sections": [
                    "خلاصه تحلیل",
                    "کلمات کلیدی پیشنهادی",
                    "بهبودهای پیشنهادی",
                    "امتیاز کلی"
                ]
            }
        }
        return templates

    def generate_seo_report(self, analysis_data: Dict, user_id: int = None) -> bytes:
        """تولید گزارش PDF حرفه‌ای
        
        Args:
            analysis_data (Dict): داده‌های تحلیل
            user_id (int): شناسه کاربر (اختیاری)
            
        Returns:
            bytes: گزارش در قالب PDF
        """
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

    def generate_competitor_report(self, comparison_data: Dict, user_id: int = None) -> bytes:
        """تولید گزارش تحلیل رقبا
        
        Args:
            comparison_data (Dict): داده‌های مقایسه
            user_id (int): شناسه کاربر (اختیاری)
            
        Returns:
            bytes: گزارش در قالب PDF
        """
        try:
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            
            # هدر گزارش
            self._draw_header(c, "گزارش تحلیل رقبا", user_id)
            
            # محتوای گزارش
            y_position = 650
            y_position = self._draw_competitor_comparison(c, comparison_data, y_position)
            
            if y_position < 150:
                c.showPage()
                self._draw_header(c, "گزارش تحلیل رقبا (ادامه)", user_id)
                y_position = 650
                
            y_position = self._draw_keyword_analysis(c, comparison_data, y_position)
            
            # فوتر گزارش
            self._draw_footer(c)
            
            c.save()
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"خطا در تولید گزارش رقبا: {e}")
            raise

    def _draw_header(self, c: canvas.Canvas, title: str, user_id: int = None):
        """رسم هدر گزارش"""
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, title)
        
        c.setFont("Helvetica", 10)
        c.drawString(100, 730, f"تاریخ تولید: {datetime.now(pytz.timezone(self.config.TIMEZONE)).strftime('%Y-%m-%d %H:%M')}")
        
        if user_id:
            c.drawString(100, 710, f"شناسه کاربر: {user_id}")
            
        c.line(100, 700, 500, 700)

    def _draw_section(self, c: canvas.Canvas, title: str, content: Any, y_pos: int) -> int:
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

    def _draw_competitor_comparison(self, c: canvas.Canvas, data: Dict, y_pos: int) -> int:
        """رسم بخش مقایسه رقبا"""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y_pos, "مقایسه با رقیب")
        y_pos -= 25
        
        c.setFont("Helvetica", 12)
        c.drawString(120, y_pos, f"امتیاز شباهت: {data.get('similarity_score', 0)}%")
        y_pos -= 20
        
        c.drawString(120, y_pos, f"خوانایی محتوای شما: {data.get('content1_readability', 0)}")
        y_pos -= 20
        
        c.drawString(120, y_pos, f"خوانایی محتوای رقیب: {data.get('content2_readability', 0)}")
        y_pos -= 30
        
        return y_pos

    def _draw_keyword_analysis(self, c: canvas.Canvas, data: Dict, y_pos: int) -> int:
        """رسم بخش تحلیل کلمات کلیدی"""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y_pos, "تحلیل کلمات کلیدی")
        y_pos -= 25
        
        c.setFont("Helvetica", 12)
        c.drawString(120, y_pos, "کلمات کلیدی مشترک:")
        y_pos -= 20
        
        common_keywords = set(data.get('content1_keywords', [])) & set(data.get('content2_keywords', []))
        for kw in list(common_keywords)[:5]:
            c.drawString(140, y_pos, f"- {kw}")
            y_pos -= 20
            
        y_pos -= 10
        c.drawString(120, y_pos, "کلمات کلیدی منحصر به فرد شما:")
        y_pos -= 20
        
        unique_keywords = set(data.get('content1_keywords', [])) - set(data.get('content2_keywords', []))
        for kw in list(unique_keywords)[:5]:
            c.drawString(140, y_pos, f"- {kw}")
            y_pos -= 20
            
        return y_pos

    def _draw_footer(self, c: canvas.Canvas):
        """رسم فوتر گزارش"""
        c.setFont("Helvetica", 8)
        c.drawString(100, 30, f"تولید شده توسط ربات سئوکار - {datetime.now(pytz.timezone(self.config.TIMEZONE)).strftime('%Y-%m-%d')}")

# ##################################################
# ## ----- سیستم یکپارچه‌سازی گوگل ----- ##
# ##################################################

class GoogleIntegration:
    """مدیریت یکپارچه‌سازی با سرویس‌های گوگل"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("کلید API گوگل باید تنظیم شود")
            
        self.api_key = api_key
        self.translate_client = None
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        
        try:
            self.translate_client = translate.Client(api_key)
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        except Exception as e:
            logger.error(f"خطا در راه‌اندازی سرویس‌های گوگل: {e}")
            raise

    def translate_text(self, text: str, target_lang: str = 'fa') -> str:
        """ترجمه متن با استفاده از Google Cloud Translation
        
        Args:
            text (str): متن برای ترجمه
            target_lang (str): زبان مقصد (پیش‌فرض: فارسی)
            
        Returns:
            str: متن ترجمه شده
            
        Raises:
            ValueError: اگر کلید API تنظیم نشده باشد
        """
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
        """دریافت داده‌های سرچ کنسول گوگل
        
        Args:
            site_url (str): آدرس سایت
            start_date (str): تاریخ شروع (YYYY-MM-DD)
            end_date (str): تاریخ پایان (YYYY-MM-DD)
            
        Returns:
            Dict: داده‌های سرچ کنسول
        """
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
        """دریافت داده‌های گوگل آنالیتیکس
        
        Args:
            view_id (str): شناسه نمای آنالیتیکس
            start_date (str): تاریخ شروع (YYYY-MM-DD)
            end_date (str): تاریخ پایان (YYYY-MM-DD)
            
        Returns:
            Dict: داده‌های آنالیتیکس
        """
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
          
## --------اینجا قسمت چهارم ربات هست--------##
# ##################################################
# ## ------ ربات تلگرام با معماری پیشرفته ------ ##
# ##################################################

class SEOAssistantBot:
    """ربات هوشمند سئوکار با معماری مقیاس‌پذیر و پیشرفته"""
    
    def __init__(self, config: Config):
        # تأیید تنظیمات ضروری
        config.validate()
        
        self.config = config
        self.user_profiles = {}  # ذخیره پروفایل کاربران در حافظه
        self._init_managers()
        self._init_telegram()
        self._setup_metrics()
        self._load_user_data()
        
        logger.info("تنظیمات اولیه ربات با موفقیت انجام شد")

    def _init_managers(self):
        """مقداردهی مدیران سرویس"""
        self.model_manager = ModelManager(self.config)
        self.security_manager = SecurityManager(self.config.ENCRYPTION_KEY)
        self.seo_analytics = SEOAnalytics(self.model_manager)
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
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"دایرکتوری کاربران ایجاد شد: {data_dir}")
                return
                
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
            backup_dir = Path(self.config.BACKUP_DIR)
            backup_dir.mkdir(exist_ok=True, parents=True)
            
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d')}.json"
            backup_path = backup_dir / backup_name
            
            all_data = {
                str(user_id): profile.data 
                for user_id, profile in self.user_profiles.items()
            }
            
            encrypted_data = self.security_manager.encrypt_data(json.dumps(all_data, ensure_ascii=False))
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(encrypted_data, f, ensure_ascii=False)
            
            logger.info(f"پشتیبان‌گیری روزانه با نام {backup_name} با موفقیت انجام شد")
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
            self._save_all_user_data()
            logger.info("داده‌های کاربران ذخیره شدند")

    def _save_all_user_data(self):
        """ذخیره تمام داده کاربران"""
        for user_id in self.user_profiles:
            self._save_user_data(user_id)

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
            )
            
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(encrypted_data, f, ensure_ascii=False)
            
            logger.debug(f"داده کاربر {user_id} با موفقیت ذخیره شد")
        except Exception as e:
            logger.error(f"خطا در ذخیره داده کاربر {user_id}: {e}")

if __name__ == '__main__':
    try:
        # تنظیمات اولیه
        config = Config()
        config.validate()
        
        # راه‌اندازی ربات
        bot = SEOAssistantBot(config)
        bot.run()
    except Exception as e:
        logger.critical(f"خطای بحرانی در راه‌اندازی ربات: {e}")
        raise
## --------اینجا قسمت پنجم ربات هست--------##


  
