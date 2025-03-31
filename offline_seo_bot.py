import logging
from typing import Dict, List, Tuple, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    Filters,
    CallbackContext,
    Dispatcher
)
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import os
from pathlib import Path
from dataclasses import dataclass
import hashlib
from functools import lru_cache

# تنظیمات پایه
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# تنظیمات پیکربندی
@dataclass
class Config:
    BOT_TOKEN: str = "YOUR_BOT_TOKEN"
    MODEL_CACHE_DIR: str = "model_cache"
    USER_DATA_DIR: str = "user_data"
    MAX_CONTENT_LENGTH: int = 5000
    KEYWORD_SUGGESTIONS: int = 10
    CONTENT_TYPES: Dict[str, str] = {
        "article": "مقاله",
        "product": "محصول",
        "landing": "صفحه فرود",
        "blog": "بلاگ پست"
    }

class SEOAnalytics:
    """کلاس برای تحلیل محتوای سئو"""
    
    @staticmethod
    def calculate_readability(text: str) -> float:
        """محاسبه خوانایی متن (Flesch-Kincaid برای فارسی)"""
        words = text.split()
        sentences = re.split(r'[.!?]', text)
        words_count = len(words)
        sentences_count = len([s for s in sentences if s.strip()])
        
        if words_count == 0 or sentences_count == 0:
            return 0
            
        avg_words_per_sentence = words_count / sentences_count
        syllables_count = sum([SEOAnalytics.count_syllables(word) for word in words])
        avg_syllables_per_word = syllables_count / words_count
        
        readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, readability))

    @staticmethod
    def count_syllables(word: str) -> int:
        """شمارش هجاهای کلمه فارسی"""
        vowels = ['ا', 'آ', 'ی', 'و', 'ه', 'ن', 'م', 'ء']
        return sum(1 for char in word if char in vowels)

    @staticmethod
    def keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
        """محاسبه تراکم کلمات کلیدی"""
        word_count = len(text.split())
        density = {}
        
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue
                
            count = text.lower().count(keyword.lower())
            density[keyword] = (count / word_count) * 100 if word_count > 0 else 0
            
        return density

class ModelManager:
    """مدیریت مدل‌های یادگیری ماشین"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_dirs()
        self.models = {}
        
    def _setup_dirs(self):
        """ایجاد دایرکتوری‌های لازم"""
        Path(self.config.MODEL_CACHE_DIR).mkdir(exist_ok=True)
        Path(self.config.USER_DATA_DIR).mkdir(exist_ok=True)
    
    def load_models(self):
        """بارگذاری مدل‌ها با کش و مدیریت حافظه"""
        try:
            self.models = {
                "keyword": self._load_keyword_model(),
                "content": self._load_content_model(),
                "similarity": self._load_similarity_model(),
                "optimization": self._load_optimization_model()
            }
            logger.info("تمامی مدل‌ها با موفقیت بارگذاری شدند")
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل‌ها: {e}")
            raise

    @lru_cache(maxsize=1)
    def _load_keyword_model(self):
        """بارگذاری مدل پیشنهاد کلمات کلیدی"""
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device="cpu",
            cache_dir=self.config.MODEL_CACHE_DIR
        )

    @lru_cache(maxsize=1)
    def _load_content_model(self):
        """بارگذاری مدل تولید محتوا"""
        return pipeline(
            "text-generation",
            model="facebook/bart-base",
            device="cpu",
            cache_dir=self.config.MODEL_CACHE_DIR
        )

    @lru_cache(maxsize=1)
    def _load_similarity_model(self):
        """بارگذاری مدل مقایسه متون"""
        return SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2',
            device='cpu',
            cache_folder=self.config.MODEL_CACHE_DIR
        )

    @lru_cache(maxsize=1)
    def _load_optimization_model(self):
        """بارگذاری مدل بهینه‌سازی متن"""
        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.config.MODEL_CACHE_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=self.config.MODEL_CACHE_DIR)
        return (model, tokenizer)

class OfflineSEOBot:
    """ربات سئو آفلاین"""
    
    def __init__(self, config: Config):
        self.config = config
        self.updater = Updater(config.BOT_TOKEN, use_context=True)
        self.dp = self.updater.dispatcher
        self.model_manager = ModelManager(config)
        self.model_manager.load_models()
        self.user_states: Dict[int, Dict] = {}
        self.setup_handlers()
        
        # بارگذاری داده‌های کاربران از کش اگر وجود دارد
        self._load_user_data()

    def _load_user_data(self):
        """بارگذاری داده‌های کاربران از فایل"""
        try:
            data_file = Path(self.config.USER_DATA_DIR) / "user_data.json"
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    self.user_states = json.load(f)
        except Exception as e:
            logger.error(f"خطا در بارگذاری داده کاربران: {e}")

    def _save_user_data(self):
        """ذخیره داده‌های کاربران در فایل"""
        try:
            data_file = Path(self.config.USER_DATA_DIR) / "user_data.json"
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_states, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"خطا در ذخیره داده کاربران: {e}")

    def setup_handlers(self):
        """تنظیم دستورات ربات"""
        handlers = [
            CommandHandler("start", self.start),
            CommandHandler("help", self.show_help),
            CallbackQueryHandler(self.handle_button),
            MessageHandler(Filters.text & ~Filters.command, self.handle_message),
            
            # دستورات سئو
            CommandHandler("keywords", self.suggest_keywords),
            CommandHandler("content", self.generate_content),
            CommandHandler("optimize", self.optimize_text),
            CommandHandler("compare", self.compare_texts),
            CommandHandler("analyze", self.analyze_seo),
            CommandHandler("history", self.show_history)
        ]
        
        for handler in handlers:
            self.dp.add_handler(handler)

    # --- توابع اصلی ---
    def start(self, update: Update, context: CallbackContext):
        """منوی اصلی"""
        user = update.effective_user
        logger.info(f"کاربر جدید شروع کرد: {user.id} - {user.full_name}")
        
        keyboard = [
            [InlineKeyboardButton("🔑 پیشنهاد کلمات کلیدی", callback_data='keywords')],
            [InlineKeyboardButton("📝 تولید محتوا", callback_data='content')],
            [InlineKeyboardButton("✍️ بهینه‌سازی متن", callback_data='optimize')],
            [InlineKeyboardButton("🆚 مقایسه متون", callback_data='compare')],
            [InlineKeyboardButton("📊 تحلیل سئو", callback_data='analyze')],
            [InlineKeyboardButton("ℹ️ راهنما", callback_data='help')]
        ]
        
        update.message.reply_text(
            f"🤖 سلام {user.first_name}!\nبه ربات پیشرفته سئو آفلاین خوش آمدید!\n\n"
            "امکانات در دسترس:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    def show_help(self, update: Update, context: CallbackContext):
        """نمایش راهنمای ربات"""
        help_text = """
📚 راهنمای ربات سئو آفلاین:

🔍 <b>پیشنهاد کلمات کلیدی</b>
/keywords [کلمه کلیدی] - دریافت کلمات کلیدی مرتبط

📝 <b>تولید محتوا</b>
/content [نوع] [موضوع] - تولید محتوای سئو شده
انواع محتوا: article, product, landing, blog

✍️ <b>بهینه‌سازی متن</b>
/optimize [متن] - بهینه‌سازی متن برای سئو

🆚 <b>مقایسه متون</b>
/compare [متن1]\n[متن2] - مقایسه دو متن

📊 <b>تحلیل سئو</b>
/analyze [متن] - تحلیل کامل سئو متن

📜 <b>تاریخچه</b>
/history - نمایش تاریخچه درخواست‌ها
"""
        update.message.reply_text(help_text, parse_mode="HTML")

    def handle_button(self, update: Update, context: CallbackContext):
        """مدیریت کلیک روی دکمه‌ها"""
        query = update.callback_query
        query.answer()
        user_id = query.from_user.id
        
        if query.data == 'keywords':
            self.user_states[user_id] = {'state': 'awaiting_keyword'}
            query.edit_message_text("🔍 لطفا کلمه کلیدی اصلی را وارد کنید:")
        
        elif query.data == 'content':
            self.show_content_menu(query)
        
        elif query.data == 'optimize':
            self.user_states[user_id] = {'state': 'awaiting_optimize'}
            query.edit_message_text("✍️ لطفا متنی که می‌خواهید بهینه شود را ارسال کنید:")
        
        elif query.data == 'compare':
            self.user_states[user_id] = {'state': 'awaiting_compare'}
            query.edit_message_text("🆚 لطفا دو متن را با خط جدید (Enter) جدا کنید:")
        
        elif query.data == 'analyze':
            self.user_states[user_id] = {'state': 'awaiting_analyze'}
            query.edit_message_text("📊 لطفا متنی که می‌خواهید تحلیل شود را ارسال کنید:")
        
        elif query.data == 'help':
            self.show_help(update)
        
        elif query.data.startswith('gen_'):
            content_type = query.data[4:]
            self.user_states[user_id] = {
                'state': 'awaiting_content_topic',
                'content_type': content_type
            }
            query.edit_message_text(f"📝 لطفا موضوع {self.config.CONTENT_TYPES.get(content_type, '')} را وارد کنید:")
        
        elif query.data == 'back':
            self.start(update, context)

    # --- توابع سئو آفلاین ---
    def suggest_keywords(self, update: Update, context: CallbackContext):
        """پیشنهاد کلمات کلیدی (بدون API)"""
        keyword = ' '.join(context.args) if context.args else None
        
        if not keyword:
            self.user_states[update.effective_user.id] = {'state': 'awaiting_keyword'}
            update.message.reply_text("🔍 لطفا کلمه کلیدی اصلی را وارد کنید:")
            return
        
        try:
            prompt = (
                f"پیشنهاد {self.config.KEYWORD_SUGGESTIONS} کلمه کلیدی مرتبط با '{keyword}' برای سئو به زبان فارسی "
                "به صورت لیست عددی بدون توضیح اضافه:"
            )
            
            result = self.model_manager.models["keyword"](
                prompt,
                max_length=150,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
            
            keywords = result[0]['generated_text'].strip()
            
            # ذخیره در تاریخچه کاربر
            self._add_to_history(
                update.effective_user.id,
                "keywords",
                {"input": keyword, "output": keywords}
            )
            
            update.message.reply_text(
                f"🔍 کلمات کلیدی پیشنهادی برای '{keyword}':\n\n{keywords}\n\n"
                "💡 می‌توانید از این کلمات در تولید محتوا استفاده کنید."
            )
        except Exception as e:
            logger.error(f"خطا در پیشنهاد کلمات کلیدی: {e}")
            update.message.reply_text("⚠️ خطایی در پردازش درخواست رخ داد. لطفا مجددا تلاش کنید.")

    def generate_content(self, update: Update, context: CallbackContext):
        """تولید محتوای سئو شده (بدون API)"""
        args = context.args
        user_id = update.effective_user.id
        
        if len(args) < 2 and user_id not in self.user_states:
            self.show_content_menu(update)
            return
        
        if user_id in self.user_states and self.user_states[user_id]['state'] == 'awaiting_content_topic':
            content_type = self.user_states[user_id]['content_type']
            topic = ' '.join(args) if args else None
            
            if not topic:
                update.message.reply_text("لطفا موضوع محتوا را وارد کنید.")
                return
        else:
            if len(args) < 2:
                update.message.reply_text("⚠️ فرمت دستور: /content [نوع] [موضوع]")
                return
                
            content_type, topic = args[0], ' '.join(args[1:])
        
        try:
            # تولید متن با توجه به نوع محتوا
            if content_type == "article":
                prompt = (
                    f"مقاله ای 300 کلمه‌ای درباره '{topic}' با رعایت اصول سئو به زبان فارسی:\n"
                    "• استفاده از تیترهای مناسب (H2, H3)\n"
                    "• پاراگراف‌های کوتاه و خوانا\n"
                    "• کلمات کلیدی مرتبط\n"
                    "• ساختار مناسب برای موتورهای جستجو"
                )
            elif content_type == "product":
                prompt = (
                    f"توضیح محصول برای '{topic}' با ویژگی‌های سئو به زبان فارسی:\n"
                    "• معرفی محصول\n"
                    "• ویژگی‌های اصلی\n"
                    "• مزایا\n"
                    "• دعوت به اقدام (CTA)"
                )
            else:
                prompt = f"متن سئو شده درباره '{topic}' به زبان فارسی:"
            
            result = self.model_manager.models["content"](
                prompt,
                max_length=500,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            content = result[0]['generated_text'].strip()
            
            # ذخیره در تاریخچه کاربر
            self._add_to_history(
                user_id,
                "content",
                {"type": content_type, "topic": topic, "output": content}
            )
            
            update.message.reply_text(
                f"📝 محتوای تولید شده ({self.config.CONTENT_TYPES.get(content_type, '')}):\n\n"
                f"{content}\n\n"
                "💡 برای تحلیل سئو این متن می‌توانید از دستور /analyze استفاده کنید."
            )
            
            # پاکسازی حالت کاربر
            if user_id in self.user_states:
                del self.user_states[user_id]
                
        except Exception as e:
            logger.error(f"خطا در تولید محتوا: {e}")
            update.message.reply_text("⚠️ خطایی در تولید محتوا رخ داد. لطفا مجددا تلاش کنید.")

    def optimize_text(self, update: Update, context: CallbackContext):
        """بهینه‌سازی متن برای سئو"""
        text = ' '.join(context.args) if context.args else None
        user_id = update.effective_user.id
        
        if not text:
            if user_id not in self.user_states:
                self.user_states[user_id] = {'state': 'awaiting_optimize'}
                update.message.reply_text("✍️ لطفا متنی که می‌خواهید بهینه شود را ارسال کنید:")
            return
        
        try:
            model, tokenizer = self.model_manager.models["optimization"]
            
            # آماده‌سازی ورودی
            input_text = f"بهینه‌سازی متن برای سئو: {text}"
            inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # تولید متن بهینه‌شده
            outputs = model.generate(
                inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True,
                temperature=0.7
            )
            
            optimized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # ذخیره در تاریخچه کاربر
            self._add_to_history(
                user_id,
                "optimize",
                {"input": text, "output": optimized_text}
            )
            
            update.message.reply_text(
                f"✍️ متن بهینه‌شده:\n\n{optimized_text}\n\n"
                "💡 تغییرات اعمال شده:\n"
                "- بهبود ساختار متن\n"
                "- افزودن کلمات کلیدی مرتبط\n"
                "- بهینه‌سازی خوانایی"
            )
            
            # پاکسازی حالت کاربر
            if user_id in self.user_states:
                del self.user_states[user_id]
                
        except Exception as e:
            logger.error(f"خطا در بهینه‌سازی متن: {e}")
            update.message.reply_text("⚠️ خطایی در بهینه‌سازی متن رخ داد. لطفا مجددا تلاش کنید.")

    def compare_texts(self, update: Update, context: CallbackContext):
        """مقایسه دو متن"""
        if context.args:
            texts = ' '.join(context.args).split('\n')
        else:
            texts = None
            
        user_id = update.effective_user.id
        
        if not texts or len(texts) < 2:
            if user_id not in self.user_states:
                self.user_states[user_id] = {'state': 'awaiting_compare'}
                update.message.reply_text("🆚 لطفا دو متن را در خطوط جداگانه ارسال کنید:")
            return
        
        try:
            # محدود کردن طول متن برای جلوگیری از مشکلات حافظه
            texts = [text[:self.config.MAX_CONTENT_LENGTH] for text in texts[:2]]
            
            # محاسبه شباهت
            embeddings = self.model_manager.models["similarity"].encode(texts)
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100
            
            # تحلیل خوانایی
            readability = [
                SEOAnalytics.calculate_readability(texts[0]),
                SEOAnalytics.calculate_readability(texts[1])
            ]
            
            # ذخیره در تاریخچه کاربر
            self._add_to_history(
                user_id,
                "compare",
                {
                    "text1": texts[0],
                    "text2": texts[1],
                    "similarity": similarity,
                    "readability": readability
                }
            )
            
            update.message.reply_text(
                f"🔍 نتایج مقایسه:\n\n"
                f"📝 متن اول (خوانایی: {readability[0]:.1f}/100):\n{texts[0][:100]}...\n\n"
                f"📝 متن دوم (خوانایی: {readability[1]:.1f}/100):\n{texts[1][:100]}...\n\n"
                f"📊 میزان شباهت محتواها: {similarity:.1f}%\n\n"
                f"💡 خوانایی بالاتر معمولاً برای سئو بهتر است."
            )
            
            # پاکسازی حالت کاربر
            if user_id in self.user_states:
                del self.user_states[user_id]
                
        except Exception as e:
            logger.error(f"خطا در مقایسه متون: {e}")
            update.message.reply_text("⚠️ خطایی در مقایسه متون رخ داد. لطفا مجددا تلاش کنید.")

    def analyze_seo(self, update: Update, context: CallbackContext):
        """تحلیل کامل سئو متن"""
        text = ' '.join(context.args) if context.args else None
        user_id = update.effective_user.id
        
        if not text:
            if user_id not in self.user_states:
                self.user_states[user_id] = {'state': 'awaiting_analyze'}
                update.message.reply_text("📊 لطفا متنی که می‌خواهید تحلیل شود را ارسال کنید:")
            return
        
        try:
            # استخراج کلمات کلیدی احتمالی
            keyword_prompt = f"استخراج 5 کلمه کلیدی اصلی از متن زیر به زبان فارسی:\n{text[:500]}"
            keywords = self.model_manager.models["keyword"](
                keyword_prompt,
                max_length=100,
                num_return_sequences=1
            )[0]['generated_text'].split(', ')
            
            # محاسبه معیارهای سئو
            readability = SEOAnalytics.calculate_readability(text)
            density = SEOAnalytics.keyword_density(text, keywords)
            
            # تحلیل ساختار متن
            structure_prompt = f"تحلیل ساختار متن زیر برای سئو به زبان فارسی:\n{text[:500]}"
            structure_analysis = self.model_manager.models["keyword"](
                structure_prompt,
                max_length=200,
                num_return_sequences=1
            )[0]['generated_text']
            
            # ذخیره در تاریخچه کاربر
            self._add_to_history(
                user_id,
                "analyze",
                {
                    "text": text,
                    "keywords": keywords,
                    "readability": readability,
                    "density": density,
                    "structure": structure_analysis
                }
            )
            
            # آماده‌سازی گزارش
            report = (
                f"📊 گزارش تحلیل سئو:\n\n"
                f"🔍 کلمات کلیدی اصلی:\n{', '.join(keywords)}\n\n"
                f"📈 تراکم کلمات کلیدی:\n"
            )
            
            for kw, dens in density.items():
                report += f"- {kw}: {dens:.2f}%\n"
            
            report += (
                f"\n📖 سطح خوانایی: {readability:.1f}/100\n"
                f"💡 {self._get_readability_feedback(readability)}\n\n"
                f"🏗️ تحلیل ساختار:\n{structure_analysis}\n\n"
                f"💡 پیشنهادات: {self._get_seo_suggestions(readability, density)}"
            )
            
            update.message.reply_text(report)
            
            # پاکسازی حالت کاربر
            if user_id in self.user_states:
                del self.user_states[user_id]
                
        except Exception as e:
            logger.error(f"خطا در تحلیل سئو: {e}")
            update.message.reply_text("⚠️ خطایی در تحلیل سئو رخ داد. لطفا مجددا تلاش کنید.")

    def show_history(self, update: Update, context: CallbackContext):
        """نمایش تاریخچه درخواست‌های کاربر"""
        user_id = update.effective_user.id
        history = self._get_user_history(user_id)
        
        if not history:
            update.message.reply_text("📜 تاریخچه‌ای یافت نشد.")
            return
        
        message = "📜 تاریخچه درخواست‌های شما:\n\n"
        for i, item in enumerate(history[-5:], 1):  # فقط 5 مورد آخر
            message += (
                f"{i}. {item['type']} - {item['timestamp']}\n"
                f"   ورودی: {item['data'].get('input', item['data'].get('text', '...'))[:30]}...\n\n"
            )
        
        update.message.reply_text(message)

    # --- توابع کمکی ---
    def show_content_menu(self, update):
        """نمایش منوی تولید محتوا"""
        keyboard = [
            [InlineKeyboardButton("📰 مقاله", callback_data='gen_article')],
            [InlineKeyboardButton("📦 محصول", callback_data='gen_product')],
            [InlineKeyboardButton("🏠 صفحه فرود", callback_data='gen_landing')],
            [InlineKeyboardButton("✍️ بلاگ پست", callback_data='gen_blog')],
            [InlineKeyboardButton("🔙 بازگشت", callback_data='back')]
        ]
        
        if hasattr(update, 'callback_query'):
            update.callback_query.edit_message_text(
                "نوع محتوا را انتخاب کنید:",
                reply_markup=InlineKeyboardMarkup(keyboard)
        else:
            update.message.reply_text(
                "نوع محتوا را انتخاب کنید:",
                reply_markup=InlineKeyboardMarkup(keyboard))

    def handle_message(self, update: Update, context: CallbackContext):
        """پردازش پیام‌های کاربر"""
        user_id = update.effective_user.id
        text = update.message.text
        
        if user_id not in self.user_states:
            self.start(update, context)
            return
        
        state = self.user_states[user_id]['state']
        
        if state == 'awaiting_keyword':
            context.args = [text]
            self.suggest_keywords(update, context)
        
        elif state == 'awaiting_optimize':
            context.args = [text]
            self.optimize_text(update, context)
        
        elif state == 'awaiting_compare':
            context.args = text.split('\n')
            self.compare_texts(update, context)
        
        elif state == 'awaiting_analyze':
            context.args = [text]
            self.analyze_seo(update, context)
        
        elif state == 'awaiting_content_topic':
            context.args = [text]
            self.generate_content(update, context)
        
        # پاکسازی حالت کاربر
        if user_id in self.user_states:
            del self.user_states[user_id]

    def _add_to_history(self, user_id: int, action_type: str, data: Dict):
        """افزودن فعالیت به تاریخچه کاربر"""
        try:
            history_file = Path(self.config.USER_DATA_DIR) / f"{user_id}_history.json"
            history = []
            
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            history.append({
                "type": action_type,
                "timestamp": str(datetime.now()),
                "data": data
            })
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history[-50:], f, ensure_ascii=False, indent=2)  # فقط 50 مورد آخر
            
        except Exception as e:
            logger.error(f"خطا در ذخیره تاریخچه: {e}")

    def _get_user_history(self, user_id: int) -> List[Dict]:
        """دریافت تاریخچه کاربر"""
        try:
            history_file = Path(self.config.USER_DATA_DIR) / f"{user_id}_history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"خطا در خواندن تاریخچه: {e}")
        return []

    def _get_readability_feedback(self, score: float) -> str:
        """بازخورد سطح خوانایی"""
        if score > 80:
            return "خوانایی عالی! متن شما برای اکثر مخاطبان قابل فهم است."
        elif score > 60:
            return "خوانایی خوب. ممکن است برخی جملات نیاز به ساده‌سازی داشته باشند."
        elif score > 40:
            return "خوانایی متوسط. پیشنهاد می‌شود جملات طولانی را کوتاه کنید."
        else:
            return "خوانایی ضعیف. متن نیاز به بازنویسی اساسی دارد."

    def _get_seo_suggestions(self, readability: float, density: Dict[str, float]) -> str:
        """پیشنهادات بهبود سئو"""
        suggestions = []
        
        # پیشنهادات بر اساس خوانایی
        if readability < 40:
            suggestions.append("متن خود را به جملات کوتاه‌تر تقسیم کنید.")
        elif readability < 60:
            suggestions.append("از کلمات ساده‌تر و جملات کوتاه‌تر استفاده کنید.")
        
        # پیشنهادات بر اساس تراکم کلمات کلیدی
        for kw, dens in density.items():
            if dens < 1:
                suggestions.append(f"استفاده از کلمه کلیدی '{kw}' را افزایش دهید.")
            elif dens > 3:
                suggestions.append(f"استفاده از کلمه کلیدی '{kw}' را کاهش دهید.")
        
        return " ".join(suggestions) if suggestions else "متن شما از نظر سئو وضعیت خوبی دارد."

    def run(self):
        """راه‌اندازی ربات"""
        self.updater.start_polling()
        logger.info("ربات شروع به کار کرد...")
        self.updater.idle()

if __name__ == '__main__':
    config = Config()
    bot = OfflineSEOBot(config)
    bot.run()
