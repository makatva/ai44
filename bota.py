import asyncio
import logging
import os
import sys
import textwrap
import json
import time
import traceback
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import base64
from io import BytesIO

# Сторонние библиотеки
import aiohttp
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, CommandObject
from aiogram.types import BotCommand, Message, ReplyParameters, ChatMemberUpdated
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter

# Внешние модули: БД
from motor.motor_asyncio import AsyncIOMotorClient

# =================================================================
# [1] GLOBAL CONFIGURATION & CONSTANTS
# =================================================================

class Config:
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
    MONGO_URL = os.getenv("MONGO_URL")
    ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
    
    # AI Engine Parameters
    MODEL_VISION = "qwen/qwen-2-vl-7b-instruct:free"  # Универсальная модель (текст + vision)
    
    MAX_CONTEXT_LEN = 40  # Глубина истории сообщений
    MAX_TOKENS_OUT = 4096
    TEMPERATURE = 0.8
    
    # Bot Settings
    TRIGGERS = ["бот", "bot", "дельфин", "dolphin", "эй"]
    
    # Timing
    REQUEST_TIMEOUT = 120

# Настройка детализированного вывода логов
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Dolphin_Apex_VPS")

# Глобальная переменная для режима обслуживания
MAINTENANCE_MODE = False

# =================================================================
# [1.1] TRIGGER DETECTION HELPERS
# =================================================================

def check_triggers(text: str) -> bool:
    """
    Проверяет наличие триггерных слов как отдельных слов (не частей других слов).
    Возвращает True, если найден хотя бы один триггер.
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Разбиваем текст на слова (по пробелам и знакам препинания)
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Проверяем каждое триггерное слово
    for trigger in Config.TRIGGERS:
        trigger_lower = trigger.lower()
        # Проверяем как отдельное слово
        if trigger_lower in words:
            return True
        # Проверяем в начале текста (с учетом того, что может быть сразу знак препинания)
        if text_lower.startswith(trigger_lower):
            # Проверяем, что после триггера идёт пробел или знак препинания
            if len(text_lower) == len(trigger_lower) or text_lower[len(trigger_lower)] in ' ,.!?;:\n':
                return True
    
    return False

def clean_thinking_blocks(text: str) -> str:
    """
    Удаляет блоки <thinking>...</thinking> из ответа модели.
    Поддерживает многострочные блоки и вложенные теги.
    """
    if not text:
        return text
    
    # Удаляем все блоки <thinking>...</thinking>
    # Флаг re.DOTALL позволяет . соответствовать переводам строк
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Удаляем лишние пустые строки, которые могли остаться
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    
    # Удаляем начальные и конечные пробелы
    cleaned = cleaned.strip()
    
    return cleaned

# =================================================================
# [2] PERSISTENCE LAYER (MONGODB HIGH-LEVEL ENGINE)
# =================================================================

class ApexDatabase:
    """
    Ультимативный менеджер данных. 
    Управляет тремя уровнями промптов, историей и метаданными пользователей.
    """
    def __init__(self, uri: str):
        self.uri = uri
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.connected = False

    async def connect(self):
        """Инициализация подключения с проверкой топологии"""
        if not self.uri:
            logger.critical("🚨 КРИТИЧЕСКАЯ ОШИБКА: MONGO_URL пуст!")
            return False
        
        try:
            # Авто-детекция имени базы данных
            db_name = "dolphin_apex_v23"
            if "/" in self.uri.split("://")[-1]:
                uri_path = self.uri.split('/')[-1].split('?')[0]
                if uri_path:
                    db_name = uri_path

            self.client = AsyncIOMotorClient(
                self.uri, 
                serverSelectionTimeoutMS=20000,
                maxPoolSize=50,
                retryWrites=True
            )
            
            # Проверка доступности
            await self.client.admin.command('ping')
            self.db = self.client[db_name]
            
            # Генерация индексов для оптимизации запросов
            await self.db.history.create_index("chat_id", unique=True)
            await self.db.users.create_index("user_id", unique=True)
            await self.db.groups.create_index("group_id", unique=True)
            await self.db.settings.create_index("key", unique=True)
            
            self.connected = True
            logger.info(f"💾 APEX DB: Успешно. База: [{db_name}]")
            return True
        except Exception as e:
            logger.error(f"🚨 DB FAILURE: {traceback.format_exc()}")
            return False

    # --- Управление промптами (3 Уровня) ---

    async def get_global_prompt(self) -> str:
        """1 уровень: Глобальный промпт от владельца бота"""
        if not self.connected: return "You are a helpful AI assistant."
        doc = await self.db.settings.find_one({"key": "global_prompt"})
        return doc.get("value") if doc else "Ты Dolphin Apex — мощный ИИ, способный анализировать контекст групп."

    async def set_global_prompt(self, prompt: str):
        await self.db.settings.update_one(
            {"key": "global_prompt"}, 
            {"$set": {"value": prompt, "updated_at": datetime.now(timezone.utc)}}, 
            upsert=True
        )

    async def get_group_prompt(self, group_id: Union[int, str]) -> str:
        """2 уровень: Настройка конкретной группы (от админа группы)"""
        if not self.connected: return ""
        doc = await self.db.groups.find_one({"group_id": str(group_id)})
        return doc.get("prompt", "") if doc else ""

    async def set_group_prompt(self, group_id: Union[int, str], prompt: str):
        await self.db.groups.update_one(
            {"group_id": str(group_id)}, 
            {"$set": {"prompt": prompt, "updated_at": datetime.now(timezone.utc)}}, 
            upsert=True
        )

    async def get_personal_prompt(self, user_id: int) -> str:
        """3 уровень: Личная настройка пользователя (в ЛС)"""
        if not self.connected: return ""
        doc = await self.db.users.find_one({"user_id": user_id})
        return doc.get("personal_prompt", "") if doc else ""

    async def set_personal_prompt(self, user_id: int, prompt: str):
        await self.db.users.update_one(
            {"user_id": user_id}, 
            {"$set": {"personal_prompt": prompt}}, 
            upsert=True
        )

    # --- Работа с историей и статистикой ---

    async def update_user_profile(self, user: types.User):
        """Обновление метаданных пользователя при каждом обращении"""
        if not self.connected: return
        await self.db.users.update_one(
            {"user_id": user.id},
            {
                "$set": {
                    "username": user.username,
                    "name": user.full_name,
                    "last_seen": datetime.now(timezone.utc)
                },
                "$inc": {"total_requests": 1}
            },
            upsert=True
        )

    async def get_chat_history(self, chat_id: Union[int, str]) -> List[Dict]:
        """Загрузка истории с санитарной проверкой ролей"""
        if not self.connected: return []
        doc = await self.db.history.find_one({"chat_id": str(chat_id)})
        if not doc: return []
        
        messages = doc.get("messages", [])
        return [{"role": m["role"], "content": m["content"]} for m in messages][-Config.MAX_CONTEXT_LEN:]

    async def save_interaction(self, chat_id: Union[int, str], author: str, query: str, response: str):
        """Сохранение диалога с именами пользователей для группового осознания"""
        if not self.connected: return
        
        # Формат: [Имя]: Текст (помогает ИИ различать людей в группе)
        entry_user = {"role": "user", "content": f"[{author}]: {query}", "ts": time.time()}
        entry_bot = {"role": "assistant", "content": response, "ts": time.time()}
        
        await self.db.history.update_one(
            {"chat_id": str(chat_id)},
            {
                "$push": {
                    "messages": {
                        "$each": [entry_user, entry_bot],
                        "$slice": -(Config.MAX_CONTEXT_LEN * 2)
                    }
                },
                "$set": {"last_update": datetime.now(timezone.utc)}
            },
            upsert=True
        )

    async def get_broadcast_list(self) -> List[int]:
        """Список всех уникальных пользователей для рассылки"""
        cursor = self.db.users.find({}, {"user_id": 1})
        return [doc["user_id"] async for doc in cursor]

# =================================================================
# [3] ИНТЕЛЛЕКТУАЛЬНОЕ ЯДРО (NEURAL)
# =================================================================

class Brain:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    async def talk(self, messages: List[Dict], temp: float = Config.TEMPERATURE, model: str = None, max_tokens: int = None) -> Optional[str]:
        """Запрос к LLM через OpenRouter с обработкой ошибок сети"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://vps-server.com",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model or Config.MODEL_VISION,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens or Config.MAX_TOKENS_OUT
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.endpoint, headers=headers, json=payload, timeout=Config.REQUEST_TIMEOUT) as r:
                    if r.status != 200:
                        err_text = await r.text()
                        logger.error(f"🧠 AI Error ({r.status}): {err_text}")
                        return None
                    data = await r.json()
                    response = data['choices'][0]['message'].get('content', '')
                    
                    # Очищаем thinking блоки из ответа
                    cleaned_response = clean_thinking_blocks(response)
                    
                    return cleaned_response
            except Exception as e:
                logger.error(f"🧠 AI Connection Failed: {e}")
                return None

    async def process_vision(self, image_data: bytes, prompt: str) -> Optional[str]:
        """Обработка изображений через vision-модель OpenRouter"""
        try:
            # Конвертация в base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            response = await self.talk(messages, model=Config.MODEL_VISION, max_tokens=4096)
            
            # Дополнительная очистка на случай, если thinking проскочил
            if response:
                response = clean_thinking_blocks(response)
            
            return response
        except Exception as e:
            logger.error(f"🖼 Vision Error: {e}")
            return None

# =================================================================
# [4] ГЛАВНЫЙ ПРОЦЕССОР (APEX ORCHESTRATOR)
# =================================================================

db = ApexDatabase(Config.MONGO_URL)
ai = Brain(Config.OPENROUTER_KEY)
bot = Bot(token=Config.TOKEN)
dp = Dispatcher()

async def apex_pipeline(message: types.Message, raw_text: str, media_type: str = None, media_data: bytes = None):
    """Основной конвейер обработки сообщения"""
    chat_id = message.chat.id
    user = message.from_user
    
    # Визуальный отклик
    await bot.send_chat_action(chat_id, "typing")
    await db.update_user_profile(user)

    # 1. Сбор контекста (3 уровня промптов + история)
    global_p = await db.get_global_prompt()
    group_p = await db.get_group_prompt(chat_id) if message.chat.type != 'private' else ""
    personal_p = await db.get_personal_prompt(user.id)
    history = await db.get_chat_history(chat_id)

    # 2. Проверка наличия изображения
    has_image = False
    
    if media_type and media_data:
        if media_type in ["изображение", "стикер", "документ-изображение", "видео-превью"]:
            has_image = True
            logger.info(f"🖼 Обнаружено изображение, будет использована vision модель: {Config.MODEL_VISION}")
        elif media_type in ["голосовое сообщение", "видео сообщение", "аудио"]:
            return await message.reply("🎤 Распознавание голоса и аудио временно недоступно.")

    # 3. Синтез системного промпта
    group_context = f"Это группа '{message.chat.title}'. " if message.chat.type != 'private' else "Это приватный чат. "
    
    system_instruction = (
        f"{global_p}\n\n"
        f"КОНТЕКСТ СРЕДЫ: {group_context}\n"
        f"УСТАНОВКА ГРУППЫ: {group_p if group_p else 'Отсутствует'}\n"
        f"ЛИЧНЫЕ ПРЕДПОЧТЕНИЯ {user.full_name}: {personal_p if personal_p else 'Не заданы'}\n\n"
        "ПРАВИЛА:\n"
        "1. Сообщения пользователей приходят в формате [Имя]: Текст. Используй это для обращения.\n"
        "2. Отвечай на языке, на котором говорит пользователь.\n"
        "3. НЕ повторяй имя пользователя в своем ответе, если он только что написал.\n"
        "4. Отвечай естественно, как в живом диалоге."
    )

    # 4. Генерация ответа - ВСЕГДА используем vision модель
    answer = None
    
    if has_image:
        # ПУТЬ 1: С изображением - vision модель + картинка
        await bot.send_chat_action(chat_id, "typing")
        
        vision_user_prompt = f"{system_instruction}\n\n[{user.full_name}]: {raw_text}" if raw_text else f"{system_instruction}\n\nОпиши это {media_type} детально на русском языке."
        
        answer = await ai.process_vision(media_data, vision_user_prompt)
        
        if not answer:
            return await message.reply("😔 Не удалось проанализировать изображение. Попробуйте позже.")
            
    else:
        # ПУТЬ 2: Без изображения - vision модель для текста
        # Qwen VL отлично работает с обычным текстом
        user_final_content = f"[{user.full_name}]: {raw_text}"

        messages = [{"role": "system", "content": system_instruction}] + history + [{"role": "user", "content": user_final_content}]

        # Используем vision модель (она универсальная для текста и картинок)
        answer = await ai.talk(messages, model=Config.MODEL_VISION)
        
        if not answer:
            return await message.reply("💤 Система временно не отвечает. Попробуйте через 30 секунд.")
    
    # 5. Обработка и отправка ответа
    if answer:
        # Убираем случайное повторение имени пользователя в начале ответа
        answer = re.sub(rf'^\[?{re.escape(user.full_name)}\]?\s*:\s*', '', answer, flags=re.IGNORECASE)
        
        # Сохранение в базу (сохраняем исходный текст без префикса имени для истории)
        clean_query = re.sub(rf'^\[{re.escape(user.full_name)}\]:\s*', '', raw_text)
        await db.save_interaction(chat_id, user.full_name, clean_query, answer)
        
        # Отправка пользователю с обработкой лимитов Telegram
        try:
            if len(answer) > 4090:
                parts = textwrap.wrap(answer, 4000, replace_whitespace=False)
                for p in parts:
                    await message.answer(p)
            else:
                try:
                    await message.reply(answer, parse_mode="Markdown")
                except:
                    await message.reply(answer)
        except Exception as e:
            logger.error(f"Send error: {e}")
            await message.reply(answer)

# =================================================================
# [5] ХЕНДЛЕРЫ КОМАНД (ADMIN & USER INTERFACE)
# =================================================================

# --- АДМИНИСТРАТИВНЫЕ (БОТ) ---

@dp.message(Command("setglobal"))
async def cmd_setglobal(m: Message, command: CommandObject):
    if m.from_user.id != Config.ADMIN_ID: return
    if not command.args:
        return await m.reply("Укажите текст промпта.")
    await db.set_global_prompt(command.args)
    await m.reply("💠 Глобальный промпт обновлен.")

@dp.message(Command("broadcast"))
async def cmd_broadcast(m: Message, command: CommandObject):
    if m.from_user.id != Config.ADMIN_ID: return
    if not command.args: return await m.reply("Введите сообщение.")
    
    users = await db.get_broadcast_list()
    count = 0
    status = await m.answer(f"⏳ Рассылка запущена ({len(users)} чел.)...")
    
    for uid in users:
        try:
            await bot.send_message(uid, f"📣 **Объявление:**\n\n{command.args}", parse_mode="Markdown")
            count += 1
            await asyncio.sleep(0.05) 
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except: continue
    
    await status.edit_text(f"✅ Рассылка завершена. Получили: {count} пользователей.")

@dp.message(Command("maintenance"))
async def cmd_maintenance(m: Message):
    """Включение/выключение режима обслуживания"""
    if m.from_user.id != Config.ADMIN_ID: 
        return await m.reply("❌ Только для администратора.")
    
    global MAINTENANCE_MODE
    MAINTENANCE_MODE = not MAINTENANCE_MODE
    
    status = "ВКЛЮЧЕН 🔧" if MAINTENANCE_MODE else "ВЫКЛЮЧЕН ✅"
    await m.reply(f"Режим обслуживания: {status}")
    logger.info(f"🔧 Maintenance mode: {MAINTENANCE_MODE}")

@dp.message(Command("settings"))
async def cmd_settings(m: Message):
    """Просмотр текущих настроек пользователя/группы"""
    user_id = m.from_user.id
    chat_id = m.chat.id
    
    # Получаем все промпты
    global_p = await db.get_global_prompt()
    personal_p = await db.get_personal_prompt(user_id)
    
    # Информация о текущем режиме
    global MAINTENANCE_MODE
    maintenance_status = "🔧 ВКЛЮЧЕН" if MAINTENANCE_MODE else "✅ ВЫКЛЮЧЕН"
    is_admin = user_id == Config.ADMIN_ID
    
    if m.chat.type == 'private':
        # Личные сообщения
        txt = (
            "⚙️ **Ваши настройки:**\n\n"
            f"**Личный промпт:**\n"
            f"{personal_p if personal_p else '_Не установлен_'}\n\n"
            f"**Глобальный промпт:**\n"
            f"_{global_p[:100]}..._\n\n"
        )
        
        if is_admin:
            txt += f"**Режим обслуживания:** {maintenance_status}\n\n"
        
        txt += (
            "**Команды:**\n"
            "`/myprompt` - изменить личный промпт\n"
            "`/clear` - очистить историю\n"
        )
    else:
        # Группы
        group_p = await db.get_group_prompt(chat_id)
        
        txt = (
            f"⚙️ **Настройки группы '{m.chat.title}':**\n\n"
            f"**Групповой промпт:**\n"
            f"{group_p if group_p else '_Не установлен_'}\n\n"
            f"**Ваш личный промпт:**\n"
            f"{personal_p if personal_p else '_Не установлен_'}\n\n"
            "**Команды:**\n"
            "`/groupprompt` - изменить групповой промпт (админ)\n"
            "`/clear` - очистить историю\n"
        )
    
    await m.reply(txt, parse_mode="Markdown")

# --- АДМИНИСТРАТИВНЫЕ (ГРУППА) ---

@dp.message(Command("groupprompt"))
async def cmd_groupprompt(m: Message, command: CommandObject):
    if m.chat.type == 'private':
        return await m.reply("❌ Эта команда только для групп.")
    
    # Проверка прав администратора группы
    member = await bot.get_chat_member(m.chat.id, m.from_user.id)
    is_admin = member.status in ["administrator", "creator"]
    is_owner = m.from_user.id == Config.ADMIN_ID

    if not (is_admin or is_owner):
        return await m.reply("❌ Только администраторы группы могут менять настройки.")

    if not command.args:
        return await m.reply("Использование: `/groupprompt Текст промпта`.")
    
    await db.set_group_prompt(m.chat.id, command.args)
    await m.reply(f"🏛 **Настройка группы обновлена:**\n{command.args}")

# --- ПОЛЬЗОВАТЕЛЬСКИЕ ---

@dp.message(Command("myprompt"))
async def cmd_myprompt(m: Message, command: CommandObject):
    if m.chat.type != 'private':
        return await m.reply("❌ Настроить личный промпт можно только в ЛС с ботом.")
    
    if not command.args:
        return await m.reply("Пример: `/myprompt Общайся со мной на 'ты' и дерзко.`")
    
    await db.set_personal_prompt(m.from_user.id, command.args)
    await m.reply("👤 Ваш личный стиль общения сохранен.")

@dp.message(Command("clear"))
async def cmd_clear(m: Message):
    if db.connected:
        await db.db.history.delete_one({"chat_id": str(m.chat.id)})
    await m.reply("🧼 Память чата очищена.")

@dp.message(Command("start"))
async def cmd_start(m: Message):
    txt = (
        "🐬 **Dolphin Apex v23 VPS Edition**\n\n"
        "Я — многоуровневая ИИ-система на VPS.\n\n"
        "**Мои слои сознания:**\n"
        "1️⃣ **Глобальный** (от владельца)\n"
        "2️⃣ **Групповой** (команда `/groupprompt` от админа группы)\n"
        "3️⃣ **Личный** (команда `/myprompt` в ЛС)\n\n"
        "**Возможности:**\n"
        "🖼 Анализирую фото, стикеры, документы\n"
        "🎥 Анализирую видео (по превью)\n"
        "🎯 Помню контекст беседы\n"
        "👥 Вижу, кто говорит в группах\n\n"
        "**Модель:**\n"
        f"🧠 Qwen 2 VL 7B (универсальная - текст + vision)\n\n"
        "**Команды:**\n"
        "`/settings` - посмотреть настройки\n"
        "`/clear` - очистить память\n"
        "`/myprompt` - личные настройки (ЛС)\n"
        "`/groupprompt` - настройки группы (админ)\n"
        "`/maintenance` - режим обслуживания (владелец)"
    )
    await m.answer(txt, parse_mode="Markdown")

# =================================================================
# [6] ОБРАБОТЧИКИ МЕДИА
# =================================================================

async def download_media(file_id: str) -> Optional[bytes]:
    """Скачивание файла из Telegram"""
    try:
        file = await bot.get_file(file_id)
        file_data = await bot.download_file(file.file_path)
        return file_data.read()
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None

@dp.message(F.photo)
async def handle_photo(m: Message):
    """Обработка фотографий"""
    if m.caption and m.caption.startswith('/'): return
    
    content = m.caption or ""
    in_private = m.chat.type == 'private'
    is_triggered = check_triggers(content)
    is_reply_to_me = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    
    is_mentioned = False
    if m.caption and m.caption_entities:
        bot_info = await bot.get_me()
        for ent in m.caption_entities:
            if ent.type == "mention" and f"@{bot_info.username}" in content:
                is_mentioned = True
    
    # Если ни один триггер не сработал - игнорируем
    if not (in_private or is_triggered or is_reply_to_me or is_mentioned):
        return
    
    # ТОЛЬКО ЕСЛИ фото адресовано боту - проверяем maintenance
    global MAINTENANCE_MODE
    if MAINTENANCE_MODE and m.from_user.id != Config.ADMIN_ID:
        return await m.reply("🔧 Бот на техническом обслуживании. Попробуйте позже.")
    
    photo = m.photo[-1]
    photo_data = await download_media(photo.file_id)
    
    if photo_data:
        await apex_pipeline(m, content or "Что на этом изображении?", media_type="изображение", media_data=photo_data)

@dp.message(F.sticker)
async def handle_sticker(m: Message):
    """Обработка стикеров"""
    in_private = m.chat.type == 'private'
    is_reply_to_me = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    
    # Если ни один триггер не сработал - игнорируем
    if not (in_private or is_reply_to_me):
        return
    
    # ТОЛЬКО ЕСЛИ стикер адресован боту - проверяем maintenance
    global MAINTENANCE_MODE
    if MAINTENANCE_MODE and m.from_user.id != Config.ADMIN_ID:
        return await m.reply("🔧 Бот на техническом обслуживании. Попробуйте позже.")
    
    if m.sticker.is_animated or m.sticker.is_video:
        await m.reply("😅 Я пока не умею анализировать анимированные стикеры, только статичные.")
        return
    
    sticker_data = await download_media(m.sticker.file_id)
    
    if sticker_data:
        await apex_pipeline(m, "Опиши этот стикер", media_type="стикер", media_data=sticker_data)

@dp.message(F.document)
async def handle_document(m: Message):
    """Обработка документов (изображения)"""
    if m.caption and m.caption.startswith('/'): return
    
    content = m.caption or ""
    in_private = m.chat.type == 'private'
    is_triggered = check_triggers(content)
    is_reply_to_me = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    
    is_mentioned = False
    if m.caption and m.caption_entities:
        bot_info = await bot.get_me()
        for ent in m.caption_entities:
            if ent.type == "mention" and f"@{bot_info.username}" in content:
                is_mentioned = True
    
    # Если ни один триггер не сработал - игнорируем
    if not (in_private or is_triggered or is_reply_to_me or is_mentioned):
        return
    
    # ТОЛЬКО ЕСЛИ документ адресован боту - проверяем maintenance
    global MAINTENANCE_MODE
    if MAINTENANCE_MODE and m.from_user.id != Config.ADMIN_ID:
        return await m.reply("🔧 Бот на техническом обслуживании. Попробуйте позже.")
    
    if m.document.mime_type and m.document.mime_type.startswith('image/'):
        doc_data = await download_media(m.document.file_id)
        
        if doc_data:
            await apex_pipeline(m, content or "Проанализируй это изображение", media_type="документ-изображение", media_data=doc_data)
    else:
        await m.reply("📄 Я работаю только с изображениями.")

@dp.message(F.voice)
async def handle_voice(m: Message):
    """Обработка голосовых сообщений"""
    in_private = m.chat.type == 'private'
    is_reply_to_me = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    
    # Если не адресовано боту - игнорируем
    if not (in_private or is_reply_to_me):
        return
    
    # ТОЛЬКО ЕСЛИ адресовано боту - проверяем maintenance
    global MAINTENANCE_MODE
    if MAINTENANCE_MODE and m.from_user.id != Config.ADMIN_ID:
        return await m.reply("🔧 Бот на техническом обслуживании. Попробуйте позже.")
    
    await m.reply("🎤 Распознавание голоса временно недоступно.")

@dp.message(F.video_note)
async def handle_video_note(m: Message):
    """Обработка видео сообщений (кружочков)"""
    in_private = m.chat.type == 'private'
    is_reply_to_me = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    
    # Если ни один триггер не сработал - игнорируем
    if not (in_private or is_reply_to_me):
        return
    
    # ТОЛЬКО ЕСЛИ адресовано боту - проверяем maintenance
    global MAINTENANCE_MODE
    if MAINTENANCE_MODE and m.from_user.id != Config.ADMIN_ID:
        return await m.reply("🔧 Бот на техническом обслуживании. Попробуйте позже.")
    
    if m.video_note.thumbnail:
        thumb_data = await download_media(m.video_note.thumbnail.file_id)
        
        if thumb_data:
            await apex_pipeline(m, "Что на этом видео?", media_type="видео-превью", media_data=thumb_data)
    else:
        await m.reply("🎥 Не удалось получить превью видео.")

@dp.message(F.video)
async def handle_video(m: Message):
    """Обработка видео файлов"""
    if m.caption and m.caption.startswith('/'): return
    
    content = m.caption or ""
    in_private = m.chat.type == 'private'
    is_triggered = check_triggers(content)
    is_reply_to_me = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    
    is_mentioned = False
    if m.caption and m.caption_entities:
        bot_info = await bot.get_me()
        for ent in m.caption_entities:
            if ent.type == "mention" and f"@{bot_info.username}" in content:
                is_mentioned = True
    
    # Если ни один триггер не сработал - игнорируем
    if not (in_private or is_triggered or is_reply_to_me or is_mentioned):
        return
    
    # ТОЛЬКО ЕСЛИ адресовано боту - проверяем maintenance
    global MAINTENANCE_MODE
    if MAINTENANCE_MODE and m.from_user.id != Config.ADMIN_ID:
        return await m.reply("🔧 Бот на техническом обслуживании. Попробуйте позже.")
    
    if m.video.thumbnail:
        thumb_data = await download_media(m.video.thumbnail.file_id)
        
        if thumb_data:
            video_prompt = f"{content}\n\nПримечание: Это превью видео, полный анализ видео пока недоступен." if content else "Что на этом видео? (анализ по превью)"
            await apex_pipeline(m, video_prompt, media_type="видео-превью", media_data=thumb_data)
    else:
        await m.reply("🎥 Не удалось получить превью видео. Попробуйте отправить скриншот.")

@dp.message(F.audio)
async def handle_audio(m: Message):
    """Обработка аудио файлов"""
    in_private = m.chat.type == 'private'
    is_reply_to_me = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    
    # Если не адресовано боту - игнорируем
    if not (in_private or is_reply_to_me):
        return
    
    # ТОЛЬКО ЕСЛИ адресовано боту - проверяем maintenance
    global MAINTENANCE_MODE
    if MAINTENANCE_MODE and m.from_user.id != Config.ADMIN_ID:
        return await m.reply("🔧 Бот на техническом обслуживании. Попробуйте позже.")
    
    await m.reply("🎵 Распознавание аудио временно недоступно.")

# =================================================================
# [7] ENGINE STARTUP & INFRASTRUCTURE
# =================================================================

@dp.message(F.text)
async def message_router(m: Message):
    """Главный распределитель входящих текстовых сообщений"""
    if m.text and m.text.startswith('/'): return
    
    content = m.text or ""
    if not content: return

    # Сначала проверяем триггеры
    in_private = m.chat.type == 'private'
    is_triggered = check_triggers(content)
    is_reply_to_me = m.reply_to_message and m.reply_to_message.from_user.id == bot.id
    is_mentioned = False
    
    if m.entities:
        bot_info = await bot.get_me()
        for ent in m.entities:
            if ent.type == "mention" and f"@{bot_info.username}" in content:
                is_mentioned = True

    # Если сообщение НЕ адресовано боту - игнорируем
    if not (in_private or is_triggered or is_reply_to_me or is_mentioned):
        return
    
    # ТОЛЬКО ЕСЛИ сообщение адресовано боту - проверяем maintenance
    global MAINTENANCE_MODE
    if MAINTENANCE_MODE and m.from_user.id != Config.ADMIN_ID:
        return await m.reply("🔧 Бот на техническом обслуживании. Попробуйте позже.")
    
    # Всё проверили - обрабатываем
    await apex_pipeline(m, content)

async def main():
    """Главная функция запуска бота"""
    logger.info("🚀 Запуск Dolphin Apex VPS Edition...")
    
    # Подключаемся к БД
    await db.connect()
    
    # Удаляем webhook если был (на всякий случай)
    await bot.delete_webhook(drop_pending_updates=True)
    logger.info("✅ Webhook удалён")
    
    # Регистрация команд
    public_cmds = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="settings", description="Текущие настройки"),
        BotCommand(command="myprompt", description="Личная настройка (в ЛС)"),
        BotCommand(command="groupprompt", description="Настройка группы (Админ)"),
        BotCommand(command="clear", description="Очистить память")
    ]
    await bot.set_my_commands(public_cmds)
    logger.info("✅ Команды зарегистрированы")
    
    # Запуск polling
    logger.info("✅ Бот запущен в режиме polling")
    logger.info(f"🤖 Модель: {Config.MODEL_VISION}")
    logger.info(f"👤 Админ ID: {Config.ADMIN_ID}")
    
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        logger.error(traceback.format_exc())
